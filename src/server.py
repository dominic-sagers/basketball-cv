"""
server.py — standalone FastAPI backend for game archiving.

No Qt, no torch, no GPU. Designed to run on the NUC production box.

Android session lifecycle:
    POST /api/v1/sessions                  create (cam A: camera_id + team)
    POST /api/v1/sessions/{run_id}/join    join (cam B: camera_id + team)
    POST /api/v1/chunks/upload             upload chunk (metadata must include run_id + camera_id)
    POST /api/v1/sessions/{run_id}/end     end → concat + dvc push in background
    GET  /api/v1/sessions/{run_id}         poll status

Chunks are stored per session + camera to avoid seq-number collisions:
    store/chunks/{run_id}/{camera_id}/{chunk_id}.mp4

Concatenated output lands in:
    store/output/games/{run_id}/game_{camera_id}_raw.mp4

Session state is persisted to:
    store/output/games/{run_id}/session.json

Run:
    python src/server.py
    python src/server.py --config config.game.yaml --port 8000
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game_archive import dvc_push

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    chunks_root: Path = Path("store/chunks")
    games_root: Path = Path("store/output/games")

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "ServerConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            logger.warning("Config not found at %s — using defaults", cfg_path)
            return cls()
        with cfg_path.open() as f:
            full = yaml.safe_load(f) or {}
        recv = full.get("receiver", {}) or {}
        return cls(
            host=recv.get("host", "0.0.0.0"),
            port=int(recv.get("port", 8000)),
            chunks_root=Path(recv.get("storage_root", "store/chunks")),
            games_root=Path("store/output/games"),
        )


# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

WAITING = "WAITING"
RECORDING = "RECORDING"
ARCHIVING = "ARCHIVING"
DONE = "DONE"
FAILED = "FAILED"


@dataclass
class CameraRecord:
    camera_id: str
    team: str
    joined_at: str
    chunks: list[str] = field(default_factory=list)


@dataclass
class GameSession:
    run_id: str
    state: str
    created_at: str
    cameras: dict[str, CameraRecord] = field(default_factory=dict)
    ended_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "cameras": {
                cid: {
                    "camera_id": c.camera_id,
                    "team": c.team,
                    "joined_at": c.joined_at,
                    "chunk_count": len(c.chunks),
                }
                for cid, c in self.cameras.items()
            },
        }

    def persist(self, games_root: Path) -> None:
        out_dir = games_root / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_chunks"] = {cid: c.chunks for cid, c in self.cameras.items()}
        (out_dir / "session.json").write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GameSession":
        data = json.loads(path.read_text())
        session = cls(
            run_id=data["run_id"],
            state=data["state"],
            created_at=data["created_at"],
            ended_at=data.get("ended_at"),
            error=data.get("error"),
        )
        chunk_lists = data.get("_chunks", {})
        for cid, cam in data.get("cameras", {}).items():
            session.cameras[cid] = CameraRecord(
                camera_id=cid,
                team=cam.get("team", ""),
                joined_at=cam.get("joined_at", ""),
                chunks=chunk_lists.get(cid, []),
            )
        return session


class SessionRegistry:
    """Thread-safe in-memory session store. Persists to session.json on every write."""

    def __init__(self, games_root: Path) -> None:
        self._games_root = games_root
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.Lock()
        self._restore()

    def _restore(self) -> None:
        """Reload sessions from disk on startup (survives server restarts)."""
        if not self._games_root.exists():
            return
        for f in self._games_root.glob("*/session.json"):
            try:
                s = GameSession.load(f)
                if s.state in (WAITING, RECORDING, ARCHIVING):
                    s.state = FAILED
                    s.error = "Server restarted while session was active"
                    s.persist(self._games_root)
                self._sessions[s.run_id] = s
                logger.info("Restored session %s (%s)", s.run_id, s.state)
            except Exception:
                logger.exception("Could not restore session from %s", f)

    def create(self, camera_id: str, team: str) -> GameSession:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session = GameSession(
            run_id=run_id,
            state=RECORDING,
            created_at=datetime.now().isoformat(),
            cameras={
                camera_id: CameraRecord(
                    camera_id=camera_id,
                    team=team,
                    joined_at=datetime.now().isoformat(),
                )
            },
        )
        with self._lock:
            self._sessions[run_id] = session
        session.persist(self._games_root)
        logger.info("Session created: %s  cam=%s team=%s", run_id, camera_id, team)
        return session

    def join(self, run_id: str, camera_id: str, team: str) -> GameSession:
        with self._lock:
            session = self._require(run_id)
            if session.state != RECORDING:
                raise HTTPException(400, f"Session {run_id} is {session.state}, expected RECORDING")
            if camera_id in session.cameras:
                raise HTTPException(400, f"Camera {camera_id} already in session {run_id}")
            session.cameras[camera_id] = CameraRecord(
                camera_id=camera_id,
                team=team,
                joined_at=datetime.now().isoformat(),
            )
        session.persist(self._games_root)
        logger.info("Camera %s (team %s) joined session %s", camera_id, team, run_id)
        return session

    def add_chunk(self, run_id: str, camera_id: str, chunk_path: str) -> None:
        with self._lock:
            session = self._require(run_id)
            if session.state != RECORDING:
                raise HTTPException(400, f"Session {run_id} is {session.state}, not RECORDING")
            if camera_id not in session.cameras:
                raise HTTPException(
                    400,
                    f"Camera {camera_id} not in session {run_id} — call /join first",
                )
            session.cameras[camera_id].chunks.append(chunk_path)
        session.persist(self._games_root)

    def end(self, run_id: str) -> GameSession:
        with self._lock:
            session = self._require(run_id)
            if session.state != RECORDING:
                raise HTTPException(400, f"Session {run_id} is {session.state}, not RECORDING")
            session.state = ARCHIVING
            session.ended_at = datetime.now().isoformat()
        session.persist(self._games_root)
        return session

    def set_state(self, run_id: str, state: str, error: str | None = None) -> None:
        with self._lock:
            session = self._sessions.get(run_id)
            if session is None:
                return
            session.state = state
            if error:
                session.error = error
        if session:
            session.persist(self._games_root)

    def get(self, run_id: str) -> GameSession | None:
        with self._lock:
            return self._sessions.get(run_id)

    def _require(self, run_id: str) -> GameSession:
        session = self._sessions.get(run_id)
        if session is None:
            raise HTTPException(404, f"Session {run_id} not found")
        return session


# ---------------------------------------------------------------------------
# Archiving helpers (no ML imports)
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        str(Path.home() / ".cache/ripcheck/ffstatic/ffmpeg"),
    ):
        if Path(candidate).exists():
            return candidate
    return None


def _concat(chunk_files: list[str], output_path: str) -> bool:
    if not chunk_files:
        logger.warning("No chunks to concatenate.")
        return False
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.error("ffmpeg not found — cannot concat. Install ffmpeg.")
        return False
    list_path = Path(output_path).with_suffix(".concat_list.txt")
    list_path.write_text("\n".join(f"file '{Path(c).resolve()}'" for c in chunk_files))
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", output_path]
    logger.info("Concatenating %d chunks → %s", len(chunk_files), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode == 0:
        logger.info("Saved: %s", output_path)
        return True
    logger.error("ffmpeg concat failed:\n%s", result.stderr[-600:])
    return False


def _parse_seq(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _crc32_hex(path: Path) -> str:
    crc = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            crc = zlib.crc32(chunk, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


# DVC is not safe for concurrent CLI invocations against one repo — two sessions
# ending close together previously collided on DVC's internal lock, and the failed
# one still got reported as archived DONE. Serialize all dvc_push() calls here.
_dvc_lock = threading.Lock()


def _run_archive(session: GameSession, cfg: ServerConfig, registry: SessionRegistry) -> None:
    """Background thread: concat each camera's chunks → raw MP4 → dvc push → DONE."""
    out_dir = cfg.games_root / session.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for camera_id, cam in session.cameras.items():
        if not cam.chunks:
            logger.warning("[%s] Camera %s: no chunks — skipping", session.run_id, camera_id)
            continue
        ordered = sorted(cam.chunks, key=lambda p: _parse_seq(Path(p).stem))
        output = str(out_dir / f"game_{camera_id}_raw.mp4")
        if not _concat(ordered, output):
            logger.error("[%s] Camera %s concat failed", session.run_id, camera_id)
            all_ok = False

    if not all_ok:
        registry.set_state(session.run_id, FAILED, error="Concat failed for one or more cameras")
        logger.error("Session %s archive failed → FAILED", session.run_id)
        return

    with _dvc_lock:
        pushed = dvc_push()

    if pushed:
        registry.set_state(session.run_id, DONE)
        logger.info("Session %s archived → DONE", session.run_id)
    else:
        registry.set_state(session.run_id, FAILED, error="DVC push failed or timed out — see server logs")
        logger.error("Session %s archive failed → FAILED (dvc push)", session.run_id)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(cfg: ServerConfig) -> FastAPI:
    registry = SessionRegistry(cfg.games_root)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        cfg.chunks_root.mkdir(parents=True, exist_ok=True)
        cfg.games_root.mkdir(parents=True, exist_ok=True)
        logger.info("Server ready on %s:%d", cfg.host, cfg.port)
        yield

    app = FastAPI(title="basketball-cv game server", lifespan=_lifespan)

    # --- Session endpoints ---------------------------------------------------

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session(body: dict[str, Any]) -> dict[str, Any]:
        """Create a new session. Body: {camera_id, team}. Returns {run_id}."""
        camera_id = str(body.get("camera_id") or "").strip()
        team = str(body.get("team") or "").strip()
        if not camera_id:
            raise HTTPException(400, "camera_id is required")
        if not team:
            raise HTTPException(400, "team is required")
        session = registry.create(camera_id, team)
        return {"run_id": session.run_id, "state": session.state}

    @app.post("/api/v1/sessions/{run_id}/join", status_code=200)
    async def join_session(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Register a second camera. Body: {camera_id, team}."""
        camera_id = str(body.get("camera_id") or "").strip()
        team = str(body.get("team") or "").strip()
        if not camera_id:
            raise HTTPException(400, "camera_id is required")
        if not team:
            raise HTTPException(400, "team is required")
        session = registry.join(run_id, camera_id, team)
        return session.to_dict()

    @app.post("/api/v1/sessions/{run_id}/end", status_code=202)
    async def end_session(run_id: str) -> dict[str, Any]:
        """End the session. Triggers concat + DVC push in the background."""
        session = registry.end(run_id)
        t = threading.Thread(
            target=_run_archive,
            args=(session, cfg, registry),
            name=f"archive-{run_id}",
            daemon=False,
        )
        t.start()
        return {"run_id": run_id, "state": session.state, "message": "Archiving started"}

    @app.get("/api/v1/sessions/{run_id}")
    async def get_session(run_id: str) -> JSONResponse:
        """Poll session status."""
        session = registry.get(run_id)
        if session is None:
            raise HTTPException(404, f"Session {run_id} not found")
        code = 200 if session.state in (DONE, FAILED) else 202
        return JSONResponse(status_code=code, content=session.to_dict())

    # --- Chunk upload --------------------------------------------------------

    @app.post("/api/v1/chunks/upload", status_code=202)
    async def upload_chunk(
        metadata: str = Form(...),
        video: UploadFile = File(...),
        checksum: str = Form(""),
    ) -> dict[str, Any]:
        """
        Receive a chunk from the Android recorder.

        metadata JSON must include: chunk_id, run_id, camera_id, expected_frame_count.
        checksum (optional): CRC32 hex — if provided, chunk is rejected on mismatch.
        """
        try:
            header = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"metadata JSON invalid: {exc}")

        chunk_id = str(header.get("chunk_id") or "").strip()
        run_id = str(header.get("run_id") or "").strip()
        camera_id = str(header.get("camera_id") or "").strip()

        if not chunk_id:
            raise HTTPException(400, "metadata.chunk_id is required")
        if not run_id:
            raise HTTPException(400, "metadata.run_id is required")
        if not camera_id:
            raise HTTPException(400, "metadata.camera_id is required")
        if "/" in chunk_id or ".." in chunk_id:
            raise HTTPException(400, "chunk_id contains illegal characters")

        chunk_dir = cfg.chunks_root / run_id / camera_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"{chunk_id}.mp4"

        bytes_written = 0
        with chunk_path.open("wb") as f:
            while buf := await video.read(1 << 20):
                f.write(buf)
                bytes_written += len(buf)

        # CRC32 check — reject and delete if mismatch
        app_checksum = checksum.strip().lower() or str(header.get("checksum_value", "")).lower()
        if app_checksum:
            actual = _crc32_hex(chunk_path)
            if actual != app_checksum:
                chunk_path.unlink(missing_ok=True)
                raise HTTPException(
                    422,
                    f"Checksum mismatch for {chunk_id}: expected {app_checksum}, got {actual}",
                )

        registry.add_chunk(run_id, camera_id, str(chunk_path))

        logger.info(
            "Chunk received: %s  session=%s cam=%s  %.1f MB",
            chunk_id, run_id, camera_id, bytes_written / 1024 / 1024,
        )
        return {
            "status": "received",
            "chunk_id": chunk_id,
            "run_id": run_id,
            "camera_id": camera_id,
            "bytes": bytes_written,
            "timestamp_received_ms": int(time.time() * 1000),
        }

    # --- Health --------------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        sessions = registry._sessions
        return {
            "status": "ok",
            "host": cfg.host,
            "port": cfg.port,
            "sessions": {
                state: sum(1 for s in sessions.values() if s.state == state)
                for state in (RECORDING, ARCHIVING, DONE, FAILED)
            },
        }

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="basketball-cv game server — no GPU required")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = ServerConfig.from_yaml(args.config)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port

    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
