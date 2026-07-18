"""
server.py — standalone FastAPI backend for game archiving.

No Qt, no torch, no GPU. Designed to run on the NUC production box.

Android session lifecycle:
    POST /api/v1/sessions                  create (cam A: camera_id + team) → WAITING
    POST /api/v1/sessions/{run_id}/join    join (cam B: camera_id + team) — WAITING or RECORDING
    POST /api/v1/sessions/{run_id}/start   confirm positioning → RECORDING (uploads now accepted)
    POST /api/v1/chunks/upload             upload chunk (metadata must include run_id + camera_id)
    POST /api/v1/sessions/{run_id}/metrics upload one camera's device-health log (JSONL, any session state)
    POST /api/v1/sessions/{run_id}/end     end → DRAINING (uploads still accepted) → concat + dvc push;
                                            ending while still WAITING → CANCELLED (nothing to archive)
    GET  /api/v1/sessions/{run_id}         poll status
    GET  /api/v1/sessions                  list recent sessions (newest first)
    GET  /api/v1/push-status               background dvc push status (global, not per-session)

A session sits in WAITING from creation until a device confirms it has finished positioning
the camera and started capturing — mirrors the Android app's own Positioning screen, so a
session backed out of before Continue never gets reported as "RECORDING" with zero footage.

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

from src.game_archive import dvc_add_local, dvc_push_background, get_push_status

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
    # End-of-session drain: how long to keep accepting other cameras' upload
    # backlog before archiving (basketball-cv#15).
    drain_quiet_seconds: float = 20.0    # camera that never reported a total (crashed device) counts as drained after this much upload silence
    drain_timeout_seconds: float = 900.0 # hard cap — archive whatever arrived by then
    drain_poll_seconds: float = 2.0      # how often the drain watcher re-checks
    # Test environments run without a mounted .dvc/.git — set false there so archiving
    # completes as DONE locally instead of failing on a missing DVC repo.
    dvc_enabled: bool = True

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
            drain_quiet_seconds=float(recv.get("drain_quiet_seconds", 20.0)),
            drain_timeout_seconds=float(recv.get("drain_timeout_seconds", 900.0)),
            drain_poll_seconds=float(recv.get("drain_poll_seconds", 2.0)),
            dvc_enabled=bool(recv.get("dvc_enabled", True)),
        )


# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

WAITING = "WAITING"      # created (or joined), but no camera has confirmed positioning yet
RECORDING = "RECORDING"  # at least one camera has confirmed positioning and is capturing
DRAINING = "DRAINING"
ARCHIVING = "ARCHIVING"
DONE = "DONE"
FAILED = "FAILED"
CANCELLED = "CANCELLED"  # ended while still WAITING — no camera ever started, nothing to archive

# States a client should read as "the session is over — stop capturing".
ENDED_STATES = (DRAINING, ARCHIVING, DONE, FAILED, CANCELLED)


@dataclass
class CameraRecord:
    camera_id: str
    team: str
    joined_at: str
    chunks: list[str] = field(default_factory=list)
    # Total segments the device says it captured (sent with its end request) —
    # when known, "drained" means the server has received exactly that many.
    # None means the device never got to report (it crashed or lost its
    # connection for good); the quiet-period heuristic covers that camera so
    # one dead device can't stall the archive until the drain timeout.
    reported_total: int | None = None
    # time.monotonic() of the last received chunk; transient, not persisted.
    last_chunk_at: float | None = None


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
                    "reported_total": c.reported_total,
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
                reported_total=cam.get("reported_total"),
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
                # WAITING included: interrupted mid-positioning needs recovery too.
                if s.state in (WAITING, RECORDING, DRAINING, ARCHIVING):
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
            state=WAITING,
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
            # WAITING: the creator hasn't confirmed positioning yet — the second device can
            # still join and position in parallel. RECORDING: the creator already started;
            # a later joiner just starts its own capture immediately after positioning.
            if session.state not in (WAITING, RECORDING):
                raise HTTPException(400, f"Session {run_id} is {session.state}, expected WAITING or RECORDING")
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

    def begin_recording(self, run_id: str, camera_id: str | None = None) -> GameSession:
        """
        WAITING → RECORDING: a device has confirmed positioning and started its camera.

        Whichever device calls this first (creator or joiner — each decides independently
        when to leave its own positioning screen) unblocks uploads for the whole session, so
        the other device's first chunk is never rejected regardless of which one continues
        first. Idempotent: a second caller (the other device) just gets the current state
        back, same as end() tolerates a second caller.
        """
        with self._lock:
            session = self._require(run_id)
            if session.state == WAITING:
                session.state = RECORDING
            elif session.state != RECORDING:
                raise HTTPException(400, f"Session {run_id} is {session.state}, cannot start recording")
        session.persist(self._games_root)
        logger.info("Session %s recording confirmed by %s", run_id, camera_id or "?")
        return session

    def add_chunk(self, run_id: str, camera_id: str, chunk_path: str) -> None:
        with self._lock:
            session = self._require(run_id)
            if session.state not in (RECORDING, DRAINING):
                raise HTTPException(
                    400, f"Session {run_id} is {session.state}, not accepting uploads"
                )
            if camera_id not in session.cameras:
                raise HTTPException(
                    400,
                    f"Camera {camera_id} not in session {run_id} — call /join first",
                )
            cam = session.cameras[camera_id]
            cam.chunks.append(chunk_path)
            cam.last_chunk_at = time.monotonic()
        session.persist(self._games_root)

    def end(
        self,
        run_id: str,
        camera_id: str | None = None,
        captured_count: int | None = None,
    ) -> tuple[GameSession, bool]:
        """
        Mark the session ended. Returns (session, newly_ended) — newly_ended is True only
        for the RECORDING → DRAINING transition (the caller uses it to decide whether to
        kick off the drain-then-archive background thread).

        Ending an already-ended session is not an error — the joiner device may
        legitimately call end after the creator did (android#21); it gets the
        current state back, and its captured_count is still recorded so the
        drain watcher knows when that camera's backlog is fully received.

        Ending a WAITING session (positioning was cancelled before any camera started —
        confirmPositioning() was never reached) goes straight to CANCELLED: no camera ever
        ran, so there's nothing to drain or archive.
        """
        with self._lock:
            session = self._require(run_id)
            if camera_id and captured_count is not None and camera_id in session.cameras:
                session.cameras[camera_id].reported_total = captured_count
            if session.state == WAITING:
                session.state = CANCELLED
                session.ended_at = datetime.now().isoformat()
                newly_ended = False
            else:
                newly_ended = session.state == RECORDING
                if newly_ended:
                    session.state = DRAINING
                    session.ended_at = datetime.now().isoformat()
        session.persist(self._games_root)
        return session, newly_ended

    def begin_archiving(self, run_id: str) -> GameSession | None:
        """DRAINING → ARCHIVING; returns None if the session is no longer draining."""
        with self._lock:
            session = self._sessions.get(run_id)
            if session is None or session.state != DRAINING:
                return None
            session.state = ARCHIVING
        session.persist(self._games_root)
        return session

    def all_drained(self, run_id: str, quiet_seconds: float, drain_start: float) -> bool:
        """
        True when every camera's upload backlog appears fully received: a camera
        with a client-reported total is drained once that many chunks arrived;
        one without is drained after quiet_seconds with no new chunk.
        """
        with self._lock:
            session = self._sessions.get(run_id)
            if session is None:
                return True
            now = time.monotonic()
            for cam in session.cameras.values():
                if cam.reported_total is not None:
                    if len(cam.chunks) < cam.reported_total:
                        return False
                else:
                    last_activity = cam.last_chunk_at if cam.last_chunk_at is not None else drain_start
                    if now - last_activity < quiet_seconds:
                        return False
        return True

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

    def list_sessions(self, limit: int = 20) -> list[GameSession]:
        with self._lock:
            sessions = list(self._sessions.values())
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)[:limit]

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


# dvc add is not safe for concurrent CLI invocations against one repo — two sessions
# ending close together previously collided on DVC's internal lock, and the failed
# one still got reported as archived DONE. Serialize dvc_add_local() calls here.
# The background dvc_push_background() has its own single-flight guard and does
# not need this lock — see game_archive.py.
_dvc_lock = threading.Lock()


def _drain_then_archive(session: GameSession, cfg: ServerConfig, registry: SessionRegistry) -> None:
    """
    Background thread: hold the session in DRAINING until every camera's upload
    backlog has been received (or the drain times out), then archive.

    Ending a session used to archive immediately, discarding whatever the other
    cameras hadn't uploaded yet — the normal game-night flow lost the joiner's
    tail footage (basketball-cv#15).
    """
    drain_start = time.monotonic()
    deadline = drain_start + cfg.drain_timeout_seconds
    while time.monotonic() < deadline:
        if registry.all_drained(session.run_id, cfg.drain_quiet_seconds, drain_start):
            logger.info("[%s] All cameras drained after %.1fs", session.run_id, time.monotonic() - drain_start)
            break
        time.sleep(cfg.drain_poll_seconds)
    else:
        logger.warning(
            "[%s] Drain timed out after %.0fs — archiving what was received",
            session.run_id, cfg.drain_timeout_seconds,
        )

    if registry.begin_archiving(session.run_id) is None:
        logger.warning("[%s] No longer DRAINING — skipping archive", session.run_id)
        return
    _run_archive(session, cfg, registry)


def _run_archive(session: GameSession, cfg: ServerConfig, registry: SessionRegistry) -> None:
    """
    Background thread: concat each camera's chunks → raw MP4 → dvc add (local) → DONE.

    A session counts as archived once its data is safely in the local DVC cache
    on this machine — that's disk-bound and fast. Pushing that cache on to the
    remote (Dominic's tailnet machine) happens separately in its own background
    thread and does not gate this session's completion; over the current
    relay-bound link a full push can take hours, and nothing should wait on it.
    """
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

    if not cfg.dvc_enabled:
        registry.set_state(session.run_id, DONE)
        logger.info("Session %s archived → DONE (local, DVC disabled)", session.run_id)
        return

    with _dvc_lock:
        added = dvc_add_local()

    if not added:
        registry.set_state(session.run_id, FAILED, error="dvc add failed — see server logs")
        logger.error("Session %s archive failed → FAILED (dvc add)", session.run_id)
        return

    registry.set_state(session.run_id, DONE)
    logger.info("Session %s archived → DONE (local)", session.run_id)

    threading.Thread(
        target=dvc_push_background,
        name=f"dvc-push-{session.run_id}",
        daemon=True,
    ).start()


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

    @app.get("/api/v1/sessions")
    async def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions, newest first — backs the Android app's status page."""
        return [s.to_dict() for s in registry.list_sessions(limit=limit)]

    @app.get("/api/v1/push-status")
    async def push_status() -> dict[str, Any]:
        """
        Status of the background dvc push to the remote (Dominic's tailnet machine).

        Global, not per-session — a session reaching DONE only means its data is
        safe in the local DVC cache on this machine; this reflects whether that
        cache has also been pushed on to the remote.
        """
        return get_push_status()

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

    @app.post("/api/v1/sessions/{run_id}/start", status_code=200)
    async def start_recording(run_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Confirm positioning is done and this device's camera has started capturing.
        WAITING → RECORDING; a no-op if another device already made this call. Optional
        body: {camera_id} — informational, logged only.
        """
        camera_id = str((body or {}).get("camera_id") or "").strip() or None
        session = registry.begin_recording(run_id, camera_id)
        return session.to_dict()

    @app.post("/api/v1/sessions/{run_id}/end", status_code=202)
    async def end_session(run_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        End the session: drain remaining uploads from all cameras, then concat +
        DVC push in the background.

        Optional body: {camera_id, captured_count} — the calling device's total
        captured-segment count, so the drain knows when that camera is complete.
        Ending an already-ended session returns its current state (not an error).
        """
        camera_id = str((body or {}).get("camera_id") or "").strip() or None
        try:
            raw_count = (body or {}).get("captured_count")
            captured_count = int(raw_count) if raw_count is not None else None
        except (TypeError, ValueError):
            raise HTTPException(400, "captured_count must be an integer")

        session, newly_ended = registry.end(run_id, camera_id, captured_count)
        if newly_ended:
            t = threading.Thread(
                target=_drain_then_archive,
                args=(session, cfg, registry),
                name=f"archive-{run_id}",
                daemon=False,
            )
            t.start()
            message = "Draining uploads, then archiving"
        elif session.state == CANCELLED:
            message = "Session cancelled before recording started"
        else:
            message = f"Session already ended ({session.state})"
        return {"run_id": run_id, "state": session.state, "message": message}

    @app.get("/api/v1/sessions/{run_id}")
    async def get_session(run_id: str) -> JSONResponse:
        """Poll session status."""
        session = registry.get(run_id)
        if session is None:
            raise HTTPException(404, f"Session {run_id} not found")
        code = 200 if session.state in (DONE, FAILED, CANCELLED) else 202
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

    # --- Device metrics --------------------------------------------------------

    @app.post("/api/v1/sessions/{run_id}/metrics", status_code=202)
    async def upload_metrics(
        run_id: str,
        camera_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """
        Receive one camera's device-health log for this session: a JSONL file of
        periodic battery/thermal/CPU samples taken by the Android app for the
        duration of the recording. Purely diagnostic — never gates session state,
        and accepted regardless of state as long as the session exists (unlike chunk
        upload, this isn't part of the drain-count bookkeeping). Stored alongside the
        session's archived footage so a phone's health over a game can be checked
        without touching the device.
        """
        if registry.get(run_id) is None:
            raise HTTPException(404, f"Session {run_id} not found")
        camera_id = camera_id.strip()
        if not camera_id:
            raise HTTPException(400, "camera_id is required")
        if "/" in camera_id or ".." in camera_id:
            raise HTTPException(400, "camera_id contains illegal characters")

        out_dir = cfg.games_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"metrics_{camera_id}.jsonl"

        bytes_written = 0
        with dest.open("wb") as f:
            while buf := await file.read(1 << 20):
                f.write(buf)
                bytes_written += len(buf)

        logger.info(
            "Metrics received: session=%s cam=%s %d bytes", run_id, camera_id, bytes_written,
        )
        return {
            "status": "received",
            "run_id": run_id,
            "camera_id": camera_id,
            "bytes": bytes_written,
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
                for state in (WAITING, RECORDING, DRAINING, ARCHIVING, DONE, FAILED, CANCELLED)
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
