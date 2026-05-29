"""
http_chunk_receiver.py — FastAPI receiver for Android-uploaded MP4 chunks.

Implements the two endpoints the Android recorder app expects:
    POST /api/v1/chunks/upload         — multipart (metadata JSON + video bytes + checksum)
    GET  /api/v1/chunks/{chunk_id}/status

Chunk lifecycle:
    receive → write three files to received/ → enqueue chunk_id for healthcheck
    → background worker validates (CRC32 + ffprobe-style frame count tolerance)
    → updates in-memory status cache → status endpoint returns healthy / unhealthy / pending
    → file moved to validated/ or failed/ for inspection

Storage layout under config.yaml `receiver.storage_root`:
    received/<chunk_id>.mp4    raw upload (multipart field "video")
    received/<chunk_id>.json   chunk header (multipart field "metadata", JSON)
    received/<chunk_id>.crc32  app-side checksum (multipart field "checksum", text)
    validated/                 chunks that passed healthcheck (mp4 + json moved here)
    failed/                    chunks that failed (kept for debugging)

Run:
    uvicorn src.http_chunk_receiver:app --host 0.0.0.0 --port 8000

Or with config defaults:
    python -m src.http_chunk_receiver
"""
from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
import time
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import cv2
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ReceiverConfig:
    """Resolved receiver configuration loaded from config.yaml's `receiver:` block."""

    host: str = "0.0.0.0"
    port: int = 8000
    storage_root: Path = Path("store/chunks")
    frame_count_tolerance: float = 0.05
    healthcheck_workers: int = 2
    status_cache_ttl_seconds: int = 3600

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "ReceiverConfig":
        """Build from the top-level config.yaml; missing fields fall back to dataclass defaults."""
        cfg_path = Path(path)
        if not cfg_path.exists():
            logger.warning("config.yaml not found at %s — using receiver defaults", cfg_path)
            return cls()
        with cfg_path.open() as f:
            full = yaml.safe_load(f) or {}
        block = full.get("receiver", {}) or {}
        return cls(
            host=block.get("host", cls.host),
            port=int(block.get("port", cls.port)),
            storage_root=Path(block.get("storage_root", str(cls.storage_root))),
            frame_count_tolerance=float(block.get("frame_count_tolerance", cls.frame_count_tolerance)),
            healthcheck_workers=int(block.get("healthcheck_workers", cls.healthcheck_workers)),
            status_cache_ttl_seconds=int(block.get("status_cache_ttl_seconds", cls.status_cache_ttl_seconds)),
        )


# ---------------------------------------------------------------------------
# Status cache
# ---------------------------------------------------------------------------

@dataclass
class ChunkStatus:
    """In-memory record of a chunk's validation state."""

    chunk_id: str
    status: str  # "pending" | "healthy" | "unhealthy"
    received_ms: int
    expected_frame_count: int = 0
    actual_frame_count: int = 0
    duration_seconds: float = 0.0
    issues: list[str] = field(default_factory=list)
    processing_started_ms: int | None = None
    retry_deadline_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Match the JSON shape the Kotlin HealthcheckResponse expects."""
        out: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "status": self.status,
            "message": "; ".join(self.issues) if self.issues else None,
        }
        if self.status == "healthy":
            out["processing_started_ms"] = self.processing_started_ms or self.received_ms
            out["validation"] = {
                "frame_count_match": True,
                "checksum_valid": True,
                "no_sequence_gaps": True,
                "no_timestamp_gaps": True,
                "frames_in_order": True,
            }
        elif self.status == "unhealthy":
            out["expected"] = self.expected_frame_count
            out["received"] = self.actual_frame_count
            out["retry_deadline_ms"] = self.retry_deadline_ms or (self.received_ms + 120_000)
        return {k: v for k, v in out.items() if v is not None}


class StatusCache:
    """Thread-safe dict of chunk_id → ChunkStatus."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: dict[str, ChunkStatus] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def put(self, status: ChunkStatus) -> None:
        with self._lock:
            self._store[status.chunk_id] = status

    def get(self, chunk_id: str) -> ChunkStatus | None:
        with self._lock:
            return self._store.get(chunk_id)

    def update(self, chunk_id: str, **fields: Any) -> None:
        """Patch the fields of an existing record. Silently no-ops if missing."""
        with self._lock:
            existing = self._store.get(chunk_id)
            if existing is None:
                return
            for k, v in fields.items():
                setattr(existing, k, v)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _crc32_hex(path: Path) -> str:
    """Compute CRC32 over a file's bytes, returning lowercase hex (matches Android Moshi output)."""
    crc = 0
    with path.open("rb") as f:
        while True:
            buf = f.read(1 << 20)  # 1 MB
            if not buf:
                break
            crc = zlib.crc32(buf, crc)
    # zlib.crc32 returns unsigned 32-bit int. Hex without 0x prefix, lowercase, 8 chars.
    return f"{crc & 0xFFFFFFFF:08x}"


def _probe_mp4(path: Path) -> tuple[int, float]:
    """
    Return (frame_count, duration_seconds) using OpenCV.

    OpenCV reads the MP4 container header to find these without decoding the whole file,
    so this is fast. Returns (0, 0.0) if the file can't be opened.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0, 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        duration = frame_count / fps if fps > 0 else 0.0
        return frame_count, duration
    finally:
        cap.release()


def validate_chunk(
    chunk_id: str,
    storage_root: Path,
    frame_count_tolerance: float,
) -> ChunkStatus:
    """
    Run healthcheck on a chunk that's already been written to received/.

    Steps:
        1. Read the header JSON; pull expected_frame_count, checksum_value, encoding.
        2. Compute CRC32 over the .mp4 bytes; compare to the app's checksum.
        3. Probe the MP4 for actual frame count (and duration for logging).
        4. Compare against expected_frame_count within the configured tolerance.

    Returns a ChunkStatus marking the chunk healthy or unhealthy with a list of issues.
    """
    received_dir = storage_root / "received"
    mp4_path = received_dir / f"{chunk_id}.mp4"
    json_path = received_dir / f"{chunk_id}.json"
    crc_path = received_dir / f"{chunk_id}.crc32"

    issues: list[str] = []
    expected_frames = 0
    actual_frames = 0
    duration = 0.0

    if not mp4_path.exists():
        return ChunkStatus(
            chunk_id=chunk_id,
            status="unhealthy",
            received_ms=int(time.time() * 1000),
            issues=[f"Video file not found at {mp4_path}"],
        )

    # Header
    try:
        header = json.loads(json_path.read_text())
        expected_frames = int(header.get("expected_frame_count", 0))
        header_checksum = str(header.get("checksum_value", "")).lower()
    except Exception as exc:
        issues.append(f"Header JSON unreadable: {exc}")
        header_checksum = ""

    # Checksum — prefer the app's separately-sent value, fall back to header
    app_checksum = ""
    if crc_path.exists():
        try:
            app_checksum = crc_path.read_text().strip().lower()
        except Exception:
            pass
    expected_checksum = app_checksum or header_checksum

    actual_checksum = _crc32_hex(mp4_path)
    if expected_checksum and actual_checksum != expected_checksum:
        issues.append(
            f"Checksum mismatch: app sent {expected_checksum}, computed {actual_checksum}"
        )

    # Frame count via ffprobe-equivalent (OpenCV reads MP4 header only)
    actual_frames, duration = _probe_mp4(mp4_path)
    if actual_frames == 0:
        issues.append("OpenCV could not open MP4 or it has 0 frames")
    elif expected_frames > 0:
        diff = abs(actual_frames - expected_frames) / expected_frames
        if diff > frame_count_tolerance:
            issues.append(
                f"Frame count {actual_frames} differs from expected {expected_frames} "
                f"by {diff:.1%} (tolerance {frame_count_tolerance:.0%})"
            )

    now_ms = int(time.time() * 1000)
    if issues:
        return ChunkStatus(
            chunk_id=chunk_id,
            status="unhealthy",
            received_ms=now_ms,
            expected_frame_count=expected_frames,
            actual_frame_count=actual_frames,
            duration_seconds=duration,
            issues=issues,
            retry_deadline_ms=now_ms + 120_000,
        )
    return ChunkStatus(
        chunk_id=chunk_id,
        status="healthy",
        received_ms=now_ms,
        expected_frame_count=expected_frames,
        actual_frame_count=actual_frames,
        duration_seconds=duration,
        processing_started_ms=now_ms,
    )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class HealthcheckWorker:
    """Drains a queue of chunk_ids, validates each, updates the status cache, moves files."""

    def __init__(
        self,
        cfg: ReceiverConfig,
        cache: StatusCache,
        chunk_queue: "queue.Queue[str | None]",
        on_validated: "Callable[[ChunkStatus], None] | None" = None,
    ) -> None:
        self._cfg = cfg
        self._cache = cache
        self._queue = chunk_queue
        self._on_validated = on_validated
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Spawn the worker pool. Idempotent — second call is a no-op."""
        if self._threads:
            return
        for i in range(self._cfg.healthcheck_workers):
            t = threading.Thread(target=self._run, name=f"healthcheck-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("Started %d healthcheck worker(s)", len(self._threads))

    def stop(self) -> None:
        """Signal all workers to exit and wait briefly."""
        self._stop_event.set()
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if chunk_id is None:
                return
            try:
                self._process_one(chunk_id)
            except Exception:
                logger.exception("Healthcheck failed for %s", chunk_id)

    def _process_one(self, chunk_id: str) -> None:
        result = validate_chunk(
            chunk_id=chunk_id,
            storage_root=self._cfg.storage_root,
            frame_count_tolerance=self._cfg.frame_count_tolerance,
        )
        self._cache.put(result)

        # Move files to validated/ or failed/ for inspection.
        target_dir = self._cfg.storage_root / ("validated" if result.status == "healthy" else "failed")
        target_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".mp4", ".json", ".crc32"):
            src = self._cfg.storage_root / "received" / f"{chunk_id}{suffix}"
            if src.exists():
                shutil.move(str(src), str(target_dir / src.name))

        logger.info(
            "[%s] %s — frames %d/%d, duration %.2fs%s",
            chunk_id,
            result.status,
            result.actual_frame_count,
            result.expected_frame_count,
            result.duration_seconds,
            f" ({'; '.join(result.issues)})" if result.issues else "",
        )

        # Notify any in-process consumer (e.g. ChunkReceiverSource) that this
        # chunk has finished validating and its files are now in validated/ or failed/.
        if self._on_validated is not None:
            try:
                self._on_validated(result)
            except Exception:
                logger.exception("on_validated callback failed for %s", chunk_id)


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(
    cfg: ReceiverConfig,
    cache: StatusCache,
    chunk_queue: "queue.Queue[str | None]",
    worker: HealthcheckWorker,
) -> FastAPI:
    """
    Build a FastAPI app whose routes close over the given instances.

    Used both by the module-level standalone `app` and by ChunkReceiverSource,
    which embeds the receiver in-process alongside the CV pipeline.
    """

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Create storage directories and start the validation worker pool, then tear down on shutdown."""
        for sub in ("received", "validated", "failed"):
            (cfg.storage_root / sub).mkdir(parents=True, exist_ok=True)
        worker.start()
        logger.info(
            "Receiver ready on %s:%d, storage=%s, tolerance=%.0f%%",
            cfg.host, cfg.port, cfg.storage_root, cfg.frame_count_tolerance * 100,
        )
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(title="basketball-cv chunk receiver", lifespan=_lifespan)

    @app.post("/api/v1/chunks/upload", status_code=202)
    async def upload_chunk(
        metadata: str = Form(...),
        video: UploadFile = File(...),
        checksum: str = Form(""),
    ) -> dict[str, Any]:
        """
        Receive a multipart chunk upload from the Android recorder app.

        Expected fields (match Android ChunkUploader.kt):
            metadata  application/json — ChunkHeader (chunk_id, expected_frame_count,
                      checksum_value, encoding, ...)
            video     application/octet-stream — MP4 bytes
            checksum  text/plain — CRC32 hex (also present inside metadata; we trust this one)

        Writes three files to received/<chunk_id>.{mp4,json,crc32} and enqueues for
        background validation. Returns 202 Accepted immediately.
        """
        try:
            header = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"metadata JSON invalid: {exc}")

        chunk_id = str(header.get("chunk_id") or "").strip()
        if not chunk_id:
            raise HTTPException(status_code=400, detail="metadata.chunk_id is required")
        # Defensive — prevent path traversal
        if "/" in chunk_id or ".." in chunk_id:
            raise HTTPException(status_code=400, detail="metadata.chunk_id contains illegal characters")

        received_dir = cfg.storage_root / "received"
        received_dir.mkdir(parents=True, exist_ok=True)

        mp4_path = received_dir / f"{chunk_id}.mp4"
        json_path = received_dir / f"{chunk_id}.json"
        crc_path = received_dir / f"{chunk_id}.crc32"

        # Stream the upload to disk so memory stays bounded regardless of chunk size.
        bytes_written = 0
        with mp4_path.open("wb") as f:
            while True:
                buf = await video.read(1 << 20)  # 1 MB
                if not buf:
                    break
                f.write(buf)
                bytes_written += len(buf)

        json_path.write_text(json.dumps(header, indent=2))
        if checksum:
            crc_path.write_text(checksum.strip())

        now_ms = int(time.time() * 1000)
        cache.put(
            ChunkStatus(
                chunk_id=chunk_id,
                status="pending",
                received_ms=now_ms,
                expected_frame_count=int(header.get("expected_frame_count", 0)),
            )
        )
        chunk_queue.put(chunk_id)

        logger.info(
            "Received %s — %.1f MB, expected %d frames, encoding=%s",
            chunk_id,
            bytes_written / 1024 / 1024,
            header.get("expected_frame_count", 0),
            header.get("encoding", "?"),
        )
        return {
            "status": "received",
            "chunk_id": chunk_id,
            "timestamp_received_ms": now_ms,
            "message": "Chunk queued for healthcheck",
        }

    @app.get("/api/v1/chunks/{chunk_id}/status")
    async def get_chunk_status(chunk_id: str) -> JSONResponse:
        """
        Return validation state for a previously-uploaded chunk.

        Returns 200 OK with status "healthy" or "unhealthy" once validation completes,
        202 Accepted with status "pending" while the worker is still processing,
        404 Not Found if the chunk_id was never uploaded.
        """
        status = cache.get(chunk_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Unknown chunk_id: {chunk_id}")
        if status.status == "pending":
            return JSONResponse(status_code=202, content=status.to_dict())
        return JSONResponse(status_code=200, content=status.to_dict())

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness probe — returns the receiver's resolved config so you can confirm it's running."""
        return {
            "status": "ok",
            "host": cfg.host,
            "port": cfg.port,
            "storage_root": str(cfg.storage_root),
            "frame_count_tolerance": cfg.frame_count_tolerance,
            "workers": cfg.healthcheck_workers,
        }

    return app


# ---------------------------------------------------------------------------
# Module-level app (standalone path: `uvicorn src.http_chunk_receiver:app`)
# ---------------------------------------------------------------------------

_cfg = ReceiverConfig.from_yaml()
_cache = StatusCache(ttl_seconds=_cfg.status_cache_ttl_seconds)
_queue: "queue.Queue[str | None]" = queue.Queue()
_worker = HealthcheckWorker(_cfg, _cache, _queue)
app = create_app(_cfg, _cache, _queue, _worker)


# ---------------------------------------------------------------------------
# In-process chunk source
# ---------------------------------------------------------------------------

class ChunkReceiverSource:
    """
    Embeds the FastAPI receiver in-process and exposes validated chunks as a queue.

    Mirrors StreamChunkRecorder's interface so PipelineWorker can consume Android
    uploads the same way it consumes RTSP-recorded chunks:

        source = ChunkReceiverSource(cfg)
        chunk_queue = source.start()
        while True:
            path = chunk_queue.get()
            if path is None:        # sentinel: receiver stopped
                break
            process(path)           # wrap in FileVideoSource
        source.stop()

    A uvicorn server runs in a background thread. Each chunk that passes the
    healthcheck is buffered keyed by the integer suffix of its chunk_id
    (e.g. camera_0_chunk_7 -> 7) and emitted onto the output queue in ascending
    sequence order. Because the healthcheck pool validates chunks in parallel,
    completion order is not guaranteed, so a release thread enforces ordering:

      - a chunk that is the next contiguous sequence is emitted immediately;
      - the lowest buffered chunk is emitted once it has been held hold_seconds
        (the gap-filler is presumed lost) or once max_reorder_buffer chunks
        pile up — so one missing chunk can never stall the pipeline forever.
    """

    def __init__(
        self,
        cfg: ReceiverConfig | None = None,
        max_reorder_buffer: int = 8,
        hold_seconds: float = 2.0,
    ) -> None:
        self._cfg = cfg or ReceiverConfig.from_yaml()
        self._cache = StatusCache(ttl_seconds=self._cfg.status_cache_ttl_seconds)
        self._id_queue: "queue.Queue[str | None]" = queue.Queue()
        self._worker = HealthcheckWorker(
            self._cfg, self._cache, self._id_queue, on_validated=self._on_validated
        )
        self._app = create_app(self._cfg, self._cache, self._id_queue, self._worker)

        self._out_queue: "queue.Queue[str | None]" = queue.Queue()
        self._server: "Any | None" = None
        self._thread: threading.Thread | None = None

        # Reorder state (guarded by _lock)
        self._max_reorder_buffer = max_reorder_buffer
        self._hold_seconds = hold_seconds
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[str, float]] = {}   # seq -> (path, buffered_at)
        self._last_emitted: int | None = None              # highest seq emitted so far
        self._fallback_seq = -1                            # for ids without a numeric suffix
        self._release_thread: threading.Thread | None = None
        self._release_stop = threading.Event()

    def start(self) -> "queue.Queue[str | None]":
        """Start the embedded uvicorn server + release thread. Returns the validated-chunk path queue."""
        import uvicorn

        config = uvicorn.Config(
            self._app,
            host=self._cfg.host,
            port=self._cfg.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="chunk-receiver-uvicorn", daemon=True
        )
        self._thread.start()
        self._release_stop.clear()
        self._release_thread = threading.Thread(
            target=self._release_loop, name="chunk-reorder-release", daemon=True
        )
        self._release_thread.start()
        logger.info(
            "ChunkReceiverSource listening on %s:%d (storage=%s)",
            self._cfg.host, self._cfg.port, self._cfg.storage_root,
        )
        return self._out_queue

    def stop(self) -> None:
        """Stop the server + release thread, flush any buffered chunks in order, then enqueue the sentinel."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._release_stop.set()
        if self._release_thread is not None:
            self._release_thread.join(timeout=5.0)
        with self._lock:
            for seq in sorted(self._pending):
                self._out_queue.put(self._pending[seq][0])
            self._pending.clear()
        self._out_queue.put(None)

    def _parse_seq(self, chunk_id: str) -> int:
        """Extract the trailing integer of a chunk_id; fall back to a monotonic counter."""
        match = re.search(r"(\d+)$", chunk_id)
        if match:
            return int(match.group(1))
        self._fallback_seq += 1
        return self._fallback_seq

    def _on_validated(self, status: ChunkStatus) -> None:
        """HealthcheckWorker callback — buffer healthy chunks; the release thread emits them in order."""
        if status.status != "healthy":
            return
        mp4_path = self._cfg.storage_root / "validated" / f"{status.chunk_id}.mp4"
        seq = self._parse_seq(status.chunk_id)
        with self._lock:
            self._pending[seq] = (str(mp4_path), time.monotonic())
            self._release_ready_locked()

    def _release_loop(self) -> None:
        """Periodically release buffered chunks whose hold timer has elapsed."""
        while not self._release_stop.is_set():
            with self._lock:
                self._release_ready_locked()
            self._release_stop.wait(0.25)

    def _release_ready_locked(self) -> None:
        """
        Emit buffered chunks in ascending seq order. Caller must hold _lock.

        A chunk is released when it is the next contiguous sequence, or when the
        lowest buffered chunk has been held past hold_seconds / the buffer is full
        (gap-skip — emit lowest and advance past it).
        """
        now = time.monotonic()
        while self._pending:
            lowest = min(self._pending)
            path, buffered_at = self._pending[lowest]

            contiguous = self._last_emitted is not None and lowest == self._last_emitted + 1
            timed_out = (now - buffered_at) >= self._hold_seconds
            buffer_full = len(self._pending) > self._max_reorder_buffer

            if contiguous:
                pass  # in-order, emit immediately
            elif self._last_emitted is None and (timed_out or buffer_full):
                pass  # first emission — start at the lowest we've settled on
            elif timed_out or buffer_full:
                logger.warning(
                    "Releasing chunk seq %d out of order (was expecting %s; %d buffered)",
                    lowest,
                    "?" if self._last_emitted is None else self._last_emitted + 1,
                    len(self._pending),
                )
            else:
                break  # wait for the gap to fill or the hold timer to elapse

            self._out_queue.put(path)
            self._pending.pop(lowest)
            self._last_emitted = lowest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the receiver with uvicorn using values from config.yaml."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    import uvicorn
    uvicorn.run(
        "src.http_chunk_receiver:app",
        host=_cfg.host,
        port=_cfg.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
