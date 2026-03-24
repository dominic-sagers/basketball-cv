"""
video_source.py — unified video source abstraction for the basketball CV pipeline.

Supports three source types, all sharing the same interface:
  - FileVideoSource   : pre-recorded video files (mp4, avi, mov, etc.)
  - RTSPVideoSource   : network RTSP streams with automatic reconnection
  - USBCameraSource   : local USB/built-in cameras

Usage:
    source = VideoSourceFactory.from_config(source_cfg)
    source.open()
    while source.is_opened():
        ok, frame = source.read()
        if ok:
            ...
    source.release()

All sources are configured from config.yaml — see docs/architecture.md for the
config schema. Never hardcode paths, indices, or URLs in source code.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VideoSource(ABC):
    """Abstract interface for all video sources in the basketball CV pipeline."""

    @abstractmethod
    def open(self) -> bool:
        """Open and initialise the source. Returns True on success."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next (or latest) frame.

        Returns:
            (True, frame) on success
            (False, None) on failure or end-of-stream
        """

    @abstractmethod
    def release(self) -> None:
        """Release all resources."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Return True if the source is currently open and usable."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second of this source."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height) of this source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for logging."""


# ---------------------------------------------------------------------------
# File source
# ---------------------------------------------------------------------------

class FileVideoSource(VideoSource):
    """
    Video source backed by a pre-recorded file.

    Useful for development and testing before live cameras are available.
    Set loop=True to replay the file indefinitely (handy for iterating on
    detection logic with a fixed test clip).
    """

    def __init__(self, path: str, name: str = "file", loop: bool = False) -> None:
        self._path = path
        self._name = name
        self._loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._resolution: tuple[int, int] = (1920, 1080)

    def open(self) -> bool:
        if not Path(self._path).exists():
            logger.error("[%s] File not found: %s", self._name, self._path)
            return False

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            logger.error("[%s] OpenCV could not open file: %s", self._name, self._path)
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (w, h)
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / self._fps if self._fps > 0 else 0

        logger.info(
            "[%s] Opened %s — %dx%d @ %.1f FPS, %d frames (%.1fs)%s",
            self._name, Path(self._path).name, w, h, self._fps,
            total, duration, " [looping]" if self._loop else "",
        )
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()

        if not ok:
            if self._loop:
                logger.debug("[%s] EOF — looping back to start", self._name)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                logger.info("[%s] End of file.", self._name)
                return False, None

        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.debug("[%s] Released.", self._name)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# RTSP source
# ---------------------------------------------------------------------------

class RTSPVideoSource(VideoSource):
    """
    Video source backed by an RTSP network stream.

    Runs a background reader thread that continuously pulls frames and stores
    only the latest one. This prevents the processing pipeline from falling
    behind the stream (lag buildup) — the pipeline always gets the most recent
    frame rather than working through a growing backlog.

    Automatically reconnects if the stream drops, with a configurable delay
    between attempts.
    """

    def __init__(
        self,
        url: str,
        name: str = "rtsp",
        reconnect_delay: float = 3.0,
        connect_timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._name = name
        self._reconnect_delay = reconnect_delay
        self._connect_timeout = connect_timeout

        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._resolution: tuple[int, int] = (1920, 1080)

        # Latest frame shared between reader thread and main thread
        self._frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._opened = False

    def open(self) -> bool:
        self._stop_event.clear()
        self._opened = self._connect()
        if self._opened:
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name=f"rtsp-reader-{self._name}",
                daemon=True,
            )
            self._reader_thread.start()
        return self._opened

    def _connect(self) -> bool:
        """Attempt to open the RTSP connection. Returns True on success."""
        logger.info("[%s] Connecting to %s ...", self._name, self._url)

        # RTSP-specific OpenCV flags for lower latency
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        deadline = time.time() + self._connect_timeout
        while not cap.isOpened() and time.time() < deadline:
            time.sleep(0.5)

        if not cap.isOpened():
            logger.error("[%s] Failed to connect within %.1fs", self._name, self._connect_timeout)
            cap.release()
            return False

        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (w, h)
        self._cap = cap

        logger.info("[%s] Connected — %dx%d @ %.1f FPS", self._name, w, h, self._fps)
        return True

    def _reader_loop(self) -> None:
        """Background thread: continuously read frames and keep only the latest."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                logger.warning("[%s] Stream lost — reconnecting in %.1fs ...", self._name, self._reconnect_delay)
                time.sleep(self._reconnect_delay)
                if not self._connect():
                    continue

            ok, frame = self._cap.read()
            if not ok:
                logger.warning("[%s] Frame read failed — stream may have dropped.", self._name)
                if self._cap:
                    self._cap.release()
                    self._cap = None
                continue

            # Overwrite the queue with the latest frame (drop stale frames)
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(frame)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened or self._stop_event.is_set():
            return False, None
        try:
            frame = self._frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            logger.debug("[%s] No frame available (stream may be reconnecting).", self._name)
            return False, None

    def release(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.debug("[%s] Released.", self._name)

    def is_opened(self) -> bool:
        return self._opened and not self._stop_event.is_set()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# USB camera source
# ---------------------------------------------------------------------------

class USBCameraSource(VideoSource):
    """
    Video source backed by a locally attached USB or built-in camera.

    Wraps cv2.VideoCapture with an integer index. Use camera_test.py to
    discover available indices on your system before setting config.yaml.
    """

    def __init__(
        self,
        index: int,
        name: str = "usb",
        resolution: tuple[int, int] = (1920, 1080),
        fps: int = 30,
    ) -> None:
        self._index = index
        self._name = name
        self._target_resolution = resolution
        self._target_fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = float(fps)
        self._resolution: tuple[int, int] = resolution

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            logger.error("[%s] Could not open USB camera at index %d", self._name, self._index)
            return False

        # Request target resolution and FPS (camera may not honour exactly)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._resolution = (actual_w, actual_h)
        self._fps = actual_fps or float(self._target_fps)

        logger.info(
            "[%s] Opened USB camera %d — %dx%d @ %.1f FPS",
            self._name, self._index, actual_w, actual_h, self._fps,
        )
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("[%s] Frame read failed.", self._name)
        return ok, frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.debug("[%s] Released.", self._name)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class VideoSourceFactory:
    """Creates the correct VideoSource subclass from a config dict."""

    @staticmethod
    def from_config(cfg: dict[str, Any]) -> VideoSource:
        """
        Build a VideoSource from a single entry under config.yaml `sources`.

        Expected keys:
            type (str):         "file" | "rtsp" | "usb"
            name (str):         human-readable label, e.g. "basket_1"

        Type-specific keys:
            file:
                path (str):     path to the video file
                loop (bool):    whether to loop at EOF, default False
            rtsp:
                url (str):      RTSP URL, e.g. "rtsp://192.168.1.100:554/stream"
                reconnect_delay (float):  seconds between reconnect attempts, default 3.0
                connect_timeout (float):  seconds to wait for initial connect, default 10.0
            usb:
                index (int):    camera device index, e.g. 0
                resolution (list): [width, height], e.g. [1920, 1080]
                fps (int):      target FPS, e.g. 30

        Raises:
            ValueError: if type is unknown or required keys are missing
        """
        source_type = cfg.get("type", "").lower()
        name = cfg.get("name", source_type)

        if source_type == "file":
            path = cfg.get("path")
            if not path:
                raise ValueError(f"[{name}] 'path' is required for file sources")
            return FileVideoSource(
                path=path,
                name=name,
                loop=cfg.get("loop", False),
            )

        if source_type == "rtsp":
            url = cfg.get("url")
            if not url:
                raise ValueError(f"[{name}] 'url' is required for rtsp sources")
            return RTSPVideoSource(
                url=url,
                name=name,
                reconnect_delay=cfg.get("reconnect_delay", 3.0),
                connect_timeout=cfg.get("connect_timeout", 10.0),
            )

        if source_type == "usb":
            index = cfg.get("index")
            if index is None:
                raise ValueError(f"[{name}] 'index' is required for usb sources")
            resolution = cfg.get("resolution", [1920, 1080])
            return USBCameraSource(
                index=int(index),
                name=name,
                resolution=(resolution[0], resolution[1]),
                fps=cfg.get("fps", 30),
            )

        raise ValueError(
            f"Unknown source type '{source_type}'. Expected: 'file', 'rtsp', or 'usb'."
        )


# ---------------------------------------------------------------------------
# Stream chunk recorder
# ---------------------------------------------------------------------------

class StreamChunkRecorder:
    """
    Records a live RTSP/HTTP stream to sequential fixed-duration chunk files on disk.

    A background thread reads every frame from the stream and writes them to
    chunk_{n:04d}.mp4 files. Completed chunk paths are placed on a queue for
    the pipeline to consume at its own pace. This fully decouples capture from
    inference — no frames are dropped due to slow processing, only disk write
    speed is the limit (far faster than 720p @ 25 FPS).

    Usage:
        recorder = StreamChunkRecorder(url, chunk_dir, chunk_seconds=30)
        chunk_queue = recorder.start()
        while True:
            chunk_path = chunk_queue.get(timeout=...)
            if chunk_path is None:          # sentinel: stream ended
                break
            process(chunk_path)             # use FileVideoSource
        recorder.stop()
    """

    def __init__(
        self,
        url: str,
        chunk_dir: str,
        chunk_seconds: float = 30.0,
    ) -> None:
        self._url = url
        self._chunk_dir = Path(chunk_dir)
        self._chunk_seconds = chunk_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._chunk_queue: queue.Queue[str | None] = queue.Queue()

    def start(self) -> "queue.Queue[str | None]":
        """Start the background recording thread. Returns the completed-chunk queue."""
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._record_loop,
            name="stream-chunk-recorder",
            daemon=True,
        )
        self._thread.start()
        return self._chunk_queue

    def stop(self) -> None:
        """Signal recording to stop and wait for the current chunk to finalise."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _record_loop(self) -> None:
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("[StreamChunkRecorder] Could not open stream: %s", self._url)
            self._chunk_queue.put(None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_chunk = max(1, int(fps * self._chunk_seconds))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        logger.info(
            "[StreamChunkRecorder] %dx%d @ %.1f FPS | %.0fs chunks (%d frames each) → %s",
            w, h, fps, self._chunk_seconds, frames_per_chunk, self._chunk_dir,
        )

        chunk_idx = 0
        frame_in_chunk = 0
        writer: cv2.VideoWriter | None = None
        current_path: str | None = None

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                logger.info("[StreamChunkRecorder] Stream ended.")
                break

            if writer is None:
                current_path = str(self._chunk_dir / f"chunk_{chunk_idx:04d}.mp4")
                writer = cv2.VideoWriter(current_path, fourcc, fps, (w, h))
                logger.info("[StreamChunkRecorder] Recording chunk %04d ...", chunk_idx)

            writer.write(frame)
            frame_in_chunk += 1

            if frame_in_chunk >= frames_per_chunk:
                writer.release()
                writer = None
                logger.info("[StreamChunkRecorder] Chunk %04d complete → queued", chunk_idx)
                self._chunk_queue.put(current_path)
                chunk_idx += 1
                frame_in_chunk = 0
                current_path = None

        # Finalise any partial chunk
        if writer is not None:
            writer.release()
            if frame_in_chunk > 0 and current_path:
                logger.info("[StreamChunkRecorder] Final partial chunk %04d queued", chunk_idx)
                self._chunk_queue.put(current_path)

        cap.release()
        self._chunk_queue.put(None)  # sentinel — recording done
