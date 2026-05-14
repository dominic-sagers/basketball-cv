"""
tracker.py — ByteTrack multi-object tracking wrapper.

Wraps Ultralytics' built-in ByteTrack (called via model.track) and returns
structured Track objects with persistent session IDs.

ByteTrack uses two-pass association:
  1. High-confidence detections matched to existing tracks
  2. Low-confidence detections matched to unmatched tracks (recovers occluded players)

Track IDs are stable within a session but reset on pipeline restart.

Usage:
    tracker = Tracker.from_config(cfg["model"], cfg["tracking"])
    tracker.load()
    while running:
        tracks = tracker.track(frame)
        for t in tracks:
            print(t.track_id, t.class_name, t.bbox)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A tracked object with a persistent session ID."""

    track_id: int
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        """Center pixel of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def to_detection(self) -> Detection:
        """Convenience: convert back to a Detection (drops track_id)."""
        return Detection(
            bbox=self.bbox,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
        )


class Tracker:
    """
    Runs detection + ByteTrack in one call via model.track().

    This is the recommended Ultralytics pattern: tracking is handled
    internally by the model and persist=True keeps the Kalman filter
    state across frames for stable IDs.

    The Detector class is not used when Tracker is active — both do
    inference, so only one should run per frame. Use Detector alone
    only when you need raw detections without tracking (e.g. for the
    ball, which gets its own dedicated tracker).
    """

    def __init__(
        self,
        weights: str,
        device: int | str,
        confidence: float,
        iou_threshold: float,
        input_size: int,
        tracker_type: str,
        class_map: dict[int, str] | None = None,
    ) -> None:
        self._weights = weights
        self._device = device
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._tracker_type = tracker_type  # "bytetrack" or "botsort"
        self._class_map = class_map or {0: "person", 32: "sports ball"}
        self._model = None

    def load(self) -> None:
        """Load model weights and warm up. Call once before the pipeline loop."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics not installed — run: pip install ultralytics") from exc

        import torch
        if not torch.cuda.is_available() and self._device != "cpu":
            raise RuntimeError(
                "CUDA not available. Check GPU drivers and PyTorch CUDA build."
            )

        logger.info("Loading tracker model: %s on device %s", self._weights, self._device)
        self._model = YOLO(self._weights)

        # Warm up
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self._model.track(
            dummy,
            device=self._device,
            persist=True,
            verbose=False,
            tracker=f"{self._tracker_type}.yaml",
        )
        logger.info("Tracker loaded and warmed up.")

    def track(self, frame: np.ndarray) -> list[Track]:
        """
        Run detection + tracking on one frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Track objects for all objects with active track IDs.
            Objects detected but not yet assigned a track ID are excluded.
        """
        if self._model is None:
            raise RuntimeError("Tracker not loaded — call tracker.load() first.")

        results = self._model.track(
            frame,
            device=self._device,
            persist=True,                         # keep Kalman state across frames
            conf=self._confidence,
            iou=self._iou_threshold,
            imgsz=self._input_size,
            tracker=f"{self._tracker_type}.yaml",
            verbose=False,
        )

        tracks: list[Track] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            for box in boxes:
                if box.id is None:
                    continue
                cls_id = int(box.cls[0])
                if cls_id not in self._class_map:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                tracks.append(Track(
                    track_id=int(box.id[0]),
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=self._class_map[cls_id],
                    confidence=float(box.conf[0]),
                ))

        return tracks

    def reset(self) -> None:
        """
        Reset tracker state (clears all track IDs).

        Call this between game periods or when restarting a session to
        prevent stale track IDs from being re-used.
        """
        if self._model is not None:
            # Ultralytics resets state by re-running on a blank frame without persist
            dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            self._model.track(dummy, device=self._device, persist=False, verbose=False)
        logger.info("Tracker state reset.")

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any], tracking_cfg: dict[str, Any]) -> "Tracker":
        """Build a Tracker from the model + tracking sections of config.yaml."""
        class_map: dict[int, str] | None = None
        raw_map = model_cfg.get("class_map")
        if raw_map:
            class_map = {int(k): v for k, v in raw_map.items()}

        return cls(
            weights=model_cfg["weights"],
            device=model_cfg.get("device", 0),
            confidence=model_cfg.get("confidence", 0.4),
            iou_threshold=model_cfg.get("iou_threshold", 0.5),
            input_size=model_cfg.get("input_size", 1280),
            tracker_type=tracking_cfg.get("tracker", "bytetrack"),
            class_map=class_map,
        )
