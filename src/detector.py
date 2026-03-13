"""
detector.py — YOLOv11 inference wrapper for the basketball CV pipeline.

Returns structured Detection objects rather than raw Ultralytics result tensors,
so the rest of the pipeline never needs to import ultralytics directly.

Classes detected from the base COCO model:
    - person      (class 0)
    - sports ball (class 32)
Custom fine-tuned model additionally detects:
    - hoop        (class depends on training config)

Usage:
    detector = Detector.from_config(cfg["model"])
    detections = detector.detect(frame)
    for d in detections:
        print(d.class_name, d.confidence, d.bbox)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs the pipeline cares about (base model).
# Fine-tuned model may remap these — set via config.yaml model.class_map.
_DEFAULT_CLASS_MAP: dict[int, str] = {
    0: "person",
    32: "sports ball",
}


@dataclass
class Detection:
    """A single detected object in one frame."""

    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixel coords
    class_id: int
    class_name: str
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        """Center pixel of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Bounding box area in pixels²."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class Detector:
    """
    Wraps a YOLOv11 model and returns Detection objects.

    CUDA is required — raises RuntimeError on CPU-only environments.
    Set device="cpu" in config only for quick smoke tests (will not meet FPS target).
    """

    def __init__(
        self,
        weights: str,
        device: int | str,
        confidence: float,
        iou_threshold: float,
        input_size: int,
        class_map: dict[int, str] | None = None,
    ) -> None:
        self._weights = weights
        self._device = device
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._class_map = class_map or _DEFAULT_CLASS_MAP
        self._model = None

    def load(self) -> None:
        """Load model weights and warm up the GPU. Call once before the pipeline loop."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics not installed — run: pip install ultralytics") from exc

        import torch
        if not torch.cuda.is_available() and self._device != "cpu":
            raise RuntimeError(
                "CUDA not available. Check GPU drivers and PyTorch CUDA build. "
                "Set device='cpu' in config.yaml only for testing (will be slow)."
            )

        logger.info("Loading model: %s on device %s", self._weights, self._device)
        self._model = YOLO(self._weights)

        # Warm-up pass — prevents first-frame latency spike from CUDA graph compilation
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self._model.predict(dummy, device=self._device, verbose=False)
        self._model.predict(dummy, device=self._device, verbose=False)
        logger.info("Model loaded and warmed up.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on one frame and return filtered detections.

        Only returns classes present in the class_map (person, ball, hoop).
        Skips all other COCO classes to reduce noise.

        Args:
            frame: BGR image as numpy array (OpenCV format)

        Returns:
            List of Detection objects, empty if nothing relevant found.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded — call detector.load() first.")

        results = self._model.predict(
            frame,
            device=self._device,
            conf=self._confidence,
            iou=self._iou_threshold,
            imgsz=self._input_size,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self._class_map:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=self._class_map[cls_id],
                    confidence=float(box.conf[0]),
                ))

        return detections

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Detector":
        """Build a Detector from the model section of config.yaml."""
        class_map: dict[int, str] | None = None
        raw_map = cfg.get("class_map")
        if raw_map:
            class_map = {int(k): v for k, v in raw_map.items()}

        return cls(
            weights=cfg["weights"],
            device=cfg.get("device", 0),
            confidence=cfg.get("confidence", 0.4),
            iou_threshold=cfg.get("iou_threshold", 0.5),
            input_size=cfg.get("input_size", 1280),
            class_map=class_map,
        )
