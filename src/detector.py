"""
detector.py — detection inference wrapper for the basketball CV pipeline.

Supports two backends, selected via config.yaml model.backend:
  - "yolo"   : Ultralytics YOLOv11 (default, fine-tuned weights in store/)
  - "rfdetr" : Roboflow RF-DETR-L (transformer backbone, better occlusion handling)

Both backends return the same Detection dataclass so the rest of the pipeline
never needs to know which model is running.

Usage:
    detector = Detector.from_config(cfg["model"])
    detector.load()
    detections = detector.detect(frame)   # frame is BGR numpy array
    for d in detections:
        print(d.class_name, d.confidence, d.bbox)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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
    Backend-agnostic detection wrapper.

    Instantiate via Detector.from_config(cfg["model"]) — it reads the
    `backend` field and configures the right engine automatically.

    CUDA is expected for real-time use. Set device="cpu" only for smoke tests.
    """

    def __init__(
        self,
        weights: str,
        device: int | str,
        confidence: float,
        iou_threshold: float,
        input_size: int,
        class_map: dict[int, str] | None = None,
        backend: str = "yolo",
    ) -> None:
        self._weights = weights
        self._device = device
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._class_map = class_map or _DEFAULT_CLASS_MAP
        self._backend = backend.lower()
        self._model = None

        if self._backend not in ("yolo", "rfdetr"):
            raise ValueError(f"Unknown backend '{backend}'. Use 'yolo' or 'rfdetr'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights and warm up the GPU. Call once before the pipeline loop."""
        import torch
        if not torch.cuda.is_available() and self._device != "cpu":
            raise RuntimeError(
                "CUDA not available. Check GPU drivers and PyTorch CUDA build. "
                "Set device='cpu' in config.yaml only for testing (will be slow)."
            )

        if self._backend == "yolo":
            self._load_yolo()
        else:
            self._load_rfdetr()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on one BGR frame and return filtered detections.

        Only classes present in class_map are returned; all others are dropped.

        Args:
            frame: BGR image as numpy array (OpenCV format).

        Returns:
            List of Detection objects, empty if nothing relevant found.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded — call detector.load() first.")

        if self._backend == "yolo":
            return self._detect_yolo(frame)
        else:
            return self._detect_rfdetr(frame)

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
            input_size=cfg.get("input_size", 960),
            class_map=class_map,
            backend=cfg.get("backend", "yolo"),
        )

    # ------------------------------------------------------------------
    # YOLO backend
    # ------------------------------------------------------------------

    def _load_yolo(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics not installed — run: pip install ultralytics") from exc

        logger.info("[YOLO] Loading %s on device %s", self._weights, self._device)
        self._model = YOLO(self._weights)

        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self._model.predict(dummy, device=self._device, verbose=False)
        self._model.predict(dummy, device=self._device, verbose=False)
        logger.info("[YOLO] Model loaded and warmed up.")

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
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

    # ------------------------------------------------------------------
    # RF-DETR backend
    # ------------------------------------------------------------------

    def _load_rfdetr(self) -> None:
        try:
            from rfdetr import RFDETRLarge
        except ImportError as exc:
            raise ImportError("rfdetr not installed — run: pip install rfdetr") from exc

        logger.info("[RF-DETR] Loading RF-DETR-L on device %s", self._device)

        num_classes = len(self._class_map)
        if self._weights and self._weights.endswith(".pth"):
            # Fine-tuned weights
            self._model = RFDETRLarge(
                pretrained=False,
                num_classes=num_classes,
                resolution=self._input_size,
            )
            import torch
            device_str = f"cuda:{self._device}" if isinstance(self._device, int) else self._device
            state = torch.load(self._weights, map_location=device_str)
            self._model.load_state_dict(state)
            logger.info("[RF-DETR] Loaded fine-tuned weights from %s", self._weights)
        else:
            # Pretrained COCO weights (before fine-tuning)
            self._model = RFDETRLarge(pretrained=True, resolution=self._input_size)
            logger.info("[RF-DETR] Loaded pretrained COCO weights.")

        # Warm up
        from PIL import Image
        dummy = Image.fromarray(np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8))
        self._model.predict(dummy, threshold=self._confidence)
        self._model.predict(dummy, threshold=self._confidence)
        logger.info("[RF-DETR] Model loaded and warmed up.")

    def _detect_rfdetr(self, frame: np.ndarray) -> list[Detection]:
        from PIL import Image

        # RF-DETR expects RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        pil_image = Image.fromarray(rgb)

        sv_detections = self._model.predict(pil_image, threshold=self._confidence)

        detections: list[Detection] = []
        if sv_detections is None or len(sv_detections) == 0:
            return detections

        for i in range(len(sv_detections)):
            cls_id = int(sv_detections.class_id[i])
            if cls_id not in self._class_map:
                continue
            x1, y1, x2, y2 = (int(v) for v in sv_detections.xyxy[i])
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=self._class_map[cls_id],
                confidence=float(sv_detections.confidence[i]),
            ))

        return detections
