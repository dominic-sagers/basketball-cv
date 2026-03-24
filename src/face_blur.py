"""
face_blur.py — detect and blur faces for privacy.

Two modes:

  1. Player-box head blur (default, recommended) — blurs the upper 35% of each
     player bounding box passed from the tracker. No face detector needed; works
     at any camera angle, distance, or lighting. Use this when tracker output is
     available (live pipeline or post-processing with a saved JSON log).

  2. Face-detector blur (fallback) — two sub-backends:
       a. OpenCV DNN (res10_300x300 SSD) — ~2 MB model auto-downloaded to
          store/weights/face/ on first use. Decent for close, frontal faces.
       b. OpenCV Haar cascade — zero downloads, frontal faces only.

     NOTE: Both detector backends struggle with small, distant, or angled faces
     (e.g. camera at half-court height). Use player-box mode for gym footage.

Usage:
    fb = FaceBlur()

    # Preferred — blur head region of known player boxes:
    player_boxes = [(x1, y1, x2, y2), ...]   # from tracker
    blurred = fb.blur_player_heads(frame, player_boxes)

    # Fallback — run face detector:
    blurred = fb.process(frame)
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNN model — OpenCV res10 SSD face detector (Caffe)
# ---------------------------------------------------------------------------

_DNN_MODEL_DIR  = Path("store/weights/face")
_PROTOTXT_PATH  = _DNN_MODEL_DIR / "deploy.prototxt"
_CAFFEMODEL_PATH = _DNN_MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)


def _download_dnn_model() -> bool:
    """Download res10 SSD face model if not already present. Returns True on success."""
    _DNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for path, url in [(_PROTOTXT_PATH, _PROTOTXT_URL), (_CAFFEMODEL_PATH, _CAFFEMODEL_URL)]:
        if not path.exists():
            logger.info("Downloading face model: %s", path.name)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", path.name, exc)
                return False

    return True


# ---------------------------------------------------------------------------
# FaceBlur
# ---------------------------------------------------------------------------

class FaceBlur:
    """
    Detect and blur all faces in a frame.

    Args:
        blur_strength:    Gaussian kernel size (odd integer, larger = more blurred).
                          51 gives a strong privacy blur; 31 is softer.
        dnn_confidence:   Minimum detection confidence for the DNN backend (0–1).
        haar_scale:       Scale factor for Haar cascade multiscale detection.
        haar_min_neighbors: Strictness of Haar detection (higher = fewer false positives).
    """

    def __init__(
        self,
        blur_strength: int = 51,
        dnn_confidence: float = 0.5,
        haar_scale: float = 1.1,
        haar_min_neighbors: int = 5,
    ) -> None:
        self._blur_k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self._dnn_conf = dnn_confidence
        self._haar_scale = haar_scale
        self._haar_min_neighbors = haar_min_neighbors

        self._net: cv2.dnn.Net | None = None
        self._haar: cv2.CascadeClassifier | None = None
        self._backend: str = "none"

        self._init_backend()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        # Try DNN first
        if _CAFFEMODEL_PATH.exists() and _PROTOTXT_PATH.exists():
            self._load_dnn()
        elif _download_dnn_model():
            self._load_dnn()

        if self._net is None:
            self._load_haar()

    def _load_dnn(self) -> None:
        try:
            self._net = cv2.dnn.readNetFromCaffe(
                str(_PROTOTXT_PATH), str(_CAFFEMODEL_PATH)
            )
            self._backend = "dnn"
            logger.info("FaceBlur: using OpenCV DNN backend (res10 SSD)")
        except Exception as exc:
            logger.warning("FaceBlur: DNN load failed (%s), falling back to Haar", exc)

    def _load_haar(self) -> None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(xml)
        if self._haar.empty():
            logger.error("FaceBlur: Haar cascade failed to load — face blur disabled")
            self._backend = "none"
        else:
            self._backend = "haar"
            logger.info("FaceBlur: using OpenCV Haar cascade backend (frontal faces only)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Active backend: 'dnn', 'haar', or 'none'."""
        return self._backend

    @property
    def enabled(self) -> bool:
        return self._backend != "none"

    def blur_player_heads(
        self,
        frame: np.ndarray,
        player_boxes: list[tuple[int, int, int, int]],
        head_fraction: float = 0.35,
    ) -> np.ndarray:
        """
        Blur the top `head_fraction` of each player bounding box.

        This is the recommended mode for gym footage where faces are small,
        distant, or non-frontal — no face detector is needed.

        Args:
            frame:         BGR frame to blur.
            player_boxes:  List of (x1, y1, x2, y2) bounding boxes for players.
            head_fraction: Fraction of box height to blur from the top (default 0.35).

        Returns:
            Copy of frame with head regions blurred.
        """
        out = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        for (x1, y1, x2, y2) in player_boxes:
            box_h = y2 - y1
            if box_h <= 0:
                continue
            head_h = max(1, int(box_h * head_fraction))
            # Clamp to frame bounds
            rx1 = max(0, x1)
            ry1 = max(0, y1)
            rx2 = min(w_frame, x2)
            ry2 = min(h_frame, y1 + head_h)
            roi = out[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue
            out[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(
                roi, (self._blur_k, self._blur_k), 0
            )
        return out

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a copy of `frame` with all detected faces blurred using the
        active face-detector backend (DNN or Haar).

        NOTE: For gym footage with distant/angled faces, prefer blur_player_heads().
        If no backend is available, returns the frame unchanged.
        """
        if not self.enabled:
            return frame

        out = frame.copy()
        bboxes = self._detect(frame)
        for (x1, y1, x2, y2) in bboxes:
            roi = out[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            out[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi, (self._blur_k, self._blur_k), 0
            )
        return out

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._backend == "dnn":
            return self._detect_dnn(frame)
        if self._backend == "haar":
            return self._detect_haar(frame)
        return []

    def _detect_dnn(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False,
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._dnn_conf:
                continue
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _detect_haar(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=self._haar_scale,
            minNeighbors=self._haar_min_neighbors,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return []
        return [(x, y, x + w, y + h) for (x, y, w, h) in faces]
