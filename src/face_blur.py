"""
face_blur.py — SAM2-powered face detection, segmentation, and blur.

Pipeline:
  1. YOLOv8-face (arnabdhar/YOLOv8-Face-Detection on HuggingFace) detects face
     bounding boxes on keyframes. Works at gym distances and non-frontal angles —
     far better recall than the old res10 SSD (~80% AP hard vs ~75%).
  2. SAM2 video predictor (facebook/sam2.1-hiera-large) takes those bboxes as
     prompts and propagates pixel-precise face masks across the chunk. Faces are
     tracked smoothly even when moving fast or partially occluded.
  3. Gaussian blur is applied only to the masked pixels.

Both models auto-download on first use (HuggingFace cache). No login required —
neither model is gated.

This is intentionally a post-processing tool. Per-frame real-time inference is
not practical at 30+ FPS; use blur_footage.py after a game session.

Usage:
    fb = FaceBlur()
    blurred = fb.blur_frames(frames_bgr)   # list[np.ndarray] → list[np.ndarray]
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_FACE_WEIGHTS_REPO = "arnabdhar/YOLOv8-Face-Detection"
_FACE_WEIGHTS_FILE = "model.pt"
_SAM2_MODEL_ID     = "facebook/sam2.1-hiera-large"

# Number of keyframes per chunk used for face detection.
# SAM2 propagates from these prompts so we don't need to detect on every frame.
_KEYFRAMES_PER_CHUNK = 3


class FaceBlur:
    """
    Detect and blur faces in video frames using YOLOv8-face + SAM2.

    Args:
        blur_strength:  Gaussian kernel size (odd int). 51 = strong; 31 = softer.
        face_conf:      YOLOv8-face detection confidence threshold (0–1).
                        Lower values catch more faces at the cost of false positives.
        face_imgsz:     YOLO inference resolution (px). Default 1280. Higher values
                        detect smaller/more distant faces. 1920 catches court-depth
                        faces but uses more VRAM. Must be a multiple of 32.
        chunk_size:     Frames per SAM2 video session. Reduce if you hit OOM.
        device:         Torch device string. Defaults to "cuda" if available.
    """

    def __init__(
        self,
        blur_strength: int = 51,
        face_conf: float = 0.25,
        face_imgsz: int = 1280,
        chunk_size: int = 120,
        device: str | None = None,
    ) -> None:
        self._blur_k    = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self._face_conf = face_conf
        self._face_imgsz = face_imgsz
        self._chunk_size = chunk_size
        self._device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._yolo      = None   # loaded lazily
        self._sam2      = None   # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return True

    def blur_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Blur all faces in a list of BGR frames using YOLOv8-face + SAM2.

        Frames are split into fixed-size chunks. Each chunk writes temporary
        JPEG files (required by SAM2), runs detection + propagation, then
        applies per-pixel Gaussian blur through the face masks.

        Args:
            frames: List of BGR numpy arrays, all the same resolution.

        Returns:
            List of BGR frames with detected faces blurred.
        """
        if not frames:
            return []

        self._load_models()

        results: list[np.ndarray] = []
        n_chunks = (len(frames) + self._chunk_size - 1) // self._chunk_size
        for ci, start in enumerate(range(0, len(frames), self._chunk_size)):
            chunk = frames[start : start + self._chunk_size]
            logger.info(
                "FaceBlur: chunk %d/%d — %d frames …", ci + 1, n_chunks, len(chunk)
            )
            results.extend(self._blur_chunk(chunk))

        faces_changed = sum(1 for a, b in zip(frames, results) if not np.array_equal(a, b))
        logger.info(
            "FaceBlur: complete — faces blurred in %d/%d frames", faces_changed, len(frames)
        )
        return results

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Stub for live-pipeline compatibility — SAM2 is a post-processing tool."""
        logger.warning(
            "FaceBlur.process() called on a single frame — SAM2 requires batch video input. "
            "Run blur_footage.py for post-processing."
        )
        return frame

    def blur_player_heads(
        self,
        frame: np.ndarray,
        player_boxes: Sequence,
        head_fraction: float = 0.35,
    ) -> np.ndarray:
        """Removed — use blur_footage.py instead."""
        logger.warning(
            "FaceBlur.blur_player_heads() is no longer supported. "
            "Run blur_footage.py for SAM2-based face blur."
        )
        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Lazy-load YOLOv8-face and SAM2 on first call."""
        if self._yolo is not None:
            return

        # ── YOLOv8-face ───────────────────────────────────────────────
        try:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics and huggingface_hub are required. "
                "Install with: pip install ultralytics huggingface_hub"
            ) from e

        logger.info("FaceBlur: downloading YOLOv8-face weights from HuggingFace …")
        weights_path = hf_hub_download(
            repo_id=_FACE_WEIGHTS_REPO,
            filename=_FACE_WEIGHTS_FILE,
        )
        self._yolo = YOLO(weights_path)
        logger.info("FaceBlur: YOLOv8-face ready.")

        # ── SAM2 ──────────────────────────────────────────────────────
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError as e:
            raise ImportError(
                "sam2 is required. Install with: pip install sam2"
            ) from e

        logger.info("FaceBlur: loading SAM2 from '%s' …", _SAM2_MODEL_ID)
        self._sam2 = SAM2VideoPredictor.from_pretrained(_SAM2_MODEL_ID)
        logger.info("FaceBlur: SAM2 ready.")

    def _detect_faces(self, frame: np.ndarray) -> list[list[float]]:
        """Run YOLOv8-face on one frame. Returns list of [x1,y1,x2,y2] boxes.

        Higher imgsz catches small/distant faces — YOLO internally resizes the
        frame to this resolution before inference, giving more pixels per face.
        """
        results = self._yolo(
            frame, verbose=False, conf=self._face_conf, imgsz=self._face_imgsz
        )[0]
        return [
            [float(v) for v in box.xyxy[0].tolist()]
            for box in results.boxes
        ]

    def _blur_chunk(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Process one chunk:
          1. Write frames as JPEGs to a temp dir (SAM2 requires file-backed input).
             Frames are downscaled to ≤1024px on the longest side before writing —
             SAM2 internally resizes to 1024 anyway, so this cuts I/O significantly.
          2. Detect faces on evenly-spaced keyframes (original resolution for accuracy).
          3. Init SAM2 state and add scaled bbox prompts.
          4. Propagate masks across all frames.
          5. Upscale masks to original resolution and apply Gaussian blur.
        """
        from PIL import Image

        orig_h, orig_w = frames[0].shape[:2]
        scale = min(1024 / max(orig_h, orig_w), 1.0)
        sam_w = int(orig_w * scale)
        sam_h = int(orig_h * scale)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write frames as JPEGs at SAM2-native resolution to minimise I/O
            for i, frame in enumerate(frames):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale < 1.0:
                    rgb = cv2.resize(rgb, (sam_w, sam_h), interpolation=cv2.INTER_AREA)
                Image.fromarray(rgb).save(
                    os.path.join(tmp_dir, f"{i:06d}.jpg"), quality=92
                )

            # Autocast context — bfloat16 on CUDA, float32 on CPU
            if self._device != "cpu":
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                amp_ctx = contextlib.nullcontext()

            with torch.inference_mode(), amp_ctx:
                state = self._sam2.init_state(
                    tmp_dir,
                    offload_video_to_cpu=True,   # keeps frames in RAM, not VRAM
                    offload_state_to_cpu=False,
                )

                # Detect faces on evenly-spaced keyframes and add as SAM2 prompts.
                # Detection runs on original-resolution frames for accuracy; bboxes
                # are then scaled to match the downscaled SAM2 input size.
                stride = max(1, len(frames) // _KEYFRAMES_PER_CHUNK)
                obj_id = 1
                for kf in range(0, len(frames), stride):
                    for bbox in self._detect_faces(frames[kf]):
                        # Scale bbox from original resolution to SAM2 input resolution
                        scaled = [
                            bbox[0] * scale, bbox[1] * scale,
                            bbox[2] * scale, bbox[3] * scale,
                        ]
                        self._sam2.add_new_points_or_box(
                            state,
                            frame_idx=kf,
                            obj_id=obj_id,
                            box=scaled,
                        )
                        obj_id += 1

                if obj_id == 1:
                    # No faces detected in this chunk
                    logger.debug("FaceBlur: no faces in chunk — skipping SAM2")
                    self._sam2.reset_state(state)
                    return list(frames)

                logger.debug(
                    "FaceBlur: %d face prompts across %d keyframes — propagating …",
                    obj_id - 1, min(_KEYFRAMES_PER_CHUNK, len(frames)),
                )

                # Propagate and union all face masks per frame.
                # SAM2 returns logits (pre-sigmoid), so threshold at 0.0 to get
                # binary foreground masks (logit > 0  ↔  sigmoid > 0.5).
                frame_masks: dict[int, np.ndarray] = {}
                for frame_idx, _obj_ids, masks in self._sam2.propagate_in_video(state):
                    # masks: (N, 1, H, W) float logits
                    binary = (masks.squeeze(1) > 0.0)        # (N, H, W) bool
                    combined = binary.any(dim=0)              # (H, W) bool — union of faces
                    mask_np = combined.cpu().numpy()
                    # Upscale mask back to original frame resolution if we downscaled
                    if scale < 1.0:
                        mask_np = cv2.resize(
                            mask_np.astype(np.uint8),
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    frame_masks[frame_idx] = mask_np

                self._sam2.reset_state(state)

        # Apply Gaussian blur through the collected masks
        out: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            mask = frame_masks.get(i)
            if mask is not None and mask.any():
                blurred = cv2.GaussianBlur(frame, (self._blur_k, self._blur_k), 0)
                result = frame.copy()
                result[mask] = blurred[mask]
                out.append(result)
            else:
                out.append(frame)
        return out
