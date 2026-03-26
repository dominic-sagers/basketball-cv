"""
blur_footage.py — post-process a video to blur all faces using YOLOv8-face + SAM2.

YOLOv8-face detects face bounding boxes on keyframes; SAM2 propagates pixel-precise
masks across all frames. No login or gated models required — both download automatically.

This is intentionally a post-processing step — run it on the annotated output
from pipeline_test.py or app.py after a game session.

Setup (one-time, automatic on first run):
    pip install sam2 ultralytics huggingface_hub

Usage:
    # Blur all faces in an annotated game clip
    python src/blur_footage.py store/output/2026-03-25_194841/game_camA.mp4

    # Explicit output path
    python src/blur_footage.py store/output/game.mp4 --output store/output/game_blurred.mp4

    # Adjust blur intensity (default 51 — strong; 31 = softer)
    python src/blur_footage.py store/output/game.mp4 --blur-strength 31

    # Tune face detection sensitivity (lower = more faces, more false positives)
    python src/blur_footage.py store/output/game.mp4 --face-conf 0.2

    # Reduce chunk size if you run out of GPU memory (default: 120 frames)
    python src/blur_footage.py store/output/game.mp4 --chunk-size 60
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.face_blur import FaceBlur

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def blur_video(
    input_path: str,
    output_path: str,
    blur_strength: int,
    face_conf: float,
    face_imgsz: int,
    chunk_size: int,
) -> None:
    """Read input video, blur faces via YOLOv8-face + SAM2, write output."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Cannot open: %s", input_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Input : %s  (%dx%d @ %.1f fps, %d frames)", input_path, w, h, fps, total_frames)
    logger.info("Output: %s", output_path)
    logger.info("Chunk size: %d frames | blur kernel: %d | face conf: %.2f | face imgsz: %d",
                chunk_size, blur_strength, face_conf, face_imgsz)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    fb = FaceBlur(
        blur_strength=blur_strength,
        face_conf=face_conf,
        face_imgsz=face_imgsz,
        chunk_size=chunk_size,
    )

    t_start = time.perf_counter()
    frame_num = 0

    # Read and process in chunks so GPU memory stays bounded for long videos
    while True:
        chunk_frames = []
        for _ in range(chunk_size):
            ok, frame = cap.read()
            if not ok:
                break
            chunk_frames.append(frame)

        if not chunk_frames:
            break

        blurred = fb.blur_frames(chunk_frames)
        for out_frame in blurred:
            writer.write(out_frame)

        frame_num += len(chunk_frames)
        elapsed = time.perf_counter() - t_start
        pct = 100 * frame_num / total_frames if total_frames else 0
        fps_proc = frame_num / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_num) / fps_proc if fps_proc > 0 else 0
        logger.info(
            "Progress: %d/%d frames (%.0f%%)  %.1f fps  ETA %.0fs",
            frame_num, total_frames, pct, fps_proc, eta,
        )

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Done — %d frames in %.1fs (%.1f fps avg)",
        frame_num, elapsed, frame_num / elapsed if elapsed > 0 else 0,
    )
    logger.info("Saved to: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blur faces in a video file using SAM3 segmentation"
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument(
        "--output", metavar="PATH", default=None,
        help="Output path (default: <input_stem>_blurred.mp4 alongside input)",
    )
    parser.add_argument(
        "--blur-strength", type=int, default=51, metavar="K",
        help="Gaussian kernel size — must be odd, larger = stronger blur (default: 51)",
    )
    parser.add_argument(
        "--face-conf", type=float, default=0.25, metavar="F",
        help="YOLOv8-face detection confidence threshold (0–1, default: 0.25). "
             "Lower values detect more faces but may increase false positives.",
    )
    parser.add_argument(
        "--face-imgsz", type=int, default=1280, metavar="PX",
        help="YOLO inference resolution in pixels (default: 1280, must be multiple of 32). "
             "Higher values detect smaller/more distant faces at the cost of speed. "
             "Try 1920 for court-depth faces.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=120, metavar="N",
        help="Frames per SAM2 video session (default: 120). "
             "Reduce if you run out of GPU memory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path = args.output or str(
        input_path.with_stem(input_path.stem + "_blurred").with_suffix(".mp4")
    )

    blur_video(
        input_path=str(input_path),
        output_path=output_path,
        blur_strength=args.blur_strength,
        face_conf=args.face_conf,
        face_imgsz=args.face_imgsz,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
