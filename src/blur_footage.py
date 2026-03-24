"""
blur_footage.py — post-process a video file to blur all faces / player heads.

Runs independently of the CV pipeline — takes any video (raw or annotated)
and outputs a new mp4 with heads blurred. Useful for sharing footage with
the group after a game without exposing identities.

Two modes:

  1. Player-box mode (--log, recommended): pass the JSON log produced by
     `pipeline_test.py --save-log`. Uses stored tracker bounding boxes to blur
     the top 35% of each player box. Works at any camera distance or angle.

  2. Face-detector mode (default fallback): OpenCV DNN res10 SSD with Haar
     cascade fallback. Only reliable for close, frontal faces — NOT recommended
     for wide-angle gym footage.

Usage:
    # Recommended: use tracker log for reliable head blur
    python src/blur_footage.py store/footage/raw_game.mp4 \\
        --log store/output/raw_game_log.json

    # Output to explicit path
    python src/blur_footage.py store/output/5on5-annotated.mp4 \\
        --log store/output/5on5-log.json \\
        --output store/output/5on5-blurred.mp4

    # Adjust blur strength (default 51 — strong; 31 = softer)
    python src/blur_footage.py store/footage/raw_game.mp4 \\
        --log store/output/raw_game_log.json --blur-strength 31

    # Face-detector fallback (no log available)
    python src/blur_footage.py store/footage/raw_game.mp4 --confidence 0.3
"""

from __future__ import annotations

import argparse
import json
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

_PLAYER_CLASSES = {"Player", "Player_Shooting", "person"}


def _load_frame_boxes(log_path: str) -> dict[int, list[tuple[int, int, int, int]]]:
    """
    Load per-frame player bounding boxes from a pipeline JSON log.

    Returns a dict mapping frame_number → list of (x1, y1, x2, y2) for players.
    """
    with open(log_path) as f:
        data = json.load(f)

    boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for entry in data.get("frames", []):
        frame_num = entry["frame"]
        player_boxes = []
        for obj in entry.get("objects", []):
            if obj.get("class") in _PLAYER_CLASSES:
                x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
                player_boxes.append((x1, y1, x2, y2))
        boxes[frame_num] = player_boxes

    total_players = sum(len(v) for v in boxes.values())
    logger.info(
        "Log loaded: %d frames, %d total player detections",
        len(boxes), total_players,
    )
    return boxes


def blur_video(
    input_path: str,
    output_path: str,
    blur_strength: int,
    confidence: float,
    log_path: str | None = None,
    head_fraction: float = 0.35,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Cannot open: %s", input_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    fb = FaceBlur(blur_strength=blur_strength, dnn_confidence=confidence)

    # Load tracker log if provided
    frame_boxes: dict[int, list[tuple[int, int, int, int]]] | None = None
    if log_path:
        logger.info("Using player-box head blur from log: %s", log_path)
        frame_boxes = _load_frame_boxes(log_path)
    else:
        logger.info("No log provided — using face detector backend: %s", fb.backend)
        logger.warning(
            "Face detector is unreliable for wide-angle gym footage. "
            "Run pipeline_test.py with --save-log and pass the result via --log."
        )

    logger.info("Input : %s  (%dx%d, %.1f fps, %d frames)", input_path, w, h, fps, total_frames)
    logger.info("Output: %s", output_path)

    frame_num = 0
    blurred_frames = 0
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_num += 1

        if frame_boxes is not None:
            boxes = frame_boxes.get(frame_num, [])
            if boxes:
                blurred_frames += 1
            result = fb.blur_player_heads(frame, boxes, head_fraction=head_fraction)
        else:
            result = fb.process(frame)

        writer.write(result)

        if frame_num % 100 == 0 or frame_num == total_frames:
            elapsed = time.perf_counter() - t_start
            pct = 100 * frame_num / total_frames if total_frames else 0
            fps_proc = frame_num / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_num) / fps_proc if fps_proc > 0 else 0
            logger.info(
                "Frame %d/%d  (%.0f%%)  %.1f fps  ETA %.0fs",
                frame_num, total_frames, pct, fps_proc, eta,
            )

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Done — %d frames in %.1fs (%.1f fps avg)",
        frame_num, elapsed, frame_num / elapsed if elapsed > 0 else 0,
    )
    if frame_boxes is not None:
        logger.info(
            "Player heads blurred in %d/%d frames (%.0f%%)",
            blurred_frames, frame_num,
            100 * blurred_frames / frame_num if frame_num else 0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blur player heads / faces in a video file for privacy"
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument(
        "--output", metavar="PATH", default=None,
        help="Output path (default: <input_stem>_blurred.mp4 alongside input)",
    )
    parser.add_argument(
        "--log", metavar="PATH", default=None,
        help="JSON log from pipeline_test.py --save-log (enables player-box head blur)",
    )
    parser.add_argument(
        "--head-fraction", type=float, default=0.35, metavar="F",
        help="Fraction of player box height to blur from top (default: 0.35)",
    )
    parser.add_argument(
        "--blur-strength", type=int, default=51, metavar="K",
        help="Gaussian kernel size — must be odd, larger = stronger blur (default: 51)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, metavar="F",
        help="DNN face detection confidence threshold 0–1 (fallback mode only, default: 0.5)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    if args.log and not Path(args.log).exists():
        logger.error("Log file not found: %s", args.log)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_stem(input_path.stem + "_blurred").with_suffix(".mp4"))

    blur_video(
        input_path=str(input_path),
        output_path=output_path,
        blur_strength=args.blur_strength,
        confidence=args.confidence,
        log_path=args.log,
        head_fraction=args.head_fraction,
    )


if __name__ == "__main__":
    main()
