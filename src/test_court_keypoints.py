"""
test_court_keypoints.py — quick test of Roboflow's basketball-court-detection-2
keypoint model against a local video frame.

Downloads the model weights once (cached in ~/.roboflow/) then runs inference
locally — no internet needed after the first run.

Usage:
    python src/test_court_keypoints.py --api-key YOUR_KEY [--video PATH] [--frame N]

Output:
    store/output/court_keypoint_test.jpg  — frame with detected keypoints overlaid
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_WORKSPACE = "roboflow-jvuqo"
MODEL_ID = "basketball-court-detection-2"
MODEL_VERSION = 18  # latest as of 2026-06


def extract_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(frame_index, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx} from {video_path}")
    logger.info("Extracted frame %d / %d from %s", idx, total, Path(video_path).name)
    return frame


def draw_keypoints(
    frame: np.ndarray,
    predictions: list,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    colours = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 255, 0), (0, 128, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
    ]

    for pred in predictions:
        if pred.get("confidence", 0) < conf_threshold:
            continue

        # Bounding box
        x, y, bw, bh = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(out, f"{pred['confidence']:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Keypoints
        keypoints = pred.get("keypoints", [])
        for i, kp in enumerate(keypoints):
            if kp.get("confidence", 0) < conf_threshold:
                continue
            kx, ky = int(kp["x"]), int(kp["y"])
            colour = colours[i % len(colours)]
            cv2.circle(out, (kx, ky), 6, colour, -1)
            cv2.circle(out, (kx, ky), 7, (0, 0, 0), 1)
            label = kp.get("class_name", str(i))
            cv2.putText(out, label, (kx + 8, ky + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    detected = sum(1 for p in predictions if p.get("confidence", 0) >= conf_threshold)
    cv2.putText(out, f"Detections (conf>={conf_threshold}): {detected}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def run(api_key: str, video_path: str, frame_index: int, out_path: str) -> None:
    from roboflow import Roboflow

    logger.info("Authenticating with Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(MODEL_WORKSPACE).project(MODEL_ID)
    model = project.version(MODEL_VERSION).model
    logger.info("Model loaded: %s v%d", MODEL_ID, MODEL_VERSION)

    frame = extract_frame(video_path, frame_index)

    # Save frame to temp file for Roboflow SDK (it needs a file path)
    tmp = Path("/tmp/rf_test_frame.jpg")
    cv2.imwrite(str(tmp), frame)

    logger.info("Running inference...")
    result = model.predict(str(tmp), confidence=20).json()

    preds = result.get("predictions", [])
    logger.info("Got %d prediction(s)", len(preds))
    for i, p in enumerate(preds):
        kps = p.get("keypoints", [])
        logger.info(
            "  [%d] conf=%.2f  keypoints=%d", i, p.get("confidence", 0), len(kps)
        )
        for kp in kps:
            logger.info(
                "       %-30s x=%-6.1f y=%-6.1f conf=%.2f",
                kp.get("class_name", "?"), kp["x"], kp["y"], kp.get("confidence", 0)
            )

    annotated = draw_keypoints(frame, preds)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, annotated)
    logger.info("Result saved to %s", out_path)

    if not preds:
        logger.warning(
            "No keypoints detected — the model may not generalise to this camera angle."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test basketball court keypoint detection")
    parser.add_argument(
        "--api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (defaults to ROBOFLOW_API_KEY in .env)",
    )
    parser.add_argument(
        "--video",
        default="store/footage/church-basketball-01/game _1_church.mp4",
        help="Video file to sample from",
    )
    parser.add_argument("--frame", type=int, default=300, help="Frame index to test")
    parser.add_argument(
        "--out", default="store/output/court_keypoint_test.jpg", help="Output image path"
    )
    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API key found — set ROBOFLOW_API_KEY in .env or pass --api-key")
        sys.exit(1)

    if not Path(args.video).exists():
        logger.error("Video not found: %s", args.video)
        sys.exit(1)

    run(args.api_key, args.video, args.frame, args.out)


if __name__ == "__main__":
    main()
