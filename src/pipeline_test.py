"""
pipeline_test.py — end-to-end pipeline test: source → detect/track → visualize.

Runs the full processing chain on any configured source (file, RTSP, or USB)
and displays the annotated output. Use this to verify detection quality and
tracking stability before building event logic on top.

This script is also the right place to record annotated output clips for
reviewing detection/tracking quality offline.

Usage:
    # Test all sources from config.yaml (one window per source, sequential)
    python src/pipeline_test.py

    # Test a single source by name
    python src/pipeline_test.py --source basket_1

    # Test with a one-off file (no config entry needed)
    python src/pipeline_test.py --file test_footage/basket_1.mp4

    # Detection only (no tracking — faster, useful for initial calibration)
    python src/pipeline_test.py --detect-only

    # Save annotated output to a file
    python src/pipeline_test.py --file test_footage/basket_1.mp4 --save-output output/basket_1_annotated.mp4

    # Headless (no window — useful inside Docker, still prints FPS)
    python src/pipeline_test.py --no-preview

    # Slow down playback to inspect individual detections
    python src/pipeline_test.py --file test_footage/basket_1.mp4 --playback-speed 0.25
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import Detector
from src.tracker import Tracker
from src.video_source import FileVideoSource, RTSPVideoSource, USBCameraSource, VideoSource, VideoSourceFactory
from src.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    source: VideoSource,
    detector: Detector | None,
    tracker: Tracker | None,
    *,
    show_preview: bool = True,
    save_path: str | None = None,
    playback_speed: float = 1.0,
    detect_only: bool = False,
) -> dict:
    """
    Open source, run detect/track loop, draw visualisation.

    Returns a summary dict with frame count, duration, and average FPS.

    Args:
        source:         Any VideoSource (file/RTSP/USB)
        detector:       Detector instance (used when detect_only=True)
        tracker:        Tracker instance (used when detect_only=False)
        show_preview:   Show an OpenCV window
        save_path:      If set, write annotated frames to this video file
        playback_speed: Slow down (0.5 = half speed) for frame-by-frame inspection
        detect_only:    Skip tracking — raw detections only (faster, less info)
    """
    if not source.open():
        logger.error("[%s] Failed to open source.", source.name)
        return {"ok": False}

    viz = Visualizer()
    writer: cv2.VideoWriter | None = None
    frame_number = 0
    t_start = time.perf_counter()
    fps_times: list[float] = []

    # Delay between frames for file playback speed control
    target_frame_ms = int((1000 / source.fps) / playback_speed) if source.fps > 0 else 33

    try:
        while True:
            t_frame = time.perf_counter()
            ok, frame = source.read()
            if not ok:
                logger.info("[%s] Stream ended or source closed.", source.name)
                break

            frame_number += 1

            # ── Inference ──────────────────────────────────────────────
            tracks = None
            detections = None

            if detect_only and detector:
                detections = detector.detect(frame)
            elif tracker:
                tracks = tracker.track(frame)

            # ── FPS measurement ────────────────────────────────────────
            elapsed = time.perf_counter() - t_frame
            if elapsed > 0:
                fps_times.append(1.0 / elapsed)
            recent_fps = sum(fps_times[-30:]) / len(fps_times[-30:]) if fps_times else 0.0

            # ── Visualise ──────────────────────────────────────────────
            annotated = viz.draw(
                frame,
                tracks=tracks,
                detections=detections,
                source_name=source.name,
                fps=recent_fps,
                frame_number=frame_number,
            )

            # ── Output ─────────────────────────────────────────────────
            if show_preview:
                cv2.imshow(f"basketball-cv | {source.name} — Q to quit", annotated)
                wait_ms = max(1, target_frame_ms - int((time.perf_counter() - t_frame) * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    logger.info("Stopped by user (Q).")
                    break

            if save_path and writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, source.fps, (w, h))
                logger.info("Writing annotated output to %s", save_path)
            if writer is not None:
                writer.write(annotated)

            # Log progress every 100 frames
            if frame_number % 100 == 0:
                logger.info(
                    "[%s] Frame %d | %.1f FPS | %d %s",
                    source.name, frame_number, recent_fps,
                    len(tracks) if tracks else len(detections or []),
                    "tracks" if tracks else "detections",
                )

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        source.release()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_number / total_time if total_time > 0 else 0.0

    logger.info(
        "[%s] Done — %d frames in %.1fs = %.1f avg FPS",
        source.name, frame_number, total_time, avg_fps,
    )
    return {
        "ok": True,
        "frames": frame_number,
        "duration_s": total_time,
        "avg_fps": avg_fps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline test: source → detect/track → visualize")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, metavar="NAME",
                        help="Test a single source by name from config.yaml")
    parser.add_argument("--detect-only", action="store_true",
                        help="Run detection only (no ByteTrack) — faster, less info")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable OpenCV window (headless mode)")
    parser.add_argument("--save-output", metavar="PATH", default=None,
                        help="Save annotated video to this file path")
    parser.add_argument("--playback-speed", type=float, default=1.0, metavar="X",
                        help="Playback speed multiplier for file sources (0.25 = quarter speed)")

    # One-off source overrides
    grp = parser.add_argument_group("one-off source overrides")
    grp.add_argument("--file", metavar="PATH", help="Test a video file directly")
    grp.add_argument("--rtsp", metavar="URL", help="Test an RTSP URL directly")
    grp.add_argument("--usb", type=int, metavar="INDEX", help="Test a USB camera by index")

    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config.yaml not found at %s — run from project root", config_path.resolve())
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    tracking_cfg = cfg.get("tracking", {})

    # ── Load model ────────────────────────────────────────────────────────
    detect_only = args.detect_only
    if detect_only:
        logger.info("Detection-only mode (no tracking)")
        model = Detector.from_config(model_cfg)
        model.load()
        detector, tracker = model, None
    else:
        logger.info("Detection + tracking mode (ByteTrack)")
        model = Tracker.from_config(model_cfg, tracking_cfg)
        model.load()
        detector, tracker = None, model

    # ── Resolve sources ───────────────────────────────────────────────────
    if args.file or args.rtsp or args.usb is not None:
        if args.file:
            source: VideoSource = FileVideoSource(args.file, name="file-override", loop=False)
        elif args.rtsp:
            source = RTSPVideoSource(args.rtsp, name="rtsp-override")
        else:
            source = USBCameraSource(args.usb, name=f"usb-{args.usb}")
        sources = [source]
    else:
        source_configs: list[dict] = cfg.get("sources", [])
        if not source_configs:
            logger.error("No 'sources' defined in config.yaml")
            sys.exit(1)
        if args.source:
            source_configs = [s for s in source_configs if s.get("name") == args.source]
            if not source_configs:
                logger.error("No source named '%s' in config.yaml", args.source)
                sys.exit(1)
        sources = [VideoSourceFactory.from_config(s) for s in source_configs]

    # ── Run pipeline on each source ───────────────────────────────────────
    show_preview = not args.no_preview
    for source in sources:
        logger.info("Starting pipeline on source: %s", source.name)

        # When saving multiple sources, append source name to avoid overwriting
        save_path = args.save_output
        if save_path and len(sources) > 1:
            p = Path(save_path)
            save_path = str(p.with_stem(f"{p.stem}_{source.name}"))

        run_pipeline(
            source=source,
            detector=detector,
            tracker=tracker,
            show_preview=show_preview,
            save_path=save_path,
            playback_speed=args.playback_speed,
            detect_only=detect_only,
        )

        # Reset tracker state between sources so IDs don't bleed across
        if tracker is not None:
            tracker.reset()


if __name__ == "__main__":
    main()
