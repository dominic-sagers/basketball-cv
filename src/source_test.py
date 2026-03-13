"""
source_test.py — verify video sources from config.yaml before running the main pipeline.

Opens each configured source, reads frames for a few seconds, reports actual FPS,
and optionally shows a live preview window.

Usage:
    python src/source_test.py                        # test all sources in config.yaml
    python src/source_test.py --source basket_1      # test one source by name
    python src/source_test.py --no-preview           # headless (no OpenCV window)
    python src/source_test.py --duration 10          # run for 10 seconds instead of 5

    # Quick one-off tests (no config entry needed):
    python src/source_test.py --file path/to/clip.mp4
    python src/source_test.py --rtsp rtsp://192.168.1.100:554/stream
    python src/source_test.py --usb 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.video_source import FileVideoSource, RTSPVideoSource, USBCameraSource, VideoSource, VideoSourceFactory

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def test_source(source: VideoSource, duration_sec: int = 5, show_preview: bool = True) -> bool:
    """
    Open a source, read frames for duration_sec, report FPS and resolution.

    Returns True if the source worked (at least one frame read successfully).
    """
    print(f"\n{'─' * 60}")
    print(f"  Source : {source.name}")
    print(f"  Type   : {type(source).__name__}")

    if not source.open():
        print(f"  FAIL   — could not open source")
        return False

    print(f"  Info   : {source.resolution[0]}x{source.resolution[1]} @ {source.fps:.1f} FPS (nominal)")

    frame_count = 0
    fail_count = 0
    start = time.perf_counter()
    window_title = f"{source.name} — press Q to stop"

    while (time.perf_counter() - start) < duration_sec:
        ok, frame = source.read()

        if not ok:
            fail_count += 1
            if fail_count > 30:
                print(f"  FAIL   — too many consecutive read failures")
                break
            time.sleep(0.01)
            continue

        fail_count = 0
        frame_count += 1

        if show_preview and frame is not None:
            elapsed = time.perf_counter() - start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame,
                f"{source.name}  |  {fps:.1f} FPS  |  {frame.shape[1]}x{frame.shape[0]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Stopped by user (Q).")
                break

    elapsed = time.perf_counter() - start
    source.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print(f"  FAIL   — 0 frames read in {elapsed:.1f}s")
        return False

    actual_fps = frame_count / elapsed
    print(f"  PASS   — {frame_count} frames in {elapsed:.1f}s = {actual_fps:.1f} actual FPS")
    if actual_fps < 25:
        print(f"  WARN   — FPS below 25; pipeline may struggle at 30 FPS")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test video sources from config.yaml or one-off overrides"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--source", default=None, metavar="NAME",
        help="Test a single source by name (as defined in config.yaml)"
    )
    parser.add_argument(
        "--duration", type=int, default=5,
        help="Seconds to read from each source (default: 5)"
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable OpenCV preview window (useful for headless/Docker environments)"
    )

    # One-off source overrides — skip config.yaml entirely
    group = parser.add_argument_group("one-off source overrides (skip config.yaml)")
    group.add_argument("--file", metavar="PATH", help="Test a video file directly")
    group.add_argument("--rtsp", metavar="URL", help="Test an RTSP URL directly")
    group.add_argument("--usb", type=int, metavar="INDEX", help="Test a USB camera by index")

    args = parser.parse_args()
    show_preview = not args.no_preview

    # ── One-off overrides ───────────────────────────────────────────────────
    if args.file or args.rtsp or args.usb is not None:
        if args.file:
            source: VideoSource = FileVideoSource(args.file, name="file-override", loop=False)
        elif args.rtsp:
            source = RTSPVideoSource(args.rtsp, name="rtsp-override")
        else:
            source = USBCameraSource(args.usb, name=f"usb-{args.usb}")
        success = test_source(source, duration_sec=args.duration, show_preview=show_preview)
        sys.exit(0 if success else 1)

    # ── Config-driven ───────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config.yaml not found at %s", config_path.resolve())
        logger.error("Run from the project root: python src/source_test.py")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    source_configs: list[dict] = cfg.get("sources", [])
    if not source_configs:
        logger.error("No 'sources' defined in config.yaml")
        sys.exit(1)

    # Filter to a single source if --source was given
    if args.source:
        source_configs = [s for s in source_configs if s.get("name") == args.source]
        if not source_configs:
            logger.error("No source named '%s' found in config.yaml", args.source)
            sys.exit(1)

    print(f"\nTesting {len(source_configs)} source(s) from {config_path.resolve()}")
    print("(Use --no-preview for headless environments, --duration N to change test length)\n")

    results: dict[str, bool] = {}
    for src_cfg in source_configs:
        try:
            source = VideoSourceFactory.from_config(src_cfg)
        except ValueError as exc:
            logger.error("Bad source config: %s", exc)
            results[src_cfg.get("name", "?")] = False
            continue
        results[source.name] = test_source(source, duration_sec=args.duration, show_preview=show_preview)

    # Summary
    print(f"\n{'═' * 60}")
    print("  Results")
    print(f"{'─' * 60}")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False
    print(f"{'═' * 60}\n")

    if not all_pass:
        print("One or more sources failed. Check the log above for details.")
        print("Common fixes:")
        print("  file  — verify the path exists and the file is a valid video")
        print("  rtsp  — check the URL, ensure the camera is on the network")
        print("  usb   — run python src/camera_test.py to find working indices")
    else:
        print("All sources OK. Update config.yaml and proceed to tracker.py.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
