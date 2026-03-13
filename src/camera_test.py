"""
camera_test.py — verify camera feeds before starting the main pipeline.

Run this first to confirm each camera is accessible, check resolution/FPS,
and identify the correct device index for config.yaml.

Usage:
    python src/camera_test.py
    python src/camera_test.py --index 2   # test a specific camera index
"""

import argparse
import time
import cv2


def test_camera(index: int, duration_sec: int = 5) -> None:
    """Open a camera, display the feed, and report actual FPS."""
    print(f"\nTesting camera index {index}...")
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"  FAIL — could not open camera {index}")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Opened: {int(width)}x{int(height)} @ {reported_fps:.0f}fps (reported)")

    frame_count = 0
    start = time.time()

    while (time.time() - start) < duration_sec:
        ret, frame = cap.read()
        if not ret:
            print("  WARNING — frame read failed")
            break
        frame_count += 1
        elapsed = time.time() - start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame,
            f"Camera {index} | {fps:.1f} FPS | {int(width)}x{int(height)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow(f"Camera {index} — press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    actual_fps = frame_count / (time.time() - start)
    print(f"  PASS — {frame_count} frames in {duration_sec}s = {actual_fps:.1f} actual FPS")
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test camera feeds")
    parser.add_argument("--index", type=int, default=None, help="Test a single camera index")
    parser.add_argument("--max", type=int, default=4, help="Max indices to scan (default: 4)")
    args = parser.parse_args()

    if args.index is not None:
        test_camera(args.index)
    else:
        print(f"Scanning camera indices 0–{args.max - 1}...")
        for i in range(args.max):
            test_camera(i)
        print("\nDone. Update config.yaml with the working camera indices.")


if __name__ == "__main__":
    main()
