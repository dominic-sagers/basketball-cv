"""
detection_test.py — verify YOLOv11 + CUDA inference speed before building the pipeline.

Checks:
  1. CUDA is available and on the right GPU
  2. YOLOv11 loads and runs on GPU
  3. Inference speed meets the ≥30 FPS minimum (targeting ≥60 FPS)

Usage:
    python src/detection_test.py
    python src/detection_test.py --model yolo11s.pt   # try a smaller model
    python src/detection_test.py --source 0           # benchmark on live camera feed
"""

import argparse
import time
import sys

import cv2
import numpy as np


def check_cuda() -> None:
    """Verify CUDA is available and print GPU info."""
    try:
        import torch
    except ImportError:
        print("ERROR — PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR — CUDA not available. Check your PyTorch install and GPU drivers.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {free_mem / 1e9:.1f}GB free / {total_mem / 1e9:.1f}GB total")


def benchmark_inference(model_path: str, source, n_frames: int = 100) -> float:
    """Run inference on N frames and return actual FPS."""
    from ultralytics import YOLO

    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)

    # Warm up (first inference is slow due to CUDA graph compilation)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device=0, verbose=False)
    model.predict(dummy, device=0, verbose=False)
    print("Warm-up done.")

    if isinstance(source, int):
        # Live camera
        cap = cv2.VideoCapture(source)
        frames = []
        for _ in range(n_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        if not frames:
            print(f"ERROR — could not read frames from camera {source}")
            sys.exit(1)
        print(f"Captured {len(frames)} frames from camera {source}")
    else:
        # Synthetic frames (default)
        frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(n_frames)]

    print(f"Benchmarking {len(frames)} frames at 1920x1080...")
    start = time.perf_counter()
    for frame in frames:
        model.predict(frame, device=0, verbose=False, imgsz=1280)
    elapsed = time.perf_counter() - start

    fps = len(frames) / elapsed
    print(f"\nResults: {fps:.1f} FPS ({elapsed:.2f}s for {len(frames)} frames)")

    if fps >= 60:
        print("PASS — well above 60 FPS target")
    elif fps >= 30:
        print("PASS — above 30 FPS minimum (consider yolo11s.pt or lower imgsz for headroom)")
    else:
        print("FAIL — below 30 FPS minimum. Try: smaller model (yolo11n/s), lower imgsz (640), or check GPU utilization")

    return fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv11 inference speed")
    parser.add_argument("--model", default="yolo11m.pt", help="Model weights path")
    parser.add_argument("--source", default=None, help="Camera index (int) or None for synthetic frames")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark")
    args = parser.parse_args()

    print("=== CUDA check ===")
    check_cuda()

    print("\n=== Inference benchmark ===")
    source = int(args.source) if args.source is not None else None
    benchmark_inference(args.model, source, args.frames)


if __name__ == "__main__":
    main()
