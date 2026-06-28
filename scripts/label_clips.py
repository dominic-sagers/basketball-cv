"""
label_clips.py — run Ball_in_Basket detection on each highlight clip.

Loads the model once, processes every Nth frame of each clip, and applies
the same confidence + consecutive-frame thresholds as game_state.py.

Writes results to <highlights_dir>/clip_detections.json so Claude can
upload them to the annotation spreadsheet via the Sheets MCP.

Usage:
    python scripts/label_clips.py                           # church-basketball-03 default
    python scripts/label_clips.py --run-dir store/output/church-basketball-03_2026-06-27_143017
    python scripts/label_clips.py --stride 3 --skip 1 2 3  # skip warm-up clips
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import yaml


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _clip_number(path: Path) -> int:
    """Extract the 3-digit clip number from a highlight filename."""
    # e.g. church-basketball-03_raw_highlight_005_teamA_239s.mp4 → 5
    part = path.name.split("_highlight_")[1]
    return int(part.split("_")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Ball_in_Basket in highlight clips")
    parser.add_argument("--run-dir", default=None,
                        help="Path to pipeline run directory (default: newest in store/output/)")
    parser.add_argument("--stride", type=int, default=3,
                        help="Process every Nth frame (default: 3 — ~10 fps from 30 fps source)")
    parser.add_argument("--skip", type=int, nargs="*", default=[1, 2, 3],
                        help="Clip numbers to skip (default: 1 2 3 — warm-ups)")
    args = parser.parse_args()

    cfg = _load_config()
    model_cfg = cfg["model"]
    el_cfg = cfg.get("event_logic", {})
    min_conf = el_cfg.get("basket_min_confidence", 0.60)
    min_frames = el_cfg.get("basket_min_frames", 3)
    skip_clips = set(args.skip or [])

    # Locate highlights directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        output_dirs = sorted(Path("store/output").glob("*/highlights"))
        if not output_dirs:
            raise SystemExit("No highlights directories found under store/output/")
        run_dir = output_dirs[-1].parent

    highlights_dir = run_dir / "highlights"
    if not highlights_dir.exists():
        raise SystemExit(f"No highlights/ directory found in {run_dir}")

    clips = sorted(
        [p for p in highlights_dir.glob("*_highlight_*.mp4") if "reel" not in p.name],
        key=_clip_number,
    )
    clips = [c for c in clips if _clip_number(c) not in skip_clips]

    print(f"Run dir  : {run_dir}")
    print(f"Clips    : {len(clips)}  (skipping {sorted(skip_clips)})")
    print(f"Stride   : every {args.stride} frames")
    print(f"Threshold: conf ≥ {min_conf},  min_frames = {min_frames}")
    print()

    from ultralytics import YOLO  # deferred so argparse --help is fast
    import torch
    cfg_device = model_cfg.get("device", 0)
    device = cfg_device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and cfg_device != "cpu":
        print(f"CUDA not available — falling back to CPU (config requested {cfg_device})")
    imgsz = model_cfg.get("input_size", 640)
    model = YOLO(model_cfg["weights"])

    results: list[dict] = []
    t_start = time.time()

    for i, clip in enumerate(clips, 1):
        clip_num = _clip_number(clip)
        cap = cv2.VideoCapture(str(clip))
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        streak = 0
        fired = False
        fire_frame: int | None = None
        peak_conf = 0.0
        detection_frames = 0
        processed = 0
        src_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_idx % args.stride == 0:
                preds = model.predict(frame, device=device, imgsz=imgsz,
                                      conf=min_conf, verbose=False)
                frame_peak = 0.0
                has_basket = False
                for pred in preds:
                    for box in pred.boxes:
                        cls_name = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        if cls_name == "Ball_in_Basket" and conf >= min_conf:
                            has_basket = True
                            frame_peak = max(frame_peak, conf)
                            detection_frames += 1

                if has_basket:
                    streak += 1
                    peak_conf = max(peak_conf, frame_peak)
                    if streak >= min_frames and not fired:
                        fired = True
                        fire_frame = src_idx
                else:
                    streak = 0
                processed += 1
            src_idx += 1

        cap.release()

        elapsed = time.time() - t_start
        remaining_clips = len(clips) - i
        fps_proc = processed / elapsed if elapsed > 0 else 0
        # estimate remaining time
        avg_frames_per_clip = (processed * i) / i  # = processed for this clip
        eta_s = (remaining_clips * processed / max(fps_proc, 0.01)) if fps_proc > 0 else 0

        status = "YES" if fired else "no "
        print(
            f"[{i:3d}/{len(clips)}] Clip {clip_num:3d}: {status}  "
            f"peak={peak_conf:.2f}  det_frames={detection_frames:3d}  "
            f"processed={processed}  ETA ~{eta_s/60:.0f}m"
        )

        results.append({
            "clip": clip_num,
            "filename": clip.name,
            "detected": fired,
            "peak_confidence": round(peak_conf, 3),
            "detection_frames": detection_frames,
            "fire_frame": fire_frame,
            "src_frames": total_src_frames,
            "processed_frames": processed,
        })

    out_path = run_dir / "clip_detections.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    detected = sum(1 for r in results if r["detected"])
    print(f"\nDone in {total_time/60:.1f} min")
    print(f"Detected: {detected}/{len(results)} clips")
    print(f"Results : {out_path}")


if __name__ == "__main__":
    main()
