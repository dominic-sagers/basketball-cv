"""
Export player crops and short clips per track for visual re-ID validation.

Produces:
  - contact_sheet.jpg  — grid of crops, one row per track, columns = time samples
  - clips/track_{id}.mp4 — 10s clip from the middle of each track, with bbox overlay
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CROP_H, CROP_W = 256, 128
CROPS_PER_TRACK = 8       # columns in contact sheet
CLIP_DURATION_S = 10      # seconds of clip per track
MIN_TRACK_FRAMES = 500
MAX_TRACKS = 20
FPS = 30


def load_track_index(log_path: Path) -> dict[int, list[tuple[int, list[int]]]]:
    with open(log_path) as f:
        data = json.load(f)
    index: dict[int, list] = defaultdict(list)
    for frame in data["frames"]:
        for obj in frame.get("objects", []):
            if obj["class"] == "Player":
                index[obj["track_id"]].append((frame["frame"], obj["bbox"]))
    return index


def extract_crop(img: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (CROP_W, CROP_H))


def build_contact_sheet(
    video_path: Path,
    track_index: dict[int, list],
    top_tracks: list[int],
    out_path: Path,
) -> None:
    # Evenly sample CROPS_PER_TRACK frames across each track's lifespan
    needed: dict[int, list[tuple[int, list[int]]]] = {}
    for tid in top_tracks:
        entries = track_index[tid]
        idxs = np.linspace(0, len(entries) - 1, CROPS_PER_TRACK, dtype=int)
        needed[tid] = [entries[i] for i in idxs]

    frame_lookup: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for tid, samples in needed.items():
        for frame_num, bbox in samples:
            frame_lookup[frame_num].append((tid, bbox))

    track_crops: dict[int, list[np.ndarray]] = {tid: [] for tid in top_tracks}

    cap = cv2.VideoCapture(str(video_path))
    for frame_num in sorted(frame_lookup.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ok, img = cap.read()
        if not ok:
            continue
        for tid, bbox in frame_lookup[frame_num]:
            track_crops[tid].append(extract_crop(img, bbox))
    cap.release()

    # Pad missing crops with black
    for tid in top_tracks:
        while len(track_crops[tid]) < CROPS_PER_TRACK:
            track_crops[tid].append(np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8))

    label_w = 100
    pad = 4
    row_h = CROP_H + pad * 2
    col_w = CROP_W + pad * 2
    sheet_h = row_h * len(top_tracks)
    sheet_w = label_w + col_w * CROPS_PER_TRACK
    sheet = np.full((sheet_h, sheet_w, 3), 30, dtype=np.uint8)

    for row, tid in enumerate(top_tracks):
        y0 = row * row_h
        # Label
        label = f"t{tid}\n({len(track_index[tid])}f)"
        for i, line in enumerate(label.split("\n")):
            cv2.putText(sheet, line, (4, y0 + 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for col, crop in enumerate(track_crops[tid]):
            x0 = label_w + col * col_w + pad
            sheet[y0 + pad: y0 + pad + CROP_H, x0: x0 + CROP_W] = crop

    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info(f"Contact sheet → {out_path}  ({len(top_tracks)} tracks × {CROPS_PER_TRACK} crops)")


def extract_clips(
    video_path: Path,
    track_index: dict[int, list],
    top_tracks: list[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    clip_frames = CLIP_DURATION_S * FPS

    for tid in top_tracks:
        entries = track_index[tid]
        mid_frame = entries[len(entries) // 2][0]
        start = max(0, mid_frame - clip_frames // 2)
        end = min(total_frames, start + clip_frames)

        # Build frame -> bbox lookup for this track
        bbox_lookup = {fn: bbox for fn, bbox in entries}

        out_path = out_dir / f"track_{tid:05d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for fn in range(int(start), int(end)):
            ok, frame = cap.read()
            if not ok:
                break
            if fn in bbox_lookup:
                x1, y1, x2, y2 = [int(v) for v in bbox_lookup[fn]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"track {tid}", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(frame)

        cap.release()
        writer.release()
        log.info(f"Clip → {out_path}  (frames {int(start)}–{int(end)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_json", type=Path)
    parser.add_argument("raw_video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("store/output/reid_validation"))
    parser.add_argument("--min-track-frames", type=int, default=MIN_TRACK_FRAMES)
    parser.add_argument("--max-tracks", type=int, default=MAX_TRACKS)
    parser.add_argument("--no-clips", action="store_true", help="Skip video clip export")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading track index...")
    track_index = load_track_index(args.log_json)

    top_tracks = sorted(
        [tid for tid, entries in track_index.items() if len(entries) >= args.min_track_frames],
        key=lambda tid: len(track_index[tid]),
        reverse=True,
    )[: args.max_tracks]
    log.info(f"{len(top_tracks)} tracks ≥ {args.min_track_frames} frames")

    log.info("Building contact sheet...")
    build_contact_sheet(
        args.raw_video, track_index, top_tracks,
        args.out_dir / "contact_sheet.jpg",
    )

    if not args.no_clips:
        log.info("Extracting clips...")
        extract_clips(args.raw_video, track_index, top_tracks, args.out_dir / "clips")

    log.info("Done.")


if __name__ == "__main__":
    main()
