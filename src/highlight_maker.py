"""
highlight_maker.py — cut highlight clips from a clean game recording.

Reads the event log JSON produced by pipeline_test.py --save-log and
cuts a clip around each scored basket from the raw (unannotated) video
produced by --save-raw.  Clips are trimmed with ffmpeg (stream copy —
no re-encode) and saved as individual mp4 files.

Usage:
    python src/highlight_maker.py store/output/run/game_log.json store/output/run/game_raw.mp4

    # Custom clip window
    python src/highlight_maker.py game_log.json game_raw.mp4 --pre 10 --post 7

    # Choose output directory
    python src/highlight_maker.py game_log.json game_raw.mp4 --out-dir store/output/highlights/

    # Merge all clips into a single reel
    python src/highlight_maker.py game_log.json game_raw.mp4 --reel

Parameters (all tunable via CLI or config.yaml highlights section):
    --pre       Seconds of footage to include before the basket (default: 8)
    --post      Seconds of footage to include after the basket (default: 5)
    --min-gap   Minimum seconds between two events before they are merged
                into one clip (default: 3)
    --reel      Concatenate all clips into a single highlight reel mp4
    --out-dir   Output directory for clips (default: <raw_video_dir>/highlights/)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffmpeg helper
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str | None:
    """Return path to ffmpeg, checking PATH then platform-specific locations."""
    import glob as _glob
    found = shutil.which("ffmpeg")
    if found:
        return found
    if sys.platform == "win32":
        pattern = "C:/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/bin/ffmpeg.EXE"
        matches = _glob.glob(pattern, recursive=True)
        return matches[0] if matches else None
    for candidate in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    return None


def _ffmpeg_cut(
    ffmpeg: str,
    src: str,
    start_s: float,
    duration_s: float,
    dest: str,
) -> bool:
    """
    Cut a segment from src into dest using ffmpeg stream copy (no re-encode).
    Returns True on success.
    """
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_s:.3f}",
        "-i", src,
        "-t", f"{duration_s:.3f}",
        "-c", "copy",
        dest,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg failed cutting clip:\n%s", result.stderr[-400:])
        return False
    return True


def _ffmpeg_concat(ffmpeg: str, clip_paths: list[str], dest: str) -> bool:
    """Concatenate clips into a single reel using ffmpeg concat demuxer."""
    list_path = Path(dest).with_suffix(".concat_list.txt")
    list_path.write_text("\n".join(f"file '{Path(c).resolve()}'" for c in clip_paths))
    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        dest,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        logger.error("ffmpeg concat failed:\n%s", result.stderr[-400:])
        return False
    return True


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

def _load_score_events(log_path: str) -> tuple[list[dict], float]:
    """
    Parse the pipeline JSON log and return (score_events, source_fps).

    Each score event dict has:
        frame       int   — global frame number in the concatenated video
        timestamp_s float — video time in seconds (frame / fps)
        team        str   — "A" or "B"
        points      int
    """
    with open(log_path) as f:
        log = json.load(f)

    source_fps: float = log.get("source_fps") or 30.0
    game_state = log.get("game_state", {})
    raw_events: list[dict] = game_state.get("events", [])

    score_events = []
    for ev in raw_events:
        if ev.get("type") != "score":
            continue
        frame = ev.get("frame", 0)
        score_events.append({
            "frame": frame,
            "timestamp_s": round(frame / source_fps, 3),
            "team": ev.get("team", "?"),
            "points": ev.get("points", 2),
        })

    score_events.sort(key=lambda e: e["frame"])
    return score_events, source_fps


# ---------------------------------------------------------------------------
# Clip window computation
# ---------------------------------------------------------------------------

def _compute_clip_windows(
    events: list[dict],
    video_duration_s: float,
    pre_s: float,
    post_s: float,
    min_gap_s: float,
) -> list[dict]:
    """
    Convert score events into merged clip windows.

    Events closer than min_gap_s apart are merged into one clip to avoid
    nearly-duplicate highlights (e.g. two detections from the same basket).

    Returns a list of dicts:
        start_s   float  — clip start time (clamped to 0)
        end_s     float  — clip end time (clamped to video duration)
        duration_s float
        events    list   — score events covered by this clip
    """
    if not events:
        return []

    # Merge events that fall within min_gap_s of each other
    groups: list[list[dict]] = []
    current_group: list[dict] = [events[0]]

    for ev in events[1:]:
        prev_t = current_group[-1]["timestamp_s"]
        if ev["timestamp_s"] - prev_t <= min_gap_s:
            current_group.append(ev)
        else:
            groups.append(current_group)
            current_group = [ev]
    groups.append(current_group)

    windows = []
    for group in groups:
        basket_t = group[0]["timestamp_s"]   # use first event in group as anchor
        start_s = max(0.0, basket_t - pre_s)
        end_s = min(video_duration_s, basket_t + post_s)
        windows.append({
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "events": group,
        })

    return windows


# ---------------------------------------------------------------------------
# Video duration
# ---------------------------------------------------------------------------

def _video_duration(ffmpeg: str, video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe") or ffmpeg.replace("ffmpeg", "ffprobe").replace("ffmpeg.EXE", "ffprobe.EXE")
    cmd = [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    logger.warning("Could not determine video duration — using 1 hour as fallback.")
    return 3600.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_highlights(
    log_path: str,
    raw_video_path: str,
    out_dir: str | None = None,
    pre_s: float = 8.0,
    post_s: float = 5.0,
    min_gap_s: float = 3.0,
    make_reel: bool = False,
) -> list[str]:
    """
    Cut highlight clips and return list of saved clip paths.

    Args:
        log_path:       Path to the pipeline JSON event log.
        raw_video_path: Path to the clean (unannotated) game video.
        out_dir:        Output directory. Defaults to <raw_video_dir>/highlights/.
        pre_s:          Seconds before each basket to include in the clip.
        post_s:         Seconds after each basket to include in the clip.
        min_gap_s:      Events within this many seconds are merged into one clip.
        make_reel:      If True, also concatenate all clips into a single reel.

    Returns:
        List of paths to the saved clip files.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — install ffmpeg (see docs/tech-stack.md).")

    if not Path(log_path).exists():
        raise FileNotFoundError(f"Event log not found: {log_path}")
    if not Path(raw_video_path).exists():
        raise FileNotFoundError(f"Raw video not found: {raw_video_path}")

    events, source_fps = _load_score_events(log_path)
    if not events:
        logger.warning("No score events found in log — nothing to cut.")
        return []

    video_dur = _video_duration(ffmpeg, raw_video_path)
    windows = _compute_clip_windows(events, video_dur, pre_s, post_s, min_gap_s)

    out_path = Path(out_dir) if out_dir else Path(raw_video_path).parent / "highlights"
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(raw_video_path).stem
    saved_clips: list[str] = []

    logger.info(
        "Cutting %d highlight clip(s) from %s  (pre=%.0fs, post=%.0fs)",
        len(windows), raw_video_path, pre_s, post_s,
    )

    for i, win in enumerate(windows, start=1):
        ev = win["events"][0]
        clip_name = f"{stem}_highlight_{i:03d}_team{ev['team']}_{int(win['start_s'])}s.mp4"
        clip_path = str(out_path / clip_name)

        ok = _ffmpeg_cut(ffmpeg, raw_video_path, win["start_s"], win["duration_s"], clip_path)
        if ok:
            logger.info(
                "  [%d/%d] %s  (%.1fs–%.1fs, %.0fs clip, Team %s +%d)",
                i, len(windows), clip_name,
                win["start_s"], win["end_s"], win["duration_s"],
                ev["team"], ev["points"],
            )
            saved_clips.append(clip_path)
        else:
            logger.error("  [%d/%d] Failed to cut clip — skipping.", i, len(windows))

    if make_reel and len(saved_clips) > 1:
        reel_path = str(out_path / f"{stem}_highlight_reel.mp4")
        logger.info("Concatenating %d clips into reel: %s", len(saved_clips), reel_path)
        if _ffmpeg_concat(ffmpeg, saved_clips, reel_path):
            logger.info("Reel saved: %s", reel_path)
            saved_clips.append(reel_path)

    logger.info("Done — %d clip(s) saved to %s", len([c for c in saved_clips if "reel" not in c]), out_path)
    return saved_clips


def main() -> None:
    # Load default params from config.yaml if present
    config_defaults: dict = {}
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        config_defaults = cfg.get("highlights", {})

    parser = argparse.ArgumentParser(
        description="Cut highlight clips from a clean game recording using the event log."
    )
    parser.add_argument("log", metavar="LOG_JSON",
                        help="Path to the pipeline event log JSON (from --save-log)")
    parser.add_argument("video", metavar="RAW_VIDEO",
                        help="Path to the clean (unannotated) game video (from --save-raw)")
    parser.add_argument("--pre", type=float,
                        default=config_defaults.get("pre_seconds", 8.0),
                        metavar="SECS",
                        help="Seconds before each basket to include (default: %(default)s)")
    parser.add_argument("--post", type=float,
                        default=config_defaults.get("post_seconds", 5.0),
                        metavar="SECS",
                        help="Seconds after each basket to include (default: %(default)s)")
    parser.add_argument("--min-gap", type=float,
                        default=config_defaults.get("min_gap_seconds", 3.0),
                        metavar="SECS",
                        help="Events within this gap are merged into one clip (default: %(default)s)")
    parser.add_argument("--out-dir", metavar="DIR",
                        default=config_defaults.get("out_dir", None),
                        help="Output directory for clips (default: <video_dir>/highlights/)")
    parser.add_argument("--reel", action="store_true",
                        help="Also concatenate all clips into a single highlight reel mp4")

    args = parser.parse_args()

    try:
        clips = make_highlights(
            log_path=args.log,
            raw_video_path=args.video,
            out_dir=args.out_dir,
            pre_s=args.pre,
            post_s=args.post,
            min_gap_s=args.min_gap,
            make_reel=args.reel,
        )
        if not clips:
            sys.exit(1)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
