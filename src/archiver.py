"""
archiver.py — receive game footage and archive to DVC. No CV inference, no GUI.

Designed to run on any machine (no GPU, no display required). Receives chunks
from an Android phone or an RTSP stream, concatenates them into a single raw
game video, and pushes to the DVC remote so the clip viewer can serve them.

Flow:
    receive chunks → store/output/<run_id>/game_camA_raw.mp4
                   → dvc add store && dvc push
                   → log: git add store.dvc && git commit && git push

Usage:
    # Android phone uploading chunks (two-camera: run two instances)
    python src/archiver.py --source http_chunks --camera-team A
    python src/archiver.py --source http_chunks --camera-team B

    # RTSP stream
    python src/archiver.py --source rtsp --rtsp http://192.168.1.100:8080/video

    # Explicit run ID (to pair with a processing run on another machine)
    python src/archiver.py --source http_chunks --run-id 2026-06-25_200000

Press Ctrl+C to stop recording and trigger the post-session concat + DVC push.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game_archive import dvc_push

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffmpeg concat (no dependency on pipeline_test.py or any ML module)
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    return None


def _concat(chunk_files: list[str], output_path: str) -> bool:
    """Concatenate MP4 chunks with ffmpeg stream-copy (no re-encode). Returns True on success."""
    if not chunk_files:
        logger.warning("No chunks to concatenate.")
        return False
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.error("ffmpeg not found — cannot concatenate. Install ffmpeg.")
        return False

    list_path = Path(output_path).with_suffix(".concat_list.txt")
    list_path.write_text("\n".join(f"file '{Path(c).resolve()}'" for c in chunk_files))
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", output_path]
    logger.info("Concatenating %d chunks → %s", len(chunk_files), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)

    if result.returncode == 0:
        logger.info("Saved: %s", output_path)
        return True
    logger.error("ffmpeg concat failed:\n%s", result.stderr[-600:])
    return False


# ---------------------------------------------------------------------------
# Post-session archive
# ---------------------------------------------------------------------------

def _archive_session(
    raw_chunks: list[str],
    run_id: str,
    camera_team: str,
    cam_chunk_dir: str,
    source_kind: str,
) -> None:
    """Concat chunks → store/output/<run_id>/, delete source chunks, push to DVC."""
    if not raw_chunks:
        logger.warning("[Cam %s] No chunks recorded — nothing to archive.", camera_team)
        return

    out_dir = Path("store/output/games") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = str(out_dir / f"game_cam{camera_team.upper()}_raw.mp4")

    if not _concat(raw_chunks, raw_out):
        logger.error("[Cam %s] Concat failed — footage NOT archived.", camera_team)
        return

    # Delete source chunks now that concat succeeded
    for f in Path(cam_chunk_dir).glob("chunk_*.mp4"):
        f.unlink(missing_ok=True)
    if source_kind == "http_chunks":
        for chunk_path in raw_chunks:
            p = Path(chunk_path)
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)
    logger.info("[Cam %s] Source chunks deleted.", camera_team)

    # DVC push — run synchronously so the process doesn't exit before push completes
    dvc_push()
    logger.info(
        "Commit the updated DVC pointer:\n"
        "  git add store.dvc && git commit -m 'archive: game %s' && git push",
        run_id,
    )


# ---------------------------------------------------------------------------
# Main receive loop
# ---------------------------------------------------------------------------

def _run(
    source_kind: str,
    rtsp_url: str,
    chunk_seconds: float,
    chunk_dir: str,
    camera_team: str,
    run_id: str,
) -> None:
    stop_event = threading.Event()

    def _sigint(sig, frame):
        logger.info("Ctrl+C received — stopping after current chunk …")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    cam_chunk_dir = str(Path(chunk_dir) / f"cam{camera_team.upper()}")

    if source_kind == "http_chunks":
        from src.http_chunk_receiver import ChunkReceiverSource, ReceiverConfig
        recorder = ChunkReceiverSource(ReceiverConfig.from_yaml())
        logger.info("[Cam %s] Waiting for Android uploads … (Ctrl+C to stop and archive)", camera_team)
    else:
        from src.video_source import StreamChunkRecorder
        recorder = StreamChunkRecorder(
            url=rtsp_url,
            chunk_dir=cam_chunk_dir,
            chunk_seconds=chunk_seconds,
        )
        logger.info("[Cam %s] Recording RTSP stream … (Ctrl+C to stop and archive)", camera_team)

    chunk_queue = recorder.start()
    raw_chunks: list[str] = []
    chunk_idx = 0

    while not stop_event.is_set():
        try:
            chunk_path = chunk_queue.get(timeout=1.0)
        except Exception:
            continue
        if chunk_path is None:
            break
        raw_chunks.append(chunk_path)
        chunk_idx += 1
        logger.info("[Cam %s] Chunk %04d received: %s", camera_team, chunk_idx, Path(chunk_path).name)

    recorder.stop()

    # Drain any chunks that arrived while we were stopping
    while not chunk_queue.empty():
        remaining = chunk_queue.get_nowait()
        if remaining:
            raw_chunks.append(remaining)
            chunk_idx += 1

    logger.info("[Cam %s] Session ended — %d chunk(s) received.", camera_team, chunk_idx)
    _archive_session(raw_chunks, run_id, camera_team, cam_chunk_dir, source_kind)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Archive game footage to DVC — no CV inference, no GUI."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--source", default=None, choices=("rtsp", "http_chunks"),
        help="'rtsp' or 'http_chunks' (Android uploads). Default: auto-detect from config.",
    )
    parser.add_argument("--rtsp", metavar="URL", default=None, help="RTSP/HTTP stream URL")
    parser.add_argument("--camera-team", metavar="A|B", default="A",
                        help="Which team's basket this camera covers (default: A)")
    parser.add_argument("--chunk-seconds", type=float, default=5.0, metavar="SECS")
    parser.add_argument("--chunk-dir", metavar="DIR", default="store/output/dev/stream-chunks")
    parser.add_argument("--run-id", metavar="ID", default=None,
                        help="Session ID (default: current timestamp). Use the same ID across "
                             "both cameras so their footage is grouped in the same run directory.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Auto-detect source kind from config if not specified
    source_kind = args.source
    if source_kind is None:
        source_kind = "http_chunks" if any(
            s.get("type") == "http_chunks" for s in cfg.get("sources", [])
        ) else "rtsp"

    rtsp_url = args.rtsp or ""
    if source_kind == "rtsp" and not rtsp_url:
        for src in cfg.get("sources", []):
            if src.get("type") == "rtsp":
                rtsp_url = src["url"]
                break
        if not rtsp_url:
            print("ERROR: provide --rtsp URL or define an rtsp source in config.yaml")
            sys.exit(1)

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger.info("Run ID: %s  |  Camera: %s  |  Source: %s", run_id, args.camera_team, source_kind)

    _run(
        source_kind=source_kind,
        rtsp_url=rtsp_url,
        chunk_seconds=args.chunk_seconds,
        chunk_dir=args.chunk_dir,
        camera_team=args.camera_team,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
