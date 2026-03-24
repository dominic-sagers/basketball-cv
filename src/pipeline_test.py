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
import glob as _glob
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import Detector
from src.face_blur import FaceBlur
from src.tracker import Tracker
from src.video_source import FileVideoSource, RTSPVideoSource, StreamChunkRecorder, USBCameraSource, VideoSource, VideoSourceFactory
from src.visualizer import Visualizer, BALL_CLASSES
from src.game_state import GameState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str | None:
    """Return path to ffmpeg, checking PATH then platform-specific install locations."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    if sys.platform == "win32":
        # Winget install location (Gyan.FFmpeg package)
        pattern = "C:/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/bin/ffmpeg.EXE"
        matches = _glob.glob(pattern, recursive=True)
        return matches[0] if matches else None
    # Linux / macOS common install locations
    for candidate in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    return None


def _concat_chunks(chunk_files: list[str], output_path: str) -> bool:
    """
    Concatenate annotated chunk files into a single video using ffmpeg concat demuxer.
    No re-encode — instant copy. Returns True on success.
    """
    if not chunk_files:
        logger.warning("No chunk files to concatenate.")
        return False

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.error("ffmpeg not found — cannot concatenate chunks. Install ffmpeg.")
        return False

    list_path = Path(output_path).with_suffix(".concat_list.txt")
    list_path.write_text("\n".join(f"file '{Path(c).resolve()}'" for c in chunk_files))

    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", output_path]
    logger.info("Concatenating %d chunks → %s ...", len(chunk_files), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)

    if result.returncode == 0:
        logger.info("Concatenation complete: %s", output_path)
        return True
    else:
        logger.error("ffmpeg concat failed:\n%s", result.stderr[-600:])
        return False


def _terminal_quit_watcher(stop_event: threading.Event) -> None:
    """Background thread: watch for 'q' keypress in the terminal (Windows msvcrt / POSIX select)."""
    if sys.platform == "win32":
        try:
            import msvcrt
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b"q", b"Q"):
                        logger.info("Terminal quit key pressed — stopping after current chunk.")
                        stop_event.set()
                        break
                time.sleep(0.05)
        except ImportError:
            pass
    else:
        # POSIX (Linux / macOS): use select + termios for non-blocking stdin
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while not stop_event.is_set():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        if sys.stdin.read(1).lower() == "q":
                            logger.info("Terminal quit key pressed — stopping after current chunk.")
                            stop_event.set()
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass  # stdin not a tty (e.g. piped) — rely on Ctrl+C and preview window Q


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    source: VideoSource,
    detector: Detector | None,
    tracker: Tracker | None,
    game_state: GameState | None = None,
    face_blur: FaceBlur | None = None,
    *,
    show_preview: bool = True,
    save_path: str | None = None,
    log_path: str | None = None,
    playback_speed: float = 1.0,
    detect_only: bool = False,
    inference_every: int = 1,
    frame_callback: "Callable[[np.ndarray, float], None] | None" = None,
    stop_event: "threading.Event | None" = None,
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
        log_path:       If set, write per-frame detection/tracking data to this JSON file
        playback_speed:  Slow down (0.5 = half speed) for frame-by-frame inspection
        detect_only:     Skip tracking — raw detections only (faster, less info)
        inference_every: Run inference only on every Nth frame; reuse last result for
                         intermediate frames. inference_every=2 halves inference load
                         while writing every decoded frame to output (smooth video).
                         Useful for live streams where the pipeline can't sustain source FPS.
        frame_callback:  If set, called with (annotated_frame, fps) for every frame.
                         Used by the Qt app to display frames without a cv2 window.
        stop_event:      If set, pipeline exits cleanly when the event is set.
                         Used by the Qt app to stop between frames.
    """
    if not source.open():
        logger.error("[%s] Failed to open source.", source.name)
        return {"ok": False}

    viz = Visualizer()
    writer: cv2.VideoWriter | None = None
    frame_log: list[dict] = []
    frame_number = 0
    stopped_by_user = False
    t_start = time.perf_counter()
    fps_times: list[float] = []

    # Cache last inference result so skipped frames still get annotations drawn.
    _last_tracks: list | None = None
    _last_detections: list | None = None

    # Delay between frames for file playback speed control
    target_frame_ms = int((1000 / source.fps) / playback_speed) if source.fps > 0 else 33

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("[%s] Stop event received.", source.name)
                break
            t_frame = time.perf_counter()
            ok, frame = source.read()
            if not ok:
                logger.info("[%s] Stream ended or source closed.", source.name)
                break

            frame_number += 1

            # ── Inference ──────────────────────────────────────────────
            # Run inference only on every Nth frame; reuse cached result otherwise.
            run_inference = (frame_number % inference_every == 0)

            if run_inference:
                if detect_only and detector:
                    _last_detections = detector.detect(frame)
                    _last_tracks = None
                elif tracker:
                    _last_tracks = tracker.track(frame)
                    _last_detections = None

            tracks = _last_tracks
            detections = _last_detections

            # ── Game state update ──────────────────────────────────────
            score_flash = False
            if game_state is not None and tracks and run_inference:
                events = game_state.process_frame(tracks, frame_number, source.name)
                if events:
                    score_flash = True

            # ── FPS measurement ────────────────────────────────────────
            elapsed = time.perf_counter() - t_frame
            if elapsed > 0:
                fps_times.append(1.0 / elapsed)
            recent_fps = sum(fps_times[-30:]) / len(fps_times[-30:]) if fps_times else 0.0

            # ── Frame log entry ────────────────────────────────────────
            if log_path:
                entry: dict = {
                    "frame": frame_number,
                    "timestamp_s": round(frame_number / source.fps, 3) if source.fps else None,
                    "inference_ms": round(elapsed * 1000, 1),
                    "ball_detected": False,
                    "objects": [],
                }
                if tracks:
                    for t in tracks:
                        entry["objects"].append({
                            "track_id": t.track_id,
                            "class": t.class_name,
                            "conf": round(t.confidence, 3),
                            "bbox": list(t.bbox),
                            "center": list(t.center),
                        })
                        if t.class_name in BALL_CLASSES:
                            entry["ball_detected"] = True
                elif detections:
                    for d in detections:
                        entry["objects"].append({
                            "class": d.class_name,
                            "conf": round(d.confidence, 3),
                            "bbox": list(d.bbox),
                            "center": list(d.center),
                        })
                        if d.class_name in BALL_CLASSES:
                            entry["ball_detected"] = True
                frame_log.append(entry)

            # ── Visualise ──────────────────────────────────────────────
            annotated = viz.draw(
                frame,
                tracks=tracks,
                detections=detections,
                source_name=source.name,
                fps=recent_fps,
                frame_number=frame_number,
                score=game_state.score if game_state else None,
                score_flash=score_flash,
            )

            # ── Face blur (applied after all overlays) ─────────────────
            if face_blur is not None:
                if tracks:
                    # Use player bounding boxes — reliable at any distance/angle
                    player_boxes = [
                        t.bbox for t in tracks if t.class_name in ("Player", "Player_Shooting", "person")
                    ]
                    annotated = face_blur.blur_player_heads(annotated, player_boxes)
                elif face_blur.enabled:
                    # Fallback to face detector (less reliable at gym distances)
                    annotated = face_blur.process(annotated)

            # ── Frame callback (Qt app / external display) ─────────────
            if frame_callback is not None:
                frame_callback(annotated, recent_fps)

            # ── Output ─────────────────────────────────────────────────
            if show_preview:
                cv2.imshow(f"basketball-cv | {source.name} — Q to quit", annotated)
                wait_ms = max(1, target_frame_ms - int((time.perf_counter() - t_frame) * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    logger.info("Stopped by user (Q).")
                    stopped_by_user = True
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
        if show_preview:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
        if log_path and frame_log:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            ball_frames = sum(1 for e in frame_log if e["ball_detected"])
            summary = {
                "source": source.name,
                "model": cfg_model if "cfg_model" in dir() else "unknown",
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "total_frames": frame_number,
                "source_fps": source.fps,
                "ball_detected_frames": ball_frames,
                "ball_detection_rate_pct": round(100 * ball_frames / frame_number, 1) if frame_number else 0,
                "avg_inference_ms": round(sum(e["inference_ms"] for e in frame_log) / len(frame_log), 1),
                "game_state": game_state.to_dict() if game_state else None,
                "frames": frame_log,
            }
            with open(log_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Frame log saved to %s  (ball detected in %d/%d frames = %.1f%%)",
                        log_path, ball_frames, frame_number,
                        100 * ball_frames / frame_number if frame_number else 0)

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
        "stopped_by_user": stopped_by_user,
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
    parser.add_argument("--save-log", metavar="PATH", default=None,
                        help="Save per-frame detection/tracking data to a JSON file")
    parser.add_argument("--playback-speed", type=float, default=1.0, metavar="X",
                        help="Playback speed multiplier for file sources (0.25 = quarter speed)")
    parser.add_argument("--face-blur", action="store_true",
                        help="Blur faces in output for privacy (applied after all overlays)")
    parser.add_argument("--inference-every", type=int, default=1, metavar="N",
                        help="Run inference every N frames; reuse last result for intermediate frames. "
                             "Use 2 or 3 for live streams to halve/third inference load while keeping "
                             "smooth full-FPS output. Default: 1 (every frame)")
    parser.add_argument("--stream-buffer", action="store_true",
                        help="Buffered stream mode: record every frame to chunk files on disk, then "
                             "run the pipeline through them sequentially. Eliminates dropped frames at "
                             "the cost of a processing delay (one chunk length). RTSP/HTTP sources only.")
    parser.add_argument("--chunk-seconds", type=float, default=30.0, metavar="SECS",
                        help="Chunk duration in seconds for --stream-buffer mode (default: 30)")
    parser.add_argument("--chunk-dir", metavar="DIR", default=None,
                        help="Directory to write stream chunks (default: store/output/stream-chunks/)")

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

    # ── Game state ────────────────────────────────────────────────────────
    game_state = GameState.from_config(cfg)

    # ── Face blur (optional) ──────────────────────────────────────────────
    face_blur = FaceBlur() if args.face_blur else None
    if face_blur:
        logger.info("Face blur enabled (backend: %s)", face_blur.backend)

    # ── Run pipeline on each source ───────────────────────────────────────
    show_preview = not args.no_preview

    # Buffered stream mode: one source must be RTSP/HTTP
    use_stream_buffer = args.stream_buffer and len(sources) == 1 and isinstance(sources[0], RTSPVideoSource)
    if args.stream_buffer and not use_stream_buffer:
        logger.warning("--stream-buffer requires exactly one RTSP/HTTP source — ignoring flag")

    if use_stream_buffer:
        chunk_dir = args.chunk_dir or "store/output/stream-chunks"
        recorder = StreamChunkRecorder(
            url=sources[0]._url,
            chunk_dir=chunk_dir,
            chunk_seconds=args.chunk_seconds,
        )
        chunk_queue = recorder.start()
        logger.info(
            "Buffered stream mode | %.0fs chunks → %s | lag ~%.0fs | press Q in preview or 'q' in terminal to stop + concat",
            args.chunk_seconds, chunk_dir, args.chunk_seconds,
        )

        save_path_base = Path(args.save_output) if args.save_output else None
        annotated_chunks: list[str] = []
        chunk_idx = 0

        # Terminal key watcher — lets user press 'q' without a preview window
        _quit_event = threading.Event()
        _watcher = threading.Thread(target=_terminal_quit_watcher, args=(_quit_event,), daemon=True)
        _watcher.start()

        def _process_chunk(chunk_path: str, idx: int) -> bool:
            """Run pipeline on one chunk. Returns True if user requested stop."""
            nonlocal annotated_chunks
            chunk_source = FileVideoSource(chunk_path, name=f"chunk-{idx:04d}")
            chunk_save = (
                str(save_path_base.with_stem(f"{save_path_base.stem}_chunk_{idx:04d}"))
                if save_path_base else None
            )
            logger.info("Processing chunk %04d: %s", idx, chunk_path)
            result = run_pipeline(
                source=chunk_source,
                detector=detector, tracker=tracker,
                game_state=game_state, face_blur=face_blur,
                show_preview=show_preview, save_path=chunk_save,
                detect_only=detect_only, inference_every=args.inference_every,
            )
            if chunk_save and result.get("ok"):
                annotated_chunks.append(chunk_save)
            if tracker is not None:
                tracker.reset()
            return result.get("stopped_by_user", False)

        try:
            while True:
                # Check terminal quit between chunks
                if _quit_event.is_set():
                    logger.info("Quit requested — stopping recorder, draining remaining chunks ...")
                    recorder.stop()
                    while not chunk_queue.empty():
                        remaining = chunk_queue.get_nowait()
                        if remaining:
                            _process_chunk(remaining, chunk_idx)
                            chunk_idx += 1
                    break

                try:
                    chunk_path = chunk_queue.get(timeout=1.0)
                except Exception:
                    continue  # timeout — check quit event and retry

                if chunk_path is None:
                    logger.info("Stream ended — draining final queued chunks ...")
                    while not chunk_queue.empty():
                        remaining = chunk_queue.get_nowait()
                        if remaining:
                            _process_chunk(remaining, chunk_idx)
                            chunk_idx += 1
                    break

                user_quit = _process_chunk(chunk_path, chunk_idx)
                chunk_idx += 1

                if user_quit:
                    logger.info("Quit requested — stopping recorder, draining remaining chunks ...")
                    recorder.stop()
                    _quit_event.set()
                    while not chunk_queue.empty():
                        remaining = chunk_queue.get_nowait()
                        if remaining:
                            _process_chunk(remaining, chunk_idx)
                            chunk_idx += 1
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted — stopping recorder.")
            recorder.stop()

        _quit_event.set()  # stop watcher thread

        # Auto-concatenate all annotated chunks into one file, then clean up
        if annotated_chunks and save_path_base:
            concat_out = str(save_path_base)
            if _concat_chunks(annotated_chunks, concat_out):
                for f in annotated_chunks:
                    Path(f).unlink(missing_ok=True)
                for f in Path(chunk_dir).glob("chunk_*.mp4"):
                    f.unlink(missing_ok=True)
                logger.info("Chunks deleted after successful concatenation.")
        elif not save_path_base and annotated_chunks:
            logger.info("No --save-output specified — annotated chunks not saved.")

    else:
        for source in sources:
            logger.info("Starting pipeline on source: %s", source.name)

            # When saving multiple sources, append source name to avoid overwriting
            save_path = args.save_output
            if save_path and len(sources) > 1:
                p = Path(save_path)
                save_path = str(p.with_stem(f"{p.stem}_{source.name}"))

            log_path = args.save_log
            if log_path and len(sources) > 1:
                p = Path(log_path)
                log_path = str(p.with_stem(f"{p.stem}_{source.name}"))

            run_pipeline(
                source=source,
                detector=detector,
                tracker=tracker,
                game_state=game_state,
                face_blur=face_blur,
                show_preview=show_preview,
                save_path=save_path,
                log_path=log_path,
                playback_speed=args.playback_speed,
                detect_only=detect_only,
                inference_every=args.inference_every,
            )

            # Reset tracker state between sources so IDs don't bleed across
            if tracker is not None:
                tracker.reset()


if __name__ == "__main__":
    main()
