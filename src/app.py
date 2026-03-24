"""
app.py — Basketball CV desktop application.

Unified UI: live annotated preview, log output, score management, and
pipeline start/stop control. Supports one or two simultaneous camera streams.

Camera architecture (two-camera setup):
    Camera A → watches Team A's basket → scores credited to Team A
    Camera B → watches Team B's basket → scores credited to Team B
    Each camera runs its own PipelineWorker (independent QThread).
    Both workers emit score_event signals; BasketballApp owns the
    authoritative score dict and updates the ScorePanel from both.

Usage:
    # Single camera
    python src/app.py --rtsp http://100.88.2.45:8080/video

    # Dual camera
    python src/app.py --rtsp http://100.88.2.45:8080/video \\
                      --rtsp2 http://100.88.2.46:8080/video

    # With chunked buffering (recommended for live streams)
    python src/app.py --rtsp <url> --chunk-seconds 5 --save-output store/output/game.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGroupBox, QSizePolicy,
)

from src.game_state import GameState
from src.tracker import Tracker
from src.video_source import FileVideoSource, StreamChunkRecorder
from src.pipeline_test import run_pipeline, _concat_chunks
from src.face_blur import FaceBlur

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log bridge — routes Python logging into the Qt UI
# ---------------------------------------------------------------------------

class _LogBridge(QObject):
    record = pyqtSignal(str, str)   # (levelname, formatted message)


class _QtLogHandler(logging.Handler):
    def __init__(self, bridge: _LogBridge) -> None:
        super().__init__()
        self._bridge = bridge
        self.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.record.emit(record.levelname, self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pipeline worker
# ---------------------------------------------------------------------------

class PipelineWorker(QThread):
    """
    Background thread: records an RTSP stream to chunks and runs the pipeline
    on each completed chunk via FileVideoSource.

    camera_team ("A" or "B") determines which team's score is credited when
    Ball_in_Basket is detected. Camera A watches Team A's basket; Camera B
    watches Team B's basket.
    """

    frame_ready    = pyqtSignal(object)     # numpy BGR frame
    score_event    = pyqtSignal(str, int)   # (team "A"|"B", points)
    fps_updated    = pyqtSignal(float)
    status_changed = pyqtSignal(str)
    finished       = pyqtSignal(str)        # final output path or ""

    def __init__(
        self,
        cfg: dict,
        rtsp_url: str,
        chunk_seconds: float,
        chunk_dir: str,
        save_output: str | None,
        camera_team: str = "A",
        face_blur_enabled: bool = False,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._rtsp_url = rtsp_url
        self._chunk_seconds = chunk_seconds
        self._chunk_dir = chunk_dir
        self._save_output = save_output
        self._camera_team = camera_team.upper()
        self._face_blur_enabled = face_blur_enabled
        self._stop_event = threading.Event()
        self._score_adjustments: list[tuple[str, int]] = []
        self._score_lock = threading.Lock()
        self.game_state: GameState | None = None
        self._prev_score: dict[str, int] = {"A": 0, "B": 0}

    def request_stop(self) -> None:
        self._stop_event.set()

    def adjust_score(self, team: str, delta: int) -> None:
        """Thread-safe: apply a manual score correction to the internal game_state."""
        with self._score_lock:
            self._score_adjustments.append((team.upper(), delta))

    def _frame_cb(self, frame: np.ndarray, fps: float) -> None:
        """Called by run_pipeline for every annotated frame."""
        # Apply queued manual adjustments
        with self._score_lock:
            for team, delta in self._score_adjustments:
                if self.game_state:
                    self.game_state.score[team] = max(0, self.game_state.score[team] + delta)
            self._score_adjustments.clear()

        # Detect auto score events by comparing with previous frame's score.
        # GameState always labels detected baskets as Team A internally;
        # we remap to camera_team so Camera B credits Team B correctly.
        if self.game_state:
            current = dict(self.game_state.score)
            total_now  = sum(current.values())
            total_prev = sum(self._prev_score.values())
            if total_now > total_prev:
                self.score_event.emit(self._camera_team, total_now - total_prev)
            self._prev_score = current

        self.frame_ready.emit(frame.copy())
        self.fps_updated.emit(fps)

    def _run_chunk(
        self,
        chunk_path: str,
        idx: int,
        tracker: Tracker,
        face_blur: FaceBlur | None,
        save_path_base: Path | None,
        annotated_chunks: list[str],
    ) -> bool:
        """Process one chunk file. Returns True if stop was requested."""
        chunk_save = (
            str(save_path_base.with_stem(f"{save_path_base.stem}_chunk_{idx:04d}"))
            if save_path_base else None
        )
        self.status_changed.emit(f"[Cam {self._camera_team}] Processing chunk {idx:04d} …")
        result = run_pipeline(
            source=FileVideoSource(chunk_path, name=f"cam{self._camera_team}-chunk-{idx:04d}"),
            detector=None,
            tracker=tracker,
            game_state=self.game_state,
            face_blur=face_blur,
            show_preview=False,
            save_path=chunk_save,
            frame_callback=self._frame_cb,
            stop_event=self._stop_event,
        )
        if chunk_save and result.get("ok"):
            annotated_chunks.append(chunk_save)
        tracker.reset()
        return result.get("stopped_by_user", False) or self._stop_event.is_set()

    def run(self) -> None:
        self.status_changed.emit(f"[Cam {self._camera_team}] Loading model …")
        model_cfg   = self._cfg["model"]
        tracking_cfg = self._cfg.get("tracking", {})

        tracker = Tracker.from_config(model_cfg, tracking_cfg)
        tracker.load()
        self.game_state = GameState.from_config(self._cfg)
        face_blur = FaceBlur() if self._face_blur_enabled else None

        # Use a camera-specific chunk subdirectory so two workers don't collide
        cam_chunk_dir = str(Path(self._chunk_dir) / f"cam{self._camera_team}")
        recorder = StreamChunkRecorder(
            url=self._rtsp_url,
            chunk_dir=cam_chunk_dir,
            chunk_seconds=self._chunk_seconds,
        )
        chunk_queue = recorder.start()
        self.status_changed.emit(
            f"[Cam {self._camera_team}] Recording + processing …  |  Press Stop to end"
        )

        save_path_base = Path(self._save_output) if self._save_output else None
        if save_path_base:
            # Suffix save path with camera team so two workers don't overwrite each other
            save_path_base = save_path_base.with_stem(f"{save_path_base.stem}_cam{self._camera_team}")

        annotated_chunks: list[str] = []
        chunk_idx = 0

        while not self._stop_event.is_set():
            try:
                chunk_path = chunk_queue.get(timeout=1.0)
            except Exception:
                continue
            if chunk_path is None:
                break
            user_quit = self._run_chunk(chunk_path, chunk_idx, tracker, face_blur, save_path_base, annotated_chunks)
            chunk_idx += 1
            if user_quit:
                recorder.stop()
                break

        # Drain remaining chunks
        recorder.stop()
        self.status_changed.emit(f"[Cam {self._camera_team}] Draining remaining chunks …")
        while not chunk_queue.empty():
            remaining = chunk_queue.get_nowait()
            if remaining:
                self._run_chunk(remaining, chunk_idx, tracker, face_blur, save_path_base, annotated_chunks)
                chunk_idx += 1

        # Concat + clean up
        final_path = ""
        if annotated_chunks and save_path_base:
            self.status_changed.emit(f"[Cam {self._camera_team}] Concatenating …")
            if _concat_chunks(annotated_chunks, str(save_path_base)):
                for f in annotated_chunks:
                    Path(f).unlink(missing_ok=True)
                for f in Path(cam_chunk_dir).glob("chunk_*.mp4"):
                    f.unlink(missing_ok=True)
                final_path = str(save_path_base)

        self.status_changed.emit("Idle")
        self.finished.emit(final_path)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _btn(label: str, color: str = "#3a3a3a") -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{color}; color:white; border:none;
            border-radius:6px; padding:8px 12px; font-size:13px; font-weight:bold;
        }}
        QPushButton:hover {{ background:#555; }}
        QPushButton:disabled {{ background:#2a2a2a; color:#555; }}
    """)
    return b

_GRP = "QGroupBox {{ color:white; font-size:13px; font-weight:bold; border:1px solid #444; border-radius:8px; margin-top:8px; padding-top:8px; }}"


# ---------------------------------------------------------------------------
# Score panel
# ---------------------------------------------------------------------------

class ScorePanel(QGroupBox):
    """Live score display with manual adjustment buttons for each team."""

    score_adjusted = pyqtSignal(str, int)   # (team "a"|"b", delta)

    def __init__(self) -> None:
        super().__init__("SCORE")
        self.setStyleSheet(_GRP)
        self._score_a = 0
        self._score_b = 0

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Score row: [Team A name + number] – [Team B name + number]
        score_row = QHBoxLayout()

        self._name_a  = QLabel("TEAM A")
        self._digit_a = QLabel("0")
        self._name_b  = QLabel("TEAM B")
        self._digit_b = QLabel("0")
        dash          = QLabel("–")

        for lbl in (self._name_a, self._name_b):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#888; font-size:11px; font-weight:bold;")
            lbl.setWordWrap(True)

        for lbl in (self._digit_a, self._digit_b):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:white; font-size:48px; font-weight:bold;")

        dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash.setStyleSheet("color:#666; font-size:36px;")

        col_a = QVBoxLayout()
        col_a.addWidget(self._name_a)
        col_a.addWidget(self._digit_a)
        col_b = QVBoxLayout()
        col_b.addWidget(self._name_b)
        col_b.addWidget(self._digit_b)

        score_row.addLayout(col_a)
        score_row.addWidget(dash)
        score_row.addLayout(col_b)
        root.addLayout(score_row)

        # Adjustment buttons
        root.addWidget(self._adj_row("a", "Team A"))
        root.addWidget(self._adj_row("b", "Team B"))

    def _adj_row(self, team: str, label: str) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#888; font-size:11px;")
        lbl.setFixedWidth(48)
        row.addWidget(lbl)
        for delta, text, color in [(+3,"+3","#1a7a2a"),(+2,"+2","#1a6a2a"),(+1,"+1","#155a20"),(-1,"-1","#7a1a1a")]:
            b = _btn(text, color)
            b.setFixedWidth(42)
            b.clicked.connect(lambda _, t=team, d=delta: self.score_adjusted.emit(t, d))
            row.addWidget(b)
        return w

    def set_score(self, a: int, b: int) -> None:
        self._score_a, self._score_b = a, b
        self._digit_a.setText(str(a))
        self._digit_b.setText(str(b))


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------

class ControlPanel(QGroupBox):
    """Pipeline start/stop and live status."""

    start_requested = pyqtSignal()
    stop_requested  = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("PIPELINE")
        self.setStyleSheet(_GRP)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self._status = QLabel("Idle")
        self._status.setStyleSheet("color:#aaa; font-size:11px;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)

        self._fps = QLabel("FPS: —")
        self._fps.setStyleSheet("color:#666; font-size:10px;")
        self._fps.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._btn_start = _btn("▶  Start Pipeline", "#1a5c2a")
        self._btn_stop  = _btn("■  Stop + Save",    "#7a1a1a")
        self._btn_stop.setEnabled(False)

        self._btn_start.clicked.connect(self.start_requested)
        self._btn_stop.clicked.connect(self.stop_requested)

        layout.addWidget(self._status)
        layout.addWidget(self._fps)
        layout.addSpacing(2)
        layout.addWidget(self._btn_start)
        layout.addWidget(self._btn_stop)

    def set_running(self, running: bool) -> None:
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_fps(self, fps: float) -> None:
        self._fps.setText(f"FPS: {fps:.1f}")


# ---------------------------------------------------------------------------
# Video panel
# ---------------------------------------------------------------------------

class VideoPanel(QLabel):
    """Displays annotated video frames with a team label overlay."""

    def __init__(self, team_label: str = "") -> None:
        super().__init__()
        self._team_label = team_label
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        label = f"Camera {self._team_label}\nNo stream" if self._team_label else "No stream"
        self.setText(label)
        self.setStyleSheet("background:#111; color:#444; font-size:14px; border:1px solid #333; border-radius:4px;")

    def update_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
        self.setStyleSheet("background:#000; border:1px solid #333; border-radius:4px;")


# ---------------------------------------------------------------------------
# Log panel
# ---------------------------------------------------------------------------

class LogPanel(QTextEdit):
    _COLOURS = {"DEBUG":"#777","INFO":"#ccc","WARNING":"#e8a000","ERROR":"#e04040","CRITICAL":"#ff4040"}

    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        # "Consolas" on Windows; Qt falls back to the system monospace font on Linux/macOS
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setStyleSheet("background:#111; color:#ccc; border:1px solid #333; border-radius:4px;")
        self.setMaximumHeight(170)

    def append_record(self, level: str, message: str) -> None:
        color = self._COLOURS.get(level, "#ccc")
        self.append(f'<span style="color:{color}">{message}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BasketballApp(QMainWindow):
    """
    Main application window. Supports one or two simultaneous camera streams.

    Score ownership: BasketballApp holds the authoritative score dict.
    Workers emit score_event(team, delta); the app accumulates and pushes
    updates to ScorePanel. Manual buttons go through the same path.
    """

    def __init__(
        self,
        cfg: dict,
        rtsp_url: str,
        rtsp_url2: str | None,
        chunk_seconds: float,
        chunk_dir: str,
        save_output: str | None,
        camera2_team: str = "B",
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._rtsp_url   = rtsp_url
        self._rtsp_url2  = rtsp_url2
        self._chunk_seconds = chunk_seconds
        self._chunk_dir  = chunk_dir
        self._save_output = save_output
        self._camera2_team = camera2_team.upper()

        # Authoritative score — both workers and manual buttons update this
        self._score: dict[str, int] = {"A": 0, "B": 0}

        self._worker:  PipelineWorker | None = None
        self._worker2: PipelineWorker | None = None
        self._workers_done = 0   # counts finished workers when stopping both

        self.setWindowTitle("basketball-cv")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background:#1e1e1e;")

        self._build_ui()
        self._setup_logging()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)

        # Video area: one or two panels side by side
        video_area = QHBoxLayout()
        video_area.setSpacing(6)
        self._video_a = VideoPanel("A (Team A)")
        self._video_b = VideoPanel(f"B (Team {self._camera2_team})")
        video_area.addWidget(self._video_a)
        video_area.addWidget(self._video_b)
        # Hide Camera B panel until a second URL is provided
        if not self._rtsp_url2:
            self._video_b.hide()

        video_widget = QWidget()
        video_widget.setLayout(video_area)
        top.addWidget(video_widget, stretch=3)

        # Right column
        right = QVBoxLayout()
        right.setSpacing(8)

        self._score_panel = ScorePanel()
        self._score_panel.score_adjusted.connect(self._on_score_adjusted)
        right.addWidget(self._score_panel)

        self._control_panel = ControlPanel()
        self._control_panel.start_requested.connect(self._start_pipeline)
        self._control_panel.stop_requested.connect(self._stop_pipeline)
        right.addWidget(self._control_panel)

        right.addStretch()
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(290)
        top.addWidget(right_widget)

        root.addLayout(top, stretch=1)

        log_label = QLabel("LOG")
        log_label.setStyleSheet("color:#555; font-size:10px; font-weight:bold;")
        root.addWidget(log_label)
        self._log = LogPanel()
        root.addWidget(self._log)

    def _setup_logging(self) -> None:
        self._log_bridge = _LogBridge()
        self._log_bridge.record.connect(self._log.append_record)
        logging.getLogger().addHandler(_QtLogHandler(self._log_bridge))

    # ── Score management ────────────────────────────────────────────────────

    def _apply_score(self, team: str, delta: int) -> None:
        """Single entry point for all score changes (auto + manual)."""
        key = team.upper()
        self._score[key] = max(0, self._score.get(key, 0) + delta)
        self._score_panel.set_score(self._score["A"], self._score["B"])

    def _on_score_event(self, team: str, delta: int) -> None:
        """Slot: auto score event from a worker."""
        self._apply_score(team, delta)

    def _on_score_adjusted(self, team: str, delta: int) -> None:
        """Slot: manual score button pressed."""
        self._apply_score(team, delta)
        # Keep worker game_states in sync for visualisation overlay
        for w in (self._worker, self._worker2):
            if w and w.isRunning():
                w.adjust_score(team, delta)

    # ── Pipeline control ────────────────────────────────────────────────────

    def _make_worker(self, url: str, team: str, save_suffix: str) -> PipelineWorker:
        save = None
        if self._save_output:
            p = Path(self._save_output)
            save = str(p.with_stem(f"{p.stem}{save_suffix}")) if save_suffix else self._save_output
        w = PipelineWorker(
            cfg=self._cfg,
            rtsp_url=url,
            chunk_seconds=self._chunk_seconds,
            chunk_dir=self._chunk_dir,
            save_output=save,
            camera_team=team,
        )
        w.score_event.connect(self._on_score_event)
        w.fps_updated.connect(self._control_panel.set_fps)
        w.status_changed.connect(self._control_panel.set_status)
        return w

    def _start_pipeline(self) -> None:
        if (self._worker and self._worker.isRunning()) or \
           (self._worker2 and self._worker2.isRunning()):
            return

        self._workers_done = 0

        # Camera A — always present
        self._worker = self._make_worker(self._rtsp_url, "A", "")
        self._worker.frame_ready.connect(self._video_a.update_frame)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        # Camera B — only if a second URL was provided
        if self._rtsp_url2:
            self._video_b.show()
            self._worker2 = self._make_worker(self._rtsp_url2, self._camera2_team, "_camB")
            self._worker2.frame_ready.connect(self._video_b.update_frame)
            self._worker2.finished.connect(self._on_worker_finished)
            self._worker2.start()

        self._control_panel.set_running(True)

    def _stop_pipeline(self) -> None:
        self._control_panel.set_status("Stopping …")
        self._control_panel.set_running(False)
        for w in (self._worker, self._worker2):
            if w and w.isRunning():
                w.request_stop()

    def _on_worker_finished(self, output_path: str) -> None:
        self._workers_done += 1
        expected = 2 if self._rtsp_url2 else 1
        if output_path:
            logger.info("Saved: %s", output_path)
        if self._workers_done >= expected:
            self._control_panel.set_running(False)
            self._control_panel.set_status("Idle — pipeline complete")

    def closeEvent(self, event) -> None:
        for w in (self._worker, self._worker2):
            if w and w.isRunning():
                w.request_stop()
                w.wait(5000)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Basketball CV desktop app")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rtsp",  metavar="URL", default=None,
                        help="Camera A RTSP/HTTP stream URL (Team A basket)")
    parser.add_argument("--rtsp2", metavar="URL", default=None,
                        help="Camera B RTSP/HTTP stream URL (Team B basket) — optional")
    parser.add_argument("--rtsp2-team", metavar="A|B", default="B",
                        help="Which team Camera B's basket belongs to (default: B)")
    parser.add_argument("--chunk-seconds", type=float, default=5.0, metavar="SECS")
    parser.add_argument("--chunk-dir", metavar="DIR", default="store/output/stream-chunks")
    parser.add_argument("--save-output", metavar="PATH", default="store/output/game.mp4")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve Camera A URL: CLI > first rtsp source in config
    rtsp_url = args.rtsp
    if not rtsp_url:
        for src in cfg.get("sources", []):
            if src.get("type") == "rtsp":
                rtsp_url = src["url"]
                break
    if not rtsp_url:
        print("ERROR: provide --rtsp URL or define an rtsp source in config.yaml")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = BasketballApp(
        cfg=cfg,
        rtsp_url=rtsp_url,
        rtsp_url2=args.rtsp2,
        chunk_seconds=args.chunk_seconds,
        chunk_dir=args.chunk_dir,
        save_output=args.save_output,
        camera2_team=args.rtsp2_team,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
