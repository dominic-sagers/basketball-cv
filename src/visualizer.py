"""
visualizer.py — real-time overlay rendering for the basketball CV pipeline.

Draws detection and tracking results onto frames for debugging and monitoring.
Designed to be used both during development (pipeline_test.py) and optionally
in production to show a debug feed alongside the scoreboard.

Features:
    - Bounding boxes color-coded by class (player/ball/hoop)
    - Track IDs on each tracked object
    - Ball trajectory trail (last N positions as a fading polyline)
    - Team color tinting (Phase 1d — once team assignment is added)
    - FPS counter and source info panel
    - Confidence scores (toggleable)

Usage:
    viz = Visualizer()
    annotated = viz.draw(frame, tracks=tracks, source_name="basket_1", fps=45.2)
    cv2.imshow("Debug", annotated)
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.detector import Detection
from src.tracker import Track


# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------

class Colors:
    PLAYER       = (220, 160,  50)   # steel blue
    BALL         = ( 30, 140, 255)   # orange (basketball color)
    HOOP         = ( 50, 200, 255)   # gold
    TEAM_A       = ( 80, 200,  80)   # green
    TEAM_B       = (200,  80,  80)   # red-ish
    TRACK_LABEL  = (255, 255, 255)   # white
    TRAIL_BASE   = ( 30, 140, 255)   # orange — matches ball
    INFO_BG      = ( 20,  20,  20)   # near-black panel
    INFO_TEXT    = (220, 220, 220)   # light grey
    WARN_TEXT    = ( 50, 180, 255)   # amber
    FPS_OK       = ( 60, 200,  60)   # green
    FPS_WARN     = ( 50, 180, 255)   # amber
    FPS_FAIL     = ( 60,  60, 220)   # red


CLASS_COLOR: dict[str, tuple[int, int, int]] = {
    "person":      Colors.PLAYER,
    "sports ball": Colors.BALL,
    "hoop":        Colors.HOOP,
}

TEAM_COLOR: dict[str, tuple[int, int, int]] = {
    "A": Colors.TEAM_A,
    "B": Colors.TEAM_B,
}


# ---------------------------------------------------------------------------
# Ball trail state
# ---------------------------------------------------------------------------

@dataclass
class BallTrail:
    """Stores recent ball center positions for trajectory visualisation."""

    maxlen: int = 40
    positions: collections.deque = field(default_factory=lambda: collections.deque(maxlen=40))

    def update(self, center: tuple[int, int] | None) -> None:
        """Append a new center, or None if the ball was not detected this frame."""
        self.positions.append(center)

    def clear(self) -> None:
        self.positions.clear()


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    """
    Draws detection and tracking overlays onto frames.

    All drawing is done on a copy of the input frame — the original is
    never modified.
    """

    def __init__(
        self,
        show_confidence: bool = True,
        show_track_ids: bool = True,
        trail_length: int = 40,
        box_thickness: int = 2,
        font_scale: float = 0.55,
    ) -> None:
        self._show_confidence = show_confidence
        self._show_track_ids = show_track_ids
        self._box_thickness = box_thickness
        self._font_scale = font_scale
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._ball_trail = BallTrail(maxlen=trail_length)

        # FPS smoothing
        self._fps_history: collections.deque[float] = collections.deque(maxlen=30)
        self._last_ts: float = time.perf_counter()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        *,
        tracks: list[Track] | None = None,
        detections: list[Detection] | None = None,
        source_name: str = "",
        fps: float | None = None,
        frame_number: int | None = None,
        team_assignments: dict[int, str] | None = None,   # track_id → "A" | "B"
        extra_labels: list[str] | None = None,            # freeform text in info panel
    ) -> np.ndarray:
        """
        Draw all overlays onto a copy of `frame` and return the result.

        Pass either `tracks` (after tracker runs) or `detections` (raw detections
        only). Passing both is fine — tracks take visual priority.

        Args:
            frame:            BGR frame from any VideoSource
            tracks:           List of Track objects from Tracker.track()
            detections:       List of Detection objects from Detector.detect()
            source_name:      Label shown in the info panel (e.g. "basket_1")
            fps:              Measured pipeline FPS; auto-computed if None
            frame_number:     Optional frame index for file sources
            team_assignments: Optional dict mapping track_id → team label
            extra_labels:     Additional text lines in the info panel

        Returns:
            Annotated BGR frame (same resolution as input).
        """
        canvas = frame.copy()

        # Measure FPS internally if not provided
        now = time.perf_counter()
        dt = now - self._last_ts
        self._last_ts = now
        if dt > 0:
            self._fps_history.append(1.0 / dt)
        measured_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0.0
        display_fps = fps if fps is not None else measured_fps

        # Ball trail — find ball position this frame
        ball_center: tuple[int, int] | None = None
        if tracks:
            for t in tracks:
                if t.class_name == "sports ball":
                    ball_center = t.center
                    break
        elif detections:
            for d in detections:
                if d.class_name == "sports ball":
                    ball_center = d.center
                    break
        self._ball_trail.update(ball_center)

        # Draw layers (order matters — trail under boxes)
        self._draw_trail(canvas)
        if tracks:
            self._draw_tracks(canvas, tracks, team_assignments)
        elif detections:
            self._draw_detections(canvas, detections)
        self._draw_info_panel(canvas, source_name, display_fps, frame_number, tracks, detections, extra_labels)

        return canvas

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_tracks(
        self,
        canvas: np.ndarray,
        tracks: list[Track],
        team_assignments: dict[int, str] | None,
    ) -> None:
        for track in tracks:
            color = self._track_color(track, team_assignments)
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self._box_thickness)

            label_parts = []
            if self._show_track_ids:
                label_parts.append(f"#{track.track_id}")
            label_parts.append(track.class_name)
            if self._show_confidence:
                label_parts.append(f"{track.confidence:.2f}")
            label = " ".join(label_parts)

            # Team badge
            if team_assignments and track.track_id in team_assignments:
                label = f"[{team_assignments[track.track_id]}] {label}"

            self._draw_label(canvas, label, x1, y1, color)

    def _draw_detections(self, canvas: np.ndarray, detections: list[Detection]) -> None:
        for det in detections:
            color = CLASS_COLOR.get(det.class_name, Colors.PLAYER)
            x1, y1, x2, y2 = det.bbox
            # Dashed-style boxes for untracked detections (thinner, alpha-blended)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

            label_parts = [det.class_name]
            if self._show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            self._draw_label(canvas, " ".join(label_parts), x1, y1, color)

    def _draw_trail(self, canvas: np.ndarray) -> None:
        """Draw a fading polyline for the ball trajectory."""
        pts = [p for p in self._ball_trail.positions if p is not None]
        if len(pts) < 2:
            return

        n = len(pts)
        for i in range(1, n):
            # Older points are more transparent (smaller alpha)
            alpha = i / n
            thickness = max(1, int(4 * alpha))
            brightness = int(80 + 175 * alpha)
            # Fade from dim to full orange
            color = (
                int(Colors.TRAIL_BASE[0] * alpha),
                int(Colors.TRAIL_BASE[1] * alpha),
                brightness,
            )
            cv2.line(canvas, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Dot at current position
        if pts:
            cv2.circle(canvas, pts[-1], 6, Colors.BALL, -1, cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], 6, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_label(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a filled background label above a bounding box."""
        (tw, th), baseline = cv2.getTextSize(text, self._font, self._font_scale, 1)
        pad = 3
        bg_y1 = max(0, y - th - 2 * pad)
        bg_y2 = y
        bg_x2 = min(canvas.shape[1], x + tw + 2 * pad)

        # Semi-transparent label background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        cv2.putText(
            canvas, text,
            (x + pad, y - pad),
            self._font, self._font_scale,
            Colors.TRACK_LABEL, 1, cv2.LINE_AA,
        )

    def _draw_info_panel(
        self,
        canvas: np.ndarray,
        source_name: str,
        fps: float,
        frame_number: int | None,
        tracks: list[Track] | None,
        detections: list[Detection] | None,
        extra_labels: list[str] | None,
    ) -> None:
        """Draw a semi-transparent info panel in the top-left corner."""
        n_players = sum(1 for t in (tracks or []) if t.class_name == "person")
        n_balls   = sum(1 for t in (tracks or []) if t.class_name == "sports ball")
        n_raw_det = len(detections) if detections else 0
        mode = "TRACK" if tracks else "DETECT"

        fps_color = Colors.FPS_OK if fps >= 30 else (Colors.FPS_WARN if fps >= 15 else Colors.FPS_FAIL)

        lines = [
            f"FPS  {fps:.1f}",
            f"SRC  {source_name}" if source_name else None,
            f"MODE {mode}",
            f"PLYR {n_players}  BALL {n_balls}",
            f"DET  {n_raw_det}" if detections else None,
            f"FRM  {frame_number}" if frame_number is not None else None,
        ]
        if extra_labels:
            lines.extend(extra_labels)

        lines = [l for l in lines if l is not None]

        line_h = 22
        pad = 8
        panel_h = len(lines) * line_h + pad * 2
        panel_w = 220

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), Colors.INFO_BG, -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        for i, line in enumerate(lines):
            color = fps_color if line.startswith("FPS") else Colors.INFO_TEXT
            cv2.putText(
                canvas, line,
                (pad, pad + (i + 1) * line_h - 4),
                self._font, 0.5,
                color, 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _track_color(
        self,
        track: Track,
        team_assignments: dict[int, str] | None,
    ) -> tuple[int, int, int]:
        """Color by team if assigned, otherwise by class."""
        if team_assignments and track.track_id in team_assignments:
            return TEAM_COLOR.get(team_assignments[track.track_id], Colors.PLAYER)
        return CLASS_COLOR.get(track.class_name, Colors.PLAYER)

    def reset_trail(self) -> None:
        """Clear the ball trail — call between game periods or on source switch."""
        self._ball_trail.clear()
