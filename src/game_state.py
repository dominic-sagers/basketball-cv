"""
game_state.py — single source of truth for the current game.

Tracks score, possession, and a timestamped event log. All CV pipeline
modules write here; the scoreboard UI reads from here.

Phase 1: team-level stats only. Team assignment comes from which camera
detected the event — camera config maps source name → team ("A" or "B").

Shot debouncing: Ball_in_Basket detected across ~30-60 consecutive frames
per make. A cooldown window prevents one real basket from being counted
multiple times.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.three_point_zone import ThreePointZone

logger = logging.getLogger(__name__)

# Class names that indicate a made basket
MADE_BASKET_CLASSES: frozenset[str] = frozenset({"Ball_in_Basket"})

# How many frames back to look for the last shooter when a basket is detected
_SHOOTER_LOOKBACK_FRAMES = 90  # ~3 seconds at 30 fps


@dataclass
class GameEvent:
    """A discrete timestamped game event."""
    frame: int
    timestamp_s: float
    event_type: str          # "score", "shot_attempt", "rebound", etc.
    team: str                # "A", "B", or "unknown"
    points: int = 0
    detail: str = ""


class GameState:
    """
    Tracks score and events for a single game session.

    Designed to be updated once per frame by the pipeline and read
    at any time by the scoreboard overlay.

    Args:
        shot_cooldown_frames:  Frames to suppress duplicate score events
                               after a basket is detected. At 30fps, 45
                               frames = 1.5 seconds — enough to cover the
                               Ball_in_Basket detection window for one make.
        source_team_map:       Maps source name → team label.
                               e.g. {"basket_1": "A", "basket_2": "B"}
                               If a source isn't in the map, defaults to "A".
    """

    def __init__(
        self,
        shot_cooldown_frames: int = 45,
        source_team_map: dict[str, str] | None = None,
        three_point_zone: "ThreePointZone | None" = None,
    ) -> None:
        self.score: dict[str, int] = {"A": 0, "B": 0}
        self.events: list[GameEvent] = []
        self._shot_cooldown_frames = shot_cooldown_frames
        self._source_team_map = source_team_map or {}
        self._three_point_zone = three_point_zone
        self._cooldown_remaining: int = 0
        self._session_start = time.time()
        self._last_event: GameEvent | None = None
        # Rolling buffer of (frame_number, foot_xy) for Player_Shooting detections
        self._shooter_history: deque[tuple[int, tuple[float, float]]] = deque(
            maxlen=_SHOOTER_LOOKBACK_FRAMES
        )

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def process_frame(
        self,
        tracks: list[Any],          # list[Track] — avoid circular import
        frame_number: int,
        source_name: str = "",
    ) -> list[GameEvent]:
        """
        Inspect tracks for this frame and fire any scoring events.

        Call once per frame, per camera source.

        Args:
            tracks:       Track list from Tracker.track()
            frame_number: Current frame index (for event log)
            source_name:  Source name from VideoSource.name (used to
                          determine which team's basket this camera covers)

        Returns:
            List of new GameEvent objects fired this frame (usually empty).
        """
        self._cooldown_remaining = max(0, self._cooldown_remaining - 1)

        team = self._source_team_map.get(source_name, "A")
        timestamp_s = round(time.time() - self._session_start, 2)
        fired: list[GameEvent] = []

        # Record any Player_Shooting detections for 2pt/3pt lookback
        for t in tracks:
            if t.class_name == "Player_Shooting":
                x1, y1, x2, y2 = t.bbox
                foot_xy = ((x1 + x2) / 2.0, float(y2))  # bottom-centre = approx floor
                self._shooter_history.append((frame_number, foot_xy))

        ball_in_basket = any(t.class_name in MADE_BASKET_CLASSES for t in tracks)

        if ball_in_basket and self._cooldown_remaining == 0:
            points = self._classify_shot(tracks, frame_number)
            detail = f"{points}pt — Ball_in_Basket detected by {source_name or 'camera'}"
            event = GameEvent(
                frame=frame_number,
                timestamp_s=timestamp_s,
                event_type="score",
                team=team,
                points=points,
                detail=detail,
            )
            self.score[team] += points
            self._cooldown_remaining = self._shot_cooldown_frames
            self._last_event = event
            self.events.append(event)
            fired.append(event)
            logger.info(
                "SCORE — Team %s +%d  |  Score: A %d – B %d  (frame %d)",
                team, points, self.score["A"], self.score["B"], frame_number,
            )

        return fired

    def _classify_shot(self, tracks: list[Any], frame_number: int) -> int:
        """
        Return 2 or 3 for the most recent shooter position vs the 3pt zone.

        Falls back to 2 if the zone is not calibrated or no shooter was seen
        recently.
        """
        if self._three_point_zone is None or not self._shooter_history:
            return 2

        # Find basket bbox for zone anchoring
        basket_center: tuple[float, float] | None = None
        basket_width: float = 0.0
        for t in tracks:
            if t.class_name == "Basket":
                x1, y1, x2, y2 = t.bbox
                basket_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                basket_width = float(x2 - x1)
                break

        if basket_center is None or basket_width == 0:
            return 2

        # Use the most recent shooter within the lookback window
        _, foot_xy = self._shooter_history[-1]

        return self._three_point_zone.classify(foot_xy, basket_center, basket_width)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def three_point_zone(self) -> "ThreePointZone | None":
        return self._three_point_zone

    @property
    def total_makes(self) -> int:
        return sum(1 for e in self.events if e.event_type == "score")

    @property
    def last_event(self) -> GameEvent | None:
        return self._last_event

    @property
    def cooldown_active(self) -> bool:
        """True when a recent score is still in its suppression window."""
        return self._cooldown_remaining > 0

    def score_display(self) -> str:
        """Short string for overlays: 'A  4 – 6  B'"""
        return f"A  {self.score['A']} – {self.score['B']}  B"

    def reset(self) -> None:
        """Full reset — call at game start or between halves."""
        self.score = {"A": 0, "B": 0}
        self.events.clear()
        self._cooldown_remaining = 0
        self._last_event = None
        self._shooter_history.clear()
        self._session_start = time.time()
        logger.info("GameState reset.")

    def to_dict(self) -> dict:
        """Serialise to dict for JSON logging."""
        return {
            "score": dict(self.score),
            "total_makes": self.total_makes,
            "events": [
                {
                    "frame": e.frame,
                    "timestamp_s": e.timestamp_s,
                    "type": e.event_type,
                    "team": e.team,
                    "points": e.points,
                }
                for e in self.events
            ],
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "GameState":
        """Build from the config.yaml sources section."""
        from src.three_point_zone import ThreePointZone

        source_team_map = {
            s["name"]: s.get("team", "A")
            for s in cfg.get("sources", [])
            if "name" in s
        }
        cooldown = cfg.get("event_logic", {}).get("shot_cooldown_frames", 45)
        zone = ThreePointZone.from_config(cfg)
        return cls(
            shot_cooldown_frames=cooldown,
            source_team_map=source_team_map,
            three_point_zone=zone,
        )
