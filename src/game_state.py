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
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Class names that indicate a made basket
MADE_BASKET_CLASSES: frozenset[str] = frozenset({"Ball_in_Basket"})


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
    ) -> None:
        self.score: dict[str, int] = {"A": 0, "B": 0}
        self.events: list[GameEvent] = []
        self._shot_cooldown_frames = shot_cooldown_frames
        self._source_team_map = source_team_map or {}
        self._cooldown_remaining: int = 0
        self._session_start = time.time()
        self._last_event: GameEvent | None = None

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

        ball_in_basket = any(
            t.class_name in MADE_BASKET_CLASSES for t in tracks
        )

        if ball_in_basket and self._cooldown_remaining == 0:
            event = GameEvent(
                frame=frame_number,
                timestamp_s=timestamp_s,
                event_type="score",
                team=team,
                points=2,
                detail=f"Ball_in_Basket detected by {source_name or 'camera'}",
            )
            self.score[team] += 2
            self._cooldown_remaining = self._shot_cooldown_frames
            self._last_event = event
            self.events.append(event)
            fired.append(event)
            logger.info(
                "SCORE — Team %s +2  |  Score: A %d – B %d  (frame %d)",
                team, self.score["A"], self.score["B"], frame_number,
            )

        return fired

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

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
        source_team_map = {
            s["name"]: s.get("team", "A")
            for s in cfg.get("sources", [])
            if "name" in s
        }
        cooldown = cfg.get("event_logic", {}).get("shot_cooldown_frames", 45)
        return cls(shot_cooldown_frames=cooldown, source_team_map=source_team_map)
