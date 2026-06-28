"""
Tests for GameState — the team-level scoring/event logic (Phase 1).

This is the closest thing to the event-rule engine right now (event_logic.py
doesn't exist yet), so the make-detection, cooldown debounce, team-mapping,
and 2pt/3pt classification behaviour are pinned down here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.game_state import GameState
from src.three_point_zone import ThreePointZone


@dataclass
class FakeTrack:
    """Stands in for a Tracker Track."""
    class_name: str
    bbox: tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 10, 10))
    confidence: float = 1.0


BALL = [FakeTrack("Ball_in_Basket")]
NO_BALL = [FakeTrack("Player"), FakeTrack("Basketball")]


def test_made_basket_scores_two_for_team_a():
    gs = GameState(shot_cooldown_frames=0, basket_min_frames=1)
    fired = gs.process_frame(BALL, frame_number=0)
    assert len(fired) == 1
    assert fired[0].event_type == "score"
    assert fired[0].points == 2
    assert gs.score == {"A": 2, "B": 0}


def test_no_make_no_event():
    gs = GameState(shot_cooldown_frames=0, basket_min_frames=1)
    assert gs.process_frame(NO_BALL, frame_number=0) == []
    assert gs.score == {"A": 0, "B": 0}


def test_cooldown_debounces_duplicate_makes():
    # Ball_in_Basket persists across consecutive frames; one make must count once.
    gs = GameState(shot_cooldown_frames=3, basket_min_frames=1)
    fired = []
    for f in range(4):  # frame 0 scores, 1-2 suppressed, frame 3 scores again
        fired += gs.process_frame(BALL, frame_number=f)
    assert len(fired) == 2
    assert gs.score["A"] == 4
    assert gs.total_makes == 2


def test_cooldown_active_flag():
    gs = GameState(shot_cooldown_frames=5, basket_min_frames=1)
    assert gs.cooldown_active is False
    gs.process_frame(BALL, frame_number=0)
    assert gs.cooldown_active is True


def test_team_mapping_credits_correct_team():
    gs = GameState(shot_cooldown_frames=0, source_team_map={"basket_2": "B"}, basket_min_frames=1)
    fired = gs.process_frame(BALL, frame_number=0, source_name="basket_2")
    assert fired[0].team == "B"
    assert gs.score == {"A": 0, "B": 2}


def test_unknown_source_defaults_to_team_a():
    gs = GameState(shot_cooldown_frames=0, source_team_map={"basket_2": "B"}, basket_min_frames=1)
    fired = gs.process_frame(BALL, frame_number=0, source_name="mystery_cam")
    assert fired[0].team == "A"


def test_from_config_builds_map_and_cooldown():
    cfg = {
        "sources": [
            {"name": "basket_1", "team": "A"},
            {"name": "basket_2", "team": "B"},
        ],
        "event_logic": {"shot_cooldown_frames": 10},
    }
    gs = GameState.from_config(cfg)
    assert gs._source_team_map == {"basket_1": "A", "basket_2": "B"}
    assert gs._shot_cooldown_frames == 10


def test_from_config_defaults_cooldown_when_absent():
    gs = GameState.from_config({"sources": []})
    assert gs._shot_cooldown_frames == 45


def test_to_dict_shape():
    gs = GameState(shot_cooldown_frames=0, basket_min_frames=1)
    gs.process_frame(BALL, frame_number=5)
    d = gs.to_dict()
    assert d["score"]["A"] == 2
    assert d["total_makes"] == 1
    assert len(d["events"]) == 1
    event = d["events"][0]
    assert event["type"] == "score"
    assert event["team"] == "A"
    assert event["points"] == 2
    assert event["frame"] == 5


def test_reset_clears_state():
    gs = GameState(shot_cooldown_frames=0, basket_min_frames=1)
    gs.process_frame(BALL, frame_number=0)
    gs.reset()
    assert gs.score == {"A": 0, "B": 0}
    assert gs.events == []
    assert gs.last_event is None
    assert gs.total_makes == 0


# ---------------------------------------------------------------------------
# 2pt / 3pt classification via ThreePointZone
# ---------------------------------------------------------------------------

import math


def _simple_zone(radius: float = 2.0) -> ThreePointZone:
    """Circular arc at `radius` basket-widths — easy to reason about in tests."""
    n = 9
    angles = [math.radians(-90 + 180 * i / (n - 1)) for i in range(n)]
    return ThreePointZone([(radius * math.cos(a), radius * math.sin(a)) for a in angles])


def _basket_track(cx: int = 100, cy: int = 100, w: int = 20) -> FakeTrack:
    """A Basket track centred at (cx, cy) with width w."""
    half = w // 2
    return FakeTrack("Basket", bbox=(cx - half, cy - half, cx + half, cy + half))


def _shooter_track(foot_x: int, foot_y: int) -> FakeTrack:
    """A Player_Shooting track whose foot lands at (foot_x, foot_y)."""
    return FakeTrack("Player_Shooting", bbox=(foot_x - 10, foot_y - 40, foot_x + 10, foot_y))


def test_no_zone_always_scores_two():
    gs = GameState(shot_cooldown_frames=0, three_point_zone=None, basket_min_frames=1)
    fired = gs.process_frame(BALL, frame_number=0)
    assert fired[0].points == 2


def test_zone_present_no_shooter_history_scores_two():
    gs = GameState(shot_cooldown_frames=0, three_point_zone=_simple_zone(), basket_min_frames=1)
    tracks = [FakeTrack("Ball_in_Basket"), _basket_track()]
    fired = gs.process_frame(tracks, frame_number=0)
    assert fired[0].points == 2


def test_zone_present_no_basket_in_frame_scores_two():
    gs = GameState(shot_cooldown_frames=0, three_point_zone=_simple_zone(), basket_min_frames=1)
    # Frame 0: shooter recorded but no basket
    gs.process_frame([_shooter_track(100, 150)], frame_number=0)
    # Frame 1: ball in basket but still no basket bbox
    fired = gs.process_frame([FakeTrack("Ball_in_Basket")], frame_number=1)
    assert fired[0].points == 2


def test_shooter_inside_arc_scores_two():
    # Basket at (100, 100), width 20 → arc radius = 40px
    # Shooter foot at (100, 120) = 20px below basket → inside arc
    gs = GameState(shot_cooldown_frames=0, three_point_zone=_simple_zone(radius=2.0), basket_min_frames=1)
    basket = _basket_track(cx=100, cy=100, w=20)
    shooter = _shooter_track(foot_x=100, foot_y=120)   # 20px from basket < 40px arc

    gs.process_frame([shooter, basket], frame_number=0)
    fired = gs.process_frame([FakeTrack("Ball_in_Basket"), basket], frame_number=1)
    assert fired[0].points == 2
    assert gs.score["A"] == 2


def test_shooter_outside_arc_scores_three():
    # Shooter foot at (100, 165) = 65px below basket → outside 40px arc
    gs = GameState(shot_cooldown_frames=0, three_point_zone=_simple_zone(radius=2.0), basket_min_frames=1)
    basket = _basket_track(cx=100, cy=100, w=20)
    shooter = _shooter_track(foot_x=100, foot_y=165)   # 65px from basket > 40px arc

    gs.process_frame([shooter, basket], frame_number=0)
    fired = gs.process_frame([FakeTrack("Ball_in_Basket"), basket], frame_number=1)
    assert fired[0].points == 3
    assert gs.score["A"] == 3


def test_three_point_score_accumulates_correctly():
    gs = GameState(shot_cooldown_frames=0, three_point_zone=_simple_zone(radius=2.0), basket_min_frames=1)
    basket = _basket_track(cx=100, cy=100, w=20)
    shooter = _shooter_track(foot_x=100, foot_y=165)

    for frame in range(0, 6, 2):
        gs.process_frame([shooter, basket], frame_number=frame)
        gs.process_frame([FakeTrack("Ball_in_Basket"), basket], frame_number=frame + 1)

    assert gs.score["A"] == 9   # 3 × 3pt
    assert gs.total_makes == 3


def test_shooter_history_uses_most_recent_position():
    """Earlier inside-arc position is overwritten by later outside-arc position."""
    gs = GameState(shot_cooldown_frames=0, three_point_zone=_simple_zone(radius=2.0), basket_min_frames=1)
    basket = _basket_track(cx=100, cy=100, w=20)

    gs.process_frame([_shooter_track(100, 120), basket], frame_number=0)  # inside
    gs.process_frame([_shooter_track(100, 165), basket], frame_number=1)  # outside (most recent)
    fired = gs.process_frame([FakeTrack("Ball_in_Basket"), basket], frame_number=2)
    assert fired[0].points == 3


def test_three_point_zone_property_exposed():
    zone = _simple_zone()
    gs = GameState(three_point_zone=zone, basket_min_frames=1)
    assert gs.three_point_zone is zone


# ---------------------------------------------------------------------------
# basket_min_frames streak filter + basket_min_confidence
# ---------------------------------------------------------------------------

def test_basket_min_frames_filters_single_frame_noise():
    """A single Ball_in_Basket frame must not score when basket_min_frames=3."""
    gs = GameState(shot_cooldown_frames=0, basket_min_frames=3)
    fired = gs.process_frame(BALL, frame_number=0)
    assert fired == []
    assert gs.score == {"A": 0, "B": 0}


def test_basket_min_frames_scores_after_streak():
    """Score fires on the Nth consecutive Ball_in_Basket frame."""
    gs = GameState(shot_cooldown_frames=0, basket_min_frames=3)
    results = []
    for f in range(3):
        results += gs.process_frame(BALL, frame_number=f)
    assert len(results) == 1
    assert results[0].points == 2


def test_basket_min_frames_streak_resets_on_gap():
    """A gap in detections resets the streak — must see N consecutive again."""
    gs = GameState(shot_cooldown_frames=0, basket_min_frames=3)
    for f in range(2):
        gs.process_frame(BALL, frame_number=f)
    gs.process_frame(NO_BALL, frame_number=2)
    results = []
    for f in range(3, 6):
        results += gs.process_frame(BALL, frame_number=f)
    assert len(results) == 1


def test_basket_min_confidence_filters_low_conf():
    """Ball_in_Basket below basket_min_confidence must not score."""
    low_conf = [FakeTrack("Ball_in_Basket", confidence=0.4)]
    gs = GameState(shot_cooldown_frames=0, basket_min_confidence=0.60, basket_min_frames=1)
    fired = gs.process_frame(low_conf, frame_number=0)
    assert fired == []
    assert gs.score == {"A": 0, "B": 0}


def test_basket_min_confidence_allows_high_conf():
    """Ball_in_Basket at or above basket_min_confidence scores normally."""
    high_conf = [FakeTrack("Ball_in_Basket", confidence=0.75)]
    gs = GameState(shot_cooldown_frames=0, basket_min_confidence=0.60, basket_min_frames=1)
    fired = gs.process_frame(high_conf, frame_number=0)
    assert len(fired) == 1
    assert fired[0].points == 2
