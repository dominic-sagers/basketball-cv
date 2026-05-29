"""
Tests for GameState — the team-level scoring/event logic (Phase 1).

This is the closest thing to the event-rule engine right now (event_logic.py
doesn't exist yet), so the make-detection, cooldown debounce, and team-mapping
behaviour are pinned down here.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.game_state import GameState


@dataclass
class FakeTrack:
    """Stands in for a Tracker Track — process_frame only reads .class_name."""
    class_name: str


BALL = [FakeTrack("Ball_in_Basket")]
NO_BALL = [FakeTrack("Player"), FakeTrack("Basketball")]


def test_made_basket_scores_two_for_team_a():
    gs = GameState(shot_cooldown_frames=0)
    fired = gs.process_frame(BALL, frame_number=0)
    assert len(fired) == 1
    assert fired[0].event_type == "score"
    assert fired[0].points == 2
    assert gs.score == {"A": 2, "B": 0}


def test_no_make_no_event():
    gs = GameState(shot_cooldown_frames=0)
    assert gs.process_frame(NO_BALL, frame_number=0) == []
    assert gs.score == {"A": 0, "B": 0}


def test_cooldown_debounces_duplicate_makes():
    # Ball_in_Basket persists across consecutive frames; one make must count once.
    gs = GameState(shot_cooldown_frames=3)
    fired = []
    for f in range(4):  # frame 0 scores, 1-2 suppressed, frame 3 scores again
        fired += gs.process_frame(BALL, frame_number=f)
    assert len(fired) == 2
    assert gs.score["A"] == 4
    assert gs.total_makes == 2


def test_cooldown_active_flag():
    gs = GameState(shot_cooldown_frames=5)
    assert gs.cooldown_active is False
    gs.process_frame(BALL, frame_number=0)
    assert gs.cooldown_active is True


def test_team_mapping_credits_correct_team():
    gs = GameState(shot_cooldown_frames=0, source_team_map={"basket_2": "B"})
    fired = gs.process_frame(BALL, frame_number=0, source_name="basket_2")
    assert fired[0].team == "B"
    assert gs.score == {"A": 0, "B": 2}


def test_unknown_source_defaults_to_team_a():
    gs = GameState(shot_cooldown_frames=0, source_team_map={"basket_2": "B"})
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
    gs = GameState(shot_cooldown_frames=0)
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
    gs = GameState(shot_cooldown_frames=0)
    gs.process_frame(BALL, frame_number=0)
    gs.reset()
    assert gs.score == {"A": 0, "B": 0}
    assert gs.events == []
    assert gs.last_event is None
    assert gs.total_makes == 0
