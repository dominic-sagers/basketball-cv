"""
tests/test_three_point_zone.py — unit tests for ThreePointZone.

Covers:
    pixel_boundary()    — scaling and offset math
    is_three_point()    — inside/outside angular distance test
    classify()          — 2pt / 3pt return value
    draw()              — runs without error, returns frame
    from_config()       — missing file → None; valid file → correct zone
    save()              — writes correct YAML; round-trip via from_config
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.three_point_zone import ThreePointZone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _arc(radius: float = 2.0, n: int = 9) -> list[tuple[float, float]]:
    """
    Return `n` points forming a semicircle of `radius` basket-widths,
    spanning angles -90° → +90° (below and to the sides of the basket).
    """
    angles = [math.radians(-90 + 180 * i / (n - 1)) for i in range(n)]
    return [(radius * math.cos(a), radius * math.sin(a)) for a in angles]


@pytest.fixture
def zone() -> ThreePointZone:
    """A simple circular arc at 2.0 basket-widths radius."""
    return ThreePointZone(_arc(radius=2.0))


@pytest.fixture
def basket():
    """Basket at pixel (100, 100) with width 20. Arc radius = 40px."""
    return (100.0, 100.0), 20.0  # (center, width)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_fewer_than_3_points_raises():
    with pytest.raises(ValueError, match="≥3"):
        ThreePointZone([(0.0, 1.0), (1.0, 0.0)])


def test_exactly_3_points_accepted():
    zone = ThreePointZone([(0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)])
    assert zone is not None


# ---------------------------------------------------------------------------
# pixel_boundary
# ---------------------------------------------------------------------------

def test_pixel_boundary_shape(zone, basket):
    center, width = basket
    pts = zone.pixel_boundary(center, width)
    assert pts.shape == (9, 2)
    assert pts.dtype == np.float32


def test_pixel_boundary_offset_and_scale(basket):
    """A single point at (1.0, 0.0) in relative coords should map to (cx + width, cy)."""
    center, width = basket
    cx, cy = center
    zone = ThreePointZone([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
    pts = zone.pixel_boundary(center, width)
    assert pts[0, 0] == pytest.approx(cx + width)
    assert pts[0, 1] == pytest.approx(cy)


def test_pixel_boundary_negative_offset(basket):
    """A point at (-1.0, 0.0) maps to (cx - width, cy)."""
    center, width = basket
    cx, cy = center
    zone = ThreePointZone([(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    pts = zone.pixel_boundary(center, width)
    assert pts[0, 0] == pytest.approx(cx - width)


# ---------------------------------------------------------------------------
# is_three_point / classify
# ---------------------------------------------------------------------------

def test_player_inside_arc_is_two(zone, basket):
    center, width = basket
    cx, cy = center
    # 20px directly below basket — inside 40px arc radius
    assert zone.is_three_point((cx, cy + 20), center, width) is False


def test_player_outside_arc_is_three(zone, basket):
    center, width = basket
    cx, cy = center
    # 60px directly below basket — outside 40px arc radius
    assert zone.is_three_point((cx, cy + 60), center, width) is True


def test_player_at_basket_center_is_two(zone, basket):
    center, width = basket
    assert zone.is_three_point(center, center, width) is False


def test_classify_returns_2_inside(zone, basket):
    center, width = basket
    cx, cy = center
    assert zone.classify((cx, cy + 20), center, width) == 2


def test_classify_returns_3_outside(zone, basket):
    center, width = basket
    cx, cy = center
    assert zone.classify((cx, cy + 60), center, width) == 3


def test_corner_three_correctly_classified(basket):
    """
    Player at 45° (diagonal) outside the arc should still be a 3.
    Uses a square-ish arc to make the geometry clear.
    """
    center, width = basket
    cx, cy = center
    # Arc at 2.0 basket-widths; player at 3.0 basket-widths diagonally
    zone = ThreePointZone(_arc(radius=2.0, n=13))
    far_x = cx + 3.0 * width * math.cos(math.radians(45))
    far_y = cy + 3.0 * width * math.sin(math.radians(45))
    assert zone.is_three_point((far_x, far_y), center, width) is True


def test_inside_corner_is_two(basket):
    center, width = basket
    cx, cy = center
    zone = ThreePointZone(_arc(radius=2.0, n=13))
    near_x = cx + 0.5 * width * math.cos(math.radians(45))
    near_y = cy + 0.5 * width * math.sin(math.radians(45))
    assert zone.is_three_point((near_x, near_y), center, width) is False


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------

def test_draw_returns_frame(zone, basket):
    import numpy as np
    center, width = basket
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = zone.draw(frame, center, width)
    assert result is frame   # in-place, same object


def test_draw_does_not_raise_on_small_frame(zone, basket):
    """Arc may extend outside the frame — cv2 clips silently, no crash."""
    import numpy as np
    center, width = basket
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    zone.draw(frame, center, width)   # just must not raise


# ---------------------------------------------------------------------------
# from_config / save — persistence round-trip
# ---------------------------------------------------------------------------

def test_from_config_missing_file_returns_none(tmp_path):
    cfg = {"court": {"calibration_dir": str(tmp_path / "cal")}}
    assert ThreePointZone.from_config(cfg) is None


def test_from_config_loads_boundary(tmp_path, zone):
    cfg = {"court": {"calibration_dir": str(tmp_path / "cal")}}
    zone.save(cfg)
    loaded = ThreePointZone.from_config(cfg)
    assert loaded is not None
    assert len(loaded._boundary) == len(zone._boundary)


def test_save_writes_valid_yaml(tmp_path, zone):
    cfg = {"court": {"calibration_dir": str(tmp_path / "cal")}}
    path = zone.save(cfg)
    data = yaml.safe_load(path.read_text())
    assert "boundary" in data
    assert len(data["boundary"]) == len(zone._boundary)
    assert all(len(pt) == 2 for pt in data["boundary"])


def test_round_trip_preserves_geometry(tmp_path, basket):
    """Save then load; classification results must match."""
    center, width = basket
    cx, cy = center
    cfg = {"court": {"calibration_dir": str(tmp_path / "cal")}}

    original = ThreePointZone(_arc(radius=2.0))
    original.save(cfg)
    loaded = ThreePointZone.from_config(cfg)

    # Inside point
    assert loaded.is_three_point((cx, cy + 20), center, width) is False
    # Outside point
    assert loaded.is_three_point((cx, cy + 60), center, width) is True


def test_from_config_too_few_points_returns_none(tmp_path):
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    (cal_dir / "three_point_zone.yaml").write_text(
        yaml.dump({"boundary": [[0.0, 1.0], [1.0, 0.0]]})
    )
    cfg = {"court": {"calibration_dir": str(cal_dir)}}
    assert ThreePointZone.from_config(cfg) is None
