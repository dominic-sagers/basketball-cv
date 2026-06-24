"""
three_point_zone.py — basket-anchored 3-point boundary for 2pt/3pt classification.

The 3-point zone is defined as a polyline in basket-relative coordinates:
    - Origin at the basket bounding box centre
    - Units: basket bounding box width (scale-invariant across camera positions)
    - +x rightward, +y downward in image space

The boundary is calibrated once by running calibrate_zone.py, which writes
store/calibration/three_point_zone.yaml.  At runtime the polyline is
re-anchored to wherever the Basket detection appears in each frame — so
small camera repositions between games are handled automatically.

Classification uses an angular distance test: a player is a 3-point shooter
if they are farther from the basket than the arc point in the same angular
direction.  This works for any camera angle and correctly handles corner 3s.

Later this can be replaced by a proper CourtHomography without changing
anything in game_state.py.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_CALIBRATION_DIR_DEFAULT = "store/calibration"
_CAL_FILENAME = "three_point_zone.yaml"


class ThreePointZone:
    """
    Classifies a floor position as inside (2pt) or outside (3pt) the arc.

    Boundary is stored as (dx, dy) offsets from the basket centre in
    basket-width units.  At runtime it is projected to pixel space using
    the detected basket bbox and compared against the shooter position
    using an angular distance test.
    """

    def __init__(self, boundary_relative: list[tuple[float, float]]) -> None:
        """
        Args:
            boundary_relative: Points along the 3-point line as (dx, dy)
                               offsets from basket centre in basket-width units.
        """
        if len(boundary_relative) < 3:
            raise ValueError(
                f"Need ≥3 boundary points, got {len(boundary_relative)}."
            )
        self._boundary = np.array(boundary_relative, dtype=np.float32)

    # ------------------------------------------------------------------
    # Runtime geometry
    # ------------------------------------------------------------------

    def pixel_boundary(
        self,
        basket_center: tuple[float, float],
        basket_width: float,
    ) -> np.ndarray:
        """
        Project the stored boundary into pixel coordinates.

        Returns an (N, 2) float32 array of pixel (x, y) points along the arc.
        """
        cx, cy = basket_center
        pts = self._boundary.copy() * basket_width
        pts[:, 0] += cx
        pts[:, 1] += cy
        return pts

    def is_three_point(
        self,
        player_foot_xy: tuple[float, float],
        basket_center: tuple[float, float],
        basket_width: float,
        frame_shape: tuple[int, int] = (10000, 10000),
    ) -> bool:
        """
        Return True if the player's floor position is outside the 3pt line.

        Uses an angular distance test: find the arc point in the same
        angular direction as the player (from the basket), then compare
        radial distances.  Works for any camera angle including corners.

        Args:
            player_foot_xy: Bottom-centre of the player bounding box (approx floor).
            basket_center:  Centre of the Basket detection bbox this frame.
            basket_width:   Width of the Basket detection bbox this frame.
            frame_shape:    Unused — kept for API compatibility with future homography.
        """
        arc_pts = self.pixel_boundary(basket_center, basket_width)
        bcx, bcy = basket_center
        px, py = player_foot_xy

        # Player's squared distance from basket
        player_dist_sq = (px - bcx) ** 2 + (py - bcy) ** 2

        # Find the arc point in the same angular direction as the player
        player_angle = math.atan2(py - bcy, px - bcx)
        arc_angles = np.arctan2(arc_pts[:, 1] - bcy, arc_pts[:, 0] - bcx)

        # Smallest absolute angular difference (handles ±π wrap)
        angle_diff = np.abs(((player_angle - arc_angles) + math.pi) % (2 * math.pi) - math.pi)
        nearest_idx = int(np.argmin(angle_diff))
        nearest_pt = arc_pts[nearest_idx]

        # Arc boundary squared distance in same direction
        arc_dist_sq = (nearest_pt[0] - bcx) ** 2 + (nearest_pt[1] - bcy) ** 2

        return player_dist_sq > arc_dist_sq

    def classify(
        self,
        player_foot_xy: tuple[float, float],
        basket_center: tuple[float, float],
        basket_width: float,
        frame_shape: tuple[int, int] = (10000, 10000),
    ) -> int:
        """Return 3 if behind the 3pt line, else 2."""
        return 3 if self.is_three_point(
            player_foot_xy, basket_center, basket_width, frame_shape
        ) else 2

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        basket_center: tuple[float, float],
        basket_width: float,
        colour: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Overlay the 3pt boundary arc on *frame* (in-place, returns frame)."""
        pts = self.pixel_boundary(basket_center, basket_width).astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), colour, thickness)
        return frame

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ThreePointZone | None":
        """
        Load calibration from store/calibration/three_point_zone.yaml.
        Returns None if the file doesn't exist yet.
        """
        cal_dir = Path(cfg.get("court", {}).get("calibration_dir", _CALIBRATION_DIR_DEFAULT))
        cal_file = cal_dir / _CAL_FILENAME

        if not cal_file.exists():
            logger.warning(
                "No 3pt zone calibration found at %s — "
                "run `python src/calibrate_zone.py` to set it up. "
                "All baskets will be counted as 2pt until then.",
                cal_file,
            )
            return None

        with cal_file.open() as f:
            data = yaml.safe_load(f)

        boundary = data.get("boundary", [])
        if len(boundary) < 3:
            logger.error(
                "3pt zone calibration at %s has only %d point(s) — need ≥3.",
                cal_file, len(boundary),
            )
            return None

        logger.info(
            "Loaded 3pt zone calibration: %d boundary points from %s",
            len(boundary), cal_file,
        )
        return cls(boundary_relative=boundary)

    def save(self, cfg: dict[str, Any]) -> Path:
        """Persist boundary to the calibration file. Returns the written path."""
        cal_dir = Path(cfg.get("court", {}).get("calibration_dir", _CALIBRATION_DIR_DEFAULT))
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_file = cal_dir / _CAL_FILENAME

        data = {"boundary": [list(map(float, pt)) for pt in self._boundary]}
        with cal_file.open("w") as f:
            yaml.safe_dump(data, f)

        logger.info("Saved 3pt zone calibration to %s", cal_file)
        return cal_file
