"""
court_homography.py — pixel-to-court coordinate mapping for shot classification.

CourtHomography wraps an OpenCV homography matrix computed from manually-clicked
landmark correspondences (pixel ↔ real-world feet). Once calibrated it converts
any pixel coordinate to court coordinates and classifies a shooter's floor
position as a 2-point or 3-point attempt.

Coordinate system (all positions in feet, origin at the basket center projected
to the floor):
    +x — toward half court (away from this basket's baseline)
    +y — toward the left sideline when facing the court from behind the basket

With this system:
    - Baseline is at x = -basket_from_baseline_ft (e.g. -5.25 for NBA)
    - Free throw line is at x = (key_length_ft - basket_from_baseline_ft)
    - 3-point arc is a circle of radius three_point_radius_ft centred at (0, 0)
    - Corner 3-point line (NBA-style) is a straight segment at |y| = three_point_corner_ft

Calibration data is stored per-camera in store/calibration/<source_name>.yaml.
Run src/calibrate_court.py once per camera to generate those files.
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


class CourtHomography:
    """Maps camera pixel coordinates to court coordinates in feet."""

    def __init__(
        self,
        image_points: list[tuple[float, float]],
        court_points: list[tuple[float, float]],
        arc_radius_ft: float,
        basket_from_baseline_ft: float,
        corner_ft: float | None = None,
    ) -> None:
        """
        Args:
            image_points: pixel (x, y) coordinates of calibration landmarks.
            court_points: corresponding court (x, y) coordinates in feet
                          (basket-origin system described in module docstring).
            arc_radius_ft: 3-point arc radius from basket centre (e.g. 20.75 NCAA).
            basket_from_baseline_ft: distance from basket centre to baseline.
            corner_ft: lateral distance from basket centre line to corner 3pt
                       straight segment. None = pure arc only (NCAA-style).
        """
        if len(image_points) < 4:
            raise ValueError(
                f"Need ≥4 calibration points, got {len(image_points)}."
            )
        if len(image_points) != len(court_points):
            raise ValueError("image_points and court_points must be the same length.")

        self._arc_radius = arc_radius_ft
        self._basket_from_baseline = basket_from_baseline_ft
        self._corner_ft = corner_ft

        img_pts = np.array(image_points, dtype=np.float32)
        crt_pts = np.array(court_points, dtype=np.float32)

        H, mask = cv2.findHomography(img_pts, crt_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError(
                "Homography computation failed — check that calibration points "
                "are not collinear and cover different areas of the court."
            )

        inliers = int(mask.sum()) if mask is not None else len(image_points)
        logger.info(
            "CourtHomography: computed from %d/%d inlier pairs (RANSAC)",
            inliers,
            len(image_points),
        )
        self._H = H

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def pixel_to_court(self, px: float, py: float) -> tuple[float, float]:
        """Transform a single pixel coordinate to court feet (basket-origin)."""
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        x, y = result[0][0]
        return float(x), float(y)

    def pixels_to_court(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Batch transform a list of pixel coordinates to court feet."""
        if not points:
            return []
        arr = np.array([[p] for p in points], dtype=np.float32)
        result = cv2.perspectiveTransform(arr, self._H)
        return [(float(r[0][0]), float(r[0][1])) for r in result]

    # ------------------------------------------------------------------
    # Shot classification
    # ------------------------------------------------------------------

    def is_three_point_attempt(self, court_x: float, court_y: float) -> bool:
        """
        Return True if this court position is behind (outside) the 3-point line.

        Handles both the arc and optional straight corner segments.
        Positions behind the baseline (x < -basket_from_baseline) return False
        since they are out of bounds.
        """
        if court_x < -self._basket_from_baseline:
            return False  # behind the baseline — out of bounds

        distance = math.sqrt(court_x**2 + court_y**2)

        if distance >= self._arc_radius:
            return True

        # Corner segment: straight line at |y| = corner_ft, from baseline to
        # where the arc begins. Only applies if the config specifies corner_ft.
        if self._corner_ft is not None and abs(court_y) >= self._corner_ft:
            return True

        return False

    def classify_shot(
        self, court_x: float, court_y: float
    ) -> int:
        """Return 3 if behind the 3-point line, else 2."""
        return 3 if self.is_three_point_attempt(court_x, court_y) else 2

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], source_name: str
    ) -> "CourtHomography | None":
        """
        Load calibration for *source_name* from the calibration directory.

        Calibration files live at:
            <court.calibration_dir>/<source_name>.yaml

        Returns None if the calibration file does not exist yet (run
        calibrate_court.py to generate it).
        """
        court = cfg.get("court", {})
        cal_dir = Path(court.get("calibration_dir", _CALIBRATION_DIR_DEFAULT))
        cal_file = cal_dir / f"{source_name}.yaml"

        if not cal_file.exists():
            logger.warning(
                "[%s] No calibration file found at %s — "
                "run `python src/calibrate_court.py --source %s` to calibrate.",
                source_name,
                cal_file,
                source_name,
            )
            return None

        with cal_file.open() as f:
            cal = yaml.safe_load(f)

        points = cal.get("points", [])
        if len(points) < 4:
            logger.error(
                "[%s] Calibration file has %d point(s) — need ≥4.",
                source_name,
                len(points),
            )
            return None

        image_pts = [tuple(p["image"]) for p in points]
        court_pts = [tuple(p["court"]) for p in points]

        corner_ft = court.get("three_point_corner_ft")  # None if not set

        try:
            return cls(
                image_points=image_pts,
                court_points=court_pts,
                arc_radius_ft=court.get("three_point_radius_ft", 20.75),
                basket_from_baseline_ft=court.get("basket_from_baseline_ft", 5.25),
                corner_ft=corner_ft,
            )
        except ValueError as exc:
            logger.error("[%s] Failed to build CourtHomography: %s", source_name, exc)
            return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def reprojection_errors(
        self,
        image_points: list[tuple[float, float]],
        court_points: list[tuple[float, float]],
    ) -> list[float]:
        """Return per-point distance (feet) between projected and expected court coords."""
        errors = []
        for (px, py), (cx, cy) in zip(image_points, court_points):
            tx, ty = self.pixel_to_court(px, py)
            errors.append(math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2))
        return errors
