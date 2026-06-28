"""
calibrate_zone.py — one-time 3pt zone calibration tool.

Opens a frame from the configured video source, auto-detects the basket
using the existing YOLO model, then lets you click points along the 3pt
line.  The result is saved to store/calibration/three_point_zone.yaml and
used by ThreePointZone at runtime.

Run once per camera setup (re-run if the camera angle changes significantly):
    python src/calibrate_zone.py
    python src/calibrate_zone.py --source basket_2 --frame 500

Controls:
    Left click  — add a point along the 3pt line (left to right, please)
    U           — undo last point
    S           — save and exit
    Q           — quit without saving
    R           — reset all points
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WINDOW = "3pt Zone Calibration"
POINT_COLOUR = (0, 255, 255)
LINE_COLOUR = (0, 200, 255)
BASKET_COLOUR = (0, 128, 255)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_basket(frame: np.ndarray, cfg: dict) -> tuple[tuple[int, int], int] | None:
    """
    Run YOLO to find the Basket in *frame*.

    Returns (center_xy, width) of the highest-confidence Basket detection,
    or None if no basket found.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed — cannot auto-detect basket.")
        return None

    weights = cfg.get("model", {}).get("weights", "yolo11m.pt")
    if not Path(weights).exists():
        logger.warning("Weights not found at %s — skipping auto-detection.", weights)
        return None

    model = YOLO(weights)
    results = model(frame, verbose=False)[0]

    best_conf = 0.0
    best_box = None
    class_map: dict[int, str] = cfg.get("model", {}).get("class_map", {})
    # Handle both int and str keys from yaml
    class_map = {int(k): v for k, v in class_map.items()}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_map.get(cls_id, "")
        conf = float(box.conf[0])
        if cls_name == "Basket" and conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        logger.warning(
            "No Basket detected in this frame (conf=%.2f threshold). "
            "Try --frame with a frame where the basket is clearly visible.",
            best_conf,
        )
        return None

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w = x2 - x1
    logger.info("Basket detected: centre=(%d,%d) width=%d conf=%.2f", cx, cy, w, best_conf)
    return (cx, cy), w


def extract_frame(cfg: dict, source_name: str | None, frame_index: int) -> np.ndarray:
    sources = cfg.get("sources", [])
    src_cfg = next(
        (s for s in sources if source_name is None or s["name"] == source_name),
        sources[0] if sources else None,
    )
    if src_cfg is None:
        raise RuntimeError("No sources configured in config.yaml.")

    src_type = src_cfg.get("type", "")
    if src_type == "file":
        path = src_cfg["path"]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_index, total - 1))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        logger.info("Loaded frame %d from %s", frame_index, path)
        return frame
    else:
        raise RuntimeError(
            f"Source type '{src_type}' not supported for calibration. "
            "Use a file source or capture a screenshot and pass it via --image."
        )


def to_relative(
    pixel_points: list[tuple[int, int]],
    basket_center: tuple[int, int],
    basket_width: int,
) -> list[tuple[float, float]]:
    cx, cy = basket_center
    return [
        ((px - cx) / basket_width, (py - cy) / basket_width)
        for px, py in pixel_points
    ]


def rebuild_pixel_arc(
    relative_pts: list[tuple[float, float]],
    basket_center: tuple[int, int],
    basket_width: int,
) -> list[tuple[int, int]]:
    cx, cy = basket_center
    return [
        (int(cx + dx * basket_width), int(cy + dy * basket_width))
        for dx, dy in relative_pts
    ]


def draw_state(
    base_frame: np.ndarray,
    basket_center: tuple[int, int] | None,
    basket_width: int,
    clicked_pts: list[tuple[int, int]],
) -> np.ndarray:
    out = base_frame.copy()
    h, w = out.shape[:2]

    # Draw basket bbox
    if basket_center is not None:
        cx, cy = basket_center
        hw = basket_width // 2
        cv2.rectangle(out, (cx - hw, cy - hw), (cx + hw, cy + hw), BASKET_COLOUR, 2)
        cv2.circle(out, (cx, cy), 5, BASKET_COLOUR, -1)
        cv2.putText(out, "BASKET", (cx - hw, cy - hw - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BASKET_COLOUR, 1)

    # Draw arc points and connecting lines
    for i, pt in enumerate(clicked_pts):
        cv2.circle(out, pt, 6, POINT_COLOUR, -1)
        cv2.circle(out, pt, 7, (0, 0, 0), 1)
        cv2.putText(out, str(i + 1), (pt[0] + 8, pt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, POINT_COLOUR, 1)
        if i > 0:
            cv2.line(out, clicked_pts[i - 1], pt, LINE_COLOUR, 2)

    # Draw preview 2pt zone polygon (close through basket centre)
    if basket_center is not None and len(clicked_pts) >= 2:
        polygon = np.array(
            clicked_pts + [basket_center], dtype=np.int32
        )
        overlay = out.copy()
        cv2.fillPoly(overlay, [polygon], (0, 60, 0))
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        cv2.polylines(out, [np.array(clicked_pts, dtype=np.int32)], False, LINE_COLOUR, 2)

    # HUD
    n = len(clicked_pts)
    status = f"Points: {n}  |  Click the 3pt line left-to-right"
    if n >= 3:
        status += "  |  S=save  U=undo  R=reset  Q=quit"
    else:
        status += "  |  Need ≥3 points"
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


def run(
    cfg: dict,
    source_name: str | None,
    frame_index: int,
    image_path: str | None,
) -> None:
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.three_point_zone import ThreePointZone

    if image_path:
        base_frame = cv2.imread(image_path)
        if base_frame is None:
            logger.error("Could not load image: %s", image_path)
            sys.exit(1)
    else:
        base_frame = extract_frame(cfg, source_name, frame_index)

    basket_result = detect_basket(base_frame, cfg)
    if basket_result is None:
        logger.warning(
            "Basket not auto-detected. Click the basket centre first (it will be point 0), "
            "then the 3pt boundary. Or try a different frame with --frame N."
        )
        basket_center: tuple[int, int] | None = None
        basket_width = 0
    else:
        basket_center, basket_width = basket_result

    clicked_pts: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal basket_center, basket_width
        if event == cv2.EVENT_LBUTTONDOWN:
            if basket_center is None:
                # First click sets basket centre manually
                basket_center = (x, y)
                basket_width = 50  # placeholder; user can re-run with YOLO weights
                logger.info("Basket centre set manually to (%d, %d)", x, y)
            else:
                clicked_pts.append((x, y))
                logger.info(
                    "Point %d: pixel=(%d,%d)  relative=(%.2f, %.2f)",
                    len(clicked_pts), x, y,
                    (x - basket_center[0]) / max(basket_width, 1),
                    (y - basket_center[1]) / max(basket_width, 1),
                )

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, on_mouse)

    logger.info(
        "Click points along the 3-point line from LEFT to RIGHT.\n"
        "  U = undo  |  R = reset  |  S = save  |  Q = quit"
    )

    while True:
        display = draw_state(base_frame, basket_center, basket_width, clicked_pts)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and clicked_pts:
            removed = clicked_pts.pop()
            logger.info("Undo: removed point at %s", removed)

        elif key == ord("r"):
            clicked_pts.clear()
            logger.info("Reset: all points cleared")

        elif key == ord("s"):
            if basket_center is None:
                logger.warning("Click the basket centre first.")
                continue
            if len(clicked_pts) < 3:
                logger.warning("Need at least 3 points before saving.")
                continue
            relative = to_relative(clicked_pts, basket_center, basket_width)
            zone = ThreePointZone(boundary_relative=relative)
            saved = zone.save(cfg)
            logger.info("Saved! %d points → %s", len(relative), saved)
            break

        elif key in (ord("q"), 27):
            logger.info("Quit without saving.")
            break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the 3pt zone boundary")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, help="Source name from config (default: first)")
    parser.add_argument("--frame", type=int, default=300, help="Frame index to use")
    parser.add_argument("--image", default=None, help="Use a specific image file instead of video")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.source, args.frame, args.image)


if __name__ == "__main__":
    main()
