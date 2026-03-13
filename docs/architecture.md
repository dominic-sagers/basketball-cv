# System Architecture

## Pipeline overview

```
[Camera 1] [Camera 2] [Camera N]
      |          |         |
      └──────────┴─────────┘
                 |
      [Frame preprocessor]        ← OpenCV capture, resize, denoise
                 |
      [YOLOv11 detector]          ← GPU inference: players, ball, hoop
                 |
       ┌─────────┼──────────┐
       |         |          |
  [ByteTrack] [Ball      [Pose
  [tracker  ] [tracker]  [estimator]
       |         |          |
       └─────────┴──────────┘
                 |
      [Event logic engine]        ← Rules: made shot, rebound, block, assist, TO
                 |
      [Game state + stat agg]     ← Running totals, possession, score
                 |
       ┌─────────┴──────────┐
       |                    |
  [Scoreboard UI]      [Data logger]
  (Phase 1.5)          (JSON → Excel Phase 2)
```

---

## Module breakdown

### `src/camera.py`
- Opens and manages N camera streams (OpenCV VideoCapture)
- Handles frame sync across multiple cameras
- Config: camera indices or RTSP URLs, target resolution, FPS

### `src/preprocessor.py`
- Resize frames to model input size
- Optional: denoise, contrast adjustment for poor gym lighting
- Court ROI crop (remove scoreboard / bleachers from inference region)

### `src/detector.py`
- Wraps Ultralytics YOLOv11
- Runs inference on each frame
- Returns: bounding boxes, class labels, confidence scores
- Classes to detect: `person`, `sports ball`, `hoop` (custom class)
- CUDA required — raises on CPU-only run

### `src/tracker.py`
- Wraps ByteTrack (via Ultralytics built-in)
- Maintains persistent player track IDs within a session
- Outputs: list of tracked players with IDs + bounding boxes each frame
- Handles occlusion recovery (ByteTrack's low-confidence second-pass association)

### `src/ball_tracker.py`
- Tracks ball position across frames
- Computes trajectory vector (direction, speed)
- Detects hoop zone entry with downward motion filter (anti-false-positive)
- Outputs: ball position, trajectory, `shot_made` / `shot_attempted` events

### `src/pose_estimator.py`
- Wraps YOLOv11-pose (or AlphaPose as alternative)
- Returns 17-keypoint skeleton per detected person each frame
- Used by event logic for block detection and action classification

### `src/event_logic.py`
- **The core of Phase 1** — all basketball rules live here
- Consumes outputs from tracker, ball_tracker, pose_estimator
- Emits typed events: `ShotMade`, `ShotAttempted`, `Rebound`, `Block`, `Assist`, `Turnover`
- Rules are tunable via `config.yaml` (proximity thresholds, frame windows, etc.)
- Stateless per frame — game state managed by `game_state.py`

### `src/game_state.py`
- Single source of truth for the current game
- Tracks: score (Team A / Team B), possession, event log, FG attempts/makes
- Thread-safe (updated by CV pipeline, read by scoreboard UI)
- Serializes to JSON for data logger

### `src/scoreboard.py` (Phase 1.5)
- Pygame or OpenCV overlay rendering
- Reads from `GameState` — no direct CV coupling
- Displays: score, FG%, possession indicator, recent event feed
- Runs in a separate thread / process

### `src/data_logger.py`
- Writes timestamped event log to `output/game_YYYYMMDD.json`
- Phase 2: converts to Excel with pandas + openpyxl

### `config.yaml`
- All tunable parameters (never hardcoded in source):
  - Camera indices / RTSP URLs
  - Model weights path
  - Confidence thresholds (detection, tracking)
  - Court dimensions (feet) and homography points
  - Event logic thresholds (rebound proximity radius, shot attempt angle, etc.)
  - Output paths

### `main.py`
- Entry point — reads config, wires up all modules, starts pipeline loop

---

## Camera placement recommendations

### Minimum (2 cameras)
```
         [Cam 1]                    [Cam 2]
         ~10ft up                   ~10ft up
         behind/above basket 1      behind/above basket 2
```
Each camera covers its basket end — catches shot attempts and makes cleanly.

### Recommended (3 cameras)
```
         [Cam 1]     [Cam 3]        [Cam 2]
         basket 1    halfcourt      basket 2
                     ~15ft up
```
Cam 3 at halfcourt gives full-court player tracking for assist/turnover logic.

### Camera specs (minimum viable)
- 1080p @ 30fps (60fps preferred for fast action)
- Adequate low-light sensitivity (gym fluorescents are dim)
- Options: Logitech BRIO, GoPro Hero 12, Insta360 Link, USB webcam with manual exposure
- Avoid heavy fisheye lenses — distortion breaks court coordinate mapping

---

## Performance targets

| Component | Target | Notes |
|---|---|---|
| YOLOv11m inference | ≥60 FPS @ 1280px | RTX 4080 Super, single stream |
| ByteTrack overhead | <2ms per frame | Negligible vs. inference |
| Pose estimation | ≥30 FPS | Can run every N frames if needed |
| Full pipeline | ≥30 FPS | Hard minimum for real-time |
| Multi-camera | 2 streams @ 30 FPS | Each camera on its own thread |

---

## Key design decisions

**Why YOLOv11 + ByteTrack?**
Dominant real-time combo in 2025. ByteTrack's two-stage association recovers players during occlusions (screens, pile-ups) better than SORT or DeepSORT at this speed.

**Why not a single overhead camera?**
Occlusion from above is severe in basketball. Basket-angle cameras are essential for made-shot detection. Overhead adds full-court context.

**Why event logic is rule-based (not ML) for Phase 1?**
Collecting labeled event data from your specific gym/camera setup takes many games. Rule-based logic with tunable thresholds gets to a working system faster and is fully explainable when it's wrong.

**Why defer foul detection?**
Travelling and shooting fouls require precise multi-frame pose sequences. The training data for your specific court angles doesn't exist yet. Phase 2 data collection will enable this.
