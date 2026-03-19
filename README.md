# basketball-cv

Real-time basketball stat tracking and scoreboard system using computer vision.
Built for a weekly indoor pickup game — no official scoring, no jerseys, just hoops.

Runs locally on an RTX 4080 Super (16GB VRAM). No cloud, no internet dependency at game time.

---

## What it does (target)

- Detects players, ball, and hoop from 2 fixed cameras — one per basket
- Tracks every player and the ball frame-to-frame with persistent IDs (ByteTrack)
- Detects basketball events: made shots, rebounds, blocks, assists, turnovers
- Aggregates stats at the **team level** (Phase 1) — no player identity needed yet
- Displays a live scoreboard overlay on a second monitor or projector (Phase 1.5)
- Exports per-player stats to Excel after the game (Phase 2)

---

## Current state (Phase 1 — in progress)

### Done
- [x] `video_source.py` — file / RTSP (threaded, auto-reconnect) / USB camera abstraction
- [x] `source_test.py` — verify any source opens and delivers frames at target FPS
- [x] `detector.py` — YOLOv11 wrapper, returns typed `Detection` objects
- [x] `tracker.py` — ByteTrack via `model.track(persist=True)`, returns typed `Track` objects
- [x] `visualizer.py` — bounding boxes, track IDs, fading ball trajectory trail, FPS info panel
- [x] `pipeline_test.py` — source → detect/track → visualize, `--save-output` and `--save-log` for offline analysis
- [x] Docker setup with CUDA 12.4 support (GPU passthrough, camera mounts)
- [x] `config.yaml` — all thresholds, source types, and paths; never hardcoded in source
- [x] Test footage recorded at gym-approximated angles (`test_footage/basketballcv-sample-1.mov`)
- [x] Base model benchmarked: 14.8% ball detection rate, 48.5 FPS at imgsz=960 with `yolo11m.pt`
- [x] Basketball-specific dataset downloaded (9,599 images, 5 classes incl. Ball_in_Basket, Basket)
- [x] Fine-tuning pipeline in place — see `docs/training.md`

### In progress
- [ ] Fine-tuning `yolo11m` on basketball dataset (`Basketball-detection-1/`)
- [ ] Target: ≥50% ball detection rate (up from 14.8% baseline)

### Up next
- [ ] `preprocessor.py` — court ROI crop, resize, denoise
- [ ] `ball_tracker.py` — ball trajectory analysis, shot detection, hoop zone entry
- [ ] `pose_estimator.py` — YOLOv11-pose for block/rebound detection
- [ ] `game_state.py` — single source of truth for score, possession, event log
- [ ] `event_logic.py` — basketball rules engine (made shot, rebound, assist, block, turnover)

See [`docs/roadmap.md`](docs/roadmap.md) for the full phase breakdown.

---

## Project structure

```
basketball-cv/
├── config.yaml              # all tunable parameters — start here
├── requirements.txt
├── Dockerfile               # CUDA 12.4 + Python 3.13
├── docker-compose.yml
│
├── src/
│   ├── video_source.py      # FileVideoSource / RTSPVideoSource / USBCameraSource
│   ├── detector.py          # YOLOv11 inference → Detection dataclass
│   ├── tracker.py           # ByteTrack → Track dataclass (persistent IDs)
│   ├── visualizer.py        # annotated frame rendering (boxes, trail, FPS panel)
│   │
│   ├── pipeline_test.py     # end-to-end test: source → detect/track → visualize
│   ├── source_test.py       # verify a video source opens and delivers frames
│   ├── camera_test.py       # scan USB camera indices
│   └── detection_test.py    # CUDA check + YOLOv11 FPS benchmark
│
├── test_footage/            # drop video files here (gitignored)
├── weights/                 # fine-tuned .pt files + training runs (gitignored)
├── output/                  # annotated clips, game logs, frame logs (gitignored)
├── Basketball-detection-1/  # Roboflow dataset — 9,599 images, 5 classes (gitignored)
│
└── docs/
    ├── roadmap.md           # phase-by-phase feature plan
    ├── architecture.md      # pipeline diagram and module breakdown
    ├── tech-stack.md        # dependency rationale and setup
    ├── models.md            # model options, Roboflow datasets, evaluation results
    ├── training.md          # how to download dataset and run fine-tuning
    └── docker.md            # Docker setup and camera passthrough guide
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Verify GPU

```bash
python src/detection_test.py
# Expected: RTX 4080 SUPER, ≥60 FPS on synthetic 1080p frames
```

### 3. Test a video file

Drop a clip into `test_footage/` then:

```bash
python src/pipeline_test.py --file test_footage/your_clip.mp4
```

Keyboard shortcuts in the preview window:
- `Q` — quit

### 4. Useful flags

```bash
# Detection only (no tracking) — good for calibrating confidence threshold
python src/pipeline_test.py --file test_footage/clip.mp4 --detect-only

# Slow playback to inspect frame by frame
python src/pipeline_test.py --file test_footage/clip.mp4 --playback-speed 0.25

# Save annotated output clip
python src/pipeline_test.py --file test_footage/clip.mp4 --save-output output/annotated.mp4

# Save per-frame detection log as JSON for offline analysis
python src/pipeline_test.py --file test_footage/clip.mp4 --no-preview --save-log output/log.json

# Headless (no window — Docker or SSH)
python src/pipeline_test.py --file test_footage/clip.mp4 --no-preview

# Test an RTSP stream
python src/pipeline_test.py --rtsp rtsp://192.168.1.100:554/stream
```

### 5. Docker

```bash
docker-compose build
docker-compose run --rm basketball-cv python src/detection_test.py
```

See [`docs/docker.md`](docs/docker.md) for GPU passthrough, camera setup, and headless use.

---

## Camera setup

### Current plan (2 cameras)

```
[Cam 1]                        [Cam 2]
~8-10ft up                     ~8-10ft up
behind basket 1 (Team A end)   behind basket 2 (Team B end)
```

Each camera covers its own basket for shot detection. Future: add a halfcourt overhead camera for full-court tracking (assists, turnovers).

### Video source types (config.yaml)

| Type | Use case | Config key |
|---|---|---|
| `file` | Pre-recorded test footage | `path:` |
| `rtsp` | Live network cameras at the gym | `url:` |
| `usb`  | Direct USB cameras on the host | `index:` |

Switch between types by editing `config.yaml` — no code changes needed.

---

## Model weights

The base `yolo11m.pt` (COCO) is used by default and auto-downloads on first run. A fine-tuned basketball model is in training — see [`docs/training.md`](docs/training.md) for how to reproduce it.

| Model | Ball detection | FPS (imgsz=960) | Classes |
|---|---|---|---|
| `yolo11m.pt` (COCO baseline) | 14.8% | 48.5 | person, sports ball |
| `basketball-ft/best.pt` (fine-tuned) | TBD | ~48 | Ball, Ball_in_Basket, Player, Basket, Player_Shooting |

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4080 Super (16GB VRAM) |
| CPU | Intel i7-10700K |
| RAM | 32GB |
| OS | Windows 11 / WSL2 for Docker |
| Cameras | TBD — targeting 1080p @ 30fps |

---

## Docs

| File | Contents |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | Feature phases and completion criteria |
| [`docs/architecture.md`](docs/architecture.md) | Pipeline diagram and module responsibilities |
| [`docs/tech-stack.md`](docs/tech-stack.md) | Dependency rationale, setup steps |
| [`docs/models.md`](docs/models.md) | Model options, Roboflow datasets, evaluation results |
| [`docs/training.md`](docs/training.md) | Dataset download, fine-tuning, resuming training |
| [`docs/docker.md`](docs/docker.md) | Docker setup, GPU access, camera passthrough |
