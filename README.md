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
- [x] Video source abstraction: file, RTSP, and USB camera inputs via unified interface
- [x] YOLOv11 detection wrapper (`detector.py`)
- [x] ByteTrack multi-object tracking wrapper (`tracker.py`)
- [x] Real-time visualizer: bounding boxes, track IDs, ball trajectory trail, FPS panel (`visualizer.py`)
- [x] Pipeline test harness: source → detect/track → visualize with save-to-file support (`pipeline_test.py`)
- [x] Docker setup with CUDA 12.1 support (GPU passthrough, camera mounts)
- [x] Config-driven — all thresholds, paths, and camera settings in `config.yaml`

### In progress
- [ ] Capturing test footage at gym angles for iterating on detection quality
- [ ] Evaluating pretrained basketball weights (see `docs/models.md`)

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
├── Dockerfile               # CUDA 12.1 + Python 3.11
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
├── test_footage/            # drop .mp4 files here (gitignored)
├── weights/                 # custom fine-tuned .pt files (gitignored)
├── output/                  # game logs, annotated clips (gitignored)
│
└── docs/
    ├── roadmap.md           # phase-by-phase feature plan
    ├── architecture.md      # pipeline diagram and module breakdown
    ├── tech-stack.md        # dependency rationale and setup
    ├── models.md            # pretrained basketball model options
    └── docker.md            # Docker setup and camera passthrough guide
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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

The base `yolo11m.pt` model (auto-downloaded on first run) detects `person` and `sports ball` from COCO. For better basketball accuracy, see [`docs/models.md`](docs/models.md) for pretrained basketball-specific weights and Roboflow Universe datasets.

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
| [`docs/models.md`](docs/models.md) | Pretrained basketball model options and download guide |
| [`docs/docker.md`](docs/docker.md) | Docker setup, GPU access, camera passthrough |
