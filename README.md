# basketball-cv

Real-time basketball stat tracking and scoreboard system using computer vision.
Built for a weekly indoor pickup game — no official scoring, no jerseys, just hoops.

Runs locally on an RTX 4080 Super (16GB VRAM). No cloud, no internet dependency at game time.

## Demo

https://github.com/user-attachments/assets/7f8c7bc9-2876-44ff-992e-df60d2721fb2

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
- [x] `detector.py` — YOLOv11 inference → typed `Detection` objects
- [x] `tracker.py` — ByteTrack via `model.track(persist=True)` → typed `Track` objects
- [x] `visualizer.py` — bounding boxes, track IDs, fading ball trail, score overlay, FPS panel
- [x] `game_state.py` — score tracking, Ball_in_Basket detection, 45-frame shot debounce
- [x] `pipeline_test.py` — full source → detect/track → score → visualize loop with `--save-output` / `--save-log`
- [x] `train.py` — reads all training params from `config.yaml`; swap dataset in one line
- [x] `config.yaml` — all thresholds, source types, paths, and training params; nothing hardcoded
- [x] Docker setup with CUDA 12.4 support (GPU passthrough, camera mounts)
- [x] `store/` — single DVC-tracked directory for all large assets (weights, dataset, footage, output)
- [x] DVC initialised — `store.dvc` committed; SSH remote setup documented in `docs/dvc-setup.md`
- [x] Base model benchmarked: 14.8% ball detection, 48.5 FPS at imgsz=960 (`yolo11m.pt` COCO)
- [x] Fine-tuned on Roboflow basketball dataset (9,599 imgs, 5 classes) — **27.1% ball detection, mAP50=0.956**
- [x] Multi-dataset support — each dataset in `store/dataset/<name>/`; active dataset set in `config.yaml`
- [x] Test footage recorded at gym-approximated angles

### Up next
- [ ] DVC remote configured on Ubuntu server (`dvc push` / `dvc pull` for weights + dataset)
- [ ] `preprocessor.py` — court ROI crop, resize, denoise
- [ ] `ball_tracker.py` — ball trajectory analysis, hoop zone entry detection
- [ ] `pose_estimator.py` — YOLOv11-pose for block/rebound detection
- [ ] `event_logic.py` — basketball rules engine (rebound, assist, block, turnover)

See [`docs/roadmap.md`](docs/roadmap.md) for the full phase breakdown.

---

## Project structure

```
basketball-cv/
├── config.yaml              # all tunable parameters — start here
├── store.dvc                # DVC pointer to store/ (committed; data pulled separately)
├── requirements.txt
├── Dockerfile               # CUDA 12.4 + Python 3.13
├── docker-compose.yml
│
├── src/
│   ├── video_source.py      # FileVideoSource / RTSPVideoSource / USBCameraSource
│   ├── detector.py          # YOLOv11 inference → Detection dataclass
│   ├── tracker.py           # ByteTrack → Track dataclass (persistent IDs)
│   ├── visualizer.py        # annotated frame rendering (boxes, trail, score, FPS panel)
│   ├── game_state.py        # score, event log, shot debouncing
│   │
│   ├── train.py             # fine-tune YOLOv11 from config.yaml (swap dataset in one line)
│   ├── pipeline_test.py     # end-to-end test: source → detect/track → score → visualize
│   ├── source_test.py       # verify a video source opens and delivers frames
│   ├── camera_test.py       # scan USB camera indices
│   └── detection_test.py    # CUDA check + YOLOv11 FPS benchmark
│
├── store/                   # DVC-tracked (gitignored) — run: dvc pull
│   ├── footage/             # test video files
│   ├── weights/             # fine-tuned .pt files + training run artifacts
│   ├── output/              # annotated clips, game logs, frame logs
│   └── dataset/             # one subdirectory per dataset
│       └── basketball-srfkd/    # Roboflow basketball-detection-srfkd (9,599 imgs)
│
└── docs/
    ├── roadmap.md           # phase-by-phase feature plan
    ├── architecture.md      # pipeline diagram and module breakdown
    ├── tech-stack.md        # dependency rationale and setup
    ├── models.md            # model options, Roboflow datasets, evaluation results
    ├── training.md          # dataset layout, fine-tuning, multi-dataset switching
    ├── dvc-setup.md         # DVC SSH remote setup and collaborator onboarding
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

### 2. Pull data (weights + dataset + footage)

```bash
# Configure your DVC SSH credentials first — see docs/dvc-setup.md
dvc pull
```

Or download the dataset manually and place it under `store/dataset/<name>/` — see [`docs/training.md`](docs/training.md).

### 3. Verify GPU

```bash
python src/detection_test.py
# Expected: RTX 4080 SUPER, ≥60 FPS on synthetic 1080p frames
```

### 4. Run the pipeline

```bash
python src/pipeline_test.py --file store/footage/your_clip.mp4
```

Keyboard shortcuts in the preview window:
- `Q` — quit

### 5. Useful flags

```bash
# Detection only (no tracking) — good for calibrating confidence threshold
python src/pipeline_test.py --file store/footage/clip.mp4 --detect-only

# Slow playback to inspect frame by frame
python src/pipeline_test.py --file store/footage/clip.mp4 --playback-speed 0.25

# Save annotated output clip
python src/pipeline_test.py --file store/footage/clip.mp4 --save-output store/output/annotated.mp4

# Save per-frame detection log as JSON
python src/pipeline_test.py --file store/footage/clip.mp4 --no-preview --save-log store/output/log.json

# Headless (no window — Docker or SSH)
python src/pipeline_test.py --file store/footage/clip.mp4 --no-preview

# Live RTSP stream
python src/pipeline_test.py --rtsp rtsp://192.168.1.100:554/stream
```

### 6. Docker

```bash
docker-compose build
docker-compose run --rm basketball-cv python src/detection_test.py
```

See [`docs/docker.md`](docs/docker.md) for GPU passthrough, camera setup, and headless use.

---

## Training

All training parameters live in `config.yaml → training:`. To fine-tune:

```bash
python src/train.py
```

To switch dataset, change one line in `config.yaml`:

```yaml
training:
  dataset: "store/dataset/my-gym-footage/data.yaml"
```

See [`docs/training.md`](docs/training.md) for dataset layout, adding new datasets, and resuming interrupted runs.

---

## Camera setup

### Current plan (2 cameras)

```
[Cam 1]                        [Cam 2]
~8-10ft up                     ~8-10ft up
behind basket 1 (Team A end)   behind basket 2 (Team B end)
```

Each camera covers its own basket for shot detection. Future: halfcourt overhead camera for full-court tracking.

### Video source types (config.yaml)

| Type | Use case | Config key |
|---|---|---|
| `file` | Pre-recorded test footage | `path:` |
| `rtsp` | Live network cameras at the gym | `url:` |
| `usb`  | Direct USB cameras on the host | `index:` |

Switch between types by editing `config.yaml` — no code changes needed.

---

## Model weights

| Model | Ball detection | FPS (imgsz=960) | mAP50 | Classes |
|---|---|---|---|---|
| `yolo11m.pt` (COCO baseline) | 14.8% | 48.5 | — | person, sports ball |
| `basketball-ft/best.pt` (fine-tuned) | **27.1%** | ~48 | **0.956** | Ball, Ball_in_Basket, Player, Basket, Player_Shooting |

Weights live in `store/weights/` (DVC-tracked). Pull via `dvc pull` or reproduce via `python src/train.py`. See [`docs/training.md`](docs/training.md).

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
| [`docs/training.md`](docs/training.md) | Dataset layout, fine-tuning, multi-dataset switching |
| [`docs/dvc-setup.md`](docs/dvc-setup.md) | DVC SSH remote setup and collaborator onboarding |
| [`docs/docker.md`](docs/docker.md) | Docker setup, GPU access, camera passthrough |
