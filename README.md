# basketball-cv

Real-time basketball stat tracking and scoreboard system using computer vision.
Built for a weekly indoor pickup game — no official scoring, no jerseys, just hoops.

Runs locally on an RTX 4080 Super (16GB VRAM). Cameras stream over Tailscale from Android phones at the gym to the desktop at home — no cloud, no internet dependency beyond the VPN tunnel.

## Demo

https://github.com/user-attachments/assets/7f8c7bc9-2876-44ff-992e-df60d2721fb2

---

## What it does (target)

- Detects players, ball, and hoop from 2 phone cameras — one per basket
- Streams live from the gym to a home desktop over Tailscale VPN
- Tracks every player and the ball frame-to-frame with persistent IDs (ByteTrack)
- Detects basketball events: made shots, rebounds, blocks, assists, turnovers
- Aggregates stats at the **team level** (Phase 1) — no player identity needed yet
- Displays a live scoreboard in a desktop app (PyQt6) with manual score adjustment
- Post-processes recorded footage to blur faces for privacy (YOLOv8-face + SAM2)
- Exports per-player stats to Excel after the game (Phase 2)

---

## How it's used

A typical game session:

1. **At the gym** — two Android phones running [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) are positioned behind each basket, connected to Tailscale
2. **At home** — the desktop app connects to both phone streams over Tailscale and runs the full CV pipeline in real time
3. **After the game** — recorded footage is post-processed with `blur_footage.py` to blur all faces before sharing

```bash
# Launch the desktop app — single camera
python src/app.py --rtsp http://<phone-tailscale-ip>:8080/video --chunk-seconds 5

# Dual camera (one per basket)
python src/app.py --rtsp http://<camA-ip>:8080/video --rtsp2 http://<camB-ip>:8080/video --chunk-seconds 5

# With output saved (auto-concatenated on stop)
python src/app.py --rtsp http://<camA-ip>:8080/video --save-output store/output/game.mp4

# Post-process recorded footage to blur faces
python src/blur_footage.py store/output/game.mp4

# Adjust face detection sensitivity (higher imgsz catches far-away faces)
python src/blur_footage.py store/output/game.mp4 --face-imgsz 1920 --face-conf 0.1
```

---

## Current state

### Phase 1 — Player-agnostic tracking (in progress)

**Done**
- [x] `video_source.py` — file / RTSP (threaded, auto-reconnect) / USB camera abstraction; `StreamChunkRecorder` captures live streams to chunk files so no frames are dropped
- [x] `detector.py` — YOLOv11 inference → typed `Detection` objects
- [x] `tracker.py` — ByteTrack via `model.track(persist=True)` → typed `Track` objects
- [x] `visualizer.py` — bounding boxes, track IDs, fading ball trail, score overlay, FPS panel
- [x] `game_state.py` — score tracking, `Ball_in_Basket` detection, 45-frame shot debounce
- [x] `pipeline_test.py` — full source → detect/track → score → visualize loop with `--save-output` / `--save-log`
- [x] `train.py` — reads all training params from `config.yaml`; swap dataset in one line
- [x] `config.yaml` — all thresholds, source types, paths, and training params; nothing hardcoded
- [x] Docker setup with CUDA 12.4 support (GPU passthrough, camera mounts)
- [x] `store/` — single DVC-tracked directory for all large assets (weights, dataset, footage, output)
- [x] DVC over Tailscale SSH — `store.dvc` committed; setup documented in `docs/dvc-setup.md`
- [x] Base model benchmarked: 14.8% ball detection, 48.5 FPS at imgsz=960 (`yolo11m.pt` COCO)
- [x] Fine-tuned on Roboflow basketball dataset (9,599 imgs, 5 classes) — **27.1% ball detection, mAP50=0.956**
- [x] Multi-dataset support — each dataset in `store/dataset/<name>/`; active dataset set in `config.yaml`
- [x] Test footage recorded at gym-approximated angles

**Up next**
- [ ] `preprocessor.py` — court ROI crop, resize, denoise
- [ ] `ball_tracker.py` — ball trajectory analysis, hoop zone entry detection
- [ ] `pose_estimator.py` — YOLOv11-pose for block/rebound detection
- [ ] `event_logic.py` — basketball rules engine (rebound, assist, block, turnover)
- [ ] Court homography: map camera pixels to court coordinates

### Phase 1.5 — Real-time scoreboard overlay ✓

- [x] `app.py` — PyQt6 desktop app: live video preview (raw + annotated), score panel, log panel, start/stop
- [x] Live score with manual +1/+2/+3/−1 adjustment buttons per team
- [x] Dual-camera support — Camera A → Team A basket, Camera B → Team B basket
- [x] Raw RTSP preview panel alongside the annotated output (single-cam mode)
- [x] Per-run timestamped output directories — each session saved separately
- [x] Stream health monitoring — warns on dropped frames / poor connection

### Privacy

- [x] `face_blur.py` + `blur_footage.py` — post-processing face blur using YOLOv8-face detection + SAM2 pixel-precise segmentation. Detects faces on keyframes, propagates masks across all frames, applies Gaussian blur only to face pixels. Handles distant court faces via `--face-imgsz`.

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
│   ├── app.py               # PyQt6 desktop app — primary interface for live games
│   ├── video_source.py      # FileVideoSource / RTSPVideoSource / USBCameraSource / StreamChunkRecorder
│   ├── detector.py          # YOLOv11 inference → Detection dataclass
│   ├── tracker.py           # ByteTrack → Track dataclass (persistent IDs)
│   ├── visualizer.py        # annotated frame rendering (boxes, trail, score, FPS panel)
│   ├── game_state.py        # score, event log, shot debouncing
│   ├── face_blur.py         # YOLOv8-face + SAM2 face segmentation and blur
│   ├── blur_footage.py      # CLI: post-process a recorded video to blur all faces
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
│   ├── output/              # annotated clips, game logs (one subdirectory per run)
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
# Requires Tailscale connection to the DVC remote — see docs/dvc-setup.md
dvc pull
```

Or download the dataset manually and place it under `store/dataset/<name>/` — see [`docs/training.md`](docs/training.md).

### 3. Verify GPU

```bash
python src/detection_test.py
# Expected: RTX 4080 SUPER, ≥60 FPS on synthetic 1080p frames
```

### 4. Run the desktop app (live game)

```bash
# Phone streaming via IP Webcam over Tailscale
python src/app.py --rtsp http://<phone-tailscale-ip>:8080/video --chunk-seconds 5
```

The app shows a raw preview and annotated output side by side, with live score management and a log panel.

### 5. Run on a pre-recorded file

```bash
python src/pipeline_test.py --file store/footage/your_clip.mp4
```

### 6. Blur faces in recorded footage

```bash
# Basic (auto-downloads YOLOv8-face + SAM2 on first run)
python src/blur_footage.py store/output/game.mp4

# Higher sensitivity for far-away faces
python src/blur_footage.py store/output/game.mp4 --face-imgsz 1920

# Tune all options
python src/blur_footage.py store/output/game.mp4 --face-imgsz 1920 --face-conf 0.1 --chunk-size 60
```

### 7. Useful pipeline flags

```bash
# Detection only (no tracking) — good for calibrating confidence threshold
python src/pipeline_test.py --file store/footage/clip.mp4 --detect-only

# Slow playback to inspect frame by frame
python src/pipeline_test.py --file store/footage/clip.mp4 --playback-speed 0.25

# Save annotated output clip
python src/pipeline_test.py --file store/footage/clip.mp4 --save-output store/output/annotated.mp4

# Headless (no window — Docker or SSH)
python src/pipeline_test.py --file store/footage/clip.mp4 --no-preview

# Live RTSP stream (buffered — no dropped frames)
python src/pipeline_test.py --rtsp http://<phone-ip>:8080/video --stream-buffer --chunk-seconds 5
```

### 8. Docker

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

### Current setup (phones over Tailscale)

```
[Phone A — IP Webcam]          [Phone B — IP Webcam]
Tailscale IP                   Tailscale IP
behind basket (Team A end)     behind basket (Team B end)
        │                               │
        └───────────── Tailscale VPN ───┘
                            │
                    [Desktop — RTX 4080 Super]
                    running app.py
```

Phones stream MJPEG/H.264 over Tailscale to the desktop. `StreamChunkRecorder` captures each stream to local chunk files; the pipeline processes them sequentially at its own pace — every frame is preserved even if inference is slower than the stream rate.

### Recommended phone settings (IP Webcam)

- Resolution: 720p (1080p if signal is strong)
- FPS: 25
- Encoder: H.264
- Quality: 50–60%

### Video source types (`config.yaml`)

| Type | Use case | Config key |
|---|---|---|
| `rtsp` / `http` | Live phone streams at the gym | `url:` |
| `file` | Pre-recorded test footage | `path:` |
| `usb`  | Direct USB cameras on the host | `index:` |

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
| CPU | Intel i7-14700K |
| RAM | 32GB |
| OS | Windows 11 |
| Cameras | Android phones running IP Webcam (1080p / 720p @ 25fps) |
| Networking | Tailscale VPN — gym phones → home desktop |

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
