# System Architecture

## Pipeline overview

```
[Phone/Camera A]        [Phone/Camera B]
     |                        |
[StreamChunkRecorder]  [StreamChunkRecorder]   ← background thread each; writes N-second chunk files
     |                        |
[chunk_queue]          [chunk_queue]
     |                        |
[run_pipeline]         [run_pipeline]           ← processes FileVideoSource chunks sequentially
     |                        |
[YOLOv11 + ByteTrack]  [YOLOv11 + ByteTrack]   ← GPU inference per chunk
     |                        |
[GameState (A)]        [GameState (B)]           ← independent per camera; score events emitted via Qt signal
     |                        |
     └──────────┬─────────────┘
                |
         [BasketballApp]                         ← Qt6 main window owns authoritative score
                |
     ┌──────────┼──────────┐
     |          |           |
[VideoPanel] [ScorePanel] [LogPanel]
(live frames) (+1/+2/+3)  (all log output)
```

---

## Module breakdown

### Built ✓

### `src/video_source.py`
- `FileVideoSource`, `RTSPVideoSource` (threaded, always-latest-frame, auto-reconnect), `USBCameraSource`
- `StreamChunkRecorder` — records a live RTSP/HTTP stream to sequential N-second `.mp4` chunk files on disk in a background thread. Completed chunk paths are placed on a queue for the pipeline to consume at its own pace. Eliminates dropped frames at the cost of processing delay (one chunk length). Each camera gets its own chunk subdirectory.
- `VideoSourceFactory.from_config()` — instantiates the right source from config.yaml
- All sources share the `VideoSource` ABC: `open()`, `read()`, `release()`, `fps`, `name`

### `src/detector.py`
- Wraps Ultralytics YOLOv11; returns typed `Detection` dataclass (bbox, class_name, confidence, center)
- `Detector.from_config(model_cfg)` — loads weights, device, confidence, class_map from config
- Classes: `Ball`, `Ball_in_Basket`, `Player`, `Basket`, `Player_Shooting` (fine-tuned model)

### `src/tracker.py`
- Wraps ByteTrack via `model.track(persist=True)`; returns typed `Track` dataclass (extends Detection with track_id)
- `Tracker.reset()` — clears track state between sources / between chunks

### `src/visualizer.py`
- Bounding boxes colour-coded by class, track ID labels, confidence scores
- Fading ball trajectory trail (40-frame deque)
- Centred scoreboard overlay with team colours and orange flash on basket
- FPS + source info panel

### `src/game_state.py`
- Single source of truth for score (`{"A": 0, "B": 0}`), event log
- `process_frame(tracks, frame_number, source_name)` — detects `Ball_in_Basket`, increments score, applies 45-frame shot debounce
- `from_config(cfg)` — reads `event_logic.shot_cooldown_frames` from config
- Each `PipelineWorker` has its own independent `GameState`; the Qt app owns the authoritative merged score

### `src/pipeline_test.py`
- End-to-end loop: source → detect/track → game_state → visualize → optional output
- `run_pipeline()` — core loop, accepts optional `frame_callback(frame, fps)` and `stop_event` for Qt integration
- `--save-output PATH` — write annotated mp4
- `--save-log PATH` — write per-frame JSON with ball detection rate
- `--stream-buffer` — record RTSP stream to chunk files via `StreamChunkRecorder`, process chunks sequentially; eliminates dropped frames at the cost of a one-chunk delay
- `--chunk-seconds N` — chunk duration (default 30; use 5 for low-latency preview)
- `--chunk-dir DIR` — where chunk files are written
- `--inference-every N` — run inference every N frames, reuse last result for intermediate frames; halves inference load at N=2
- `--detect-only`, `--playback-speed`, `--no-preview` flags
- Auto-concatenates annotated chunks on stop and deletes chunk files after successful concat

### `src/app.py`
- PyQt6 desktop application — the primary interface for live game sessions
- **`PipelineWorker(QThread)`** — runs `StreamChunkRecorder` + `run_pipeline` in a background thread; emits Qt signals for frames, score events, FPS, status
- **`ScorePanel`** — live score display with manual +1/+2/+3/−1 adjustment buttons per team
- **`ControlPanel`** — Start / Stop+Save buttons, live status and FPS display
- **`VideoPanel`** — displays annotated frames from `frame_callback` without a cv2 window
- **`LogPanel`** — scrolling log widget capturing all Python logging output via `_QtLogHandler`
- **Dual-camera support**: `--rtsp2 URL` adds Camera B; both workers run in parallel, each with its own `GameState`; `camera_team` parameter routes score events to the correct team; Camera B panel hidden until second URL is provided
- **Score ownership**: `BasketballApp` holds the authoritative `_score` dict; all score changes (auto detection + manual buttons) flow through `_apply_score(team, delta)`

Usage:
```bash
# Single camera
python src/app.py --rtsp http://100.88.2.45:8080/video --chunk-seconds 5

# Dual camera
python src/app.py --rtsp http://<camA>/video --rtsp2 http://<camB>/video --chunk-seconds 5

# With save output (auto-concatenated on stop)
python src/app.py --rtsp http://<url>/video --chunk-seconds 5 --save-output store/output/game.mp4
```

### `src/face_blur.py`
- `FaceBlur` class with two backends: OpenCV DNN res10 SSD (primary) → Haar cascade fallback
- `blur_player_heads(frame, player_boxes)` — blurs the upper 30% of each player bounding box (head region); reliable at any camera angle and distance, uses existing tracker output
- `process(frame)` — falls back to face detector if no tracker boxes available
- Auto-downloads DNN model weights to `store/weights/face/` on first use

### `src/blur_footage.py`
- Standalone post-processing script: reads any recorded video, outputs a face-blurred mp4
- Uses `FaceBlur.blur_player_heads` with bounding boxes re-derived from the tracker
- Usage: `python src/blur_footage.py input.mp4 --output output_blurred.mp4`

### `src/train.py`
- Reads all training params from `config.yaml → training:`
- `--resume` flag continues from last checkpoint (`store/weights/<output_name>/weights/last.pt`)
- Saves weights to `store/weights/<output_name>/`
- Usage: `python src/train.py`

### `src/camera_test.py` / `src/source_test.py` / `src/detection_test.py`
- Diagnostic scripts: scan USB indices, verify a source opens, benchmark CUDA inference FPS

---

### Planned

### `src/preprocessor.py`
- Resize frames to model input size
- Optional: denoise, contrast adjustment for poor gym lighting
- Court ROI crop (remove scoreboard / bleachers from inference region)

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

### `src/data_logger.py`
- Writes timestamped event log to `store/output/game_YYYYMMDD.json`
- Phase 2: converts to Excel with pandas + openpyxl

---

## Camera placement recommendations

### Current setup (phones on Tailscale)
```
         [Phone A]                   [Phone B]
         IP Webcam app               IP Webcam app
         ~head height                ~head height
         one basket end              other basket end
         Team A basket               Team B basket
```
Phones stream over Tailscale VPN to the desktop. `StreamChunkRecorder` captures each stream independently. Camera A credits Team A on basket detection; Camera B credits Team B.

### Minimum fixed-camera setup (2 cameras)
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
- 1080p @ 30fps (720p acceptable for phone streams)
- Adequate low-light sensitivity (gym fluorescents are dim)
- Options: Android phone with IP Webcam app (current), Logitech BRIO, GoPro Hero 12, USB webcam
- Avoid heavy fisheye lenses — distortion breaks court coordinate mapping

---

## Performance targets

| Component | Target | Notes |
|---|---|---|
| YOLOv11m inference | ≥30 FPS @ 720p | RTX 4080 Super; varies with player count (6–75 FPS observed) |
| ByteTrack overhead | <2ms per frame | Negligible vs. inference |
| Stream capture | 25 FPS | `StreamChunkRecorder` reads at full source rate, no dropped frames |
| Pipeline lag | ~1 chunk behind | At chunk-seconds=5, preview is ~10s behind stream |
| Chunk concat | <5s | ffmpeg copy — no re-encode |

---

## Key design decisions

**Why chunked streaming instead of always-latest-frame?**
The pipeline (12–20 FPS with players) can't sustain 25 FPS source rate. The always-latest-frame approach in `RTSPVideoSource` drops frames silently, causing choppy annotated output. `StreamChunkRecorder` captures every frame to disk at full rate; the pipeline processes chunks sequentially via `FileVideoSource` — same path as pre-recorded footage, zero dropped frames, at the cost of one chunk of delay.

**Why PyQt6 for the app instead of Pygame?**
Pygame is adequate for a scoreboard overlay but doesn't support proper app-style layouts (buttons, log panel, dual video feeds). PyQt6 provides a real layout engine, thread-safe signals for pipeline→UI communication, and looks like a proper desktop application.

**Why independent GameState per camera?**
Sharing a single `GameState` between two pipeline threads requires locking and complicates the remap logic (which team does each basket belong to?). Independent `GameState` objects per camera are simpler and correct: each camera detects baskets and emits `score_event(team, delta)` to the Qt app, which owns the authoritative score.

**Why YOLOv11 + ByteTrack?**
Dominant real-time combo in 2025. ByteTrack's two-stage association recovers players during occlusions (screens, pile-ups) better than SORT or DeepSORT at this speed.

**Why event logic is rule-based (not ML) for Phase 1?**
Collecting labeled event data from your specific gym/camera setup takes many games. Rule-based logic with tunable thresholds gets to a working system faster and is fully explainable when it's wrong.
