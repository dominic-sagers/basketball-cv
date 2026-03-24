# Project Roadmap

## Phase 1 — Player-agnostic tracking

Goal: track basketball events without knowing who any player is. All stats are team-level.

### 1a — Detection baseline
- [x] Camera capture pipeline — `FileVideoSource`, `RTSPVideoSource`, `USBCameraSource` in `src/video_source.py`
- [x] YOLOv11 inference on live frames (players, ball, hoop) — `src/detector.py`
- [x] ByteTrack multi-object tracking with persistent IDs per session — `src/tracker.py`
- [x] FPS benchmark — pipeline sustains 30–60 FPS on RTX 4080 Super depending on scene complexity
- [ ] Court homography: map camera pixels to court coordinates

### 1b — Ball and hoop tracking
- [x] Made basket detection: `Ball_in_Basket` class detected by fine-tuned YOLOv11m with 45-frame debounce
- [ ] Ball trajectory tracking across frames
- [ ] Hoop zone definition (scoring region polygon)
- [ ] Shot attempt detection: ball in shooting trajectory toward hoop
- [ ] FG% running tally (made / attempted) per team

### 1c — Pose and action events
- [ ] Multi-person pose estimation (YOLOv11-pose or AlphaPose)
- [ ] Rebound detection: ball contacts rim/board → nearest player gains possession
- [ ] Block detection: defensive player hand intersects ball trajectory near basket
- [ ] Assist attribution: last pass before a made basket (traced via possession chain)
- [ ] Turnover detection: possession change not caused by a shot (heuristic, will need tuning)

### 1d — Game state and event log
- [x] `GameState` class: score `{"A": 0, "B": 0}`, event log, shot debounce — `src/game_state.py`
- [x] Event log: timestamped entries for every detected basket event
- [ ] Team assignment: heuristic based on jersey color clustering (no player identity)

### Completion criteria for Phase 1
- System correctly detects ≥85% of made baskets in test footage
- FG% tracked within ±5% of manual ground truth over a full game
- Rebound and block detection at ≥70% precision (acceptable for pickup game)
- All output is team-level only (Team A / Team B)

---

## Phase 1.5 — Real-time scoreboard overlay ✓ (Qt desktop app)

Goal: display a live scoreboard during the game, either projected or on a second monitor.

- [x] Desktop app UI — `src/app.py` (PyQt6): live preview, score panel, log panel, start/stop control
- [x] Live score display with manual +1/+2/+3/−1 adjustment buttons per team
- [x] Pipeline reads from `GameState` in real time — no coupling to CV internals
- [x] Dual-camera architecture implemented: Camera A → Team A basket, Camera B → Team B basket
- [ ] Optional: shot clock countdown
- [ ] OBS Studio integration for projector output (optional)

---

## Phase 2 — Player profiles and stat export

Goal: attach all Phase 1 stats to individual players.

### 2a — Player identification
- [ ] Jersey number OCR (SmolVLM2 fine-tuned or TrOCR on cropped player regions)
- [ ] Body re-identification for frames where jersey number is not visible
  - Gait signature, build, color histogram per tracked ID
- [ ] Face recognition as fallback (limited utility at gym camera angles)
- [ ] Player profile registry: name → jersey number mapping (manual setup before game)

### 2b — Per-player stat tracking
- [ ] All Phase 1 events re-attributed to individual players
- [ ] Per-player: points, FG%, rebounds, assists, blocks, turnovers
- [ ] Running leaderboard in scoreboard overlay

### 2c — Data export
- [ ] Post-game export to Excel (pandas + openpyxl)
- [ ] Per-player stat sheet
- [ ] Game timeline (event log as a readable table)
- [ ] Optional: matplotlib charts embedded in Excel

### 2d — Historical tracking
- [ ] SQLite database for game-over-game stats
- [ ] Season leaderboard
- [ ] Per-player trend charts

---

## Deferred / future

- Foul detection (travelling, shooting foul) — complex pose sequences, low priority
- 3D court reconstruction (multi-camera)
- Web dashboard for post-game review
- Automatic highlight clip generation
