# Project Roadmap

## Phase 1 — Player-agnostic tracking

Goal: track basketball events without knowing who any player is. All stats are team-level.

### 1a — Detection baseline
- [ ] Camera capture pipeline (1–3 cameras via OpenCV)
- [ ] YOLOv11 inference on live frames (players, ball, hoop)
- [ ] ByteTrack multi-object tracking with persistent IDs per session
- [ ] Court homography: map camera pixels to court coordinates
- [ ] FPS benchmark — must sustain ≥30 FPS on RTX 4080 Super

### 1b — Ball and hoop tracking
- [ ] Ball trajectory tracking across frames
- [ ] Hoop zone definition (scoring region polygon)
- [ ] Made basket detection: ball enters hoop zone with downward motion over N frames
- [ ] Shot attempt detection: ball in shooting trajectory toward hoop
- [ ] FG% running tally (made / attempted) per team

### 1c — Pose and action events
- [ ] Multi-person pose estimation (YOLOv11-pose or AlphaPose)
- [ ] Rebound detection: ball contacts rim/board → nearest player gains possession
- [ ] Block detection: defensive player hand intersects ball trajectory near basket
- [ ] Assist attribution: last pass before a made basket (traced via possession chain)
- [ ] Turnover detection: possession change not caused by a shot (heuristic, will need tuning)

### 1d — Game state and event log
- [ ] `GameState` class: score, possession, shot clock (optional), foul count
- [ ] Event log: timestamped JSON entries for every detected event
- [ ] Team assignment: heuristic based on jersey color clustering (no player identity)

### Completion criteria for Phase 1
- System correctly detects ≥85% of made baskets in test footage
- FG% tracked within ±5% of manual ground truth over a full game
- Rebound and block detection at ≥70% precision (acceptable for pickup game)
- All output is team-level only (Team A / Team B)

---

## Phase 1.5 — Real-time scoreboard overlay

Goal: display a live scoreboard during the game, either projected or on a second monitor.

- [ ] Scoreboard UI (Pygame or OpenCV overlay)
- [ ] Live score, FG%, possession indicator
- [ ] Optional: shot clock countdown
- [ ] OBS Studio integration for projector output (optional)
- [ ] Overlay reads from `GameState` in real time — no coupling to CV pipeline internals

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
