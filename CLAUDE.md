# basketball-cv — Claude Code Context

## Project summary
Real-time basketball stat tracking and scoreboard system using computer vision.
Runs on a local machine (RTX 4080 Super 16GB, i7-10700K, 32GB RAM, Windows/Linux).
Cameras stream to the system during a live Monday indoor pickup game.

## Current phase
**Phase 1 — Player-agnostic tracking**
Do not build player identity features yet. All stats are team-level or event-level only.

## Phase roadmap
See `docs/roadmap.md` for full detail.
- Phase 1: Team FG%, made baskets, rebounds, blocks, assists, turnovers (no player identity)
- Phase 1.5: Real-time scoreboard overlay (projected or second monitor)
- Phase 2: Player profiles via jersey number OCR + body re-ID, Excel stat export

## Tech stack
See `docs/tech-stack.md` for rationale and versions.
- Python 3.11+
- Ultralytics YOLOv11 (detection + ByteTrack multi-object tracking)
- OpenCV (camera capture, frame preprocessing, court homography)
- YOLOv11-pose or AlphaPose (multi-person pose estimation)
- Pygame or OpenCV overlay (Phase 1.5 scoreboard)
- Pandas + openpyxl (Phase 2 Excel export)
- Roboflow datasets for fine-tuning (basketball-specific pretrained weights)

## Architecture overview
See `docs/architecture.md` for the full pipeline diagram and module breakdown.
Pipeline: cameras → frame preprocessing → YOLOv11 detection → [ByteTrack | ball tracker | pose estimator] → event logic engine → stat aggregator → scoreboard overlay + data log

## Coding conventions
- All modules in `src/` with clear single responsibilities
- Config values (camera indices, confidence thresholds, court dimensions) in `config.yaml` — never hardcoded
- Every detection/tracking param must be tunable from config without touching source code
- Use Python logging (not print) for debug output
- Type hints on all function signatures
- Write short docstrings on every class and public method

## Key constraints
- Inference must run at ≥30 FPS on RTX 4080 Super using CUDA — benchmark before adding features
- Do not add dependencies that require internet access at runtime (game is in a gym)
- Phase 1 produces no player-level stats — only team A / team B aggregates + event log

## Important context
- Indoor court, fixed camera positions, consistent lighting (gym fluorescents)
- ~10 players on court, no official jerseys (pickup game — mixed colors)
- Single source of truth for game state lives in `src/game_state.py`
- Event logic rules (what counts as a rebound, assist, etc.) live in `src/event_logic.py` and should be easy to tune

## Where to start
1. Read `docs/roadmap.md` and `docs/architecture.md` before writing any code
2. Run `python src/camera_test.py` to verify camera feeds are working
3. Run `python src/detection_test.py` to verify YOLOv11 + CUDA inference speed
4. Only then start on `src/tracker.py`
