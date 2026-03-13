# basketball-cv source modules
# See docs/architecture.md for full description of each module

# Suggested build order:
# 1. camera_test.py   — verify camera feeds
# 2. detection_test.py — verify CUDA + YOLOv11 FPS
# 3. camera.py        — camera capture abstraction
# 4. preprocessor.py  — frame preprocessing
# 5. detector.py      — YOLOv11 wrapper
# 6. tracker.py       — ByteTrack wrapper
# 7. ball_tracker.py  — ball + hoop logic
# 8. pose_estimator.py
# 9. game_state.py
# 10. event_logic.py  — the core Phase 1 work
# 11. data_logger.py
# 12. scoreboard.py   — Phase 1.5
# 13. main.py         — wire everything together
