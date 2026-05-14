# basketball-cv source modules
# See docs/architecture.md for full description of each module

# Suggested build order:
# 1. camera_test.py   — verify USB camera feeds
# 2. detection_test.py — verify CUDA + YOLOv11 FPS
# 3. video_source.py  — unified source abstraction (file/RTSP/USB) ✓
# 4. source_test.py   — verify all configured sources work ✓
# 5. detector.py      — YOLOv11 wrapper ✓
# 6. tracker.py       — ByteTrack wrapper ✓
# 7. visualizer.py    — detection/tracking overlay ✓
# 8. pipeline_test.py — source → detect/track → visualize ✓
# 9. preprocessor.py  — frame preprocessing
# 10. ball_tracker.py — ball + hoop logic
# 11. pose_estimator.py
# 12. game_state.py
# 13. event_logic.py  — the core Phase 1 work
# 14. data_logger.py
# 15. scoreboard.py   — Phase 1.5
# 16. main.py         — wire everything together
