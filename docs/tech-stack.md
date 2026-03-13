# Tech Stack

## Core dependencies

### Python 3.11+
Required for performance improvements and type hint support used throughout.

### Ultralytics YOLOv11
```
pip install ultralytics
```
- Handles detection, ByteTrack tracking, and pose estimation in one package
- CUDA-accelerated out of the box with PyTorch
- Built-in multi-object tracking: `model.track(source, tracker="bytetrack.yaml")`
- Model variants: use `yolo11m.pt` as the starting point (good speed/accuracy balance on 4080 Super)
- Basketball-specific fine-tuned weights available on Roboflow Universe

### OpenCV
```
pip install opencv-python
```
- Camera capture (VideoCapture)
- Frame preprocessing (resize, denoise, ROI crop)
- Court homography (perspectiveTransform)
- Optional: overlay rendering if not using Pygame

### PyTorch + CUDA
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
- CUDA 12.1 build for RTX 4080 Super
- Verify with: `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`

### Pandas + openpyxl
```
pip install pandas openpyxl
```
- Phase 2 stat export to Excel
- Install now to avoid dependency pain later

### Pygame (Phase 1.5 scoreboard)
```
pip install pygame
```
- Lightweight overlay for scoreboard rendering
- Runs in separate thread reading from GameState

---

## Optional / alternative dependencies

### AlphaPose (alternative to YOLOv11-pose)
- Better multi-person pose accuracy in crowded scenes
- Heavier to set up (separate install, CUDA build)
- Use if YOLOv11-pose accuracy is insufficient for block/action detection
- Repo: https://github.com/MVIG-SJTU/AlphaPose

### Roboflow
```
pip install roboflow
```
- Access to basketball-specific pretrained datasets for fine-tuning
- Key datasets to look at on Roboflow Universe:
  - "basketball-players" detection datasets
  - "basketball" ball detection
- Only needed if you fine-tune; not required at runtime

### OBS Studio (Phase 1.5, not a Python dep)
- For projecting the scoreboard overlay to a second display or projector
- Virtual camera output from Python → OBS → projector

---

## Hardware setup checklist

### CUDA verification
```python
import torch
print(torch.cuda.is_available())       # must be True
print(torch.cuda.get_device_name(0))   # RTX 4080 SUPER
print(torch.cuda.mem_get_info())       # should show ~16GB
```

### Camera test
```python
import cv2
cap = cv2.VideoCapture(0)  # try 0, 1, 2 for each camera
ret, frame = cap.read()
print(ret, frame.shape)    # True, (1080, 1920, 3) or similar
```

### Inference speed benchmark
```python
from ultralytics import YOLO
import time
model = YOLO("yolo11m.pt")
# warm up
model.predict("path/to/test_frame.jpg", device=0)
# benchmark
start = time.time()
for _ in range(100):
    model.predict("path/to/test_frame.jpg", device=0, verbose=False)
print(f"{100 / (time.time() - start):.1f} FPS")
```
Target: ≥60 FPS on a single 1080p frame with yolo11m.

---

## Project setup steps

```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install core deps
pip install ultralytics opencv-python pandas openpyxl pygame pyyaml

# 3. Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Download base model weights (cached locally after first run)
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

# 5. Verify CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 6. Run camera test
python src/camera_test.py

# 7. Run inference benchmark
python src/detection_test.py
```

---

## config.yaml structure (reference)

```yaml
cameras:
  - index: 0           # or "rtsp://..." for IP cameras
    name: "basket_1"
    resolution: [1920, 1080]
    fps: 30
  - index: 1
    name: "basket_2"
    resolution: [1920, 1080]
    fps: 30

model:
  weights: "weights/yolo11m-basketball.pt"  # fine-tuned, or yolo11m.pt for base
  device: 0            # GPU index (0 = first GPU)
  confidence: 0.4
  iou_threshold: 0.5
  input_size: 1280     # px — larger = more accurate, slower

tracking:
  tracker: "bytetrack"
  track_high_thresh: 0.25
  track_low_thresh: 0.1
  track_buffer: 30

court:
  width_ft: 94
  height_ft: 50
  homography_points: []  # set after calibration

event_logic:
  rebound_proximity_px: 80     # how close a player must be to ball at possession change
  shot_downward_frames: 3      # frames ball must travel downward through hoop zone
  assist_possession_window: 5  # seconds — look back window for last pass
  block_hand_ball_dist_px: 40  # max hand-to-ball distance to count as block

output:
  log_dir: "output/"
  scoreboard_display: 1        # monitor index for scoreboard
```
