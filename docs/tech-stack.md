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

### RF-DETR (alternative detection backbone)
```
pip install rfdetr
```
- Roboflow's real-time detection transformer (released March 2025, Apache 2.0 for N/S/M/L sizes)
- DINOv2 vision transformer backbone — handles player occlusion better than CNN-based YOLO
- Trade-off: ~20-25 FPS on T4 GPU at high accuracy; RTX 4080 Super is significantly faster
- Not yet integrated — evaluate against test footage if ByteTrack loses tracks during screens
- GitHub: https://github.com/roboflow/rf-detr
- See `docs/models.md` for detailed comparison vs. YOLOv11

### Roboflow
```
pip install roboflow
```
- Access to basketball-specific pretrained datasets for fine-tuning
- See `docs/models.md` for the full dataset list with URLs and download instructions
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

# 6. Verify sources and pipeline
python src/source_test.py --file test_footage/your_clip.mp4
python src/pipeline_test.py --file test_footage/your_clip.mp4
```

---

## config.yaml structure (reference)

The actual `config.yaml` at the project root is the authoritative reference.
Key top-level sections:

```yaml
sources:
  - name: "basket_1"
    type: "file"          # "file" | "rtsp" | "usb"
    path: "test_footage/basket_1.mp4"
    loop: true
    team: "A"

model:
  weights: "yolo11m.pt"   # or "weights/your_finetuned.pt"
  device: 0               # GPU index; "cpu" for testing without GPU
  confidence: 0.4
  iou_threshold: 0.5
  input_size: 1280
  class_map:
    0:  "person"
    32: "sports ball"     # extend with hoop class once fine-tuned

tracking:
  tracker: "bytetrack"
  track_high_thresh: 0.25
  track_low_thresh: 0.1
  track_buffer: 30

event_logic:
  rebound_proximity_px: 80
  shot_downward_frames: 3
  assist_possession_window: 5
  block_hand_ball_dist_px: 40

output:
  log_dir: "output/"
  scoreboard_display: 1
```
