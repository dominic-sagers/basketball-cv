# Tech Stack

## Core dependencies

### Python 3.11+ (3.13 tested ✓)
Required for performance improvements and type hint support used throughout.
Python 3.13.1 is the active version — note: use CUDA 12.4 PyTorch wheels (cu121 has no 3.13 wheels).

### Ultralytics YOLOv11
```
pip install ultralytics
```
- Handles detection, ByteTrack tracking, and pose estimation in one package
- CUDA-accelerated out of the box with PyTorch
- Built-in multi-object tracking: `model.track(source, tracker="bytetrack.yaml")`
- Model variants: use `yolo11m.pt` as the starting point (good speed/accuracy balance on 4080 Super)
- Basketball-specific fine-tuned weights in `store/weights/basketball-ft.pt`

### OpenCV
```
pip install opencv-python
```
- Camera capture (VideoCapture), frame preprocessing, RTSP stream reading
- **Important**: do not install `opencv-python-headless` alongside `opencv-python` — they conflict and the headless build wins, breaking `cv2.imshow`. If Roboflow (or another package) installs it automatically, uninstall it: `pip uninstall opencv-python-headless`

### PyTorch + CUDA
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
- CUDA 12.4 build — required for Python 3.13 (cu121 has no 3.13 wheels)
- Verify with: `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`

### PyQt6
```
pip install PyQt6
```
- Desktop application framework for `src/app.py`
- Provides layout engine, thread-safe signals (pipeline → UI), and proper widget toolkit
- Used for: live video panel, score management, log output, start/stop control

### ffmpeg (system install)
- Required for chunk concatenation in stream-buffer mode
- **Windows**: `winget install Gyan.FFmpeg` — auto-discovered in the winget install path if not on PATH
- **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora)
- **macOS**: `brew install ffmpeg`
- The pipeline checks PATH first, then falls back to platform-specific install locations

### Pandas + openpyxl
```
pip install pandas openpyxl
```
- Phase 2 stat export to Excel
- Install now to avoid dependency pain later

### SAM2 + YOLOv8-face (face blur — post-processing)
```
pip install sam2 ultralytics huggingface_hub
```
- **SAM2** (`facebook/sam2.1-hiera-large`) — pixel-precise face mask propagation across video chunks; auto-downloads from HuggingFace, not gated
- **YOLOv8-face** (`arnabdhar/YOLOv8-Face-Detection`) — detects face bounding boxes on keyframes; weights auto-downloaded from HuggingFace
- Used by `src/face_blur.py` and `src/blur_footage.py` for post-game privacy blur
- Not needed at runtime during a game — only for post-processing recorded footage
- Note: `sam2` package requires Linux or WSL2 for the Triton kernel; on Windows run inside Docker or WSL2

### Pygame (Phase 1.5 overlay — superseded by PyQt6 app)
```
pip install pygame
```
- Originally planned for scoreboard overlay; `src/app.py` (PyQt6) now serves this role
- Still a dependency in `requirements.txt` for potential secondary overlay use

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

### Roboflow
```
pip install roboflow
```
- Access to basketball-specific pretrained datasets for fine-tuning
- See `docs/models.md` for the full dataset list with URLs and download instructions
- Only needed if you fine-tune; not required at runtime
- **Note**: Roboflow installs `opencv-python-headless` as a dependency — uninstall it after

### DVC
```
pip install "dvc[ssh]"
```
- Data Version Control — tracks large files (weights, dataset, footage) separately from git
- SSH remote over Tailscale — see `docs/dvc-setup.md` for setup and access
- `store.dvc` committed to git; actual data pulled with `dvc pull`

### IP Webcam (Android app — not a Python dep)
- Streams phone camera as HTTP MJPEG or RTSP over the network
- Used as live camera source during games
- Connect via Tailscale: `http://<phone-tailscale-ip>:8080/video`
- Recommended settings: 720p, 25 FPS, H.264, quality 50–60%

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
print(torch.cuda.mem_get_info())       # should show ~16GB total
# Hardware: i7-14700K, RTX 4080 Super 16GB, 32GB RAM, Windows 11
```

### OpenCV conflict check
```python
import cv2
print(cv2.__version__)
# verify cv2.imshow exists (headless build won't have it)
print(hasattr(cv2, 'imshow'))  # must be True
```

### Inference speed benchmark
```python
from ultralytics import YOLO
import time
model = YOLO("store/weights/basketball-ft.pt")
model.predict("store/footage/your_clip.mp4", device=0)  # warm up
start = time.time()
for _ in range(100):
    model.predict("store/footage/your_clip.mp4", device=0, verbose=False)
print(f"{100 / (time.time() - start):.1f} FPS")
```
Target: ≥30 FPS on 720p with yolo11m at full player count.

---

## Project setup steps

```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install all other dependencies
pip install -r requirements.txt

# 4. Fix OpenCV conflict if Roboflow was installed
pip uninstall opencv-python-headless -y

# 5. Pull data — weights, dataset, footage (requires Tailscale + DVC remote access)
# See docs/dvc-setup.md for access setup
dvc pull

# 6. Verify CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 7. Run on a recorded file
python src/pipeline_test.py --file store/footage/your_clip.mp4

# 8. Launch the desktop app (requires phone running IP Webcam + Tailscale connected)
python src/app.py --rtsp http://<phone-tailscale-ip>:8080/video --chunk-seconds 5

# 9. (Optional) Post-process recorded footage to blur faces
python src/blur_footage.py store/output/game.mp4 --face-imgsz 1920
```

---

## config.yaml structure (reference)

The actual `config.yaml` at the project root is the authoritative reference.
Key top-level sections:

```yaml
sources:
  - name: "basket_1"
    type: "file"          # "file" | "rtsp" | "usb"
    path: "store/footage/basket_1.mp4"
    loop: true
    team: "A"

model:
  weights: "store/weights/basketball-ft.pt"   # fine-tuned; or "yolo11m.pt" for COCO baseline
  device: 0               # GPU index; "cpu" for testing without GPU
  confidence: 0.4
  iou_threshold: 0.5
  input_size: 960
  class_map:
    0: "Ball"
    1: "Ball_in_Basket"
    2: "Player"
    3: "Basket"
    4: "Player_Shooting"

training:
  dataset: "store/dataset/basketball-srfkd/data.yaml"
  base_model: "yolo11m.pt"
  epochs: 50
  imgsz: 960
  batch: 8
  device: 0
  output_name: "basketball-ft"

tracking:
  tracker: "bytetrack"
  track_high_thresh: 0.25
  track_low_thresh: 0.1
  track_buffer: 30

event_logic:
  shot_cooldown_frames: 45   # debounce window after a detected basket

output:
  log_dir: "store/output/"
  scoreboard_display: 1
```
