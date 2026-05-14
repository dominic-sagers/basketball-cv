# Docker Setup Guide

## Overview

This project includes Docker support for running the basketball CV system in a containerized environment with full CUDA/GPU acceleration. The Docker setup is optimized for the RTX 4080 Super but works with any NVIDIA GPU that supports CUDA 12.1+.

---

## Prerequisites

### 1. NVIDIA Container Toolkit (Linux)

Docker GPU support requires the NVIDIA Container Toolkit:

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Windows with WSL2 + Docker Desktop

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Enable WSL2 backend in Docker Desktop settings
3. Ensure NVIDIA GPU drivers are installed on Windows host
4. GPU passthrough to WSL2 is automatic with recent drivers

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

### Build the image

```bash
docker-compose build
```

This will:
- Create a CUDA 12.1 + Python 3.11 environment
- Install all dependencies (ultralytics, opencv, pytorch, etc.)
- Download base YOLOv11m weights
- Verify CUDA availability

Build time: ~10-15 minutes on first run (layers are cached afterward).

### Run the container

```bash
# Start container in interactive mode
docker-compose up -d

# Attach to running container
docker-compose exec basketball-cv bash

# Or run a specific script directly
docker-compose run --rm basketball-cv python3.11 src/camera_test.py
```

### Test CUDA and inference

```bash
# From inside the container
python3.11 src/detection_test.py

# Expected output:
# GPU: NVIDIA GeForce RTX 4080 SUPER
# VRAM: 15.xGB free / 16.0GB total
# Results: 60+ FPS
```

---

## Volume Mounts

The `docker-compose.yml` includes these volume mounts:

| Host Path | Container Path | Purpose |
|---|---|---|
| `./src` | `/app/src` | Live code editing (development mode) |
| `./config.yaml` | `/app/config.yaml` | Configuration |
| `./output` | `/app/output` | Game logs and stats export |
| `./weights` | `/app/weights` | Custom fine-tuned model weights |

**For production:** Comment out the `./src` mount to bake code into the image.

---

## Camera Access

### Linux (USB cameras)

Uncomment these lines in `docker-compose.yml`:

```yaml
devices:
  - /dev/video0:/dev/video0
  - /dev/video1:/dev/video1
```

Adjust device indices based on `ls /dev/video*`.

### RTSP network cameras

No special configuration needed. Update `config.yaml` with RTSP URLs:

```yaml
cameras:
  - index: "rtsp://192.168.1.100:554/stream1"
    name: "basket_1"
```

### Windows + WSL2

USB camera passthrough to WSL2 is limited. Recommended approach:
- Use RTSP network cameras, OR
- Run the pipeline natively on Windows (not in Docker)

---

## GUI / OpenCV Windows

### Linux with X11

To display OpenCV windows (e.g., `camera_test.py`):

```bash
# Allow Docker to connect to X server
xhost +local:docker

# Add to docker-compose.yml:
environment:
  - DISPLAY=${DISPLAY}
volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix
```

### Headless mode (no GUI)

For production (game night), run without OpenCV display:
- Scoreboard renders to a Pygame window or file output
- Use OBS Studio on the host to capture and project

---

## Performance Tuning

### GPU memory allocation

By default, PyTorch allocates all available VRAM. To limit usage:

```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Multi-camera throughput

Each camera stream runs in its own thread. For 2-3 cameras at 30 FPS:
- Ensure `imgsz` in `config.yaml` is ≤1280 (default)
- Use `yolo11m.pt` or smaller (`yolo11s.pt`) if FPS drops below 30

Monitor GPU utilization inside the container:

```bash
watch -n 1 nvidia-smi
```

Target: ~80-90% GPU utilization during live inference.

---

## Dockerfile Breakdown

### Base image
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```
Includes CUDA 12.1 runtime and cuDNN 8 for optimized PyTorch operations.

### Python 3.11
```dockerfile
RUN apt-get install -y python3.11 python3.11-dev python3-pip
```
Required for type hints and performance improvements in this project.

### PyTorch CUDA build
```dockerfile
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Matches CUDA 12.1 runtime for RTX 4080 Super.

### Pre-download YOLOv11 weights
```dockerfile
RUN python3.11 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```
Caches base model weights in the image (~50MB). Fine-tuned weights should be mounted via `./weights` volume.

---

## Common Issues

### "CUDA not available" inside container

**Cause:** NVIDIA Container Toolkit not installed or Docker runtime not configured.

**Fix:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### "Could not open camera 0"

**Cause:** Camera device not passed through to container.

**Fix:**
- Linux: Add `devices:` section to `docker-compose.yml`
- Windows/WSL2: Use RTSP cameras or run natively
- Verify with: `ls /dev/video*` (on host and in container)

### Low FPS (<30) during inference

**Cause:** Wrong PyTorch build (CPU-only) or GPU not being used.

**Fix:**
```python
# Inside container
python3.11 -c "import torch; print(torch.cuda.is_available())"  # must be True
```

If False, rebuild image ensuring PyTorch CUDA install step succeeded.

### "Out of memory" error

**Cause:** VRAM exhausted (multi-camera or high `imgsz`).

**Fix:**
- Reduce `imgsz` in `config.yaml` (try 960 or 640)
- Use a smaller model (`yolo11s.pt` instead of `yolo11m.pt`)
- Process cameras sequentially instead of in parallel (slower but lower memory)

---

## Production Deployment

### 1. Bake code into image

Comment out the `./src` volume mount in `docker-compose.yml`:

```yaml
volumes:
  # - ./src:/app/src         # Remove for production
  - ./config.yaml:/app/config.yaml
  - ./output:/app/output
  - ./weights:/app/weights
```

Rebuild: `docker-compose build`

### 2. Auto-start on boot

```yaml
restart: always
```

And enable Docker service auto-start:
```bash
sudo systemctl enable docker
```

### 3. Run main pipeline

Update `docker-compose.yml`:

```yaml
command: python3.11 main.py
```

### 4. Monitor logs

```bash
docker-compose logs -f basketball-cv
```

---

## Development Workflow

### Live code editing

With `./src` mounted, edit files on host and restart the container:

```bash
docker-compose restart
```

No rebuild needed — code changes are reflected immediately.

### Adding dependencies

1. Update `requirements.txt`
2. Rebuild: `docker-compose build`
3. Restart: `docker-compose up -d`

### Debugging inside container

```bash
docker-compose exec basketball-cv bash
python3.11  # interactive shell
```

---

## Next Steps

1. **Verify setup:**
   ```bash
   docker-compose run --rm basketball-cv python3.11 src/detection_test.py
   ```
   Target: ≥60 FPS on synthetic frames.

2. **Test cameras:**
   ```bash
   docker-compose run --rm basketball-cv python3.11 src/camera_test.py
   ```

3. **Start building pipeline:**
   - Once Docker is working, proceed with implementing `src/tracker.py`, `src/ball_tracker.py`, etc.
   - See `docs/roadmap.md` for Phase 1 checklist

---

## References

- [NVIDIA Container Toolkit docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose GPU access](https://docs.docker.com/compose/gpu-support/)
- [Ultralytics YOLOv11 Docker guide](https://docs.ultralytics.com/guides/docker-quickstart/)
