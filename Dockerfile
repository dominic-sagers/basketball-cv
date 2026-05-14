# Basketball CV - Real-time stat tracking system
# CUDA-enabled Docker image for YOLOv11 + ByteTrack inference

# Use NVIDIA CUDA base image with Python 3.11
# Using Ubuntu 22.04 with CUDA 12.4 for RTX 4080 Super + Python 3.13 compatibility
FROM nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# PyTorch with CUDA 12.1 support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Verify CUDA is available
RUN python3.11 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Download YOLOv11 base weights on build (cached layer)
RUN python3.11 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

# Expose ports (if needed for future web dashboard or API)
# EXPOSE 8000

# Default command - runs camera test
CMD ["python3.11", "src/camera_test.py"]
