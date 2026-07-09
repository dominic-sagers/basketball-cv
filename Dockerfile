# basketball-cv archive server
# No GPU, no CV inference — runs on any machine (NUC production box).
# For V1 real-time inference, a separate CUDA image will be needed.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY src/ src/
COPY config.yaml .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/server.py"]
