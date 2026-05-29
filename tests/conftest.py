"""Shared pytest fixtures for the basketball-cv backend tests."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.http_chunk_receiver import ReceiverConfig, _crc32_hex


@pytest.fixture
def make_mp4(tmp_path):
    """
    Factory that writes a tiny real MP4 and returns its path.

    Frames are solid-colour 64x48 so the file is valid and OpenCV can read a
    frame count back out of the container header.
    """
    def _make(n_frames: int = 25, fps: int = 30, name: str = "clip.mp4") -> Path:
        path = tmp_path / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
        assert writer.isOpened(), "OpenCV could not open an mp4v VideoWriter"
        for i in range(n_frames):
            frame = np.full((48, 64, 3), (i * 7) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        assert path.exists() and path.stat().st_size > 0
        return path

    return _make


@pytest.fixture
def receiver_cfg(tmp_path) -> ReceiverConfig:
    """A ReceiverConfig rooted in a temp dir with the storage subdirs created."""
    root = tmp_path / "store_chunks"
    for sub in ("received", "validated", "failed"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return ReceiverConfig(
        host="127.0.0.1",
        port=0,  # tests never bind a socket
        storage_root=root,
        frame_count_tolerance=0.05,
        healthcheck_workers=1,
    )


@pytest.fixture
def write_received(receiver_cfg, make_mp4):
    """
    Factory that drops a chunk's three files into received/ as the upload
    endpoint would, computing a correct checksum unless one is overridden.

    Returns the chunk_id.
    """
    def _write(
        chunk_id: str = "camera_0_chunk_0",
        n_frames: int = 25,
        fps: int = 30,
        expected_frame_count: int | None = None,
        checksum: str | None = None,
    ) -> str:
        received = receiver_cfg.storage_root / "received"
        mp4_src = make_mp4(n_frames=n_frames, fps=fps, name=f"{chunk_id}.mp4")
        mp4_dst = received / f"{chunk_id}.mp4"
        mp4_dst.write_bytes(mp4_src.read_bytes())

        real_crc = _crc32_hex(mp4_dst)
        crc = checksum if checksum is not None else real_crc
        header = {
            "chunk_id": chunk_id,
            "expected_frame_count": expected_frame_count if expected_frame_count is not None else n_frames,
            "fps": fps,
            "encoding": "mp4",
            "checksum_algorithm": "crc32",
            "checksum_value": crc,
        }
        (received / f"{chunk_id}.json").write_text(json.dumps(header))
        (received / f"{chunk_id}.crc32").write_text(crc)
        return chunk_id

    return _write
