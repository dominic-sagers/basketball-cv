"""
End-to-end test for ChunkReceiverSource — real uvicorn server, real HTTP, real MP4.

The FastAPI tests in test_upload_endpoint.py go through TestClient, which
bypasses uvicorn entirely. This file covers the path the gym-box run actually
takes: phone → HTTP POST → uvicorn-in-thread → healthcheck worker → on_validated
callback → reorder buffer → out_queue.
"""
from __future__ import annotations

import json
import socket
import time
import zlib
from pathlib import Path

import httpx
import pytest

from src.http_chunk_receiver import ChunkReceiverSource, ReceiverConfig


def _free_port() -> int:
    """Ask the OS for an unused port. Small race window but fine for tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_cfg(tmp_path) -> ReceiverConfig:
    root = tmp_path / "store_chunks"
    for sub in ("received", "validated", "failed"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return ReceiverConfig(
        host="127.0.0.1",
        port=_free_port(),
        storage_root=root,
        frame_count_tolerance=0.05,
        healthcheck_workers=1,
    )


def _post_chunk(url: str, chunk_id: str, data: bytes, n_frames: int = 25) -> httpx.Response:
    crc = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
    header = {
        "chunk_id": chunk_id,
        "expected_frame_count": n_frames,
        "fps": 30,
        "encoding": "mp4",
        "checksum_algorithm": "crc32",
        "checksum_value": crc,
    }
    return httpx.post(
        url,
        data={"metadata": json.dumps(header), "checksum": crc},
        files={"video": ("chunk.bin", data, "application/octet-stream")},
        timeout=5.0,
    )


def _drain_queue(src: ChunkReceiverSource, expected: int, timeout_s: float = 5.0) -> list[str]:
    """Pull at most `expected` paths off the out queue, blocking up to timeout_s total."""
    paths: list[str] = []
    deadline = time.monotonic() + timeout_s
    while len(paths) < expected and time.monotonic() < deadline:
        try:
            item = src._out_queue.get(timeout=0.1)
        except Exception:
            continue
        if item is None:
            break
        paths.append(item)
    return paths


def test_uploads_flow_through_to_out_queue_in_order(live_cfg, make_mp4):
    src = ChunkReceiverSource(live_cfg, hold_seconds=0.5)
    src.start()
    try:
        url = f"http://{live_cfg.host}:{live_cfg.port}/api/v1/chunks/upload"
        for seq in (0, 1, 2):
            data = make_mp4(n_frames=25, fps=30, name=f"clip_{seq}.mp4").read_bytes()
            r = _post_chunk(url, f"camera_0_chunk_{seq}", data)
            assert r.status_code == 202, r.text

        paths = _drain_queue(src, expected=3)
        seqs = [int(Path(p).stem.rsplit("_", 1)[-1]) for p in paths]
        assert seqs == [0, 1, 2]
        for p in paths:
            assert Path(p).exists(), f"validated chunk file is missing: {p}"
            assert Path(p).parent == live_cfg.storage_root / "validated"
    finally:
        src.stop()


def test_healthz_responds_over_real_socket(live_cfg):
    src = ChunkReceiverSource(live_cfg)
    src.start()
    try:
        r = httpx.get(f"http://{live_cfg.host}:{live_cfg.port}/healthz", timeout=5.0)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
    finally:
        src.stop()


def test_port_collision_raises_clear_error(live_cfg):
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind((live_cfg.host, live_cfg.port))
    try:
        src = ChunkReceiverSource(live_cfg)
        with pytest.raises(OSError, match="Cannot bind"):
            src.start()
    finally:
        blocker.close()


def test_stop_enqueues_sentinel_after_live_run(live_cfg):
    src = ChunkReceiverSource(live_cfg, hold_seconds=0.2)
    src.start()
    src.stop()
    items: list = []
    while True:
        try:
            items.append(src._out_queue.get_nowait())
        except Exception:
            break
    assert items[-1] is None
