"""
End-to-end HTTP contract tests for the receiver, driven through Starlette's
TestClient (no real socket). Covers the multipart upload, the background
validation handoff, and the status state machine — plus a regression for the
checksum-as-filename bug that made the Android client get 422s.
"""
from __future__ import annotations

import json
import queue as queue_mod
import time
import zlib

import pytest
from fastapi.testclient import TestClient

from src.http_chunk_receiver import HealthcheckWorker, StatusCache, create_app

UPLOAD = "/api/v1/chunks/upload"


def _crc(data: bytes) -> str:
    return f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"


@pytest.fixture
def client(receiver_cfg):
    cache = StatusCache()
    q: "queue_mod.Queue[str | None]" = queue_mod.Queue()
    worker = HealthcheckWorker(receiver_cfg, cache, q)
    app = create_app(receiver_cfg, cache, q, worker)
    # `with` runs the lifespan, which starts the healthcheck worker pool.
    with TestClient(app) as c:
        yield c


def _poll_status(client: TestClient, chunk_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/api/v1/chunks/{chunk_id}/status").json()
        if body["status"] != "pending":
            return body
        time.sleep(0.05)
    raise AssertionError(f"chunk {chunk_id} never left 'pending'")


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_validates_and_moves_to_validated(client, receiver_cfg, make_mp4):
    data = make_mp4(n_frames=25, fps=30).read_bytes()
    crc = _crc(data)
    header = {
        "chunk_id": "camera_0_chunk_0",
        "expected_frame_count": 25,
        "fps": 30,
        "encoding": "mp4",
        "checksum_algorithm": "crc32",
        "checksum_value": crc,
    }
    r = client.post(
        UPLOAD,
        data={"metadata": json.dumps(header), "checksum": crc},
        files={"video": ("chunk.bin", data, "application/octet-stream")},
    )
    assert r.status_code == 202
    assert r.json()["chunk_id"] == "camera_0_chunk_0"

    status = _poll_status(client, "camera_0_chunk_0")
    assert status["status"] == "healthy"
    assert (receiver_cfg.storage_root / "validated" / "camera_0_chunk_0.mp4").exists()


def test_upload_bad_checksum_lands_in_failed(client, receiver_cfg, make_mp4):
    data = make_mp4(n_frames=25).read_bytes()
    header = {"chunk_id": "camera_0_chunk_9", "expected_frame_count": 25, "checksum_value": "deadbeef"}
    r = client.post(
        UPLOAD,
        data={"metadata": json.dumps(header), "checksum": "deadbeef"},
        files={"video": ("chunk.bin", data, "application/octet-stream")},
    )
    assert r.status_code == 202
    status = _poll_status(client, "camera_0_chunk_9")
    assert status["status"] == "unhealthy"
    assert (receiver_cfg.storage_root / "failed" / "camera_0_chunk_9.mp4").exists()


def test_checksum_sent_as_file_part_is_rejected(client, make_mp4):
    # Regression: the Android client originally sent checksum with a filename,
    # so FastAPI parsed it as an UploadFile instead of a Form field -> 422.
    data = make_mp4().read_bytes()
    header = {"chunk_id": "camera_0_chunk_0", "expected_frame_count": 25}
    r = client.post(
        UPLOAD,
        data={"metadata": json.dumps(header)},
        files={
            "video": ("chunk.bin", data, "application/octet-stream"),
            "checksum": ("checksum.txt", b"deadbeef", "text/plain"),
        },
    )
    assert r.status_code == 422


def test_missing_chunk_id_returns_400(client, make_mp4):
    data = make_mp4().read_bytes()
    r = client.post(
        UPLOAD,
        data={"metadata": json.dumps({"expected_frame_count": 25}), "checksum": "abc"},
        files={"video": ("chunk.bin", data, "application/octet-stream")},
    )
    assert r.status_code == 400


def test_invalid_metadata_json_returns_400(client, make_mp4):
    data = make_mp4().read_bytes()
    r = client.post(
        UPLOAD,
        data={"metadata": "{not json", "checksum": "abc"},
        files={"video": ("chunk.bin", data, "application/octet-stream")},
    )
    assert r.status_code == 400


def test_status_unknown_chunk_returns_404(client):
    assert client.get("/api/v1/chunks/never-seen/status").status_code == 404
