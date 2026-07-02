"""
tests/test_server.py — unit + integration tests for src/server.py.

Covers:
    Session lifecycle   create → join → chunk upload → end → DONE
    State guards        join/upload/end rejected when session is wrong state
    CRC32 rejection     bad checksum → 422
    Persistence         session.json written on every mutation
    Two-camera e2e      both cameras upload chunks; concat produces two output files
"""
from __future__ import annotations

import io
import json
import shutil
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.server import (
    ARCHIVING,
    DONE,
    FAILED,
    RECORDING,
    GameSession,
    ServerConfig,
    _crc32_hex,
    create_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crc32_of_bytes(data: bytes) -> str:
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return f"{crc:08x}"


class _SyncThread:
    """Replaces src.server's threading.Thread so archive runs synchronously inside tests.

    Patched via patch("src.server.threading.Thread") — scoped only to server.py's
    import so TestClient's internal threads are unaffected.
    """

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None, **kw):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self) -> None:
        if self._target:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path) -> ServerConfig:
    return ServerConfig(
        host="127.0.0.1",
        port=8000,
        chunks_root=tmp_path / "chunks",
        games_root=tmp_path / "games",
    )


@pytest.fixture
def client(cfg) -> TestClient:
    app = create_app(cfg)
    return TestClient(app)


@pytest.fixture
def tiny_mp4(make_mp4) -> bytes:
    """Real MP4 bytes (5 frames) usable as chunk payload."""
    return make_mp4(n_frames=5, name="tiny.mp4").read_bytes()


def _upload_chunk(
    client: TestClient,
    run_id: str,
    camera_id: str,
    chunk_id: str,
    video_bytes: bytes,
    bad_checksum: bool = False,
) -> dict:
    checksum = "deadbeef" if bad_checksum else _crc32_of_bytes(video_bytes)
    meta = json.dumps({
        "chunk_id": chunk_id,
        "run_id": run_id,
        "camera_id": camera_id,
        "expected_frame_count": 5,
        "checksum_value": checksum,
    })
    resp = client.post(
        "/api/v1/chunks/upload",
        data={"metadata": meta, "checksum": checksum},
        files={"video": (f"{chunk_id}.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    return resp


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionCreate:
    def test_creates_with_run_id(self, client):
        resp = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"})
        assert resp.status_code == 201
        body = resp.json()
        assert "run_id" in body
        assert body["state"] == RECORDING

    def test_missing_camera_id_400(self, client):
        resp = client.post("/api/v1/sessions", json={"team": "A"})
        assert resp.status_code == 400

    def test_missing_team_400(self, client):
        resp = client.post("/api/v1/sessions", json={"camera_id": "cam_a"})
        assert resp.status_code == 400

    def test_session_json_written(self, client, cfg):
        resp = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"})
        run_id = resp.json()["run_id"]
        session_file = cfg.games_root / run_id / "session.json"
        assert session_file.exists()
        data = json.loads(session_file.read_text())
        assert data["run_id"] == run_id
        assert data["state"] == RECORDING
        assert "cam_a" in data["cameras"]


class TestSessionJoin:
    def _create(self, client) -> str:
        return client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]

    def test_join_registers_second_camera(self, client):
        run_id = self._create(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        assert resp.status_code == 200
        body = resp.json()
        assert "cam_a" in body["cameras"]
        assert "cam_b" in body["cameras"]

    def test_duplicate_camera_400(self, client):
        run_id = self._create(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_a", "team": "A"})
        assert resp.status_code == 400

    def test_unknown_session_404(self, client):
        resp = client.post("/api/v1/sessions/no-such-id/join", json={"camera_id": "cam_b", "team": "B"})
        assert resp.status_code == 404

    def test_missing_camera_id_400(self, client):
        run_id = self._create(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"team": "B"})
        assert resp.status_code == 400


class TestSessionGet:
    def test_recording_returns_202(self, client):
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        resp = client.get(f"/api/v1/sessions/{run_id}")
        assert resp.status_code == 202
        assert resp.json()["state"] == RECORDING

    def test_unknown_returns_404(self, client):
        assert client.get("/api/v1/sessions/ghost").status_code == 404


# ---------------------------------------------------------------------------
# Chunk upload
# ---------------------------------------------------------------------------

class TestChunkUpload:
    def _session(self, client) -> str:
        return client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]

    def test_valid_chunk_accepted(self, client, tiny_mp4):
        run_id = self._session(client)
        resp = _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "received"
        assert body["run_id"] == run_id
        assert body["camera_id"] == "cam_a"

    def test_chunk_file_written_to_correct_path(self, client, cfg, tiny_mp4):
        run_id = self._session(client)
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        expected = cfg.chunks_root / run_id / "cam_a" / "cam_a_chunk_0000.mp4"
        assert expected.exists()
        assert expected.stat().st_size == len(tiny_mp4)

    def test_bad_checksum_rejected_422(self, client, tiny_mp4):
        run_id = self._session(client)
        resp = _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4, bad_checksum=True)
        assert resp.status_code == 422

    def test_bad_checksum_file_deleted(self, client, cfg, tiny_mp4):
        run_id = self._session(client)
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4, bad_checksum=True)
        # File should be cleaned up on mismatch
        assert not (cfg.chunks_root / run_id / "cam_a" / "cam_a_chunk_0000.mp4").exists()

    def test_unregistered_camera_rejected(self, client, tiny_mp4):
        run_id = self._session(client)
        resp = _upload_chunk(client, run_id, "cam_b", "cam_b_chunk_0000", tiny_mp4)
        assert resp.status_code == 400

    def test_unknown_session_rejected(self, client, tiny_mp4):
        resp = _upload_chunk(client, "no-session", "cam_a", "cam_a_chunk_0000", tiny_mp4)
        assert resp.status_code == 404

    def test_chunk_tracked_in_session(self, client, cfg, tiny_mp4):
        run_id = self._session(client)
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        data = json.loads((cfg.games_root / run_id / "session.json").read_text())
        assert len(data["_chunks"]["cam_a"]) == 1

    def test_missing_run_id_in_metadata(self, client, tiny_mp4):
        meta = json.dumps({"chunk_id": "x", "camera_id": "cam_a"})
        resp = client.post(
            "/api/v1/chunks/upload",
            data={"metadata": meta, "checksum": ""},
            files={"video": ("x.mp4", io.BytesIO(tiny_mp4), "video/mp4")},
        )
        assert resp.status_code == 400

    def test_path_traversal_rejected(self, client, tiny_mp4):
        run_id = self._session(client)
        resp = _upload_chunk(client, run_id, "cam_a", "../evil", tiny_mp4)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# State guards
# ---------------------------------------------------------------------------

class TestStateGuards:
    def _ended_session(self, client) -> str:
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        with patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"), \
             patch("src.server.threading.Thread", _SyncThread):
            client.post(f"/api/v1/sessions/{run_id}/end")
        return run_id

    def test_end_nonrecording_session_400(self, client):
        run_id = self._ended_session(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/end")
        assert resp.status_code == 400

    def test_join_done_session_400(self, client):
        run_id = self._ended_session(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        assert resp.status_code == 400

    def test_upload_to_done_session_400(self, client, tiny_mp4):
        run_id = self._ended_session(client)
        resp = _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# End-to-end: two cameras, real concat (skipped if ffmpeg not installed)
# ---------------------------------------------------------------------------

pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    and not Path("/usr/bin/ffmpeg").exists()
    and not Path(Path.home() / ".cache/ripcheck/ffstatic/ffmpeg").exists(),
    reason="ffmpeg not installed",
)


@pytestmark_ffmpeg
class TestTwoCameraE2E:
    def test_two_cameras_produce_two_output_files(self, client, cfg, make_mp4):
        # Create session (cam_a)
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]

        # Join (cam_b)
        client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})

        # Upload 3 chunks per camera
        for i in range(3):
            for cam in ("cam_a", "cam_b"):
                video = make_mp4(n_frames=10, name=f"{cam}_chunk_{i:04d}.mp4").read_bytes()
                resp = _upload_chunk(client, run_id, cam, f"{cam}_chunk_{i:04d}", video)
                assert resp.status_code == 202, resp.text

        # End session — runs archive synchronously via _SyncThread
        with patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"), \
             patch("src.server.threading.Thread", _SyncThread):
            resp = client.post(f"/api/v1/sessions/{run_id}/end")
        assert resp.status_code == 202

        # Both output files should exist
        out_dir = cfg.games_root / run_id
        assert (out_dir / "game_cam_a_raw.mp4").exists()
        assert (out_dir / "game_cam_b_raw.mp4").exists()

        # Session should be DONE
        status = client.get(f"/api/v1/sessions/{run_id}")
        assert status.json()["state"] == DONE

    def test_session_json_reflects_done_state(self, client, cfg, make_mp4):
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        video = make_mp4(n_frames=5, name="chunk.mp4").read_bytes()
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", video)

        with patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"), \
             patch("src.server.threading.Thread", _SyncThread):
            client.post(f"/api/v1/sessions/{run_id}/end")

        data = json.loads((cfg.games_root / run_id / "session.json").read_text())
        assert data["state"] == DONE
        assert data["ended_at"] is not None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthz:
    def test_healthz_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
