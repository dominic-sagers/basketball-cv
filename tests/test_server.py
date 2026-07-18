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
import time
import zlib
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.server import (
    ARCHIVING,
    CANCELLED,
    DONE,
    DRAINING,
    FAILED,
    RECORDING,
    WAITING,
    GameSession,
    ServerConfig,
    SessionRegistry,
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
    # drain_quiet_seconds=0 makes every camera without a reported captured_count
    # count as drained immediately, so tests that end a session synchronously
    # (via _SyncThread) archive right away like they did pre-drain.
    return ServerConfig(
        host="127.0.0.1",
        port=8000,
        chunks_root=tmp_path / "chunks",
        games_root=tmp_path / "games",
        drain_quiet_seconds=0.0,
        drain_timeout_seconds=5.0,
        drain_poll_seconds=0.01,
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
    def test_creates_in_waiting_state(self, client):
        # WAITING, not RECORDING: no camera has confirmed positioning yet.
        resp = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"})
        assert resp.status_code == 201
        body = resp.json()
        assert "run_id" in body
        assert body["state"] == WAITING

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
        assert data["state"] == WAITING
        assert "cam_a" in data["cameras"]


class TestSessionStart:
    def _create(self, client) -> str:
        return client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]

    def test_start_transitions_to_recording(self, client):
        run_id = self._create(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        assert resp.status_code == 200
        assert resp.json()["state"] == RECORDING

    def test_start_is_idempotent_for_second_device(self, client):
        # Creator and joiner each confirm positioning independently — whichever calls
        # /start second must not error.
        run_id = self._create(client)
        client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        resp = client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_b"})
        assert resp.status_code == 200
        assert resp.json()["state"] == RECORDING

    def test_start_unknown_session_404(self, client):
        resp = client.post("/api/v1/sessions/no-such-id/start", json={"camera_id": "cam_a"})
        assert resp.status_code == 404

    def test_start_after_done_400(self, client):
        run_id = self._create(client)
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        with patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"), \
             patch("src.server.threading.Thread", _SyncThread):
            client.post(f"/api/v1/sessions/{run_id}/end")
        resp = client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        assert resp.status_code == 400


class TestSessionJoin:
    def _create(self, client) -> str:
        return client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]

    def test_join_registers_second_camera_while_waiting(self, client):
        # The second device must be able to join and start positioning in parallel,
        # before the creator has confirmed its own positioning (still WAITING).
        run_id = self._create(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        assert resp.status_code == 200
        body = resp.json()
        assert "cam_a" in body["cameras"]
        assert "cam_b" in body["cameras"]

    def test_join_after_creator_started_recording(self, client):
        run_id = self._create(client)
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        assert resp.status_code == 200

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
    def test_waiting_returns_202(self, client):
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        resp = client.get(f"/api/v1/sessions/{run_id}")
        assert resp.status_code == 202
        assert resp.json()["state"] == WAITING

    def test_recording_returns_202(self, client):
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        resp = client.get(f"/api/v1/sessions/{run_id}")
        assert resp.status_code == 202
        assert resp.json()["state"] == RECORDING

    def test_unknown_returns_404(self, client):
        assert client.get("/api/v1/sessions/ghost").status_code == 404


class TestSessionList:
    def test_empty_when_no_sessions(self, client):
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created_sessions(self, client):
        with patch("src.server.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2020, 1, 1)
            run_id_a = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        with patch("src.server.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2020, 1, 2)
            run_id_b = client.post("/api/v1/sessions", json={"camera_id": "cam_b", "team": "B"}).json()["run_id"]
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        run_ids = {s["run_id"] for s in resp.json()}
        assert run_ids == {run_id_a, run_id_b}

    def test_sorted_newest_first(self, client):
        # created_at has second resolution — pin distinct timestamps via datetime.now()
        # rather than relying on real wall-clock gaps between the two creation calls.
        with patch("src.server.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2020, 1, 1)
            run_id_older = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        with patch("src.server.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2020, 1, 2)
            run_id_newer = client.post("/api/v1/sessions", json={"camera_id": "cam_b", "team": "B"}).json()["run_id"]

        resp = client.get("/api/v1/sessions")
        run_ids_in_order = [s["run_id"] for s in resp.json()]
        assert run_ids_in_order[0] == run_id_newer
        assert run_ids_in_order[-1] == run_id_older

    def test_limit_respected(self, client):
        # run_id has second resolution, so distinct timestamps are needed to avoid
        # same-second sessions colliding on the same run_id (a separate, pre-existing
        # behavior, not something this test is meant to exercise).
        for i in range(3):
            with patch("src.server.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2020, 1, 1 + i)
                client.post("/api/v1/sessions", json={"camera_id": f"cam_{i}", "team": "A"})
        resp = client.get("/api/v1/sessions", params={"limit": 2})
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# Chunk upload
# ---------------------------------------------------------------------------

class TestChunkUpload:
    def _session(self, client) -> str:
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        return run_id

    def test_upload_while_waiting_rejected(self, client, tiny_mp4):
        # Uploads must not be accepted before a device confirms recording via /start —
        # mirrors the app not capturing anything until Continue on the Positioning screen.
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        resp = _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        assert resp.status_code == 400

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
# Device metrics
# ---------------------------------------------------------------------------

class TestMetricsUpload:
    def _session(self, client) -> str:
        # Deliberately doesn't call /start, unlike TestChunkUpload._session — metrics
        # upload isn't state-gated (test_valid_metrics_accepted below proves it's
        # accepted in WAITING, where chunk upload would be rejected with 400).
        return client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]

    def _upload(self, client, run_id: str, camera_id: str, body: bytes):
        return client.post(
            f"/api/v1/sessions/{run_id}/metrics",
            data={"camera_id": camera_id},
            files={"file": ("metrics.jsonl", io.BytesIO(body), "application/x-ndjson")},
        )

    def test_valid_metrics_accepted(self, client):
        run_id = self._session(client)
        body = b'{"ts_ms": 1, "battery_pct": 90}\n{"ts_ms": 2, "battery_pct": 89}\n'
        resp = self._upload(client, run_id, "cam_a", body)
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "received"
        assert data["run_id"] == run_id
        assert data["camera_id"] == "cam_a"
        assert data["bytes"] == len(body)

    def test_metrics_file_written_to_correct_path(self, client, cfg):
        run_id = self._session(client)
        body = b'{"ts_ms": 1}\n'
        self._upload(client, run_id, "cam_a", body)
        expected = cfg.games_root / run_id / "metrics_cam_a.jsonl"
        assert expected.exists()
        assert expected.read_bytes() == body

    def test_unknown_session_rejected(self, client):
        resp = self._upload(client, "no-session", "cam_a", b"{}")
        assert resp.status_code == 404

    def test_blank_camera_id_rejected(self, client):
        # A literal empty string gets dropped entirely by multipart encoding (→ FastAPI's
        # own 422 for a missing required Form field) — whitespace-only reaches our
        # explicit .strip() check instead, which is what this test is for.
        run_id = self._session(client)
        resp = client.post(
            f"/api/v1/sessions/{run_id}/metrics",
            data={"camera_id": "   "},
            files={"file": ("metrics.jsonl", io.BytesIO(b"{}"), "application/x-ndjson")},
        )
        assert resp.status_code == 400

    def test_path_traversal_rejected(self, client):
        run_id = self._session(client)
        resp = self._upload(client, run_id, "../evil", b"{}")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# State guards
# ---------------------------------------------------------------------------

class TestStateGuards:
    def _ended_session(self, client) -> str:
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        with patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"), \
             patch("src.server.threading.Thread", _SyncThread):
            client.post(f"/api/v1/sessions/{run_id}/end")
        return run_id

    def test_end_already_ended_returns_current_state(self, client):
        # A joiner ending after the creator did must not 400 — that wedged its
        # UI on "Archiving footage…" (android#21). It gets the state instead.
        run_id = self._ended_session(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/end")
        assert resp.status_code == 202
        assert resp.json()["state"] == DONE

    def test_join_done_session_400(self, client):
        run_id = self._ended_session(client)
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        assert resp.status_code == 400

    def test_upload_to_done_session_400(self, client, tiny_mp4):
        run_id = self._ended_session(client)
        resp = _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        assert resp.status_code == 400

    def test_end_while_waiting_cancels(self, client):
        # Backing out of Positioning before Continue — no camera ever started, so this
        # goes straight to CANCELLED rather than draining/archiving anything.
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        resp = client.post(f"/api/v1/sessions/{run_id}/end")
        assert resp.status_code == 202
        assert resp.json()["state"] == CANCELLED

    def test_end_cancelled_is_idempotent(self, client):
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/end")
        resp = client.post(f"/api/v1/sessions/{run_id}/end")
        assert resp.status_code == 202
        assert resp.json()["state"] == CANCELLED

    def test_get_cancelled_returns_200(self, client):
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/end")
        resp = client.get(f"/api/v1/sessions/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["state"] == CANCELLED

    def test_join_cancelled_session_400(self, client):
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/end")
        resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test-environment archiving (receiver.dvc_enabled: false — see config.test.yaml)
# ---------------------------------------------------------------------------

class TestArchiveDvcDisabled:
    @pytest.fixture
    def dvc_disabled_cfg(self, tmp_path) -> ServerConfig:
        return ServerConfig(
            host="127.0.0.1",
            port=8000,
            chunks_root=tmp_path / "chunks",
            games_root=tmp_path / "games",
            drain_quiet_seconds=0.0,
            drain_timeout_seconds=5.0,
            drain_poll_seconds=0.01,
            dvc_enabled=False,
        )

    @pytest.fixture
    def dvc_disabled_client(self, dvc_disabled_cfg) -> TestClient:
        return TestClient(create_app(dvc_disabled_cfg))

    def test_end_reaches_done_without_calling_dvc(self, dvc_disabled_client):
        client = dvc_disabled_client
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        with patch("src.server.dvc_add_local") as mock_add, \
             patch("src.server.dvc_push_background") as mock_push, \
             patch("src.server.threading.Thread", _SyncThread):
            resp = client.post(f"/api/v1/sessions/{run_id}/end")
        assert resp.status_code == 202
        assert resp.json()["state"] == DONE
        mock_add.assert_not_called()
        mock_push.assert_not_called()


# ---------------------------------------------------------------------------
# End-of-session drain (basketball-cv#15)
# ---------------------------------------------------------------------------

def _wait_for_state(client: TestClient, run_id: str, want: str, timeout: float = 5.0) -> dict:
    """Poll GET /sessions/{run_id} until the session reaches `want` (or fail)."""
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/sessions/{run_id}").json()
        if body["state"] == want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"Session {run_id} never reached {want}; last state: {body.get('state')}")


class TestDrain:
    """Drain runs on real background threads here (no _SyncThread), with dvc and
    concat patched out, so uploads can land while the session is DRAINING."""

    @pytest.fixture
    def drain_cfg(self, tmp_path) -> ServerConfig:
        # Long quiet period: only reported captured_counts (or the timeout)
        # can finish the drain, so DRAINING is observable mid-test.
        return ServerConfig(
            host="127.0.0.1",
            port=8000,
            chunks_root=tmp_path / "chunks",
            games_root=tmp_path / "games",
            drain_quiet_seconds=30.0,
            drain_timeout_seconds=5.0,
            drain_poll_seconds=0.02,
        )

    @pytest.fixture
    def drain_client(self, drain_cfg) -> TestClient:
        return TestClient(create_app(drain_cfg))

    def test_end_drains_backlog_before_archiving(self, drain_client, tiny_mp4):
        """The two-cam e2e failure mode: creator ends while the joiner still has
        chunks queued — those uploads must be accepted and make the archive."""
        client = drain_client
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_b", "team": "B"})
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        _upload_chunk(client, run_id, "cam_b", "cam_b_chunk_0000", tiny_mp4)

        concat_calls: list[list[str]] = []

        def fake_concat(chunks: list[str], output: str) -> bool:
            concat_calls.append(list(chunks))
            return True

        with patch("src.server._concat", side_effect=fake_concat), \
             patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"):
            # Creator ends, reporting totals: cam_a done at 1, cam_b captured 2
            # but only uploaded 1 so far.
            resp = client.post(
                f"/api/v1/sessions/{run_id}/end",
                json={"camera_id": "cam_a", "captured_count": 1},
            )
            assert resp.status_code == 202
            assert resp.json()["state"] == DRAINING

            # Joiner's own end reports its total — legal while DRAINING.
            resp = client.post(
                f"/api/v1/sessions/{run_id}/end",
                json={"camera_id": "cam_b", "captured_count": 2},
            )
            assert resp.status_code == 202

            # The straggler chunk is still accepted while DRAINING…
            resp = _upload_chunk(client, run_id, "cam_b", "cam_b_chunk_0001", tiny_mp4)
            assert resp.status_code == 202

            # …and once counts match, the session archives.
            _wait_for_state(client, run_id, DONE)

        cam_b_chunks = next(c for c in concat_calls if any("cam_b" in p for p in c))
        assert len(cam_b_chunks) == 2

    def test_upload_after_archiving_still_rejected(self, drain_client, tiny_mp4):
        client = drain_client
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        with patch("src.server._concat", return_value=True), \
             patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"):
            client.post(f"/api/v1/sessions/{run_id}/end", json={"camera_id": "cam_a", "captured_count": 1})
            _wait_for_state(client, run_id, DONE)
        resp = _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0001", tiny_mp4)
        assert resp.status_code == 400

    def test_join_while_draining_rejected(self, drain_client, tiny_mp4):
        client = drain_client
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        with patch("src.server._concat", return_value=True), \
             patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"):
            # captured_count of 1 with nothing uploaded keeps it DRAINING.
            client.post(f"/api/v1/sessions/{run_id}/end", json={"camera_id": "cam_a", "captured_count": 1})
            resp = client.post(f"/api/v1/sessions/{run_id}/join", json={"camera_id": "cam_c", "team": "B"})
            assert resp.status_code == 400
            # Drain the session so its non-daemon watcher thread exits promptly.
            _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
            _wait_for_state(client, run_id, DONE)

    def test_drain_timeout_archives_partial(self, tmp_path, tiny_mp4):
        cfg = ServerConfig(
            host="127.0.0.1", port=8000,
            chunks_root=tmp_path / "chunks", games_root=tmp_path / "games",
            drain_quiet_seconds=30.0, drain_timeout_seconds=0.2, drain_poll_seconds=0.02,
        )
        client = TestClient(create_app(cfg))
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        with patch("src.server._concat", return_value=True), \
             patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"):
            # Reported total never arrives — the timeout must still archive.
            client.post(f"/api/v1/sessions/{run_id}/end", json={"camera_id": "cam_a", "captured_count": 5})
            _wait_for_state(client, run_id, DONE)

    def test_quiet_fallback_drains_camera_that_never_reported(self, tmp_path, tiny_mp4):
        cfg = ServerConfig(
            host="127.0.0.1", port=8000,
            chunks_root=tmp_path / "chunks", games_root=tmp_path / "games",
            drain_quiet_seconds=0.1, drain_timeout_seconds=5.0, drain_poll_seconds=0.02,
        )
        client = TestClient(create_app(cfg))
        run_id = client.post("/api/v1/sessions", json={"camera_id": "cam_a", "team": "A"}).json()["run_id"]
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
        _upload_chunk(client, run_id, "cam_a", "cam_a_chunk_0000", tiny_mp4)
        with patch("src.server._concat", return_value=True), \
             patch("src.server.dvc_add_local", return_value=True), \
             patch("src.server.dvc_push_background"):
            # Device died before reporting its captured_count (end arrived with no
            # body) — the quiet period must finish the drain rather than stalling
            # until the hard timeout.
            resp = client.post(f"/api/v1/sessions/{run_id}/end")
            assert resp.status_code == 202
            _wait_for_state(client, run_id, DONE)

    def test_restore_marks_draining_session_failed(self, tmp_path):
        games_root = tmp_path / "games"
        session = GameSession(
            run_id="2026-01-01_000000", state=DRAINING,
            created_at="2026-01-01T00:00:00",
        )
        session.persist(games_root)
        registry = SessionRegistry(games_root)
        restored = registry.get("2026-01-01_000000")
        assert restored is not None
        assert restored.state == FAILED

    def test_restore_marks_waiting_session_failed(self, tmp_path):
        # A device killed mid-positioning (never reached /start) needs recovery too,
        # same as one killed mid-recording — it shouldn't sit as WAITING forever.
        games_root = tmp_path / "games"
        session = GameSession(
            run_id="2026-01-01_000001", state=WAITING,
            created_at="2026-01-01T00:00:00",
        )
        session.persist(games_root)
        registry = SessionRegistry(games_root)
        restored = registry.get("2026-01-01_000001")
        assert restored is not None
        assert restored.state == FAILED


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
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})

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
        client.post(f"/api/v1/sessions/{run_id}/start", json={"camera_id": "cam_a"})
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
# Push status
# ---------------------------------------------------------------------------

class TestPushStatus:
    def test_reflects_current_status(self, client):
        fake = {"state": "PUSHING", "started_at": "2026-07-05T12:00:00", "finished_at": None, "error": None}
        with patch("src.server.get_push_status", return_value=fake):
            resp = client.get("/api/v1/push-status")
        assert resp.status_code == 200
        assert resp.json() == fake

    def test_never_run_by_default(self, client):
        fake = {"state": "NEVER_RUN", "started_at": None, "finished_at": None, "error": None}
        with patch("src.server.get_push_status", return_value=fake):
            resp = client.get("/api/v1/push-status")
        assert resp.json()["state"] == "NEVER_RUN"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthz:
    def test_healthz_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
