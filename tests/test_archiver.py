"""
tests/test_archiver.py — unit tests for src/archiver.py.

Covers:
    _concat()          — ffmpeg chunk concat (skipped if ffmpeg not installed)
    _archive_session() — post-session orchestration: concat + cleanup + dvc_push

End-to-end receive-loop tests (ChunkReceiverSource + live HTTP uploads) are
deferred to tests/test_archiver_e2e.py and require a network-capable environment.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest

from src.archiver import _archive_session, _concat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_have_ffmpeg = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not _have_ffmpeg, reason="ffmpeg not installed")


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ---------------------------------------------------------------------------
# _concat
# ---------------------------------------------------------------------------

class TestConcat:
    @needs_ffmpeg
    def test_two_chunks_produces_output(self, make_mp4, tmp_path):
        a = make_mp4(n_frames=15, name="a.mp4")
        b = make_mp4(n_frames=10, name="b.mp4")
        out = tmp_path / "out.mp4"

        assert _concat([str(a), str(b)], str(out))
        assert out.exists()

    @needs_ffmpeg
    def test_frame_count_is_sum_of_inputs(self, make_mp4, tmp_path):
        chunks = [make_mp4(n_frames=10, name=f"c{i}.mp4") for i in range(3)]
        out = tmp_path / "out.mp4"

        _concat([str(c) for c in chunks], str(out))

        assert _frame_count(out) == pytest.approx(30, abs=3)

    def test_empty_list_returns_false(self, tmp_path):
        assert _concat([], str(tmp_path / "out.mp4")) is False
        assert not (tmp_path / "out.mp4").exists()

    def test_no_ffmpeg_returns_false(self, make_mp4, tmp_path):
        a = make_mp4(n_frames=10, name="a.mp4")
        out = tmp_path / "out.mp4"

        with patch("src.archiver._find_ffmpeg", return_value=None):
            assert _concat([str(a)], str(out)) is False
        assert not out.exists()

    @needs_ffmpeg
    def test_bad_input_path_returns_false(self, tmp_path):
        out = tmp_path / "out.mp4"
        assert _concat([str(tmp_path / "nonexistent.mp4")], str(out)) is False
        assert not out.exists()


# ---------------------------------------------------------------------------
# _archive_session — patches _concat and dvc_push so no ffmpeg/DVC needed
# ---------------------------------------------------------------------------

class TestArchiveSession:
    """
    All tests in this class patch _concat (so ffmpeg is never needed) and
    dvc_push (so no DVC remote is required). Behaviour under test is the
    directory creation, chunk cleanup, and control flow.
    """

    def _chunks_in(self, cam_dir: Path, names: list[str]) -> list[str]:
        """Write empty placeholder files and return their string paths."""
        paths = []
        for name in names:
            p = cam_dir / name
            p.write_bytes(b"placeholder")
            paths.append(str(p))
        return paths

    def test_output_file_created_at_correct_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cam_dir = tmp_path / "cam"
        cam_dir.mkdir()
        chunks = self._chunks_in(cam_dir, ["chunk_0000.mp4"])

        with patch("src.archiver._concat", return_value=True) as mock_concat, \
             patch("src.archiver.dvc_push", return_value=True):
            _archive_session(chunks, "2026-06-25_200000", "A", str(cam_dir), "rtsp")

        out_arg = mock_concat.call_args[0][1]
        assert Path(out_arg).name == "game_camA_raw.mp4"
        assert "2026-06-25_200000" in out_arg

    def test_output_directory_is_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cam_dir = tmp_path / "cam"
        cam_dir.mkdir()
        chunks = self._chunks_in(cam_dir, ["chunk_0000.mp4"])

        with patch("src.archiver._concat", return_value=True), \
             patch("src.archiver.dvc_push", return_value=True):
            _archive_session(chunks, "run_abc", "B", str(cam_dir), "rtsp")

        assert (tmp_path / "store" / "output" / "run_abc").is_dir()

    def test_rtsp_chunks_deleted_after_concat(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cam_dir = tmp_path / "cam"
        cam_dir.mkdir()
        # StreamChunkRecorder names files chunk_NNNN.mp4
        chunks = self._chunks_in(cam_dir, ["chunk_0000.mp4", "chunk_0001.mp4"])

        with patch("src.archiver._concat", return_value=True), \
             patch("src.archiver.dvc_push", return_value=True):
            _archive_session(chunks, "run1", "A", str(cam_dir), "rtsp")

        assert not any(cam_dir.glob("chunk_*.mp4"))

    def test_http_chunks_deleted_with_json_sidecars(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        validated = tmp_path / "store" / "chunks" / "validated"
        validated.mkdir(parents=True)
        cam_dir = tmp_path / "cam"
        cam_dir.mkdir()

        # Receiver writes .mp4 + .json sidecar for each chunk
        chunk = validated / "phone_chunk_0.mp4"
        chunk.write_bytes(b"placeholder")
        sidecar = chunk.with_suffix(".json")
        sidecar.write_text('{"chunk_id": "phone_chunk_0"}')

        with patch("src.archiver._concat", return_value=True), \
             patch("src.archiver.dvc_push", return_value=True):
            _archive_session([str(chunk)], "run1", "B", str(cam_dir), "http_chunks")

        assert not chunk.exists()
        assert not sidecar.exists()

    def test_dvc_push_called_on_success(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cam_dir = tmp_path / "cam"
        cam_dir.mkdir()
        chunks = self._chunks_in(cam_dir, ["chunk_0000.mp4"])

        with patch("src.archiver._concat", return_value=True), \
             patch("src.archiver.dvc_push", return_value=True) as mock_push:
            _archive_session(chunks, "run1", "A", str(cam_dir), "rtsp")

        mock_push.assert_called_once()

    def test_no_chunks_skips_concat_and_dvc_push(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with patch("src.archiver._concat") as mock_concat, \
             patch("src.archiver.dvc_push") as mock_push:
            _archive_session([], "run1", "A", str(tmp_path), "rtsp")

        mock_concat.assert_not_called()
        mock_push.assert_not_called()

    def test_concat_failure_skips_dvc_push(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cam_dir = tmp_path / "cam"
        cam_dir.mkdir()
        chunks = self._chunks_in(cam_dir, ["chunk_0000.mp4"])

        with patch("src.archiver._concat", return_value=False), \
             patch("src.archiver.dvc_push") as mock_push:
            _archive_session(chunks, "run1", "A", str(cam_dir), "rtsp")

        mock_push.assert_not_called()

    def test_concat_failure_leaves_source_chunks_intact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cam_dir = tmp_path / "cam"
        cam_dir.mkdir()
        chunks = self._chunks_in(cam_dir, ["chunk_0000.mp4", "chunk_0001.mp4"])

        with patch("src.archiver._concat", return_value=False), \
             patch("src.archiver.dvc_push"):
            _archive_session(chunks, "run1", "A", str(cam_dir), "rtsp")

        # Chunks should NOT be deleted if concat failed
        assert len(list(cam_dir.glob("chunk_*.mp4"))) == 2

    def test_camera_team_reflected_in_output_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        for team in ("A", "B"):
            cam_dir = tmp_path / f"cam{team}"
            cam_dir.mkdir()
            chunks = self._chunks_in(cam_dir, ["chunk_0000.mp4"])

            with patch("src.archiver._concat", return_value=True) as mock_concat, \
                 patch("src.archiver.dvc_push"):
                _archive_session(chunks, "run1", team, str(cam_dir), "rtsp")

            out_arg = mock_concat.call_args[0][1]
            assert f"game_cam{team}_raw.mp4" in out_arg
