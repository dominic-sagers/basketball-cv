"""
tests/test_game_archive.py — unit tests for src/game_archive.py.

Covers:
    get_push_status() / dvc_push_background()   push state transitions
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import src.game_archive as game_archive
from src.game_archive import (
    PUSH_FAILED,
    PUSH_NEVER_RUN,
    PUSH_PENDING,
    PUSH_PUSHED,
    dvc_push_background,
    get_push_status,
)


def _reset_push_status() -> None:
    game_archive._push_status = game_archive.PushStatus()
    game_archive._push_proc = None


def _fake_proc(returncode: int, out: str = "") -> MagicMock:
    proc = MagicMock()
    proc.communicate.return_value = (out, None)
    proc.returncode = returncode
    proc.poll.return_value = None
    return proc


class TestPushStatus:
    def setup_method(self) -> None:
        _reset_push_status()

    def teardown_method(self) -> None:
        _reset_push_status()

    def test_never_run_before_any_push(self):
        status = get_push_status()
        assert status["state"] == PUSH_NEVER_RUN
        assert status["started_at"] is None
        assert status["error"] is None

    def test_success_sets_pushed(self):
        with patch("src.game_archive._find_dvc", return_value="dvc"), \
             patch("src.game_archive.subprocess.Popen", return_value=_fake_proc(0)):
            dvc_push_background()
        status = get_push_status()
        assert status["state"] == PUSH_PUSHED
        assert status["started_at"] is not None
        assert status["finished_at"] is not None
        assert status["error"] is None

    def test_failure_sets_failed_with_error(self):
        with patch("src.game_archive._find_dvc", return_value="dvc"), \
             patch("src.game_archive.subprocess.Popen", return_value=_fake_proc(1, "connection refused")):
            dvc_push_background()
        status = get_push_status()
        assert status["state"] == PUSH_FAILED
        assert "connection refused" in status["error"]

    def test_interrupted_sets_pending_not_failed(self):
        with patch("src.game_archive._find_dvc", return_value="dvc"), \
             patch("src.game_archive.subprocess.Popen", return_value=_fake_proc(-15)):
            dvc_push_background()
        status = get_push_status()
        assert status["state"] == PUSH_PENDING
        assert status["error"] is None

    def test_missing_dvc_leaves_status_unchanged(self):
        with patch("src.game_archive._find_dvc", return_value=None):
            dvc_push_background()
        assert get_push_status()["state"] == PUSH_NEVER_RUN
