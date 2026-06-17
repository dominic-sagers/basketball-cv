"""Tests for the chunk healthcheck: validate_chunk, _crc32_hex, _probe_mp4."""
from __future__ import annotations

import zlib

from src.http_chunk_receiver import _crc32_hex, _probe_mp4, validate_chunk


def test_crc32_hex_matches_zlib(tmp_path):
    data = b"basketball" * 1000
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    assert _crc32_hex(f) == f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"


def test_probe_mp4_reads_frame_count(make_mp4):
    path = make_mp4(n_frames=25, fps=30)
    frames, duration = _probe_mp4(path)
    assert abs(frames - 25) <= 1
    assert duration > 0


def test_probe_mp4_missing_file_returns_zero(tmp_path):
    frames, duration = _probe_mp4(tmp_path / "nope.mp4")
    assert (frames, duration) == (0, 0.0)


def test_validate_chunk_healthy(receiver_cfg, write_received):
    chunk_id = write_received(n_frames=25, expected_frame_count=25)
    status = validate_chunk(chunk_id, receiver_cfg.storage_root, receiver_cfg.frame_count_tolerance)
    assert status.status == "healthy"
    assert status.issues == []
    assert abs(status.actual_frame_count - 25) <= 1


def test_validate_chunk_checksum_mismatch(receiver_cfg, write_received):
    chunk_id = write_received(checksum="deadbeef")
    status = validate_chunk(chunk_id, receiver_cfg.storage_root, receiver_cfg.frame_count_tolerance)
    assert status.status == "unhealthy"
    assert any("hecksum" in issue for issue in status.issues)


def test_validate_chunk_frame_count_out_of_tolerance(receiver_cfg, write_received):
    chunk_id = write_received(n_frames=25, expected_frame_count=100)
    status = validate_chunk(chunk_id, receiver_cfg.storage_root, receiver_cfg.frame_count_tolerance)
    assert status.status == "unhealthy"
    assert any("rame count" in issue for issue in status.issues)
    assert status.expected_frame_count == 100


def test_validate_chunk_frame_count_within_tolerance(receiver_cfg, write_received):
    # 25 actual vs 26 expected = 4% diff, under the 5% tolerance
    chunk_id = write_received(n_frames=25, expected_frame_count=26)
    status = validate_chunk(chunk_id, receiver_cfg.storage_root, receiver_cfg.frame_count_tolerance)
    assert status.status == "healthy"


def test_validate_chunk_missing_video(receiver_cfg):
    status = validate_chunk("ghost_chunk", receiver_cfg.storage_root, receiver_cfg.frame_count_tolerance)
    assert status.status == "unhealthy"
    assert any("not found" in issue for issue in status.issues)
