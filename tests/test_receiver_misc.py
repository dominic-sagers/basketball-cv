"""Tests for ChunkStatus.to_dict (Kotlin contract), StatusCache, ReceiverConfig."""
from __future__ import annotations

from src.http_chunk_receiver import ChunkStatus, ReceiverConfig, StatusCache


# --- ChunkStatus.to_dict --------------------------------------------------

def test_to_dict_pending_is_minimal():
    d = ChunkStatus("c1", "pending", received_ms=1000).to_dict()
    assert d == {"chunk_id": "c1", "status": "pending"}


def test_to_dict_healthy_shape():
    s = ChunkStatus(
        "c1", "healthy", received_ms=1000,
        processing_started_ms=2000, expected_frame_count=25, actual_frame_count=25,
    )
    d = s.to_dict()
    assert d["status"] == "healthy"
    assert d["processing_started_ms"] == 2000
    assert d["validation"]["frame_count_match"] is True
    assert d["validation"]["checksum_valid"] is True
    assert "expected" not in d  # unhealthy-only field


def test_to_dict_healthy_defaults_processing_started_to_received():
    d = ChunkStatus("c1", "healthy", received_ms=1000).to_dict()
    assert d["processing_started_ms"] == 1000


def test_to_dict_unhealthy_shape():
    s = ChunkStatus(
        "c1", "unhealthy", received_ms=1000,
        expected_frame_count=30, actual_frame_count=10,
        issues=["frame count off", "checksum bad"], retry_deadline_ms=5000,
    )
    d = s.to_dict()
    assert d["status"] == "unhealthy"
    assert d["expected"] == 30
    assert d["received"] == 10
    assert d["retry_deadline_ms"] == 5000
    assert d["message"] == "frame count off; checksum bad"


def test_to_dict_unhealthy_default_retry_deadline():
    d = ChunkStatus("c1", "unhealthy", received_ms=1000, issues=["x"]).to_dict()
    assert d["retry_deadline_ms"] == 1000 + 120_000


# --- StatusCache ----------------------------------------------------------

def test_status_cache_put_get():
    cache = StatusCache()
    status = ChunkStatus("a", "pending", received_ms=0)
    cache.put(status)
    assert cache.get("a") is status
    assert cache.get("missing") is None


def test_status_cache_update_patches_fields():
    cache = StatusCache()
    cache.put(ChunkStatus("a", "pending", received_ms=0))
    cache.update("a", status="healthy", actual_frame_count=25)
    got = cache.get("a")
    assert got.status == "healthy"
    assert got.actual_frame_count == 25


def test_status_cache_update_missing_is_noop():
    cache = StatusCache()
    cache.update("ghost", status="healthy")  # must not raise
    assert cache.get("ghost") is None


# --- ReceiverConfig.from_yaml --------------------------------------------

def test_from_yaml_reads_receiver_block(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "receiver:\n"
        "  host: 1.2.3.4\n"
        "  port: 9999\n"
        "  storage_root: /tmp/x\n"
        "  frame_count_tolerance: 0.1\n"
        "  healthcheck_workers: 4\n"
    )
    cfg = ReceiverConfig.from_yaml(str(cfg_file))
    assert cfg.host == "1.2.3.4"
    assert cfg.port == 9999
    assert cfg.frame_count_tolerance == 0.1
    assert cfg.healthcheck_workers == 4
    assert str(cfg.storage_root) == "/tmp/x"


def test_from_yaml_missing_file_uses_defaults(tmp_path):
    cfg = ReceiverConfig.from_yaml(str(tmp_path / "does-not-exist.yaml"))
    assert cfg.port == 8000
    assert cfg.host == "0.0.0.0"


def test_from_yaml_missing_block_uses_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("model: {}\n")
    cfg = ReceiverConfig.from_yaml(str(cfg_file))
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
