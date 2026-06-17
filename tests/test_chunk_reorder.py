"""
Tests for ChunkReceiverSource's reorder buffer.

These exercise the ordering logic directly without starting uvicorn — the
release function is pure and runs against the in-memory output queue. This is
the logic that previously stranded earlier chunks behind a later one, so the
ordering guarantees are pinned down here.
"""
from __future__ import annotations

import queue
from pathlib import Path

from src.http_chunk_receiver import ChunkReceiverSource, ChunkStatus


def _healthy(chunk_id: str) -> ChunkStatus:
    return ChunkStatus(chunk_id=chunk_id, status="healthy", received_ms=0)


def _drain(src: ChunkReceiverSource) -> list[str]:
    out: list[str] = []
    while True:
        try:
            out.append(src._out_queue.get_nowait())
        except queue.Empty:
            break
    return out


def _seqs(paths: list[str]) -> list[int]:
    return [int(Path(p).stem.rsplit("_", 1)[-1]) for p in paths]


def _backdate_and_release(src: ChunkReceiverSource) -> None:
    """Mark every buffered chunk as held long ago, then run the release pass."""
    with src._lock:
        src._pending = {seq: (path, 0.0) for seq, (path, _) in src._pending.items()}
        src._release_ready_locked()


def test_parse_seq_numeric_suffix(receiver_cfg):
    src = ChunkReceiverSource(receiver_cfg)
    assert src._parse_seq("camera_0_chunk_7") == 7
    assert src._parse_seq("camera_0_chunk_42") == 42


def test_parse_seq_fallback_is_monotonic(receiver_cfg):
    src = ChunkReceiverSource(receiver_cfg)
    a = src._parse_seq("no-digits-here")
    b = src._parse_seq("still-none")
    assert b == a + 1


def test_contiguous_chunks_emit_immediately(receiver_cfg):
    # hold_seconds is huge: only the contiguous fast-path can emit.
    src = ChunkReceiverSource(receiver_cfg, hold_seconds=1000)
    src._last_emitted = 0
    src._on_validated(_healthy("camera_0_chunk_1"))
    src._on_validated(_healthy("camera_0_chunk_2"))
    assert _seqs(_drain(src)) == [1, 2]


def test_out_of_order_uploads_emit_in_order(receiver_cfg):
    src = ChunkReceiverSource(receiver_cfg, hold_seconds=1000)
    for cid in ["camera_0_chunk_2", "camera_0_chunk_0", "camera_0_chunk_1", "camera_0_chunk_3"]:
        src._on_validated(_healthy(cid))
    # Nothing emits yet: first emission needs the hold timer, which hasn't elapsed.
    assert _drain(src) == []
    _backdate_and_release(src)
    assert _seqs(_drain(src)) == [0, 1, 2, 3]


def test_gap_skipped_after_hold_elapses(receiver_cfg):
    src = ChunkReceiverSource(receiver_cfg, hold_seconds=1000)
    src._last_emitted = 0
    src._on_validated(_healthy("camera_0_chunk_2"))  # gap at seq 1
    assert _drain(src) == []                          # still waiting for 1
    _backdate_and_release(src)
    assert _seqs(_drain(src)) == [2]                  # 1 presumed lost, skipped
    assert src._last_emitted == 2


def test_buffer_full_forces_progress(receiver_cfg):
    src = ChunkReceiverSource(receiver_cfg, hold_seconds=1000, max_reorder_buffer=3)
    src._last_emitted = 0
    for seq in (2, 3, 4, 5):  # gap at 1; once >3 buffered, the gap is forced
        src._on_validated(_healthy(f"camera_0_chunk_{seq}"))
    out = _seqs(_drain(src))
    assert out == [2, 3, 4, 5]
    assert src._last_emitted == 5


def test_unhealthy_chunks_are_not_emitted(receiver_cfg):
    src = ChunkReceiverSource(receiver_cfg, hold_seconds=1000)
    src._on_validated(ChunkStatus(chunk_id="camera_0_chunk_0", status="unhealthy", received_ms=0))
    assert _drain(src) == []
    assert src._pending == {}


def test_stop_flushes_remaining_in_order_with_sentinel(receiver_cfg):
    src = ChunkReceiverSource(receiver_cfg, hold_seconds=1000)
    for cid in ["camera_0_chunk_1", "camera_0_chunk_0"]:
        src._on_validated(_healthy(cid))
    assert _drain(src) == []          # nothing released yet
    src.stop()                        # server/threads never started — guarded
    drained = _drain(src)
    assert drained[-1] is None        # sentinel terminates the queue
    assert _seqs(drained[:-1]) == [0, 1]
