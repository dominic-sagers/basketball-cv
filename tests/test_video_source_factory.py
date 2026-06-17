"""Tests for VideoSourceFactory.from_config — type dispatch and validation."""
from __future__ import annotations

import pytest

from src.video_source import (
    FileVideoSource,
    RTSPVideoSource,
    USBCameraSource,
    VideoSourceFactory,
)


def test_file_source(tmp_path):
    src = VideoSourceFactory.from_config(
        {"type": "file", "name": "basket_1", "path": str(tmp_path / "x.mp4"), "loop": True}
    )
    assert isinstance(src, FileVideoSource)
    assert src.name == "basket_1"


def test_file_source_requires_path():
    with pytest.raises(ValueError, match="path"):
        VideoSourceFactory.from_config({"type": "file", "name": "basket_1"})


def test_rtsp_source():
    src = VideoSourceFactory.from_config(
        {"type": "rtsp", "name": "cam", "url": "rtsp://10.0.0.1/stream"}
    )
    assert isinstance(src, RTSPVideoSource)


def test_rtsp_source_requires_url():
    with pytest.raises(ValueError, match="url"):
        VideoSourceFactory.from_config({"type": "rtsp", "name": "cam"})


def test_usb_source():
    src = VideoSourceFactory.from_config(
        {"type": "usb", "name": "cam", "index": 0, "resolution": [1280, 720], "fps": 30}
    )
    assert isinstance(src, USBCameraSource)


def test_usb_source_requires_index():
    with pytest.raises(ValueError, match="index"):
        VideoSourceFactory.from_config({"type": "usb", "name": "cam"})


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown source type"):
        VideoSourceFactory.from_config({"type": "carrier_pigeon", "name": "x"})


def test_type_is_case_insensitive(tmp_path):
    src = VideoSourceFactory.from_config(
        {"type": "FILE", "name": "c", "path": str(tmp_path / "x.mp4")}
    )
    assert isinstance(src, FileVideoSource)
