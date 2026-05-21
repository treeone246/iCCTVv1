"""Tests for graceful DeepStream dependency failures."""

from pathlib import Path

import pytest

from app.deepstream.config_builder import DeepStreamSettings
from app.deepstream.deepstream_pipeline import DeepStreamPipelineRunner, DeepStreamUnavailableError


def test_deepstream_missing_dependency_error_points_to_setup_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error():
        raise ImportError("gi missing")

    monkeypatch.setattr(
        "app.deepstream.deepstream_pipeline.import_deepstream_modules",
        _raise_import_error,
    )
    settings = DeepStreamSettings(
        project_root=Path("/tmp"),
        enabled=True,
        source_uris=["file:///tmp/video.mp4"],
        camera_ids=["cam"],
        batch_size=1,
        width=1280,
        height=720,
        live_source=True,
        gie_config=Path("configs/deepstream/config_infer_primary_yolo.txt"),
        tracker_config=Path("configs/deepstream/config_tracker_NvDCF.yml"),
        tracker_ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        output_metadata_topic="frame_payload",
        enable_osd=False,
        enable_display=False,
        drop_frame_interval=0,
        target_fps=30.0,
        emit_jpeg=True,
        jpeg_quality=75,
        jpeg_interval=1,
        appsink_max_buffers=4,
        engine_path=Path("models/best2.engine"),
        onnx_fallback_path=Path("models/best2.onnx"),
        labels_fallback_path=Path("configs/deepstream/labels_ppe.txt"),
    )
    runner = DeepStreamPipelineRunner(
        settings=settings,
        label_map={},
        person_classes={"person"},
        ppe_classes={"helmet"},
        alias_to_canonical={},
    )
    with pytest.raises(DeepStreamUnavailableError) as exc:
        runner.start()
    assert "docs/deepstream_jetson_setup.md" in str(exc.value)
