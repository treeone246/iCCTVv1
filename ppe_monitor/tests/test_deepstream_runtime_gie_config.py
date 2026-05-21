"""Tests for runtime materialization of DeepStream nvinfer config paths."""

from pathlib import Path

from app.deepstream.config_builder import DeepStreamSettings
from app.deepstream.deepstream_pipeline import DeepStreamPipelineRunner


def _make_runner(tmp_path: Path, gie_text: str) -> tuple[DeepStreamPipelineRunner, Path]:
    cfg_dir = tmp_path / "configs" / "deepstream"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    gie = cfg_dir / "config_infer_primary_yolo.txt"
    gie.write_text(gie_text, encoding="utf-8")
    tracker = cfg_dir / "config_tracker_NvDCF.yml"
    tracker.write_text("BaseConfig:\n", encoding="utf-8")
    labels = cfg_dir / "labels_ppe.txt"
    labels.write_text("person\nhelmet\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "best2.engine").write_bytes(b"engine")
    (models / "best2.onnx").write_bytes(b"onnx")

    settings = DeepStreamSettings(
        project_root=tmp_path,
        enabled=True,
        source_uris=["file:///tmp/a.mp4"],
        camera_ids=["cam_a"],
        batch_size=1,
        width=1280,
        height=720,
        live_source=True,
        gie_config=gie,
        tracker_config=tracker,
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
        engine_path=models / "best2.engine",
        onnx_fallback_path=models / "best2.onnx",
        labels_fallback_path=labels,
    )
    runner = DeepStreamPipelineRunner(
        settings=settings,
        label_map={},
        person_classes={"person"},
        ppe_classes={"helmet"},
        alias_to_canonical={},
    )
    return runner, gie


def test_materialize_gie_config_rewrites_relative_paths(tmp_path: Path) -> None:
    runner, _ = _make_runner(
        tmp_path,
        gie_text=(
            "onnx-file=models/best2.onnx\n"
            "model-engine-file=models/best2.engine\n"
            "labelfile-path=configs/deepstream/labels_ppe.txt\n"
        ),
    )
    out = runner._materialize_gie_config()
    text = out.read_text(encoding="utf-8")
    assert f"model-engine-file={(tmp_path / 'models' / 'best2.engine').as_posix()}" in text
    assert f"onnx-file={(tmp_path / 'models' / 'best2.onnx').as_posix()}" in text
    assert f"labelfile-path={(tmp_path / 'configs' / 'deepstream' / 'labels_ppe.txt').as_posix()}" in text


def test_materialize_gie_config_appends_missing_keys(tmp_path: Path) -> None:
    runner, _ = _make_runner(
        tmp_path,
        gie_text=(
            "[property]\n"
            "batch-size=1\n"
        ),
    )
    out = runner._materialize_gie_config()
    text = out.read_text(encoding="utf-8")
    assert "model-engine-file=" in text
    assert "onnx-file=" in text
    assert "labelfile-path=" in text
