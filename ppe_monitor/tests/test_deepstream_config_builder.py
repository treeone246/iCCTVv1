"""Tests for DeepStream config builder."""

from pathlib import Path

from app.deepstream.config_builder import build_deepstream_settings


def test_build_settings_uses_source_uris_and_pads_camera_ids(tmp_path: Path) -> None:
    cfg = {
        "video": {"source": "videos/fallback.mp4"},
        "deepstream": {
            "enabled": True,
            "source_uris": ["rtsp://cam/a", "rtsp://cam/b"],
            "camera_ids": ["cam_a"],
            "batch_size": 1,
            "gie_config": "configs/deepstream/config_infer_primary_yolo.txt",
            "tracker_config": "configs/deepstream/config_tracker_NvDCF.yml",
            "engine_path": "models/best2.engine",
            "camera_id": "rig_cam",
        },
        "models": {"ppe": "models/best2.onnx"},
    }
    out = build_deepstream_settings(cfg, project_root=tmp_path)
    assert out.source_uris == ["rtsp://cam/a", "rtsp://cam/b"]
    assert out.camera_ids == ["cam_a", "rig_cam_1"]
    assert out.batch_size == 2


def test_build_settings_falls_back_to_single_source(tmp_path: Path) -> None:
    cfg = {
        "video": {"source": "videos/simulation.mp4"},
        "deepstream": {
            "enabled": True,
            "source_uri": "",
            "gie_config": "configs/deepstream/config_infer_primary_yolo.txt",
            "tracker_config": "configs/deepstream/config_tracker_NvDCF.yml",
            "engine_path": "models/best2.engine",
        },
        "models": {"ppe": "models/best2.onnx"},
    }
    out = build_deepstream_settings(cfg, project_root=tmp_path)
    assert out.source_uris == ["videos/simulation.mp4"]
    assert out.camera_ids == ["rig_floor_cam_01"]
    assert out.batch_size == 1
