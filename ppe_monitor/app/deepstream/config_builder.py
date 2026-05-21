"""DeepStream backend config parsing and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ..runtime_backend import resolve_project_path


@dataclass
class DeepStreamSettings:
    enabled: bool
    source_uris: list[str]
    camera_ids: list[str]
    batch_size: int
    width: int
    height: int
    live_source: bool
    gie_config: Path
    tracker_config: Path
    tracker_ll_lib_file: str
    output_metadata_topic: str
    enable_osd: bool
    enable_display: bool
    drop_frame_interval: int
    target_fps: float
    emit_jpeg: bool
    jpeg_quality: int
    jpeg_interval: int
    appsink_max_buffers: int
    engine_path: Path


def build_deepstream_settings(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    source_override: Optional[str] = None,
    engine_override: Optional[str] = None,
    camera_id_override: Optional[str] = None,
) -> DeepStreamSettings:
    ds_cfg = dict(config.get("deepstream", {}) or {})
    video_cfg = dict(config.get("video", {}) or {})
    models_cfg = dict(config.get("models", {}) or {})

    cfg_source_uris = ds_cfg.get("source_uris", [])
    source_uris: list[str] = []
    if isinstance(cfg_source_uris, list):
        source_uris = [str(x).strip() for x in cfg_source_uris if str(x).strip()]
    if source_override:
        source_uris = [str(source_override).strip()]
    if not source_uris:
        single_source = str(ds_cfg.get("source_uri") or video_cfg.get("source") or "").strip()
        source_uris = [single_source] if single_source else ["videos/simulationTest2.mp4"]

    cfg_camera_ids = ds_cfg.get("camera_ids", [])
    base_camera_id = str(ds_cfg.get("camera_id", "rig_floor_cam_01")).strip() or "rig_floor_cam_01"
    camera_ids: list[str] = []
    if isinstance(cfg_camera_ids, list):
        camera_ids = [str(x).strip() for x in cfg_camera_ids if str(x).strip()]
    if camera_id_override:
        if len(source_uris) == 1:
            camera_ids = [str(camera_id_override).strip()]
        elif not camera_ids:
            camera_ids = [str(camera_id_override).strip()]
    while len(camera_ids) < len(source_uris):
        if len(source_uris) == 1:
            camera_ids.append(base_camera_id)
        else:
            camera_ids.append(f"{base_camera_id}_{len(camera_ids)}")
    if len(camera_ids) > len(source_uris):
        camera_ids = camera_ids[: len(source_uris)]

    engine_path_value = str(engine_override or ds_cfg.get("engine_path", "models/best2.engine"))
    if not engine_path_value:
        engine_path_value = "models/best2.engine"
    engine_path = resolve_project_path(project_root, engine_path_value)

    # Keep existing model path as fallback reference in docs/logging only.
    _ = models_cfg.get("ppe", "")

    gie_config = resolve_project_path(
        project_root,
        str(ds_cfg.get("gie_config", "configs/deepstream/config_infer_primary_yolo.txt")),
    )
    tracker_config = resolve_project_path(
        project_root,
        str(ds_cfg.get("tracker_config", "configs/deepstream/config_tracker_NvDCF.yml")),
    )

    configured_batch = max(1, int(ds_cfg.get("batch_size", 1)))
    batch_size = max(configured_batch, len(source_uris))

    return DeepStreamSettings(
        enabled=bool(ds_cfg.get("enabled", False)),
        source_uris=source_uris,
        camera_ids=camera_ids,
        batch_size=batch_size,
        width=max(64, int(ds_cfg.get("width", 1280))),
        height=max(64, int(ds_cfg.get("height", 720))),
        live_source=bool(ds_cfg.get("live_source", True)),
        gie_config=gie_config,
        tracker_config=tracker_config,
        tracker_ll_lib_file=str(
            ds_cfg.get(
                "tracker_ll_lib_file",
                "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            )
        ),
        output_metadata_topic=str(ds_cfg.get("output_metadata_topic", "frame_payload")),
        enable_osd=bool(ds_cfg.get("enable_osd", False)),
        enable_display=bool(ds_cfg.get("enable_display", False)),
        drop_frame_interval=max(0, int(ds_cfg.get("drop_frame_interval", 0))),
        target_fps=max(1.0, float(ds_cfg.get("target_fps", 30))),
        emit_jpeg=bool(ds_cfg.get("emit_jpeg", True)),
        jpeg_quality=max(30, min(95, int(ds_cfg.get("jpeg_quality", 75)))),
        jpeg_interval=max(1, int(ds_cfg.get("jpeg_interval", 1))),
        appsink_max_buffers=max(1, int(ds_cfg.get("appsink_max_buffers", 4))),
        engine_path=engine_path,
    )
