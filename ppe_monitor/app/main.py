"""FastAPI app for live PPE monitoring stream and dashboard hosting."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .alert_feedback import AlertFeedbackStore
from .ai_behavior_agent.agent import BehaviorAgentService
from .ai_behavior_agent.schemas import empty_agent_output
from .deepstream import DeepStreamPipelineRunner, build_deepstream_settings
from .deepstream.compat import fill_unavailable_keypoints
from .jetson_exporter_bridge import JetsonExporterBridge
from .metrics_exporter import CONTENT_TYPE_LATEST, PrometheusMetricsExporter
from .performance_logger import PerformanceLogWriter
from .pipeline import MonitoringPipeline
from .ppe_detector import build_alias_index
from .runtime_backend import get_runtime_backend
from .runtime_acceleration import summarize_runtime_acceleration
from .schemas import AlertAcknowledgeRequest
from .startup_check import load_runtime_components
from .stream_guard import StreamClientGate
from .video_source import VideoSource


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_bool_query(value: str | None) -> bool | None:
    """Parse bool-like websocket query values.

    Returns:
    - True / False when recognized
    - None when value is unset or unrecognized
    """
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    backend = get_runtime_backend(config)
    runtime = load_runtime_components(config, PROJECT_ROOT, skip_ppe=(backend == "deepstream"))
    behavior_service = None

    pipeline = MonitoringPipeline(
        pose_tracker=runtime.pose_tracker,
        ppe_detector=runtime.ppe_detector,
        verifier=runtime.verifier,
        config=config,
    )
    video_source = None
    deepstream_runner = None
    deepstream_emit_jpeg = True
    deepstream_source_id = 0
    deepstream_camera_id = str(config.get("deepstream", {}).get("camera_id", "rig_floor_cam_01"))

    if backend == "deepstream":
        ds_settings = build_deepstream_settings(config, project_root=PROJECT_ROOT)
        if not ds_settings.enabled:
            raise RuntimeError(
                "runtime.backend is 'deepstream' but deepstream.enabled is false. "
                "Set deepstream.enabled: true in config.yaml."
            )
        alias_map = build_alias_index(config.get("ppe_label_aliases", {}))
        ds_cfg = config.get("deepstream", {}) or {}
        person_classes = set(str(x) for x in ds_cfg.get("person_classes", ["person"]))
        ppe_classes = set(str(x) for x in config.get("required_ppe", []))
        ppe_classes.update({"no_gloves", "no_boots", "harness"})
        deepstream_runner = DeepStreamPipelineRunner(
            settings=ds_settings,
            label_map=ds_cfg.get("label_map", {}),
            person_classes=person_classes,
            ppe_classes=ppe_classes,
            alias_to_canonical=alias_map,
        )
        deepstream_runner.start()
        deepstream_emit_jpeg = bool(ds_settings.emit_jpeg)
        source_cfg = {"source_uris": ds_settings.source_uris, "target_fps": ds_settings.target_fps}
        print(
            json.dumps(
                {
                    "event_type": "video_source_config",
                    "backend": backend,
                    "source_count": len(ds_settings.source_uris),
                    "source_uris": ds_settings.source_uris,
                    "camera_ids": ds_settings.camera_ids,
                    "target_fps": source_cfg["target_fps"],
                }
            )
        )
    else:
        source_cfg = config["video"]
        print(
            json.dumps(
                {
                    "event_type": "video_source_config",
                    "backend": backend,
                    "source": source_cfg["source"],
                    "source_type": type(source_cfg["source"]).__name__,
                    "target_fps": source_cfg["target_fps"],
                }
            )
        )
        video_source = VideoSource(
            source=source_cfg["source"],
            target_fps=float(source_cfg["target_fps"]),
            drop_grab_limit=int(source_cfg.get("drop_grab_limit", 3)),
        )
        video_source.open()

    app.state.config = config
    app.state.runtime_backend = backend
    app.state.runtime = runtime
    app.state.video_source = video_source
    app.state.deepstream_runner = deepstream_runner
    app.state.deepstream_emit_jpeg = deepstream_emit_jpeg
    app.state.deepstream_source_id = deepstream_source_id
    app.state.deepstream_camera_id = deepstream_camera_id
    app.state.pipeline = pipeline
    prom_cfg = config.get("prometheus", {}) or {}
    app.state.metrics_exporter = PrometheusMetricsExporter(enabled=bool(prom_cfg.get("enabled", True)))
    app.state.jetson_bridge = JetsonExporterBridge.from_app_config(config)
    app.state.performance_logger = PerformanceLogWriter(config)
    feedback_cfg = config.get("alert_feedback", {}) or {}
    feedback_path = _resolve_project_path(str(feedback_cfg.get("path", "outputs/alert_feedback.jsonl")))
    app.state.alert_feedback_store = AlertFeedbackStore(
        path=feedback_path,
        enabled=bool(feedback_cfg.get("enabled", True)),
        max_recent_records=int(feedback_cfg.get("max_recent_records", 10000)),
    )
    app.state.alert_feedback_camera_id = str(
        (config.get("event_stream", {}) or {}).get(
            "camera_id",
            (config.get("performance_logging", {}) or {}).get("camera_id", "cam_01"),
        )
    )
    dashboard_cfg = config.get("dashboard", {}) or {}
    max_stream_clients = max(1, int(dashboard_cfg.get("max_stream_clients", 1)))
    app.state.stream_jpeg_enabled = bool(dashboard_cfg.get("stream_jpeg_enabled", True))
    app.state.stream_gate = StreamClientGate(max_clients=max_stream_clients)

    behavior_cfg = config.get("behavior_agent", {}) or {}
    if bool(behavior_cfg.get("enabled", False)):
        behavior_service = BehaviorAgentService.from_config(config=config, project_root=PROJECT_ROOT)
        behavior_service.start()
        app.state.behavior_agent_service = behavior_service

    print(json.dumps({"event_type": "app_started"}))
    try:
        yield
    finally:
        if behavior_service is not None:
            behavior_service.stop()
        app.state.performance_logger.close()
        if app.state.deepstream_runner is not None:
            app.state.deepstream_runner.stop()
        pipeline.event_writer.close()
        pipeline.close()
        if video_source is not None:
            video_source.close()
        print(json.dumps({"event_type": "app_stopped"}))


app = FastAPI(title="PPE Monitoring", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "backend": getattr(app.state, "runtime_backend", "python"),
        "provider_info": app.state.runtime.provider_info,
    }


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    gate = getattr(app.state, "stream_gate", None)
    gate_acquired = False
    if gate is not None:
        gate_acquired = await gate.try_acquire()
        if not gate_acquired:
            snapshot = await gate.snapshot()
            await websocket.accept()
            await websocket.send_json(
                {
                    "event_type": "stream_rejected",
                    "reason": "max_stream_clients_reached",
                    "active_clients": snapshot["active_clients"],
                    "max_clients": snapshot["max_clients"],
                }
            )
            await websocket.close(code=1013, reason="stream_busy")
            return

    await websocket.accept()
    frame_id = 0
    backend = str(getattr(app.state, "runtime_backend", "python"))
    target_fps = float(app.state.config.get("video", {}).get("target_fps", 20))
    jpeg_interval_frames = max(1, int(app.state.config.get("dashboard", {}).get("jpeg_interval_frames", 1)))
    jpeg_enabled = bool(getattr(app.state, "stream_jpeg_enabled", True))
    jpeg_query = _parse_bool_query(websocket.query_params.get("jpeg"))
    if jpeg_query is not None:
        jpeg_enabled = jpeg_query
    if backend == "deepstream":
        target_fps = float(app.state.config.get("deepstream", {}).get("target_fps", target_fps))
    frame_interval = 1.0 / max(1.0, target_fps)
    last_tick = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            send_jpeg_for_frame = (frame_id % jpeg_interval_frames) == 0
            deepstream_emit_jpeg = bool(getattr(app.state, "deepstream_emit_jpeg", True))
            include_stream_jpeg = jpeg_enabled and send_jpeg_for_frame and (
                backend != "deepstream" or deepstream_emit_jpeg
            )
            if backend == "deepstream":
                bundle = await asyncio.to_thread(
                    app.state.deepstream_runner.read_bundle,
                    1.0,
                )
                if bundle is None:
                    await asyncio.sleep(frame_interval)
                    continue
                app.state.deepstream_source_id = int(bundle.source_id)
                app.state.deepstream_camera_id = str(bundle.camera_id)
                frame = bundle.frame_bgr
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                fill_unavailable_keypoints(bundle.adapted.persons)
                payload, jpeg = await asyncio.to_thread(
                    app.state.pipeline.process_frame,
                    frame,
                    frame_id,
                    ppe_detections_override=bundle.adapted.ppe_detections,
                    backend="deepstream",
                    include_stream_jpeg=include_stream_jpeg,
                )
                input_fps = float(bundle.input_fps)
                primary_latency_ms = float(bundle.primary_infer_latency_ms)
                tracker_latency_ms = float(bundle.tracker_latency_ms)
                end_to_end_ms = float(bundle.end_to_end_latency_ms)
            else:
                lag = max(0.0, now - last_tick - frame_interval)
                requested_drops = int(lag / frame_interval) if frame_interval > 0 else 0

                frame, dropped = await asyncio.to_thread(
                    app.state.video_source.read_latest,
                    requested_drops,
                )
                app.state.pipeline.increment_dropped_frames(dropped)

                if frame is None:
                    await asyncio.sleep(frame_interval)
                    continue

                payload, jpeg = await asyncio.to_thread(
                    app.state.pipeline.process_frame,
                    frame,
                    frame_id,
                    include_stream_jpeg=include_stream_jpeg,
                )
                input_fps = float(payload.metrics.fps)
                primary_latency_ms = 0.0
                tracker_latency_ms = 0.0
                end_to_end_ms = 0.0
            camera_for_log = (
                str(getattr(app.state, "deepstream_camera_id", ""))
                if backend == "deepstream"
                else None
            )
            per_camera_fps = bundle.camera_fps_map if backend == "deepstream" else None
            jetson_snapshot = app.state.jetson_bridge.read_snapshot()
            app.state.metrics_exporter.update(
                payload.metrics.model_dump(),
                event_stream_dropped=int(getattr(app.state.pipeline.event_writer, "dropped", 0)),
                jetson=jetson_snapshot,
            )
            app.state.performance_logger.emit(
                frame_id=frame_id,
                metrics=payload.metrics.model_dump(),
                jetson=jetson_snapshot,
                source="websocket_stream" if backend == "python" else "websocket_stream_deepstream",
                backend=backend,
                source_id=int(getattr(app.state, "deepstream_source_id", 0)),
                input_fps=input_fps,
                primary_infer_latency_ms=primary_latency_ms,
                tracker_latency_ms=tracker_latency_ms,
                end_to_end_latency_ms=end_to_end_ms,
                camera_id=camera_for_log,
                per_camera_fps=per_camera_fps,
                timestamp=payload.timestamp,
            )
            feedback_store = getattr(app.state, "alert_feedback_store", None)
            if feedback_store is not None:
                active_alert_dicts: list[dict] = []
                for alert in payload.active_alerts:
                    acknowledged = feedback_store.is_acknowledged(alert.alert_id)
                    alert.acknowledged = acknowledged
                    alert.feedback_label = feedback_store.feedback_label(alert.alert_id)
                    active_alert_dicts.append(alert.model_dump())
                feedback_camera_id = (
                    str(getattr(app.state, "deepstream_camera_id", "cam_01"))
                    if backend == "deepstream"
                    else str(getattr(app.state, "alert_feedback_camera_id", "cam_01"))
                )
                feedback_store.observe_active_alerts(
                    alerts=active_alert_dicts,
                    camera_id=feedback_camera_id,
                    observed_ts=float(payload.timestamp),
                )
            if include_stream_jpeg:
                await websocket.send_bytes(jpeg)
            await websocket.send_json(payload.model_dump())
            frame_id += 1

            elapsed = time.perf_counter() - now
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            last_tick = now
    except WebSocketDisconnect:
        return
    finally:
        if gate_acquired and gate is not None:
            await gate.release()


@app.get("/api/behavior-agent/latest")
async def behavior_agent_latest() -> dict:
    cfg = app.state.config.get("behavior_agent", {}) if hasattr(app.state, "config") else {}
    model = str(cfg.get("model", "qwen3:4b"))
    path = _resolve_project_path(str(cfg.get("output_dir", "outputs/behavior_agent"))) / "latest_behavior_insight.json"
    return _safe_load_json(path, default=empty_agent_output(model=model, time_window={"start": None, "end": None, "event_count": 0}))


@app.get("/api/behavior-agent/history")
async def behavior_agent_history() -> list[dict]:
    cfg = app.state.config.get("behavior_agent", {}) if hasattr(app.state, "config") else {}
    history_dir = _resolve_project_path(str(cfg.get("output_dir", "outputs/behavior_agent"))) / "history"
    if not history_dir.exists():
        return []
    out: list[dict] = []
    for path in sorted(history_dir.glob("behavior_agent_*.json"), reverse=True):
        doc = _safe_load_json(path, default=None)
        if isinstance(doc, dict):
            out.append(doc)
    return out


@app.get("/api/behavior-agent/memory")
async def behavior_agent_memory() -> dict:
    cfg = app.state.config.get("behavior_agent", {}) if hasattr(app.state, "config") else {}
    path = _resolve_project_path(str(cfg.get("memory_path", "outputs/person_behavior_memory.json")))
    doc = _safe_load_json(path, default={})
    return doc if isinstance(doc, dict) else {}


@app.get("/api/jetson/stats")
async def jetson_stats() -> dict:
    bridge = getattr(app.state, "jetson_bridge", None)
    if bridge is None:
        return {
            "enabled": False,
            "available": False,
            "error": "jetson_bridge_not_initialized",
        }
    snap = bridge.read_snapshot()
    return {
        "enabled": snap.enabled,
        "available": snap.available,
        "error": snap.error,
        "source_url": snap.source_url,
        "cpu_utilization_pct": snap.cpu_utilization_pct,
        "gpu_utilization_pct": snap.gpu_utilization_pct,
        "memory_utilization_pct": snap.memory_utilization_pct,
        "memory_used_mb": snap.memory_used_mb,
        "memory_total_mb": snap.memory_total_mb,
        "temperature_c": snap.temperature_c,
        "power_w": snap.power_w,
        "fan_pwm_pct": snap.fan_pwm_pct,
    }


@app.get("/api/runtime/acceleration")
async def runtime_acceleration() -> dict:
    runtime = getattr(app.state, "runtime", None)
    provider_info = getattr(runtime, "provider_info", {}) if runtime is not None else {}
    summary = summarize_runtime_acceleration(provider_info)
    summary["provider_info"] = provider_info
    return summary


@app.post("/api/alerts/acknowledge")
async def acknowledge_alert(request: AlertAcknowledgeRequest) -> dict:
    store = getattr(app.state, "alert_feedback_store", None)
    if store is None:
        return {"ok": False, "error": "alert_feedback_store_not_initialized"}
    camera_id = str(getattr(app.state, "alert_feedback_camera_id", "cam_01"))
    record = store.acknowledge(
        alert_id=request.alert_id,
        person_id=int(request.person_id),
        display_id=str(request.display_id),
        item=str(request.item),
        camera_id=camera_id,
        note=str(request.note),
        acknowledged=bool(request.acknowledged),
        positive_conf=float(request.positive_conf),
        negative_conf=float(request.negative_conf),
    )
    return {
        "ok": True,
        "record": record,
        "stats": store.stats(),
    }


@app.get("/api/alerts/feedback/stats")
async def alert_feedback_stats() -> dict:
    store = getattr(app.state, "alert_feedback_store", None)
    if store is None:
        return {"enabled": False, "error": "alert_feedback_store_not_initialized"}
    return store.stats()


@app.get("/metrics")
async def metrics() -> Response:
    exporter = getattr(app.state, "metrics_exporter", None)
    if exporter is None:
        return Response(content="# metrics exporter not initialized\n", media_type="text/plain", status_code=503)
    bridge = getattr(app.state, "jetson_bridge", None)
    snap = bridge.read_snapshot() if bridge is not None else None
    exporter.update(
        {},
        event_stream_dropped=int(getattr(app.state.pipeline.event_writer, "dropped", 0))
        if hasattr(app.state, "pipeline")
        else 0,
        jetson=snap,
    )
    return Response(content=exporter.render(), media_type=CONTENT_TYPE_LATEST)


def _safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "static"), html=True), name="static")
