"""FastAPI app for live PPE monitoring stream and dashboard hosting."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .ai_behavior_agent.agent import BehaviorAgentService
from .ai_behavior_agent.schemas import empty_agent_output
from .pipeline import MonitoringPipeline
from .startup_check import load_runtime_components
from .video_source import VideoSource


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    runtime = load_runtime_components(config, PROJECT_ROOT)
    behavior_service = None

    source_cfg = config["video"]
    print(
        json.dumps(
            {
                "event_type": "video_source_config",
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

    pipeline = MonitoringPipeline(
        pose_tracker=runtime.pose_tracker,
        ppe_detector=runtime.ppe_detector,
        verifier=runtime.verifier,
        config=config,
    )

    app.state.config = config
    app.state.runtime = runtime
    app.state.video_source = video_source
    app.state.pipeline = pipeline

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
        pipeline.event_writer.close()
        video_source.close()
        print(json.dumps({"event_type": "app_stopped"}))


app = FastAPI(title="PPE Monitoring", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "provider_info": app.state.runtime.provider_info,
    }


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    frame_id = 0
    target_fps = float(app.state.config["video"]["target_fps"])
    frame_interval = 1.0 / max(1.0, target_fps)
    last_tick = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
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
            )
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
