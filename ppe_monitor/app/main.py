"""FastAPI app for live PPE monitoring stream and dashboard hosting."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

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

    print(json.dumps({"event_type": "app_started"}))
    try:
        yield
    finally:
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


app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "static"), html=True), name="static")
