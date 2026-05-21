"""CLI entrypoint to run DeepStream backend directly."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from ..main import load_config
from ..pipeline import MonitoringPipeline
from ..ppe_detector import build_alias_index
from ..startup_check import load_runtime_components
from .compat import fill_unavailable_keypoints
from .config_builder import build_deepstream_settings
from .deepstream_pipeline import DeepStreamPipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepStream backend for ppe_monitor.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--source", type=str, default="", help="Override source URI")
    parser.add_argument("--engine", type=str, default="", help="Override TensorRT engine path")
    parser.add_argument("--camera-id", type=str, default="", help="Override camera id")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _load_cfg(project_root: Path, config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if path == (project_root / "config.yaml"):
        return load_config()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = _load_cfg(project_root, args.config)
    config.setdefault("runtime", {})["backend"] = "deepstream"

    runtime = load_runtime_components(config, project_root, skip_ppe=True)
    pipeline = MonitoringPipeline(
        pose_tracker=runtime.pose_tracker,
        ppe_detector=runtime.ppe_detector,
        verifier=runtime.verifier,
        config=config,
    )
    ds_settings = build_deepstream_settings(
        config,
        project_root=project_root,
        source_override=args.source or None,
        engine_override=args.engine or None,
        camera_id_override=args.camera_id or None,
    )
    if not ds_settings.enabled:
        raise RuntimeError("DeepStream runner requested but deepstream.enabled is false in config.")

    alias_map = build_alias_index(config.get("ppe_label_aliases", {}))
    person_classes = set(str(x) for x in config.get("deepstream", {}).get("person_classes", ["person"]))
    ppe_classes = set(str(x) for x in config.get("required_ppe", []))
    ppe_classes.update({"no_gloves", "no_boots", "harness"})

    runner = DeepStreamPipelineRunner(
        settings=ds_settings,
        label_map=config.get("deepstream", {}).get("label_map", {}),
        person_classes=person_classes,
        ppe_classes=ppe_classes,
        alias_to_canonical=alias_map,
    )

    frame_counter = 0
    start = time.time()
    try:
        runner.start()
        while True:
            bundle = runner.read_bundle(timeout_seconds=1.0)
            if bundle is None:
                continue
            if bundle.frame_bgr is None:
                continue

            # Decision 1b default: keep pose tracking in Python, use DeepStream PPE detections.
            fill_unavailable_keypoints(bundle.adapted.persons)
            payload, _ = pipeline.process_frame(
                bundle.frame_bgr,
                frame_counter,
                ppe_detections_override=bundle.adapted.ppe_detections,
                backend="deepstream",
            )
            frame_counter += 1
            print(
                json.dumps(
                    {
                        "event_type": "deepstream_frame_processed",
                        "frame_id": payload.frame_id,
                        "source_id": bundle.source_id,
                        "camera_id": bundle.camera_id,
                        "tracked_count": payload.metrics.tracked_count,
                        "active_violations": payload.metrics.active_violations,
                        "fps": payload.metrics.fps,
                        "input_fps": bundle.input_fps,
                        "per_camera_fps": bundle.camera_fps_map or {},
                        "ppe_detections": len(bundle.adapted.ppe_detections),
                    }
                )
            )
            if args.max_frames > 0 and frame_counter >= args.max_frames:
                break
    finally:
        runner.stop()
        pipeline.event_writer.close()
        elapsed = max(1e-6, time.time() - start)
        print(
            json.dumps(
                {
                    "event_type": "deepstream_run_stopped",
                    "frames": frame_counter,
                    "elapsed_s": round(elapsed, 2),
                    "avg_fps": round(frame_counter / elapsed, 2),
                }
            )
        )


if __name__ == "__main__":
    main()
