"""Run live PPE pipeline inference in an OpenCV window (no web app).

Supports ONNX or TensorRT engine model paths through config overrides.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from app.main import load_config
from app.pipeline import MonitoringPipeline
from app.schemas import Classification, OverallStatus
from app.startup_check import load_runtime_components


STATUS_COLOR = {
    "COMPLIANT": (40, 180, 40),
    "VIOLATION": (30, 30, 220),
    "INDETERMINATE": (0, 165, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live inference window using PPE pipeline models.")
    parser.add_argument("--source", type=str, default="0", help='Video source: webcam index like "0" or video file path.')
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--pose-model", type=str, default="", help="Override pose model path (.onnx or .engine).")
    parser.add_argument("--ppe-model", type=str, default="", help="Override PPE model path (.onnx or .engine).")
    parser.add_argument("--verifier-model", type=str, default="", help="Override verifier model path (.onnx or .engine).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf-pose", type=float, default=-1.0, help="Override pose confidence threshold.")
    parser.add_argument("--conf-ppe", type=float, default=-1.0, help="Override PPE confidence threshold.")
    parser.add_argument("--conf-verifier", type=float, default=-1.0, help="Override verifier confidence threshold.")
    parser.add_argument("--show-skeleton", action="store_true", help="Draw pose keypoints.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit (0 = infinite).")
    return parser.parse_args()


def parse_source(source_arg: str) -> int | str:
    return int(source_arg) if source_arg.isdigit() else source_arg


def draw_overlay(frame: Any, payload: Any, show_skeleton: bool) -> Any:
    out = frame.copy()

    for det in payload.ppe_detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 120, 0), 2)
        cv2.putText(
            out,
            f"{det.label}:{det.conf:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 120, 0),
            1,
            cv2.LINE_AA,
        )

    for person in payload.persons:
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        color = STATUS_COLOR.get(person.overall_status, (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"ID {person.person_id} {person.overall_status}",
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        line_y = y2 + 16
        for item, state in person.per_item_state.items():
            state_color = STATUS_COLOR.get(state, (128, 128, 128))
            cv2.putText(
                out,
                f"{item}:{state}",
                (x1, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                state_color,
                1,
                cv2.LINE_AA,
            )
            line_y += 15

        if show_skeleton:
            for kp in person.keypoints.values():
                if kp.conf < 0.4:
                    continue
                cv2.circle(out, (int(kp.x), int(kp.y)), 2, color, -1)

    m = payload.metrics
    top = f"FPS:{m.fps:.1f} tracked:{m.tracked_count} active:{m.active_violations} dropped:{m.dropped_frames} verifier/s:{m.verifier_calls_last_sec}"
    cv2.putText(out, top, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(out, top, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    config = load_config() if config_path == (project_root / "config.yaml") else yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.pose_model:
        config["models"]["pose"] = args.pose_model
    if args.ppe_model:
        config["models"]["ppe"] = args.ppe_model
    if args.verifier_model:
        config["models"]["verifier"] = args.verifier_model

    config["inference"]["imgsz"] = int(args.imgsz)
    if args.conf_pose >= 0:
        config["inference"]["conf_threshold_pose"] = float(args.conf_pose)
    if args.conf_ppe >= 0:
        config["inference"]["conf_threshold_ppe"] = float(args.conf_ppe)
    if args.conf_verifier >= 0:
        config["inference"]["conf_threshold_verifier"] = float(args.conf_verifier)

    runtime = load_runtime_components(config, project_root)
    pipeline = MonitoringPipeline(
        pose_tracker=runtime.pose_tracker,
        ppe_detector=runtime.ppe_detector,
        verifier=runtime.verifier,
        config=config,
    )

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    window_name = "PPE Monitor Live (q or ESC to quit)"
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            payload, _ = pipeline.process_frame(frame, frame_id)
            vis = draw_overlay(frame, payload, show_skeleton=args.show_skeleton)
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
