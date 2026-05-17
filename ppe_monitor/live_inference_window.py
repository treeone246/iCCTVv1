"""Run live PPE pipeline inference in an OpenCV window (no web app).

Supports ONNX or TensorRT engine model paths through config overrides.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import yaml

from app.main import load_config
from app.pipeline import MonitoringPipeline
from app.startup_check import load_runtime_components


STATUS_COLOR = {
    "COMPLIANT": (40, 180, 40),
    "VIOLATION": (30, 30, 220),
    "INDETERMINATE": (0, 165, 255),
}

REASON_LEGEND = {
    "detected_and_spatially_bound": "Detected and correctly worn",
    "keypoint_not_visible_or_out_of_frame": "Cannot assess (limb/keypoint not visible)",
    "detected_but_held_not_worn": "Detected but held, not worn",
    "direct_violation": "Direct violation from association",
    "verifier_cache_compliant": "Compliant from verifier cache",
    "verifier_cache_violation": "Violation from verifier cache",
    "verifier_yoloe_compliant": "Compliant by YOLOE verifier",
    "verifier_yoloe_violation": "Violation by YOLOE verifier",
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
    parser.add_argument("--show-reason-legend", action="store_true", help="Show reason code legend on screen.")
    parser.add_argument(
        "--legend-position",
        type=str,
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="top-right",
        help="Legend position for reason panel.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit (0 = infinite).")
    return parser.parse_args()


def parse_source(source_arg: str) -> int | str:
    return int(source_arg) if source_arg.isdigit() else source_arg


def _human_reason(reason: str) -> str:
    if not reason:
        return ""
    return REASON_LEGEND.get(reason, reason.replace("_", " "))


def draw_overlay(
    frame: Any,
    payload: Any,
    show_skeleton: bool,
    show_reason_legend: bool,
    legend_position: str,
) -> Any:
    out = frame.copy()

    for det in payload.ppe_detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        if getattr(det, "source", None) == "yoloe_aux":
            box_color = (60, 220, 220)  # yellow-cyan for ensemble detector
            source_tag = "YOLOE"
        else:
            box_color = (200, 120, 0)  # blue-ish for primary detector
            source_tag = "BEST"
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            out,
            f"{det.label}:{det.conf:.2f} [{source_tag}]",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
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
            reason = person.per_item_reason.get(item, "") if hasattr(person, "per_item_reason") else ""
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
            if reason and state != "COMPLIANT":
                cv2.putText(
                    out,
                    f"  reason:{_human_reason(reason)}",
                    (x1, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    state_color,
                    1,
                    cv2.LINE_AA,
                )
                line_y += 14

        if show_skeleton:
            for kp in person.keypoints.values():
                if kp.conf < 0.4:
                    continue
                cv2.circle(out, (int(kp.x), int(kp.y)), 2, color, -1)

    m = payload.metrics
    top = f"FPS:{m.fps:.1f} tracked:{m.tracked_count} active:{m.active_violations} dropped:{m.dropped_frames} verifier/s:{m.verifier_calls_last_sec}"
    cv2.putText(out, top, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(out, top, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    if show_reason_legend:
        lines = [
            "Reason Legend:",
            "detected_and_spatially_bound = Detected and worn",
            "keypoint_not_visible_or_out_of_frame = Cannot assess",
            "detected_but_held_not_worn = Held PPE, not worn",
            "direct_violation = Association violation",
            "verifier_cache_compliant/violation = Cached verifier result",
            "verifier_yoloe_compliant/violation = YOLOE verifier result",
            "BBox colors: BEST=blue, YOLOE ensemble=yellow",
        ]
        margin = 8
        line_h = 14
        panel_w = 500
        panel_h = margin * 2 + line_h * len(lines)
        h, w = out.shape[:2]

        if legend_position == "top-left":
            x = 12
            y = 40
        elif legend_position == "top-right":
            x = max(12, w - panel_w - 12)
            y = 40
        elif legend_position == "bottom-left":
            x = 12
            y = max(24, h - panel_h + 12)
        else:  # bottom-right
            x = max(12, w - panel_w - 12)
            y = max(24, h - panel_h + 12)

        cv2.rectangle(out, (x - 6, y - 14), (x - 6 + panel_w, y - 14 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(out, (x - 6, y - 14), (x - 6 + panel_w, y - 14 + panel_h), (140, 140, 140), 1)
        for i, line in enumerate(lines):
            ly = y + i * line_h
            cv2.putText(out, line, (x, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
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
            vis = draw_overlay(
                frame,
                payload,
                show_skeleton=args.show_skeleton,
                show_reason_legend=args.show_reason_legend,
                legend_position=args.legend_position,
            )
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
