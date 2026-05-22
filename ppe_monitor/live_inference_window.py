"""Run live PPE pipeline inference in an OpenCV window (no web app).

Supports ONNX or TensorRT engine model paths through config overrides.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.main import load_config
from app.jetson_exporter_bridge import JetsonExporterBridge
from app.performance_logger import PerformanceLogWriter
from app.pipeline import MonitoringPipeline
from app.compliance_events import ComplianceEventWriter
from app.ppe_memory import (
    PPEMemoryConfig,
    PPEMemoryManager,
    PersonState,
    build_ppe_observations_for_person,
)
from app.startup_check import load_runtime_components

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


STATUS_COLOR = {
    "COMPLIANT": (40, 180, 40),
    "VIOLATION": (30, 30, 220),
    "INDETERMINATE": (0, 165, 255),
}

MEMORY_STATE_COLOR = {
    PersonState.COMPLIANT_CONFIRMED.value: (40, 180, 40),      # green
    PersonState.COMPLIANT_CANDIDATE.value: (0, 200, 200),      # yellow-ish
    PersonState.VIOLATION_CANDIDATE.value: (0, 165, 255),      # orange
    PersonState.VIOLATION_CONFIRMED.value: (30, 30, 220),      # red
    PersonState.UNKNOWN.value: (128, 128, 128),                # gray
}

SKELETON_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

REASON_LEGEND = {
    "detected_and_spatially_bound": "Detected and correctly worn",
    "keypoint_not_visible_or_out_of_frame": "Cannot assess (limb/keypoint not visible)",
    "detected_but_held_not_worn": "Detected but held, not worn",
    "direct_violation": "Direct violation from association",
    "verifier_cache_compliant": "Compliant from verifier cache",
    "verifier_cache_violation": "Violation from verifier cache",
    "verifier_yoloe_compliant": "Compliant by YOLOE verifier",
    "verifier_yoloe_violation": "Violation by YOLOE verifier",
    "violation_candidate": "Timer stage: candidate (vote threshold reached)",
    "violation_confirmed": "Timer stage: confirmed (stable duration reached)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live inference window using PPE pipeline models.")
    parser.add_argument("--source", type=str, default="0", help='Video source: webcam index like "0" or video file path.')
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--pose-model", type=str, default="", help="Override pose model path (.onnx or .engine).")
    parser.add_argument("--ppe-model", type=str, default="", help="Override PPE model path (.onnx or .engine).")
    parser.add_argument("--verifier-model", type=str, default="", help="Override verifier model path (.onnx or .engine).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default="", help='Optional device hint: "cpu", "auto", or GPU index like "0".')
    parser.add_argument("--conf-pose", type=float, default=-1.0, help="Override pose confidence threshold.")
    parser.add_argument("--conf-ppe", type=float, default=-1.0, help="Override PPE confidence threshold.")
    parser.add_argument("--conf-verifier", type=float, default=-1.0, help="Override verifier confidence threshold.")
    parser.add_argument("--show-skeleton", action="store_true", help="Draw pose keypoints.")
    parser.add_argument("--skeleton-conf", type=float, default=0.25, help="Keypoint confidence floor for drawing skeleton.")
    parser.add_argument("--show-reason-legend", action="store_true", help="Show reason code legend on screen.")
    parser.add_argument("--enable-memory", action="store_true", help="Enable per-track PPE compliance memory anti-spam.")
    parser.add_argument("--memory-window", type=int, default=30, help="Memory vote window frames.")
    parser.add_argument("--memory-min-frames", type=int, default=15, help="Minimum valid votes before memory item decision.")
    parser.add_argument("--alert-cooldown", type=float, default=60.0, help="Cooldown seconds between repeated alerts.")
    parser.add_argument(
        "--events-jsonl",
        type=str,
        default="outputs/compliance_events.jsonl",
        help="JSONL path for confirmed violation events when memory is enabled.",
    )
    parser.add_argument(
        "--hide-status-dashboard",
        action="store_true",
        help="Hide separate status dashboard window.",
    )
    parser.add_argument(
        "--legend-position",
        type=str,
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="top-right",
        help="Legend position for reason panel.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit (0 = infinite).")
    parser.add_argument(
        "--image-interval-ms",
        type=int,
        default=0,
        help="Autoplay interval for image mode (0 = manual next/prev controls).",
    )
    return parser.parse_args()


def parse_source(source_arg: str) -> int | str:
    return int(source_arg) if source_arg.isdigit() else source_arg


def collect_image_paths(source_arg: str, project_root: Path) -> list[Path]:
    src = Path(source_arg)
    if not src.is_absolute():
        src = (project_root / src).resolve()
    if src.is_file() and src.suffix.lower() in IMAGE_SUFFIXES:
        return [src]
    if src.is_dir():
        return sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    return []


def _human_reason(reason: str) -> str:
    if not reason:
        return ""
    return REASON_LEGEND.get(reason, reason.replace("_", " "))


def draw_overlay(
    frame: Any,
    payload: Any,
    show_skeleton: bool,
    skeleton_conf: float,
    show_reason_legend: bool,
    legend_position: str,
    memory_state_by_person: dict[int, str] | None = None,
    memory_label_by_person: dict[int, str] | None = None,
) -> Any:
    out = frame.copy()
    source_counts = {"BEST": 0, "YOLOE": 0}

    for det in payload.ppe_detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        if getattr(det, "source", None) == "yoloe_aux":
            box_color = (0, 215, 255)  # orange for YOLOE auxiliary detector
            source_tag = "YOLOE"
            source_counts["YOLOE"] += 1
        else:
            box_color = (255, 0, 0)  # strong blue for BEST2 primary detector
            source_tag = "BEST"
            source_counts["BEST"] += 1
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
        pid = int(person.person_id)
        memory_state = memory_state_by_person.get(pid) if memory_state_by_person else None
        if memory_state:
            color = MEMORY_STATE_COLOR.get(memory_state, (128, 128, 128))
            person_line = memory_label_by_person.get(pid, f"ID {pid} | {memory_state}") if memory_label_by_person else f"ID {pid} | {memory_state}"
        else:
            color = STATUS_COLOR.get(person.overall_status, (128, 128, 128))
            person_line = f"ID {person.person_id} {person.overall_status}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            person_line,
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
            for a, b in SKELETON_PAIRS:
                pa = person.keypoints.get(a)
                pb = person.keypoints.get(b)
                if pa is None or pb is None:
                    continue
                if float(pa.conf) < skeleton_conf or float(pb.conf) < skeleton_conf:
                    continue
                cv2.line(
                    out,
                    (int(pa.x), int(pa.y)),
                    (int(pb.x), int(pb.y)),
                    color,
                    1,
                    cv2.LINE_AA,
                )
            for kp in person.keypoints.values():
                if float(kp.conf) < skeleton_conf:
                    continue
                cv2.circle(out, (int(kp.x), int(kp.y)), 3, color, -1)

    m = payload.metrics
    top = (
        f"FPS:{m.fps:.1f} tracked:{m.tracked_count} active:{m.active_violations} "
        f"dropped:{m.dropped_frames} verifier/s:{m.verifier_calls_last_sec} "
        f"raw[BEST:{getattr(m, 'ppe_primary_raw', 0)} YOLOE:{getattr(m, 'verifier_aux_raw', 0)}] "
        f"out[BEST:{source_counts['BEST']} YOLOE:{source_counts['YOLOE']}]"
    )
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


def _state_value(state: Any) -> str:
    if hasattr(state, "value"):
        return str(state.value)
    return str(state)


def _status_color_bgr(state: str) -> tuple[int, int, int]:
    return STATUS_COLOR.get(state, (128, 128, 128))


def render_status_dashboard(payload: Any) -> np.ndarray:
    width = 640
    row_h = 64
    item_col_w = 80
    items = ["helmet", "gloves", "coverall", "boots", "goggles"]
    people = list(payload.persons)
    total_rows = 4 + len(people)
    height = max(220, 24 + total_rows * row_h)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (24, 24, 24)

    cv2.putText(
        canvas,
        "PPE Status Dashboard",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    m = payload.metrics
    model_line = (
        f"PPE model: {Path(getattr(m, 'ppe_model', '')).name or 'unknown'} "
        f"({getattr(m, 'ppe_task', 'detect')}, fusion={getattr(m, 'ppe_fusion_mode', 'nms')})"
    )
    infer_line = (
        f"infer calls: BEST2={getattr(m, 'ppe_infer_calls', 0)} "
        f"YOLOEaux={getattr(m, 'verifier_aux_infer_calls', 0)}"
    )
    raw_line = (
        f"raw/frame: BEST2={getattr(m, 'ppe_primary_raw', 0)} "
        f"YOLOEaux={getattr(m, 'verifier_aux_raw', 0)} merged={getattr(m, 'ppe_merged', 0)}"
    )
    cv2.putText(canvas, model_line, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, infer_line, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, raw_line, (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Per-item global counters
    y = 104
    item_counts = {it: {"COMPLIANT": 0, "VIOLATION": 0, "INDETERMINATE": 0} for it in items}
    for person in people:
        for item in items:
            state = _state_value(person.per_item_state.get(item, "INDETERMINATE"))
            if state not in item_counts[item]:
                state = "INDETERMINATE"
            item_counts[item][state] += 1

    for item in items:
        compliant = item_counts[item]["COMPLIANT"]
        violation = item_counts[item]["VIOLATION"]
        indet = item_counts[item]["INDETERMINATE"]
        text = f"{item:<9} ok:{compliant:>2}  bad:{violation:>50}  unk:{indet:>20}"
        cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        y += 20

    y += 8
    cv2.line(canvas, (12, y), (width - 12, y), (75, 75, 75), 1)
    y += 18

    header = "Person".ljust(8) + "".join(i[:6].upper().ljust(9) for i in items) + "OVERALL"
    cv2.putText(canvas, header, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    y += 12

    for person in people:
        y += row_h
        pid = f"P{person.person_id}"
        cv2.putText(canvas, pid, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        x = 86
        for item in items:
            state = _state_value(person.per_item_state.get(item, "INDETERMINATE"))
            if state == "COMPLIANT":
                mark = "OK"
            elif state == "VIOLATION":
                mark = "X"
            else:
                mark = "?"
            color = _status_color_bgr(state)
            cv2.rectangle(canvas, (x, y - 20), (x + item_col_w - 10, y + 4), (55, 55, 55), 1)
            cv2.putText(canvas, mark, (x + 26, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
            x += item_col_w

        overall = _state_value(person.overall_status)
        cv2.putText(canvas, overall, (x + 8, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _status_color_bgr(overall), 2, cv2.LINE_AA)

    return canvas


def _item_visible_from_keypoints(person: Any, item: str, conf_floor: float = 0.25) -> bool:
    keypoint_groups = {
        "helmet": ["nose", "left_eye", "right_eye"],
        "safety_glasses": ["left_eye", "right_eye"],
        "gloves": ["left_wrist", "right_wrist"],
        "boots": ["left_ankle", "right_ankle"],
        "coverall": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    }
    names = keypoint_groups.get(item, [])
    if not names:
        return True
    kp_map = getattr(person, "keypoints", {}) or {}
    visible = 0
    for name in names:
        kp = kp_map.get(name)
        if kp is None:
            continue
        conf = float(getattr(kp, "conf", 0.0))
        if conf >= conf_floor:
            visible += 1
    return visible > 0


def _status_to_classification_text(item_status: str) -> str:
    if item_status == "OK":
        return "COMPLIANT"
    if item_status == "VIOLATION":
        return "VIOLATION"
    return "INDETERMINATE"


def apply_memory_layer(
    payload: Any,
    manager: PPEMemoryManager,
    events: ComplianceEventWriter | None,
    camera_id: str,
    keypoint_conf_floor: float = 0.25,
) -> tuple[dict[int, str], dict[int, str]]:
    memory_state_by_person: dict[int, str] = {}
    memory_label_by_person: dict[int, str] = {}

    for person in payload.persons:
        pid = int(person.person_id)
        person_box = tuple(float(v) for v in person.bbox)
        observations = build_ppe_observations_for_person(
            person_box=person_box,
            detections=payload.ppe_detections,
            names=None,
        )

        # Use keypoint visibility to assign None when body part is not evaluable.
        for item in ["helmet", "coverall", "gloves", "safety_glasses", "boots"]:
            if not _item_visible_from_keypoints(person, item, conf_floor=keypoint_conf_floor):
                observations[item] = None

        # TODO: Production mode should always rely on real tracker IDs from ByteTrack/BoT-SORT/DeepStream nvtracker.
        memory = manager.update_track(camera_id=camera_id, track_id=pid, observations=observations, bbox=person_box)
        if events is not None and memory.should_emit_alert():
            events.write_alert(memory)

        item_stats = memory.item_statuses()
        mapped = {
            "helmet": _status_to_classification_text(item_stats["helmet"].value),
            "coverall": _status_to_classification_text(item_stats["coverall"].value),
            "gloves": _status_to_classification_text(item_stats["gloves"].value),
            "goggles": _status_to_classification_text(item_stats["safety_glasses"].value),
            "boots": _status_to_classification_text(item_stats["boots"].value),
        }
        mapped_reason = {
            "helmet": item_stats["helmet"].value.lower(),
            "coverall": item_stats["coverall"].value.lower(),
            "gloves": item_stats["gloves"].value.lower(),
            "goggles": item_stats["safety_glasses"].value.lower(),
            "boots": item_stats["boots"].value.lower(),
        }
        person.per_item_state = mapped
        person.per_item_reason = mapped_reason

        state = memory.state.value
        memory_state_by_person[pid] = state

        if state == PersonState.COMPLIANT_CONFIRMED.value:
            person.overall_status = "COMPLIANT"
        elif state == PersonState.VIOLATION_CONFIRMED.value:
            person.overall_status = "VIOLATION"
        else:
            person.overall_status = "INDETERMINATE"

        short = (
            f"H:{item_stats['helmet'].value[0]} "
            f"C:{item_stats['coverall'].value[0]} "
            f"G:{item_stats['gloves'].value[0]} "
            f"E:{item_stats['safety_glasses'].value[0]} "
            f"B:{item_stats['boots'].value[0]}"
        )
        memory_label_by_person[pid] = f"ID {pid} | {state} | {short}"

    manager.cleanup()
    return memory_state_by_person, memory_label_by_person


def run_image_mode(
    args: argparse.Namespace,
    pipeline: MonitoringPipeline,
    image_paths: list[Path],
    memory_manager: PPEMemoryManager | None,
    event_writer: ComplianceEventWriter | None,
    performance_logger: PerformanceLogWriter | None,
    jetson_bridge: JetsonExporterBridge | None,
) -> None:
    if not image_paths:
        raise RuntimeError("Image mode requested but no images were found.")

    window_name = "PPE Monitor Live (Image Mode)"
    dashboard_window_name = "PPE Status Dashboard"
    idx = 0
    frame_id = 0
    shown = 0

    while True:
        image_path = image_paths[idx]
        frame = cv2.imread(image_path.as_posix())
        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path.as_posix()}")

        payload, _ = pipeline.process_frame(frame, frame_id)
        if performance_logger is not None:
            jetson_snapshot = jetson_bridge.read_snapshot() if jetson_bridge is not None else None
            performance_logger.emit(
                frame_id=frame_id,
                metrics=payload.metrics.model_dump(),
                jetson=jetson_snapshot,
                source="live_window_image_mode",
                timestamp=payload.timestamp,
            )
        memory_state_by_person: dict[int, str] = {}
        memory_label_by_person: dict[int, str] = {}
        if memory_manager is not None:
            memory_state_by_person, memory_label_by_person = apply_memory_layer(
                payload=payload,
                manager=memory_manager,
                events=event_writer,
                camera_id="image_mode",
                keypoint_conf_floor=float(args.skeleton_conf),
            )

        vis = draw_overlay(
            frame,
            payload,
            show_skeleton=args.show_skeleton,
            skeleton_conf=float(args.skeleton_conf),
            show_reason_legend=args.show_reason_legend,
            legend_position=args.legend_position,
            memory_state_by_person=memory_state_by_person,
            memory_label_by_person=memory_label_by_person,
        )
        cv2.putText(
            vis,
            f"Image {idx + 1}/{len(image_paths)}: {image_path.name} | keys: n-next p-prev q-quit",
            (12, max(40, vis.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, vis)
        if not args.hide_status_dashboard:
            status_dash = render_status_dashboard(payload)
            cv2.imshow(dashboard_window_name, status_dash)

        wait_ms = max(1, int(args.image_interval_ms)) if args.image_interval_ms > 0 else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q") or key == 27:
            break
        if key in (ord("n"), ord("d"), 32):  # next
            idx = (idx + 1) % len(image_paths)
            frame_id += 1
            shown += 1
        elif key in (ord("p"), ord("a")):  # previous
            idx = (idx - 1) % len(image_paths)
        elif args.image_interval_ms > 0:
            idx = (idx + 1) % len(image_paths)
            frame_id += 1
            shown += 1
        else:
            # manual mode: any other key means stay on current image
            pass

        if args.max_frames > 0 and shown >= args.max_frames:
            break

    cv2.destroyAllWindows()


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
    if args.device:
        d = str(args.device).strip().lower()
        if d == "cpu":
            config["inference"]["device"] = "cpu"
        elif d in {"auto", "cuda"}:
            config["inference"]["device"] = "auto"
        else:
            # Numeric GPU index maps to auto/CUDA path in current startup loader.
            config["inference"]["device"] = "auto"
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

    memory_manager: PPEMemoryManager | None = None
    event_writer: ComplianceEventWriter | None = None
    performance_logger = PerformanceLogWriter(config)
    jetson_bridge = JetsonExporterBridge.from_app_config(config)
    if args.enable_memory:
        mem_cfg = PPEMemoryConfig(
            vote_window_frames=int(args.memory_window),
            min_frames_for_decision=int(args.memory_min_frames),
            alert_cooldown_sec=float(args.alert_cooldown),
        )
        memory_manager = PPEMemoryManager(mem_cfg)
        event_writer = ComplianceEventWriter(path=args.events_jsonl)

    try:
        image_paths = collect_image_paths(args.source, project_root)
        if image_paths:
            run_image_mode(
                args=args,
                pipeline=pipeline,
                image_paths=image_paths,
                memory_manager=memory_manager,
                event_writer=event_writer,
                performance_logger=performance_logger,
                jetson_bridge=jetson_bridge,
            )
            return

        source = parse_source(args.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open source: {args.source}")

        window_name = "PPE Monitor Live (q or ESC to quit)"
        dashboard_window_name = "PPE Status Dashboard"
        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                payload, _ = pipeline.process_frame(frame, frame_id)
                jetson_snapshot = jetson_bridge.read_snapshot() if jetson_bridge is not None else None
                performance_logger.emit(
                    frame_id=frame_id,
                    metrics=payload.metrics.model_dump(),
                    jetson=jetson_snapshot,
                    source="live_window_video_mode",
                    timestamp=payload.timestamp,
                )
                memory_state_by_person: dict[int, str] = {}
                memory_label_by_person: dict[int, str] = {}
                if memory_manager is not None:
                    memory_state_by_person, memory_label_by_person = apply_memory_layer(
                        payload=payload,
                        manager=memory_manager,
                        events=event_writer,
                        camera_id=str(args.source),
                        keypoint_conf_floor=float(args.skeleton_conf),
                    )
                vis = draw_overlay(
                    frame,
                    payload,
                    show_skeleton=args.show_skeleton,
                    skeleton_conf=float(args.skeleton_conf),
                    show_reason_legend=args.show_reason_legend,
                    legend_position=args.legend_position,
                    memory_state_by_person=memory_state_by_person,
                    memory_label_by_person=memory_label_by_person,
                )
                cv2.imshow(window_name, vis)
                if not args.hide_status_dashboard:
                    status_dash = render_status_dashboard(payload)
                    cv2.imshow(dashboard_window_name, status_dash)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

                frame_id += 1
                if args.max_frames > 0 and frame_id >= args.max_frames:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    finally:
        performance_logger.close()
        pipeline.event_writer.close()
        pipeline.close()


if __name__ == "__main__":
    main()
