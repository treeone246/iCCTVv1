"""Evaluate base-only vs base+verifier PPE decisions on labeled video frames.

Annotation schema (JSON):
{
  "items": ["helmet", "goggles", "gloves", "boots", "coverall"],
  "frames": [
    {
      "frame_id": 10,
      "persons": [
        {
          "bbox": [x1, y1, x2, y2],
          "items": {
            "helmet": "COMPLIANT",
            "gloves": "VIOLATION",
            "boots": "INDETERMINATE"
          }
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml

from app.association import AssociationEngine, iou
from app.pipeline import MonitoringPipeline
from app.schemas import Classification
from app.startup_check import load_runtime_components


BBox = Tuple[float, float, float, float]
VALID_GT = {"COMPLIANT", "VIOLATION", "INDETERMINATE"}


@dataclass
class BinaryMetrics:
    item: str
    evaluated: int = 0
    ignored_indeterminate: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, gt: str, pred: str) -> None:
        self.evaluated += 1
        if gt == "COMPLIANT" and pred == "COMPLIANT":
            self.tp += 1
        elif gt == "VIOLATION" and pred == "COMPLIANT":
            self.fp += 1
        elif gt == "VIOLATION" and pred == "VIOLATION":
            self.tn += 1
        elif gt == "COMPLIANT" and pred == "VIOLATION":
            self.fn += 1

    def to_report(self) -> dict:
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        false_clear_rate = safe_div(self.fp, self.fp + self.tn)
        violation_recall = safe_div(self.tn, self.tn + self.fp)
        return {
            "item": self.item,
            "evaluated": self.evaluated,
            "ignored_indeterminate": self.ignored_indeterminate,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_clear_rate": false_clear_rate,
            "violation_recall": violation_recall,
        }


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pipeline lift from verifier.")
    parser.add_argument("--video", type=str, required=True, help="Path to annotated video.")
    parser.add_argument("--labels", type=str, required=True, help="Path to frame-level labels JSON.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--iou-match-threshold", type=float, default=0.30, help="IoU threshold for GT-person matching.")
    parser.add_argument("--report-json", type=str, default="", help="Optional output JSON report path.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    frame_map: Dict[int, dict] = {}
    for frame in frames:
        frame_id = int(frame["frame_id"])
        frame_map[frame_id] = frame
    payload["_frame_map"] = frame_map
    return payload


def normalize_gt(value: str) -> str:
    text = value.strip().upper()
    if text not in VALID_GT:
        raise ValueError(f"Invalid ground-truth label: {value}. Expected one of {sorted(VALID_GT)}.")
    return text


def cls_to_binary_base(cls: Classification) -> Optional[str]:
    if cls == Classification.COMPLIANT:
        return "COMPLIANT"
    if cls == Classification.INDETERMINATE:
        return None
    return "VIOLATION"


def cls_to_binary_full(cls: Classification) -> Optional[str]:
    if cls == Classification.COMPLIANT:
        return "COMPLIANT"
    if cls == Classification.INDETERMINATE:
        return None
    return "VIOLATION"


def match_person(gt_bbox: BBox, tracked: list, threshold: float) -> Optional[int]:
    best_idx: Optional[int] = None
    best_iou = 0.0
    for idx, person in enumerate(tracked):
        score = iou(gt_bbox, person.bbox)
        if score > best_iou:
            best_iou = score
            best_idx = idx
    if best_idx is None or best_iou < threshold:
        return None
    return best_idx


def aggregate(reports: List[dict]) -> dict:
    tp = sum(r["tp"] for r in reports)
    fp = sum(r["fp"] for r in reports)
    tn = sum(r["tn"] for r in reports)
    fn = sum(r["fn"] for r in reports)
    evaluated = sum(r["evaluated"] for r in reports)
    ignored_indeterminate = sum(r["ignored_indeterminate"] for r in reports)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_clear_rate = safe_div(fp, fp + tn)
    violation_recall = safe_div(tn, tn + fp)
    return {
        "evaluated": evaluated,
        "ignored_indeterminate": ignored_indeterminate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_clear_rate": false_clear_rate,
        "violation_recall": violation_recall,
    }


def print_mode_table(title: str, items: List[dict], summary: dict) -> None:
    print(title)
    print("=" * 100)
    for r in items:
        print(
            f"{r['item']:10s} eval={r['evaluated']:4d} "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            f"false_clear={r['false_clear_rate']:.3f} vio_recall={r['violation_recall']:.3f} "
            f"(tp={r['tp']}, fp={r['fp']}, tn={r['tn']}, fn={r['fn']}, ignore={r['ignored_indeterminate']})"
        )
    print("-" * 100)
    print(
        f"OVERALL    eval={summary['evaluated']:4d} "
        f"P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['f1']:.3f} "
        f"false_clear={summary['false_clear_rate']:.3f} vio_recall={summary['violation_recall']:.3f} "
        f"(tp={summary['tp']}, fp={summary['fp']}, tn={summary['tn']}, fn={summary['fn']}, ignore={summary['ignored_indeterminate']})"
    )


def print_lift(base_summary: dict, full_summary: dict) -> None:
    print("LIFT (base+verifier minus base-only)")
    print("=" * 100)
    print(f"precision        {full_summary['precision'] - base_summary['precision']:+.3f}")
    print(f"recall           {full_summary['recall'] - base_summary['recall']:+.3f}")
    print(f"f1               {full_summary['f1'] - base_summary['f1']:+.3f}")
    print(f"false_clear_rate {full_summary['false_clear_rate'] - base_summary['false_clear_rate']:+.3f}")
    print(f"violation_recall {full_summary['violation_recall'] - base_summary['violation_recall']:+.3f}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve() if Path(args.config).is_absolute() else (project_root / args.config).resolve()
    video_path = Path(args.video).resolve()
    labels_path = Path(args.labels).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path.as_posix()}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path.as_posix()}")

    config = load_config(config_path)
    labels = load_labels(labels_path)
    frame_map: Dict[int, dict] = labels["_frame_map"]
    if not frame_map:
        raise ValueError("No annotated frames found in labels file.")

    items = labels.get("items") or list(config["required_ppe"])
    base_metrics = {item: BinaryMetrics(item=item) for item in items}
    full_metrics = {item: BinaryMetrics(item=item) for item in items}

    runtime = load_runtime_components(config, project_root)
    pipeline = MonitoringPipeline(
        pose_tracker=runtime.pose_tracker,
        ppe_detector=runtime.ppe_detector,
        verifier=runtime.verifier,
        config=config,
    )
    association = AssociationEngine(config)

    cap = cv2.VideoCapture(video_path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path.as_posix()}")

    max_frame_id = max(frame_map.keys())
    frame_id = 0
    matched_persons = 0
    unmatched_persons = 0
    annotated_persons = 0

    started = time.perf_counter()
    try:
        while frame_id <= max_frame_id:
            ok, frame = cap.read()
            if not ok:
                break

            annotations = frame_map.get(frame_id)
            tracked_people = runtime.pose_tracker.track(frame)
            ppe_detections = runtime.ppe_detector.detect(frame)

            if annotations is not None:
                for gt_person in annotations.get("persons", []):
                    annotated_persons += 1
                    gt_bbox = tuple(float(v) for v in gt_person["bbox"])
                    match_idx = match_person(gt_bbox, tracked_people, float(args.iou_match_threshold))
                    if match_idx is None:
                        unmatched_persons += 1
                        continue
                    matched_persons += 1

                    tracked = tracked_people[match_idx]
                    detection_dicts = [
                        {"label": det.label, "bbox": det.bbox, "conf": det.conf}
                        for det in ppe_detections
                    ]

                    for item in items:
                        gt_map = gt_person.get("items", {})
                        if item not in gt_map:
                            continue
                        gt = normalize_gt(str(gt_map[item]))
                        if gt == "INDETERMINATE":
                            base_metrics[item].ignored_indeterminate += 1
                            full_metrics[item].ignored_indeterminate += 1
                            continue

                        base_cls, _ = association.classify_item(
                            item=item,
                            keypoints=tracked.keypoints,
                            keypoint_confidences=tracked.keypoint_confidences,
                            ppe_detections=detection_dicts,
                            frame_shape=frame.shape,
                        )
                        full_cls = pipeline._classify_person_item(tracked, ppe_detections, item, frame)

                        base_pred = cls_to_binary_base(base_cls)
                        full_pred = cls_to_binary_full(full_cls)
                        if base_pred is None:
                            base_metrics[item].ignored_indeterminate += 1
                        else:
                            base_metrics[item].add(gt, base_pred)
                        if full_pred is None:
                            full_metrics[item].ignored_indeterminate += 1
                        else:
                            full_metrics[item].add(gt, full_pred)

            frame_id += 1
    finally:
        cap.release()

    base_reports = [base_metrics[item].to_report() for item in items]
    full_reports = [full_metrics[item].to_report() for item in items]
    base_summary = aggregate(base_reports)
    full_summary = aggregate(full_reports)

    elapsed = time.perf_counter() - started
    runtime_stats = {
        "frames_processed": frame_id,
        "max_annotated_frame_id": max_frame_id,
        "elapsed_seconds": elapsed,
        "effective_fps": safe_div(frame_id, elapsed),
        "annotated_persons": annotated_persons,
        "matched_persons": matched_persons,
        "unmatched_persons": unmatched_persons,
        "person_match_rate": safe_div(matched_persons, annotated_persons),
    }

    print_mode_table("BASE ONLY (association without verifier)", base_reports, base_summary)
    print("")
    print_mode_table("BASE + VERIFIER (current pipeline behavior)", full_reports, full_summary)
    print("")
    print_lift(base_summary, full_summary)
    print("")
    print("MATCH / THROUGHPUT")
    print("=" * 100)
    print(json.dumps(runtime_stats, indent=2))

    report = {
        "video": video_path.as_posix(),
        "labels": labels_path.as_posix(),
        "config": config_path.as_posix(),
        "items": items,
        "runtime": runtime_stats,
        "base_only": {"per_item": base_reports, "summary": base_summary},
        "base_plus_verifier": {"per_item": full_reports, "summary": full_summary},
        "lift": {
            "precision": full_summary["precision"] - base_summary["precision"],
            "recall": full_summary["recall"] - base_summary["recall"],
            "f1": full_summary["f1"] - base_summary["f1"],
            "false_clear_rate": full_summary["false_clear_rate"] - base_summary["false_clear_rate"],
            "violation_recall": full_summary["violation_recall"] - base_summary["violation_recall"],
        },
    }

    if args.report_json:
        out_path = Path(args.report_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
