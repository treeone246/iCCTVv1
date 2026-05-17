"""Offline verifier benchmark on labeled PPE crops.

Dataset layout:
  <dataset_root>/
    helmet/
      compliant/*.jpg|*.png
      violation/*.jpg|*.png
    goggles/
      compliant/*.jpg|*.png
      violation/*.jpg|*.png
    gloves/
      compliant/*.jpg|*.png
      violation/*.jpg|*.png
    boots/
      compliant/*.jpg|*.png
      violation/*.jpg|*.png
    coverall/
      compliant/*.jpg|*.png
      violation/*.jpg|*.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
from ultralytics import YOLO

from app.schemas import VerifierVerdict
from app.verifier import YOLOEVerifier


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ItemMetrics:
    item: str
    samples: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    false_clear_rate: float
    violation_recall: float


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


def list_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOE verifier on labeled PPE crops.")
    parser.add_argument("--model", type=str, default="models/yoloe-ppe.onnx", help="Verifier model path.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root folder with item/compliant|violation structure.")
    parser.add_argument(
        "--items",
        nargs="+",
        default=["helmet", "goggles", "gloves", "boots", "coverall"],
        help="PPE items to evaluate.",
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Verifier confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--report-json", type=str, default="", help="Optional output JSON report path.")
    return parser.parse_args()


def evaluate_item(verifier: YOLOEVerifier, dataset_root: Path, item: str) -> ItemMetrics:
    compliant_dir = dataset_root / item / "compliant"
    violation_dir = dataset_root / item / "violation"

    tp = fp = tn = fn = 0
    samples = 0

    labeled_paths: List[tuple[Path, VerifierVerdict]] = []
    labeled_paths.extend((p, VerifierVerdict.COMPLIANT) for p in list_images(compliant_dir))
    labeled_paths.extend((p, VerifierVerdict.VIOLATION) for p in list_images(violation_dir))

    for image_path, ground_truth in labeled_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        samples += 1
        result = verifier.verify(image, item)
        pred = result.verdict

        if ground_truth == VerifierVerdict.COMPLIANT and pred == VerifierVerdict.COMPLIANT:
            tp += 1
        elif ground_truth == VerifierVerdict.VIOLATION and pred == VerifierVerdict.COMPLIANT:
            fp += 1
        elif ground_truth == VerifierVerdict.VIOLATION and pred == VerifierVerdict.VIOLATION:
            tn += 1
        elif ground_truth == VerifierVerdict.COMPLIANT and pred == VerifierVerdict.VIOLATION:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_clear_rate = safe_div(fp, fp + tn)
    violation_recall = safe_div(tn, tn + fp)

    return ItemMetrics(
        item=item,
        samples=samples,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_clear_rate=false_clear_rate,
        violation_recall=violation_recall,
    )


def aggregate(items: List[ItemMetrics]) -> Dict[str, float]:
    tp = sum(x.tp for x in items)
    fp = sum(x.fp for x in items)
    tn = sum(x.tn for x in items)
    fn = sum(x.fn for x in items)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_clear_rate = safe_div(fp, fp + tn)
    violation_recall = safe_div(tn, tn + fp)
    return {
        "samples": sum(x.samples for x in items),
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


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root.as_posix()}")

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.as_posix()}")

    model = YOLO(str(model_path), task="detect")
    verifier = YOLOEVerifier(model=model, conf_threshold=float(args.conf), imgsz=int(args.imgsz))

    item_reports: List[ItemMetrics] = []
    for item in args.items:
        report = evaluate_item(verifier, dataset_root, item)
        item_reports.append(report)

    summary = aggregate(item_reports)

    print("Verifier evaluation report")
    print("=" * 80)
    for report in item_reports:
        print(
            f"{report.item:10s} samples={report.samples:4d} "
            f"P={report.precision:.3f} R={report.recall:.3f} F1={report.f1:.3f} "
            f"false_clear={report.false_clear_rate:.3f} violation_recall={report.violation_recall:.3f} "
            f"(tp={report.tp}, fp={report.fp}, tn={report.tn}, fn={report.fn})"
        )
    print("-" * 80)
    print(
        f"OVERALL    samples={summary['samples']:4.0f} "
        f"P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['f1']:.3f} "
        f"false_clear={summary['false_clear_rate']:.3f} violation_recall={summary['violation_recall']:.3f} "
        f"(tp={summary['tp']:.0f}, fp={summary['fp']:.0f}, tn={summary['tn']:.0f}, fn={summary['fn']:.0f})"
    )

    report_payload = {
        "model": model_path.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "items": [asdict(r) for r in item_reports],
        "summary": summary,
    }
    if args.report_json:
        out_path = Path(args.report_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
