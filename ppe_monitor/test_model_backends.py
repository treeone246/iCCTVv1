"""Standalone model backend benchmark (no web app).

Use this script to compare ONNX vs TensorRT engine performance for:
- Pose model
- PPE detector model
- Verifier model

It runs each model on the same decoded frames and reports latency/FPS.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX and TensorRT engine models without the PPE web app."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Video source: webcam index ("0"), video file path, or image file path.',
    )
    parser.add_argument(
        "--pose-models",
        nargs="*",
        default=[],
        help="Pose model paths to benchmark (e.g., models/yolo26n-pose.onnx models/yolo26n-pose.engine).",
    )
    parser.add_argument(
        "--ppe-models",
        nargs="*",
        default=[],
        help="PPE detector model paths to benchmark (e.g., models/best.onnx models/best.engine).",
    )
    parser.add_argument(
        "--verifier-models",
        nargs="*",
        default=[],
        help="Verifier model paths to benchmark (e.g., models/yoloe-ppe.onnx models/yoloe-ppe.engine).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "0" or "cpu".')
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum frames/images to benchmark.")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth decoded frame.")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Warmup frames per model.")
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional JSON output report path.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing model files with warning instead of raising error.",
    )
    return parser.parse_args()


def parse_source(source_arg: str) -> Tuple[int | str, str]:
    if source_arg.isdigit():
        return int(source_arg), f"webcam:{source_arg}"

    source_path = Path(source_arg)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_arg}")
    return str(source_path), source_path.as_posix()


def load_frames(source: int | str, max_frames: int, frame_step: int) -> List[Any]:
    if isinstance(source, str):
        suffix = Path(source).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image = cv2.imread(source)
            if image is None:
                raise RuntimeError(f"Failed to read image: {source}")
            return [image]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    frames: List[Any] = []
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_step > 1 and (decoded % frame_step) != 0:
                decoded += 1
                continue
            frames.append(frame)
            decoded += 1
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("No frames loaded from source.")
    return frames


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * p))
    idx = max(0, min(idx, len(values_sorted) - 1))
    return values_sorted[idx]


def benchmark_model(
    model_path: Path,
    task: str,
    frames: List[Any],
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    warmup_frames: int,
) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.as_posix()}")

    model = YOLO(str(model_path), task=task)

    warmup_n = min(max(0, warmup_frames), len(frames))
    for idx in range(warmup_n):
        _ = model.predict(
            source=frames[idx],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

    latencies_ms: List[float] = []
    total_boxes = 0

    for frame in frames:
        start = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        if results and results[0].boxes is not None:
            total_boxes += len(results[0].boxes)

    total_ms = sum(latencies_ms)
    frames_n = len(latencies_ms)
    avg_ms = total_ms / frames_n if frames_n else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return {
        "model_path": model_path.as_posix(),
        "task": task,
        "frames": frames_n,
        "avg_ms": round(avg_ms, 3),
        "median_ms": round(statistics.median(latencies_ms), 3) if latencies_ms else 0.0,
        "p95_ms": round(percentile(latencies_ms, 0.95), 3),
        "fps": round(fps, 3),
        "total_boxes": int(total_boxes),
        "avg_boxes_per_frame": round(total_boxes / frames_n, 3) if frames_n else 0.0,
    }


def print_report(section: str, reports: List[Dict[str, Any]]) -> None:
    if not reports:
        return
    print(f"\n=== {section} ===")
    for rep in reports:
        print(
            f"{rep['model_path']} | task={rep['task']} | fps={rep['fps']} | "
            f"avg={rep['avg_ms']}ms median={rep['median_ms']}ms p95={rep['p95_ms']}ms | "
            f"avg_boxes={rep['avg_boxes_per_frame']}"
        )


def main() -> None:
    args = parse_args()
    source, source_label = parse_source(args.source)

    model_groups = {
        "pose": [Path(p) for p in args.pose_models],
        "ppe": [Path(p) for p in args.ppe_models],
        "verifier": [Path(p) for p in args.verifier_models],
    }
    if not any(model_groups.values()):
        raise ValueError("No models provided. Use --pose-models and/or --ppe-models and/or --verifier-models.")

    frames = load_frames(source, max_frames=args.max_frames, frame_step=args.frame_step)
    print(f"Loaded {len(frames)} frames from {source_label}")

    results: Dict[str, List[Dict[str, Any]]] = {"pose": [], "ppe": [], "verifier": []}

    missing_models: List[str] = []
    for task, paths in model_groups.items():
        for model_path in paths:
            print(f"Benchmarking {task} model: {model_path.as_posix()}")
            try:
                rep = benchmark_model(
                    model_path=model_path,
                    task="pose" if task == "pose" else "detect",
                    frames=frames,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    warmup_frames=args.warmup_frames,
                )
                results[task].append(rep)
            except FileNotFoundError as exc:
                if args.skip_missing:
                    print(f"WARNING: {exc}")
                    missing_models.append(model_path.as_posix())
                    continue
                raise

    print_report("POSE", results["pose"])
    print_report("PPE", results["ppe"])
    print_report("VERIFIER", results["verifier"])

    report_payload = {
        "source": source_label,
        "frames": len(frames),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "results": results,
    }

    if args.report_json:
        out_path = Path(args.report_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"\nWrote report: {out_path.as_posix()}")

    if missing_models:
        print("\nMissing models skipped:")
        for path in missing_models:
            print(f"- {path}")


if __name__ == "__main__":
    main()
