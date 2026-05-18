"""Standalone best2 model test (ONNX or TensorRT engine).

Use this to validate best2.onnx or best2.engine inference without the web app.
Supports image/video/webcam, optional live display, optional saved annotated video,
and optional JSON benchmark output.
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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test best2.onnx or best2.engine model standalone.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best2.onnx",
        help="Path to model (.onnx or .engine).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Video source: webcam index like "0", or path to video/image.',
    )
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"], help="Model task.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0" or "cpu".')
    parser.add_argument("--show", action="store_true", help="Show live inference window.")
    parser.add_argument(
        "--save-video",
        type=str,
        default="",
        help="Optional output video path (e.g. outputs/best2_test.mp4).",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default="",
        help="Optional output image path for image inference.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit (0 = no limit).")
    parser.add_argument("--report-json", type=str, default="", help="Optional JSON report path.")
    return parser.parse_args()


def parse_source(source_arg: str) -> int | str:
    if source_arg.isdigit():
        return int(source_arg)
    source_path = Path(source_arg)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_arg}")
    return source_path.as_posix()


def safe_label(names: Dict[int, str], class_id: int) -> str:
    return str(names.get(class_id, f"class_{class_id}"))


def draw_detections_safe(frame: Any, result: Any) -> Tuple[Any, Dict[str, int], List[int], int]:
    out = frame.copy()
    counts: Dict[str, int] = {}
    unknown_ids: List[int] = []

    names = result.names if isinstance(result.names, dict) else {}
    if result.boxes is None or result.boxes.xyxy is None:
        return out, counts, unknown_ids, 0

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
    cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

    for i, box in enumerate(xyxy):
        class_id = int(cls[i]) if i < len(cls) else -1
        label = safe_label(names, class_id)
        if class_id not in names:
            unknown_ids.append(class_id)

        score = float(conf[i]) if i < len(conf) else 0.0
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 180, 255) if class_id in names else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label}:{score:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        counts[label] = counts.get(label, 0) + 1

    return out, counts, sorted(set(unknown_ids)), len(xyxy)


def draw_overlay_stats(frame: Any, fps: float, total_boxes: int, counts: Dict[str, int]) -> None:
    cv2.putText(frame, f"best2 test | FPS:{fps:.1f} boxes:{total_boxes}", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(frame, f"best2 test | FPS:{fps:.1f} boxes:{total_boxes}", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    y = 40
    for label, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:8]:
        cv2.putText(frame, f"{label}: {cnt}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 210), 1, cv2.LINE_AA)
        y += 18


def run_image(model: YOLO, source: str, args: argparse.Namespace) -> dict:
    image = cv2.imread(source)
    if image is None:
        raise RuntimeError(f"Failed to read image: {source}")

    t0 = time.perf_counter()
    results = model.predict(
        source=image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    result = results[0]

    annotated, counts, unknown_ids, box_count = draw_detections_safe(image, result)
    draw_overlay_stats(annotated, fps=(1000.0 / max(1e-6, dt_ms)), total_boxes=box_count, counts=counts)

    out_path = Path(args.save_image) if args.save_image else Path("outputs") / f"{Path(source).stem}_best2_test.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path.as_posix(), annotated)

    report = {
        "mode": "image",
        "model": args.model,
        "task": args.task,
        "source": source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "inference_ms": round(dt_ms, 3),
        "boxes": box_count,
        "class_counts": counts,
        "unknown_class_ids": unknown_ids,
        "output_image": out_path.as_posix(),
    }
    return report


def run_video(model: YOLO, source: int | str, args: argparse.Namespace) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    writer = None
    if args.save_video:
        out_path = Path(args.save_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to create output video: {out_path.as_posix()}")

    frame_count = 0
    lat_ms: List[float] = []
    total_boxes = 0
    class_totals: Dict[str, int] = {}
    unknown_id_counts: Dict[int, int] = {}

    t_prev = time.perf_counter()
    fps_smoothed = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            lat_ms.append(dt_ms)
            result = results[0]

            annotated, counts, unknown_ids, box_count = draw_detections_safe(frame, result)
            total_boxes += box_count
            for k, v in counts.items():
                class_totals[k] = class_totals.get(k, 0) + v
            for uid in unknown_ids:
                unknown_id_counts[uid] = unknown_id_counts.get(uid, 0) + 1

            t_now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps_smoothed = inst_fps if fps_smoothed == 0.0 else (0.8 * fps_smoothed + 0.2 * inst_fps)
            draw_overlay_stats(annotated, fps_smoothed, box_count, counts)

            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("best2 standalone test (q or ESC to quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    avg_ms = statistics.mean(lat_ms) if lat_ms else 0.0
    med_ms = statistics.median(lat_ms) if lat_ms else 0.0
    p95_ms = statistics.quantiles(lat_ms, n=100)[94] if len(lat_ms) >= 20 else max(lat_ms, default=0.0)

    report = {
        "mode": "video",
        "model": args.model,
        "task": args.task,
        "source": args.source,
        "frames": frame_count,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "avg_ms": round(avg_ms, 3),
        "median_ms": round(med_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "fps": round(1000.0 / max(1e-6, avg_ms), 3) if avg_ms > 0 else 0.0,
        "total_boxes": int(total_boxes),
        "avg_boxes_per_frame": round((total_boxes / frame_count), 3) if frame_count > 0 else 0.0,
        "class_totals": dict(sorted(class_totals.items(), key=lambda x: (-x[1], x[0]))),
        "unknown_class_ids": {str(k): v for k, v in sorted(unknown_id_counts.items())},
        "saved_video": args.save_video if args.save_video else "",
    }
    return report


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.as_posix()}")

    model = YOLO(model_path.as_posix(), task=args.task)
    source = parse_source(args.source)

    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
        report = run_image(model, source, args)
    else:
        report = run_video(model, source, args)

    print(json.dumps(report, indent=2))

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {out.as_posix()}")


if __name__ == "__main__":
    main()
