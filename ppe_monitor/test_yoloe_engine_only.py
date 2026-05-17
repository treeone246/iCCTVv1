"""Standalone YOLOE PPE engine test (no web app, no ensemble).

Use this to verify whether yoloe-ppe.engine detects PPE classes by itself.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test yoloe-ppe engine/onnx model alone.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yoloe-ppe.engine",
        help="Path to YOLOE model (.engine or .onnx).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Video source: webcam index like "0", or path to video/image.',
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--device", type=str, default="0", help='Device, e.g. "0" or "cpu".')
    parser.add_argument("--show", action="store_true", help="Show live inference window.")
    parser.add_argument(
        "--save-video",
        type=str,
        default="",
        help="Optional output video path (e.g. outputs/yoloe_only.mp4).",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit (0 = no limit).")
    return parser.parse_args()


def parse_source(source_arg: str) -> int | str:
    if source_arg.isdigit():
        return int(source_arg)
    source_path = Path(source_arg)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_arg}")
    return source_path.as_posix()


def draw_counts(frame, counts: Dict[str, int], fps: float) -> None:
    y = 20
    cv2.putText(
        frame,
        f"YOLOE only | FPS:{fps:.1f}",
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"YOLOE only | FPS:{fps:.1f}",
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    y += 22
    for label, count in counts.items():
        cv2.putText(
            frame,
            f"{label}: {count}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 210, 210),
            1,
            cv2.LINE_AA,
        )
        y += 18


def infer_single_image(model: YOLO, image_path: str, args: argparse.Namespace) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    results = model.predict(
        source=image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    result = results[0]
    annotated = result.plot()
    out_path = Path("outputs") / f"{Path(image_path).stem}_yoloe_only.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path.as_posix(), annotated)
    print(f"Saved: {out_path.as_posix()}")

    names = result.names if isinstance(result.names, dict) else {}
    if result.boxes is not None and result.boxes.cls is not None:
        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        print("Detections:")
        for i, c in enumerate(cls):
            label = str(names.get(int(c), int(c)))
            score = float(conf[i]) if i < len(conf) else 0.0
            print(f"- {label}: {score:.3f}")
    else:
        print("No detections.")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.as_posix()}")

    model = YOLO(model_path.as_posix(), task="detect")
    source = parse_source(args.source)

    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        infer_single_image(model, source, args)
        return

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
        writer = cv2.VideoWriter(
            out_path.as_posix(),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to create output video: {out_path.as_posix()}")

    frame_count = 0
    t_prev = time.perf_counter()
    fps_smoothed = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()

            counts: Dict[str, int] = {}
            names = result.names if isinstance(result.names, dict) else {}
            if result.boxes is not None and result.boxes.cls is not None:
                cls = result.boxes.cls.cpu().numpy()
                for c in cls:
                    label = str(names.get(int(c), int(c)))
                    counts[label] = counts.get(label, 0) + 1

            t_now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps_smoothed = inst_fps if fps_smoothed == 0.0 else (0.8 * fps_smoothed + 0.2 * inst_fps)
            draw_counts(annotated, counts, fps_smoothed)

            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("YOLOE PPE Only (q or ESC to quit)", annotated)
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

    print(f"Done. Frames processed: {frame_count}")
    if args.save_video:
        print(f"Saved video: {Path(args.save_video).as_posix()}")


if __name__ == "__main__":
    main()
