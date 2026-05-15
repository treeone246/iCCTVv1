import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOE ONNX inference on a video, preview, and save annotated output."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yoloe-ppe.onnx",
        help="Path to ONNX model (default: yoloe-ppe.onnx in alternatif).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=r"C:\iCCTVv1\videos\simulationTest2.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output annotated video path (default: outputs/<video>_yoloe_ppe_onnx.mp4).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device, e.g. "cpu" or "0".',
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live inference window while processing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    return parser.parse_args()


def resolve_path(candidate: str, search_project_root: bool = False) -> Path:
    raw = Path(candidate)
    if raw.exists():
        return raw.resolve()

    in_script_dir = SCRIPT_DIR / candidate
    if in_script_dir.exists():
        return in_script_dir.resolve()

    if search_project_root:
        in_project_root = PROJECT_ROOT / candidate
        if in_project_root.exists():
            return in_project_root.resolve()

    raise FileNotFoundError(
        f"Path not found: {candidate}. Checked: {raw.as_posix()}, {in_script_dir.as_posix()}"
    )


def default_output_path(video_path: Path) -> Path:
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{video_path.stem}_yoloe_ppe_onnx.mp4"


def draw_fallback(frame, result):
    annotated = frame.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"cls:{cls_id} {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def main() -> None:
    args = parse_args()

    model_path = resolve_path(args.model, search_project_root=True)
    video_path = resolve_path(args.video, search_project_root=True)
    output_video = Path(args.output_video).resolve() if args.output_video else default_output_path(video_path)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path), task="segment")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path.as_posix()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video.as_posix()}")

    processed = 0
    window_name = "YOLOE ONNX Inference (press q or ESC to stop)"

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
            try:
                annotated = result.plot()
            except KeyError:
                annotated = draw_fallback(frame, result)
            writer.write(annotated)

            if args.show:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            processed += 1
            if args.max_frames is not None and processed >= args.max_frames:
                break
            if processed % 30 == 0:
                if frame_count > 0:
                    print(f"Processed {processed}/{frame_count} frames...")
                else:
                    print(f"Processed {processed} frames...")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("Done.")
    print(f"Model: {model_path.as_posix()}")
    print(f"Input video: {video_path.as_posix()}")
    print(f"Output video: {output_video.as_posix()}")
    print(f"Processed frames: {processed}")


if __name__ == "__main__":
    main()
