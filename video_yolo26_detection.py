import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26 detection on a video using best.pt."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best2.pt",
        help="Path to YOLO model weights (default: best.pt).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="simulationTest2.mp4",
        help="Input video path or filename in videos/ (default: simulationTesting.mp4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output annotated video path (default: outputs/<video>_yolo26_detectionNew.mp4).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold (default: 0.20).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.12,
        help="IoU threshold for NMS (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to run on, e.g. "cpu", "0", "0,1" (default: auto).',
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Optional class IDs to detect (example: --classes 0 2 3).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window while processing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for number of frames to process.",
    )
    return parser.parse_args()


def resolve_video_path(video_arg: str) -> Path:
    video_path = Path(video_arg)
    if video_path.exists():
        return video_path

    in_videos_dir = Path("videos") / video_arg
    if in_videos_dir.exists():
        return in_videos_dir

    raise FileNotFoundError(
        f"Video not found: {video_arg}. Also checked: {in_videos_dir.as_posix()}"
    )


def default_output_path(video_path: Path) -> Path:
    return Path("outputs") / f"{video_path.stem}_yolo26_detectionNew.mp4"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.as_posix()}")

    video_path = resolve_video_path(args.video)
    output_path = Path(args.output) if args.output else default_output_path(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
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
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path.as_posix()}")

    processed = 0
    window_name = "YOLO26 Video Detection (press q to quit)"

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
                classes=args.classes,
                verbose=False,
            )

            annotated = results[0].plot()
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

    print(f"Done. Annotated video saved to: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
