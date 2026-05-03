import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time webcam pose detection with an Ultralytics YOLO pose model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-pose.pt",
        help="Path or name of the YOLO pose model (default: yolo26n-pose.pt).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 1).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
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
        help="Optional class IDs to detect (example: --classes 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height (default: 720).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 1 or another index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Ultralytics YOLO26 Webcam Pose (press q to quit)"
    prev_time = time.perf_counter()
    warned_no_keypoints = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
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

            result = results[0]
            if result.keypoints is None and not warned_no_keypoints:
                print(
                    "Warning: no keypoints in output. "
                    "Check that you loaded a pose model (e.g., *-pose.pt)."
                )
                warned_no_keypoints = True

            annotated = result.plot()

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
