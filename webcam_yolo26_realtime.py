import argparse
import time

import cv2
from ultralytics import YOLO

import config
from model_adapter import adapt_yolo_detections
from windows_sender import ZmqDetectionSender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time webcam object detection with an Ultralytics YOLO model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Path or name of the YOLO model (default: yolo26.pt).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Webcam index (default: 0).",
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
        help="Optional class IDs to detect (example: --classes 0 2 3).",
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
    parser.add_argument(
        "--host",
        type=str,
        default=config.ZMQ_HOST,
        help="ZMQ receiver host/IP (default from config.py).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.ZMQ_PORT,
        help="ZMQ receiver port (default from config.py).",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default=config.DEFAULT_FRAME_ID,
        help="frame_id field for outgoing payloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    sender = ZmqDetectionSender(config.zmq_endpoint(host=args.host, port=args.port))

    if not cap.isOpened():
        sender.close()
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 1 or another index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Ultralytics YOLO26 Webcam Detection (press q to quit)"
    prev_time = time.perf_counter()

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
            payload = adapt_yolo_detections(
                yolo_boxes=result.boxes,
                frame_id=args.frame_id,
                class_names=model.names,
            )
            sender.send_payload(payload)

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
        sender.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
