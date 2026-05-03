import argparse
import time

import cv2
import numpy as np
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time webcam object detection with RFDETRNano."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Webcam index (default: 1).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
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
    return parser.parse_args()


def class_name_from_id(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return str(COCO_CLASSES[class_id])
    return f"class_{class_id}"


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    out = frame.copy()

    for xyxy, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        label = f"{class_name_from_id(int(class_id))} {float(score):.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        y_label = max(y1 - 8, text_h + baseline + 4)
        cv2.rectangle(
            out,
            (x1, y_label - text_h - baseline - 6),
            (x1 + text_w + 6, y_label),
            (0, 220, 0),
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 3, y_label - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def main() -> None:
    args = parse_args()
    class_filter = set(args.classes) if args.classes else None

    model = RFDETRNano()
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 0 or another index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "RFDETRNano Webcam Detection (press q to quit)"
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            # RF-DETR expects RGB images.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(frame_rgb, threshold=args.conf)

            boxes = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))))
            scores = np.asarray(getattr(detections, "confidence", np.empty((0,))))
            class_ids = np.asarray(getattr(detections, "class_id", np.empty((0,))))

            if boxes.size > 0:
                valid = np.ones(len(boxes), dtype=bool)

                if class_filter is not None and class_ids.size > 0:
                    valid &= np.isin(class_ids.astype(int), list(class_filter))

                boxes = boxes[valid]
                if scores.size > 0:
                    scores = scores[valid]
                else:
                    scores = np.zeros((len(boxes),), dtype=float)

                if class_ids.size > 0:
                    class_ids = class_ids[valid].astype(int)
                else:
                    class_ids = np.full((len(boxes),), -1, dtype=int)
            else:
                boxes = np.empty((0, 4), dtype=float)
                scores = np.empty((0,), dtype=float)
                class_ids = np.empty((0,), dtype=int)

            annotated = draw_detections(frame, boxes, scores, class_ids)

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
