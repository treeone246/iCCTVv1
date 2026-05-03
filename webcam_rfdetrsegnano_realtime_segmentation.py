import argparse
import time

import cv2
import numpy as np
from rfdetr import RFDETRSegNano
from rfdetr.assets.coco_classes import COCO_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time webcam instance segmentation with RFDETRSegNano."
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
        help="Optional class IDs to segment (example: --classes 0 1).",
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
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Mask overlay alpha in [0,1] (default: 0.45).",
    )
    return parser.parse_args()


def class_name_from_id(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return str(COCO_CLASSES[class_id])
    return f"class_{class_id}"


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 12345)
    bgr = rng.integers(40, 255, size=3, dtype=np.int32)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_segmentations(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    mask_alpha: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]

    for xyxy, score, class_id, mask in zip(boxes, scores, class_ids, masks):
        class_id = int(class_id)
        color = color_for_class(class_id)

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask.astype(bool)
        if np.any(mask_bool):
            overlay = out.copy()
            overlay[mask_bool] = color
            out = cv2.addWeighted(overlay, mask_alpha, out, 1.0 - mask_alpha, 0)

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name_from_id(class_id)} {float(score):.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        y_label = max(y1 - 8, text_h + baseline + 4)
        cv2.rectangle(
            out,
            (x1, y_label - text_h - baseline - 6),
            (x1 + text_w + 6, y_label),
            color,
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
    mask_alpha = float(np.clip(args.mask_alpha, 0.0, 1.0))

    model = RFDETRSegNano()
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 0 or another index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "RFDETRSegNano Webcam Segmentation (press q to quit)"
    prev_time = time.perf_counter()
    warned_no_masks = False

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
            masks = np.asarray(getattr(detections, "mask", np.empty((0, 0, 0))))

            if boxes.ndim == 1 and boxes.size == 4:
                boxes = boxes[None, :]
            if scores.ndim == 0:
                scores = scores[None]
            if class_ids.ndim == 0:
                class_ids = class_ids[None]
            if masks.ndim == 2:
                masks = masks[None, ...]

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

                if masks.ndim == 3 and masks.shape[0] == len(valid):
                    masks = masks[valid]
                else:
                    masks = np.zeros(
                        (len(boxes), frame.shape[0], frame.shape[1]), dtype=np.uint8
                    )
            else:
                boxes = np.empty((0, 4), dtype=float)
                scores = np.empty((0,), dtype=float)
                class_ids = np.empty((0,), dtype=int)
                masks = np.empty((0, frame.shape[0], frame.shape[1]), dtype=np.uint8)

            if masks.shape[0] == 0 and not warned_no_masks:
                print(
                    "Warning: model returned no masks. Confirm RFDETRSegNano weights are loaded."
                )
                warned_no_masks = True

            annotated = draw_segmentations(
                frame=frame,
                boxes=boxes,
                scores=scores,
                class_ids=class_ids,
                masks=masks,
                mask_alpha=mask_alpha,
            )

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
