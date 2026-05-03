import argparse
import json
import pickle
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-time activity/state inference from YOLO pose keypoints + trained model."
        )
    )
    parser.add_argument("--pose-model", type=str, default="yolo26n-pose.pt")
    parser.add_argument(
        "--activity-model",
        type=str,
        default="models/pose_state_baseline.pkl",
        help="Path to trained classifier pickle from train_pose_state_baseline.py.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/pose_state_baseline_meta.json",
        help="Optional metadata JSON from training to auto-load window/min conf.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Frames per sequence window. 0 = load from metadata or use 30.",
    )
    parser.add_argument(
        "--min-kpt-conf",
        type=float,
        default=-1.0,
        help="Minimum keypoint confidence for normalization. -1 = metadata or 0.2.",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=45,
        help="Remove person history if unseen for this many frames.",
    )
    return parser.parse_args()


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            if hasattr(value, "tolist"):
                value = value.tolist()
    return np.asarray(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(float(value))
    except Exception:
        return default


def _choose_center(points_xy: np.ndarray, conf: np.ndarray, min_kpt_conf: float) -> np.ndarray:
    idx_pairs = [(11, 12), (5, 6)]
    for a, b in idx_pairs:
        if a < len(points_xy) and b < len(points_xy):
            if conf[a] >= min_kpt_conf and conf[b] >= min_kpt_conf:
                return (points_xy[a] + points_xy[b]) / 2.0

    valid = conf >= min_kpt_conf
    if np.any(valid):
        return np.mean(points_xy[valid], axis=0)
    return np.nanmean(points_xy, axis=0)


def _choose_scale(points_xy: np.ndarray, conf: np.ndarray, bbox_h: float, min_kpt_conf: float) -> float:
    candidates: list[float] = []
    idx_pairs = [(5, 6), (11, 12), (5, 11), (6, 12)]
    for a, b in idx_pairs:
        if a < len(points_xy) and b < len(points_xy):
            if conf[a] >= min_kpt_conf and conf[b] >= min_kpt_conf:
                dist = float(np.linalg.norm(points_xy[a] - points_xy[b]))
                if dist > 1e-6:
                    candidates.append(dist)

    if bbox_h > 1e-6:
        candidates.append(float(bbox_h))
    if candidates:
        return max(np.median(candidates), 1e-3)
    return 1.0


def _normalize_frame(keypoints: np.ndarray, bbox_h: float, min_kpt_conf: float) -> np.ndarray:
    points_xy = keypoints[:, :2].copy()
    conf = keypoints[:, 2].copy()

    points_xy = np.nan_to_num(points_xy, nan=0.0, posinf=0.0, neginf=0.0)
    center = _choose_center(points_xy, conf, min_kpt_conf=min_kpt_conf)
    scale = _choose_scale(points_xy, conf, bbox_h=bbox_h, min_kpt_conf=min_kpt_conf)
    norm_xy = (points_xy - center) / scale
    return np.concatenate([norm_xy, conf[:, None]], axis=1)


def _window_to_feature(window_keypoints: np.ndarray) -> np.ndarray:
    xy = window_keypoints[:, :, :2]
    conf = window_keypoints[:, :, 2:]
    delta_xy = np.diff(xy, axis=0, prepend=xy[:1])
    feature = np.concatenate(
        [
            window_keypoints.reshape(-1),
            delta_xy.reshape(-1),
            conf.mean(axis=0).reshape(-1),
        ]
    )
    return feature.astype(np.float32)


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_tracks(result: Any) -> list[dict[str, Any]]:
    keypoints_obj = getattr(result, "keypoints", None)
    boxes_obj = getattr(result, "boxes", None)
    if keypoints_obj is None:
        return []

    kpt_xy = _to_numpy(getattr(keypoints_obj, "xy", None))
    if kpt_xy is None or kpt_xy.size == 0:
        return []
    if kpt_xy.ndim == 2:
        kpt_xy = np.expand_dims(kpt_xy, axis=0)

    kpt_conf = _to_numpy(getattr(keypoints_obj, "conf", None))
    if kpt_conf is None or kpt_conf.size == 0:
        kpt_conf = np.ones((kpt_xy.shape[0], kpt_xy.shape[1]), dtype=np.float32)
    elif kpt_conf.ndim == 1:
        kpt_conf = np.expand_dims(kpt_conf, axis=0)

    boxes_xyxy = _to_numpy(getattr(boxes_obj, "xyxy", None))
    boxes_id = _to_numpy(getattr(boxes_obj, "id", None))
    boxes_conf = _to_numpy(getattr(boxes_obj, "conf", None))

    tracks: list[dict[str, Any]] = []
    for i in range(kpt_xy.shape[0]):
        if boxes_xyxy is not None and i < len(boxes_xyxy):
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i].tolist()]
        else:
            x1 = float(np.min(kpt_xy[i, :, 0]))
            y1 = float(np.min(kpt_xy[i, :, 1]))
            x2 = float(np.max(kpt_xy[i, :, 0]))
            y2 = float(np.max(kpt_xy[i, :, 1]))

        bbox_h = max(y2 - y1, 1e-3)
        person_id = (
            _to_int(boxes_id[i], default=-1)
            if boxes_id is not None and i < len(boxes_id)
            else -1
        )
        det_conf = (
            _to_float(boxes_conf[i], default=float(np.mean(kpt_conf[i])))
            if boxes_conf is not None and i < len(boxes_conf)
            else float(np.mean(kpt_conf[i]))
        )

        points = np.concatenate(
            [kpt_xy[i], kpt_conf[i].reshape(-1, 1)],
            axis=1,
        ).astype(np.float32)

        tracks.append(
            {
                "idx": i,
                "person_id": person_id,
                "bbox": (x1, y1, x2, y2),
                "bbox_h": bbox_h,
                "model_confidence": det_conf,
                "keypoints": points,
            }
        )
    return tracks


def _label_color(label: str) -> tuple[int, int, int]:
    label_key = label.strip().lower()
    semantic_colors = {
        "running": (60, 190, 255),
        "working": (90, 220, 120),
        "sleeping": (210, 140, 90),
        "falling": (70, 90, 255),
        "fainted": (120, 90, 230),
        "collecting...": (160, 160, 160),
    }
    if label_key in semantic_colors:
        return semantic_colors[label_key]

    palette = [(255, 120, 80), (80, 230, 180), (120, 170, 255), (255, 210, 90)]
    idx = abs(hash(label_key)) % len(palette)
    return palette[idx]


def _draw_translucent_rect(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def _draw_confidence_bar(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    value = float(max(0.0, min(1.0, value)))
    cv2.rectangle(image, (x, y), (x + width, y + height), (70, 70, 70), 1)
    fill_w = int(width * value)
    if fill_w > 0:
        cv2.rectangle(image, (x + 1, y + 1), (x + fill_w - 1, y + height - 1), color, -1)


def _draw_track_card(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    person_id: int,
    label: str,
    prob: float,
    progress_ratio: float,
    color: tuple[int, int, int],
) -> None:
    card_w = 260
    card_h = 64
    card_x1 = max(8, min(image.shape[1] - card_w - 8, x1))
    card_y1 = max(8, y1 - card_h - 10)
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h

    _draw_translucent_rect(image, card_x1, card_y1, card_x2, card_y2, (25, 25, 25), 0.55)
    cv2.rectangle(image, (card_x1, card_y1), (card_x2, card_y2), color, 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    title = f"ID {person_id}  {label.upper()}"
    cv2.putText(
        image,
        title,
        (card_x1 + 10, card_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    conf_text = f"conf {prob:.2f}" if prob > 0 else "conf --"
    cv2.putText(
        image,
        conf_text,
        (card_x1 + 10, card_y1 + 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    _draw_confidence_bar(
        image=image,
        x=card_x1 + 95,
        y=card_y1 + 34,
        width=150,
        height=10,
        value=prob if prob > 0 else progress_ratio,
        color=color,
    )


def _draw_header_hud(
    image: np.ndarray,
    fps: float,
    window_size: int,
    tracks: int,
    ready_tracks: int,
) -> None:
    x1, y1, x2, y2 = 10, 10, 470, 78
    _draw_translucent_rect(image, x1, y1, x2, y2, (18, 18, 18), 0.52)
    cv2.rectangle(image, (x1, y1), (x2, y2), (90, 210, 255), 2)

    cv2.putText(
        image,
        "Pose Activity Monitor",
        (x1 + 12, y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )
    stats_text = (
        f"FPS {fps:>4.1f}    tracks {tracks}    ready {ready_tracks}    window {window_size}"
    )
    cv2.putText(
        image,
        stats_text,
        (x1 + 12, y1 + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (200, 235, 245),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    meta = _load_meta(Path(args.metadata))

    window_size = args.window_size if args.window_size > 0 else int(meta.get("window_size", 30))
    min_kpt_conf = (
        args.min_kpt_conf
        if args.min_kpt_conf >= 0
        else float(meta.get("min_kpt_conf", 0.2))
    )

    with Path(args.activity_model).open("rb") as f:
        activity_model = pickle.load(f)

    pose_model = YOLO(args.pose_model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 0 or another index."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    histories: dict[int, deque[np.ndarray]] = defaultdict(
        lambda: deque(maxlen=window_size)
    )
    last_seen: dict[int, int] = {}
    predictions: dict[int, tuple[str, float]] = {}
    missing_id_counter = 100000

    frame_idx = 0
    prev_time = time.perf_counter()
    window_name = "YOLO26 Pose Activity Inference (q/Esc to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            results = pose_model.track(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                classes=args.classes,
                tracker=args.tracker,
                persist=True,
                verbose=False,
            )
            result = results[0]
            tracks = _extract_tracks(result)
            annotated = result.plot()

            active_ids: set[int] = set()
            ready_ids = 0
            for t in tracks:
                pid = int(t["person_id"])
                if pid < 0:
                    pid = missing_id_counter + t["idx"]
                active_ids.add(pid)
                last_seen[pid] = frame_idx

                normalized = _normalize_frame(
                    keypoints=t["keypoints"],
                    bbox_h=t["bbox_h"],
                    min_kpt_conf=min_kpt_conf,
                )
                histories[pid].append(normalized)

                if len(histories[pid]) >= window_size:
                    window = np.stack(list(histories[pid]), axis=0)
                    feature = _window_to_feature(window).reshape(1, -1)
                    label = str(activity_model.predict(feature)[0])
                    prob = 0.0
                    if hasattr(activity_model, "predict_proba"):
                        probs = activity_model.predict_proba(feature)[0]
                        prob = float(np.max(probs))
                    predictions[pid] = (label, prob)
                    ready_ids += 1

                label, prob = predictions.get(pid, ("collecting...", 0.0))
                x1, y1, x2, y2 = t["bbox"]
                color = _label_color(label)
                progress_ratio = min(1.0, len(histories[pid]) / max(float(window_size), 1.0))
                _draw_track_card(
                    image=annotated,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    person_id=pid,
                    label=label,
                    prob=prob,
                    progress_ratio=progress_ratio,
                    color=color,
                )

            stale_ids = [
                pid
                for pid, seen_frame in last_seen.items()
                if frame_idx - seen_frame > args.max_missing_frames
            ]
            for pid in stale_ids:
                histories.pop(pid, None)
                predictions.pop(pid, None)
                last_seen.pop(pid, None)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            _draw_header_hud(
                image=annotated,
                fps=fps,
                window_size=window_size,
                tracks=len(active_ids),
                ready_tracks=ready_ids,
            )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
