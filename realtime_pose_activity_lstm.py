import argparse
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime activity inference from YOLO pose + LSTM model "
            "(streaming with warmup, smoothing, and uncertainty handling)."
        )
    )
    parser.add_argument("--pose-model", type=str, default="yolo26n-pose.pt")
    parser.add_argument("--activity-model", type=str, default="models/pose_activity_lstm_fine.pt")
    parser.add_argument(
        "--source",
        type=str,
        default="videos/CNN_TF_YOLO_TEST_INFERENCE.mp4",
        help="Video source path (e.g., videos/CNN_TF_YOLO_TEST_INFERENCE.mp4). If empty, uses webcam.",
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
        help="Override model window size. 0 = use checkpoint config.",
    )
    parser.add_argument(
        "--min-live-frames",
        type=int,
        default=8,
        help="Start streaming predictions after this many frames per track.",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--unknown-threshold", type=float, default=0.58)
    parser.add_argument("--decision-hold", type=int, default=3)
    parser.add_argument("--max-missing-frames", type=int, default=45)
    parser.add_argument(
        "--disable-optical-flow",
        action="store_true",
        help="Disable optical-flow layer and rely on pose/LSTM only.",
    )
    parser.add_argument(
        "--flow-min-mag-px",
        type=float,
        default=0.8,
        help="Minimum flow magnitude (pixels) treated as real motion.",
    )
    parser.add_argument(
        "--flow-low-threshold",
        type=float,
        default=1.2,
        help="Mean flow threshold for light movement.",
    )
    parser.add_argument(
        "--flow-active-threshold",
        type=float,
        default=2.2,
        help="Mean flow threshold for active movement.",
    )
    parser.add_argument(
        "--flow-heavy-threshold",
        type=float,
        default=3.2,
        help="Mean flow threshold for heavy movement.",
    )
    parser.add_argument(
        "--flow-accident-p90-threshold",
        type=float,
        default=4.0,
        help="P90 flow magnitude threshold for accident-like bursts.",
    )
    parser.add_argument(
        "--flow-accident-vertical-ratio",
        type=float,
        default=0.58,
        help="Minimum vertical-motion ratio for accident-like movement.",
    )
    parser.add_argument(
        "--label-scope",
        type=str,
        choices=["fine", "baseline"],
        default="fine",
        help="Display detailed labels or baseline labels (working/not_working/accident).",
    )
    parser.add_argument(
        "--hide-general-state",
        action="store_true",
        help="Hide derived general-state text (working/not_working/accident) on the track card.",
    )
    parser.add_argument(
        "--accident-min-instability",
        type=float,
        default=1.30,
        help="Minimum instability ratio required to show accident label.",
    )
    parser.add_argument(
        "--accident-min-conf",
        type=float,
        default=0.62,
        help="Minimum model confidence for accident decisions after gating.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Optional path to save annotated output video (e.g., outputs/rigAccident_pred.mp4).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help=(
            "When using --source video file, auto-save annotated output to "
            "outputs/<video>_realtime_pose_activity_lstm.mp4 if --save-output is not set."
        ),
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


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def _choose_center(points_xy: np.ndarray, conf: np.ndarray, min_kpt_conf: float) -> np.ndarray:
    for a, b in [(11, 12), (5, 6)]:
        if a < len(points_xy) and b < len(points_xy):
            if conf[a] >= min_kpt_conf and conf[b] >= min_kpt_conf:
                return (points_xy[a] + points_xy[b]) / 2.0

    valid = conf >= min_kpt_conf
    if np.any(valid):
        return np.mean(points_xy[valid], axis=0)
    return np.mean(points_xy, axis=0)


def _choose_scale(points_xy: np.ndarray, conf: np.ndarray, bbox_h: float, min_kpt_conf: float) -> float:
    candidates: list[float] = []
    for a, b in [(5, 6), (11, 12), (5, 11), (6, 12)]:
        if a < len(points_xy) and b < len(points_xy):
            if conf[a] >= min_kpt_conf and conf[b] >= min_kpt_conf:
                d = float(np.linalg.norm(points_xy[a] - points_xy[b]))
                if d > 1e-6:
                    candidates.append(d)
    if bbox_h > 1e-6:
        candidates.append(float(bbox_h))
    if not candidates:
        return 1.0
    return max(float(np.median(candidates)), 1e-3)


def normalize_frame(keypoints_xyc: np.ndarray, bbox_h: float, min_kpt_conf: float) -> np.ndarray:
    xy = keypoints_xyc[:, :2].copy()
    conf = keypoints_xyc[:, 2].copy()
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    center = _choose_center(xy, conf, min_kpt_conf=min_kpt_conf)
    scale = _choose_scale(xy, conf, bbox_h=bbox_h, min_kpt_conf=min_kpt_conf)
    nxy = (xy - center) / scale
    return np.concatenate([nxy, conf[:, None]], axis=1).astype(np.float32)


class LSTMActivityClassifier(nn.Module):
    def __init__(
        self,
        kpt_count: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        input_size = kpt_count * 3
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, k, c = x.shape
        z = x.reshape(b, t, k * c)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :])


def _extract_tracks(result: Any) -> list[dict[str, Any]]:
    kpts_obj = getattr(result, "keypoints", None)
    boxes_obj = getattr(result, "boxes", None)
    if kpts_obj is None:
        return []

    kpt_xy = _to_numpy(getattr(kpts_obj, "xy", None))
    if kpt_xy is None or kpt_xy.size == 0:
        return []
    if kpt_xy.ndim == 2:
        kpt_xy = np.expand_dims(kpt_xy, axis=0)

    kpt_conf = _to_numpy(getattr(kpts_obj, "conf", None))
    if kpt_conf is None or kpt_conf.size == 0:
        kpt_conf = np.ones((kpt_xy.shape[0], kpt_xy.shape[1]), dtype=np.float32)
    elif kpt_conf.ndim == 1:
        kpt_conf = np.expand_dims(kpt_conf, axis=0)

    boxes_xyxy = _to_numpy(getattr(boxes_obj, "xyxy", None))
    boxes_id = _to_numpy(getattr(boxes_obj, "id", None))

    tracks: list[dict[str, Any]] = []
    for i in range(kpt_xy.shape[0]):
        if boxes_xyxy is not None and i < len(boxes_xyxy):
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i].tolist()]
        else:
            x1 = float(np.min(kpt_xy[i, :, 0]))
            y1 = float(np.min(kpt_xy[i, :, 1]))
            x2 = float(np.max(kpt_xy[i, :, 0]))
            y2 = float(np.max(kpt_xy[i, :, 1]))
        pid = (
            _to_int(boxes_id[i], default=-1)
            if boxes_id is not None and i < len(boxes_id)
            else -1
        )
        bbox_h = max(y2 - y1, 1e-3)
        arr = np.concatenate([kpt_xy[i], kpt_conf[i].reshape(-1, 1)], axis=1).astype(np.float32)
        tracks.append({"idx": i, "person_id": pid, "bbox": (x1, y1, x2, y2), "bbox_h": bbox_h, "keypoints": arr})
    return tracks


def _color_for_label(label: str) -> tuple[int, int, int]:
    palette = {
        "standing": (100, 220, 120),
        "walking": (90, 200, 255),
        "sitting": (190, 170, 90),
        "manual_working": (95, 230, 190),
        "drilling_operation": (70, 120, 255),
        "pipe_handling": (90, 220, 255),
        "crouching": (225, 160, 80),
        "bending": (235, 140, 120),
        "falling": (60, 80, 255),
        "lying": (150, 110, 220),
        "working": (80, 215, 150),
        "not_working": (210, 175, 90),
        "accident": (65, 85, 255),
        "uncertain": (150, 150, 150),
        "warming_up": (170, 170, 170),
    }
    return palette.get(label, (120, 220, 220))


def _draw_overlay_box(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float = 0.45,
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


def _draw_track_card(
    image: np.ndarray,
    pid: int,
    bbox: tuple[float, float, float, float],
    label: str,
    general_label: str,
    conf: float,
    warm_ratio: float,
    instability: float,
    show_general: bool,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = _color_for_label(label)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    card_w = 300
    card_h = 84 if show_general else 70
    card_x1 = max(8, min(image.shape[1] - card_w - 8, x1))
    card_y1 = max(8, y1 - card_h - 8)
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h
    _draw_overlay_box(image, card_x1, card_y1, card_x2, card_y2, (25, 25, 25), alpha=0.58)
    cv2.rectangle(image, (card_x1, card_y1), (card_x2, card_y2), color, 2)

    title = f"ID {pid}  {label.upper()}"
    cv2.putText(
        image,
        title,
        (card_x1 + 10, card_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    if label == "warming_up":
        stat_text = f"window fill {warm_ratio * 100:.0f}%"
        bar_val = warm_ratio
    else:
        stat_text = f"conf {conf:.2f}  instab x{instability:.2f}"
        bar_val = conf

    cv2.putText(
        image,
        stat_text,
        (card_x1 + 10, card_y1 + 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if show_general:
        general_text = f"state: {general_label}"
        cv2.putText(
            image,
            general_text,
            (card_x1 + 10, card_y1 + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 235, 235),
            1,
            cv2.LINE_AA,
        )

    bar_x, bar_y, bar_w, bar_h = card_x1 + 140, card_y1 + 40, 145, 11
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    fill_w = int(max(0.0, min(1.0, bar_val)) * bar_w)
    if fill_w > 0:
        cv2.rectangle(
            image,
            (bar_x + 1, bar_y + 1),
            (bar_x + fill_w - 1, bar_y + bar_h - 1),
            color,
            -1,
        )


def _draw_header(image: np.ndarray, fps: float, tracks: int, ready: int, window_size: int) -> None:
    _draw_overlay_box(image, 10, 10, 140, 44, (7, 7, 7), alpha=0.55)
    cv2.putText(
        image,
        f"FPS {fps:.1f}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (205, 230, 240),
        2,
        cv2.LINE_AA,
    )


def _flow_roi_features(
    flow: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    min_mag_px: float,
) -> dict[str, float]:
    if flow is None or flow.size == 0:
        return {
            "mean_mag": 0.0,
            "p90_mag": 0.0,
            "std_mag": 0.0,
            "vertical_ratio": 0.0,
            "moving_ratio": 0.0,
        }

    h, w = flow.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return {
            "mean_mag": 0.0,
            "p90_mag": 0.0,
            "std_mag": 0.0,
            "vertical_ratio": 0.0,
            "moving_ratio": 0.0,
        }

    roi = flow[y1:y2, x1:x2]
    fx = roi[:, :, 0]
    fy = roi[:, :, 1]
    mag = np.sqrt((fx * fx) + (fy * fy)).astype(np.float32)

    mean_mag = float(np.mean(mag)) if mag.size else 0.0
    p90_mag = float(np.percentile(mag, 90)) if mag.size else 0.0
    std_mag = float(np.std(mag)) if mag.size else 0.0

    abs_fx = np.abs(fx)
    abs_fy = np.abs(fy)
    denom = abs_fx + abs_fy + 1e-6
    vertical_ratio = float(np.mean(abs_fy / denom)) if denom.size else 0.0
    moving_ratio = float(np.mean(mag >= min_mag_px)) if mag.size else 0.0

    return {
        "mean_mag": mean_mag,
        "p90_mag": p90_mag,
        "std_mag": std_mag,
        "vertical_ratio": vertical_ratio,
        "moving_ratio": moving_ratio,
    }


def _flow_motion_state(mean_mag: float, low: float, active: float, heavy: float) -> str:
    if mean_mag < low:
        return "still"
    if mean_mag < active:
        return "light"
    if mean_mag < heavy:
        return "active"
    return "heavy"


def map_to_baseline(label: str) -> str:
    l = label.strip().lower()
    if l in {"manual_working", "drilling_operation", "pipe_handling", "working"}:
        return "working"
    if l in {"falling", "lying", "accident"}:
        return "accident"
    if l in {
        "standing",
        "walking",
        "running",
        "sitting",
        "sleeping",
        "crouching",
        "bending",
        "not_working",
    }:
        return "not_working"
    return l


def main() -> None:
    args = parse_args()

    ckpt = torch.load(Path(args.activity_model), map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {int(k): v for k, v in ckpt.get("idx_to_label", {}).items()}
    if not idx_to_label:
        idx_to_label = {v: k for k, v in label_to_idx.items()}
    cfg = ckpt["config"]

    window_size = args.window_size if args.window_size > 0 else int(cfg.get("window_size", 32))
    min_kpt_conf = float(cfg.get("min_kpt_conf", 0.2))
    kpt_count = int(cfg["kpt_count"])
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_layers"])
    dropout = float(cfg["dropout"])
    num_classes = len(label_to_idx)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = LSTMActivityClassifier(
        kpt_count=kpt_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pose_model = YOLO(args.pose_model)
    use_video_file = bool(args.source.strip())
    source_value: str | int = args.source if use_video_file else args.camera
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        if use_video_file:
            raise RuntimeError(f"Could not open video source: {args.source}")
        raise RuntimeError(f"Could not open webcam index {args.camera}. Try --camera 0 or another index.")

    if not use_video_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    writer = None
    out_path: Path | None = None
    if args.save_output.strip():
        out_path = Path(args.save_output)
    elif use_video_file and args.save_video:
        source_path = Path(args.source)
        out_path = Path("outputs") / f"{source_path.stem}_realtime_pose_activity_lstm.mp4"

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, source_fps, (frame_w, frame_h))
        print(f"Saving annotated output to: {out_path.as_posix()}")

    histories: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=window_size))
    ema_probs: dict[int, np.ndarray] = {}
    stable_label: dict[int, str] = {}
    candidate_label: dict[int, str] = {}
    candidate_count: dict[int, int] = defaultdict(int)
    last_seen: dict[int, int] = {}
    instability_hist: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=12))
    prev_norm_frame: dict[int, np.ndarray] = {}
    flow_hist: dict[int, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=10))
    prev_gray: np.ndarray | None = None

    frame_idx = 0
    prev = time.perf_counter()
    window_name = "Realtime Drilling Activity (LSTM) - q/Esc to quit"
    missing_offset = 100000

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if use_video_file:
                    print("Reached end of video.")
                else:
                    print("Failed to read frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_field: np.ndarray | None = None
            if not args.disable_optical_flow and prev_gray is not None:
                flow_field = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )

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
            ready = 0

            for t in tracks:
                pid = int(t["person_id"])
                if pid < 0:
                    pid = missing_offset + int(t["idx"])
                active_ids.add(pid)
                last_seen[pid] = frame_idx

                norm = normalize_frame(t["keypoints"], bbox_h=t["bbox_h"], min_kpt_conf=min_kpt_conf)
                if norm.shape[0] != kpt_count:
                    # Skip inconsistent skeleton sizes to keep model input stable.
                    continue
                histories[pid].append(norm)
                h_len = len(histories[pid])
                instability_ratio = 1.0

                prev_n = prev_norm_frame.get(pid)
                if prev_n is not None and prev_n.shape == norm.shape:
                    delta = np.linalg.norm(norm[:, :2] - prev_n[:, :2], axis=1)
                    conf_mask = norm[:, 2] >= min_kpt_conf
                    if np.any(conf_mask):
                        step_motion = float(np.mean(delta[conf_mask]))
                    else:
                        step_motion = float(np.mean(delta))
                    instability_hist[pid].append(step_motion)
                prev_norm_frame[pid] = norm.copy()

                if len(instability_hist[pid]) >= 4:
                    hist = np.asarray(instability_hist[pid], dtype=np.float32)
                    m = float(np.mean(hist))
                    s = float(np.std(hist))
                    instability_ratio = (m + s) / max(m, 1e-4)

                label = "warming_up"
                general_label = "warming_up"
                conf = 0.0
                warm_ratio = min(1.0, h_len / max(float(window_size), 1.0))

                if h_len >= max(args.min_live_frames, 2):
                    ready += 1
                    seq = list(histories[pid])
                    if h_len < window_size:
                        pad = [seq[0]] * (window_size - h_len)
                        seq = pad + seq
                    x = np.stack(seq[-window_size:], axis=0).astype(np.float32)
                    with torch.no_grad():
                        x_t = torch.tensor(x[None, ...].tolist(), dtype=torch.float32, device=device)
                        logits = model(x_t)
                        probs = np.asarray(
                            torch.softmax(logits, dim=1).cpu().tolist()[0],
                            dtype=np.float32,
                        )

                    if pid not in ema_probs:
                        ema_probs[pid] = probs.copy()
                    else:
                        ema_probs[pid] = (
                            args.ema_alpha * probs + (1.0 - args.ema_alpha) * ema_probs[pid]
                        )
                    smooth = ema_probs[pid]
                    top_idx = int(np.argmax(smooth))
                    top_conf = float(smooth[top_idx])

                    raw_detail = idx_to_label.get(top_idx, str(top_idx))
                    if top_conf < args.unknown_threshold:
                        raw_detail = "uncertain"
                    raw_general = map_to_baseline(raw_detail)

                    # Accident gating: require unstable/spiky motion and enough confidence.
                    if raw_general == "accident":
                        if instability_ratio < args.accident_min_instability or top_conf < args.accident_min_conf:
                            raw_general = "not_working"
                            if raw_detail in {"falling", "lying", "accident"}:
                                raw_detail = "uncertain"

                    flow_feat = _flow_roi_features(
                        flow=flow_field,
                        bbox=t["bbox"],
                        min_mag_px=args.flow_min_mag_px,
                    )
                    flow_hist[pid].append(flow_feat)
                    flow_mean = float(np.mean([f["mean_mag"] for f in flow_hist[pid]]))
                    flow_p90 = float(np.mean([f["p90_mag"] for f in flow_hist[pid]]))
                    flow_vertical = float(np.mean([f["vertical_ratio"] for f in flow_hist[pid]]))
                    flow_state = _flow_motion_state(
                        mean_mag=flow_mean,
                        low=args.flow_low_threshold,
                        active=args.flow_active_threshold,
                        heavy=args.flow_heavy_threshold,
                    )

                    # Optical-flow extra layer:
                    # 1) suppress accident if movement is not bursty/vertical enough.
                    # 2) promote heavy-motion hints for high-movement cases.
                    if not args.disable_optical_flow:
                        if raw_general == "accident":
                            if flow_p90 < args.flow_accident_p90_threshold or flow_vertical < args.flow_accident_vertical_ratio:
                                raw_general = "not_working"
                                raw_detail = "uncertain" if args.label_scope == "fine" else raw_detail

                        if args.label_scope == "fine":
                            if raw_general != "accident" and flow_state == "heavy":
                                if raw_detail in {"standing", "sitting", "crouching"}:
                                    raw_detail = "walking"

                    raw_label = raw_general if args.label_scope == "baseline" else raw_detail

                    prev_cand = candidate_label.get(pid)
                    if prev_cand == raw_label:
                        candidate_count[pid] += 1
                    else:
                        candidate_label[pid] = raw_label
                        candidate_count[pid] = 1

                    if candidate_count[pid] >= max(1, args.decision_hold):
                        stable_label[pid] = raw_label

                    label = stable_label.get(pid, raw_label)
                    general_label = map_to_baseline(label) if args.label_scope == "fine" else label
                    conf = top_conf
                else:
                    flow_feat = _flow_roi_features(
                        flow=flow_field,
                        bbox=t["bbox"],
                        min_mag_px=args.flow_min_mag_px,
                    )
                    flow_hist[pid].append(flow_feat)
                    flow_mean = float(np.mean([f["mean_mag"] for f in flow_hist[pid]]))
                    flow_state = _flow_motion_state(
                        mean_mag=flow_mean,
                        low=args.flow_low_threshold,
                        active=args.flow_active_threshold,
                        heavy=args.flow_heavy_threshold,
                    )
                    if flow_state == "heavy":
                        general_label = "working"
                    elif flow_state in {"light", "active"}:
                        general_label = "not_working"
                    else:
                        general_label = "warming_up"

                if label not in {"warming_up", "uncertain"} and not args.hide_general_state:
                    # Surface motion hint without changing classification text.
                    if flow_hist[pid]:
                        flow_mean_hint = float(np.mean([f["mean_mag"] for f in flow_hist[pid]]))
                        flow_state_hint = _flow_motion_state(
                            mean_mag=flow_mean_hint,
                            low=args.flow_low_threshold,
                            active=args.flow_active_threshold,
                            heavy=args.flow_heavy_threshold,
                        )
                        general_label = f"{general_label} | move:{flow_state_hint}"

                _draw_track_card(
                    image=annotated,
                    pid=pid,
                    bbox=t["bbox"],
                    label=label,
                    general_label=general_label,
                    conf=conf,
                    warm_ratio=warm_ratio,
                    instability=instability_ratio,
                    show_general=not args.hide_general_state,
                )

            stale = [
                pid
                for pid, seen in last_seen.items()
                if frame_idx - seen > args.max_missing_frames
            ]
            for pid in stale:
                histories.pop(pid, None)
                ema_probs.pop(pid, None)
                stable_label.pop(pid, None)
                candidate_label.pop(pid, None)
                candidate_count.pop(pid, None)
                last_seen.pop(pid, None)
                instability_hist.pop(pid, None)
                prev_norm_frame.pop(pid, None)
                flow_hist.pop(pid, None)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            _draw_header(annotated, fps=fps, tracks=len(active_ids), ready=ready, window_size=window_size)

            cv2.imshow(window_name, annotated)
            if writer is not None:
                writer.write(annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            frame_idx += 1
            prev_gray = gray
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
