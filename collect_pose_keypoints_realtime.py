import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-time pose + tracking collector for activity/state-model training. "
            "Saves JSONL, CSV, and NPY from one recording session."
        )
    )
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable tracking and run pose prediction only.",
    )
    parser.add_argument(
        "--activity-label",
        type=str,
        required=True,
        help='Activity/state label for this recording (example: "running").',
    )
    parser.add_argument("--subject-id", type=str, default="subject_00")
    parser.add_argument("--camera-id", type=str, default="cam_00")
    parser.add_argument(
        "--output-root",
        type=str,
        default="datasets/pose_sessions",
        help="Root folder where session folders are created.",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Optional explicit session folder name.",
    )
    parser.add_argument(
        "--exports",
        type=str,
        nargs="+",
        choices=["jsonl", "csv", "npy"],
        default=["jsonl", "csv", "npy"],
        help="Output formats to save.",
    )
    parser.add_argument(
        "--save-annotated-video",
        action="store_true",
        help="Save plotted output to annotated.mp4 in the session folder.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.0,
        help="Annotated video fps override. 0 means auto from camera.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. 0 means unlimited until q/Esc.",
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
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _to_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        out = int(float(value))
        return out
    except Exception:
        return default


def _safe_session_name(activity_label: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", activity_label.strip().lower())
    safe_label = safe_label.strip("_") or "unknown"
    return f"{safe_label}_{stamp}"


def _resolve_label(class_names: Any, cls_id: int) -> str:
    if class_names is None:
        return str(cls_id)
    if isinstance(class_names, dict):
        return str(class_names.get(cls_id, cls_id))
    if isinstance(class_names, (list, tuple)):
        if 0 <= cls_id < len(class_names):
            return str(class_names[cls_id])
    return str(cls_id)


def _extract_pose_records(
    result: Any,
    frame_index: int,
    timestamp: float,
    frame_width: int,
    frame_height: int,
    activity_label: str,
    subject_id: str,
    camera_id: str,
    session_name: str,
) -> list[dict[str, Any]]:
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

    kpt_xyn = _to_numpy(getattr(keypoints_obj, "xyn", None))
    if kpt_xyn is not None and kpt_xyn.ndim == 2:
        kpt_xyn = np.expand_dims(kpt_xyn, axis=0)

    boxes_xyxy = _to_numpy(getattr(boxes_obj, "xyxy", None))
    boxes_conf = _to_numpy(getattr(boxes_obj, "conf", None))
    boxes_cls = _to_numpy(getattr(boxes_obj, "cls", None))
    boxes_id = _to_numpy(getattr(boxes_obj, "id", None))

    class_names = getattr(result, "names", None)
    n_person, n_keypoints = kpt_xy.shape[0], kpt_xy.shape[1]
    records: list[dict[str, Any]] = []

    for i in range(n_person):
        if boxes_xyxy is not None and i < len(boxes_xyxy):
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i].tolist()]
        else:
            xs = kpt_xy[i, :, 0]
            ys = kpt_xy[i, :, 1]
            x1, y1 = float(np.min(xs)), float(np.min(ys))
            x2, y2 = float(np.max(xs)), float(np.max(ys))

        bbox_w = max(x2 - x1, 0.0)
        bbox_h = max(y2 - y1, 0.0)

        model_conf = (
            _to_float(boxes_conf[i], default=0.0)
            if boxes_conf is not None and i < len(boxes_conf)
            else float(np.mean(kpt_conf[i]))
        )
        class_id = (
            _to_int(boxes_cls[i], default=0)
            if boxes_cls is not None and i < len(boxes_cls)
            else 0
        )
        class_label = _resolve_label(class_names, class_id)

        person_id = (
            _to_int(boxes_id[i], default=-1)
            if boxes_id is not None and i < len(boxes_id)
            else -1
        )

        keypoints_list: list[dict[str, float | int]] = []
        for k in range(n_keypoints):
            x = _to_float(kpt_xy[i, k, 0], default=0.0)
            y = _to_float(kpt_xy[i, k, 1], default=0.0)
            c = _to_float(kpt_conf[i, k], default=0.0)

            if kpt_xyn is not None and i < len(kpt_xyn):
                xn = _to_float(kpt_xyn[i, k, 0], default=x / max(frame_width, 1))
                yn = _to_float(kpt_xyn[i, k, 1], default=y / max(frame_height, 1))
            else:
                xn = x / max(frame_width, 1)
                yn = y / max(frame_height, 1)

            keypoints_list.append(
                {
                    "kpt_index": k,
                    "x": x,
                    "y": y,
                    "xn": xn,
                    "yn": yn,
                    "conf": c,
                }
            )

        records.append(
            {
                "session_name": session_name,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "activity_label": activity_label,
                "subject_id": subject_id,
                "camera_id": camera_id,
                "person_id": person_id,
                "class_id": class_id,
                "class_label": class_label,
                "model_confidence": model_conf,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": bbox_w,
                    "h": bbox_h,
                },
                "keypoints": keypoints_list,
            }
        )

    return records


def _write_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    max_k = 0
    for rec in records:
        max_k = max(max_k, len(rec.get("keypoints", [])))

    headers = [
        "session_name",
        "frame_index",
        "timestamp",
        "activity_label",
        "subject_id",
        "camera_id",
        "person_id",
        "class_id",
        "class_label",
        "model_confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_w",
        "bbox_h",
        "n_keypoints",
    ]
    for k in range(max_k):
        prefix = f"kp{k:02d}"
        headers.extend(
            [
                f"{prefix}_x",
                f"{prefix}_y",
                f"{prefix}_xn",
                f"{prefix}_yn",
                f"{prefix}_conf",
            ]
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            row: dict[str, Any] = {
                "session_name": rec["session_name"],
                "frame_index": rec["frame_index"],
                "timestamp": rec["timestamp"],
                "activity_label": rec["activity_label"],
                "subject_id": rec["subject_id"],
                "camera_id": rec["camera_id"],
                "person_id": rec["person_id"],
                "class_id": rec["class_id"],
                "class_label": rec["class_label"],
                "model_confidence": rec["model_confidence"],
                "bbox_x1": rec["bbox"]["x1"],
                "bbox_y1": rec["bbox"]["y1"],
                "bbox_x2": rec["bbox"]["x2"],
                "bbox_y2": rec["bbox"]["y2"],
                "bbox_w": rec["bbox"]["w"],
                "bbox_h": rec["bbox"]["h"],
                "n_keypoints": len(rec["keypoints"]),
            }
            for kp in rec["keypoints"]:
                prefix = f"kp{int(kp['kpt_index']):02d}"
                row[f"{prefix}_x"] = kp["x"]
                row[f"{prefix}_y"] = kp["y"]
                row[f"{prefix}_xn"] = kp["xn"]
                row[f"{prefix}_yn"] = kp["yn"]
                row[f"{prefix}_conf"] = kp["conf"]
            writer.writerow(row)


def _write_npy(records: list[dict[str, Any]], npy_path: Path) -> None:
    if not records:
        bundle = {
            "features": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype="<U1"),
            "person_ids": np.empty((0,), dtype=np.int32),
            "frame_indices": np.empty((0,), dtype=np.int32),
            "sessions": np.empty((0,), dtype="<U1"),
            "keypoint_count": np.asarray([0], dtype=np.int32),
        }
        np.save(npy_path, bundle, allow_pickle=True)
        return

    keypoint_count = max(len(rec["keypoints"]) for rec in records)
    feature_size = 6 + (keypoint_count * 5)
    features = np.full((len(records), feature_size), np.nan, dtype=np.float32)
    labels: list[str] = []
    person_ids: list[int] = []
    frame_indices: list[int] = []
    sessions: list[str] = []

    for i, rec in enumerate(records):
        bbox = rec["bbox"]
        base = [
            float(rec["model_confidence"]),
            float(bbox["x1"]),
            float(bbox["y1"]),
            float(bbox["x2"]),
            float(bbox["y2"]),
            float(rec["person_id"]),
        ]
        features[i, :6] = np.asarray(base, dtype=np.float32)

        for kp in rec["keypoints"]:
            k = int(kp["kpt_index"])
            start = 6 + (k * 5)
            features[i, start : start + 5] = np.asarray(
                [kp["x"], kp["y"], kp["xn"], kp["yn"], kp["conf"]], dtype=np.float32
            )

        labels.append(str(rec["activity_label"]))
        person_ids.append(int(rec["person_id"]))
        frame_indices.append(int(rec["frame_index"]))
        sessions.append(str(rec["session_name"]))

    bundle = {
        "features": features,
        "labels": np.asarray(labels),
        "person_ids": np.asarray(person_ids, dtype=np.int32),
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "sessions": np.asarray(sessions),
        "keypoint_count": np.asarray([keypoint_count], dtype=np.int32),
    }
    np.save(npy_path, bundle, allow_pickle=True)


def main() -> None:
    args = parse_args()

    session_name = args.session_name or _safe_session_name(args.activity_label)
    output_root = Path(args.output_root)
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = session_dir / "keypoints.jsonl"
    csv_path = session_dir / "keypoints.csv"
    npy_path = session_dir / "keypoints.npy"
    meta_path = session_dir / "session_meta.json"

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 0 or another index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)

    video_writer = None
    if args.save_annotated_video:
        out_fps = args.video_fps
        if out_fps <= 0:
            out_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(session_dir / "annotated.mp4"),
            fourcc,
            out_fps,
            (actual_w, actual_h),
        )

    window_name = "YOLO26 Pose Keypoint Collector (q/Esc to stop)"
    frame_index = 0
    record_count = 0
    all_records: list[dict[str, Any]] = []
    start_ts = time.time()
    prev_time = time.perf_counter()

    jsonl_file = None
    try:
        if "jsonl" in args.exports:
            jsonl_file = jsonl_path.open("w", encoding="utf-8")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            ts = time.time()
            if args.no_track:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    classes=args.classes,
                    verbose=False,
                )
            else:
                results = model.track(
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
            records = _extract_pose_records(
                result=result,
                frame_index=frame_index,
                timestamp=ts,
                frame_width=actual_w,
                frame_height=actual_h,
                activity_label=args.activity_label,
                subject_id=args.subject_id,
                camera_id=args.camera_id,
                session_name=session_name,
            )

            if records:
                all_records.extend(records)
                record_count += len(records)
                if jsonl_file is not None:
                    for rec in records:
                        jsonl_file.write(json.dumps(rec) + "\n")

            annotated = result.plot()
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  Frame: {frame_index}  Records: {record_count}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Session: {session_name}  Label: {args.activity_label}",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)
            if video_writer is not None:
                video_writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if jsonl_file is not None:
            jsonl_file.close()
        cv2.destroyAllWindows()

    if "csv" in args.exports:
        _write_csv(all_records, csv_path)
    if "npy" in args.exports:
        _write_npy(all_records, npy_path)

    end_ts = time.time()
    meta = {
        "session_name": session_name,
        "model": args.model,
        "activity_label": args.activity_label,
        "subject_id": args.subject_id,
        "camera_id": args.camera_id,
        "camera_index": args.camera,
        "tracking_enabled": not args.no_track,
        "tracker": None if args.no_track else args.tracker,
        "frame_width": actual_w,
        "frame_height": actual_h,
        "frames_processed": frame_index + 1,
        "records_saved": record_count,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "duration_sec": max(end_ts - start_ts, 0.0),
        "exports": args.exports,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Session saved to: {session_dir}")
    print(f"Total records: {record_count}")
    for fmt, path in [
        ("jsonl", jsonl_path),
        ("csv", csv_path),
        ("npy", npy_path),
        ("meta", meta_path),
    ]:
        if fmt == "meta" or fmt in args.exports:
            print(f"- {fmt}: {path}")


if __name__ == "__main__":
    main()
