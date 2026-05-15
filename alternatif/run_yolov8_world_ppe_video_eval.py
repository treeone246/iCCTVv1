import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOv8-World PPE detection on a video and generate "
            "detection-based evaluation metrics."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-world-ppe.pt",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Input video path. If omitted, the newest .mp4 in videos/ is used."
        ),
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Path for annotated video output.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path for evaluation JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path for per-frame CSV output.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device, for example "cpu", "0", "0,1".',
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frame count for quick tests.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 means all frames).",
    )
    return parser.parse_args()


def newest_mp4(videos_dir: Path) -> Path:
    mp4s = sorted(videos_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 video found in {videos_dir.as_posix()}")
    return mp4s[0]


def resolve_model_path(model_arg: str) -> Path:
    candidate = Path(model_arg)
    if candidate.exists():
        return candidate

    in_script_dir = SCRIPT_DIR / model_arg
    if in_script_dir.exists():
        return in_script_dir

    in_project_root = PROJECT_ROOT / model_arg
    if in_project_root.exists():
        return in_project_root

    raise FileNotFoundError(
        f"Model not found: {model_arg}. Checked: "
        f"{candidate.as_posix()}, {in_script_dir.as_posix()}, {in_project_root.as_posix()}"
    )


def resolve_video_path(video_arg: Optional[str]) -> Path:
    if video_arg:
        p = Path(video_arg)
        if p.exists():
            return p
        in_videos = PROJECT_ROOT / "videos" / video_arg
        if in_videos.exists():
            return in_videos
        raise FileNotFoundError(
            f"Video not found: {video_arg}. Also checked {in_videos.as_posix()}"
        )
    return newest_mp4(PROJECT_ROOT / "videos")


def safe_name_lookup(names: Dict[int, str], class_id: int) -> str:
    return names.get(class_id, f"class_{class_id}")


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    model_path = resolve_model_path(args.model)

    video_path = resolve_video_path(args.video)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = (
        Path(args.output_video)
        if args.output_video
        else output_dir / f"{video_path.stem}_yolov8s_world_ppe.mp4"
    )
    output_json = (
        Path(args.output_json)
        if args.output_json
        else output_dir / f"{video_path.stem}_yolov8s_world_ppe_eval.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else output_dir / f"{video_path.stem}_yolov8s_world_ppe_frame_metrics.csv"
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else {}

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

    processed_frames = 0
    sampled_frames = 0
    frames_with_detection = 0
    total_detections = 0
    conf_sum_all = 0.0

    per_class_det_count = Counter()
    per_class_conf_sum = defaultdict(float)
    per_class_frames_present = Counter()

    with output_csv.open("w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(
            [
                "frame_idx",
                "timestamp_sec",
                "num_detections",
                "avg_conf",
                "max_conf",
                "class_counts",
            ]
        )

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if processed_frames % args.frame_step != 0:
                    processed_frames += 1
                    continue

                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                result = results[0]
                boxes = result.boxes
                frame_num_dets = 0
                frame_avg_conf = 0.0
                frame_max_conf = 0.0
                frame_class_counts = Counter()

                if boxes is not None and len(boxes) > 0:
                    cls_ids = boxes.cls.tolist()
                    confs = boxes.conf.tolist()

                    frame_num_dets = len(cls_ids)
                    frame_avg_conf = sum(confs) / frame_num_dets
                    frame_max_conf = max(confs)

                    for cls_val, conf_val in zip(cls_ids, confs):
                        class_id = int(cls_val)
                        class_name = safe_name_lookup(names, class_id)
                        frame_class_counts[class_name] += 1
                        per_class_det_count[class_name] += 1
                        per_class_conf_sum[class_name] += float(conf_val)

                    for class_name in frame_class_counts:
                        per_class_frames_present[class_name] += 1

                    frames_with_detection += 1
                    total_detections += frame_num_dets
                    conf_sum_all += sum(confs)

                annotated = result.plot()
                writer.write(annotated)

                timestamp_sec = processed_frames / fps
                class_counts_text = ";".join(
                    f"{k}:{v}" for k, v in sorted(frame_class_counts.items())
                )
                csv_writer.writerow(
                    [
                        processed_frames,
                        f"{timestamp_sec:.3f}",
                        frame_num_dets,
                        f"{frame_avg_conf:.4f}",
                        f"{frame_max_conf:.4f}",
                        class_counts_text,
                    ]
                )

                sampled_frames += 1
                processed_frames += 1
                if args.max_frames is not None and sampled_frames >= args.max_frames:
                    break
                if sampled_frames % 30 == 0:
                    if frame_count > 0:
                        print(
                            f"Processed {sampled_frames} sampled frames "
                            f"from {processed_frames}/{frame_count} source frames..."
                        )
                    else:
                        print(
                            f"Processed {sampled_frames} sampled frames "
                            f"from {processed_frames} source frames..."
                        )
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

    avg_dets_per_frame = (
        float(total_detections) / sampled_frames if sampled_frames > 0 else 0.0
    )
    detection_coverage = (
        float(frames_with_detection) / sampled_frames if sampled_frames > 0 else 0.0
    )
    avg_conf_all = float(conf_sum_all) / total_detections if total_detections > 0 else 0.0

    # This is a proxy score, not true accuracy, because no frame-level ground truth is provided.
    proxy_accuracy = detection_coverage * avg_conf_all

    per_class_metrics = {}
    for class_name, det_count in per_class_det_count.items():
        per_class_metrics[class_name] = {
            "detections": int(det_count),
            "avg_confidence": (
                per_class_conf_sum[class_name] / det_count if det_count > 0 else 0.0
            ),
            "frames_present": int(per_class_frames_present[class_name]),
            "frame_presence_rate": (
                per_class_frames_present[class_name] / sampled_frames
                if sampled_frames > 0
                else 0.0
            ),
        }

    summary = {
        "video_path": str(video_path.as_posix()),
        "model_path": str(model_path.as_posix()),
        "thresholds": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        },
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "source_frame_count": frame_count,
            "sampled_frame_step": args.frame_step,
            "processed_source_frames": processed_frames,
            "processed_sampled_frames": sampled_frames,
            "sampled_duration_sec": (processed_frames / fps if fps > 0 else 0.0),
        },
        "detection_metrics": {
            "total_detections": int(total_detections),
            "frames_with_detection": int(frames_with_detection),
            "detection_coverage_rate": detection_coverage,
            "avg_detections_per_frame": avg_dets_per_frame,
            "avg_confidence_all_detections": avg_conf_all,
        },
        "evaluation_note": (
            "No ground-truth labels were supplied for this video. "
            "proxy_accuracy_score is detection-based and not mAP/true accuracy. "
            "Metrics are computed on sampled frames if frame_step > 1."
        ),
        "proxy_accuracy_score": proxy_accuracy,
        "proxy_accuracy_percent": proxy_accuracy * 100.0,
        "per_class_metrics": per_class_metrics,
        "outputs": {
            "annotated_video": str(output_video.as_posix()),
            "per_frame_csv": str(output_csv.as_posix()),
            "evaluation_json": str(output_json.as_posix()),
        },
    }

    with output_json.open("w", encoding="utf-8") as f_json:
        json.dump(summary, f_json, indent=2)

    print("Done.")
    print(f"Video input: {video_path.as_posix()}")
    print(f"Annotated video: {output_video.as_posix()}")
    print(f"Per-frame CSV: {output_csv.as_posix()}")
    print(f"Evaluation JSON: {output_json.as_posix()}")
    print(f"Processed source frames: {processed_frames}")
    print(f"Processed sampled frames: {sampled_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Proxy accuracy (%): {summary['proxy_accuracy_percent']:.2f}")


if __name__ == "__main__":
    main()
