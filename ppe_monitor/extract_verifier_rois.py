"""Extract verifier ROI crops from video using the current pipeline crop logic.

This tool uses the same item-specific ROI crop function as runtime
(`MonitoringPipeline._crop_for_item`) so the exported crops match what the
verifier actually sees online.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import cv2
import yaml

from app.pipeline import MonitoringPipeline
from app.schemas import Classification
from app.startup_check import load_runtime_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract verifier ROI crops from video with pipeline-consistent ROI logic."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path.")
    parser.add_argument("--output-dir", type=str, default="verifier_eval_data_auto", help="Output dataset root.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "tentative", "non_compliant"],
        default="all",
        help="all: save every item ROI; tentative: only base VIOLATION_TENTATIVE; non_compliant: VIOLATION or VIOLATION_TENTATIVE.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many decoded frames (0 = full video).")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="Saved crop JPEG quality.")
    parser.add_argument("--min-side-px", type=int, default=24, help="Skip tiny crops.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name tag. If omitted, uses timestamp like run_20260515_153000.",
    )
    parser.add_argument(
        "--bucket-by",
        type=str,
        choices=["base", "final"],
        default="base",
        help="Choose whether output folders are based on base association classification or final classification after verifier.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def should_save(mode: str, base_cls: Classification) -> bool:
    if mode == "all":
        return True
    if mode == "tentative":
        return base_cls == Classification.VIOLATION_TENTATIVE
    return base_cls in (Classification.VIOLATION, Classification.VIOLATION_TENTATIVE)


def bucket_for(base_cls: Classification) -> str:
    if base_cls == Classification.COMPLIANT:
        return "compliant_candidate"
    if base_cls == Classification.INDETERMINATE:
        return "indeterminate_candidate"
    return "violation_candidate"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    video_path = Path(args.video).resolve()
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()
    run_name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_name = run_name.replace(" ", "_")
    out_root = out_root / run_name

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path.as_posix()}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.as_posix()}")

    config = load_config(config_path)
    runtime = load_runtime_components(config, project_root)
    pipeline = MonitoringPipeline(
        pose_tracker=runtime.pose_tracker,
        ppe_detector=runtime.ppe_detector,
        verifier=runtime.verifier,
        config=config,
    )

    items: List[str] = list(config.get("required_ppe", []))
    if not items:
        raise ValueError("No required_ppe items configured.")

    for item in items:
        for bucket in ("compliant_candidate", "violation_candidate", "indeterminate_candidate"):
            (out_root / item / bucket).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path.as_posix()}")

    meta_path = out_root / "metadata.jsonl"
    written = 0
    seen_frames = 0
    processed_frames = 0
    skipped_by_mode = 0
    skipped_tiny = 0

    try:
        with meta_path.open("w", encoding="utf-8") as meta:
            frame_id = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                seen_frames += 1

                if args.max_frames > 0 and seen_frames > args.max_frames:
                    break

                if args.frame_step > 1 and (frame_id % args.frame_step != 0):
                    frame_id += 1
                    continue

                tracked_people = runtime.pose_tracker.track(frame)
                ppe_detections = runtime.ppe_detector.detect(frame)
                det_dicts: List[Dict[str, object]] = [
                    {"label": det.label, "bbox": det.bbox, "conf": det.conf}
                    for det in ppe_detections
                ]

                for person in tracked_people:
                    for item in items:
                        base_cls, bind = pipeline.association.classify_item(
                            item=item,
                            keypoints=person.keypoints,
                            keypoint_confidences=person.keypoint_confidences,
                            ppe_detections=det_dicts,
                            frame_shape=frame.shape,
                        )
                        if not should_save(args.mode, base_cls):
                            skipped_by_mode += 1
                            continue

                        crop = pipeline._crop_for_item(frame, person, item)
                        h, w = crop.shape[:2]
                        if h < args.min_side_px or w < args.min_side_px:
                            skipped_tiny += 1
                            continue

                        final_cls = pipeline._classify_person_item(person, ppe_detections, item, frame)
                        bucket_cls = base_cls if args.bucket_by == "base" else final_cls
                        bucket = bucket_for(bucket_cls)
                        out_dir = out_root / item / bucket
                        filename = (
                            f"{run_name}_f{frame_id:06d}_p{person.person_id:04d}_"
                            f"{item}_{base_cls.value.lower()}_{written:07d}.jpg"
                        )
                        out_path = out_dir / filename
                        ok_write = cv2.imwrite(
                            out_path.as_posix(),
                            crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                        )
                        if not ok_write:
                            continue

                        record = {
                            "file": out_path.relative_to(out_root).as_posix(),
                            "frame_id": frame_id,
                            "person_id": person.person_id,
                            "item": item,
                            "base_classification": base_cls.value,
                            "final_classification": final_cls.value,
                            "bucket_by": args.bucket_by,
                            "bucket_classification": bucket_cls.value,
                            "run_name": run_name,
                            "bbox_person": [float(v) for v in person.bbox],
                            "crop_shape": [int(h), int(w)],
                            "bind": {
                                "bound": bool(bind.bound),
                                "held": bool(bind.held),
                                "confidence": float(bind.confidence),
                                "reason": str(bind.reason),
                            }
                            if bind is not None
                            else None,
                        }
                        meta.write(json.dumps(record) + "\n")
                        written += 1

                processed_frames += 1
                frame_id += 1
    finally:
        cap.release()

    print("ROI extraction complete")
    print(f"Video: {video_path.as_posix()}")
    print(f"Output: {out_root.as_posix()}")
    print(f"Run name: {run_name}")
    print(f"Frames seen: {seen_frames}")
    print(f"Frames processed: {processed_frames}")
    print(f"Crops written: {written}")
    print(f"Skipped by mode: {skipped_by_mode}")
    print(f"Skipped tiny crops: {skipped_tiny}")
    print(f"Metadata: {meta_path.as_posix()}")


if __name__ == "__main__":
    main()
