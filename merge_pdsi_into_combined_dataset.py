import argparse
import json
import re
import shutil
import os
from pathlib import Path


TARGET_CLASSES = [
    "Coveralls",
    "helmet/hardhat",
    "boots/shoes",
    "mask",
    "gloves",
    "goggles/glasses",
]

# PDSI class index -> unified target class index
PDSI_TO_TARGET = {
    0: 2,   # boots -> boots/shoes
    1: 4,   # gloves -> gloves
    2: 5,   # goggles -> goggles/glasses
    3: 1,   # helmet -> helmet/hardhat
    5: 0,   # coverall -> Coveralls
    # 4 and 10 are empty in source
    # 6..11 are "no-*" negatives and are skipped
}

NEGATIVE_OR_SKIP = {4, 6, 7, 8, 9, 10, 11}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge PDSI_Data into combined_ppe_selected.yolo26 and remap labels "
            "to the unified 6-class schema."
        )
    )
    parser.add_argument(
        "--pdsi-root",
        type=str,
        default="YOLO26_DATASETS/PDSI_Data",
        help="Path to PDSI_Data folder.",
    )
    parser.add_argument(
        "--combined-root",
        type=str,
        default="YOLO26_DATASETS/combined_ppe_selected.yolo26",
        help="Path to existing combined dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Target split in combined dataset for imported PDSI samples.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pdsi",
        help="Prefix for output file names to avoid collisions.",
    )
    parser.add_argument(
        "--copy-empty-label-images",
        action="store_true",
        help="Keep images even when all boxes map to skipped classes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview counts only; do not copy/write files.",
    )
    parser.add_argument(
        "--image-mode",
        type=str,
        choices=["hardlink", "copy"],
        default="hardlink",
        help=(
            "How to place images into combined dataset. "
            "'hardlink' avoids duplicating disk usage on same volume."
        ),
    )
    return parser.parse_args()


def _find_image_by_task_id(images_dir: Path, task_id: int) -> Path | None:
    stem = str(task_id)
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _parse_class_index(result_item: dict) -> int | None:
    value = result_item.get("value") or {}
    cls_raw = value.get("class_index")
    if cls_raw is not None:
        try:
            return int(float(cls_raw))
        except Exception:
            pass

    labels = value.get("rectanglelabels") or []
    if labels:
        name = str(labels[0]).strip().lower()
        # Fallback by label text if class_index is missing/corrupted.
        if "coverall" in name:
            return 5
        if "helmet" in name:
            return 3
        if "boot" in name:
            return 0
        if "glove" in name:
            return 1
        if "goggle" in name or "glass" in name:
            return 2
        if re.search(r"\bno[\s\-_]*", name):
            return 6
    return None


def _pick_annotation(task: dict) -> dict | None:
    anns = task.get("annotations") or []
    if not anns:
        return None

    valid = [a for a in anns if not a.get("was_cancelled", False)]
    if not valid:
        valid = anns

    # Prefer latest updated annotation when multiple exist.
    valid.sort(key=lambda a: str(a.get("updated_at", "")))
    return valid[-1]


def _results_to_yolo_rows(results: list[dict]) -> tuple[list[str], dict[str, int]]:
    rows: list[str] = []
    stats = {
        "all_boxes": 0,
        "mapped_boxes": 0,
        "skipped_negative_or_empty": 0,
        "skipped_unmapped": 0,
        "skipped_bad_geom": 0,
    }

    for r in results:
        if r.get("type") != "rectanglelabels":
            continue
        value = r.get("value") or {}
        stats["all_boxes"] += 1

        src_cls = _parse_class_index(r)
        if src_cls is None:
            stats["skipped_unmapped"] += 1
            continue
        if src_cls in NEGATIVE_OR_SKIP:
            stats["skipped_negative_or_empty"] += 1
            continue

        dst_cls = PDSI_TO_TARGET.get(src_cls)
        if dst_cls is None:
            stats["skipped_unmapped"] += 1
            continue

        try:
            x = float(value.get("x"))
            y = float(value.get("y"))
            w = float(value.get("width"))
            h = float(value.get("height"))
        except Exception:
            stats["skipped_bad_geom"] += 1
            continue

        if w <= 0.0 or h <= 0.0:
            stats["skipped_bad_geom"] += 1
            continue

        xc = _clamp01((x + (w / 2.0)) / 100.0)
        yc = _clamp01((y + (h / 2.0)) / 100.0)
        wn = _clamp01(w / 100.0)
        hn = _clamp01(h / 100.0)

        if wn <= 0.0 or hn <= 0.0:
            stats["skipped_bad_geom"] += 1
            continue

        rows.append(f"{dst_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        stats["mapped_boxes"] += 1

    return rows, stats


def _ensure_yaml(combined_root: Path) -> None:
    yaml_path = combined_root / "data.yaml"
    if yaml_path.exists():
        return
    lines = [
        f"path: {combined_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(TARGET_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(TARGET_CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _place_image(src: Path, dst: Path, image_mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if image_mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    pdsi_root = Path(args.pdsi_root).resolve()
    combined_root = Path(args.combined_root).resolve()
    images_src = pdsi_root / "images"
    json_src = pdsi_root / "horus_dataset.json"

    if not images_src.exists():
        raise FileNotFoundError(f"Missing images folder: {images_src}")
    if not json_src.exists():
        raise FileNotFoundError(f"Missing annotation file: {json_src}")
    if not combined_root.exists():
        raise FileNotFoundError(f"Combined dataset not found: {combined_root}")

    out_img_dir = combined_root / "images" / args.split
    out_lbl_dir = combined_root / "labels" / args.split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    _ensure_yaml(combined_root)

    tasks = json.loads(json_src.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise RuntimeError("horus_dataset.json is expected to be a list of tasks.")

    summary = {
        "tasks_total": len(tasks),
        "images_found": 0,
        "images_missing": 0,
        "images_written": 0,
        "labels_written": 0,
        "empty_labels_written": 0,
        "all_boxes": 0,
        "mapped_boxes": 0,
        "skipped_negative_or_empty": 0,
        "skipped_unmapped": 0,
        "skipped_bad_geom": 0,
        "split": args.split,
        "dry_run": args.dry_run,
        "image_mode": args.image_mode,
    }

    for task in tasks:
        task_id = int(task.get("id"))
        img_path = _find_image_by_task_id(images_src, task_id)
        if img_path is None:
            summary["images_missing"] += 1
            continue
        summary["images_found"] += 1

        ann = _pick_annotation(task)
        results = [] if ann is None else (ann.get("result") or [])
        rows, stats = _results_to_yolo_rows(results)

        summary["all_boxes"] += stats["all_boxes"]
        summary["mapped_boxes"] += stats["mapped_boxes"]
        summary["skipped_negative_or_empty"] += stats["skipped_negative_or_empty"]
        summary["skipped_unmapped"] += stats["skipped_unmapped"]
        summary["skipped_bad_geom"] += stats["skipped_bad_geom"]

        if not rows and not args.copy_empty_label_images:
            continue

        out_stem = f"{args.prefix}__{task_id}"
        out_img = out_img_dir / f"{out_stem}{img_path.suffix.lower()}"
        out_lbl = out_lbl_dir / f"{out_stem}.txt"

        if not args.dry_run:
            _place_image(img_path, out_img, args.image_mode)
            out_lbl.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

        summary["images_written"] += 1
        summary["labels_written"] += 1
        if not rows:
            summary["empty_labels_written"] += 1

    report_path = combined_root / "merge_report_pdsi.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("PDSI merge complete.")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if not args.dry_run:
        print(f"report: {report_path.as_posix()}")


if __name__ == "__main__":
    main()
