import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TARGET_CLASSES = [
    "Coveralls",
    "helmet/hardhat",
    "boots/shoes",
    "mask",
    "gloves",
    "goggles/glasses",
]


@dataclass
class DatasetPaths:
    dataset_root: Path
    split_to_images: dict[str, Path]
    class_map: dict[int, int]  # source class id -> target class id
    source_names: dict[int, str]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple YOLO26 PPE datasets into one unified dataset with "
            "filtered/remapped classes and generated data.yaml."
        )
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default="YOLO26_DATASETS",
        help="Root folder containing source dataset subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="YOLO26_DATASETS/combined_ppe_selected.yolo26",
        help="Output folder for merged dataset.",
    )
    parser.add_argument(
        "--copy-empty-label-images",
        action="store_true",
        help=(
            "Also keep images whose labels become empty after class filtering "
            "(background-only samples)."
        ),
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output folder before writing merged dataset.",
    )
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("_", " ").replace("-", " ").replace("/", " ")
    value = re.sub(r"[^a-z0-9\s]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_negative_label(raw_name: str) -> bool:
    raw = raw_name.strip().lower()
    norm = _normalize_text(raw_name)
    if raw in {"none", "background"} or norm in {"none", "background"}:
        return True
    # "no_helmet", "no hard-hat", "without_coveralls", etc.
    if re.search(r"(^|[\s_\-])(no|without)([\s_\-]|$)", raw):
        return True
    if re.search(r"(^| )no( |$)", norm):
        return True
    if "without " in norm:
        return True
    return False


def _map_label_to_target(name: str) -> int | None:
    norm = _normalize_text(name)
    if not norm:
        return None
    if _is_negative_label(name):
        return None
    if "person" in norm or norm == "vest":
        return None

    if "coverall" in norm:
        return 0  # Coveralls
    if "helmet" in norm or "hardhat" in norm or "hard hat" in norm:
        return 1  # helmet/hardhat
    if "boot" in norm or "shoe" in norm:
        return 2  # boots/shoes
    if "mask" in norm:
        return 3  # mask
    if "glove" in norm:
        return 4  # gloves
    if "goggle" in norm or "glasses" in norm or "glass" in norm:
        return 5  # goggles/glasses

    return None


def _extract_names_from_text(text: str) -> dict[int, str]:
    out: dict[int, str] = {}

    # Pattern 1: "0: helmet"
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s*[:\-]\s*(.+?)\s*$", line)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip().strip("'\"")
        out[idx] = name

    if out:
        return out

    # Pattern 2: "names: ['a', 'b', ...]"
    m_list = re.search(r"names\s*:\s*(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL)
    if m_list:
        try:
            arr = ast.literal_eval(m_list.group(1))
            if isinstance(arr, list):
                return {i: str(v) for i, v in enumerate(arr)}
        except Exception:
            pass

    # Pattern 3: YAML-like block
    # names:
    #   0: helmet
    #   1: gloves
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*names\s*:\s*$", line, flags=re.IGNORECASE):
            start = i + 1
            break
    if start is not None:
        for line in lines[start:]:
            if not line.strip():
                continue
            if re.match(r"^\S", line):  # new top-level key
                break
            m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if m:
                out[int(m.group(1))] = m.group(2).strip().strip("'\"")
        if out:
            return out

    return {}


def _read_names(dataset_root: Path) -> dict[int, str]:
    readme_candidates = sorted(dataset_root.glob("labelledDataCustomREADME*"))
    if readme_candidates:
        text = readme_candidates[0].read_text(encoding="utf-8", errors="ignore")
        names = _extract_names_from_text(text)
        if names:
            return names

    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        text = data_yaml.read_text(encoding="utf-8", errors="ignore")
        names = _extract_names_from_text(text)
        if names:
            return names

    return {}


def _read_yaml_splits(dataset_root: Path) -> dict[str, str]:
    """
    Parse train/val/test from data.yaml with a lightweight regex parser.
    """
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        return {}
    text = data_yaml.read_text(encoding="utf-8", errors="ignore")
    result: dict[str, str] = {}
    for key in ("train", "val", "valid", "test"):
        m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("'\"")
            result[key] = value
    return result


def _resolve_split_path(dataset_root: Path, split_value: str) -> Path | None:
    """
    Resolve split path from data.yaml.
    Handles common Roboflow-style '../train/images' and local 'images/train' paths.
    """
    if not split_value:
        return None

    candidate = (dataset_root / split_value).resolve()
    if candidate.exists():
        return candidate

    # Some data.yaml use paths relative to parent with ../train/images.
    candidate2 = (dataset_root.parent / split_value).resolve()
    if candidate2.exists():
        return candidate2

    # Fallback: relative without leading ../
    cleaned = split_value.replace("../", "").replace("..\\", "")
    candidate3 = (dataset_root / cleaned).resolve()
    if candidate3.exists():
        return candidate3
    return None


def _discover_split_images(dataset_root: Path) -> dict[str, Path]:
    yaml_splits = _read_yaml_splits(dataset_root)
    split_to_images: dict[str, Path] = {}

    def set_if_exists(split_name: str, p: Path | None) -> None:
        if p is not None and p.exists():
            split_to_images[split_name] = p

    if yaml_splits:
        set_if_exists("train", _resolve_split_path(dataset_root, yaml_splits.get("train", "")))
        val_path = yaml_splits.get("val", yaml_splits.get("valid", ""))
        set_if_exists("val", _resolve_split_path(dataset_root, val_path))
        set_if_exists("test", _resolve_split_path(dataset_root, yaml_splits.get("test", "")))

    # Fallback discovery for non-standard layouts
    if not split_to_images:
        candidates = {
            "train": [dataset_root / "train" / "images", dataset_root / "images" / "train"],
            "val": [
                dataset_root / "valid" / "images",
                dataset_root / "val" / "images",
                dataset_root / "images" / "val",
            ],
            "test": [dataset_root / "test" / "images", dataset_root / "images" / "test"],
        }
        for split, paths in candidates.items():
            for p in paths:
                if p.exists():
                    split_to_images[split] = p.resolve()
                    break
    return split_to_images


def _image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _labels_dir_for_images_dir(images_dir: Path) -> Path | None:
    """
    map .../images/<split> -> .../labels/<split>
    map .../<split>/images -> .../<split>/labels
    """
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        cand = Path(*parts[:idx], "labels", *parts[idx + 1 :])
        if cand.exists():
            return cand
    # Roboflow style: train/images => train/labels
    if images_dir.name == "images":
        cand2 = images_dir.parent / "labels"
        if cand2.exists():
            return cand2
    return None


def _build_dataset_paths(dataset_root: Path) -> DatasetPaths | None:
    names = _read_names(dataset_root)
    if not names:
        return None

    class_map: dict[int, int] = {}
    for src_id, src_name in names.items():
        mapped = _map_label_to_target(src_name)
        if mapped is not None:
            class_map[src_id] = mapped

    splits = _discover_split_images(dataset_root)
    if not splits:
        return None

    return DatasetPaths(
        dataset_root=dataset_root,
        split_to_images=splits,
        class_map=class_map,
        source_names=names,
    )


def _write_yaml(output_root: Path) -> None:
    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(TARGET_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(TARGET_CLASSES):
        lines.append(f"  {i}: {name}")
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _on_rm_error(func, path, _exc_info):
    # Windows can keep read-only flags; clear and retry.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _safe_rmtree(path: Path, retries: int = 5, sleep_sec: float = 0.3) -> None:
    if not path.exists():
        return
    for i in range(retries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(sleep_sec)


def main() -> None:
    args = parse_args()
    datasets_root = Path(args.datasets_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets root not found: {datasets_root}")

    if args.clear_output and output_root.exists():
        _safe_rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted([d for d in datasets_root.iterdir() if d.is_dir()])
    usable: list[DatasetPaths] = []
    skipped: list[str] = []

    for d in dataset_dirs:
        info = _build_dataset_paths(d)
        if info is None:
            skipped.append(d.name)
            continue
        usable.append(info)

    if not usable:
        raise RuntimeError(
            "No usable datasets found. Ensure each folder has class names and split folders."
        )

    report: dict[str, object] = {
        "datasets_root": str(datasets_root),
        "output_root": str(output_root),
        "target_classes": TARGET_CLASSES,
        "skipped_datasets": skipped,
        "dataset_summaries": [],
        "totals": {
            "images_scanned": 0,
            "images_written": 0,
            "labels_written": 0,
            "label_rows_written": 0,
        },
    }

    for ds in usable:
        ds_slug = re.sub(r"[^a-zA-Z0-9]+", "_", ds.dataset_root.name).strip("_").lower()
        ds_stats = {
            "dataset": ds.dataset_root.name,
            "source_classes": ds.source_names,
            "mapped_class_ids": ds.class_map,
            "splits": {},
        }

        for split in ("train", "val", "test"):
            img_dir = ds.split_to_images.get(split)
            if img_dir is None:
                continue

            lbl_dir = _labels_dir_for_images_dir(img_dir)
            if lbl_dir is None:
                # Continue scanning images, but labels may be absent.
                lbl_dir = img_dir

            split_stats = {
                "images_scanned": 0,
                "images_written": 0,
                "label_rows_written": 0,
            }

            for img_path in _image_files(img_dir):
                split_stats["images_scanned"] += 1
                report["totals"]["images_scanned"] += 1

                rel = img_path.relative_to(img_dir)
                rel_tag = re.sub(r"[^a-zA-Z0-9]+", "_", rel.as_posix().rsplit(".", 1)[0]).strip("_")
                base_stem = f"{ds_slug}__{rel_tag}" if rel_tag else f"{ds_slug}__{img_path.stem}"
                digest = hashlib.sha1(str(img_path).encode("utf-8")).hexdigest()[:10]
                if len(base_stem) > 120:
                    base_stem = base_stem[:120].rstrip("_")
                out_stem = f"{base_stem}__{digest}"

                label_src = (lbl_dir / rel).with_suffix(".txt")
                kept_rows: list[str] = []

                if label_src.exists():
                    for line in label_src.read_text(encoding="utf-8", errors="ignore").splitlines():
                        row = line.strip()
                        if not row:
                            continue
                        cols = row.split()
                        if not cols:
                            continue
                        try:
                            src_cls = int(float(cols[0]))
                        except Exception:
                            continue
                        dst_cls = ds.class_map.get(src_cls)
                        if dst_cls is None:
                            continue
                        cols[0] = str(dst_cls)
                        kept_rows.append(" ".join(cols))

                if not kept_rows and not args.copy_empty_label_images:
                    continue

                out_img = output_root / "images" / split / f"{out_stem}{img_path.suffix.lower()}"
                out_lbl = output_root / "labels" / split / f"{out_stem}.txt"

                shutil.copy2(img_path, out_img)
                out_lbl.write_text("\n".join(kept_rows) + ("\n" if kept_rows else ""), encoding="utf-8")

                split_stats["images_written"] += 1
                split_stats["label_rows_written"] += len(kept_rows)
                report["totals"]["images_written"] += 1
                report["totals"]["labels_written"] += 1
                report["totals"]["label_rows_written"] += len(kept_rows)

            ds_stats["splits"][split] = split_stats

        report["dataset_summaries"].append(ds_stats)

    _write_yaml(output_root)
    (output_root / "merge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Combined dataset created at: {output_root}")
    print("Target classes:")
    for i, name in enumerate(TARGET_CLASSES):
        print(f"  {i}: {name}")
    print("Totals:")
    print(f"  images_scanned: {report['totals']['images_scanned']}")
    print(f"  images_written: {report['totals']['images_written']}")
    print(f"  labels_written: {report['totals']['labels_written']}")
    print(f"  label_rows_written: {report['totals']['label_rows_written']}")
    if skipped:
        print("Skipped datasets:")
        for name in skipped:
            print(f"  - {name}")
    print(f"YAML: {output_root / 'data.yaml'}")
    print(f"Report: {output_root / 'merge_report.json'}")


if __name__ == "__main__":
    main()
