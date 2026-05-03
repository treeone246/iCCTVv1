import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a baseline activity/state classifier from keypoint CSV files "
            "produced by collect_pose_keypoints_realtime.py"
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/pose_sessions",
        help="Root folder containing session folders with keypoints.csv.",
    )
    parser.add_argument(
        "--data-json",
        type=str,
        default="",
        help=(
            "Optional JSON dataset path (e.g., datasets/pose_dummy_activity_data_varied.json). "
            "If set, training rows are loaded from this JSON instead of CSV sessions."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Sequence length in frames (e.g., 30 for ~1 sec at 30 FPS).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sliding window stride in frames.",
    )
    parser.add_argument(
        "--min-kpt-conf",
        type=float,
        default=0.2,
        help="Minimum keypoint confidence considered valid for normalization.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Session-level holdout fraction.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/pose_state_baseline.pkl",
        help="Path to save the trained baseline model.",
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default="models/pose_state_baseline_meta.json",
        help="Path to save model metadata.",
    )
    return parser.parse_args()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _discover_keypoint_count(headers: list[str]) -> int:
    count = 0
    while f"kp{count:02d}_x" in headers:
        count += 1
    return count


def _load_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        kpt_count = _discover_keypoint_count(headers)

        for row in reader:
            keypoints = np.zeros((kpt_count, 3), dtype=np.float32)
            for k in range(kpt_count):
                keypoints[k, 0] = _to_float(row.get(f"kp{k:02d}_x"), default=np.nan)
                keypoints[k, 1] = _to_float(row.get(f"kp{k:02d}_y"), default=np.nan)
                keypoints[k, 2] = _to_float(row.get(f"kp{k:02d}_conf"), default=0.0)

            rows.append(
                {
                    "session_name": row.get("session_name", "unknown_session"),
                    "frame_index": _to_int(row.get("frame_index"), default=0),
                    "timestamp": _to_float(row.get("timestamp"), default=0.0),
                    "activity_label": row.get("activity_label", "unknown"),
                    "person_id": _to_int(row.get("person_id"), default=-1),
                    "bbox_h": _to_float(row.get("bbox_h"), default=1.0),
                    "keypoints": keypoints,
                }
            )
    return rows


def _choose_center(points_xy: np.ndarray, conf: np.ndarray, min_kpt_conf: float) -> np.ndarray:
    # COCO-style fallback priority: hips -> shoulders -> confidence-weighted centroid.
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


def _normalize_frame(
    keypoints: np.ndarray, bbox_h: float, min_kpt_conf: float
) -> np.ndarray:
    points_xy = keypoints[:, :2].copy()
    conf = keypoints[:, 2].copy()

    points_xy = np.nan_to_num(points_xy, nan=0.0, posinf=0.0, neginf=0.0)
    center = _choose_center(points_xy, conf, min_kpt_conf)
    scale = _choose_scale(points_xy, conf, bbox_h=bbox_h, min_kpt_conf=min_kpt_conf)

    norm_xy = (points_xy - center) / scale
    return np.concatenate([norm_xy, conf[:, None]], axis=1)


def _window_to_feature(window_keypoints: np.ndarray) -> np.ndarray:
    # window_keypoints shape: [T, K, 3], channels = [xn_norm, yn_norm, conf]
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


def _build_dataset(
    rows: list[dict[str, Any]],
    window_size: int,
    stride: int,
    min_kpt_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    # Group by session + person so windows contain temporal continuity.
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["session_name"], row["person_id"], row["activity_label"])
        grouped[key].append(row)

    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    seq_lengths: list[int] = []

    for (session_name, _person_id, activity_label), seq in grouped.items():
        seq = sorted(seq, key=lambda x: x["frame_index"])
        seq_lengths.append(len(seq))
        if len(seq) < window_size:
            continue

        normalized_frames = [
            _normalize_frame(
                keypoints=item["keypoints"],
                bbox_h=item["bbox_h"],
                min_kpt_conf=min_kpt_conf,
            )
            for item in seq
        ]
        arr = np.stack(normalized_frames, axis=0)

        for start in range(0, len(arr) - window_size + 1, stride):
            window = arr[start : start + window_size]
            features.append(_window_to_feature(window))
            labels.append(activity_label)
            groups.append(session_name)

    if not features:
        stats = {
            "num_sequences": len(seq_lengths),
            "max_sequence_length": max(seq_lengths) if seq_lengths else 0,
            "min_sequence_length": min(seq_lengths) if seq_lengths else 0,
            "avg_sequence_length": (
                float(np.mean(seq_lengths)) if seq_lengths else 0.0
            ),
        }
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype="<U1"),
            np.empty((0,), dtype="<U1"),
            stats,
        )

    stats = {
        "num_sequences": len(seq_lengths),
        "max_sequence_length": max(seq_lengths) if seq_lengths else 0,
        "min_sequence_length": min(seq_lengths) if seq_lengths else 0,
        "avg_sequence_length": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
    }
    return (
        np.stack(features, axis=0),
        np.asarray(labels),
        np.asarray(groups),
        stats,
    )


def _load_all_rows(data_root: Path) -> list[dict[str, Any]]:
    csv_files = sorted(data_root.glob("**/keypoints.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No keypoints.csv found under: {data_root}")

    rows: list[dict[str, Any]] = []
    for csv_file in csv_files:
        rows.extend(_load_csv_rows(csv_file))
    return rows


def _load_rows_from_json(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON dataset not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"JSON dataset has no non-empty 'samples' list: {json_path}")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        keypoints_raw = sample.get("keypoints", [])
        if not isinstance(keypoints_raw, list) or not keypoints_raw:
            continue

        k_max = max(int(k.get("kpt_index", i)) for i, k in enumerate(keypoints_raw)) + 1
        keypoints = np.zeros((k_max, 3), dtype=np.float32)
        keypoints[:, 0] = np.nan
        keypoints[:, 1] = np.nan

        for i, kp in enumerate(keypoints_raw):
            idx = _to_int(kp.get("kpt_index"), default=i)
            if idx < 0 or idx >= k_max:
                continue
            keypoints[idx, 0] = _to_float(kp.get("x"), default=np.nan)
            keypoints[idx, 1] = _to_float(kp.get("y"), default=np.nan)
            keypoints[idx, 2] = _to_float(kp.get("conf"), default=0.0)

        bbox = sample.get("bbox", {})
        rows.append(
            {
                "session_name": str(sample.get("session_name", "unknown_session")),
                "frame_index": _to_int(sample.get("frame_index"), default=0),
                "timestamp": _to_float(sample.get("timestamp"), default=0.0),
                "activity_label": str(sample.get("activity_label", "unknown")),
                "person_id": _to_int(sample.get("person_id"), default=-1),
                "bbox_h": _to_float(bbox.get("h"), default=1.0),
                "keypoints": keypoints,
            }
        )

    if not rows:
        raise ValueError(f"No usable rows parsed from JSON dataset: {json_path}")
    return rows


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_path = Path(args.output_model)
    meta_path = Path(args.output_metadata)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if args.data_json.strip():
        rows = _load_rows_from_json(Path(args.data_json))
    else:
        rows = _load_all_rows(data_root)
    X, y, groups, ds_stats = _build_dataset(
        rows=rows,
        window_size=args.window_size,
        stride=args.stride,
        min_kpt_conf=args.min_kpt_conf,
    )

    if len(X) == 0:
        raise RuntimeError(
            "No training windows were generated. "
            f"window_size={args.window_size}, "
            f"max_sequence_length={ds_stats['max_sequence_length']}, "
            f"avg_sequence_length={ds_stats['avg_sequence_length']:.2f}. "
            "Reduce --window-size (must be <= max sequence length) or collect longer sessions."
        )

    labels = sorted(set(y.tolist()))
    if len(labels) < 2:
        raise RuntimeError(
            f"Need at least 2 labels to train a classifier, found: {labels}"
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_seed
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Keep pipeline explicit so future feature subsets can be added without code churn.
    passthrough_cols = list(range(X.shape[1]))
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), passthrough_cols)],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=args.random_seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    print("=== Classification Report ===")
    print(report)
    print(f"Train windows: {len(X_train)}")
    print(f"Test windows: {len(X_test)}")
    print(f"Classes: {labels}")

    with model_path.open("wb") as f:
        pickle.dump(pipeline, f)

    metadata = {
        "data_root": str(data_root),
        "window_size": args.window_size,
        "stride": args.stride,
        "min_kpt_conf": args.min_kpt_conf,
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "n_features": int(X.shape[1]),
        "labels": labels,
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "model_path": str(model_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
