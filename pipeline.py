import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


SOURCE = "videos/simulationTest2.mp4"
PPE_MODEL_PATH = "best.pt"
PERSON_MODEL_PATH = "yolo26n.pt"

PPE_CLASSES = ("coveralls", "helmet", "gloves", "safety_glasses", "safety_boots")
CONF_THRESH = {
    "coveralls": 0.35,
    "helmet": 0.12,
    "gloves": 0.12,
    "safety_glasses": 0.10,
    "safety_boots": 0.15,
}

WINDOW = 15
PRESENCE_HITS = 4
ABSENCE_FRAMES = 30
MIN_OVERLAP = 0.50
PRESENCE_HITS_BY_CLASS = {
    "coveralls": 3,
    "helmet": 2,
    "gloves": 2,
    "safety_glasses": 2,
    "safety_boots": 2,
}
MIN_OVERLAP_BY_CLASS = {
    "coveralls": 0.35,
    "helmet": 0.12,
    "gloves": 0.10,
    "safety_glasses": 0.08,
    "safety_boots": 0.12,
}

PERSON_DET_CONF = 0.35
PPE_DET_CONF = 0.10
PERSON_IMGSZ = 960
PPE_IMGSZ = 1280

# Map possible model label variants into the canonical PPE class names.
CLASS_ALIASES = {
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "helmet": "helmet",
    "glasses": "safety_glasses",
    "safetyglasses": "safety_glasses",
    "safety_glasses": "safety_glasses",
    "goggles": "safety_glasses",
    "coverall": "coveralls",
    "coveralls": "coveralls",
    "overall": "coveralls",
    "overalls": "coveralls",
    "boot": "safety_boots",
    "boots": "safety_boots",
    "safety_boot": "safety_boots",
    "safety_boots": "safety_boots",
    "glove": "gloves",
    "gloves": "gloves",
}


@dataclass
class PPEState:
    history: dict
    status: dict
    last_seen: dict
    fulfilled: dict

    @classmethod
    def create(cls, window: int) -> "PPEState":
        return cls(
            history=defaultdict(lambda: defaultdict(lambda: deque(maxlen=window))),
            status=defaultdict(dict),
            last_seen=defaultdict(dict),
            fulfilled=defaultdict(set),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPE detection + tracking pipeline")
    parser.add_argument("--source", default=SOURCE, help="Video path, webcam index, or RTSP URL")
    parser.add_argument("--ppe-model", default=PPE_MODEL_PATH, help="PPE detector model path")
    parser.add_argument("--person-model", default=PERSON_MODEL_PATH, help="Person detector model path")
    parser.add_argument("--output", default="", help="Optional output video path")
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview window")
    return parser.parse_args()


def resolve_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def get_model_names(model: YOLO):
    return model.names


def class_name(names, cid: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(cid), str(cid)))
    return str(names[int(cid)])


def normalize_class_label(label: str) -> str:
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(key, key)


def filter_ppe_detections(ppe: sv.Detections, names, valid_classes, conf_thresh):
    if len(ppe) == 0:
        return ppe, np.empty((0,), dtype=object)

    class_ids = ppe.class_id if ppe.class_id is not None else np.full(len(ppe), -1, dtype=int)
    confidence = ppe.confidence if ppe.confidence is not None else np.zeros(len(ppe), dtype=np.float32)

    labels = np.array([normalize_class_label(class_name(names, cid)) for cid in class_ids], dtype=object)
    thresholds = np.array([conf_thresh.get(label, 1.01) for label in labels], dtype=np.float32)
    keep = np.isin(labels, np.array(valid_classes, dtype=object)) & (confidence >= thresholds)

    return ppe[keep], labels[keep]


def overlap_ratio_matrix(person_xyxy: np.ndarray, ppe_xyxy: np.ndarray) -> np.ndarray:
    if person_xyxy.size == 0 or ppe_xyxy.size == 0:
        return np.zeros((len(person_xyxy), len(ppe_xyxy)), dtype=np.float32)

    px1 = person_xyxy[:, None, 0]
    py1 = person_xyxy[:, None, 1]
    px2 = person_xyxy[:, None, 2]
    py2 = person_xyxy[:, None, 3]

    ox1 = ppe_xyxy[None, :, 0]
    oy1 = ppe_xyxy[None, :, 1]
    ox2 = ppe_xyxy[None, :, 2]
    oy2 = ppe_xyxy[None, :, 3]

    inter_x1 = np.maximum(px1, ox1)
    inter_y1 = np.maximum(py1, oy1)
    inter_x2 = np.minimum(px2, ox2)
    inter_y2 = np.minimum(py2, oy2)

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    ppe_area = np.clip((ox2 - ox1) * (oy2 - oy1), a_min=1.0, a_max=None)
    return inter / ppe_area


def ppe_center_inside_person_matrix(person_xyxy: np.ndarray, ppe_xyxy: np.ndarray) -> np.ndarray:
    if person_xyxy.size == 0 or ppe_xyxy.size == 0:
        return np.zeros((len(person_xyxy), len(ppe_xyxy)), dtype=bool)

    px1 = person_xyxy[:, None, 0]
    py1 = person_xyxy[:, None, 1]
    px2 = person_xyxy[:, None, 2]
    py2 = person_xyxy[:, None, 3]

    cx = ((ppe_xyxy[:, 0] + ppe_xyxy[:, 2]) * 0.5)[None, :]
    cy = ((ppe_xyxy[:, 1] + ppe_xyxy[:, 3]) * 0.5)[None, :]
    return (cx >= px1) & (cx <= px2) & (cy >= py1) & (cy <= py2)


def associate_ppe_to_tracks(
    persons: sv.Detections,
    ppe: sv.Detections,
    ppe_names: np.ndarray,
    min_overlap: float,
    min_overlap_by_class: dict,
):
    detected_now = defaultdict(set)
    if len(persons) == 0 or len(ppe) == 0 or persons.tracker_id is None:
        return detected_now

    overlaps = overlap_ratio_matrix(persons.xyxy, ppe.xyxy)
    centers_inside = ppe_center_inside_person_matrix(persons.xyxy, ppe.xyxy)
    thresholds = np.array([min_overlap_by_class.get(name, min_overlap) for name in ppe_names], dtype=np.float32)

    for i, tracker_id in enumerate(persons.tracker_id):
        if tracker_id is None:
            continue
        matched_mask = (overlaps[i] >= thresholds) | centers_inside[i]
        if matched_mask.any():
            detected_now[int(tracker_id)].update(ppe_names[matched_mask].tolist())
    return detected_now


def lock_fulfilled_ppe_for_person(state: PPEState, person_id: int, ppe_class: str) -> None:
    state.fulfilled[person_id].add(ppe_class)


def is_ppe_fulfilled_for_person(state: PPEState, person_id: int, ppe_class: str) -> bool:
    return ppe_class in state.fulfilled.get(person_id, set())


def update_ppe_state(
    frame_idx: int,
    persons: sv.Detections,
    detected_now_by_pid,
    state: PPEState,
    classes,
    presence_hits: int,
    presence_hits_by_class: dict,
    absence_frames: int,
):
    if persons.tracker_id is None:
        return

    active_ids = {int(tid) for tid in persons.tracker_id if tid is not None}
    stale_ids = [pid for pid in state.status.keys() if pid not in active_ids]
    for pid in stale_ids:
        state.history.pop(pid, None)
        state.status.pop(pid, None)
        state.last_seen.pop(pid, None)
        state.fulfilled.pop(pid, None)

    for pid in active_ids:
        detected_now = detected_now_by_pid.get(pid, set())
        for cls_name in classes:
            seen = cls_name in detected_now
            state.history[pid][cls_name].append(seen)
            if seen:
                state.last_seen[pid][cls_name] = frame_idx

            if is_ppe_fulfilled_for_person(state=state, person_id=pid, ppe_class=cls_name):
                state.status[pid][cls_name] = "present"
                continue

            recent_hits = sum(state.history[pid][cls_name])
            last_hit = state.last_seen[pid].get(cls_name)
            required_hits = presence_hits_by_class.get(cls_name, presence_hits)

            if recent_hits >= required_hits:
                state.status[pid][cls_name] = "present"
                lock_fulfilled_ppe_for_person(state=state, person_id=pid, ppe_class=cls_name)
                continue

            if last_hit is not None and (frame_idx - last_hit) > absence_frames:
                state.status[pid][cls_name] = "absent"
                continue

            if len(state.history[pid][cls_name]) == state.history[pid][cls_name].maxlen and recent_hits == 0:
                state.status[pid][cls_name] = "absent"
                continue

            state.status[pid].setdefault(cls_name, "unknown")


def build_labels(persons: sv.Detections, state: PPEState):
    labels = []
    if persons.tracker_id is None:
        return labels

    for tracker_id in persons.tracker_id:
        if tracker_id is None:
            labels.append("#-1")
            continue

        pid = int(tracker_id)
        ppe_status = state.status.get(pid, {})
        present = sorted([c for c, s in ppe_status.items() if s == "present"])
        absent = sorted([c for c, s in ppe_status.items() if s == "absent"])
        unknown = sorted([c for c, s in ppe_status.items() if s == "unknown"])
        labels.append(
            f"#{pid} | OK:{','.join(present) or '-'} | MISS:{','.join(absent) or '-'} | ?:{','.join(unknown) or '-'}"
        )
    return labels


def build_ppe_labels(ppe_names: np.ndarray, ppe_confidence: np.ndarray):
    if len(ppe_names) == 0:
        return []
    return [f"{name} {conf:.2f}" for name, conf in zip(ppe_names.tolist(), ppe_confidence.tolist())]


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)

    ppe_model = YOLO(args.ppe_model)
    person_model = YOLO(args.person_model)
    ppe_names_map = get_model_names(ppe_model)
    print(f"[INFO] PPE model classes: {ppe_names_map}")
    raw_names = ppe_names_map.values() if isinstance(ppe_names_map, dict) else ppe_names_map
    normalized_names = sorted({normalize_class_label(str(n)) for n in raw_names})
    print(f"[INFO] Normalized model labels: {normalized_names}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    warnings.filterwarnings(
        "ignore",
        message="The `ByteTrack` was deprecated",
        category=FutureWarning,
    )
    tracker = sv.ByteTrack(lost_track_buffer=90, frame_rate=float(fps))
    state = PPEState.create(window=WINDOW)

    person_box_annotator = sv.BoxAnnotator(thickness=2)
    person_label_annotator = sv.LabelAnnotator(text_scale=0.45)
    ppe_box_annotator = sv.BoxAnnotator(thickness=2)
    ppe_label_annotator = sv.LabelAnnotator(text_scale=0.5)

    writer = None
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (frame_w, frame_h),
        )

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            person_result = person_model(
                frame,
                conf=PERSON_DET_CONF,
                imgsz=PERSON_IMGSZ,
                classes=[0],
                verbose=False,
            )[0]
            persons = sv.Detections.from_ultralytics(person_result)
            persons = tracker.update_with_detections(persons)

            ppe_result = ppe_model(
                frame,
                conf=PPE_DET_CONF,
                imgsz=PPE_IMGSZ,
                verbose=False,
            )[0]
            ppe = sv.Detections.from_ultralytics(ppe_result)
            ppe, ppe_names = filter_ppe_detections(
                ppe=ppe,
                names=ppe_names_map,
                valid_classes=PPE_CLASSES,
                conf_thresh=CONF_THRESH,
            )

            detected_now_by_pid = associate_ppe_to_tracks(
                persons=persons,
                ppe=ppe,
                ppe_names=ppe_names,
                min_overlap=MIN_OVERLAP,
                min_overlap_by_class=MIN_OVERLAP_BY_CLASS,
            )
            update_ppe_state(
                frame_idx=frame_idx,
                persons=persons,
                detected_now_by_pid=detected_now_by_pid,
                state=state,
                classes=PPE_CLASSES,
                presence_hits=PRESENCE_HITS,
                presence_hits_by_class=PRESENCE_HITS_BY_CLASS,
                absence_frames=ABSENCE_FRAMES,
            )

            labels = build_labels(persons=persons, state=state)
            ppe_conf = ppe.confidence if ppe.confidence is not None else np.zeros(len(ppe), dtype=np.float32)
            ppe_labels = build_ppe_labels(ppe_names=ppe_names, ppe_confidence=ppe_conf)

            frame = ppe_box_annotator.annotate(scene=frame, detections=ppe)
            frame = ppe_label_annotator.annotate(scene=frame, detections=ppe, labels=ppe_labels)
            frame = person_box_annotator.annotate(scene=frame, detections=persons)
            frame = person_label_annotator.annotate(scene=frame, detections=persons, labels=labels)

            if writer is not None:
                writer.write(frame)

            if args.show or not args.output:
                cv2.imshow("ppe_pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
