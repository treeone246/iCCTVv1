# SOP: Pose Keypoint Data Collection and State-Model Training

This document is a practical Standard Operating Procedure (SOP) for collecting real-time pose keypoints and training an activity/state model (for cases like `running`, `working`, `sleeping`, `falling`, `fainted`).

## 1) Goal and Scope

Use `yolo26n-pose.pt` to capture per-person keypoint time series from webcam video, then train a state/activity classifier from those sequences.

Included in this repository:

- `collect_pose_keypoints_realtime.py`: real-time data collection and export (`jsonl`, `csv`, `npy`)
- `train_pose_state_baseline.py`: baseline classifier training from collected CSV files

## 2) Data Schema Overview

Each detected person at each frame becomes one record with:

- session metadata (`session_name`, `activity_label`, `subject_id`, `camera_id`)
- time/index (`timestamp`, `frame_index`)
- identity (`person_id` from tracker when available)
- detection context (`bbox`, `model_confidence`)
- pose keypoints list:
  - `kpt_index`
  - `x`, `y` pixel coordinates
  - `xn`, `yn` normalized coordinates
  - `conf` keypoint confidence

## 3) Folder Layout (Recommended)

Use one session per recording run:

```text
datasets/
  pose_sessions/
    running_20260428_101500/
      keypoints.jsonl
      keypoints.csv
      keypoints.npy
      session_meta.json
      annotated.mp4   (if enabled)
```

## 4) Collection Procedure (Per Activity Class)

Collect each activity as separate sessions with `--activity-label`.

Examples:

```powershell
# Running
python collect_pose_keypoints_realtime.py --model yolo26n-pose.pt --camera 1 --activity-label running --subject-id s01 --camera-id cam_lab --save-annotated-video

# Working (desk/computer movement)
python collect_pose_keypoints_realtime.py --model yolo26n-pose.pt --camera 1 --activity-label working --subject-id s01 --camera-id cam_lab

# Sleeping (lying still for long duration)
python collect_pose_keypoints_realtime.py --model yolo26n-pose.pt --camera 1 --activity-label sleeping --subject-id s01 --camera-id cam_room

# Falling event sessions
python collect_pose_keypoints_realtime.py --model yolo26n-pose.pt --camera 1 --activity-label falling --subject-id s01 --camera-id cam_lab

# Fainted-like sessions (lying + low movement)
python collect_pose_keypoints_realtime.py --model yolo26n-pose.pt --camera 1 --activity-label fainted --subject-id s01 --camera-id cam_lab
```

Stop recording with `q` or `Esc`.

### Recommended capture settings

- 20 to 60 seconds per session
- at least 10 to 30 sessions per class initially
- multiple subjects, clothes, and body types
- multiple camera angles/heights/distances
- varied lighting/background

### Class-specific guidance

- `running`: include start/stop transitions, direction changes
- `working`: include natural hand/upper-body movement at desk
- `sleeping`: include long static lying intervals
- `falling`: include pre-fall and post-fall frames in same session
- `fainted`: include collapse/posture transition and sustained stillness

## 5) Label Quality Rules

- Label one dominant activity per session.
- If activity changes mid-session, stop and start a new session with new label.
- Keep at least one reviewer pass on hard classes (`falling` vs `fainted` vs `sleeping`).
- Do not mix train/test by subject and session randomly per frame; split by session (or subject) to avoid leakage.

## 6) Output File Examples

## 6.1 JSONL example (`keypoints.jsonl`)

One JSON object per line:

```json
{
  "session_name": "running_20260428_101500",
  "frame_index": 42,
  "timestamp": 1777365312.154,
  "activity_label": "running",
  "subject_id": "s01",
  "camera_id": "cam_lab",
  "person_id": 3,
  "class_id": 0,
  "class_label": "person",
  "model_confidence": 0.94,
  "bbox": {"x1": 412.2, "y1": 139.5, "x2": 705.9, "y2": 694.7, "w": 293.7, "h": 555.2},
  "keypoints": [
    {"kpt_index": 0, "x": 520.1, "y": 183.3, "xn": 0.4063, "yn": 0.2546, "conf": 0.88},
    {"kpt_index": 1, "x": 512.7, "y": 176.4, "xn": 0.4005, "yn": 0.2450, "conf": 0.81}
  ]
}
```

## 6.2 CSV example (`keypoints.csv`)

Flat columns (truncated):

```csv
session_name,frame_index,timestamp,activity_label,subject_id,camera_id,person_id,class_id,class_label,model_confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,bbox_w,bbox_h,n_keypoints,kp00_x,kp00_y,kp00_xn,kp00_yn,kp00_conf,kp01_x,kp01_y,kp01_xn,kp01_yn,kp01_conf
running_20260428_101500,42,1777365312.154,running,s01,cam_lab,3,0,person,0.94,412.2,139.5,705.9,694.7,293.7,555.2,17,520.1,183.3,0.4063,0.2546,0.88,512.7,176.4,0.4005,0.2450,0.81
```

## 6.3 NPY example (`keypoints.npy`)

Saved arrays:

- `features`: shape `[N, 6 + keypoint_count*5]` where 6 base fields are:
  - `model_confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, person_id`
- `labels`: shape `[N]`
- `person_ids`: shape `[N]`
- `frame_indices`: shape `[N]`
- `sessions`: shape `[N]`
- `keypoint_count`: shape `[1]`

Load example:

```python
import numpy as np
bundle = np.load(
    "datasets/pose_sessions/running_20260428_101500/keypoints.npy",
    allow_pickle=True,
).item()
print(bundle["features"].shape)
print(bundle["labels"][:5])
```

## 7) Baseline Training SOP (State Model)

Train baseline model from collected session CSV files:

```powershell
python train_pose_state_baseline.py --data-root datasets/pose_sessions --window-size 30 --stride 10
```

What training script does:

- loads all `keypoints.csv`
- groups sequence by `(session_name, person_id, activity_label)`
- normalizes pose per frame (center + scale normalization)
- builds sliding windows of `window_size` frames
- creates features (pose + temporal delta)
- splits by session (`GroupShuffleSplit`)
- trains a baseline `RandomForestClassifier`
- saves model and metadata:
  - `models/pose_state_baseline.pkl`
  - `models/pose_state_baseline_meta.json`

## 8) Practical Modeling Approach (Production-Oriented)

Recommended architecture in deployment:

1. Pose extraction layer:
   - `yolo26n-pose.pt` + tracker (`bytetrack.yaml`)
2. Temporal state model:
   - classify short windows into `running/working/sleeping/falling/fainted`
3. Event logic layer:
   - rules for persistence, cooldown, and confidence gating
4. Alert layer:
   - notify only when event condition is stable for N frames

For hard classes:

- `falling`: abrupt posture + vertical velocity change + post-fall lying
- `sleeping`: lying posture + long-duration low motion + context (bed/couch)
- `fainted`: often similar to lying still; improve with transition dynamics and context signals

## 9) Data Quantity Targets (Initial)

- 5 classes x 20 sessions/class minimum (starter)
- 30 to 90 seconds/session
- at least 3 to 5 subjects
- balanced class counts

As a rough start, this usually gives enough data for a usable baseline and fast iteration.

## 10) Common Pitfalls

- Missing tracking IDs (`person_id=-1`) if tracking disabled or unstable
- Label leakage from random per-frame split
- Single-subject overfitting
- Ambiguous labels between `sleeping` and `fainted`
- Very short sessions that cannot form temporal windows

## 11) Suggested Next Improvements

- Train deep sequence model (LSTM/TCN/Transformer/ST-GCN) after baseline
- Add explicit motion features (velocity/acceleration of torso/hips)
- Add scene priors (bed/floor/desk zones)
- Calibrate per-class thresholds with validation set
- Add false-alarm suppression rules in runtime logic
