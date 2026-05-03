# Drilling Activity Recognition: Practical Solution

This document explains the recommended approach for drilling-site worker activity recognition using pose + temporal models, and how to leverage unlabeled data safely.

## Why your previous realtime output looked wrong

- The baseline model was trained on generic synthetic classes, not drilling-specific workflows.
- Single-frame or weak temporal features make actions like `standing` vs `manual_working` unstable.
- Realtime needs causal windowing and smoothing, not only raw per-frame argmax predictions.

## Recommended model stack

1. Pose extraction:
   - `yolo26n-pose.pt` for 17 human keypoints.
2. Temporal classifier:
   - LSTM on normalized keypoint sequences (`train_pose_activity_lstm.py`).
3. Streaming decision layer:
   - warmup threshold
   - EMA probability smoothing
   - uncertainty gate (`unknown/uncertain`)
   - short hold/hysteresis to avoid label flicker
4. Optional context fusion:
   - combine pose output with rig telemetry (depth, RPM, torque, hook load, pressure) and/or tool/person detection.

## Activity scope (current dummy + model)

- `standing`
- `walking`
- `sitting`
- `manual_working`
- `drilling_operation`
- `pipe_handling`
- `crouching`
- `bending`
- `falling`
- `lying`
- unlabeled marker: `unlabeled` (for semi-supervised reinforcement)

## Unlabeled data strategy

The LSTM trainer supports pseudo-label reinforcement:

- Train supervised on labeled sessions.
- Run model on `unlabeled` windows.
- Keep only high-confidence pseudo labels.
- Fine-tune for a few epochs on labeled + pseudo-labeled windows.

Use:

```powershell
python train_pose_activity_lstm.py --data-json datasets/pose_dummy_drilling_activity_v2.json --window-size 32 --stride 4 --use-unlabeled --pseudo-threshold 0.94
```

## Critical production note

Dummy data is only for pipeline validation.  
For field deployment, real rig video + real annotation is mandatory.

Minimum high-value labels for drilling operations:

- `connection/make-up`
- `tripping in`
- `tripping out`
- `reaming/back-reaming`
- `circulating`
- `manual handling / wrenching`
- `monitoring/idle`
- `abnormal posture / fall`

## References used

- Ultralytics pose docs (`yolo26n-pose`, COCO keypoints, 17 keypoints):
  - https://github.com/ultralytics/ultralytics/blob/main/docs/en/tasks/pose.md
- Oil well drilling activity recognition with hierarchical classifier on real rig data:
  - https://www.sciencedirect.com/science/article/pii/S0920410520309402
- LSTM-based deep formation drilling condition recognition (open access):
  - https://link.springer.com/article/10.1007/s42452-025-08001-1
- Skeleton action recognition model family reference (ST-GCN):
  - https://arxiv.org/abs/1801.07455
- Semi-supervised confidence-based pseudo-labeling reference (FixMatch):
  - https://arxiv.org/abs/2001.07685
- Industrial worker-activity dataset example (wearable, real plant, expert labels):
  - https://digitalinnovationlab.github.io/mppdataset
