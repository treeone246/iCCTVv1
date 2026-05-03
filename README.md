# Windows ZeroMQ Sender (to WSL ROS 2 Bridge)

This Windows-side Python app sends object detection payloads as JSON over ZeroMQ.

- Windows app role: `PUSH` sender
- WSL ROS 2 bridge role: `PULL` receiver
- Default endpoint: `tcp://127.0.0.1:5555`

If localhost forwarding does not work in your WSL network setup, use your WSL IP instead of `127.0.0.1`.

## Files

- `windows_sender.py`: ZeroMQ transport and send loop CLI
- `model_adapter.py`: model output -> JSON schema conversion helpers
- `config.py`: host/port/frame/send interval configuration
- `requirements.txt`: Python dependencies

## JSON Schema Sent

```json
{
  "timestamp": 1710000000.123,
  "frame_id": "camera_front",
  "detections": [
    {
      "label": "person",
      "score": 0.95,
      "x1": 100,
      "y1": 120,
      "x2": 260,
      "y2": 500
    }
  ]
}
```

## Windows Setup

```powershell
cd C:\iCCTVv1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Sender

One-shot fake payload:

```powershell
python windows_sender.py
```

Continuous loop:

```powershell
python windows_sender.py --loop
```

Override endpoint manually:

```powershell
python windows_sender.py --host 172.28.112.1 --port 5555
```

## Troubleshooting

1. Localhost does not work
- Try WSL IP instead of `127.0.0.1`.
- In WSL, get IP with:
```bash
hostname -I
```
- Use the first IP in `--host <ip>`.

2. Need WSL IP instead of localhost
- Update `ZMQ_HOST` in `config.py`, or pass `--host` on the CLI.

3. Windows Firewall blocking port
- Ensure outbound/inbound TCP traffic on port `5555` is allowed for your environment.
- Temporarily test with firewall disabled only if your security policy allows it.

4. Malformed JSON payloads
- Ensure detections contain:
  - `label` (string)
  - `score` (float)
  - `x1`, `y1`, `x2`, `y2` (numeric pixel coordinates)
- `windows_sender.py` logs JSON serialization errors.

## How to connect your real model

1. Run inference in your existing Windows CV loop.
2. Convert model output using helpers in `model_adapter.py`.
3. Send with `ZmqDetectionSender.send_payload(payload)`.

Minimal example:

```python
from windows_sender import ZmqDetectionSender
from model_adapter import adapt_parsed_detections
import config

sender = ZmqDetectionSender(config.zmq_endpoint())

# Replace this with detections from your model output
parsed = [
    {"label": "person", "score": 0.93, "x1": 80, "y1": 120, "x2": 240, "y2": 460}
]

payload = adapt_parsed_detections(parsed, frame_id=config.DEFAULT_FRAME_ID)
sender.send_payload(payload)
sender.close()
```

## Pose Activity Pipeline (Baseline + LSTM Streaming)

Files for building activity recognition from pose:

- `webcam_yolo26_realtime_pose.py`: real-time pose visualization
- `collect_pose_keypoints_realtime.py`: collect labeled pose sessions (`jsonl/csv/npy`)
- `train_pose_state_baseline.py`: baseline classical ML trainer (RandomForest)
- `realtime_pose_activity_inference.py`: baseline realtime predictor with styled overlay
- `generate_drilling_pose_dummy_data.py`: synthetic drilling-scope dummy data generator
- `train_pose_activity_lstm.py`: LSTM sequence trainer (supports pseudo-labeling from unlabeled sessions)
- `realtime_pose_activity_lstm.py`: realtime streaming LSTM predictor (warmup + smoothing + uncertainty gating)
- `POSE_KEYPOINT_DATA_SOP.md`: collection and labeling SOP

### Quick Start (LSTM path)

```powershell
pip install -r requirements_activity.txt

# 1) Generate drilling-scope dummy data
python generate_drilling_pose_dummy_data.py --output datasets/pose_dummy_drilling_activity_v2.json

# 2) Train LSTM (supervised + optional pseudo-label reinforcement)
python train_pose_activity_lstm.py --data-json datasets/pose_dummy_drilling_activity_v2.json --window-size 32 --stride 4 --use-unlabeled

# 3) Realtime prediction
python realtime_pose_activity_lstm.py --pose-model yolo26n-pose.pt --activity-model models/pose_activity_lstm.pt --camera 1
```

### Real Data Collection Reminder

For production reliability, train with real site recordings:

```powershell
python collect_pose_keypoints_realtime.py --model yolo26n-pose.pt --camera 1 --activity-label standing --subject-id worker01 --camera-id rig_cam_a
python collect_pose_keypoints_realtime.py --model yolo26n-pose.pt --camera 1 --activity-label drilling_operation --subject-id worker01 --camera-id rig_cam_a
```
