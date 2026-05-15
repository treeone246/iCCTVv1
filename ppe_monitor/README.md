# PPE Compliance Monitoring Prototype

Real-time PPE compliance monitoring using a three-model ONNX cascade:

- Pose + tracking: `models/yolo26n-pose.onnx`
- PPE detection: `models/best.onnx`
- Verifier: `models/yoloe-ppe.onnx`

The system uses tracked persons, keypoint-aware PPE association, verifier cache TTLs, and a per-item alert state machine with hysteresis.

## Quick Start

```bash
pip install -r requirements.txt
```

Drop models into `models/`:

- `models/yolo26n-pose.onnx`
- `models/best.onnx`
- `models/yoloe-ppe.onnx`

Run:

```bash
uvicorn app.main:app --reload
```

Open dashboard at `http://localhost:8000`.

## Model Preparation

- Export ONNX with opset 17+ when possible.
- Use `dynamic=True` when you hit shape compatibility issues.
- Keep task settings aligned:
  - pose model with `task='pose'`
  - PPE/verifier models with `task='detect'`

If preprocessing errors occur, inspect input/output tensor names and shapes with Netron and compare with startup logs.

## Configuration

All thresholds and behavior are in `config.yaml`.

- `video.source`: webcam index, file path, or RTSP URL
- `video.target_fps`: desired streaming cadence
- `video.drop_grab_limit`: max extra `.grab()` operations to catch up when slow
- `models.pose|ppe|verifier`: ONNX paths
- `models.allow_mock_models`: if true, missing models are replaced with mocks
- `inference.device`: `auto`, `cuda`, or `cpu`
- `inference.conf_threshold_pose|ppe|verifier`: score thresholds per model
- `inference.keypoint_conf_floor`: below this keypoint confidence => `INDETERMINATE`
- `inference.imgsz`: inference image size
- `association.*`: PPE binding distances, keypoint sets, coverall IoU threshold, held-distance ratio
- `verifier_cache.ttl_compliant_seconds|ttl_violation_seconds`: cache TTLs per verifier result
- `state_machine.window_size`: rolling window length
- `state_machine.violation_threshold`: raise alert on this many violations in window
- `state_machine.clear_threshold`: clear alert after this many consecutive compliant frames
- `required_ppe`: PPE items to enforce
- `dashboard.jpeg_quality`: JPEG quality used for stream frames
- `dashboard.metrics_window_minutes`: time window used for dashboard aggregate metrics

## ONNX Caveats

- TensorRT export is the next optimization step once ONNX flow is stable.
- Verify each model's input shape and output assumptions before debugging association logic.
- YOLOE text-prompt inference usually does not survive ONNX export. The verifier is implemented as fixed-class detection and filtered by expected item.

## Troubleshooting

- CUDA expected but CPU used:
  - check startup provider logs for each model
  - verify GPU runtime installation and provider availability
- Too many dropped frames:
  - lower input resolution, reduce FPS, or move to GPU
- Alert flicker:
  - increase `window_size` or `violation_threshold`
  - tune `clear_threshold` and verifier cache TTLs

## Testing

Run unit tests:

```bash
pytest -q
```

CI can run with `models.allow_mock_models: true` (no real model files required).
