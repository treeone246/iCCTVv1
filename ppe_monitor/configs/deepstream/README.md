# DeepStream Configs (Phase 1)

This folder contains Phase 1 DeepStream templates for `ppe_monitor`.

## Locked Decisions

1. Pose handling in Phase 1: **Decision 1b**
- Keep pose in Python for association quality.
- DeepStream provides primary PPE detections + tracker metadata.
- Existing Python compliance logic stays active.

2. Camera scope in Phase 1: **single-camera implementation with mux pattern retained**
- `nvstreammux` is configured with `batch-size=1`.
- Source-batch architecture is kept so multi-camera expansion is a config/code extension, not a rewrite.

Phase 2 implementation now supports multi-camera source lists through:

- `deepstream.source_uris`
- `deepstream.camera_ids`

## Files

- `config_infer_primary_yolo.txt`: primary `nvinfer` template
- `config_tracker_NvDCF.yml`: tracker template (DeepStream 7.x YAML format)
- `labels_ppe.txt`: label list used by primary detector
- `deepstream_app_ppe.txt`: `deepstream-app` smoke-test config

## Engine Requirements

Phase 1 expects:

- `models/best2.engine`

If missing, export first:

```bash
yolo export model=models/best2.pt format=engine device=0 half=True imgsz=640
```

Or from ONNX:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/best2.onnx \
  --saveEngine=models/best2.engine \
  --fp16 \
  --minShapes=input.1:1x3x640x640 \
  --optShapes=input.1:1x3x640x640 \
  --maxShapes=input.1:1x3x640x640
```

## Smoke Test

Run this before Python integration:

```bash
deepstream-app -c configs/deepstream/deepstream_app_ppe.txt
```

If this fails, fix DeepStream/model wiring first.
