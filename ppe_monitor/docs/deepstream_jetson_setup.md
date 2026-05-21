# DeepStream Jetson Setup (Orin / DeepStream 7.1)

## Verify Versions

```bash
deepstream-app --version-all
cat /etc/nv_tegra_release
apt show nvidia-jetpack 2>/dev/null | grep Version
```

Expected family:

- DeepStreamSDK 7.1.0
- CUDA 12.6
- TensorRT 10.7

## Verify Core Plugins

```bash
gst-inspect-1.0 nvinfer
gst-inspect-1.0 nvstreammux
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvtracker
```

If missing:

```bash
ls /opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins/
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins/${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}
```

## Install Python Binding (`pyds`)

Preferred: install prebuilt wheel matching DeepStream 7.1 + JetPack.

```bash
pip install pyds-*.whl
python -c "import pyds; print(getattr(pyds, '__version__', 'ok'))"
```

## Jetson Performance Mode

```bash
sudo nvpmodel -q
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Engine Export

Preferred:

```bash
yolo export model=models/best2.pt format=engine device=0 half=True imgsz=640
```

ONNX fallback:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/best2.onnx \
  --saveEngine=models/best2.engine \
  --fp16 \
  --minShapes=input.1:1x3x640x640 \
  --optShapes=input.1:1x3x640x640 \
  --maxShapes=input.1:1x3x640x640
```

## Smoke Test With `deepstream-app` (Before Python)

```bash
deepstream-app -c configs/deepstream/deepstream_app_ppe.txt
```

If this fails, fix DS/model wiring first.

## Run `ppe_monitor` DeepStream Backend

Set in `config.yaml`:

```yaml
runtime:
  backend: "deepstream"
deepstream:
  enabled: true
  engine_path: "models/best2.engine"
```

Multi-camera (Phase 2):

```yaml
runtime:
  backend: "deepstream"
deepstream:
  enabled: true
  source_uris:
    - "rtsp://user:pass@cam-a/stream1"
    - "rtsp://user:pass@cam-b/stream1"
  camera_ids: ["rig_floor_cam_01", "rig_floor_cam_02"]
  batch_size: 2
  engine_path: "models/best2.engine"
```

Run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Standalone runner:

```bash
python -m app.deepstream.run --config config.yaml --source rtsp://user:pass@camera/stream1 --engine models/best2.engine
```

## Troubleshooting

- `DeepStream backend selected but dependencies are missing`:
  install `pyds` and verify DeepStream plugin paths.
- `TensorRT engine not found`:
  export `models/best2.engine` first.
- `Failed to create DeepStream elements`:
  one or more GStreamer/DeepStream plugins not visible (`GST_PLUGIN_PATH`).
- `Could not find output coverage layer for parsing objects`:
  your model output format does not match DeepStream default detector parser.
  Use a custom parser (`parse-func=0`, `custom-lib-path`, `parse-bbox-func-name`)
  for YOLO-style outputs.
