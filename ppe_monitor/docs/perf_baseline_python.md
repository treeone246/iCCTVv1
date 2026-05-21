# Phase 0 Performance Baseline (Python Backend)

Date:
Device:
JetPack/L4T:
DeepStream:
TensorRT:

## Workload

- Source type: RTSP / file
- Resolution:
- FPS input:
- Duration measured:
- Scenario: person visible + PPE transitions + verifier path active

## Commands Used

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
tegrastats --interval 500 --logfile /tmp/tegrastats_python.log
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Observed Metrics

- Average processed FPS:
- CPU utilization (%):
- GPU utilization (%):
- RAM usage (%):
- Swap usage:
- Temperature (C):
- Dropped frames:

## Symptom Table Match

Matched row from decision table:

- [ ] GPU <30%, one CPU core pinned
- [ ] GPU 60-80%, balanced CPU
- [ ] GPU spiky periodic stalls
- [ ] Memory pressure / swap
- [ ] Low GPU, low CPU, waiting
- [ ] ONNX runtime CPU EP fallback
- [ ] Ollama called in per-frame hot path

## Quick-Win Checks

1. ONNX providers:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Result:

2. Ollama/VLM temporarily disabled:

Result:

3. Behavior agent disabled:

Result:

## Decision

- Proceed with DeepStream backend: YES / NO
- Primary bottleneck identified:
- Immediate one-line or small fix applied:

## Baseline Summary

Record final baseline values after quick-win checks:

- FPS:
- CPU:
- GPU:
- RAM:
- Notes:
