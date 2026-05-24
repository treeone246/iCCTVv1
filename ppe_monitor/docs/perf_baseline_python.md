# Phase 0 Performance Baseline (N=2, LLM-Aware)

Date:
Device:
JetPack/L4T:
DeepStream:
TensorRT:
Thermal mode (`nvpmodel`/`jetson_clocks`):

## Workload

- Source:
- Resolution:
- Input FPS:
- Duration per test:
- Scenario: 2-person footage (same file across all tests)

## Shared Commands

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
tegrastats --interval 500 --logfile /tmp/tegrastats_testX.log
sudo py-spy record -o profile_testX.svg --pid <pid> --duration 120
```

## Test 1: CV-Only (Both LLM Paths Off)

Config toggles:

```yaml
behavior_agent:
  enabled: false
verifier:
  enable_vlm: false
```

Measurements:

- Mean FPS:
- CPU %:
- GPU %:
- RAM used:
- Swap used:
- VLM calls:
- Behavior cycles:
- Top hotspots (py-spy):

## Test 2: CV + VLM (Behavior Agent Off)

Config toggles:

```yaml
behavior_agent:
  enabled: false
verifier:
  enable_vlm: true
```

Measurements:

- Mean FPS:
- CPU %:
- GPU %:
- RAM used:
- Swap used:
- VLM calls (count):
- VLM mean duration:
- Behavior cycles:
- Top hotspots (py-spy):

## Test 3: Full Production-Like (All On)

Config toggles:

```yaml
behavior_agent:
  enabled: true
verifier:
  enable_vlm: true
```

Measurements:

- Mean FPS:
- CPU %:
- GPU %:
- RAM used:
- Swap used:
- VLM calls (count):
- VLM mean duration:
- Behavior cycles (count):
- Behavior mean duration:
- Top hotspots (py-spy):

## Decision Matrix Result

Matched row:

- [ ] CV healthy, LLM-dominated
- [ ] CV + LLM both heavy
- [ ] Foundational runtime issue (e.g. ONNX CPU EP fallback)
- [ ] Behavior-agent dominant
- [ ] VLM trigger-rate dominant

Decision:

- Proceed Phase 1 architecture optimizations: YES / NO
- Re-tune VLM escalation thresholds first: YES / NO
- Revisit LLM placement (local/cloud) first: YES / NO
- Notes:

## Phase Ordering for This Repo

1. 
2. 
3. 

## Re-Profile Summary (Post-Phase-1 / Post-Phase-2)

| Test | Before | After Phase 1 | After Phase 2 |
|---|---:|---:|---:|
| Test 1 (CV only) FPS |  |  |  |
| Test 2 (CV + VLM) FPS |  |  |  |
| Test 3 (Full) FPS |  |  |  |

## Final Baseline Notes

- Memory ceiling observations:
- Swap behavior:
- Thermal behavior:
- Operational recommendation:
