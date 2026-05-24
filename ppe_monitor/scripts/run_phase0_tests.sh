#!/usr/bin/env bash
set -euo pipefail

# Run the three Phase-0 tests (CV-only, CV+VLM, full) with reproducible artifacts.
# Artifacts:
# - uvicorn logs
# - tegrastats logs
# - py-spy flame graphs (when available)
# - websocket metrics summaries
#
# Usage:
#   bash scripts/run_phase0_tests.sh
#   bash scripts/run_phase0_tests.sh --duration 120 --port 8000 --output-dir outputs/phase0_custom

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DURATION="120"
PORT="8000"
HOST="127.0.0.1"
OUTPUT_DIR=""
CONFIG_PATH="$ROOT_DIR/config.yaml"
FORCE_IMGSZ_640="1"
SKIP_PYSPY="0"
WS_NO_JPEG="0"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_phase0_tests.sh [options]

Options:
  --duration <seconds>      Duration per test (default: 120)
  --port <port>             Uvicorn port (default: 8000)
  --host <host>             Host used by websocket client (default: 127.0.0.1)
  --output-dir <path>       Output directory for artifacts
  --no-force-imgsz-640      Do not force inference.imgsz=640 during test runs
  --skip-pyspy              Do not run py-spy even if installed
  --ws-no-jpeg              Request JSON-only websocket stream (no JPEG frames)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-force-imgsz-640)
      FORCE_IMGSZ_640="0"
      shift
      ;;
    --skip-pyspy)
      SKIP_PYSPY="1"
      shift
      ;;
    --ws-no-jpeg)
      WS_NO_JPEG="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/outputs/phase0_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

if ! command -v curl >/dev/null 2>&1; then
  echo "[error] curl is required for health checks." >&2
  exit 1
fi
if ! python -c "import uvicorn" >/dev/null 2>&1; then
  echo "[error] uvicorn is not importable in the active Python environment." >&2
  exit 1
fi
if ! python -c "import websockets" >/dev/null 2>&1; then
  echo "[error] websockets package is required for ws_phase0_client.py." >&2
  exit 1
fi

CONFIG_BAK="$OUTPUT_DIR/config.yaml.backup"
cp "$CONFIG_PATH" "$CONFIG_BAK"

UVICORN_PID=""
TEGRA_PID=""
PYSPY_PID=""

cleanup() {
  set +e
  if [[ -n "$PYSPY_PID" ]]; then
    kill "$PYSPY_PID" >/dev/null 2>&1 || true
    wait "$PYSPY_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$TEGRA_PID" ]]; then
    kill "$TEGRA_PID" >/dev/null 2>&1 || true
    wait "$TEGRA_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$UVICORN_PID" ]]; then
    kill "$UVICORN_PID" >/dev/null 2>&1 || true
    wait "$UVICORN_PID" >/dev/null 2>&1 || true
  fi
  cp "$CONFIG_BAK" "$CONFIG_PATH"
}
trap cleanup EXIT INT TERM

have_sudo_nopass() {
  command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1
}

start_tegrastats() {
  local outfile="$1"
  if ! command -v tegrastats >/dev/null 2>&1; then
    echo "[warn] tegrastats not found; skipping system telemetry." | tee -a "$outfile"
    return 0
  fi
  if have_sudo_nopass; then
    sudo tegrastats --interval 500 --logfile "$outfile" >/dev/null 2>&1 &
  else
    tegrastats --interval 500 --logfile "$outfile" >/dev/null 2>&1 &
  fi
  TEGRA_PID="$!"
}

start_pyspy() {
  local uvicorn_pid="$1"
  local svg_out="$2"
  local log_out="$3"
  if [[ "$SKIP_PYSPY" == "1" ]]; then
    echo "[info] py-spy explicitly skipped." >"$log_out"
    return 0
  fi
  if ! command -v py-spy >/dev/null 2>&1; then
    echo "[warn] py-spy not found; skipping flame graph." >"$log_out"
    return 0
  fi
  if have_sudo_nopass; then
    sudo py-spy record -o "$svg_out" --pid "$uvicorn_pid" --duration "$DURATION" >"$log_out" 2>&1 &
  else
    py-spy record -o "$svg_out" --pid "$uvicorn_pid" --duration "$DURATION" >"$log_out" 2>&1 &
  fi
  PYSPY_PID="$!"
}

apply_test_config() {
  local behavior_enabled="$1"
  local enable_vlm="$2"
  local test_label="$3"
  python - "$CONFIG_PATH" "$behavior_enabled" "$enable_vlm" "$FORCE_IMGSZ_640" "$test_label" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
behavior_enabled = sys.argv[2].lower() == "true"
enable_vlm = sys.argv[3].lower() == "true"
force_imgsz_640 = sys.argv[4] == "1"
test_label = sys.argv[5]

cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
cfg.setdefault("behavior_agent", {})["enabled"] = behavior_enabled
cfg.setdefault("verifier", {})["enable_vlm"] = enable_vlm
if force_imgsz_640:
    cfg.setdefault("inference", {})["imgsz"] = 640
cfg.setdefault("_phase0_meta", {})
cfg["_phase0_meta"]["active_test"] = test_label
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
}

wait_for_health() {
  local health_url="http://127.0.0.1:${PORT}/health"
  for _ in $(seq 1 60); do
    if curl -fsS "$health_url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

stop_processes() {
  if [[ -n "$PYSPY_PID" ]]; then
    kill "$PYSPY_PID" >/dev/null 2>&1 || true
    wait "$PYSPY_PID" >/dev/null 2>&1 || true
    PYSPY_PID=""
  fi
  if [[ -n "$TEGRA_PID" ]]; then
    kill "$TEGRA_PID" >/dev/null 2>&1 || true
    wait "$TEGRA_PID" >/dev/null 2>&1 || true
    TEGRA_PID=""
  fi
  if [[ -n "$UVICORN_PID" ]]; then
    kill "$UVICORN_PID" >/dev/null 2>&1 || true
    wait "$UVICORN_PID" >/dev/null 2>&1 || true
    UVICORN_PID=""
  fi
}

run_test() {
  local test_name="$1"
  local behavior_enabled="$2"
  local enable_vlm="$3"
  local test_dir="$OUTPUT_DIR/$test_name"
  mkdir -p "$test_dir"

  echo "=== Running ${test_name} ==="
  apply_test_config "$behavior_enabled" "$enable_vlm" "$test_name"
  cp "$CONFIG_PATH" "$test_dir/config_used.yaml"

  python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >"$test_dir/uvicorn.log" 2>&1 &
  UVICORN_PID="$!"

  if ! wait_for_health; then
    echo "[error] ${test_name}: health endpoint did not become ready." | tee -a "$test_dir/uvicorn.log"
    stop_processes
    return 1
  fi

  start_tegrastats "$test_dir/tegrastats.log"
  start_pyspy "$UVICORN_PID" "$test_dir/profile.svg" "$test_dir/pyspy.log"

  local ws_jpeg_args=()
  if [[ "$WS_NO_JPEG" == "1" ]]; then
    ws_jpeg_args+=(--no-jpeg)
  fi

  python scripts/ws_phase0_client.py \
    --url "ws://${HOST}:${PORT}/ws/stream" \
    --duration "$DURATION" \
    --output "$test_dir/ws_summary.json" \
    --metrics-jsonl "$test_dir/ws_metrics.jsonl" \
    "${ws_jpeg_args[@]}" \
    >"$test_dir/ws_client.log" 2>&1

  stop_processes

  python - "$test_dir/ws_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("[warn] ws summary missing")
    raise SystemExit(0)
doc = json.loads(path.read_text(encoding="utf-8"))
print(
    f"[summary] fps_mean={doc.get('mean_fps_from_payload', 0.0):.2f} "
    f"json_rate={doc.get('json_rate_hz', 0.0):.2f} "
    f"jpeg_rate={doc.get('jpeg_rate_hz', 0.0):.2f}"
)
PY
}

cat >"$OUTPUT_DIR/run_manifest.txt" <<EOF
phase0_run_started=$(date -Iseconds)
duration_s=$DURATION
host=$HOST
port=$PORT
output_dir=$OUTPUT_DIR
force_imgsz_640=$FORCE_IMGSZ_640
skip_pyspy=$SKIP_PYSPY
ws_no_jpeg=$WS_NO_JPEG
EOF

run_test "test1_cv_only" "false" "false"
run_test "test2_cv_plus_vlm" "false" "true"
run_test "test3_full_system" "true" "true"

echo "Phase-0 run completed. Artifacts stored in: $OUTPUT_DIR"
