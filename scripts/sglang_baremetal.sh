#!/usr/bin/env bash
# N SGLang replicas + the kvbench router WITHOUT Docker (RunPod, Vast.ai, ...).
#
#   pip install "sglang[all]"        # ideally in its own venv; point SGLANG_PY at its python
#   N=2 GPUS=0,0 MEM_FRACTIONS=0.40,0.80 bash scripts/sglang_baremetal.sh   # 2 replicas, one GPU
#   N=2 GPUS=0,1 MEM_FRACTIONS=0.85,0.85 bash scripts/sglang_baremetal.sh   # one replica per GPU
#
# SGLang sizes memory as a fraction of the WHOLE GPU, so a second replica on an already-used
# GPU needs a larger fraction. --max-total-tokens pins the KV pool so replicas are comparable.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
N=${N:-2}
GPUS=${GPUS:-0,0}
MEM_FRACTIONS=${MEM_FRACTIONS:-0.40,0.80}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
SCHEDULE_POLICY=${SCHEDULE_POLICY:-fcfs}
SGLANG_EXTRA_ARGS=${SGLANG_EXTRA_ARGS:---max-total-tokens 131072}
SGLANG_PY=${SGLANG_PY:-python3}
KVBENCH_BIN=${KVBENCH_BIN:-kvbench}
PORT_BASE=${PORT_BASE:-9101}
ROUTER_PORT=${ROUTER_PORT:-8000}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
IFS=, read -ra GPU_LIST <<< "$GPUS"
IFS=, read -ra MF_LIST <<< "$MEM_FRACTIONS"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM HUP

# refuse to start if anything already answers on our ports (leftovers or the image's own services)
for port in "$ROUTER_PORT" $(seq "$PORT_BASE" $((PORT_BASE + N - 1))); do
  if curl -s -o /dev/null --max-time 2 "localhost:$port/"; then
    echo "ERROR: something is already listening on :$port  (identify: curl -s -i localhost:$port/ | head -n 5)"
    echo "  earlier run? pkill -f sglang; pkill -f 'kvbench router'   otherwise: PORT_BASE=9201 ROUTER_PORT=8100 bash $0"
    exit 1
  fi
done

ROUTER_CFG="$LOG_DIR/router_sglang_local.yaml"
sed '/^workers:/q' configs/router/sglang.yaml > "$ROUTER_CFG"

for ((i = 0; i < N; i++)); do
  port=$((PORT_BASE + i))
  gpu=${GPU_LIST[$i]:-${GPU_LIST[0]}}
  mf=${MF_LIST[$i]:-${MF_LIST[0]}}
  echo "starting sglang$i on GPU $gpu :$port mem_fraction=$mf (log: $LOG_DIR/sglang$i.log)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu "$SGLANG_PY" -m sglang.launch_server --model-path "$MODEL" \
    --host 0.0.0.0 --port "$port" --served-model-name kvbench-model \
    --mem-fraction-static "$mf" --context-length "$MAX_MODEL_LEN" --max-running-requests "$MAX_NUM_SEQS" \
    --schedule-policy "$SCHEDULE_POLICY" --enable-metrics --enable-cache-report $SGLANG_EXTRA_ARGS \
    > "$LOG_DIR/sglang$i.log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo -n "  waiting for health "
  # only accept OUR server: /v1/models must list the served model name
  until curl -sf --max-time 5 "localhost:$port/v1/models" 2>/dev/null | grep -q kvbench-model; do
    if ! kill -0 "$pid" 2>/dev/null; then echo; echo "sglang$i died; see: tail -n 40 $LOG_DIR/sglang$i.log"; exit 1; fi
    sleep 5; echo -n "."
  done
  echo " ok"
  echo "  - {name: sglang$i, url: \"http://127.0.0.1:$port\", backend: sglang, role: both}" >> "$ROUTER_CFG"
done

echo "router config: $ROUTER_CFG"
echo "starting router on :$ROUTER_PORT — then: kvbench bench --experiment configs/experiments/sglang.yaml --router-url http://localhost:$ROUTER_PORT"
"$KVBENCH_BIN" router --config "$ROUTER_CFG" --port "$ROUTER_PORT"
