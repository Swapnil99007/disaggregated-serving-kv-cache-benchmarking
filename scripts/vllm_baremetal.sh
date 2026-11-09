#!/usr/bin/env bash
# N vLLM replicas + the kvbench router WITHOUT Docker.
# Use this on container-based GPU clouds (RunPod, Vast.ai, ...) where Docker isn't available.
#
#   pip install vllm                 # ideally in its own venv; point VLLM_BIN at it
#   N=2 GPUS=0,0 bash scripts/vllm_baremetal.sh          # 2 replicas sharing GPU 0
#   N=2 GPUS=0,1 GPU_MEM_UTIL=0.85 bash scripts/vllm_baremetal.sh   # one replica per GPU
#
# Replicas listen on :PORT_BASE.. (default 9101; low 8xxx ports are often taken by the cloud
# image's own services, e.g. RunPod's nginx); the router on :ROUTER_PORT (default 8000).
# Ctrl-C stops everything. Run it inside tmux.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
N=${N:-2}
GPUS=${GPUS:-0,0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.40}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:---num-gpu-blocks-override 8192}
VLLM_BIN=${VLLM_BIN:-vllm}
KVBENCH_BIN=${KVBENCH_BIN:-kvbench}
PORT_BASE=${PORT_BASE:-9101}
ROUTER_PORT=${ROUTER_PORT:-8000}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
IFS=, read -ra GPU_LIST <<< "$GPUS"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM HUP

# Refuse to start if anything already answers on our ports (leftover run, or a service
# baked into the cloud image). curl succeeds on ANY HTTP response, so this catches both.
for port in "$ROUTER_PORT" $(seq "$PORT_BASE" $((PORT_BASE + N - 1))); do
  if curl -s -o /dev/null --max-time 2 "localhost:$port/"; then
    echo "ERROR: something is already listening on :$port"
    echo "  identify it:   curl -s -i localhost:$port/ | head -n 5"
    echo "  if it's an earlier run:  pkill -f vllm-env; pkill -f 'kvbench router'"
    echo "  otherwise pick other ports, e.g.  PORT_BASE=9201 ROUTER_PORT=8100 bash $0"
    exit 1
  fi
done

# router config with N localhost workers, derived from configs/router/vllm.yaml
ROUTER_CFG="$LOG_DIR/router_vllm_local.yaml"
sed '/^workers:/q' configs/router/vllm.yaml > "$ROUTER_CFG"

for ((i = 0; i < N; i++)); do
  port=$((PORT_BASE + i))
  gpu=${GPU_LIST[$i]:-${GPU_LIST[0]}}
  echo "starting vllm$i on GPU $gpu :$port (log: $LOG_DIR/vllm$i.log)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu VLLM_SERVER_DEV_MODE=1 "$VLLM_BIN" serve "$MODEL" \
    --host 0.0.0.0 --port "$port" --served-model-name kvbench-model \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching --enable-prompt-tokens-details $VLLM_EXTRA_ARGS \
    > "$LOG_DIR/vllm$i.log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo -n "  waiting for health "
  # Start replicas one at a time (overlapping memory profiling on a shared GPU fails), and
  # only accept OUR server: /v1/models must list the served model name.
  until curl -sf --max-time 5 "localhost:$port/v1/models" 2>/dev/null | grep -q kvbench-model; do
    if ! kill -0 "$pid" 2>/dev/null; then echo; echo "vllm$i died; see: tail -n 40 $LOG_DIR/vllm$i.log"; exit 1; fi
    sleep 5; echo -n "."
  done
  echo " ok"
  echo "  - {name: vllm$i, url: \"http://127.0.0.1:$port\", backend: vllm, role: both}" >> "$ROUTER_CFG"
done

echo "router config: $ROUTER_CFG"
echo "starting router on :$ROUTER_PORT — run benchmarks from another shell, e.g.:"
echo "  kvbench bench --experiment configs/experiments/vllm_quick.yaml --router-url http://localhost:$ROUTER_PORT"
"$KVBENCH_BIN" router --config "$ROUTER_CFG" --port "$ROUTER_PORT"
