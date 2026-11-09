#!/usr/bin/env bash
# vLLM prefill/decode disaggregation WITHOUT Docker (2 GPUs, `pip install vllm nixl` in this env).
# This mirrors vLLM's own NixlConnector examples and is the most reliable way to run P/D.
# Usage:  MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/vllm_disagg_baremetal.sh
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
KV_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

common=(--served-model-name kvbench-model --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
        --enable-prefix-caching --enable-prompt-tokens-details --kv-transfer-config "$KV_CFG")

echo "starting prefill instance on GPU ${PREFILL_GPU:-0} :8100"
CUDA_VISIBLE_DEVICES=${PREFILL_GPU:-0} VLLM_SERVER_DEV_MODE=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  vllm serve "$MODEL" --port 8100 "${common[@]}" > "$LOG_DIR/prefill.log" 2>&1 &
P_PID=$!
echo "starting decode instance on GPU ${DECODE_GPU:-1} :8200"
CUDA_VISIBLE_DEVICES=${DECODE_GPU:-1} VLLM_SERVER_DEV_MODE=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  vllm serve "$MODEL" --port 8200 "${common[@]}" > "$LOG_DIR/decode.log" 2>&1 &
D_PID=$!
trap 'kill $P_PID $D_PID 2>/dev/null || true' EXIT INT TERM

for port in 8100 8200; do
  echo -n "waiting for :$port "
  until curl -sf "localhost:$port/health" > /dev/null; do
    sleep 5; echo -n "."
    if ! kill -0 $P_PID 2>/dev/null || ! kill -0 $D_PID 2>/dev/null; then
      echo; echo "a vLLM process died; see $LOG_DIR/"; exit 1
    fi
  done
  echo " ok"
done

echo "starting router on :8000 (Ctrl-C stops everything)"
kvbench router --config configs/router/vllm_disagg.yaml --port 8000
