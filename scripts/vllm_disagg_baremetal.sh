#!/usr/bin/env bash
# vLLM prefill/decode disaggregation (NixlConnector) on ONE host with 2 GPUs, without Docker.
#
# Run each part in its own tmux window (full walkthrough: docs/RUN_DISAGG_2GPU.md):
#
#   bash scripts/vllm_disagg_baremetal.sh check     # pre-flight: GPUs, versions, NIXL, ports
#   bash scripts/vllm_disagg_baremetal.sh prefill   # prefill instance on GPU 0 (stays in foreground)
#   bash scripts/vllm_disagg_baremetal.sh decode    # decode instance on GPU 1 (stays in foreground)
#   bash scripts/vllm_disagg_baremetal.sh router    # waits for both instances, then runs the router
#   bash scripts/vllm_disagg_baremetal.sh verify    # proves KV really moves between the GPUs
#
# Engine env (its own venv; pinned to the versions vLLM 0.11.0 was released with):
#   python3 -m venv ~/vllm-env
#   ~/vllm-env/bin/pip install vllm==0.11.0 "transformers==4.56.2" "openai>=1.99.1,<2" "nixl==0.6.0" hf_transfer
#
# Engine settings match scripts/vllm_baremetal.sh (model, KV cap of 8192 blocks per instance,
# max-model-len, max-num-seqs) so colocated and disaggregated runs on the same GPUs are comparable.
# Every setting below can be overridden with an environment variable of the same name.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
# one instance per GPU; the KV cache size is fixed by --num-gpu-blocks-override, so this only
# sets headroom (0.40 is too tight on 24 GB cards). Use the same value for the colocated phase.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:---num-gpu-blocks-override 8192}
VLLM_BIN=${VLLM_BIN:-$HOME/vllm-env/bin/vllm}
VLLM_PY=${VLLM_PY:-$(dirname "$VLLM_BIN")/python}
KVBENCH_BIN=${KVBENCH_BIN:-kvbench}
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}
# HTTP ports start at 9101: low 8xxx ports are often taken by the cloud image (RunPod's nginx)
PREFILL_PORT=${PREFILL_PORT:-9101}
DECODE_PORT=${DECODE_PORT:-9102}
PREFILL_NIXL_PORT=${PREFILL_NIXL_PORT:-5600}   # NIXL handshake (side channel); unique per instance
DECODE_NIXL_PORT=${DECODE_NIXL_PORT:-5601}
ROUTER_PORT=${ROUTER_PORT:-8000}
LOG_DIR=${LOG_DIR:-logs}
KV_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
mkdir -p "$LOG_DIR"

port_free() {   # true if nothing is bound to the port (catches HTTP servers and NIXL/ZMQ sockets)
  python3 -c '
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(1)' "$1"
}

require_free() {
  if ! port_free "$1"; then
    echo "ERROR: port $1 is already in use."
    echo "  Leftover run?  pkill -f vllm-env; pkill -f 'kvbench router'   (then retry)"
    echo "  Something else? pick other ports, e.g. PREFILL_PORT=9201 DECODE_PORT=9202 for ALL subcommands"
    exit 1
  fi
}

ready() {   # only accept OUR server: /v1/models must list the served model name
  curl -sf --max-time 5 "localhost:$1/v1/models" 2>/dev/null | grep -q kvbench-model
}

cmd_check() {
  local ok=1 n
  echo "== GPUs"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || ok=0
  n=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$n" -lt 2 ]; then echo "  FAIL: need 2 GPUs, found $n"; ok=0; fi
  echo "== GPU interconnect (informational)"
  nvidia-smi topo -m 2>/dev/null | head -n 3 || echo "  (topology not available in this container)"
  echo "== engine env: $VLLM_PY"
  if [ ! -x "$VLLM_BIN" ]; then
    echo "  FAIL: $VLLM_BIN not found; run the install step first"
    ok=0
  else
    "$VLLM_PY" - <<'EOF' || ok=0
import importlib.metadata as md
import sys

want = {"vllm": "0.11.0", "transformers": "4.56.2", "nixl": "0.6.0"}
bad = False
for pkg, v in want.items():
    try:
        have = md.version(pkg)
    except md.PackageNotFoundError:
        have = None
    status = "ok" if have == v else "FAIL"
    bad |= status != "ok"
    print(f"  {pkg:13s} {have} (want {v})  {status}")
import os
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").lower() in ("1", "true", "yes"):
    try:
        import hf_transfer  # noqa: F401
        print("  hf_transfer   ok (HF_HUB_ENABLE_HF_TRANSFER is set)")
    except ImportError:
        print("  hf_transfer   FAIL: HF_HUB_ENABLE_HF_TRANSFER=1 but the package is missing;"
              " model downloads will fail. Fix: ~/vllm-env/bin/pip install hf_transfer")
        bad = True
try:
    from nixl._api import nixl_agent  # noqa: F401  (what vLLM 0.11.0's NixlConnector imports)
    print("  nixl import   ok")
except Exception as e:  # noqa: BLE001
    print(f"  nixl import   FAIL: {e}")
    bad = True
import torch

n = torch.cuda.device_count()
print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}, sees {n} GPUs")
if n >= 2:
    print(f"  GPU0 <-> GPU1 peer access: {torch.cuda.can_device_access_peer(0, 1)} (informational)")
sys.exit(1 if bad else 0)
EOF
  fi
  echo "== kvbench"
  if command -v "$KVBENCH_BIN" >/dev/null; then echo "  ok ($(command -v "$KVBENCH_BIN"))"
  else echo "  FAIL: kvbench not found; activate the project venv:  . .venv/bin/activate"; ok=0; fi
  if grep -q "X-Request-Id" src/kvbench/router/app.py; then echo "  router sends a shared X-Request-Id: ok"
  else echo "  FAIL: old router code (no shared X-Request-Id); run: git pull"; ok=0; fi
  echo "== ports"
  for p in "$PREFILL_PORT" "$DECODE_PORT" "$PREFILL_NIXL_PORT" "$DECODE_NIXL_PORT" "$ROUTER_PORT"; do
    if port_free "$p"; then echo "  :$p free"; else echo "  :$p IN USE  FAIL"; ok=0; fi
  done
  if [ "$ok" = 1 ]; then echo; echo "ALL CHECKS PASSED"; else echo; echo "SOME CHECKS FAILED (see FAIL lines above)"; exit 1; fi
}

serve() {   # serve <role> <gpu> <http port> <nixl port>
  local role=$1 gpu=$2 port=$3 nixl_port=$4
  [ -x "$VLLM_BIN" ] || { echo "ERROR: $VLLM_BIN not found; run the install step"; exit 1; }
  require_free "$port"
  require_free "$nixl_port"
  echo "starting the $role instance: GPU $gpu, HTTP :$port, NIXL handshake :$nixl_port"
  echo "log is also saved to $LOG_DIR/$role.log; Ctrl-C stops it"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu VLLM_SERVER_DEV_MODE=1 VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port UCX_NET_DEVICES=all \
    "$VLLM_BIN" serve "$MODEL" --host 0.0.0.0 --port "$port" --served-model-name kvbench-model \
      --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
      --enable-prefix-caching --enable-prompt-tokens-details $VLLM_EXTRA_ARGS \
      --kv-transfer-config "$KV_CFG" 2>&1 | tee "$LOG_DIR/$role.log"
}

cmd_router() {
  local name port t0 cfg
  require_free "$ROUTER_PORT"
  for spec in "prefill:$PREFILL_PORT" "decode:$DECODE_PORT"; do
    name=${spec%%:*}
    port=${spec##*:}
    echo -n "waiting for the $name instance on :$port "
    t0=$SECONDS
    until ready "$port"; do
      if (( SECONDS - t0 > 1800 )); then
        echo; echo "ERROR: $name instance not ready after 30 min; look at its tmux window or $LOG_DIR/$name.log"; exit 1
      fi
      sleep 5; echo -n "."
    done
    echo " ok"
  done
  cfg="$LOG_DIR/router_vllm_disagg_local.yaml"
  sed '/^workers:/q' configs/router/vllm_disagg.yaml > "$cfg"
  echo "  - {name: prefill0, url: \"http://127.0.0.1:$PREFILL_PORT\", backend: vllm, role: prefill}" >> "$cfg"
  echo "  - {name: decode0, url: \"http://127.0.0.1:$DECODE_PORT\", backend: vllm, role: decode}" >> "$cfg"
  echo "router config: $cfg"
  echo "starting the router on :$ROUTER_PORT (log also in $LOG_DIR/router_disagg.log); Ctrl-C stops it"
  "$KVBENCH_BIN" router --config "$cfg" --port "$ROUTER_PORT" 2>&1 | tee "$LOG_DIR/router_disagg.log"
}

cmd_verify() {
  local ok=1 hdrs n_err xfer
  echo "== router"
  curl -sf --max-time 5 "localhost:$ROUTER_PORT/health" || { echo "  FAIL: router not answering on :$ROUTER_PORT"; exit 1; }
  echo
  echo "== one request, router response headers"
  hdrs=$(curl -s -D - -o /dev/null --max-time 300 "localhost:$ROUTER_PORT/v1/completions" \
    -H 'content-type: application/json' \
    -d '{"model": "kvbench-model", "prompt": "Explain what a KV cache is in two sentences.", "max_tokens": 16}' \
    | tr -d '\r' | grep -i '^x-kvbench' || true)
  echo "$hdrs" | sed 's/^/  /'
  if echo "$hdrs" | grep -qi 'x-kvbench-warning'; then echo "  FAIL: the prefill instance returned no KV-transfer info"; ok=0; fi
  if ! echo "$hdrs" | grep -qi 'x-kvbench-decode-worker'; then echo "  FAIL: request did not go through the prefill->decode path"; ok=0; fi
  echo "== 6 conversation turns through the router"
  "$KVBENCH_BIN" smoke --router-url "http://localhost:$ROUTER_PORT" || ok=0
  echo "== waiting 15 s for vLLM's periodic stats line ..."
  sleep 15
  echo "== NIXL transfers seen by the decode instance (must be > 0)"
  xfer=$(grep "KV Transfer metrics" "$LOG_DIR/decode.log" | tail -n 3 || true)
  if [ -n "$xfer" ]; then echo "$xfer" | sed 's/^/  /'; else echo "  FAIL: no 'KV Transfer metrics' line in $LOG_DIR/decode.log"; ok=0; fi
  echo "== request-id mismatches on the prefill instance (must be 0)"
  n_err=$(grep -c -E "Releasing expired KV blocks|unrecognized request" "$LOG_DIR/prefill.log" || true)
  echo "  $n_err"
  [ "${n_err:-0}" = 0 ] || ok=0
  if [ "$ok" = 1 ]; then echo; echo "VERIFY PASSED: KV is moving from GPU $PREFILL_GPU to GPU $DECODE_GPU over NIXL"
  else echo; echo "VERIFY FAILED (see FAIL lines above)"; exit 1; fi
}

case "${1:-}" in
  check)   cmd_check ;;
  prefill) serve prefill "$PREFILL_GPU" "$PREFILL_PORT" "$PREFILL_NIXL_PORT" ;;
  decode)  serve decode "$DECODE_GPU" "$DECODE_PORT" "$DECODE_NIXL_PORT" ;;
  router)  cmd_router ;;
  verify)  cmd_verify ;;
  *) echo "usage: bash $0 {check|prefill|decode|router|verify}"; exit 2 ;;
esac
