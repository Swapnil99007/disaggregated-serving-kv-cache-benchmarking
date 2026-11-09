# Running on real GPUs

The simulator runs everywhere. Real vLLM/SGLang numbers need a Linux box with an NVIDIA GPU.

## 1. Pick hardware

| Experiment | Minimum | Notes |
|---|---|---|
| Routing policies (vLLM or SGLang) | 1× 24 GB GPU (L4, A10G, RTX 4090/3090) | 2 replicas of Qwen2.5-1.5B-Instruct share the GPU |
| 4-replica routing | 1× 80 GB (A100/H100) or 2–4× 24 GB | `--profile x4` + `ROUTER_CONFIG=configs/router/vllm_x4.yaml` |
| P/D disaggregation | 2 GPUs on one host | NixlConnector; NVLink helps but isn't required. Step by step: [RUN_DISAGG_2GPU.md](RUN_DISAGG_2GPU.md) |

Two kinds of cloud:

- **VM clouds** (Lambda Cloud, GCP, AWS g5/g6, Azure, Paperspace): you get a full VM →
  use the **Docker** path (§3).
- **Container clouds** (RunPod, Vast.ai): your "machine" is already a container, so you
  usually can't run Docker → use the **bare-metal scripts** (§4).

## 2. Copy the project to the box

```bash
# from your laptop
rsync -av --exclude .venv --exclude results ./ user@GPU_HOST:~/llm-serving-benchmarking/
ssh user@GPU_HOST
cd ~/llm-serving-benchmarking
nvidia-smi                          # driver works?
python3 --version                   # need >= 3.10
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[gpu,dev]"
pytest -q                           # sanity: 43 passed
```

## 3. Docker path (VM clouds)

Install Docker + the NVIDIA container toolkit (Ubuntu; skip what your image already has):

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi    # must print your GPU
```

Then:

```bash
cp .env.example .env               # edit MODEL, GPU ids, VLLM_EXTRA_ARGS
docker compose -f docker/compose.vllm.yml --env-file .env up --build -d
docker compose -f docker/compose.vllm.yml logs -f vllm0     # first start downloads weights (~3 GB)
curl localhost:8000/health                                   # {"healthy_workers": ["vllm0","vllm1"], ...}
kvbench smoke                                                # 6 requests, look for cached>0 on later turns
kvbench bench --experiment configs/experiments/vllm_quick.yaml
kvbench bench --experiment configs/experiments/vllm.yaml     # the full 1,000-prompt sweep (hours)
```

Run long sweeps inside `tmux` so an SSH drop doesn't kill them.

SGLang: `docker compose -f docker/compose.vllm.yml down`, then the same with
`docker/compose.sglang.yml` and `configs/experiments/sglang.yaml`.

Disaggregated vLLM (2 GPUs): `docker compose -f docker/compose.vllm-disagg.yml --env-file .env up --build -d`,
then `kvbench bench --experiment configs/experiments/vllm_disagg.yaml`. For an iso-hardware
comparison, also run the colocated stack with one replica per GPU (`VLLM0_GPU=0 VLLM1_GPU=1
GPU_MEM_UTIL=0.85`) and analyze both result directories together.

## 4. Bare-metal path (RunPod / Vast.ai / no Docker)

```bash
python3 -m venv ~/vllm-env
~/vllm-env/bin/pip install vllm==0.11.0 "transformers==4.56.2" "openai>=1.99.1,<2"
. .venv/bin/activate                                                  # kvbench env from §2
tmux new -s serve
VLLM_BIN=~/vllm-env/bin/vllm N=2 GPUS=0,0 bash scripts/vllm_baremetal.sh
# detach (Ctrl-b d), then in another shell:
kvbench bench --experiment configs/experiments/vllm_quick.yaml
```

Lessons from running this on RunPod:

- **Pin the libraries vLLM was built with.** An unpinned install pulls the newest transformers,
  and vLLM 0.11.0 then crashes at startup with
  `Qwen2Tokenizer has no attribute all_special_tokens_extended` (transformers 5.x removed it).
- **Replica ports default to 9101+.** Cloud images often run their own web services on low
  8xxx ports. The script refuses to start if a port already answers, and only accepts a replica
  whose `/v1/models` lists `kvbench-model`.
- **Copying output from inside tmux:** Cmd-A in your local terminal copies the history from
  *before* tmux started. Mouse-select the lines instead, or send files with `runpodctl send`.

SGLang: `pip install "sglang[all]"` in its own env, then
`SGLANG_PY=~/sglang-env/bin/python N=2 bash scripts/sglang_baremetal.sh`.

P/D disaggregation (2 GPUs): add `"nixl==0.6.0"` to the engine-env install above, then follow
**[RUN_DISAGG_2GPU.md](RUN_DISAGG_2GPU.md)**. It runs `scripts/vllm_disagg_baremetal.sh`
(`check`, `prefill`, `decode`, `router`, `verify`), one tmux window per GPU, plus a colocated
baseline on the same two GPUs.

## 5. Bring results home

```bash
# on your laptop
rsync -av user@GPU_HOST:~/llm-serving-benchmarking/results/ ./results/
kvbench analyze results/vllm/<ts> results/sglang/<ts> --baseline vllm-2w/round_robin
open results/vllm/<ts>/analysis/REPORT.md
```

## 6. Making the experiment meaningful

- **Cache pressure is the experiment.** If each replica's KV pool can hold every hot
  prefix, every policy gets ~100% hits and routing doesn't matter. `--num-gpu-blocks-override 8192`
  (vLLM) / `--max-total-tokens 131072` (SGLang) caps each replica at ~131K tokens. The
  trace's hot working set is larger than one replica's cache but fits across all of them,
  which is the regime where cache-aware routing pays off. Sweep this cap if you want a
  "benefit vs cache size" plot.
- **Keep `index_capacity_tokens`** in the router config equal to the per-replica KV cap.
- **Same GPU for every comparison.** Never compare a vLLM run on an A100 with an SGLang run on an L4.
- **Check prediction accuracy.** `GET /admin/state` reports `pred_cached_frac_mean` vs
  `actual_cached_frac_mean` per worker; a big gap means the index is stale (cache smaller than configured).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Second vLLM replica OOMs at startup | Lower `GPU_MEM_UTIL` (two replicas need ≤ 0.45 each on one GPU), make sure replicas start sequentially (the compose file does this) |
| `cached_tokens` missing / prefill columns show `–` | vLLM needs `--enable-prompt-tokens-details`; SGLang needs `--enable-cache-report`. The report still has the engine-counter cross-check |
| `/admin/reset` flush shows `404` for vLLM | Set `VLLM_SERVER_DEV_MODE=1` (enables `POST /reset_prefix_cache`) |
| KV usage column empty | Metric renamed in your vLLM version: check `curl localhost:8001/metrics \| grep cache_usage` and add the name to `BACKENDS` in `src/kvbench/router/workers.py` |
| SGLang second replica: "not enough memory" | Increase `SGL1_MEM_FRACTION` (fractions are of the whole GPU), or use one GPU per replica |
| Disagg: `x-kvbench-warning: prefill-returned-no-kv_transfer_params` | The prefill instance isn't running a KV connector; check `--kv-transfer-config` and that `nixl` is installed (`python -c "import nixl"`) |
| Every policy looks identical | No cache pressure (raise reuse or lower the KV cap), or the GPU is underloaded at your concurrency levels |
| HF download 401 | Gated model: set `HF_TOKEN` and accept the license on huggingface.co |
