# Disaggregated prefill/decode on 2 real GPUs: step by step (RunPod)

This runbook measures disaggregated serving on real hardware and compares it with ordinary
(colocated) serving **on the same two GPUs, with the same engine settings**.

| Phase | GPU 0 | GPU 1 | Result folder |
|---|---|---|---|
| **A. Disaggregated (1P + 1D)** | prefill server: reads every prompt | decode server: writes every answer, pulling the KV over NIXL (staged through host memory, see below) | `results/vllm_disagg/<timestamp>` |
| **B. Colocated (2 replicas)** | full replica 0 | full replica 1 | `results/vllm_2gpu/<timestamp>` |

> **KV transfer path.** By default the KV is handed over **through host memory**
> (`KV_PATH=staged`, which sets `UCX_TLS=tcp,cuda_copy`): each GPU only copies to and from its
> own host RAM. The direct path (`KV_PATH=direct`) has the decode GPU *read* the prefill GPU's
> memory peer-to-peer. On two different RunPod hosts (2× A40, then 2× A6000, both with GPUs
> behind PCIe bridges), that took the decode GPU offline (`nvidia-smi` showed `ERR!`) within
> about 30 seconds of sweep load, even though the system reported peer access as available.
> Use `direct` only on GPUs known to support it, such as NVLink-connected ones. Staged transfers
> add tens of milliseconds per cold prompt, so report the path along with the results.

Both phases replay the same 1,000-prompt trace at 8–64 sessions with 3 repeats, using the
`cache_aware` router. The engine settings match the single-GPU runs: Qwen2.5-1.5B-Instruct,
vLLM 0.11.0, a KV cap of 8,192 blocks per instance, `--max-num-seqs 64`, and
`--max-model-len 16384`. Both phases use `GPU_MEM_UTIL=0.80`. With the KV cache size fixed,
that only sets memory headroom, not speed.

**Budget: about 3 hours**: roughly $3.50 on 2× RTX A6000 or 2× RTX PRO 4000, and about $15 on 2× RTX PRO 6000.

| Step | Time |
|---|---|
| 1. Rent the pod | 10 min |
| 2. Install and pre-flight check | 20–25 min |
| 3. Open the tmux windows | 2 min |
| 4. Phase A: start servers, verify, soak test, run the sweep | 65–85 min |
| 5. Switch phases | 3 min |
| 6. Phase B: start replicas, run the sweep | 45–60 min |
| 7. Copy results home, terminate | 5 min |

Commands below contain no `#` comments, so every block can be pasted as is. Run each block in
the window named above it.

---

## 1. Rent a pod with 2 GPUs

1. Open the RunPod console and go to **Pods → Deploy**.
2. Pick the GPU, and set **GPU count to 2**. Only GPU types showing **"2 max" or more** can do
   this. Both GPUs must be in **one pod**; two separate pods won't work. In order of preference:
   1. **2× RTX A6000** or **2× A40**: the same GPU class as the single-GPU results.
   2. **2× RTX PRO 4000** (24 GB): similar speed to the A6000, and the cheapest option.
   3. 2× L40S or 2× RTX 6000 Ada.
   4. 2× RTX PRO 6000 (96 GB): works, but it's 2–3× faster than the A6000, so the 1.5B model may
      not get busy enough at 64 sessions for a clear comparison.

   **Avoid:**
   - **consumer cards** (RTX 3090, 4090, 5090). NIXL moves KV memory directly between the two
     GPUs, and NVIDIA blocks GPU-to-GPU transfers on consumer cards;
   - **"MIG" entries**. These are slices of one physical GPU, not two GPUs, and transfers between
     slices aren't supported;
   - **H100, H200, B200**. They work, but cost 5–15× more and are far bigger than this model needs.
3. If there's a CUDA version filter, choose **12.8 or newer**. vLLM 0.11.0's wheels need it.
4. **Template:** RunPod PyTorch.
5. **Edit the pod settings:**
   - set the container disk to **80 GB** if the GPUs stay available at that size, otherwise
     **50 GB** (enough, with the cache-free install below);
   - leave the volume disk at 0;
   - no extra ports are needed.
6. Choose **On-Demand**, not Spot. A Spot pod can be taken away in the middle of a run.
7. Click **Deploy** and wait for **Running**. Then click **Connect** and copy the **SSH** command.

> The container disk is **wiped when the pod stops**. Results only survive if you copy them
> home (step 7) first.

### Renting on Vast.ai instead

Vast.ai works too. Everything from step 2 on is the same, except for the points below.

1. **Before renting:** add your SSH public key under **Account → SSH Keys**. On a Mac, print
   it with `cat ~/.ssh/id_ed25519.pub`. If that file doesn't exist, create a key with
   `ssh-keygen -t ed25519`.
2. **Filters:** GPU count **2**, GPU type RTX A6000 / A40 / L40S, **Max CUDA ≥ 12.8**,
   **verified**, reliability above 98%, **on-demand** (not interruptible).
3. **Template:** a PyTorch image with CUDA 12.8 or newer, launch mode **SSH**, disk **60–80 GB**.
   The disk size can't be changed after renting.
4. **Vast starts tmux automatically when you log in.** Turn that off once, so the guide's own
   tmux session isn't nested inside it:

   ```bash
   touch ~/.no_auto_tmux
   exit
   ```

   Then connect again.
5. If `python3 -m venv` says `ensurepip is not available`, run:

   ```bash
   apt-get update && apt-get install -y python3-venv
   ```

6. **Step 7:** there's no `runpodctl`. Copy the results from your **Mac** with the port and IP
   shown on the instance card:

   ```bash
   scp -P PORT root@IP:/tmp/pd_results.tgz ~/Documents/Projects/llm-serving-benchmarking/
   ```

7. **When you're done:** click **Destroy**, not just Stop. A stopped instance keeps charging for
   its disk.

## 2. Install and pre-flight check

**On your Mac**, paste the SSH command from RunPod.

**On the pod:**

```bash
nvidia-smi -L
```

You must see **two** lines, `GPU 0: ...` and `GPU 1: ...`. If you see one, you rented a
single-GPU pod; go back to step 1.

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/Swapnil99007/disaggregated-serving-kv-cache-benchmarking.git
cd disaggregated-serving-kv-cache-benchmarking
git log --oneline -5
```

One of the lines must mention **2-GPU disaggregation**. That's the commit with the fixed router and
launch script. The `check` step below confirms it as well.

Start tmux now, so a dropped SSH connection can't kill anything:

```bash
tmux new -s pd -n gpu0
```

Inside tmux (window `gpu0`), install the engine environment. This takes about 5–10 minutes:

```bash
cd /workspace/disaggregated-serving-kv-cache-benchmarking
python3 -m venv ~/vllm-env
~/vllm-env/bin/pip install --no-cache-dir -U pip
~/vllm-env/bin/pip install --no-cache-dir vllm==0.11.0 "transformers==4.56.2" "openai>=1.99.1,<2" "nixl==0.6.0" hf_transfer
```

`hf_transfer` is there because RunPod's PyTorch template sets `HF_HUB_ENABLE_HF_TRANSFER=1`.
Without the package, every model download fails with `Can't load the configuration of 'Qwen/...'`.
The versions are pinned on purpose. vLLM 0.11.0 crashes with newer `transformers`, and NIXL 0.6.0
is the release that was current when vLLM 0.11.0 shipped.

Then install the project itself in its own environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --no-cache-dir -e ".[gpu,dev]"
pytest -q
df -h /
```

Expected: **43 passed**, and `df` should show at least 15 GB still available.

```bash
kvbench gen-trace
```

The printed statistics must include **`"num_conversations": 369`** and
**`"prompt_chars_total": 18047003.0`**. That confirms this is the same trace as the earlier runs.

```bash
bash scripts/vllm_disagg_baremetal.sh check
```

Expected: **`ALL CHECKS PASSED`**. It checks:
- two GPUs are present;
- vLLM, transformers and NIXL are installed at the pinned versions, and NIXL imports;
- the router has the request-ID fix;
- all five ports are free.

The "peer access" line is informational. With the default staged KV path, GPU-to-GPU access
isn't used at all.

`bash scripts/vllm_disagg_baremetal.sh p2ptest` exists for anyone trying `KV_PATH=direct`. It
only exercises peer *writes*, though, and it passed on both hosts whose GPUs later failed under
peer *reads*. So it can't vouch for the direct path. Skip it with the default path.

## 3. Open the tmux windows

There are five windows. Each GPU gets its own window, so you can watch its server.

| # | Window | Phase A | Phase B |
|---|---|---|---|
| 0 | `gpu0` | prefill server (GPU 0) | replicas + router launcher |
| 1 | `gpu1` | decode server (GPU 1) | log of replica 1 (GPU 1) |
| 2 | `router` | router | log of replica 0 (GPU 0) |
| 3 | `bench` | verify + sweep | smoke + sweep |
| 4 | `monitor` | `nvidia-smi` | `nvidia-smi` |

Create them from window `gpu0`:

```bash
tmux new-window -d -n gpu1 -c /workspace/disaggregated-serving-kv-cache-benchmarking
tmux new-window -d -n router -c /workspace/disaggregated-serving-kv-cache-benchmarking
tmux new-window -d -n bench -c /workspace/disaggregated-serving-kv-cache-benchmarking
tmux new-window -d -n monitor -c /workspace/disaggregated-serving-kv-cache-benchmarking
```

**tmux keys** (press Ctrl-b, release it, then press the next key):

| Keys | Action |
|---|---|
| Ctrl-b then `0` … `4` | Switch to window 0 … 4 (the bottom bar shows which one you're in) |
| Ctrl-b then `d` | Detach; everything keeps running. Reattach with `tmux attach -t pd` |
| Ctrl-b then `[` | Scroll mode: arrow keys or PgUp; press `q` to leave |

**Window 4 (`monitor`):**

```bash
watch -n 2 nvidia-smi
```

Leave it running. It shows memory use and utilization for both GPUs.

## 4. Phase A: disaggregated, prefill on GPU 0 and decode on GPU 1

**Window 0 (`gpu0`):**

```bash
bash scripts/vllm_disagg_baremetal.sh prefill
```

**Window 1 (`gpu1`):**

```bash
bash scripts/vllm_disagg_baremetal.sh decode
```

Each takes 2–4 minutes the first time, to download the weights and compile CUDA graphs. Wait
until **both** windows show `Application startup complete.` The prefill window should also have
printed `NIXL is available` near the start.

**Window 2 (`router`):**

```bash
. .venv/bin/activate
bash scripts/vllm_disagg_baremetal.sh router
```

It waits for both servers, then prints `router up: mode=disagg`.

**Window 3 (`bench`):**

```bash
. .venv/bin/activate
bash scripts/vllm_disagg_baremetal.sh verify
```

This takes about 30 seconds and must end with **`VERIFY PASSED: KV is moving from GPU 0 to GPU 1 over NIXL`**.
It checks four things:
- requests take the prefill → decode path;
- the prefill server returns KV-transfer info;
- the decode server's log reports `num_successful_transfers` above 0;
- the prefill server logged no request-ID mismatches.

**Don't continue until verify passes.** If it fails, see
[Troubleshooting](#troubleshooting).

Then run about 2 minutes of real load (400 prompts at 16 sessions), followed by a GPU health
check. Both earlier GPU failures happened within 30 seconds of load like this, so this is where
you find out, not an hour into the sweep. Still in **window 3**:

```bash
bash scripts/vllm_disagg_baremetal.sh soak
```

It must end with **`SOAK PASSED: safe to start the full sweep`**. Its results go to
`results/vllm_disagg_soak/` and aren't used in any report.

Then, still in **window 3**, run the sweep (18 runs, about 50–70 minutes):

```bash
kvbench bench --experiment configs/experiments/vllm_disagg.yaml
```

Each run prints a summary line. It's finished when it prints
`report: results/vllm_disagg/<timestamp>/analysis/REPORT.md`. You can detach (Ctrl-b `d`)
and come back later.

Health check while it runs, from any free shell (for example, press Ctrl-C in `monitor`, then
restart `watch` afterwards):

```bash
grep -c "Releasing expired KV blocks" logs/prefill.log
```

This must stay **0**. Also check `nvidia-smi` in window 4: neither GPU may show `ERR!`. And
window 3 should print a progress line (`... 100/1000 (10%)`) every couple of minutes.

## 5. Switch from Phase A to Phase B

Stop the Phase A processes in this order, with **Ctrl-C** in each window:
1. window 2 (`router`);
2. window 1 (`gpu1`);
3. window 0 (`gpu0`).

In window 4 (`monitor`), both GPUs should drop to about 0 MiB of memory used within a few
seconds. If anything is still using memory, run this in window 3:

```bash
pkill -f vllm-env; pkill -f "kvbench router"
```

## 6. Phase B: colocated, one full replica per GPU

**Window 0 (`gpu0`):**

```bash
. .venv/bin/activate
GPU_MEM_UTIL=0.80 VLLM_BIN=~/vllm-env/bin/vllm N=2 GPUS=0,1 bash scripts/vllm_baremetal.sh
```

This starts replica 0 on GPU 0, then replica 1 on GPU 1, then the router, all from this window.
It ends with `router up: mode=colocated`. It uses the same engine settings as Phase A, minus the
KV connector.

**Window 1 (`gpu1`)**, to watch replica 1:

```bash
tail -f logs/vllm1.log
```

**Window 2 (`router`)**, to watch replica 0:

```bash
tail -f logs/vllm0.log
```

**Window 3 (`bench`):**

```bash
kvbench smoke
```

All 6 lines should show `status=ok`, and the later turns of a conversation should show
`cached=` above 0.

```bash
kvbench bench --experiment configs/experiments/vllm_2gpu.yaml
```

That's 18 runs, about 45–60 minutes. It's finished when it prints
`report: results/vllm_2gpu/<timestamp>/analysis/REPORT.md`.

**Optional, about 45–60 more minutes (about $1):** round-robin on the same setup. This measures
routing gains with each replica on its own GPU, instead of two replicas sharing one GPU as in
the single-GPU results.

```bash
kvbench bench --experiment configs/experiments/vllm_2gpu.yaml --policies round_robin
```

## 7. Copy the results home, then terminate

**Window 3 (`bench`):**

```bash
cd /workspace/disaggregated-serving-kv-cache-benchmarking
tar czf /tmp/pd_results.tgz results/vllm_disagg results/vllm_2gpu logs
runpodctl send /tmp/pd_results.tgz
```

It prints a code like `1234-alpha-bravo-charlie`. **On your Mac:**

```bash
cd ~/Documents/Projects/llm-serving-benchmarking
runpodctl receive 1234-alpha-bravo-charlie
```

Wait for the Mac to finish receiving the file, then terminate the pod in the RunPod console:
**Pods → ⋮ → Terminate**.

The combined comparison report is built from the received files:

```bash
kvbench analyze results/vllm_2gpu/<ts> results/vllm_disagg/<ts> --baseline vllm-2gpu-2w/cache_aware --out results/compare_pd_gpu
```

---

## Troubleshooting

**Collect diagnostics first.** This lets someone else look at the logs:

```bash
tail -n 80 logs/prefill.log logs/decode.log logs/router_disagg.log > /tmp/diag.txt
runpodctl send /tmp/diag.txt
```

| Symptom | Cause | Fix |
|---|---|---|
| `check`: a version shows FAIL | Unpinned or partial install | Rerun the `~/vllm-env/bin/pip install ...` line from step 2 exactly |
| `check`: `nixl import FAIL` | NIXL wheel missing or broken | `~/vllm-env/bin/pip install --force-reinstall "nixl==0.6.0"` |
| A server exits with `Can't load the configuration of 'Qwen/Qwen2.5-1.5B-Instruct'` | The template sets `HF_HUB_ENABLE_HF_TRANSFER=1`, but `hf_transfer` isn't installed | `~/vllm-env/bin/pip install --no-cache-dir hf_transfer`, then restart both servers |
| `check`: `old router code` | The clone is from before the fix | `git pull`, then rerun `check` |
| `ERROR: port ... is already in use` | A leftover server or router | `pkill -f vllm-env; pkill -f "kvbench router"`, then retry |
| A server window exits with a traceback | Usually NIXL/UCX setup, or CUDA-graph capture with the connector | Collect diagnostics. Then try the eager fallback below |
| verify: `prefill instance returned no KV-transfer info` | The connector isn't active on the prefill server | Check that window 0 printed `NIXL is available`. If not, fix the NIXL install |
| verify: no `KV Transfer metrics` line | Transfers failing, or the stats line hasn't been printed yet | Wait 30 s and run `grep "KV Transfer metrics" logs/decode.log`. If still nothing, collect diagnostics; check that windows 0 and 1 printed `KV_PATH=staged UCX_TLS=tcp,cuda_copy` |
| `Releasing expired KV blocks` in `logs/prefill.log` | Prefill and decode disagree on request IDs, so the prefill server holds every request's KV for 120 s and stalls | Make sure `check` passed the router check (`git pull`), then restart the router |
| Sweep requests time out | Usually the row above, or a crashed server | Check windows 0 and 1, then collect diagnostics |
| Sweep stalls: no progress line for 5+ min, decode log shows `Running: N reqs` with `generation throughput: 0.0`, and `nvidia-smi` shows **`ERR!`** on a GPU | That GPU stopped responding. This is the known failure of `KV_PATH=direct` on PCIe hosts (peer reads); otherwise a hardware fault. The pod can't recover | Terminate, deploy a new pod, and use the default `KV_PATH=staged` |

**Eager fallback.** Use this if a server dies while capturing CUDA graphs. It turns off CUDA
graphs, so it must be applied to **both phases**, or the comparison isn't fair:

```bash
VLLM_EXTRA_ARGS="--num-gpu-blocks-override 8192 --enforce-eager" bash scripts/vllm_disagg_baremetal.sh prefill
```

Use the same prefix for `decode`, and in Phase B:

```bash
VLLM_EXTRA_ARGS="--num-gpu-blocks-override 8192 --enforce-eager" GPU_MEM_UTIL=0.80 VLLM_BIN=~/vllm-env/bin/vllm N=2 GPUS=0,1 bash scripts/vllm_baremetal.sh
```

**Transfer path.** The default is `KV_PATH=staged` (through host memory). To try the direct
peer-to-peer path, on NVLink-connected GPUs only, start both servers with `KV_PATH=direct`:

```bash
KV_PATH=direct bash scripts/vllm_disagg_baremetal.sh prefill
```

Use the same prefix for `decode`. Each server window prints which path it's using, and the
choice is also saved to `logs/kv_path_prefill.txt` and `logs/kv_path_decode.txt`.

**Stop-loss.** If Phase A still fails after about 45 minutes of trying:
1. collect diagnostics;
2. copy home anything you have (step 7);
3. terminate the pod.

There's no point paying for idle GPUs while the logs are being read.
