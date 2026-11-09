#!/usr/bin/env bash
# One-shot on a fresh Linux GPU box (after the NVIDIA driver + container toolkit are installed):
# brings up 2 vLLM replicas + router, then runs the quick and full benchmarks.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
docker compose -f docker/compose.vllm.yml --env-file .env up --build -d
python3 -m venv .venv
. .venv/bin/activate
pip install -q -e ".[gpu,dev]"
kvbench bench --experiment configs/experiments/vllm_quick.yaml
echo "quick run done; starting the full sweep (1,000 prompts x 6 concurrency levels x 3 repeats)"
kvbench bench --experiment configs/experiments/vllm.yaml
