# Convenience targets. Everything here is also documented in README.md.
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install test traces sim-cluster sim-cluster-fast sim-disagg-cluster sim-quick sim-full sim-disagg \
        docker-sim docker-sim-disagg docker-vllm docker-sglang docker-down vllm-quick vllm-full sglang-full analyze-latest

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## create .venv and install kvbench (+dev, +gpu extras)
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev,gpu]"

test: ## run unit + end-to-end tests (simulator, no GPU)
	$(BIN)/pytest

traces: ## generate the 400- and 1,000-prompt RAG traces
	$(BIN)/kvbench gen-trace --workload configs/workloads/rag_small.yaml --out traces/rag_400.jsonl
	$(BIN)/kvbench gen-trace --workload configs/workloads/rag_default.yaml --out traces/rag_1000.jsonl

sim-cluster: ## start 4 colocated sim workers + router locally (foreground; Ctrl-C to stop)
	$(BIN)/kvbench cluster --config configs/router/sim_local.yaml

sim-cluster-fast: ## same, but the simulator runs 4x faster than real time
	$(BIN)/kvbench cluster --config configs/router/sim_local.yaml --set speed=4

sim-disagg-cluster: ## start 2 prefill + 2 decode sim workers + router locally (foreground)
	$(BIN)/kvbench cluster --config configs/router/sim_local_disagg.yaml

sim-quick: ## quick policy sweep against a running sim cluster
	$(BIN)/kvbench bench --experiment configs/experiments/sim_quick.yaml

sim-full: ## full methodology on the simulator (1,000 prompts x 6 levels x 3 repeats)
	$(BIN)/kvbench bench --experiment configs/experiments/sim_full.yaml

sim-disagg: ## disaggregated sweep against a running sim disagg cluster
	$(BIN)/kvbench bench --experiment configs/experiments/sim_disagg.yaml

docker-sim: ## sim cluster in Docker
	docker compose -f docker/compose.sim.yml up --build -d

docker-sim-disagg: ## sim disaggregated cluster in Docker
	docker compose -f docker/compose.sim-disagg.yml up --build -d

docker-vllm: ## 2 vLLM replicas + router (GPU host)
	docker compose -f docker/compose.vllm.yml --env-file .env up --build -d

docker-sglang: ## 2 SGLang replicas + router (GPU host)
	docker compose -f docker/compose.sglang.yml --env-file .env up --build -d

docker-down: ## stop every kvbench compose project
	-docker compose -f docker/compose.sim.yml down
	-docker compose -f docker/compose.sim-disagg.yml down
	-docker compose -f docker/compose.vllm.yml --profile x4 down
	-docker compose -f docker/compose.sglang.yml down
	-docker compose -f docker/compose.vllm-disagg.yml down

vllm-quick: ## quick sweep against the vLLM stack
	$(BIN)/kvbench bench --experiment configs/experiments/vllm_quick.yaml

vllm-full: ## full sweep against the vLLM stack
	$(BIN)/kvbench bench --experiment configs/experiments/vllm.yaml

sglang-full: ## full sweep against the SGLang stack
	$(BIN)/kvbench bench --experiment configs/experiments/sglang.yaml

analyze-latest: ## re-run analysis on the newest experiment directory
	$(BIN)/kvbench analyze $$(ls -dt results/*/*/ | head -1)
