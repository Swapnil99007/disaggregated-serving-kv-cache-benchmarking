"""Experiment configuration (YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kvbench.common.io import read_yaml


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    deployment: str = "default"            # label prefix, e.g. "vllm-2w" or "sim-disagg-2p2d"
    router_url: str = "http://localhost:8000"
    model: str = "kvbench-model"
    trace: dict = field(default_factory=lambda: {"path": "traces/rag_1000.jsonl",
                                                 "workload": "configs/workloads/rag_default.yaml",
                                                 "max_prompts": 0})
    policies: list[str] = field(default_factory=lambda: ["round_robin", "least_loaded", "prefix_affinity", "cache_aware"])
    decode_policy: str | None = None
    concurrency: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    repeats: int = 3
    shuffle_policy_order: bool = True
    warmup: dict = field(default_factory=lambda: {"engine_requests": 8, "prefix_warmup_prompts": 0})
    flush_between_runs: bool = True
    request: dict = field(default_factory=lambda: {"ignore_eos": True, "temperature": 0.0, "timeout_s": 900.0,
                                                   "think_time_s": 0.0, "extra_body": {}})
    telemetry: dict = field(default_factory=lambda: {"interval_s": 0.5, "nvml": True})
    ready_timeout_s: float = 900.0
    output_dir: str = "results"
    analysis: dict = field(default_factory=lambda: {"baseline_policy": "round_robin", "slo_metric": "e2e",
                                                    "slo_percentile": 95, "slo_target_s": None, "chunk_chars": 256})

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        d = read_yaml(path)
        base = cls()
        for key in ("trace", "warmup", "request", "telemetry", "analysis"):
            merged = dict(getattr(base, key))
            merged.update(d.get(key) or {})
            d[key] = merged
        return cls(**d)
