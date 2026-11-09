"""Router configuration (YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kvbench.common.io import read_yaml


@dataclass
class WorkerSpec:
    name: str
    url: str
    backend: str = "vllm"            # vllm | sglang | sim
    role: str = "both"               # both | prefill | decode
    kv_capacity_tokens: int | None = None
    sim_config: str | None = None    # only used by the local launcher (`kvbench cluster`)
    sim_overrides: dict = field(default_factory=dict)


@dataclass
class RouterConfig:
    model: str = "kvbench-model"
    mode: str = "colocated"           # colocated | disagg
    policy: str = "cache_aware"
    decode_policy: str = "least_loaded"   # disagg: how to pick the decode worker
    chunk_chars: int = 256
    chars_per_token: float = 4.0
    index_capacity_tokens: int = 131072   # default per-worker cache size estimate
    index_ttl_s: float = 600.0
    adaptive_capacity: bool = False
    scrape_interval_s: float = 0.5
    unhealthy_after_failures: int = 3
    request_timeout_s: float = 900.0
    max_retries: int = 2
    policy_params: dict = field(default_factory=dict)
    workers: list[WorkerSpec] = field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def load(cls, path: str | Path) -> "RouterConfig":
        d = read_yaml(path)
        workers = [WorkerSpec(**w) for w in d.pop("workers", [])]
        cfg = cls(**d, workers=workers)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.workers:
            raise ValueError("router config has no workers")
        names = [w.name for w in self.workers]
        if len(set(names)) != len(names):
            raise ValueError("worker names must be unique")
        if self.mode not in ("colocated", "disagg"):
            raise ValueError("mode must be 'colocated' or 'disagg'")
        if self.mode == "disagg":
            if not any(w.role == "prefill" for w in self.workers) or not any(w.role == "decode" for w in self.workers):
                raise ValueError("disagg mode needs at least one prefill and one decode worker")
        for w in self.workers:
            if w.backend not in ("vllm", "sglang", "sim"):
                raise ValueError(f"unknown backend {w.backend!r}")
            if w.role not in ("both", "prefill", "decode"):
                raise ValueError(f"unknown role {w.role!r}")
