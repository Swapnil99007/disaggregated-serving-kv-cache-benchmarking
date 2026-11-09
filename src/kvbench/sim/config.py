"""Simulator configuration and cost-model parameters."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class SimConfig:
    model_name: str = "kvbench-model"
    engine_id: str = "sim-0"
    role: str = "both"                    # both | prefill | decode (labels; any engine can do either)

    # --- KV cache / scheduler -------------------------------------------------
    block_size: int = 16                  # tokens per KV block (vLLM default)
    kv_capacity_tokens: int = 131072      # KV pool size; set to create cache pressure
    max_num_seqs: int = 64                # max concurrently running sequences
    max_num_batched_tokens: int = 2048    # per-step token budget (chunked prefill)
    enable_prefix_caching: bool = True

    # --- cost model (simulated seconds) ---------------------------------------
    # step_time = overhead
    #           + prefill_tokens / prefill_tokens_per_s                (compute-bound)
    #           + [if any decode] decode_weights_ms                     (weight streaming, memory-bound)
    #           + decode_kv_ns_per_token * sum(decode context lengths)  (KV reads, memory-bound)
    #           + decode_per_seq_ms * num_decode_seqs
    step_overhead_ms: float = 2.0
    prefill_tokens_per_s: float = 10000.0
    prefill_attn_ns_per_token_ctx: float = 0.0   # optional quadratic attention term
    decode_weights_ms: float = 8.0
    decode_kv_ns_per_token: float = 30.0
    decode_per_seq_ms: float = 0.05

    # --- memory model (for GPU-memory telemetry) ------------------------------
    gpu_mem_gb: float = 24.0
    weights_gb: float = 3.1
    kv_bytes_per_token: int = 28672       # Qwen2.5-1.5B, bf16: 28 layers * 2 * 2 kv heads * 128 * 2B

    # --- disaggregated KV transfer --------------------------------------------
    transfer_gb_per_s: float = 25.0       # effective link bandwidth (GB/s)
    transfer_latency_ms: float = 2.0      # per-transfer setup latency

    # --- misc -----------------------------------------------------------------
    speed: float = 1.0                    # wall-clock speedup (2.0 = runs twice as fast)
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown sim config keys: {sorted(unknown)}")
        return cls(**d)
