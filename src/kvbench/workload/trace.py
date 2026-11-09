"""Trace format: one JSON object per prompt, fully materialized.

Every prompt is stored verbatim (including synthetic conversation history), so a
replay is deterministic and independent of model outputs. That is what makes
"1,000 replayed prompts" comparable across routing policies and backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kvbench.common.io import read_jsonl, write_jsonl
from kvbench.common.prefix import chunk_hashes


@dataclass
class TraceRequest:
    request_id: str
    conversation_id: str
    turn: int
    tenant: int
    topic: int
    doc_ids: list[int]
    prompt: str
    max_tokens: int
    prompt_words: int = 0
    meta: dict = field(default_factory=dict)


def save_trace(path: str | Path, trace: list[TraceRequest]) -> None:
    write_jsonl(path, (asdict(r) for r in trace))


def load_trace(path: str | Path) -> list[TraceRequest]:
    return [TraceRequest(**row) for row in read_jsonl(path)]


def group_conversations(trace: list[TraceRequest]) -> list[list[TraceRequest]]:
    """Group requests into conversations, preserving first-appearance order and turn order."""
    convs: dict[str, list[TraceRequest]] = {}
    for r in trace:
        convs.setdefault(r.conversation_id, []).append(r)
    out = []
    for reqs in convs.values():
        out.append(sorted(reqs, key=lambda x: x.turn))
    return out


def truncate_trace(trace: list[TraceRequest], max_prompts: int) -> list[TraceRequest]:
    """Keep the first ``max_prompts`` prompts without breaking conversation turn order."""
    if max_prompts <= 0 or max_prompts >= len(trace):
        return trace
    return trace[:max_prompts]


def trace_stats(trace: list[TraceRequest], chunk_chars: int = 256, namespace: str = "") -> dict:
    """Compute prompt-size and prefix-sharing statistics for a trace.

    ``unique_prefix_chars`` approximates the *minimum* prefill work for the trace:
    the characters that would still need computing with a single, infinitely
    large prefix cache. Each distinct chained chunk is computed once; trailing
    partial chunks are never shareable so they always count.
    """
    lens = np.array([len(r.prompt) for r in trace], dtype=float)
    seen: set[int] = set()
    unique_chars = 0
    for r in trace:
        hs = chunk_hashes(r.prompt, chunk_chars, namespace)
        for h in hs:
            if h not in seen:
                seen.add(h)
                unique_chars += chunk_chars
        unique_chars += len(r.prompt.encode("utf-8")) - len(hs) * chunk_chars
    total = float(sum(len(r.prompt.encode("utf-8")) for r in trace))
    out_toks = np.array([r.max_tokens for r in trace], dtype=float)
    convs = group_conversations(trace)
    return {
        "num_prompts": len(trace),
        "num_conversations": len(convs),
        "mean_turns": float(np.mean([len(c) for c in convs])) if convs else 0.0,
        "prompt_chars_total": total,
        "prompt_chars_mean": float(lens.mean()) if len(lens) else 0.0,
        "prompt_chars_p50": float(np.percentile(lens, 50)) if len(lens) else 0.0,
        "prompt_chars_p95": float(np.percentile(lens, 95)) if len(lens) else 0.0,
        "prompt_chars_max": float(lens.max()) if len(lens) else 0.0,
        "max_tokens_mean": float(out_toks.mean()) if len(out_toks) else 0.0,
        "max_tokens_total": float(out_toks.sum()),
        "chunk_chars": chunk_chars,
        "unique_prefix_chars": float(unique_chars),
        "max_reuse_fraction": float(1.0 - unique_chars / total) if total else 0.0,
    }
