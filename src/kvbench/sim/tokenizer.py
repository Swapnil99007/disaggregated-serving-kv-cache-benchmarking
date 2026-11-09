"""Deterministic toy tokenizer + KV block hashing for the simulator."""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Words and punctuation marks. Prefix-stable: tokens(a) is a prefix of tokens(a+b)
    except possibly at the boundary token."""
    return _TOKEN_RE.findall(text)


def block_hashes(tokens: list[str], block_size: int, namespace: str = "") -> list[int]:
    """Chained hashes over full blocks of tokens (same scheme vLLM uses for prefix caching)."""
    prev = hashlib.blake2b(namespace.encode(), digest_size=8).digest()
    out = []
    for i in range(len(tokens) // block_size):
        blk = "\x1f".join(tokens[i * block_size:(i + 1) * block_size]).encode()
        h = hashlib.blake2b(prev + blk, digest_size=8).digest()
        out.append(int.from_bytes(h, "little"))
        prev = h
    return out
