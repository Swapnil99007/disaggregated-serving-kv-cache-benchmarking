"""Prefix identity: chained chunk hashes.

The router cannot see a worker's real KV cache, so it keeps an *approximate*
per-worker index of prompt prefixes it has sent there. Prefixes are identified
by chained hashes over fixed-size character chunks:

    h_0 = H(namespace || chunk_0)
    h_i = H(h_{i-1} || chunk_i)

Two prompts share their first k hashes iff they share their first
k * chunk_chars characters (modulo hash collisions). This mirrors how vLLM
hashes KV blocks (each block hash folds in its parent's hash), but works on
characters so the router does not need the model's tokenizer. SGLang's
cache-aware router makes the same trade-off with a character-level radix tree.

The namespace should include the model name so prefixes from different models
or tokenizers never alias each other.
"""

from __future__ import annotations

import hashlib


def chunk_hashes(text: str, chunk_chars: int, namespace: str = "") -> list[int]:
    """Return chained hashes for every *full* chunk of ``text``.

    The trailing partial chunk is not hashed: like a partial KV block it is not
    reusable by other requests.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    data = text.encode("utf-8", errors="replace")
    prev = hashlib.blake2b(namespace.encode(), digest_size=8).digest()
    out: list[int] = []
    n_full = len(data) // chunk_chars
    for i in range(n_full):
        h = hashlib.blake2b(prev + data[i * chunk_chars:(i + 1) * chunk_chars], digest_size=8).digest()
        out.append(int.from_bytes(h, "little"))
        prev = h
    return out


def extract_prompt_text(body: dict) -> str:
    """Extract routable prompt text from an OpenAI-style completions/chat body."""
    if "prompt" in body:
        p = body["prompt"]
        if isinstance(p, list):
            if p and isinstance(p[0], str):
                return "".join(p)
            # token-id prompts: stringify so identical id sequences still hash identically
            return " ".join(str(x) for x in p)
        return str(p)
    if "messages" in body:
        parts = []
        for m in body.get("messages") or []:
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            parts.append(f"<|{m.get('role', 'user')}|>{content}")
        return "\n".join(parts)
    return ""
