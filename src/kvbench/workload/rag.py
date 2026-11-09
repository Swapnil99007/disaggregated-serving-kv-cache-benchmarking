"""Synthetic long-context RAG workload generator.

Models a conversational RAG service:

* A few *tenants* (applications), each with its own long system prompt.
* A corpus of documents grouped into *topics*. Topic popularity is Zipfian, so a
  handful of topics are hot (this is what creates cross-user prefix reuse).
* Each *conversation* picks a tenant and a topic and retrieves k documents. With
  probability ``retrieval_stability`` the retriever returns the topic's
  canonical top-k in a stable order (identical prefix to other conversations on
  that topic); otherwise it returns a perturbed set (partial prefix reuse).
* A conversation has several turns. Turn t's prompt is
  ``system + retrieved docs + turns 1..t-1 + question t``, so follow-up turns
  extend the previous turn's prefix (intra-session reuse).

Prompt layout (shared prefix first, request-specific suffix last):

    <system prompt>
    ### Retrieved context
    [doc ...] ...
    ### Conversation
    User: ...
    Assistant: ...
    User: <new question>
    Assistant:

Everything is seeded: the corpus from ``corpus_seed`` and the request stream from
``seed``, so warm-up traces can reuse the corpus with a different request stream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields

import numpy as np

from kvbench.workload.trace import TraceRequest
from kvbench.workload.words import VOCAB


@dataclass
class RagWorkloadConfig:
    seed: int = 42
    corpus_seed: int = 7
    num_prompts: int = 1000
    # tenants / system prompts
    num_tenants: int = 3
    system_prompt_words: tuple[int, int] = (150, 400)
    # corpus
    num_topics: int = 60
    docs_per_topic: int = 6
    doc_words: tuple[int, int] = (300, 900)
    topic_zipf_s: float = 1.1
    # retrieval
    retrieval_k: tuple[int, int] = (3, 5)
    retrieval_stability: float = 0.7
    # conversations
    mean_turns: float = 3.0
    max_turns: int = 8
    question_words: tuple[int, int] = (10, 40)
    answer_words: tuple[int, int] = (40, 160)
    max_prompt_words: int = 9000
    # generation lengths (max_tokens, with ignore_eos at replay time)
    output_tokens_median: int = 128
    output_tokens_sigma: float = 0.5
    output_tokens_range: tuple[int, int] = (16, 512)

    @classmethod
    def from_dict(cls, d: dict) -> "RagWorkloadConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown workload keys: {sorted(unknown)}")
        kw = {}
        for k, v in d.items():
            kw[k] = tuple(v) if isinstance(v, list) else v
        return cls(**kw)


def _words(rng: random.Random, n: int) -> str:
    """Generate ``n`` words of pseudo-prose with sentence punctuation."""
    out: list[str] = []
    sentence_len = 0
    target = rng.randint(8, 20)
    for i in range(n):
        w = rng.choice(VOCAB)
        if sentence_len == 0:
            w = w.capitalize()
        sentence_len += 1
        if sentence_len >= target or i == n - 1:
            w += "."
            sentence_len = 0
            target = rng.randint(8, 20)
        out.append(w)
    return " ".join(out)


def _zipf_probs(n: int, s: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=float)
    w = 1.0 / np.power(ranks, s) if s > 0 else np.ones(n)
    return w / w.sum()


@dataclass
class _Corpus:
    system_prompts: list[str]
    topic_names: list[str]
    docs: list[str]                 # global doc id -> rendered doc text
    doc_topic: list[int]
    topic_docs: list[list[int]]     # canonical (relevance-ranked) doc ids per topic
    doc_words: list[int]
    system_words: list[int]


def build_corpus(cfg: RagWorkloadConfig) -> _Corpus:
    rng = random.Random(cfg.corpus_seed)
    system_prompts, system_words = [], []
    for t in range(cfg.num_tenants):
        n = rng.randint(*cfg.system_prompt_words)
        body = _words(rng, n)
        system_prompts.append(
            f"You are assistant #{t} for an enterprise knowledge base. Answer strictly from the "
            f"retrieved context, cite document ids, and say when the context is insufficient. "
            f"Policy and style guide follows. {body}"
        )
        system_words.append(n + 30)
    topic_names = [f"{rng.choice(VOCAB)}-{rng.choice(VOCAB)}-{i}" for i in range(cfg.num_topics)]
    docs, doc_topic, doc_words, topic_docs = [], [], [], []
    for t in range(cfg.num_topics):
        ids = []
        for j in range(cfg.docs_per_topic):
            doc_id = len(docs)
            n = rng.randint(*cfg.doc_words)
            docs.append(f"[doc:{doc_id} | topic:{topic_names[t]} | source:kb/{topic_names[t]}/{j}.md]\n{_words(rng, n)}")
            doc_topic.append(t)
            doc_words.append(n + 4)
            ids.append(doc_id)
        topic_docs.append(ids)
    return _Corpus(system_prompts, topic_names, docs, doc_topic, topic_docs, doc_words, system_words)


def _render(system: str, docs: list[str], history: list[tuple[str, str]], question: str) -> str:
    parts = [system, "", "### Retrieved context", "\n\n".join(docs), "", "### Conversation"]
    for q, a in history:
        parts.append(f"User: {q}")
        parts.append(f"Assistant: {a}")
    parts.append(f"User: {question}")
    parts.append("Assistant:")
    return "\n".join(parts)


def generate_rag_trace(cfg: RagWorkloadConfig) -> list[TraceRequest]:
    corpus = build_corpus(cfg)
    rng = random.Random(cfg.seed)
    nprng = np.random.default_rng(cfg.seed)
    topic_p = _zipf_probs(cfg.num_topics, cfg.topic_zipf_s)
    lo, hi = cfg.output_tokens_range
    trace: list[TraceRequest] = []
    conv_idx = 0
    while len(trace) < cfg.num_prompts:
        if conv_idx > 50 * cfg.num_prompts + 100:
            raise ValueError("workload config cannot produce prompts under max_prompt_words")
        tenant = rng.randrange(cfg.num_tenants)
        topic = int(nprng.choice(cfg.num_topics, p=topic_p))
        k = rng.randint(*cfg.retrieval_k)
        canonical = corpus.topic_docs[topic]
        if rng.random() < cfg.retrieval_stability:
            doc_ids = canonical[:k]
        else:
            # perturbed retrieval: a different subset/order plus one off-topic doc
            pool = list(canonical)
            rng.shuffle(pool)
            doc_ids = pool[: max(1, k - 1)]
            other = rng.randrange(len(corpus.docs))
            if other not in doc_ids:
                doc_ids.append(other)
        n_turns = min(cfg.max_turns, int(nprng.geometric(1.0 / max(cfg.mean_turns, 1.0))))
        base_words = corpus.system_words[tenant] + sum(corpus.doc_words[d] for d in doc_ids)
        history: list[tuple[str, str]] = []
        hist_words = 0
        cid = f"c{conv_idx:05d}"
        for turn in range(n_turns):
            if len(trace) >= cfg.num_prompts:
                break
            qn = rng.randint(*cfg.question_words)
            question = _words(rng, qn).rstrip(".") + "?"
            if base_words + hist_words + qn > cfg.max_prompt_words:
                break
            prompt = _render(corpus.system_prompts[tenant], [corpus.docs[d] for d in doc_ids], history, question)
            mt = int(np.clip(round(cfg.output_tokens_median * float(np.exp(nprng.normal(0, cfg.output_tokens_sigma)))), lo, hi))
            trace.append(
                TraceRequest(
                    request_id=f"{cid}-t{turn}",
                    conversation_id=cid,
                    turn=turn,
                    tenant=tenant,
                    topic=topic,
                    doc_ids=list(doc_ids),
                    prompt=prompt,
                    max_tokens=mt,
                    prompt_words=base_words + hist_words + qn,
                )
            )
            # synthetic assistant answer becomes history for the next turn
            an = rng.randint(*cfg.answer_words)
            answer = _words(rng, an)
            history.append((question, answer))
            hist_words += qn + an + 2
        conv_idx += 1
    return trace


def generate_warmup_trace(cfg: RagWorkloadConfig, num_prompts: int) -> list[TraceRequest]:
    """Same corpus, different request stream — warms prefix caches realistically."""
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    d.update(seed=cfg.seed + 1_000_003, num_prompts=num_prompts)
    trace = generate_rag_trace(RagWorkloadConfig(**d))
    for r in trace:
        r.request_id = "warm-" + r.request_id
        r.conversation_id = "warm-" + r.conversation_id
    return trace
