"""The baseline to beat: chunk-and-stuff retrieval, one model call.

This is meant to be a *fair* baseline, not a strawman:

* it sees exactly the same corpus as the graph agent (every .py file the index
  walks -- no hand-picked subset);
* chunks are line windows with overlap, so a definition straddling a boundary is
  still retrievable;
* scoring is idf-weighted lexical overlap with the same tokenizer the graph's
  `search_symbols` uses, so neither side gets a better matcher. No embeddings --
  none are available offline, and reaching for an API embedding model would make
  the token accounting incomparable;
* `k` is tuned rather than guessed. `recall_at_k` measures, for the shipped
  question set, whether the chunk containing the gold symbol makes the top-k;
  `graph/README.md` reports the sweep. The default is the smallest k that keeps
  recall at its maximum, which is the strongest honest setting for the baseline.

What it structurally cannot do is follow a reference: if the answer lives in a
function the top-k missed, there is no second chance -- that is the cost of
one-shot retrieval, not a handicap we imposed.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from common.client import MODEL

from .builder import CodeGraph
from .tokens import TokenCounter, TokenLedger
from .tools import tokenize

DEFAULT_CHUNK_LINES = 60
DEFAULT_OVERLAP = 15
DEFAULT_K = 12
DEFAULT_MAX_TOKENS = 4000

BASELINE_SYSTEM = """\
You answer questions about a Python repository.

Below are the source excerpts a retrieval system judged most relevant to the \
question. They are all you have -- there are no tools and no way to fetch more \
code. Answer from the excerpts, quoting exact identifiers and values. If the \
excerpts do not contain the answer, say what is missing rather than guessing.
"""


@dataclass(frozen=True)
class Chunk:
    path: str
    start: int
    end: int
    text: str
    tokens: tuple[str, ...] = field(default=(), repr=False)

    def render(self) -> str:
        return f"### {self.path}:{self.start}-{self.end}\n{self.text}"

    def contains(self, path: str, line: int) -> bool:
        return self.path == path and self.start <= line <= self.end


def chunk_file(
    path: str,
    source: str,
    *,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    lines = source.splitlines()
    if not lines:
        return []
    step = max(1, chunk_lines - overlap)
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + chunk_lines)
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append(
                Chunk(
                    path=path,
                    start=start + 1,
                    end=end,
                    text=text,
                    tokens=tuple(tokenize(text)),
                )
            )
        if end >= len(lines):
            break
        start += step
    return chunks


def chunk_repo(
    graph: CodeGraph,
    *,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk exactly the files the graph indexed -- same corpus, both arms."""
    chunks: list[Chunk] = []
    for rel in sorted(graph.files):
        abs_path = os.path.join(graph.root, rel)
        if not os.path.isfile(abs_path):
            continue
        with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        chunks.extend(
            chunk_file(rel, source, chunk_lines=chunk_lines, overlap=overlap)
        )
    return chunks


class LexicalRetriever:
    """idf-weighted bag-of-identifiers scoring. Stdlib only, no embeddings."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        doc_freq: dict[str, int] = {}
        for chunk in self.chunks:
            for term in set(chunk.tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        total = max(1, len(self.chunks))
        self.idf = {
            term: math.log(1 + total / (1 + freq)) for term, freq in doc_freq.items()
        }

    def score(self, chunk: Chunk, terms: Iterable[str]) -> float:
        counts: dict[str, int] = {}
        for token in chunk.tokens:
            counts[token] = counts.get(token, 0) + 1
        total = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            total += self.idf.get(term, 1.0) * (1 + math.log(tf))
        # Normalize by chunk length so a long chunk does not win on bulk alone.
        return total / math.sqrt(max(1, len(chunk.tokens)))

    def top_k(self, query: str, k: int = DEFAULT_K) -> list[Chunk]:
        terms = tokenize(query)
        scored = [(self.score(c, terms), c) for c in self.chunks]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1].path, pair[1].start))
        return [c for _, c in scored[:k]]


def build_baseline_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    body = "\n\n".join(c.render() for c in chunks)
    return (
        f"Question: {question}\n\n"
        f"Retrieved excerpts ({len(chunks)} chunks):\n\n{body}\n\n"
        f"Answer the question using only the excerpts above."
    )


@dataclass
class BaselineRun:
    question: str
    answer: str
    model_calls: int
    chunks: list[str] = field(default_factory=list)
    total_prompt_tokens: int = 0
    peak_prompt_tokens: int = 0
    wall_clock: float = 0.0
    stop_reason: str = ""
    exact_tokens: bool = False


def run_baseline(
    question: str,
    retriever: LexicalRetriever,
    client: Any,
    *,
    counter: TokenCounter,
    k: int = DEFAULT_K,
    model: str = MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BaselineRun:
    started = time.perf_counter()
    chunks = retriever.top_k(question, k=k)
    prompt = build_baseline_prompt(question, chunks)
    messages = [{"role": "user", "content": prompt}]
    ledger = TokenLedger(counter)
    ledger.record("baseline", messages, system=BASELINE_SYSTEM)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=BASELINE_SYSTEM,
        thinking={"type": "adaptive"},
        messages=messages,
    )
    texts = [
        getattr(b, "text", "")
        for b in response.content
        if getattr(b, "type", None) == "text"
    ]
    return BaselineRun(
        question=question,
        answer="\n".join(t for t in texts if t).strip(),
        model_calls=ledger.requests,
        chunks=[f"{c.path}:{c.start}-{c.end}" for c in chunks],
        total_prompt_tokens=ledger.total_prompt_tokens,
        peak_prompt_tokens=ledger.peak_prompt_tokens,
        wall_clock=time.perf_counter() - started,
        stop_reason=getattr(response, "stop_reason", "") or "",
        exact_tokens=ledger.exact,
    )


def _overlaps(chunk: Chunk, path: str, lineno: int, end_lineno: int) -> bool:
    return chunk.path == path and chunk.start <= end_lineno and chunk.end >= lineno


def recall_at_k(
    graph: CodeGraph,
    retriever: LexicalRetriever,
    questions: Sequence[Any],
    k: int,
) -> float:
    """Fraction of questions whose gold symbols all land in the top-k chunks.

    This is how `k` gets tuned without burning API calls: it measures the
    baseline's retrieval ceiling directly. A question counts as retrieved only
    if *every* gold symbol it needs is covered -- partial coverage is how
    one-shot retrieval produces confident wrong answers.
    """
    scored = 0
    hits = 0
    for question in questions:
        targets = []
        for symbol_id in getattr(question, "gold_symbols", ()) or ():
            node = graph.node(symbol_id)
            if node is not None:
                targets.append((node.path, node.lineno, node.end_lineno))
        if not targets:
            continue
        scored += 1
        chunks = retriever.top_k(question.text, k=k)
        if all(
            any(_overlaps(chunk, *target) for chunk in chunks) for target in targets
        ):
            hits += 1
    return hits / scored if scored else 0.0


K_SWEEP = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def sweep_k(
    graph: CodeGraph,
    retriever: LexicalRetriever,
    questions: Sequence[Any],
    values: Iterable[int] = K_SWEEP,
) -> list[tuple[int, float]]:
    """Recall@k for several k, so the default can be justified, not guessed."""
    return [(k, recall_at_k(graph, retriever, questions, k)) for k in values]


def choose_k(
    graph: CodeGraph,
    retriever: LexicalRetriever,
    questions: Sequence[Any],
    values: Iterable[int] = K_SWEEP,
) -> int:
    """The smallest k that reaches the best recall this retriever can manage.

    This is what keeps the baseline from being a strawman. A k picked by taste
    either starves the baseline (cheap, but wrong) or pads it (correct, but
    needlessly expensive); either way the token comparison is rigged. Measuring
    the retrieval ceiling costs no API calls, so there is no excuse for guessing.
    """
    sweep = sweep_k(graph, retriever, questions, values)
    if not sweep:
        return DEFAULT_K
    best = max(recall for _, recall in sweep)
    return min(k for k, recall in sweep if recall == best)
