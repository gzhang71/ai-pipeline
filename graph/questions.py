"""The question sets used by the head-to-head comparison.

Every question is checkable offline: correctness is a substring test over the
answer text, not a model-graded judgement. That keeps the comparison honest --
the grader cannot drift, and neither arm can win by being persuasive.

`gold_symbols` names the symbols that actually contain the answer. It is used
two ways: to tune the baseline's `k` via `baseline.recall_at_k` without spending
API calls, and to report whether the JIT agent fetched the right node.

REPO_QUESTIONS target `common/` and `graph/`, which are stable. They are
answered against the live repository, whose graph is built lazily at run time --
no snapshot is baked in anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    expect_all: tuple[str, ...] = ()
    expect_any: tuple[str, ...] = ()
    gold_symbols: tuple[str, ...] = ()
    note: str = ""

    def grade(self, answer: str) -> bool:
        """True when every required token, and at least one optional, appears."""
        if not answer:
            return False
        haystack = _normalize(answer)
        for needle in self.expect_all:
            if _normalize(needle) not in haystack:
                return False
        if self.expect_any:
            return any(_normalize(n) in haystack for n in self.expect_any)
        return True

    def missing(self, answer: str) -> list[str]:
        haystack = _normalize(answer)
        gaps = [n for n in self.expect_all if _normalize(n) not in haystack]
        if self.expect_any and not any(
            _normalize(n) in haystack for n in self.expect_any
        ):
            gaps.append("any-of:" + "|".join(self.expect_any))
        return gaps


_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


REPO_QUESTIONS: tuple[Question, ...] = (
    Question(
        id="q1-usage",
        text=(
            "In the common package, which function turns a response usage object "
            "into a plain dict, and which three token counts does it add together "
            "to produce total_prompt_tokens?"
        ),
        expect_all=(
            "usage_breakdown",
            "input_tokens",
            "cache_creation",
            "cache_read",
        ),
        gold_symbols=("common.client:usage_breakdown",),
    ),
    Question(
        id="q2-model",
        text="What exact model id string is the MODEL constant in common/client.py set to?",
        expect_all=("claude-opus-5",),
        gold_symbols=("common.client",),
    ),
    Question(
        id="q3-cache-min",
        text=(
            "According to the MIN_CACHEABLE_PREFIX_TOKENS table in common/client.py, "
            "what is the minimum cacheable prefix in tokens for claude-haiku-4-5, "
            "and for claude-opus-5?"
        ),
        expect_all=("4096", "512"),
        gold_symbols=("common.client",),
    ),
    Question(
        id="q4-credentials",
        text=(
            "Which environment variables does has_credentials() in the common "
            "package check before it falls back to looking on disk?"
        ),
        expect_all=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        expect_any=("credentials", "config", "disk"),
        gold_symbols=("common.client:has_credentials",),
    ),
    Question(
        id="q5-tools",
        text=(
            "What are the names of the four retrieval tools defined in "
            "graph/tools.py, and what is the maximum number of lines read_lines "
            "will return?"
        ),
        expect_all=(
            "search_symbols",
            "get_definition",
            "get_neighbors",
            "read_lines",
            "400",
        ),
        gold_symbols=("graph.tools",),
    ),
    Question(
        id="q6-node-kinds",
        text=(
            "Which node kinds does the code graph in graph/builder.py use, and "
            "what distinguishes a 'method' from a 'function' in that builder?"
        ),
        expect_all=("module", "class", "function", "method"),
        expect_any=("parent", "enclosing", "inside a class", "class scope"),
        gold_symbols=("graph.builder", "graph.builder:_ModuleVisitor._visit_func"),
    ),
)


FIXTURE_QUESTIONS: tuple[Question, ...] = (
    Question(
        id="f1-slugify",
        text="What separator does slugify in pkg/util.py join words with, and what is its value?",
        expect_all=("SEPARATOR",),
        expect_any=("-", "hyphen"),
        gold_symbols=("pkg.util:slugify",),
    ),
    Question(
        id="f2-retry",
        text="What is the value of RETRY_LIMIT in the sample package?",
        expect_all=("7",),
        gold_symbols=("pkg.util",),
    ),
    Question(
        id="f3-loud",
        text="Which two functions does the loud method of Widget call?",
        expect_all=("slug", "_shout"),
        gold_symbols=("pkg.core:Widget.loud",),
    ),
)


QUESTION_SETS: dict[str, tuple[Question, ...]] = {
    "repo": REPO_QUESTIONS,
    "fixture": FIXTURE_QUESTIONS,
}
