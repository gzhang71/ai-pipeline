"""Head-to-head: JIT graph retrieval vs. chunk-and-stuff, same questions.

Reported per question, per arm:

    correct              substring grading against the shipped question set
    prompt_tokens        SUM over every request of the FULL prompt -- tool
                         schemas, system prompt, question, and every tool result
                         that entered context on that turn
    peak_prompt_tokens   the largest single request, i.e. how much context the
                         approach needed at once
    model_calls          number of /v1/messages requests
    wall_clock           seconds, wall time

`prompt_tokens` is the honest number and it is deliberately unflattering to the
JIT arm: a 4-turn agent re-reads its whole transcript 4 times, and all four
readings are counted. `peak_prompt_tokens` is the number that shows what the
approach buys you in context-window terms. Both are printed.

Note on `usage`: the harness counts prompts with the count_tokens endpoint
rather than reading `usage.input_tokens` off responses, because `input_tokens`
is the *uncached remainder* only -- with caching on, total prompt size is
input + cache_creation + cache_read, and reading one field would understate a
cache-warm request. Counting the payload we are about to send sidesteps that
entirely and works identically offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from common.client import MODEL, get_client, has_credentials

from .baseline import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_K,
    DEFAULT_OVERLAP,
    LexicalRetriever,
    choose_k,
    chunk_repo,
    recall_at_k,
    run_baseline,
    sweep_k,
)
from .agent import build_system_prompt, run_jit_agent
from .builder import CodeGraph
from .index import CodeIndex
from .questions import QUESTION_SETS, Question
from .tokens import TokenCounter, default_counter
from .tools import GraphTools

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ArmResult:
    approach: str
    question_id: str
    answer: str
    correct: bool
    missing: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    peak_prompt_tokens: int = 0
    model_calls: int = 0
    wall_clock: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Comparison:
    results: list[ArmResult]
    exact_tokens: bool
    graph_stats: dict[str, int]
    k: int

    def by_approach(self, approach: str) -> list[ArmResult]:
        return [r for r in self.results if r.approach == approach]

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for approach in ("jit", "baseline"):
            rows = self.by_approach(approach)
            if not rows:
                continue
            out[approach] = {
                "questions": len(rows),
                "correct": sum(1 for r in rows if r.correct),
                "prompt_tokens": sum(r.prompt_tokens for r in rows),
                "peak_prompt_tokens": max(r.peak_prompt_tokens for r in rows),
                "model_calls": sum(r.model_calls for r in rows),
                "wall_clock": sum(r.wall_clock for r in rows),
            }
        return out


def build_graph(root: str = REPO_ROOT, *, index_path: str | None = None) -> CodeGraph:
    """Built lazily at run time -- never a baked-in snapshot."""
    return CodeIndex.load_or_build(root, index_path=index_path).graph()


def run_comparison(
    questions: Sequence[Question],
    *,
    graph: CodeGraph,
    jit_client: Any,
    baseline_client: Any,
    counter: TokenCounter | None = None,
    k: int = DEFAULT_K,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap: int = DEFAULT_OVERLAP,
    max_turns: int = 8,
    model: str = MODEL,
) -> Comparison:
    counter = counter or default_counter()
    tools = GraphTools(graph)
    system_prompt = build_system_prompt(graph)
    retriever = LexicalRetriever(
        chunk_repo(graph, chunk_lines=chunk_lines, overlap=overlap)
    )

    results: list[ArmResult] = []
    for question in questions:
        run = run_jit_agent(
            question.text,
            tools,
            jit_client,
            counter=counter,
            model=model,
            max_turns=max_turns,
            system=system_prompt,
        )
        results.append(
            ArmResult(
                approach="jit",
                question_id=question.id,
                answer=run.answer,
                correct=question.grade(run.answer),
                missing=question.missing(run.answer),
                prompt_tokens=run.total_prompt_tokens,
                peak_prompt_tokens=run.peak_prompt_tokens,
                model_calls=run.model_calls,
                wall_clock=run.wall_clock,
                detail={
                    "tool_calls": [c["name"] for c in run.tool_calls],
                    "fetched": run.fetched_symbols,
                    "gold": list(question.gold_symbols),
                    "stop_reason": run.stop_reason,
                },
            )
        )

        base = run_baseline(
            question.text,
            retriever,
            baseline_client,
            counter=counter,
            k=k,
            model=model,
        )
        results.append(
            ArmResult(
                approach="baseline",
                question_id=question.id,
                answer=base.answer,
                correct=question.grade(base.answer),
                missing=question.missing(base.answer),
                prompt_tokens=base.total_prompt_tokens,
                peak_prompt_tokens=base.peak_prompt_tokens,
                model_calls=base.model_calls,
                wall_clock=base.wall_clock,
                detail={"chunks": base.chunks, "k": k},
            )
        )

    return Comparison(
        results=results,
        exact_tokens=bool(getattr(counter, "exact", False)),
        graph_stats=graph.stats(),
        k=k,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_HEADER = (
    f"{'question':<16}{'arm':<10}{'ok':<4}{'prompt tok':>11}"
    f"{'peak tok':>10}{'calls':>7}{'sec':>8}"
)


def format_report(comparison: Comparison) -> str:
    lines: list[str] = []
    note = "exact (count_tokens endpoint)" if comparison.exact_tokens else (
        "ESTIMATED offline (chars/token heuristic; both arms share one counter)"
    )
    lines.append(f"token counts: {note}")
    lines.append(f"baseline k = {comparison.k}")
    lines.append(
        "graph: " + ", ".join(f"{k}={v}" for k, v in comparison.graph_stats.items())
    )
    lines.append("")
    lines.append(_HEADER)
    lines.append("-" * len(_HEADER))
    by_question: dict[str, list[ArmResult]] = {}
    for row in comparison.results:
        by_question.setdefault(row.question_id, []).append(row)
    for question_id, rows in by_question.items():
        for row in rows:
            lines.append(
                f"{question_id:<16}{row.approach:<10}"
                f"{'YES' if row.correct else 'no':<4}"
                f"{row.prompt_tokens:>11,}{row.peak_prompt_tokens:>10,}"
                f"{row.model_calls:>7}{row.wall_clock:>8.2f}"
            )
    lines.append("-" * len(_HEADER))
    summary = comparison.summary()
    for approach, stats in summary.items():
        lines.append(
            f"{'TOTAL':<16}{approach:<10}"
            f"{int(stats['correct'])}/{int(stats['questions'])}"
            f"{int(stats['prompt_tokens']):>10,}"
            f"{int(stats['peak_prompt_tokens']):>10,}"
            f"{int(stats['model_calls']):>7}{stats['wall_clock']:>8.2f}"
        )
    if "jit" in summary and "baseline" in summary:
        jit, base = summary["jit"], summary["baseline"]
        if jit["prompt_tokens"]:
            lines.append("")
            lines.append(
                f"total prompt tokens  jit/baseline = "
                f"{jit['prompt_tokens'] / max(1, base['prompt_tokens']):.2f}x"
            )
            lines.append(
                f"peak  prompt tokens  jit/baseline = "
                f"{jit['peak_prompt_tokens'] / max(1, base['peak_prompt_tokens']):.2f}x"
            )
            lines.append(
                f"round trips          jit/baseline = "
                f"{jit['model_calls'] / max(1, base['model_calls']):.2f}x  "
                f"(this is what JIT pays for the saving)"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=REPO_ROOT, help="repository to index")
    parser.add_argument(
        "--questions", default="repo", choices=sorted(QUESTION_SETS), help="question set"
    )
    parser.add_argument(
        "--k",
        default="auto",
        help=(
            "baseline top-k chunks, or 'auto' (default) to use the smallest k "
            "that reaches the retriever's best recall on this question set"
        ),
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--json", action="store_true", help="emit JSON, not a table")
    parser.add_argument(
        "--sweep-k",
        action="store_true",
        help="report baseline recall@k over the question set and exit (no API calls)",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="run against the offline fakes instead of the API",
    )
    args = parser.parse_args(argv)

    graph = build_graph(args.repo)
    questions = QUESTION_SETS[args.questions]

    if args.sweep_k:
        retriever = LexicalRetriever(chunk_repo(graph))
        print(f"baseline recall@k over {len(questions)} questions ({args.questions}):")
        for k, recall in sweep_k(graph, retriever, questions):
            print(f"  k={k:<3} recall={recall:.2f}")
        return 0

    if args.k == "auto":
        retriever = LexicalRetriever(chunk_repo(graph))
        k = choose_k(graph, retriever, questions)
        print(
            f"baseline k=auto -> {k} "
            f"(smallest k reaching recall "
            f"{recall_at_k(graph, retriever, questions, k):.2f} on this question set)",
            file=sys.stderr,
        )
    else:
        k = int(args.k)

    if args.fake:
        from .fake_client import FakeOneShotClient, FakeToolUsingClient

        jit_client: Any = FakeToolUsingClient(extra_reads=2)
        baseline_client: Any = FakeOneShotClient()
    else:
        if not has_credentials():
            print(
                "No API credentials found (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /"
                " an `ant auth login` profile). Re-run with --fake for the offline"
                " harness, or --sweep-k for the retrieval-only measurement.",
                file=sys.stderr,
            )
            return 2
        client = get_client()
        jit_client = client
        baseline_client = client

    started = time.perf_counter()
    comparison = run_comparison(
        questions,
        graph=graph,
        jit_client=jit_client,
        baseline_client=baseline_client,
        k=k,
        max_turns=args.max_turns,
        model=args.model,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "exact_tokens": comparison.exact_tokens,
                    "k": comparison.k,
                    "graph": comparison.graph_stats,
                    "summary": comparison.summary(),
                    "results": [r.to_dict() for r in comparison.results],
                },
                indent=2,
            )
        )
    else:
        print(format_report(comparison))
        print(f"\nharness wall clock: {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
