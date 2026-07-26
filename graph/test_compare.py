"""The comparison harness end to end, offline, against the fakes.

The fakes drive real retrieval (see `graph/fake_client.py`), so these tests
exercise search ranking, definition fetching, chunk scoring, and the token
ledger -- not the doubles.
"""

from __future__ import annotations

import pytest

from common.client import has_credentials

from graph.agent import build_outline, run_jit_agent
from graph.baseline import (
    LexicalRetriever,
    build_baseline_prompt,
    chunk_file,
    chunk_repo,
    recall_at_k,
    run_baseline,
    sweep_k,
)
from graph.builder import CodeGraph
from graph.compare import format_report, run_comparison
from graph.index import CodeIndex
from graph.fake_client import (
    FakeNeighborClient,
    FakeOneShotClient,
    FakeRefusingClient,
    FakeToolUsingClient,
)
from graph.questions import FIXTURE_QUESTIONS, Question
from graph.tokens import OfflineTokenCounter, TokenLedger
from graph.tools import GraphTools


@pytest.fixture
def counter() -> OfflineTokenCounter:
    return OfflineTokenCounter()


# -- chunking / lexical retrieval ---------------------------------------


def test_chunk_file_overlaps_so_definitions_are_not_split():
    source = "\n".join(f"line{i}" for i in range(1, 121))
    chunks = chunk_file("a.py", source, chunk_lines=50, overlap=10)
    assert chunks[0].start == 1 and chunks[0].end == 50
    assert chunks[1].start == 41  # overlap of 10 lines
    assert chunks[-1].end == 120
    covered = {line for c in chunks for line in range(c.start, c.end + 1)}
    assert covered == set(range(1, 121))


def test_lexical_retriever_finds_the_right_chunk(sample_graph: CodeGraph):
    retriever = LexicalRetriever(chunk_repo(sample_graph))
    top = retriever.top_k("slugify separator join words", k=3)
    assert any(c.path == "pkg/util.py" for c in top)


def test_baseline_prompt_contains_the_retrieved_source(sample_graph: CodeGraph):
    retriever = LexicalRetriever(chunk_repo(sample_graph))
    prompt = build_baseline_prompt("slugify", retriever.top_k("slugify", k=4))
    assert "SENTINEL_SLUGIFY_BODY" in prompt  # the baseline stuffs bodies, by design


def test_recall_at_k_is_monotonic_and_used_to_tune_k(sample_graph: CodeGraph):
    retriever = LexicalRetriever(chunk_repo(sample_graph))
    sweep = sweep_k(sample_graph, retriever, FIXTURE_QUESTIONS, values=(1, 2, 4, 8, 16))
    recalls = [r for _, r in sweep]
    assert recalls == sorted(recalls), "recall must not fall as k grows"
    assert recall_at_k(sample_graph, retriever, FIXTURE_QUESTIONS, 16) > 0


# -- the JIT agent loop --------------------------------------------------


def test_jit_agent_searches_then_reads_then_answers(
    sample_tools: GraphTools, counter: OfflineTokenCounter
):
    run = run_jit_agent(
        "What separator does slugify join words with?",
        sample_tools,
        FakeToolUsingClient(),
        counter=counter,
        max_turns=6,
    )
    assert [c["name"] for c in run.tool_calls] == ["search_symbols", "get_definition"]
    assert run.fetched_symbols == ["pkg.util:slugify"]
    assert "SENTINEL_SLUGIFY_BODY" in run.answer  # the real body reached the model
    assert run.model_calls == 3
    assert run.stop_reason == "end_turn"


def test_jit_agent_token_ledger_counts_every_turn(
    sample_tools: GraphTools, counter: OfflineTokenCounter
):
    run = run_jit_agent(
        "What separator does slugify join words with?",
        sample_tools,
        FakeToolUsingClient(),
        counter=counter,
    )
    # Three requests were counted, and the total is the sum, not the first.
    assert run.model_calls == 3
    assert run.total_prompt_tokens > run.peak_prompt_tokens
    assert run.peak_prompt_tokens > 0


def test_jit_prompt_grows_as_tool_results_enter_context(
    sample_tools: GraphTools, counter: OfflineTokenCounter
):
    """Tool results are counted -- the accounting cannot be gamed by ignoring them."""
    ledger = TokenLedger(counter)
    system = "sys"
    base = ledger.record("t0", [{"role": "user", "content": "q"}], system=system)
    definition = sample_tools.call_as_text(
        "get_definition", {"symbol_id": "pkg.core:build"}
    )
    grown = ledger.record(
        "t1",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x",
                        "content": definition,
                    }
                ],
            },
        ],
        system=system,
    )
    assert grown > base
    assert ledger.total_prompt_tokens == base + grown


def test_jit_agent_stops_on_refusal(
    sample_tools: GraphTools, counter: OfflineTokenCounter
):
    run = run_jit_agent(
        "anything", sample_tools, FakeRefusingClient(), counter=counter
    )
    assert run.stop_reason == "refusal"
    assert run.model_calls == 1


def test_jit_agent_respects_the_turn_budget(
    sample_tools: GraphTools, counter: OfflineTokenCounter
):
    class NeverStops(FakeToolUsingClient):
        def respond(self, **kwargs):
            from graph.fake_client import FakeMessage, FakeToolUseBlock

            return FakeMessage(
                content=[FakeToolUseBlock(name="search_symbols", input={"query": "x"})],
                stop_reason="tool_use",
            )

    run = run_jit_agent(
        "loop forever", sample_tools, NeverStops(), counter=counter, max_turns=3
    )
    assert run.model_calls == 3
    assert run.stop_reason == "tool_use"


# -- the baseline arm ----------------------------------------------------


def test_baseline_runs_in_exactly_one_call(
    sample_graph: CodeGraph, counter: OfflineTokenCounter
):
    retriever = LexicalRetriever(chunk_repo(sample_graph))
    run = run_baseline(
        "What separator does slugify join words with?",
        retriever,
        FakeOneShotClient(),
        counter=counter,
        k=6,
    )
    assert run.model_calls == 1
    assert run.chunks
    assert "SENTINEL_SLUGIFY_BODY" in run.answer


# -- the whole harness ---------------------------------------------------


def test_comparison_runs_end_to_end_on_the_fixture(sample_graph: CodeGraph):
    counter = OfflineTokenCounter()
    comparison = run_comparison(
        FIXTURE_QUESTIONS,
        graph=sample_graph,
        jit_client=FakeToolUsingClient(extra_reads=2),
        baseline_client=FakeOneShotClient(),
        counter=counter,
        k=8,
    )
    assert len(comparison.results) == 2 * len(FIXTURE_QUESTIONS)
    summary = comparison.summary()
    assert set(summary) == {"jit", "baseline"}

    for approach in ("jit", "baseline"):
        rows = comparison.by_approach(approach)
        assert len(rows) == len(FIXTURE_QUESTIONS)
        for row in rows:
            assert row.prompt_tokens > 0
            assert row.model_calls >= 1
            assert row.wall_clock >= 0
        # Both arms actually answer the fixture questions correctly, so the
        # token comparison below is between two working systems.
        assert all(r.correct for r in rows), [
            (r.question_id, r.missing) for r in rows if not r.correct
        ]

    assert summary["baseline"]["model_calls"] == len(FIXTURE_QUESTIONS)
    assert summary["jit"]["model_calls"] > summary["baseline"]["model_calls"]
    assert comparison.exact_tokens is False  # offline counter, honestly labelled


def test_on_a_tiny_corpus_stuffing_wins(sample_graph: CodeGraph):
    """The honest negative result, pinned so it cannot be quietly forgotten.

    The sample repo is ~120 lines. The JIT arm's fixed overhead -- four tool
    schemas plus the index -- is larger than the entire codebase, so stuffing
    is strictly better here. JIT retrieval has a floor, and small repos are
    below it.
    """
    comparison = run_comparison(
        FIXTURE_QUESTIONS,
        graph=sample_graph,
        jit_client=FakeToolUsingClient(extra_reads=2),
        baseline_client=FakeOneShotClient(),
        counter=OfflineTokenCounter(),
        k=24,  # k=24 is the whole fixture corpus
    )
    summary = comparison.summary()
    assert summary["jit"]["peak_prompt_tokens"] > summary["baseline"][
        "peak_prompt_tokens"
    ]
    assert summary["jit"]["correct"] == summary["baseline"]["correct"]


def test_pinpoint_lookup_is_a_fair_fight_for_the_baseline(large_repo: str):
    """Second honest negative: on a unique keyword, chunk retrieval is great.

    `TUNING_CONSTANT` appears in exactly one chunk, so idf ranks it first and
    k=4 suffices. There is nothing for a graph to add, and JIT's fixed overhead
    makes it the more expensive way to get the same answer.
    """
    graph = CodeIndex.build(large_repo).graph()
    question = Question(
        id="big-pinpoint",
        text="What is the value of TUNING_CONSTANT?",
        expect_all=("8675309",),
        gold_symbols=("big.util",),
    )
    retriever = LexicalRetriever(chunk_repo(graph))
    assert recall_at_k(graph, retriever, [question], 4) == 1.0

    comparison = run_comparison(
        [question],
        graph=graph,
        jit_client=FakeToolUsingClient(),
        baseline_client=FakeOneShotClient(),
        counter=OfflineTokenCounter(),
        k=4,
    )
    summary = comparison.summary()
    assert summary["jit"]["correct"] == 1
    assert summary["baseline"]["correct"] == 1
    assert summary["baseline"]["peak_prompt_tokens"] < summary["jit"][
        "peak_prompt_tokens"
    ]


def test_relational_query_is_where_the_graph_wins(large_repo: str):
    """The headline claim, on the query shape a graph actually answers better.

    "Which functions call X" needs every call site at once. Chunk retrieval must
    hold all 24 of them in context; the graph returns them as an edge list. The
    baseline is given the k it genuinely needs -- a smaller k would be cheaper
    only by being wrong.
    """
    graph = CodeIndex.build(large_repo).graph()
    callers = sorted(graph.callers["big.util:shared_helper"])
    assert len(callers) == 24, "fixture should have many call sites"

    question = Question(
        id="big-relational",
        text="Which functions call shared_helper?",
        expect_all=("op_0_0", "op_23_0"),
        gold_symbols=tuple(callers),
    )
    retriever = LexicalRetriever(chunk_repo(graph))

    fair_k = next(
        (
            k
            for k in (4, 8, 16, 24, 32, 48, 64, 96, 128, 256)
            if recall_at_k(graph, retriever, [question], k) == 1.0
        ),
        None,
    )
    assert fair_k is not None, "baseline can never retrieve every call site"
    assert fair_k >= 24, "one chunk per call site, at minimum"

    comparison = run_comparison(
        [question],
        graph=graph,
        jit_client=FakeNeighborClient(),
        baseline_client=FakeOneShotClient(),
        counter=OfflineTokenCounter(),
        k=fair_k,
    )
    summary = comparison.summary()
    assert summary["jit"]["correct"] == 1
    assert summary["baseline"]["correct"] == 1
    # Both right; JIT used a fraction of the context...
    assert summary["jit"]["peak_prompt_tokens"] * 2 < summary["baseline"][
        "peak_prompt_tokens"
    ]
    # ...and paid for it in round trips. That is the trade, stated both ways.
    assert summary["jit"]["model_calls"] > summary["baseline"]["model_calls"]


def test_large_corpus_index_falls_back_to_module_level(large_repo: str):
    """Above the symbol budget the prompt must drop to module identifiers."""
    graph = CodeIndex.build(large_repo).graph()
    outline = build_outline(graph)
    assert "MODULE INDEX" in outline
    assert "big.util" in outline  # modules are still listed
    assert "def shared_helper" not in outline  # signatures are not
    assert len(outline) < len(build_outline(graph, detail="symbols"))


def test_report_renders_and_names_the_round_trip_cost(sample_graph: CodeGraph):
    comparison = run_comparison(
        FIXTURE_QUESTIONS[:1],
        graph=sample_graph,
        jit_client=FakeToolUsingClient(),
        baseline_client=FakeOneShotClient(),
        counter=OfflineTokenCounter(),
        k=4,
    )
    report = format_report(comparison)
    assert "ESTIMATED offline" in report
    assert "round trips" in report
    assert "f1-slugify" in report


def test_question_grading_is_strict():
    question = Question(
        id="x", text="?", expect_all=("alpha", "beta"), expect_any=("one", "two")
    )
    assert question.grade("Alpha and BETA, plus one")
    assert not question.grade("alpha and beta")  # no expect_any hit
    assert not question.grade("alpha, one")  # missing beta
    assert not question.grade("")
    assert question.missing("alpha, one") == ["beta"]


# -- live path is guarded ------------------------------------------------


@pytest.mark.skipif(not has_credentials(), reason="no API credentials in this env")
def test_live_comparison_smoke(sample_graph: CodeGraph):  # pragma: no cover - live
    from common.client import get_client

    client = get_client()
    comparison = run_comparison(
        FIXTURE_QUESTIONS[:1],
        graph=sample_graph,
        jit_client=client,
        baseline_client=client,
        k=8,
    )
    assert comparison.exact_tokens is True
    assert all(r.prompt_tokens > 0 for r in comparison.results)
