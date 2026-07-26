"""The runner, the accounting, and the comparison report.

Everything here is offline. The fake client is deliberately *not* an oracle —
it answers from the context the strategy handed it and says UNKNOWN when the
fact is gone — so these tests measure information survival rather than
asserting that a mock returned what it was told to.
"""

from __future__ import annotations

import pytest

from common.client import has_credentials
from context.bench import (
    LiveClient,
    format_report,
    run_bench,
    run_task,
)
from context.fakes import FakeClient
from context.strategies import (
    Budget,
    NoteTaking,
    ServerCompaction,
    Strategy,
    StrategyResult,
    TailTruncation,
    all_strategies,
)
from context.tasks import TASKS, Task, Turn
from context.tokens import ApiTokenCounter, HeuristicTokenCounter
from context.usage import Usage
from context.validation import InvalidMessageShape

COUNTER = HeuristicTokenCounter()
BUDGET = Budget(counter=COUNTER)


def fake_client(**kwargs) -> FakeClient:
    kwargs.setdefault("compaction_threshold", BUDGET.max_tokens)
    return FakeClient(COUNTER, **kwargs)


def task_by_id(task_id: str) -> Task:
    return next(t for t in TASKS if t.id == task_id)


@pytest.fixture(scope="module")
def report():
    return run_bench()


class TestFakeClientIsNotAnOracle:
    """If the fake knew the answers, every result in this package is noise."""

    def test_answers_from_context_when_the_fact_is_present(self):
        client = fake_client()
        response = client.create(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "[FACT] K = v1"}]},
                {"role": "user", "content": [{"type": "text", "text": "[QUERY] K"}]},
            ]
        )
        assert "K = v1" in response.content[0]["text"]

    def test_says_unknown_when_the_fact_was_compacted_away(self):
        client = fake_client()
        response = client.create(
            messages=[{"role": "user", "content": [{"type": "text", "text": "[QUERY] K"}]}]
        )
        assert "K = UNKNOWN" in response.content[0]["text"]

    def test_takes_the_most_recent_value_of_a_corrected_fact(self):
        client = fake_client()
        response = client.create(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "[FACT] K = old"}]},
                {"role": "user", "content": [{"type": "text", "text": "[FACT] K = new"}]},
                {"role": "user", "content": [{"type": "text", "text": "[QUERY] K"}]},
            ]
        )
        assert "K = new" in response.content[0]["text"]

    def test_does_not_see_the_strategy_or_the_task(self):
        client = fake_client()
        assert not hasattr(client, "task")
        assert not hasattr(client, "strategy")


class TestUsageAccounting:
    def test_prompt_total_is_the_sum_of_all_three_input_fields(self):
        client = fake_client()
        response = client.create(
            messages=[{"role": "user", "content": [{"type": "text", "text": "x" * 4000}]}]
        )
        usage = Usage.from_response_usage(response.usage)
        assert usage.total_prompt_tokens == (
            usage.input_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )
        assert usage.total_tokens == usage.total_prompt_tokens + usage.output_tokens

    def test_input_tokens_alone_badly_understates_a_cache_warm_run(self):
        """The trap this accounting exists to avoid."""
        client = fake_client()
        messages = [{"role": "user", "content": [{"type": "text", "text": "y" * 8000}]}]
        client.create(messages=messages)  # warms the cache
        usage = Usage.from_response_usage(client.create(messages=messages).usage)

        assert usage.cache_read_input_tokens > 0
        cached_fraction = usage.cache_read_input_tokens / usage.total_prompt_tokens
        assert cached_fraction > 0.9, "fixture should be a ~90%-cached request"
        assert usage.input_tokens < usage.total_prompt_tokens / 10

    def test_usage_addition_accumulates_every_field(self):
        a = Usage(1, 2, 3, 4, model_calls=1)
        b = Usage(10, 20, 30, 40, model_calls=1)
        total = a + b
        assert total.as_dict() == {
            "input_tokens": 11,
            "cache_creation_input_tokens": 22,
            "cache_read_input_tokens": 33,
            "output_tokens": 44,
            "total_prompt_tokens": 66,
            "total_tokens": 110,
            "model_calls": 2,
        }
        assert sum([a, b], Usage()) == total

    def test_the_strategys_own_model_calls_are_charged_to_it(self):
        """Otherwise summarizing strategies look free and the ranking is rigged."""
        task = task_by_id("early-constant")
        summarizing = run_task(
            task, NoteTaking(), client=fake_client(), budget=BUDGET
        )
        truncating = run_task(
            task, TailTruncation(), client=fake_client(), budget=BUDGET
        )

        assert summarizing.strategy_usage.model_calls > 0
        assert summarizing.strategy_usage.total_tokens > 0
        assert truncating.strategy_usage.model_calls == 0
        assert (
            summarizing.total_tokens
            > summarizing.agent_usage.total_tokens
        ), "strategy usage vanished from the total"

    def test_agent_and_strategy_calls_are_reported_separately(self, report):
        summaries = {s.strategy: s for s in report.summaries()}
        assert summaries["TailTruncation"].strategy_calls == 0
        assert summaries["NoteTaking"].strategy_calls > 0
        # The agent loop is identical for every strategy, by construction.
        assert len({s.agent_calls for s in summaries.values()}) == 1


class TestRunner:
    def test_produces_a_result_with_the_expected_shape(self):
        result = run_task(
            task_by_id("late-fact"), TailTruncation(), client=fake_client(), budget=BUDGET
        )
        assert result.task_id == "late-fact"
        assert result.strategy == "TailTruncation"
        assert isinstance(result.success, bool)
        assert result.wall_clock_s >= 0
        assert result.turns, "per-turn records were not captured"
        assert result.agent_usage.model_calls == len(task_by_id("late-fact").turns) + 1

    def test_validates_every_request_it_sends(self):
        class BrokenStrategy(Strategy):
            name = "Broken"

            def apply(self, messages, budget):
                # A trailing assistant turn: a prefill, rejected on Opus 5.
                return StrategyResult(
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "no"}]},
                    ]
                )

        with pytest.raises(InvalidMessageShape):
            run_task(
                task_by_id("late-fact"),
                BrokenStrategy(),
                client=fake_client(),
                budget=BUDGET,
            )

    def test_records_when_a_strategy_failed_to_get_under_budget(self):
        """A strategy can be perfectly legal and still not solve the problem."""
        results = [
            run_task(t, s, client=fake_client(), budget=BUDGET)
            for s in all_strategies()
            for t in [task_by_id("early-constant")]
        ]
        by_name = {r.strategy: r for r in results}
        assert by_name["ToolResultEviction"].over_budget_turns > 0
        assert by_name["TailTruncation"].over_budget_turns == 0

    def test_the_transcript_is_identical_across_strategies(self):
        """Same scripted turns for everyone, or the comparison is meaningless."""
        counts = {
            s.name: run_task(
                task_by_id("early-constant"), s, client=fake_client(), budget=BUDGET
            ).agent_usage.model_calls
            for s in all_strategies()
        }
        assert len(set(counts.values())) == 1


class TestServerCompactionContract:
    def test_the_beta_and_edit_reach_the_client(self):
        client = fake_client()
        run_task(
            task_by_id("early-constant"), ServerCompaction(), client=client, budget=BUDGET
        )
        overrides = client.calls_log[-1].overrides
        assert overrides["betas"] == ["compact-2026-01-12"]
        assert overrides["context_management"]["edits"][0]["type"] == "compact_20260112"

    def test_the_server_actually_compacts(self):
        client = fake_client()
        run_task(
            task_by_id("early-constant"), ServerCompaction(), client=client, budget=BUDGET
        )
        assert client.compactions > 0

    def test_dropping_the_compaction_block_makes_the_server_redo_the_work(self):
        """Append only the text and the compaction state is silently lost.

        The runner appends the whole `response.content` by default. With
        `append_full_content=False` the compaction block never survives into
        the next request, so the server has to recompact from scratch every
        single turn.
        """
        task = task_by_id("early-constant")

        correct_client = fake_client()
        run_task(
            task,
            ServerCompaction(),
            client=correct_client,
            budget=BUDGET,
            append_full_content=True,
        )

        broken_client = fake_client()
        run_task(
            task,
            ServerCompaction(),
            client=broken_client,
            budget=BUDGET,
            append_full_content=False,
        )

        assert correct_client.compactions >= 1, "fixture never triggered compaction"
        assert broken_client.compactions > correct_client.compactions, (
            "dropping the compaction block should force redundant recompaction"
        )
        # Done correctly, the server compacts once and then coasts on the
        # block; done wrongly, it pays on nearly every call.
        broken_rate = broken_client.compactions / len(broken_client.calls_log)
        correct_rate = correct_client.compactions / len(correct_client.calls_log)
        assert broken_rate > correct_rate


class TestReport:
    def test_covers_every_strategy_and_every_task(self, report):
        summaries = report.summaries()
        assert {s.strategy for s in summaries} == {s.name for s in all_strategies()}
        for summary in summaries:
            assert summary.tasks == len(TASKS)

    def test_ranks_by_measured_success_then_tokens(self, report):
        summaries = report.summaries()
        keys = [(-s.success_rate, s.total_tokens) for s in summaries]
        assert keys == sorted(keys), "report is not ranked by its own measurements"

    def test_every_strategy_solves_the_control_task(self, report):
        """If a strategy fails the late-fact control, the harness is broken."""
        for result in report.results:
            if result.task_id == "late-fact":
                assert result.success, f"{result.strategy} failed the control task"

    def test_tail_truncation_loses_early_information(self, report):
        summaries = {s.strategy: s for s in report.summaries()}
        assert summaries["TailTruncation"].early_recall == 0.0

    def test_at_least_one_strategy_beats_the_baseline_on_early_recall(self, report):
        summaries = {s.strategy: s for s in report.summaries()}
        baseline = summaries["TailTruncation"].early_recall
        assert max(s.early_recall for s in summaries.values()) > baseline

    def test_early_failures_are_attributed_to_named_tasks(self, report):
        for summary in report.summaries():
            for task_id in summary.early_failures:
                assert task_id in {t.id for t in TASKS}
                assert task_id in summary.failed_tasks

    def test_no_strategy_is_perfect(self, report):
        """A clean sweep would mean the task set stopped discriminating."""
        assert any(s.successes < s.tasks for s in report.summaries())

    def test_formats_a_readable_report(self, report):
        text = format_report(report)
        for strategy in all_strategies():
            assert strategy.name in text
        assert "early-information loss" in text
        assert "input + cache_creation + cache_read" in text
        assert "SIMULATED" in text, "the simulation caveat must be stated in the output"

    def test_run_bench_accepts_a_subset(self):
        small = run_bench(
            strategies=[TailTruncation()], tasks=[task_by_id("late-fact")]
        )
        assert len(small.results) == 1
        assert small.summaries()[0].strategy == "TailTruncation"


class TestLivePathsAreGuarded:
    @pytest.mark.skipif(has_credentials(), reason="credentials are present")
    def test_live_client_refuses_to_construct_without_credentials(self):
        with pytest.raises(RuntimeError, match="no credentials"):
            LiveClient()

    @pytest.mark.skipif(not has_credentials(), reason="no API credentials")
    def test_exact_token_counting_against_the_api(self):
        counter = ApiTokenCounter()
        assert counter.count([{"role": "user", "content": "hello"}]) > 0

    @pytest.mark.skipif(not has_credentials(), reason="no API credentials")
    def test_bench_against_the_live_api(self):
        report = run_bench(
            strategies=[TailTruncation()],
            tasks=[task_by_id("late-fact")],
            client_factory=LiveClient,
        )
        assert report.results[0].agent_usage.total_prompt_tokens > 0
