"""Accuracy vs. context length, including the end-to-end synthetic task path."""

from __future__ import annotations

import pytest

from common.client import has_credentials
from loop.accuracy import (
    Observation,
    analyze_accuracy,
    observations_from_records,
    report_text,
    wilson_interval,
)
from loop.agent import LoopConfig
from loop.render import render_html
from loop.schema import validate_run
from loop.sink import MemorySink
from loop.tasks import SYNTHETIC_TASKS, Task, make_executor, run_task_set
from loop.testing import (
    FakeAnthropicClient,
    FakeMessage,
    heuristic_token_count,
    text_block,
    tool_use_block,
)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(7, 10)
    assert 0.0 <= low <= 0.7 <= high <= 1.0


def test_wilson_interval_is_well_behaved_at_the_extremes():
    assert wilson_interval(0, 5) == pytest.approx((0.0, wilson_interval(0, 5)[1]))
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_more_data_narrows_the_interval():
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------
# Bend detection
# --------------------------------------------------------------------------


def _cliff(cutoff: int, per_length: int = 10):
    """Perfect below `cutoff`, hopeless above it."""
    observations = []
    for index, tokens in enumerate((1_000, 5_000, 20_000, 60_000)):
        for repeat in range(per_length):
            observations.append(
                Observation(
                    run_id=f"r{index}_{repeat}",
                    task_id="fixed",
                    prompt_tokens=tokens + repeat,
                    success=tokens < cutoff,
                )
            )
    return observations


def test_finds_the_bend_in_an_obvious_cliff():
    report = analyze_accuracy(_cliff(cutoff=20_000), n_bins=4)
    assert report.bend_tokens is not None
    assert 5_000 < report.bend_tokens <= 20_000
    assert report.bend_drop == pytest.approx(1.0)
    assert report.significant is True


def test_reports_no_bend_when_the_curve_is_flat():
    observations = [
        Observation(f"r{i}", "fixed", 1_000 * (i + 1), success=True) for i in range(20)
    ]
    report = analyze_accuracy(observations, n_bins=4)
    assert report.bend_tokens is None
    assert any("flat or rising" in c for c in report.caveats)


def test_a_noisy_one_bin_drop_is_reported_but_flagged_insignificant():
    observations = [
        Observation("a1", "fixed", 1_000, True),
        Observation("a2", "fixed", 1_100, True),
        Observation("b1", "fixed", 9_000, True),
        Observation("b2", "fixed", 9_100, False),
    ]
    report = analyze_accuracy(observations, n_bins=2, min_bin=2)
    assert report.bend_tokens is not None
    assert report.significant is False
    assert any("not significant" in c for c in report.caveats)


def test_mixed_task_families_are_flagged_as_confounded():
    observations = [
        Observation("a", "easy", 500, True),
        Observation("b", "hard", 50_000, False),
    ]
    report = analyze_accuracy(observations, n_bins=2)
    assert any("confounded" in c for c in report.caveats)


def test_one_family_at_many_lengths_is_not_flagged_as_confounded():
    """The whole point of the synthetic design: same task, varying padding."""
    observations = [
        Observation(
            f"r{i}",
            task_id=f"invoice@{i}",
            prompt_tokens=1_000 * (i + 1),
            success=i < 3,
            metadata={"family": "invoice_amount"},
        )
        for i in range(6)
    ]
    report = analyze_accuracy(observations, n_bins=3, min_bin=1)
    assert not any("confounded" in c for c in report.caveats)


def test_too_few_bins_to_estimate_a_bend():
    report = analyze_accuracy([Observation("a", "t", 100, True)], n_bins=4, min_bin=2)
    assert report.bend_tokens is None
    assert any("fewer than two bins" in c for c in report.caveats)


def test_no_observations_is_not_a_crash():
    report = analyze_accuracy([])
    assert report.n == 0
    assert report.bend_tokens is None
    assert report.caveats == ["no observations"]
    assert "n/a" in report_text(report)


def test_small_samples_are_called_anecdotes():
    report = analyze_accuracy(_cliff(cutoff=20_000, per_length=2), n_bins=4)
    assert any("anecdote-scale" in c for c in report.caveats)


def test_bins_partition_the_observations():
    observations = _cliff(cutoff=20_000)
    report = analyze_accuracy(observations, n_bins=4)
    assert sum(b.n for b in report.bins) == len(observations)
    assert sum(b.successes for b in report.bins) == sum(o.success for o in observations)


@pytest.mark.parametrize("binning", ["quantile", "linear"])
def test_both_binning_strategies_work(binning):
    report = analyze_accuracy(_cliff(cutoff=20_000), n_bins=4, binning=binning)
    assert report.n == 40
    assert report.bins


def test_unknown_binning_is_rejected():
    with pytest.raises(ValueError):
        analyze_accuracy(_cliff(20_000), binning="magic")


def test_report_text_shows_the_curve():
    report = analyze_accuracy(_cliff(cutoff=20_000), n_bins=4)
    text = report_text(report)
    assert "accuracy vs. prompt length" in text
    assert "bend:" in text
    assert "significant" in text
    assert "#" in text  # the bar itself
    # A clean, adequately-powered single-family run has nothing to warn about.
    assert report.caveats == []


def test_report_text_surfaces_caveats_when_there_are_any():
    report = analyze_accuracy(_cliff(cutoff=20_000, per_length=2), n_bins=4)
    text = report_text(report)
    assert "caveats:" in text
    assert "anecdote-scale" in text


def test_report_record_is_json_shaped():
    report = analyze_accuracy(_cliff(cutoff=20_000), n_bins=4)
    record = report.to_record()
    assert record["n"] == 40
    assert record["bend_tokens"] == report.bend_tokens
    assert len(record["bins"]) == len(report.bins)
    assert isinstance(record["caveats"], list)


# --------------------------------------------------------------------------
# End to end over the synthetic task set
# --------------------------------------------------------------------------


def _length_sensitive_client(fail_above: int):
    """A model that answers correctly until the prompt gets long, then gives up.

    Crude, but it is a *behaviour*, not a canned answer list: the response
    depends on the prompt the loop actually built, so a bug that stopped the
    filler reaching the request would show up as a flat curve.
    """

    def respond(**kwargs):
        size = heuristic_token_count(
            messages=kwargs.get("messages", []),
            system=kwargs.get("system"),
            tools=kwargs.get("tools"),
        )
        messages = kwargs.get("messages", [])
        already_looked_up = any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for message in messages
            for block in (
                message["content"] if isinstance(message.get("content"), list) else []
            )
        )
        if size > fail_above:
            return FakeMessage(
                content=[text_block("I am not sure.")], stop_reason="end_turn"
            )
        if not already_looked_up:
            return FakeMessage(
                content=[tool_use_block("lookup_invoice", {"invoice_id": "INV-1003"}, "toolu_1")],
                stop_reason="tool_use",
            )
        return FakeMessage(content=[text_block("1204000")], stop_reason="end_turn")

    return FakeAnthropicClient([respond])


def test_task_checker_accepts_the_right_answer_and_rejects_others():
    task = SYNTHETIC_TASKS[0]
    assert task.check("1204000") is True
    assert task.check("The total is 1,204,000") is True
    assert task.check("1204001") is False
    assert task.check("I am not sure.") is False
    assert task.check("") is False


def test_filler_scales_the_prompt_without_changing_the_answer():
    short, long = SYNTHETIC_TASKS[0], SYNTHETIC_TASKS[-1]
    assert short.expected == long.expected
    assert len(long.system_prompt()) > 10 * len(short.system_prompt())
    # Padding must not smuggle in the answer or an invoice id.
    assert "INV-" not in long.system_prompt().split("</reference_material>")[0].split(
        "<reference_material>"
    )[1]
    assert long.expected not in long.system_prompt()


def test_synthetic_task_set_runs_end_to_end_and_produces_a_bent_curve():
    sink = MemorySink()
    runs = run_task_set(
        client=_length_sensitive_client(fail_above=1_500),
        sink=sink,
        counter=heuristic_token_count,
        config=LoopConfig(max_iterations=4),
    )
    assert len(runs) == len(SYNTHETIC_TASKS)
    validate_run(sink.records)

    # Short prompts succeed, long ones do not.
    successes = [r.success for r in runs]
    assert successes[0] is True
    assert successes[-1] is False

    report = analyze_accuracy([r.observation() for r in runs], n_bins=3, min_bin=1)
    assert report.bend_tokens is not None
    assert report.overall_rate < 1.0
    # Single task family, so no difficulty confound is reported.
    assert not any("confounded" in c for c in report.caveats)
    # But it is still tiny, and the report says so.
    assert any("anecdote-scale" in c for c in report.caveats)

    html = render_html(sink.records, accuracy=report)
    assert "accuracy vs. context length" in html


def test_observations_can_be_rebuilt_from_the_jsonl_alone(tmp_path):
    sink = MemorySink()
    runs = run_task_set(
        client=_length_sensitive_client(fail_above=1_500),
        sink=sink,
        counter=heuristic_token_count,
        config=LoopConfig(max_iterations=4),
    )
    outcomes = {r.result.run_id: r.success for r in runs}
    rebuilt = observations_from_records(sink.records, outcomes)
    assert len(rebuilt) == len(runs)
    direct = {o.run_id: o for o in (r.observation() for r in runs)}
    for observation in rebuilt:
        assert observation.prompt_tokens == direct[observation.run_id].prompt_tokens
        assert observation.success == direct[observation.run_id].success
        assert observation.task_id == direct[observation.run_id].task_id


def test_observations_from_records_supports_alternative_length_selectors():
    sink = MemorySink()
    runs = run_task_set(
        client=_length_sensitive_client(fail_above=10**9),
        tasks=SYNTHETIC_TASKS[:2],
        sink=sink,
        counter=heuristic_token_count,
        config=LoopConfig(max_iterations=4),
    )
    outcomes = {r.result.run_id: r.success for r in runs}
    peak = {o.run_id: o.prompt_tokens for o in observations_from_records(sink.records, outcomes)}
    first = {
        o.run_id: o.prompt_tokens
        for o in observations_from_records(sink.records, outcomes, length="first")
    }
    total = {
        o.run_id: o.prompt_tokens
        for o in observations_from_records(sink.records, outcomes, length="total")
    }
    for run_id in peak:
        assert first[run_id] <= peak[run_id] <= total[run_id]


def test_unknown_length_selector_is_rejected():
    with pytest.raises(ValueError):
        observations_from_records([], {}, length="sideways")


def test_tool_executor_rejects_unknown_invoices_and_tools():
    execute = make_executor()
    assert "1204000" in execute("lookup_invoice", {"invoice_id": "inv-1003"}, "t")
    with pytest.raises(KeyError):
        execute("lookup_invoice", {"invoice_id": "INV-9999"}, "t")
    with pytest.raises(ValueError):
        execute("something_else", {}, "t")


def test_custom_task_sets_are_supported():
    task = Task(task_id="custom", question="q", expected="1204000", filler_tokens=0)
    runs = run_task_set(
        client=_length_sensitive_client(fail_above=10**9),
        tasks=[task],
        counter=heuristic_token_count,
        config=LoopConfig(max_iterations=4),
    )
    assert len(runs) == 1
    assert runs[0].success is True


# --------------------------------------------------------------------------
# Live path stays guarded
# --------------------------------------------------------------------------


@pytest.mark.skipif(not has_credentials(), reason="no API credentials in this environment")
def test_live_smoke_against_the_real_api():  # pragma: no cover - needs credentials
    from common.client import get_client

    sink = MemorySink()
    runs = run_task_set(
        client=get_client(),
        tasks=SYNTHETIC_TASKS[:1],
        sink=sink,
        config=LoopConfig(max_iterations=4),
    )
    validate_run(sink.records)
    assert runs[0].result.turns >= 1
