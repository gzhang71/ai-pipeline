"""Diffing two runs: regressions first, and honest aggregates."""

from __future__ import annotations

import json

import pytest

from harness.diff import diff_files, diff_runs, format_json, format_report
from harness.model import ModelOutput, ScriptedClient, make_usage
from harness.runner import load_run, run
from harness.stats import mcnemar_exact, min_detectable_flips, wilson_interval


@pytest.fixture
def four_tasks(write_task):
    """One task per diff outcome: regression, improvement, and both no-changes."""
    return [
        write_task(
            "t_regress",
            'input = "a"\n[[assertions]]\nid = "has_ok"\ntype = "contains"\ntext = "ok"\n',
        ),
        write_task(
            "t_improve",
            'input = "b"\n[[assertions]]\nid = "json"\ntype = "json_valid"\n',
        ),
        write_task(
            "t_same_pass",
            'input = "c"\n[[assertions]]\nid = "has_c"\ntype = "contains"\ntext = "c"\n',
        ),
        write_task(
            "t_same_fail",
            'input = "d"\n[[assertions]]\nid = "has_z"\ntype = "contains"\ntext = "zzz"\n',
        ),
    ]


def make_run(tmp_path, prompt, tasks, texts, *, name, usage=(100, 20)):
    client = ScriptedClient(
        lambda r: ModelOutput(
            text=texts[r.task_id], stop_reason="end_turn", usage=make_usage(*usage)
        )
    )
    path = tmp_path / f"{name}.jsonl"
    run(prompt=prompt, tasks=tasks, client=client, out_path=path, concurrency=1)
    return path


@pytest.fixture
def two_runs(tmp_path, write_prompt, four_tasks):
    v1 = write_prompt("p.v1", "Version one.")
    v2 = write_prompt("p.v2", "Version two.")
    before = make_run(
        tmp_path,
        v1,
        four_tasks,
        {
            "t_regress": "ok",          # passes
            "t_improve": "not json",    # fails
            "t_same_pass": "c",         # passes
            "t_same_fail": "nope",      # fails
        },
        name="before",
        usage=(100, 40),
    )
    after = make_run(
        tmp_path,
        v2,
        four_tasks,
        {
            "t_regress": "nope",        # now fails  -> regression
            "t_improve": "{}",          # now passes -> improvement
            "t_same_pass": "c",         # still passes
            "t_same_fail": "nope",      # still fails
        },
        name="after",
        usage=(120, 25),
    )
    return before, after


def test_identifies_regression_improvement_and_no_change(two_runs):
    report = diff_files(*two_runs)
    assert [d.task_id for d in report.regressions] == ["t_regress"]
    assert [d.task_id for d in report.improvements] == ["t_improve"]
    assert [d.task_id for d in report.unchanged_pass] == ["t_same_pass"]
    assert [d.task_id for d in report.unchanged_fail] == ["t_same_fail"]
    assert report.compared == 4


def test_regression_names_the_assertion_that_broke(two_runs):
    report = diff_files(*two_runs)
    regression = report.regressions[0]
    assert regression.before is True and regression.after is False
    assert regression.newly_failing == ("has_ok",)
    assert "does not contain" in regression.detail


def test_improvement_names_the_assertion_that_was_fixed(two_runs):
    report = diff_files(*two_runs)
    assert report.improvements[0].newly_passing == ("json",)


def test_pass_rate_can_be_flat_while_a_task_regressed(two_runs):
    """The headline number hides the trade. That is the whole point."""
    report = diff_files(*two_runs)
    assert report.before_passed == 2 and report.after_passed == 2
    assert report.rate_delta == 0.0
    assert report.regressions, "a flat pass rate is not the same as no change"


def test_token_delta(two_runs):
    report = diff_files(*two_runs)
    delta = report.token_delta()
    assert delta["total_prompt_tokens"] == 4 * (120 - 100)
    assert delta["output_tokens"] == 4 * (25 - 40)
    assert report.estimated_cost("after") < report.estimated_cost("before")


def test_per_task_token_delta(two_runs):
    report = diff_files(*two_runs)
    assert all(d.token_delta == (120 + 25) - (100 + 40) for d in report.deltas)


def test_significance_of_a_one_for_one_trade(two_runs):
    report = diff_files(*two_runs)
    assert report.p_value == 1.0
    assert report.significant is False


def test_a_clean_sweep_of_regressions_is_significant(tmp_path, write_prompt, write_task):
    tasks = [
        write_task(
            f"t{i}", f'input = "x"\n[[assertions]]\ntype = "contains"\ntext = "ok"\n'
        )
        for i in range(8)
    ]
    v1 = write_prompt("p.v1", "One.")
    v2 = write_prompt("p.v2", "Two.")
    before = make_run(tmp_path, v1, tasks, {t.id: "ok" for t in tasks}, name="b")
    after = make_run(tmp_path, v2, tasks, {t.id: "no" for t in tasks}, name="a")
    report = diff_files(before, after)
    assert len(report.regressions) == 8
    assert report.significant is True
    assert report.rate_delta == -1.0


def test_report_leads_with_regressions(two_runs):
    text = format_report(diff_files(*two_runs))
    assert text.index("REGRESSIONS") < text.index("improvements")
    assert text.index("REGRESSIONS") < text.index("pass rate")
    assert "t_regress" in text


def test_report_states_the_noise_floor(two_runs):
    text = format_report(diff_files(*two_runs))
    assert "95% CI" in text
    assert "noise floor" in text
    assert "McNemar" in text


def test_report_with_no_regressions_says_so(tmp_path, write_prompt, four_tasks):
    v1 = write_prompt("p.v1", "One.")
    v2 = write_prompt("p.v2", "Two.")
    texts = {"t_regress": "ok", "t_improve": "{}", "t_same_pass": "c", "t_same_fail": "n"}
    before = make_run(tmp_path, v1, four_tasks, texts, name="b")
    after = make_run(tmp_path, v2, four_tasks, texts, name="a")
    report = diff_files(before, after)
    assert report.regressions == []
    assert "REGRESSIONS (0)" in format_report(report)


def test_json_output_is_machine_readable(two_runs):
    payload = json.loads(format_json(diff_files(*two_runs)))
    assert payload["compared"] == 4
    assert [r["task_id"] for r in payload["regressions"]] == ["t_regress"]
    assert [r["task_id"] for r in payload["improvements"]] == ["t_improve"]
    assert payload["before"]["prompt_id"] == "p.v1"
    assert payload["after"]["prompt_id"] == "p.v2"
    assert "token_delta" in payload and "p_value" in payload


# --- warnings ------------------------------------------------------------------


def test_warns_when_comparing_a_prompt_to_itself(tmp_path, write_prompt, four_tasks):
    prompt = write_prompt("p.v1", "One.")
    texts = {"t_regress": "ok", "t_improve": "{}", "t_same_pass": "c", "t_same_fail": "n"}
    before = make_run(tmp_path, prompt, four_tasks, texts, name="b")
    after = make_run(tmp_path, prompt, four_tasks, texts, name="a")
    report = diff_files(before, after)
    assert any("same prompt hash" in w for w in report.warnings)


def test_warns_when_the_task_set_changed(tmp_path, write_prompt, four_tasks, write_task):
    v1 = write_prompt("p.v1", "One.")
    v2 = write_prompt("p.v2", "Two.")
    texts = {"t_regress": "ok", "t_improve": "{}", "t_same_pass": "c", "t_same_fail": "n"}
    before = make_run(tmp_path, v1, four_tasks, texts, name="b")
    edited = write_task(
        "t_same_fail",
        'input = "d2"\n[[assertions]]\nid = "has_z"\ntype = "contains"\ntext = "zzz"\n',
    )
    after = make_run(
        tmp_path, v2, four_tasks[:3] + [edited], texts, name="a"
    )
    report = diff_files(before, after)
    assert any("task set differs" in w for w in report.warnings)


def test_warns_and_scopes_to_shared_tasks(tmp_path, write_prompt, four_tasks):
    v1 = write_prompt("p.v1", "One.")
    v2 = write_prompt("p.v2", "Two.")
    texts = {"t_regress": "ok", "t_improve": "{}", "t_same_pass": "c", "t_same_fail": "n"}
    before = make_run(tmp_path, v1, four_tasks, texts, name="b")
    after = make_run(tmp_path, v2, four_tasks[:3], texts, name="a")
    report = diff_files(before, after)
    assert report.compared == 3
    assert report.only_in_before == ["t_same_fail"]
    assert any("only in the before run" in w for w in report.warnings)


def test_warns_when_unevaluated_assertions_are_counted_as_failures(
    tmp_path, write_prompt, write_task
):
    judged = write_task(
        "t_judged",
        'input = "x"\n[[assertions]]\nid = "v"\ntype = "judge"\ncriterion = "good?"\n',
    )
    v1 = write_prompt("p.v1", "One.")
    v2 = write_prompt("p.v2", "Two.")
    before = make_run(tmp_path, v1, [judged], {"t_judged": "x"}, name="b")
    after = make_run(tmp_path, v2, [judged], {"t_judged": "x"}, name="a")
    report = diff_files(before, after)
    assert any("skipped" in w for w in report.warnings)


def test_diff_runs_accepts_loaded_records(two_runs):
    before, after = two_runs
    report = diff_runs(load_run(before), load_run(after))
    assert report.compared == 4


# --- statistics ----------------------------------------------------------------


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson_interval(24, 30)
    assert low < 24 / 30 < high
    assert high - low > 0.15, "a 30-task set has a wide interval; say so"


def test_wilson_interval_edges():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(30, 30)
    assert high == 1.0 and low < 1.0


def test_mcnemar_symmetry_and_extremes():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(2, 1) == mcnemar_exact(1, 2)
    assert mcnemar_exact(6, 0) < 0.05
    assert mcnemar_exact(2, 0) > 0.05


def test_min_detectable_flips_is_six():
    """Fewer than six one-directional flips cannot clear p=0.05, at any n."""
    assert min_detectable_flips(30) == 6
    assert min_detectable_flips(50) == 6
