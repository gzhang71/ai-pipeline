"""The runner: request construction, run records, resumption, concurrency."""

from __future__ import annotations

import json

import pytest

from harness.judge import Judge, JudgeCache
from harness.model import ModelOutput, ScriptedClient, make_usage
from harness.runner import (
    RunError,
    build_request,
    default_run_path,
    evaluate_task,
    load_run,
    run,
)

REQUIRED_META_KEYS = {
    "type",
    "run_id",
    "created_at",
    "prompt_id",
    "prompt_hash",
    "task_set_hash",
    "task_count",
    "model",
    "concurrency",
    "judge_prompt_id",
    "judge_hash",
}
REQUIRED_RESULT_KEYS = {
    "type",
    "run_id",
    "prompt_id",
    "prompt_hash",
    "task_id",
    "task_hash",
    "passed",
    "status",
    "assertions",
    "output_text",
    "output_hash",
    "tool_calls",
    "stop_reason",
    "usage",
    "model",
    "duration_ms",
    "error",
    "finished_at",
}


@pytest.fixture
def tasks(write_task):
    return [
        write_task(
            "t_pass",
            'input = "say ok"\n[[assertions]]\nid = "ok"\ntype = "contains"\ntext = "ok"\n',
        ),
        write_task(
            "t_fail",
            'input = "say json"\n[[assertions]]\nid = "j"\ntype = "json_valid"\n',
        ),
    ]


def client_for(mapping, *, usage=(100, 20)):
    def respond(request):
        text = mapping.get(request.task_id, "")
        if isinstance(text, ModelOutput):
            return text
        return ModelOutput(text=text, stop_reason="end_turn", usage=make_usage(*usage))

    return ScriptedClient(respond)


# --- request construction ------------------------------------------------------


def test_build_request_uses_prompt_and_task(write_prompt, write_task):
    prompt = write_prompt("p.v1", "Base system.", meta='effort = "high"')
    task = write_task(
        "t",
        """
        input = "the input"
        system_suffix = "Extra rule."
        max_tokens = 321

        [[tools]]
        name = "lookup"

        [[assertions]]
        type = "json_valid"
        """,
    )
    request = build_request(prompt, task)
    assert request.system == "Base system.\n\nExtra rule."
    assert request.messages == ({"role": "user", "content": "the input"},)
    assert request.max_tokens == 321
    assert request.effort == "high"
    assert request.tools[0]["name"] == "lookup"
    assert request.prompt_hash == prompt.hash and request.task_id == "t"


# --- run records ---------------------------------------------------------------


def test_run_writes_a_well_formed_record(tmp_path, simple_prompt, tasks):
    out = tmp_path / "run.jsonl"
    summary = run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client_for({"t_pass": "ok", "t_fail": "not json"}),
        out_path=out,
        concurrency=1,
    )
    lines = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    meta, results = lines[0], lines[1:]

    assert REQUIRED_META_KEYS <= set(meta)
    assert meta["type"] == "run_meta"
    assert meta["prompt_hash"] == simple_prompt.hash
    assert meta["task_count"] == 2

    assert len(results) == 2
    for record in results:
        assert REQUIRED_RESULT_KEYS <= set(record)
        assert record["run_id"] == meta["run_id"]
        assert record["prompt_hash"] == simple_prompt.hash
        assert isinstance(record["passed"], bool)
        assert record["assertions"]

    assert summary.passed == 1
    assert summary.total == 2
    assert summary.pass_rate == 0.5


def test_results_reflect_the_assertions(tmp_path, simple_prompt, tasks):
    summary = run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client_for({"t_pass": "ok", "t_fail": "not json"}),
        out_path=tmp_path / "run.jsonl",
        concurrency=1,
    )
    by_id = {r.task_id: r for r in summary.results}
    assert by_id["t_pass"].passed is True
    assert by_id["t_pass"].status == "ok"
    assert by_id["t_fail"].passed is False
    assert [a.id for a in by_id["t_fail"].failed_assertions] == ["j"]
    assert "not valid JSON" in by_id["t_fail"].failed_assertions[0].detail


def test_usage_is_captured_per_task_and_summed(tmp_path, simple_prompt, tasks):
    summary = run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client_for({"t_pass": "ok", "t_fail": "{}"}, usage=(100, 20)),
        out_path=tmp_path / "run.jsonl",
        concurrency=1,
    )
    for result in summary.results:
        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 20
        assert result.usage["total_prompt_tokens"] == 100
    total = summary.usage()
    assert total["input_tokens"] == 200
    assert total["output_tokens"] == 40


def test_cache_tokens_are_counted_in_total_prompt_tokens(tmp_path, simple_prompt, tasks):
    output = ModelOutput(
        text="ok", stop_reason="end_turn", usage=make_usage(10, 5, cache_read=990)
    )
    summary = run(
        prompt=simple_prompt,
        tasks=tasks[:1],
        client=ScriptedClient(lambda r: output),
        out_path=tmp_path / "run.jsonl",
        concurrency=1,
    )
    assert summary.results[0].usage["total_prompt_tokens"] == 1000


def test_model_error_is_recorded_not_raised(tmp_path, simple_prompt, tasks):
    client = ScriptedClient(lambda r: ModelOutput(error="APIConnectionError: down"))
    summary = run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client,
        out_path=tmp_path / "run.jsonl",
        concurrency=1,
    )
    assert summary.passed == 0
    for result in summary.results:
        assert result.status == "error"
        assert "APIConnectionError" in (result.error or "")


def test_output_text_is_truncated_but_hashed_in_full(tmp_path, simple_prompt, tasks):
    from harness.runner import MAX_STORED_OUTPUT_CHARS

    long_text = "ok" + "x" * (MAX_STORED_OUTPUT_CHARS + 100)
    summary = run(
        prompt=simple_prompt,
        tasks=tasks[:1],
        client=client_for({"t_pass": long_text}),
        out_path=tmp_path / "run.jsonl",
        concurrency=1,
    )
    result = summary.results[0]
    assert result.output_truncated is True
    assert len(result.output_text) == MAX_STORED_OUTPUT_CHARS
    from harness.hashing import hash_output

    assert result.output_hash == hash_output(long_text, ())


def test_on_result_callback_fires(tmp_path, simple_prompt, tasks):
    seen = []
    run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client_for({"t_pass": "ok", "t_fail": "{}"}),
        out_path=tmp_path / "run.jsonl",
        concurrency=1,
        on_result=seen.append,
    )
    assert {r.task_id for r in seen} == {"t_pass", "t_fail"}


# --- resumption ----------------------------------------------------------------


def test_resume_skips_already_recorded_tasks(tmp_path, simple_prompt, tasks):
    out = tmp_path / "run.jsonl"
    first = client_for({"t_pass": "ok", "t_fail": "{}"})
    run(prompt=simple_prompt, tasks=tasks[:1], client=first, out_path=out, concurrency=1)
    assert first.call_count == 1

    second = client_for({"t_pass": "ok", "t_fail": "{}"})
    summary = run(
        prompt=simple_prompt, tasks=tasks, client=second, out_path=out, concurrency=1
    )
    assert second.call_count == 1, "the completed task must not be re-run"
    assert summary.skipped == ["t_pass"]
    assert summary.total == 2
    assert summary.passed == 2


def test_a_crash_mid_run_keeps_completed_results(tmp_path, simple_prompt, tasks):
    """Simulate a crash: only one task got written before the process died."""
    out = tmp_path / "run.jsonl"
    boom = {"count": 0}

    def flaky(request):
        boom["count"] += 1
        if request.task_id == "t_fail":
            raise KeyboardInterrupt("pretend the process died")
        return ModelOutput(text="ok", stop_reason="end_turn", usage=make_usage(10, 2))

    with pytest.raises(KeyboardInterrupt):
        run(
            prompt=simple_prompt,
            tasks=tasks,
            client=ScriptedClient(flaky),
            out_path=out,
            concurrency=1,
        )

    partial = load_run(out)
    assert set(partial.results) == {"t_pass"}

    summary = run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client_for({"t_pass": "ok", "t_fail": "{}"}),
        out_path=out,
        concurrency=1,
    )
    assert summary.total == 2 and summary.passed == 2


def test_refuses_to_mix_prompt_versions_in_one_file(tmp_path, simple_prompt, write_prompt, tasks):
    out = tmp_path / "run.jsonl"
    run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client_for({"t_pass": "ok", "t_fail": "{}"}),
        out_path=out,
        concurrency=1,
    )
    edited = write_prompt("p.v1", "You are a test assistant. Answer briefly, please.")
    assert edited.hash != simple_prompt.hash
    with pytest.raises(RunError, match="corrupt the diff"):
        run(
            prompt=edited,
            tasks=tasks,
            client=client_for({"t_pass": "ok", "t_fail": "{}"}),
            out_path=out,
            concurrency=1,
        )


def test_no_resume_refuses_an_existing_file(tmp_path, simple_prompt, tasks):
    out = tmp_path / "run.jsonl"
    client = client_for({"t_pass": "ok", "t_fail": "{}"})
    run(prompt=simple_prompt, tasks=tasks, client=client, out_path=out, concurrency=1)
    with pytest.raises(RunError, match="already exists"):
        run(
            prompt=simple_prompt,
            tasks=tasks,
            client=client,
            out_path=out,
            concurrency=1,
            resume=False,
        )


def test_load_run_tolerates_a_torn_final_line(tmp_path, simple_prompt, tasks):
    out = tmp_path / "run.jsonl"
    run(
        prompt=simple_prompt,
        tasks=tasks,
        client=client_for({"t_pass": "ok", "t_fail": "{}"}),
        out_path=out,
        concurrency=1,
    )
    with out.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "result", "task_id": "t_half"')  # killed mid-write
    record = load_run(out)
    assert set(record.results) == {"t_pass", "t_fail"}


# --- concurrency ---------------------------------------------------------------


def test_concurrency_records_every_task(tmp_path, simple_prompt, write_task):
    many = [
        write_task(
            f"t{i:02d}",
            f'input = "n{i}"\n[[assertions]]\ntype = "contains"\ntext = "n{i}"\n',
        )
        for i in range(20)
    ]
    client = ScriptedClient(
        lambda r: ModelOutput(text=r.messages[0]["content"], stop_reason="end_turn",
                              usage=make_usage(5, 1))
    )
    out = tmp_path / "run.jsonl"
    summary = run(
        prompt=simple_prompt, tasks=many, client=client, out_path=out, concurrency=8
    )
    assert summary.total == 20 and summary.passed == 20
    assert len(load_run(out).results) == 20
    assert {r.task_id for r in summary.results} == {t.id for t in many}


def test_concurrency_must_be_positive(tmp_path, simple_prompt, tasks):
    with pytest.raises(RunError):
        run(
            prompt=simple_prompt,
            tasks=tasks,
            client=client_for({}),
            out_path=tmp_path / "r.jsonl",
            concurrency=0,
        )


def test_empty_task_list_is_an_error(tmp_path, simple_prompt):
    with pytest.raises(RunError, match="no tasks"):
        run(
            prompt=simple_prompt,
            tasks=[],
            client=client_for({}),
            out_path=tmp_path / "r.jsonl",
        )


# --- judge integration ---------------------------------------------------------


@pytest.fixture
def judged_task(write_task):
    return write_task(
        "t_judged",
        """
        input = "vague"

        [[assertions]]
        id = "json"
        type = "json_valid"

        [[assertions]]
        id = "verdict"
        type = "judge"
        criterion = "names the missing information"
        """,
    )


def test_judge_assertion_without_a_judge_is_recorded_as_unevaluated(
    tmp_path, simple_prompt, judged_task
):
    summary = run(
        prompt=simple_prompt,
        tasks=[judged_task],
        client=client_for({"t_judged": "{}"}),
        out_path=tmp_path / "run.jsonl",
        concurrency=1,
    )
    result = summary.results[0]
    assert result.status == "incomplete"
    assert result.passed is False
    skipped = [a for a in result.assertions if a.skipped]
    assert [a.id for a in skipped] == ["verdict"]


def test_judge_assertion_is_evaluated_and_its_usage_recorded(
    tmp_path, simple_prompt, judged_task, write_prompt
):
    import json as _json

    judge_prompt = write_prompt("judge.v1", "Grade one criterion.")
    judge_client = ScriptedClient(
        lambda r: ModelOutput(
            text=_json.dumps({"verdict": "pass", "confidence": 0.9, "reasoning": "yes"}),
            stop_reason="end_turn",
            usage=make_usage(300, 30),
        )
    )
    judge = Judge(client=judge_client, prompt=judge_prompt, cache=JudgeCache(None))
    summary = run(
        prompt=simple_prompt,
        tasks=[judged_task],
        client=client_for({"t_judged": "{}"}),
        out_path=tmp_path / "run.jsonl",
        judge=judge,
        concurrency=1,
    )
    result = summary.results[0]
    assert result.passed is True and result.status == "ok"
    assert result.judge_usage["output_tokens"] == 30
    assert summary.meta["judge_prompt_id"] == "judge.v1"
    assert summary.meta["judge_hash"] == judge_prompt.hash
    assert summary.usage()["output_tokens"] == 20 + 30


def test_evaluate_task_is_usable_standalone(simple_prompt, simple_task, make_output):
    result = evaluate_task(
        prompt=simple_prompt,
        task=simple_task,
        output=make_output("ok"),
        judge=None,
        run_id="r1",
    )
    assert result.passed is True
    assert result.run_id == "r1"


def test_default_run_path_includes_the_prompt_hash(simple_prompt):
    path = default_run_path(simple_prompt)
    assert path.name == f"p.v1.{simple_prompt.short_hash}.jsonl"
