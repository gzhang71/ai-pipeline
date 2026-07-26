"""The LLM judge: structured verdicts, versioning, and caching that actually hits."""

from __future__ import annotations

import json

import pytest

from harness.judge import VERDICT_SCHEMA, Judge, JudgeCache, cache_key
from harness.model import ModelOutput, ModelRequest, ScriptedClient, make_usage

CRITERION = "The summary names the missing information."


@pytest.fixture
def judge_prompt(write_prompt):
    return write_prompt("judge.v1", "You grade one criterion and return JSON.")


@pytest.fixture
def under_test(write_prompt):
    return write_prompt("triage.v1", "You triage tickets.")


@pytest.fixture
def task(write_task):
    return write_task(
        "t_judged",
        f'''
        input = "it broke again"

        [[assertions]]
        id = "specifics"
        type = "judge"
        criterion = "{CRITERION}"
        ''',
    )


def verdict_client(verdict="pass", *, confidence=0.9, reasoning="because", cycle=None):
    """A judge client that answers with a structured verdict and counts calls."""
    verdicts = list(cycle) if cycle else None
    state = {"i": 0}

    def respond(request: ModelRequest) -> ModelOutput:
        if verdicts is not None:
            value = verdicts[state["i"] % len(verdicts)]
            state["i"] += 1
        else:
            value = verdict
        payload = {"verdict": value, "confidence": confidence, "reasoning": reasoning}
        return ModelOutput(
            text=json.dumps(payload), stop_reason="end_turn", usage=make_usage(300, 40)
        )

    return ScriptedClient(respond)


def spec(criterion=CRITERION):
    return {"id": "specifics", "type": "judge", "criterion": criterion}


def test_pass_verdict(judge_prompt, under_test, task, make_output):
    judge = Judge(client=verdict_client("pass"), prompt=judge_prompt)
    result = judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    assert result.passed is True
    assert result.type == "judge"
    assert result.meta["judge_hash"] == judge_prompt.hash


def test_fail_verdict_carries_the_reasoning(judge_prompt, under_test, task, make_output):
    judge = Judge(client=verdict_client("fail", reasoning="never says what is missing"),
                  prompt=judge_prompt)
    result = judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    assert result.passed is False
    assert "never says what is missing" in result.detail


def test_judge_asks_for_structured_output_not_prose(judge_prompt, under_test, task, make_output):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt)
    judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    request = client.calls[0]
    assert request.output_format == {"type": "json_schema", "schema": VERDICT_SCHEMA}
    assert request.kind == "judge"
    assert request.system == judge_prompt.system


def test_judge_sees_criterion_task_input_and_candidate_output(
    judge_prompt, under_test, task, make_output
):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt)
    judge.evaluate(
        spec(), task=task, prompt=under_test,
        output=make_output("the candidate answer", tool_calls=(("lookup", {"q": 1}),)),
    )
    content = client.calls[0].messages[0]["content"]
    assert CRITERION in content
    assert "it broke again" in content
    assert "the candidate answer" in content
    assert "lookup" in content


def test_unparseable_verdict_fails_loudly(judge_prompt, under_test, task, make_output):
    client = ScriptedClient(lambda r: ModelOutput(text="looks fine to me", stop_reason="end_turn"))
    judge = Judge(client=client, prompt=judge_prompt)
    result = judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    assert result.passed is False
    assert "unparseable" in result.detail


def test_unknown_verdict_value_fails(judge_prompt, under_test, task, make_output):
    client = ScriptedClient(
        lambda r: ModelOutput(text=json.dumps({"verdict": "maybe", "confidence": 1,
                                               "reasoning": ""}), stop_reason="end_turn")
    )
    judge = Judge(client=client, prompt=judge_prompt)
    result = judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    assert result.passed is False
    assert "unknown verdict" in result.detail


def test_judge_call_failure_fails_the_assertion(judge_prompt, under_test, task, make_output):
    client = ScriptedClient(lambda r: ModelOutput(error="RateLimitError: slow down"))
    judge = Judge(client=client, prompt=judge_prompt)
    result = judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    assert result.passed is False
    assert "RateLimitError" in result.detail


def test_model_error_short_circuits_before_calling_the_judge(
    judge_prompt, under_test, task, make_output
):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt)
    result = judge.evaluate(
        spec(), task=task, prompt=under_test, output=make_output("", error="boom")
    )
    assert result.passed is False
    assert client.call_count == 0, "no point paying a judge to grade a failed call"


# --- caching -------------------------------------------------------------------


def test_cache_hits_on_identical_inputs(judge_prompt, under_test, task, make_output):
    client = verdict_client("pass")
    judge = Judge(client=client, prompt=judge_prompt, cache=JudgeCache(None))
    output = make_output("same output")

    first = judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    second = judge.evaluate(spec(), task=task, prompt=under_test, output=output)

    assert client.call_count == 1, "second evaluation must be served from cache"
    assert first.meta["cached"] is False
    assert second.meta["cached"] is True
    assert second.passed == first.passed
    assert judge.cache.hits == 1 and judge.cache.misses == 1


def test_cached_verdicts_cost_no_tokens(judge_prompt, under_test, task, make_output):
    judge = Judge(client=verdict_client(), prompt=judge_prompt, cache=JudgeCache(None))
    output = make_output("same output")
    first = judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    second = judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    assert first.meta["usage"]["output_tokens"] == 40
    assert second.meta["usage"] == {}, "a cache hit must not inflate run token totals"


def test_cache_misses_when_the_output_changes(judge_prompt, under_test, task, make_output):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt, cache=JudgeCache(None))
    judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("one"))
    judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("two"))
    assert client.call_count == 2


def test_cache_misses_when_tool_calls_differ(judge_prompt, under_test, task, make_output):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt, cache=JudgeCache(None))
    judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("same"))
    judge.evaluate(
        spec(), task=task, prompt=under_test,
        output=make_output("same", tool_calls=(("t", {}),)),
    )
    assert client.call_count == 2


def test_cache_misses_when_the_criterion_changes(judge_prompt, under_test, task, make_output):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt, cache=JudgeCache(None))
    output = make_output("same")
    judge.evaluate(spec("criterion A"), task=task, prompt=under_test, output=output)
    judge.evaluate(spec("criterion B"), task=task, prompt=under_test, output=output)
    assert client.call_count == 2


def test_cache_misses_when_the_prompt_under_test_changes(
    judge_prompt, under_test, write_prompt, task, make_output
):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt, cache=JudgeCache(None))
    output = make_output("same")
    judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    other = write_prompt("triage.v2", "You triage tickets, tersely.")
    judge.evaluate(spec(), task=task, prompt=other, output=output)
    assert client.call_count == 2


def test_cache_misses_when_the_judge_prompt_changes(
    judge_prompt, write_prompt, under_test, task, make_output
):
    """An edited judge must not silently reuse verdicts from the old judge."""
    cache = JudgeCache(None)
    output = make_output("same")
    first_client = verdict_client()
    Judge(client=first_client, prompt=judge_prompt, cache=cache).evaluate(
        spec(), task=task, prompt=under_test, output=output
    )
    edited = write_prompt("judge.v1", "You grade one criterion and return JSON. Be strict.")
    assert edited.hash != judge_prompt.hash
    second_client = verdict_client()
    Judge(client=second_client, prompt=edited, cache=cache).evaluate(
        spec(), task=task, prompt=under_test, output=output
    )
    assert first_client.call_count == 1 and second_client.call_count == 1


def test_cache_key_is_sensitive_to_every_component():
    base = dict(
        judge_hash="j", prompt_hash="p", task_id="t", output_hash="o", criterion="c"
    )
    baseline = cache_key(**base)
    for field, value in base.items():
        assert cache_key(**{**base, field: value + "!"}) != baseline


def test_cache_persists_across_instances(tmp_path, judge_prompt, under_test, task, make_output):
    path = tmp_path / "cache.json"
    output = make_output("same")
    first_client = verdict_client()
    Judge(client=first_client, prompt=judge_prompt, cache=JudgeCache(path)).evaluate(
        spec(), task=task, prompt=under_test, output=output
    )
    assert path.is_file()

    second_client = verdict_client()
    warm = JudgeCache(path)
    result = Judge(client=second_client, prompt=judge_prompt, cache=warm).evaluate(
        spec(), task=task, prompt=under_test, output=output
    )
    assert second_client.call_count == 0
    assert warm.hits == 1
    assert result.meta["cached"] is True


def test_errors_are_not_cached(tmp_path, judge_prompt, under_test, task, make_output):
    client = ScriptedClient(lambda r: ModelOutput(error="transient"))
    cache = JudgeCache(tmp_path / "cache.json")
    judge = Judge(client=client, prompt=judge_prompt, cache=cache)
    output = make_output("same")
    judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    assert client.call_count == 2
    assert len(cache) == 0


def test_no_cache_means_every_evaluation_calls_the_judge(
    judge_prompt, under_test, task, make_output
):
    client = verdict_client()
    judge = Judge(client=client, prompt=judge_prompt, cache=None)
    output = make_output("same")
    judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    assert client.call_count == 2


# --- repeated sampling ---------------------------------------------------------


def test_majority_vote_over_samples(judge_prompt, under_test, task, make_output):
    client = verdict_client(cycle=["pass", "fail", "pass"])
    judge = Judge(client=client, prompt=judge_prompt, samples=3)
    result = judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    assert client.call_count == 3
    assert result.passed is True
    assert result.meta["samples"] == ["pass", "fail", "pass"]


def test_majority_vote_can_fail(judge_prompt, under_test, task, make_output):
    client = verdict_client(cycle=["fail", "fail", "pass"])
    judge = Judge(client=client, prompt=judge_prompt, samples=3)
    result = judge.evaluate(spec(), task=task, prompt=under_test, output=make_output("x"))
    assert result.passed is False


def test_sampled_verdicts_are_cached_as_a_unit(judge_prompt, under_test, task, make_output):
    client = verdict_client(cycle=["pass", "pass", "fail"])
    judge = Judge(client=client, prompt=judge_prompt, samples=3, cache=JudgeCache(None))
    output = make_output("x")
    judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    judge.evaluate(spec(), task=task, prompt=under_test, output=output)
    assert client.call_count == 3, "the whole ballot is cached, not each sample"


def test_samples_must_be_positive(judge_prompt):
    with pytest.raises(ValueError):
        Judge(client=verdict_client(), prompt=judge_prompt, samples=0)
