"""The CLI, plus the guard that keeps live calls out of an offline environment."""

from __future__ import annotations

import json

import pytest

from common.client import api_is_usable, has_credentials
from harness import cli
from harness.model import AnthropicClient, ModelRequest
from harness.prompts import default_prompt_dir
from harness.runner import load_run
from harness.tasks import default_task_dir

FIXTURE = str(default_prompt_dir().parent / "demo_responses.json")


def test_list_prompts(capsys):
    assert cli.main(["list", "prompts"]) == 0
    out = capsys.readouterr().out
    assert "triage.v1" in out and "triage.v2" in out and "judge.rubric.v1" in out


def test_list_tasks(capsys):
    assert cli.main(["list", "tasks"]) == 0
    out = capsys.readouterr().out
    assert "t01_plain_bug" in out
    assert "judge" in out, "the tier-2 task should be visible in the listing"


def test_run_offline_against_the_fixture(tmp_path, capsys):
    out = tmp_path / "v1.jsonl"
    code = cli.main(
        [
            "run",
            "--prompt", "triage.v1",
            "--out", str(out),
            "--fake-responses", FIXTURE,
            "--judge-cache", str(tmp_path / "cache.json"),
            "--concurrency", "2",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "triage.v1@" in printed
    record = load_run(out)
    assert len(record.results) == 7
    assert record.meta["judge_prompt_id"] == "judge.rubric.v1"
    assert record.passed_count == 5


def test_run_rejects_an_unknown_prompt(tmp_path, capsys):
    code = cli.main(
        ["run", "--prompt", "nope", "--out", str(tmp_path / "x.jsonl"),
         "--fake-responses", FIXTURE]
    )
    assert code == 2
    assert "unknown prompt" in capsys.readouterr().err


def test_run_filters_by_task_id(tmp_path):
    out = tmp_path / "one.jsonl"
    code = cli.main(
        ["run", "--prompt", "triage.v1", "--out", str(out), "--fake-responses", FIXTURE,
         "--task", "t01_plain_bug", "--judge-cache", str(tmp_path / "c.json")]
    )
    assert code == 0
    assert set(load_run(out).results) == {"t01_plain_bug"}


def test_run_filters_by_tag(tmp_path):
    out = tmp_path / "tools.jsonl"
    assert cli.main(
        ["run", "--prompt", "triage.v1", "--out", str(out), "--fake-responses", FIXTURE,
         "--tag", "tools", "--judge-cache", str(tmp_path / "c.json")]
    ) == 0
    assert set(load_run(out).results) == {"t04_account_lookup"}


def test_run_with_empty_filter_is_an_error(tmp_path, capsys):
    code = cli.main(
        ["run", "--prompt", "triage.v1", "--out", str(tmp_path / "x.jsonl"),
         "--fake-responses", FIXTURE, "--tag", "nonexistent"]
    )
    assert code == 2
    assert "matched nothing" in capsys.readouterr().err


def test_run_refuses_to_go_live_without_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "has_credentials", lambda: False)
    code = cli.main(["run", "--prompt", "triage.v1", "--out", str(tmp_path / "x.jsonl")])
    assert code == 2
    assert "no Anthropic credentials" in capsys.readouterr().err


def test_diff_exit_codes(tmp_path, capsys):
    paths = []
    for prompt_id in ("triage.v1", "triage.v2"):
        path = tmp_path / f"{prompt_id}.jsonl"
        cli.main(
            ["run", "--prompt", prompt_id, "--out", str(path), "--fake-responses", FIXTURE,
             "--judge-cache", str(tmp_path / "c.json")]
        )
        paths.append(str(path))
    capsys.readouterr()

    assert cli.main(["diff", *paths]) == 1, "regressions must fail the command"
    text = capsys.readouterr().out
    assert "REGRESSIONS (1)" in text
    assert "t06_ambiguous_request" in text

    assert cli.main(["diff", *paths, "--no-fail-on-regression"]) == 0
    capsys.readouterr()

    assert cli.main(["diff", *paths, "--json"]) in (0, 1)
    payload = json.loads(capsys.readouterr().out)
    assert payload["regressions"][0]["task_id"] == "t06_ambiguous_request"
    assert payload["improvements"][0]["task_id"] == "t01_plain_bug"


def test_diff_of_a_missing_file_is_an_error(tmp_path, capsys):
    code = cli.main(["diff", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")])
    assert code == 2
    assert "no such run file" in capsys.readouterr().err


def test_demo_runs_end_to_end(tmp_path, capsys):
    assert cli.main(["demo", "--out-dir", str(tmp_path / "demo")]) == 0
    out = capsys.readouterr().out
    assert "REGRESSIONS (1)" in out
    assert "improvements (1)" in out
    assert "judge cache: 0 hit(s), 2 miss(es)" in out
    assert "noise floor" in out


def test_demo_is_idempotent_and_warms_the_judge_cache(tmp_path, capsys):
    out_dir = str(tmp_path / "demo")
    cli.main(["demo", "--out-dir", out_dir])
    capsys.readouterr()
    assert cli.main(["demo", "--out-dir", out_dir]) == 0
    second = capsys.readouterr().out
    assert "judge cache: 2 hit(s), 0 miss(es)" in second
    assert "REGRESSIONS (1)" in second


def test_bad_fixture_path_is_an_error(tmp_path, capsys):
    code = cli.main(
        ["run", "--prompt", "triage.v1", "--out", str(tmp_path / "x.jsonl"),
         "--fake-responses", str(tmp_path / "missing.json")]
    )
    assert code == 2
    assert "no such response fixture" in capsys.readouterr().err


# --- live-path guards ----------------------------------------------------------


def test_anthropic_client_refuses_to_build_without_credentials(monkeypatch):
    monkeypatch.setattr("harness.model.has_credentials", lambda: False)
    with pytest.raises(RuntimeError, match="no Anthropic credentials"):
        AnthropicClient().complete(
            ModelRequest(system="s", messages=({"role": "user", "content": "hi"},))
        )


@pytest.mark.skipif(not api_is_usable(), reason="live API not usable (no credentials, or unfunded org)")
def test_live_smoke():
    """The only test that would touch the API. Skipped by default."""
    prompts = __import__("harness").load_prompts(default_prompt_dir())
    tasks = list(__import__("harness").load_tasks(default_task_dir()).values())
    output = AnthropicClient().complete(
        __import__("harness").build_request(prompts["triage.v1"], tasks[0])
    )
    assert output.error is None
    assert output.stop_reason in {"end_turn", "tool_use", "max_tokens"}
