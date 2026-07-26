"""Offline clients backed by a JSON fixture.

This is what makes the harness demonstrable and testable with no credentials.
Only the network call is faked: the runner, every assertion, the judge prompt,
the judge's JSON parsing, the judge cache, the run record and the diff all run
the same code they run against the live API.

Fixture shape (see `data/demo_responses.json`):

    {
      "responses":   {"<prompt_id>": {"<task_id>": <ModelOutput dict>}},
      "judge":       {"<task_id>": [{"if_output_contains": "...", "verdict": "pass",
                                     "confidence": 0.9, "reasoning": "..."}]},
      "judge_usage": {"input_tokens": 890, "output_tokens": 64, ...}
    }

Judge rules are matched in order against the rendered judge input and the first
match wins; a rule with no `if_output_contains` is the fallback. Making the fake
judge decide from the candidate output (rather than from the task id alone)
keeps it honest: swap in a different candidate output and the verdict moves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .model import ModelOutput, ModelRequest, ScriptedClient, make_usage

DEFAULT_FIXTURE = Path(__file__).parent / "data" / "demo_responses.json"


class FixtureError(Exception):
    pass


def load_fixture(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_FIXTURE
    if not path.is_file():
        raise FixtureError(f"no such response fixture: {path}")
    try:
        data = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureError(f"{path}: not valid JSON: {exc}") from exc
    if "responses" not in data:
        raise FixtureError(f"{path}: fixture needs a top-level `responses` object")
    return data


def task_client(fixture: Mapping[str, Any]) -> ScriptedClient:
    """A client that replays scripted task outputs keyed by (prompt_id, task_id)."""
    responses = fixture.get("responses", {})

    def respond(request: ModelRequest) -> ModelOutput:
        bucket = responses.get(request.prompt_id)
        if bucket is None:
            return ModelOutput(
                error=f"fixture has no responses for prompt {request.prompt_id!r}"
            )
        raw = bucket.get(request.task_id)
        if raw is None:
            return ModelOutput(
                error=f"fixture has no response for task {request.task_id!r} "
                f"under prompt {request.prompt_id!r}"
            )
        return _output_from(raw)

    return ScriptedClient(respond)


def judge_client(fixture: Mapping[str, Any]) -> ScriptedClient:
    """A client that returns structured judge verdicts as JSON text."""
    rules = fixture.get("judge", {})
    usage = fixture.get("judge_usage") or {}

    def respond(request: ModelRequest) -> ModelOutput:
        content = _user_content(request)
        for rule in rules.get(request.task_id, []):
            needle = rule.get("if_output_contains")
            if needle is None or needle in content:
                verdict = {
                    "verdict": rule.get("verdict", "fail"),
                    "confidence": float(rule.get("confidence", 0.5)),
                    "reasoning": rule.get("reasoning", ""),
                }
                return ModelOutput(
                    text=json.dumps(verdict),
                    stop_reason="end_turn",
                    usage=dict(usage) or make_usage(200, 40),
                )
        return ModelOutput(
            error=f"fixture has no judge rule for task {request.task_id!r}"
        )

    return ScriptedClient(respond)


def _output_from(raw: Mapping[str, Any]) -> ModelOutput:
    data = dict(raw)
    data.setdefault("stop_reason", "end_turn")
    if "usage" not in data:
        data["usage"] = make_usage(
            int(data.get("input_tokens", 400)), int(data.get("output_tokens", 60))
        )
    return ModelOutput.from_dict(data)


def _user_content(request: ModelRequest) -> str:
    parts: list[str] = []
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(block.get("text", "")) for block in content)
    return "\n".join(parts)
