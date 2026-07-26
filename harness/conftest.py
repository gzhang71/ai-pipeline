"""Shared fixtures. Every test in this package runs offline.

The `_no_network` autouse fixture makes that a guarantee rather than a hope: it
severs socket connect for the duration of each test, so a future change that
accidentally reaches for the live client fails loudly instead of hanging or
quietly billing someone.
"""

from __future__ import annotations

import socket
import textwrap
from pathlib import Path
from typing import Any, Mapping

import pytest

from harness.model import ModelOutput, ToolCall, make_usage
from harness.prompts import Prompt, parse_prompt
from harness.tasks import Task, parse_task


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("harness tests must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


@pytest.fixture
def make_output():
    def _make(
        text: str = "",
        *,
        tool_calls: tuple[tuple[str, Mapping[str, Any]], ...] = (),
        stop_reason: str | None = "end_turn",
        input_tokens: int = 100,
        output_tokens: int = 20,
        error: str | None = None,
    ) -> ModelOutput:
        return ModelOutput(
            text=text,
            tool_calls=tuple(ToolCall(name=n, input=dict(i)) for n, i in tool_calls),
            stop_reason=stop_reason,
            usage=make_usage(input_tokens, output_tokens),
            error=error,
        )

    return _make


@pytest.fixture
def write_prompt(tmp_path: Path):
    def _write(prompt_id: str, body: str, *, meta: str = "") -> Prompt:
        directory = tmp_path / "prompts"
        directory.mkdir(exist_ok=True)
        path = directory / f"{prompt_id}.prompt.md"
        content = f"+++\n{meta}\n+++\n{body}" if meta else body
        path.write_text(textwrap.dedent(content), "utf-8")
        return parse_prompt(path.read_bytes(), path=path)

    return _write


@pytest.fixture
def write_task(tmp_path: Path):
    def _write(task_id: str, toml_body: str) -> Task:
        directory = tmp_path / "tasks"
        directory.mkdir(exist_ok=True)
        path = directory / f"{task_id}.toml"
        path.write_text(textwrap.dedent(toml_body), "utf-8")
        return parse_task(path.read_bytes(), path=path)

    return _write


@pytest.fixture
def simple_prompt(write_prompt) -> Prompt:
    return write_prompt("p.v1", "You are a test assistant. Answer briefly.")


@pytest.fixture
def simple_task(write_task) -> Task:
    return write_task(
        "t_simple",
        """
        description = "a trivial task"
        input = "say ok"

        [[assertions]]
        id = "says_ok"
        type = "contains"
        text = "ok"
        """,
    )
