"""Offline stand-ins for the Anthropic client, used by the tests.

These are deliberately local to `graph/` rather than borrowed from a sibling
package. The comparison harness reports numbers; a shared double owned by
another subproject could change underneath and silently move those numbers
without a single test in `graph/` failing. The cost is a little duplication.

Neither fake is a mock in the assert-on-calls sense. Both drive the *real*
retrieval code and return answers built from whatever that code actually
retrieved, so a test that passes here is testing the retrieval, not the double:

* `FakeToolUsingClient` runs a fixed but genuine policy -- search, then read the
  top hit, then answer with what it read. If `search_symbols` ranks badly or
  `get_definition` returns the wrong body, the answer is wrong and the test
  fails.
* `FakeOneShotClient` is a *perfect reader*: it answers with the retrieved
  excerpts verbatim. That is the most favourable possible reader for the
  baseline, so any baseline failure is a retrieval failure, never a reading one.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

_IDS = itertools.count(1)


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = field(default_factory=lambda: f"toolu_fake_{next(_IDS)}")
    type: str = "tool_use"

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str
    model: str = "fake-model"
    usage: FakeUsage = field(default_factory=FakeUsage)


class _Messages:
    def __init__(self, owner: "FakeClientBase") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> FakeMessage:
        self._owner.requests.append(kwargs)
        return self._owner.respond(**kwargs)


class FakeClientBase:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(self)

    def respond(self, **kwargs: Any) -> FakeMessage:  # pragma: no cover - abstract
        raise NotImplementedError


def _last_tool_results(messages: list[dict[str, Any]]) -> list[Any]:
    """Decode the JSON payloads the harness fed back as tool_result blocks."""
    out: list[Any] = []
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                try:
                    out.append(json.loads(block["content"]))
                except (json.JSONDecodeError, TypeError, KeyError):
                    out.append(block.get("content"))
        if out:
            break
    return out


class FakeToolUsingClient(FakeClientBase):
    """search_symbols -> get_definition(top hit) -> answer with what came back.

    `extra_reads` fetches that many additional hits from the same search before
    answering, which is how the fixture question that needs two symbols is
    satisfied.
    """

    def __init__(self, extra_reads: int = 0) -> None:
        super().__init__()
        self.extra_reads = extra_reads
        self._fetched: list[dict[str, Any]] = []
        self._pending: list[str] = []

    def respond(self, **kwargs: Any) -> FakeMessage:
        messages = kwargs.get("messages", [])
        results = _last_tool_results(messages)

        if not results:  # first turn: always search
            self._fetched = []
            self._pending = []
            question = messages[0]["content"]
            return FakeMessage(
                content=[
                    FakeToolUseBlock(name="search_symbols", input={"query": question})
                ],
                stop_reason="tool_use",
            )

        for payload in results:
            if isinstance(payload, dict) and "matches" in payload:
                ids = [m["symbol_id"] for m in payload["matches"]]
                self._pending = ids[: 1 + self.extra_reads]
            elif isinstance(payload, dict) and "source" in payload:
                self._fetched.append(payload)

        if self._pending:
            symbol_id = self._pending.pop(0)
            return FakeMessage(
                content=[
                    FakeToolUseBlock(
                        name="get_definition", input={"symbol_id": symbol_id}
                    )
                ],
                stop_reason="tool_use",
            )

        answer = "\n\n".join(
            f"{d['symbol_id']} ({d['path']}:{d['lines']}):\n{d['source']}"
            for d in self._fetched
        )
        return FakeMessage(
            content=[FakeTextBlock(text=answer or "no definitions retrieved")],
            stop_reason="end_turn",
        )


class FakeNeighborClient(FakeClientBase):
    """search_symbols -> get_neighbors(callers) -> answer with the caller list.

    The policy a graph makes possible and chunk retrieval does not: answer a
    relational question from an edge list instead of from every call site's
    source.
    """

    def __init__(self, direction: str = "callers") -> None:
        super().__init__()
        self.direction = direction

    def respond(self, **kwargs: Any) -> FakeMessage:
        messages = kwargs.get("messages", [])
        results = _last_tool_results(messages)

        if not results:
            return FakeMessage(
                content=[
                    FakeToolUseBlock(
                        name="search_symbols",
                        input={"query": messages[0]["content"]},
                    )
                ],
                stop_reason="tool_use",
            )

        for payload in results:
            if isinstance(payload, dict) and "matches" in payload:
                if not payload["matches"]:
                    return FakeMessage(
                        content=[FakeTextBlock(text="no matching symbol")],
                        stop_reason="end_turn",
                    )
                return FakeMessage(
                    content=[
                        FakeToolUseBlock(
                            name="get_neighbors",
                            input={
                                "symbol_id": payload["matches"][0]["symbol_id"],
                                "direction": self.direction,
                            },
                        )
                    ],
                    stop_reason="tool_use",
                )
            if isinstance(payload, dict) and self.direction in payload:
                names = [n["symbol_id"] for n in payload[self.direction]]
                return FakeMessage(
                    content=[
                        FakeTextBlock(
                            text=f"{len(names)} {self.direction}: " + ", ".join(names)
                        )
                    ],
                    stop_reason="end_turn",
                )
        return FakeMessage(
            content=[FakeTextBlock(text="unresolved")], stop_reason="end_turn"
        )


class FakeOneShotClient(FakeClientBase):
    """A perfect reader: echoes the retrieved excerpts back as the answer."""

    def respond(self, **kwargs: Any) -> FakeMessage:
        messages = kwargs.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt)
        return FakeMessage(
            content=[FakeTextBlock(text=prompt)], stop_reason="end_turn"
        )


class FakeRefusingClient(FakeClientBase):
    """Returns stop_reason='refusal' so the harness's guard path is exercised."""

    def respond(self, **kwargs: Any) -> FakeMessage:
        return FakeMessage(content=[], stop_reason="refusal")
