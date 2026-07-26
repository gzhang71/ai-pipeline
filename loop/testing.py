"""Offline doubles for the profiler.

There are no API credentials in this environment, so every test path runs
against these. They are deliberately *behavioural*, not mocks that merely echo
what the caller expects:

``heuristic_token_count`` is a real function of the prompt's content, with a
small deliberate non-additivity (a merge discount that grows with block count),
so the attribution code has to earn its answers rather than read them back.

``FakeAnthropicClient`` replays a script of responses and reports usage totals
that differ slightly from the counter's totals -- exactly as the live API does,
since ``count_tokens`` and billed usage are computed by different code paths.
That difference is what the reconciliation residual exists to surface.

This module is public so sibling subprojects can drive the loop offline too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from common.client import MODEL

from .attribution import normalize_messages, to_plain

# --------------------------------------------------------------------------
# A deterministic, roughly-additive token counter
# --------------------------------------------------------------------------

#: Fixed cost the API adds around any request at all.
FRAMING_TOKENS = 9
#: Per-message role wrapper.
MESSAGE_TOKENS = 4
#: Per-content-block wrapper.
BLOCK_TOKENS = 3
#: Characters per token. Crude, but we are not claiming to be a tokenizer --
#: this is a stand-in whose only job is determinism and rough additivity.
CHARS_PER_TOKEN = 4


def _text_cost(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(to_plain(value), sort_keys=True, default=str)
    return max(1, len(text) // CHARS_PER_TOKEN)


def heuristic_token_count(
    *,
    messages: Sequence[dict[str, Any]],
    system: Any = None,
    tools: Any = None,
    model: str = MODEL,
) -> int:
    """A stand-in for ``count_tokens`` that is deterministic and offline.

    Mildly non-additive on purpose: the ``merge discount`` term depends on the
    total block count, so splitting a prompt and counting the halves does *not*
    reproduce the whole. That is the property the attribution method has to
    cope with, and the property a naive per-segment counter gets wrong.
    """
    total = FRAMING_TOKENS
    blocks = 0
    for message in normalize_messages(messages):
        total += MESSAGE_TOKENS
        for block in message["content"]:
            blocks += 1
            total += BLOCK_TOKENS + _text_cost(block)
    if system is not None:
        total += 2 + _text_cost(system)
    if tools:
        total += 5 + _text_cost(tools)
    # Tokens merge across block boundaries in a real tokenizer; approximate
    # that with a discount that grows with the number of blocks.
    return total - blocks // 7


# --------------------------------------------------------------------------
# Scripted response objects
# --------------------------------------------------------------------------


@dataclass
class FakeBlock:
    type: str
    text: str | None = None
    thinking: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def text_block(text: str) -> FakeBlock:
    return FakeBlock(type="text", text=text)


def thinking_block(text: str) -> FakeBlock:
    return FakeBlock(type="thinking", thinking=text)


def tool_use_block(name: str, tool_input: dict[str, Any], block_id: str) -> FakeBlock:
    return FakeBlock(type="tool_use", id=block_id, name=name, input=tool_input)


@dataclass
class FakeUsage:
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeMessage:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = MODEL
    id: str = "msg_fake"


@dataclass
class FakeCountTokensResult:
    input_tokens: int


class _FakeMessagesResource:
    def __init__(self, client: "FakeAnthropicClient"):
        self._client = client

    def create(self, **kwargs: Any) -> FakeMessage:
        return self._client._create(**kwargs)

    def count_tokens(self, **kwargs: Any) -> FakeCountTokensResult:
        self._client.count_calls += 1
        return FakeCountTokensResult(
            heuristic_token_count(
                messages=kwargs.get("messages", []),
                system=kwargs.get("system"),
                tools=kwargs.get("tools"),
                model=kwargs.get("model", MODEL),
            )
        )


class FakeAnthropicClient:
    """Replays a script of responses through a ``messages.create``-shaped API.

    ``script`` entries are either ``FakeMessage`` objects or callables taking
    the request kwargs and returning one. When the script runs out, the last
    entry repeats -- which is what lets a test prove the ``max_iterations``
    guard actually bounds an otherwise-unbounded loop.

    ``usage_skew`` is added to the reported prompt tokens so the response's
    authoritative usage disagrees slightly with the counter, as it does live.
    """

    def __init__(
        self,
        script: Iterable[FakeMessage | Callable[..., FakeMessage]],
        *,
        usage_skew: int = 3,
        cache_read_fraction: float = 0.0,
    ):
        self.script = list(script)
        if not self.script:
            raise ValueError("script must contain at least one response")
        self.usage_skew = usage_skew
        self.cache_read_fraction = cache_read_fraction
        self.requests: list[dict[str, Any]] = []
        self.count_calls = 0
        self.messages = _FakeMessagesResource(self)

    def _create(self, **kwargs: Any) -> FakeMessage:
        self.requests.append(kwargs)
        index = min(len(self.requests) - 1, len(self.script) - 1)
        entry = self.script[index]
        response = entry(**kwargs) if callable(entry) else entry

        prompt_tokens = heuristic_token_count(
            messages=kwargs.get("messages", []),
            system=kwargs.get("system"),
            tools=kwargs.get("tools"),
            model=kwargs.get("model", MODEL),
        )
        authoritative = max(0, prompt_tokens + self.usage_skew)
        cached = int(authoritative * self.cache_read_fraction)
        usage = FakeUsage(
            input_tokens=authoritative - cached,
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=0,
            output_tokens=max(1, sum(len(str(b.to_dict())) for b in response.content) // 4),
        )
        return FakeMessage(
            content=list(response.content),
            stop_reason=response.stop_reason,
            usage=usage,
            model=kwargs.get("model", MODEL),
        )


def echo_executor(name: str, tool_input: dict[str, Any], tool_use_id: str) -> str:
    """A trivial executor: echoes its input back. Use for loop-shape tests."""
    return json.dumps({"tool": name, "input": tool_input})
