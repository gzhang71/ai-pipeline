"""Model request/response types and the clients that satisfy them.

Everything the harness executes goes through the `ModelClient` protocol, which
is one method wide. That is what lets the entire test suite run offline against
scripted fakes while the production path is a thin, real Anthropic call.

`ModelRequest` carries prompt/task identity alongside the payload. The real
client ignores those fields; fakes key their scripted responses off them. This
is deliberate -- the alternative (a side channel the fake reads) makes the fake
diverge from the interface it is standing in for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from common.client import MODEL, get_client, has_credentials, usage_breakdown

EMPTY_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
    "total_prompt_tokens": 0,
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "input": dict(self.input)}


@dataclass(frozen=True)
class ModelRequest:
    system: str
    messages: tuple[dict[str, Any], ...]
    max_tokens: int = 1024
    tools: tuple[dict[str, Any], ...] = ()
    effort: str | None = None
    output_format: Mapping[str, Any] | None = None
    # Identity, for fakes and for logging. Ignored by the live client.
    prompt_id: str = ""
    prompt_hash: str = ""
    task_id: str = ""
    kind: str = "task"  # "task" | "judge"


@dataclass(frozen=True)
class ModelOutput:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=lambda: dict(EMPTY_USAGE))
    model: str = MODEL
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "stop_reason": self.stop_reason,
            "usage": dict(self.usage),
            "model": self.model,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelOutput":
        return cls(
            text=data.get("text", ""),
            tool_calls=tuple(
                ToolCall(name=c["name"], input=c.get("input", {}))
                for c in data.get("tool_calls", [])
            ),
            stop_reason=data.get("stop_reason"),
            usage=dict(data.get("usage") or EMPTY_USAGE),
            model=data.get("model", MODEL),
            error=data.get("error"),
            duration_ms=int(data.get("duration_ms", 0)),
        )


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelOutput: ...


class AnthropicClient:
    """The live path. Requires credentials; every call hits the API.

    Adaptive thinking is on. It costs tokens, but on Opus 5 an explicitly
    disabled thinking config can emit a tool call as plain text -- the turn
    succeeds, the call never happens, and no error is raised. A harness whose
    whole job is to assert "tool X was called" cannot afford that failure mode.
    """

    def __init__(self, *, model: str = MODEL, client: Any = None) -> None:
        self.model = model
        self._client = client

    def _resolve(self) -> Any:
        if self._client is None:
            if not has_credentials():
                raise RuntimeError(
                    "no Anthropic credentials found; run offline with a fake client "
                    "(`--fake-responses`) or set ANTHROPIC_API_KEY"
                )
            self._client = get_client()
        return self._client

    def complete(self, request: ModelRequest) -> ModelOutput:
        client = self._resolve()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": list(request.messages),
            "thinking": {"type": "adaptive"},
        }
        if request.system:
            kwargs["system"] = request.system
        if request.tools:
            kwargs["tools"] = list(request.tools)
        output_config: dict[str, Any] = {}
        if request.effort:
            output_config["effort"] = request.effort
        if request.output_format is not None:
            output_config["format"] = dict(request.output_format)
        if output_config:
            kwargs["output_config"] = output_config

        started = time.perf_counter()
        try:
            with client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # surfaced as a task error, not a crash
            return ModelOutput(
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
                model=self.model,
            )
        duration_ms = int((time.perf_counter() - started) * 1000)
        return _from_message(message, model=self.model, duration_ms=duration_ms)


def _from_message(message: Any, *, model: str, duration_ms: int) -> ModelOutput:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            texts.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            calls.append(ToolCall(name=block.name, input=dict(block.input or {})))
    return ModelOutput(
        text="".join(texts),
        tool_calls=tuple(calls),
        stop_reason=getattr(message, "stop_reason", None),
        usage=usage_breakdown(getattr(message, "usage", None)),
        model=getattr(message, "model", model),
        duration_ms=duration_ms,
    )


class ScriptedClient:
    """Offline client backed by a lookup function or a nested mapping.

    Used by the tests and by `python -m harness demo`. It counts calls so tests
    can assert on cache behaviour.
    """

    def __init__(
        self,
        script: Callable[[ModelRequest], ModelOutput] | Mapping[str, Mapping[str, Any]],
        *,
        default: ModelOutput | None = None,
    ) -> None:
        self._script = script
        self._default = default
        self.calls: list[ModelRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(self, request: ModelRequest) -> ModelOutput:
        self.calls.append(request)
        if callable(self._script):
            result = self._script(request)
        else:
            bucket = self._script.get(request.prompt_id) or {}
            raw = bucket.get(request.task_id)
            result = ModelOutput.from_dict(raw) if isinstance(raw, Mapping) else raw
        if result is None:
            if self._default is not None:
                return self._default
            return ModelOutput(
                error=f"no scripted response for prompt={request.prompt_id!r} "
                f"task={request.task_id!r}",
                stop_reason=None,
            )
        return result


def make_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> dict[str, int]:
    """Build a usage dict via the shared `usage_breakdown` normalizer.

    Fakes route through the same function the live path uses, so a change in
    how usage is summarized is caught by the tests instead of being mocked away.
    """

    class _Usage:
        pass

    usage = _Usage()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation
    usage.cache_read_input_tokens = cache_read
    return usage_breakdown(usage)


def sum_usage(usages: Sequence[Mapping[str, int]]) -> dict[str, int]:
    total = dict(EMPTY_USAGE)
    for usage in usages:
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total
