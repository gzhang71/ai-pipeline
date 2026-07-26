"""An instrumented tool-use loop.

The loop itself is the ordinary request -> tool_use -> execute -> tool_result
cycle. What makes it a profiler is that before every request it decomposes the
prompt it is about to send (see ``loop.attribution``) and reconciles that
decomposition against the authoritative ``usage`` on the response.

Tools and the tool executor are supplied by the caller. Nothing here knows
what task is being run.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from common.client import MODEL, usage_breakdown

from .attribution import (
    Attribution,
    CachingTokenCounter,
    TokenCounter,
    api_token_counter,
    attribute,
    reconcile,
    to_plain,
)
from .schema import SCHEMA_ID, SCHEMA_VERSION

LIB_VERSION = "1.0.0"

#: ``stop_reason`` values from the API that end the loop rather than continue it.
TERMINAL_STOP_REASONS = ("end_turn", "max_tokens", "stop_sequence", "refusal")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class ToolResult:
    """What a tool executor may return instead of a bare string."""

    content: Any
    is_error: bool = False


#: ``executor(name, tool_input, tool_use_id) -> str | dict | list | ToolResult``
Executor = Callable[[str, dict, str], Any]


@dataclass
class LoopConfig:
    """Everything about *how* to run, with no opinion about *what* to run."""

    model: str = MODEL
    max_tokens: int = 8192
    #: Hard bound on request/tool_result round trips. The guard exists because
    #: a model that keeps calling tools will otherwise run until the context
    #: window or your budget dies, whichever is first.
    max_iterations: int = 12
    system: Any = None
    #: ``low`` | ``medium`` | ``high`` | ``xhigh`` | ``max``. Omitted when None
    #: (the API default is ``high``).
    effort: str | None = None
    #: Adaptive thinking is on by default on Opus 5; set explicitly only if you
    #: want ``{"type": "adaptive", "display": "summarized"}`` or to disable it.
    thinking: dict[str, Any] | None = None
    #: Passed through verbatim to ``messages.create``.
    extra_request: dict[str, Any] = field(default_factory=dict)
    #: See ``loop.attribution.attribute``.
    attribution: str = "block_group"
    reconcile_tolerance: float = 0.02
    #: When a tool raises, return the exception text as an error tool_result
    #: rather than aborting the run.
    catch_tool_errors: bool = True


@dataclass
class RunResult:
    run_id: str
    records: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    stop_reason: str
    final_text: str
    turns: int
    error: str | None = None

    @property
    def header(self) -> dict[str, Any]:
        return self.records[0]

    @property
    def footer(self) -> dict[str, Any]:
        return self.records[-1]

    @property
    def turn_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r["record_type"] == "turn"]

    @property
    def peak_prompt_tokens(self) -> int:
        return self.footer["totals"]["peak_prompt_tokens"]


def _text_of(content: Sequence[Any]) -> str:
    parts = []
    for block in to_plain(list(content)):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _tool_use_blocks(content: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        block
        for block in to_plain(list(content))
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _as_tool_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    return ToolResult(content=value)


def _fingerprint(value: Any) -> dict[str, Any]:
    import hashlib
    import json

    if value is None:
        return {"present": False, "sha256": None, "chars": 0}
    text = value if isinstance(value, str) else json.dumps(to_plain(value), sort_keys=True)
    return {
        "present": True,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }


def run_loop(
    *,
    client: Any,
    tools: Sequence[dict[str, Any]] | None,
    executor: Executor,
    messages: Iterable[Any] | None = None,
    prompt: str | None = None,
    config: LoopConfig | None = None,
    sink: Any = None,
    counter: TokenCounter | None = None,
    run_id: str | None = None,
    task: dict[str, Any] | None = None,
) -> RunResult:
    """Run an instrumented tool-use loop to completion.

    ``client`` needs only ``client.messages.create(**kwargs)``; anything with
    that shape works, which is how the offline tests drive it.

    ``sink`` is anything with ``.write(record)`` -- typically a
    ``loop.sink.JsonlSink``. Records are also returned on the result.
    """
    config = config or LoopConfig()
    tools = list(tools or [])
    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

    if messages is None:
        if prompt is None:
            raise ValueError("pass either `messages` or `prompt`")
        messages = [{"role": "user", "content": prompt}]
    working: list[Any] = list(messages)
    if not working:
        raise ValueError("the message list must not be empty")

    # One cache for the whole run: turn N+1's prompt prefixes are turn N's.
    cache = counter if isinstance(counter, CachingTokenCounter) else CachingTokenCounter(
        counter or api_token_counter
    )

    records: list[dict[str, Any]] = []

    def emit(record: dict[str, Any]) -> None:
        records.append(record)
        if sink is not None:
            sink.write(record)

    header = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_header",
        "run_id": run_id,
        "started_at": _now(),
        "lib_version": LIB_VERSION,
        "model": config.model,
        "max_iterations": config.max_iterations,
        "max_tokens": config.max_tokens,
        "effort": config.effort,
        "thinking": config.thinking,
        "tool_names": [t.get("name", t.get("type", "?")) for t in tools],
        "system_fingerprint": _fingerprint(config.system),
        "attribution": {
            "method": "incremental_prefix_delta",
            "granularity": config.attribution,
            "counter": getattr(counter, "__name__", type(cache.inner).__name__)
            if counter is not None
            else "api_token_counter",
            "measurement_order": list(
                ("framing", "messages", "tool_schemas", "system_prompt")
            ),
            "approximate_segments": ["framing"],
            "reconcile_tolerance_fraction": config.reconcile_tolerance,
        },
        "task": task,
    }
    emit(header)

    stop_reason = "end_turn"
    error: str | None = None
    final_text = ""
    turn_index = 0
    by_kind_total: dict[str, int] = {}
    totals = {
        "prompt_tokens_total": 0,
        "output_tokens_total": 0,
        "cache_read_total": 0,
        "cache_creation_total": 0,
        "peak_prompt_tokens": 0,
    }

    try:
        while True:
            if turn_index >= config.max_iterations:
                stop_reason = "max_iterations"
                break

            started_at = _now()
            clock = time.perf_counter()

            attribution: Attribution = attribute(
                working,
                system=config.system,
                tools=tools or None,
                model=config.model,
                counter=cache,
                granularity=config.attribution,
            )

            request: dict[str, Any] = {
                "model": config.model,
                "max_tokens": config.max_tokens,
                "messages": [to_plain(m) for m in working],
            }
            if config.system is not None:
                request["system"] = config.system
            if tools:
                request["tools"] = tools
            if config.thinking is not None:
                request["thinking"] = config.thinking
            if config.effort is not None:
                request["output_config"] = {"effort": config.effort}
            request.update(config.extra_request)

            response = client.messages.create(**request)
            duration_ms = (time.perf_counter() - clock) * 1000.0

            usage = usage_breakdown(getattr(response, "usage", None))
            content = list(getattr(response, "content", []) or [])
            api_stop = getattr(response, "stop_reason", None)
            calls = _tool_use_blocks(content)

            # Preserve thinking / tool_use blocks verbatim -- editing or
            # dropping them breaks the next turn.
            working.append({"role": "assistant", "content": content})

            tool_records: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            if api_stop == "tool_use" or calls:
                for call in calls:
                    tool_clock = time.perf_counter()
                    is_error = False
                    try:
                        raw = executor(call["name"], call.get("input", {}) or {}, call["id"])
                        result = _as_tool_result(raw)
                    except Exception as exc:  # noqa: BLE001 - surfaced to the model
                        if not config.catch_tool_errors:
                            raise
                        result = ToolResult(content=f"{type(exc).__name__}: {exc}", is_error=True)
                    is_error = result.is_error
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": result.content,
                    }
                    if is_error:
                        block["is_error"] = True
                    tool_results.append(block)
                    payload = result.content
                    tool_records.append(
                        {
                            "tool_use_id": call["id"],
                            "name": call["name"],
                            "input_chars": len(str(call.get("input", ""))),
                            "result_chars": len(payload if isinstance(payload, str) else str(payload)),
                            "is_error": is_error,
                            "duration_ms": round((time.perf_counter() - tool_clock) * 1000.0, 3),
                        }
                    )

            turn_record = {
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "record_type": "turn",
                "run_id": run_id,
                "turn_index": turn_index,
                "started_at": started_at,
                "ended_at": _now(),
                "duration_ms": round(duration_ms, 3),
                "request": {
                    "model": config.model,
                    "n_messages": len(working) - 1,
                    "n_tools": len(tools),
                    "max_tokens": config.max_tokens,
                    "effort": config.effort or "",
                },
                "prompt_tokens": attribution.to_record(),
                "usage": usage,
                "reconciliation": reconcile(
                    attribution, usage, tolerance_fraction=config.reconcile_tolerance
                ),
                "response": {
                    "stop_reason": api_stop,
                    "n_tool_use": len(calls),
                    "text_chars": len(_text_of(content)),
                    "model": getattr(response, "model", config.model),
                },
                "tool_calls": tool_records,
            }
            emit(turn_record)

            totals["prompt_tokens_total"] += attribution.counted_total
            totals["output_tokens_total"] += usage["output_tokens"]
            totals["cache_read_total"] += usage["cache_read_input_tokens"]
            totals["cache_creation_total"] += usage["cache_creation_input_tokens"]
            totals["peak_prompt_tokens"] = max(
                totals["peak_prompt_tokens"], attribution.counted_total
            )
            for kind, value in attribution.by_kind().items():
                by_kind_total[kind] = by_kind_total.get(kind, 0) + value

            turn_index += 1
            text = _text_of(content)
            if text:
                final_text = text

            if api_stop == "pause_turn":
                # A server-side tool hit its iteration cap. Re-send as-is; the
                # server resumes. Do not inject a "continue" message.
                continue
            if tool_results:
                working.append({"role": "user", "content": tool_results})
                continue
            if api_stop in TERMINAL_STOP_REASONS or api_stop is None:
                stop_reason = api_stop if api_stop in TERMINAL_STOP_REASONS else "end_turn"
                break
            # Unknown stop_reason with no tool calls: stop rather than spin.
            stop_reason = "end_turn"
            break
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised to caller
        stop_reason = "error"
        error = f"{type(exc).__name__}: {exc}"

    footer = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_footer",
        "run_id": run_id,
        "ended_at": _now(),
        "turns": turn_index,
        "stop_reason": stop_reason,
        "final_text": final_text,
        "totals": {**totals, "by_kind_total": by_kind_total},
        "counter_calls_total": cache.calls,
        "counter_lookups_total": cache.lookups,
        "error": error,
    }
    emit(footer)

    return RunResult(
        run_id=run_id,
        records=records,
        messages=working,
        stop_reason=stop_reason,
        final_text=final_text,
        turns=turn_index,
        error=error,
    )
