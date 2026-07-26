"""Public JSONL schema for context-budget profiler runs.

This module is the contract other subprojects consume. Treat every name here
as a public interface: add fields, never repurpose or silently drop them, and
bump ``SCHEMA_VERSION`` when a consumer would have to change to keep working.

A run is a JSONL file with exactly this record sequence::

    run_header        (exactly one, first)
    turn              (zero or more, turn_index 0..N-1, contiguous, ascending)
    run_footer        (exactly one, last)

Every record carries ``record_type``, ``schema_version``, ``schema_id`` and
``run_id``, so a consumer can demultiplex a concatenation of several runs.

See ``loop/README.md`` for the field-by-field prose description.
"""

from __future__ import annotations

from typing import Any, Iterable

SCHEMA_ID = "ai-pipeline/loop/context-budget"
SCHEMA_VERSION = 1

RECORD_TYPES = ("run_header", "turn", "run_footer")

#: Segment kinds that may appear in ``turn.prompt_tokens.segments[].kind`` and
#: as keys of ``turn.prompt_tokens.by_kind``. Ordered as the API renders them:
#: tools -> system -> messages.
SEGMENT_KINDS = (
    "framing",  # request scaffolding the API adds around any prompt
    "tool_schemas",  # the `tools` array
    "system_prompt",  # the `system` field
    "user_text",  # text blocks in user-role messages
    "assistant_text",  # text blocks in assistant-role messages
    "thinking",  # thinking / redacted_thinking blocks
    "tool_use",  # tool_use / server_tool_use blocks
    "tool_result",  # tool_result blocks (returned to the model)
    "messages_total",  # coarse granularity only: all messages as one segment
    "other",  # anything unrecognized (images, documents, ...)
)

#: Values that may appear in ``run_footer.stop_reason``.
RUN_STOP_REASONS = (
    "end_turn",
    "max_iterations",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "error",
)

#: Values that may appear in ``run_header.attribution.granularity``.
GRANULARITIES = ("block_group", "message", "coarse", "off")


class SchemaError(ValueError):
    """A record does not conform to the documented schema."""


def _require(record: dict, key: str, types: tuple[type, ...] | type) -> Any:
    if key not in record:
        raise SchemaError(f"missing required field {key!r}")
    value = record[key]
    if not isinstance(value, types):
        raise SchemaError(
            f"field {key!r} has type {type(value).__name__}, expected "
            f"{types if isinstance(types, tuple) else types.__name__}"
        )
    return value


def _require_envelope(record: dict) -> str:
    if not isinstance(record, dict):
        raise SchemaError(f"record must be a dict, got {type(record).__name__}")
    schema_id = _require(record, "schema_id", str)
    if schema_id != SCHEMA_ID:
        raise SchemaError(f"unknown schema_id {schema_id!r}")
    version = _require(record, "schema_version", int)
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {version} is not readable by this build "
            f"(expected {SCHEMA_VERSION})"
        )
    record_type = _require(record, "record_type", str)
    if record_type not in RECORD_TYPES:
        raise SchemaError(f"unknown record_type {record_type!r}")
    _require(record, "run_id", str)
    return record_type


def _validate_header(record: dict) -> None:
    _require(record, "started_at", str)
    _require(record, "model", str)
    _require(record, "max_iterations", int)
    _require(record, "tool_names", list)
    attribution = _require(record, "attribution", dict)
    _require(attribution, "method", str)
    granularity = _require(attribution, "granularity", str)
    if granularity not in GRANULARITIES:
        raise SchemaError(f"unknown attribution granularity {granularity!r}")
    _require(attribution, "counter", str)
    _require(attribution, "measurement_order", list)
    _require(attribution, "approximate_segments", list)
    _require(attribution, "reconcile_tolerance_fraction", (int, float))
    if "task" in record and record["task"] is not None:
        _require(record["task"], "task_id", str)


def _validate_turn(record: dict) -> None:
    turn_index = _require(record, "turn_index", int)
    if turn_index < 0:
        raise SchemaError(f"turn_index must be >= 0, got {turn_index}")
    _require(record, "started_at", str)
    _require(record, "ended_at", str)
    _require(record, "duration_ms", (int, float))

    request = _require(record, "request", dict)
    for key in ("model", "n_messages", "n_tools", "max_tokens"):
        _require(request, key, (str, int))

    prompt = _require(record, "prompt_tokens", dict)
    counted_total = _require(prompt, "counted_total", int)
    segments = _require(prompt, "segments", list)
    by_kind = _require(prompt, "by_kind", dict)
    segment_sum = _require(prompt, "segment_sum", int)
    residual = _require(prompt, "decomposition_residual", int)
    _require(prompt, "counter_calls", int)
    _require(prompt, "negative_segments", int)

    seen_ids: set[str] = set()
    running = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise SchemaError("prompt_tokens.segments entries must be dicts")
        segment_id = _require(segment, "segment_id", str)
        if segment_id in seen_ids:
            raise SchemaError(f"duplicate segment_id {segment_id!r}")
        seen_ids.add(segment_id)
        kind = _require(segment, "kind", str)
        if kind not in SEGMENT_KINDS:
            raise SchemaError(f"unknown segment kind {kind!r}")
        running += _require(segment, "tokens", int)
        _require(segment, "approximate", bool)
        if segment.get("message_index") is not None and not isinstance(
            segment["message_index"], int
        ):
            raise SchemaError("segment.message_index must be an int or null")
    if segments and running != segment_sum:
        raise SchemaError(
            f"segment_sum {segment_sum} does not equal the sum of segment "
            f"tokens {running}"
        )
    if segments and segment_sum + residual != counted_total:
        raise SchemaError(
            "segment_sum + decomposition_residual must equal counted_total "
            f"({segment_sum} + {residual} != {counted_total})"
        )
    for kind in by_kind:
        if kind not in SEGMENT_KINDS:
            raise SchemaError(f"unknown by_kind key {kind!r}")

    usage = _require(record, "usage", dict)
    for key in (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
        "total_prompt_tokens",
    ):
        _require(usage, key, int)

    recon = _require(record, "reconciliation", dict)
    _require(recon, "counted_total", int)
    _require(recon, "authoritative_total", int)
    _require(recon, "residual_tokens", int)
    _require(recon, "residual_fraction", (int, float))
    _require(recon, "within_tolerance", bool)
    _require(recon, "tolerance_fraction", (int, float))

    response = _require(record, "response", dict)
    _require(response, "stop_reason", (str, type(None)))
    _require(response, "n_tool_use", int)

    for call in _require(record, "tool_calls", list):
        if not isinstance(call, dict):
            raise SchemaError("tool_calls entries must be dicts")
        _require(call, "tool_use_id", str)
        _require(call, "name", str)
        _require(call, "is_error", bool)


def _validate_footer(record: dict) -> None:
    _require(record, "ended_at", str)
    _require(record, "turns", int)
    stop_reason = _require(record, "stop_reason", str)
    if stop_reason not in RUN_STOP_REASONS:
        raise SchemaError(f"unknown run stop_reason {stop_reason!r}")
    totals = _require(record, "totals", dict)
    for key in (
        "prompt_tokens_total",
        "output_tokens_total",
        "cache_read_total",
        "cache_creation_total",
        "peak_prompt_tokens",
    ):
        _require(totals, key, int)
    _require(totals, "by_kind_total", dict)
    if record.get("error") is not None and not isinstance(record["error"], str):
        raise SchemaError("footer.error must be a string or null")


_VALIDATORS = {
    "run_header": _validate_header,
    "turn": _validate_turn,
    "run_footer": _validate_footer,
}


def validate_record(record: dict) -> dict:
    """Validate one record. Returns it unchanged, raises `SchemaError`."""
    record_type = _require_envelope(record)
    _VALIDATORS[record_type](record)
    return record


def validate_run(records: Iterable[dict]) -> list[dict]:
    """Validate a whole run: record order, turn numbering, run_id agreement.

    Accepts a concatenation of several runs as long as each run is internally
    contiguous and its records are adjacent.
    """
    records = list(records)
    if not records:
        raise SchemaError("a run must contain at least a header and a footer")
    for record in records:
        validate_record(record)

    index = 0
    while index < len(records):
        header = records[index]
        if header["record_type"] != "run_header":
            raise SchemaError(
                f"expected a run_header at position {index}, got "
                f"{header['record_type']!r}"
            )
        run_id = header["run_id"]
        index += 1
        expected_turn = 0
        while index < len(records) and records[index]["record_type"] == "turn":
            turn = records[index]
            if turn["run_id"] != run_id:
                raise SchemaError(
                    f"turn at position {index} belongs to run {turn['run_id']!r}, "
                    f"expected {run_id!r}"
                )
            if turn["turn_index"] != expected_turn:
                raise SchemaError(
                    f"turn_index {turn['turn_index']} is out of order; expected "
                    f"{expected_turn}"
                )
            expected_turn += 1
            index += 1
        if index >= len(records) or records[index]["record_type"] != "run_footer":
            raise SchemaError(f"run {run_id!r} has no run_footer")
        footer = records[index]
        if footer["run_id"] != run_id:
            raise SchemaError(
                f"footer run_id {footer['run_id']!r} does not match header "
                f"{run_id!r}"
            )
        if footer["turns"] != expected_turn:
            raise SchemaError(
                f"footer.turns is {footer['turns']} but {expected_turn} turn "
                "records were seen"
            )
        index += 1
    return records


def group_runs(records: Iterable[dict]) -> list[dict[str, Any]]:
    """Split a validated record stream into ``{header, turns, footer}`` dicts."""
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for record in records:
        kind = record["record_type"]
        if kind == "run_header":
            current = {"header": record, "turns": [], "footer": None}
            runs.append(current)
        elif current is None:
            raise SchemaError(f"{kind} record before any run_header")
        elif kind == "turn":
            current["turns"].append(record)
        else:
            current["footer"] = record
            current = None
    return runs
