"""Assertions, in two tiers.

Tier 1 (structural) is the load-bearing tier: deterministic, offline, and fast
enough that the whole task set evaluates in microseconds once outputs exist.
Every assertion type here is exact -- it either matched the bytes or it did not.

Tier 2 (`judge`) is a single assertion type that delegates to an LLM judge; see
`judge.py`. It is evaluated by the runner, not here, because it needs a client.

An assertion spec is a plain dict, loaded from a task file:

    {"type": "contains", "text": "escalate", "case_sensitive": false}

Specs are validated at task-load time, so a typo in an assertion type fails the
run immediately instead of silently never firing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import jsonschema
from .model import ModelOutput

JUDGE_TYPE = "judge"

STRUCTURAL_TYPES = frozenset(
    {
        "contains",
        "not_contains",
        "regex",
        "json_valid",
        "json_schema",
        "length",
        "tool_called",
        "no_tool_called",
        "stop_reason",
    }
)
ALL_TYPES = STRUCTURAL_TYPES | {JUDGE_TYPE}

_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(?P<body>.*?)\n?```$", re.DOTALL)


class AssertionSpecError(Exception):
    """The assertion spec is malformed. Raised at load time."""


@dataclass(frozen=True)
class AssertionResult:
    id: str
    type: str
    passed: bool
    detail: str = ""
    skipped: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = {
            "id": self.id,
            "type": self.type,
            "passed": self.passed,
            "detail": self.detail,
        }
        if self.skipped:
            record["skipped"] = True
        if self.meta:
            record["meta"] = dict(self.meta)
        return record

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssertionResult":
        return cls(
            id=data["id"],
            type=data["type"],
            passed=bool(data["passed"]),
            detail=data.get("detail", ""),
            skipped=bool(data.get("skipped", False)),
            meta=data.get("meta") or {},
        )


def spec_id(spec: Mapping[str, Any], index: int) -> str:
    return str(spec.get("id") or f"a{index}:{spec.get('type', '?')}")


def validate_spec(spec: Mapping[str, Any], index: int = 0, *, where: str = "") -> None:
    prefix = f"{where}: " if where else ""
    kind = spec.get("type")
    if kind not in ALL_TYPES:
        raise AssertionSpecError(
            f"{prefix}assertion {index}: unknown type {kind!r}; "
            f"known types are {sorted(ALL_TYPES)}"
        )
    required: dict[str, tuple[str, ...]] = {
        "contains": ("text",),
        "not_contains": ("text",),
        "regex": ("pattern",),
        "stop_reason": ("equals",),
        JUDGE_TYPE: ("criterion",),
    }
    for key in required.get(kind, ()):
        if key not in spec:
            raise AssertionSpecError(f"{prefix}assertion {index} ({kind}): missing {key!r}")
    if kind == "regex":
        try:
            _compile(spec)
        except re.error as exc:
            raise AssertionSpecError(f"{prefix}assertion {index}: bad regex: {exc}") from exc
    if kind == "json_schema":
        schema = _schema_of(spec, prefix, index)
        try:
            jsonschema.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise AssertionSpecError(f"{prefix}assertion {index}: {exc}") from exc
    if kind == "length" and not any(
        k in spec for k in ("min_chars", "max_chars", "min_words", "max_words")
    ):
        raise AssertionSpecError(
            f"{prefix}assertion {index} (length): needs at least one of "
            "min_chars/max_chars/min_words/max_words"
        )


def evaluate(spec: Mapping[str, Any], output: ModelOutput, index: int = 0) -> AssertionResult:
    """Evaluate one structural assertion against a model output."""
    kind = str(spec.get("type"))
    ident = spec_id(spec, index)
    if kind == JUDGE_TYPE:
        raise AssertionSpecError("judge assertions are evaluated by the runner, not evaluate()")
    if kind not in STRUCTURAL_TYPES:
        return AssertionResult(ident, kind, False, f"unknown assertion type {kind!r}")
    if output.error:
        return AssertionResult(ident, kind, False, f"model error: {output.error}")

    handler = _HANDLERS[kind]
    passed, detail = handler(spec, output)
    return AssertionResult(ident, kind, passed, detail)


def evaluate_all(
    specs: tuple[Mapping[str, Any], ...], output: ModelOutput
) -> list[AssertionResult]:
    return [evaluate(spec, output, i) for i, spec in enumerate(specs)]


# --- individual assertion types ------------------------------------------------


def _contains(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    needle = str(spec["text"])
    haystack = output.text
    if not spec.get("case_sensitive", False):
        needle, haystack = needle.lower(), haystack.lower()
    found = needle in haystack
    return found, "" if found else f"output does not contain {spec['text']!r}"


def _not_contains(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    found, _ = _contains(spec, output)
    return (not found), "" if not found else f"output contains forbidden {spec['text']!r}"


def _regex(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    pattern = _compile(spec)
    matched = pattern.search(output.text) is not None
    negate = bool(spec.get("negate", False))
    passed = matched != negate
    if passed:
        return True, ""
    verb = "unexpectedly matched" if negate else "did not match"
    return False, f"output {verb} /{spec['pattern']}/"


def _json_valid(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    value, error = extract_json(output.text, allow_code_fence=spec.get("allow_code_fence", True))
    if error:
        return False, error
    _ = value
    return True, ""


def _json_schema(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    value, error = extract_json(output.text, allow_code_fence=spec.get("allow_code_fence", True))
    if error:
        return False, error
    schema = _schema_of(spec, "", 0)
    errors = jsonschema.validate(value, schema)
    if errors:
        return False, "; ".join(errors[:5])
    return True, ""


def _length(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    text = output.text
    chars, words = len(text), len(text.split())
    checks = (
        ("min_chars", chars, lambda v, b: v >= b, "chars {v} < min {b}"),
        ("max_chars", chars, lambda v, b: v <= b, "chars {v} > max {b}"),
        ("min_words", words, lambda v, b: v >= b, "words {v} < min {b}"),
        ("max_words", words, lambda v, b: v <= b, "words {v} > max {b}"),
    )
    problems = [
        msg.format(v=value, b=spec[key])
        for key, value, ok, msg in checks
        if key in spec and not ok(value, int(spec[key]))
    ]
    return (not problems), "; ".join(problems)


def _tool_called(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    name = spec.get("name")
    calls = [c for c in output.tool_calls if name is None or c.name == name]
    label = name or "<any tool>"
    if "count" in spec:
        want = int(spec["count"])
        ok = len(calls) == want
        return ok, "" if ok else f"{label} called {len(calls)} time(s), expected {want}"
    minimum = int(spec.get("min_count", 1))
    ok = len(calls) >= minimum
    if ok:
        return True, ""
    seen = ", ".join(c.name for c in output.tool_calls) or "none"
    return False, f"{label} called {len(calls)} time(s), expected >= {minimum} (called: {seen})"


def _no_tool_called(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    name = spec.get("name")
    calls = [c for c in output.tool_calls if name is None or c.name == name]
    if not calls:
        return True, ""
    return False, f"unexpected tool call(s): {', '.join(c.name for c in calls)}"


def _stop_reason(spec: Mapping[str, Any], output: ModelOutput) -> tuple[bool, str]:
    want = spec["equals"]
    got = output.stop_reason
    ok = got == want
    return ok, "" if ok else f"stop_reason is {got!r}, expected {want!r}"


_HANDLERS = {
    "contains": _contains,
    "not_contains": _not_contains,
    "regex": _regex,
    "json_valid": _json_valid,
    "json_schema": _json_schema,
    "length": _length,
    "tool_called": _tool_called,
    "no_tool_called": _no_tool_called,
    "stop_reason": _stop_reason,
}


# --- helpers -------------------------------------------------------------------


def extract_json(text: str, *, allow_code_fence: bool = True) -> tuple[Any, str]:
    """Parse `text` as JSON, optionally unwrapping one surrounding code fence.

    Returns `(value, "")` or `(None, error_message)`. Fence unwrapping is on by
    default because models fence JSON constantly; turn it off per assertion with
    `allow_code_fence = false` when the prompt promises bare JSON.
    """
    candidate = text.strip()
    if not candidate:
        return None, "output is empty, expected JSON"
    if allow_code_fence:
        match = _FENCE.match(candidate)
        if match:
            candidate = match.group("body").strip()
    try:
        return json.loads(candidate), ""
    except json.JSONDecodeError as exc:
        return None, f"output is not valid JSON ({exc.msg} at line {exc.lineno})"


def _compile(spec: Mapping[str, Any]) -> re.Pattern[str]:
    flags = 0
    for char in str(spec.get("flags", "")):
        if char not in _FLAGS:
            raise re.error(f"unknown regex flag {char!r}")
        flags |= _FLAGS[char]
    return re.compile(str(spec["pattern"]), flags)


def _schema_of(spec: Mapping[str, Any], prefix: str, index: int) -> Any:
    if "schema" in spec:
        return spec["schema"]
    if "schema_json" in spec:
        try:
            return json.loads(spec["schema_json"])
        except json.JSONDecodeError as exc:
            raise AssertionSpecError(
                f"{prefix}assertion {index}: schema_json is not valid JSON: {exc}"
            ) from exc
    raise AssertionSpecError(
        f"{prefix}assertion {index} (json_schema): needs `schema` or `schema_json`"
    )
