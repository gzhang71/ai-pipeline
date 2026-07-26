"""Deterministic serialization + a byte-level model of the rendered prefix.

Caching keys off the exact bytes of the rendered request. Two structurally
identical prompts that serialize differently (unsorted dict keys, a set
iterated in hash order, a tool list in arrival order) are two different cache
entries. Everything in this module exists to make "structurally identical"
and "byte identical" the same thing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "canonical_json",
    "canonical_bytes",
    "stable_tool_order",
    "Span",
    "RenderedPrefix",
    "render_prefix",
]


def canonical_json(obj: Any) -> str:
    """Serialize `obj` so that equal values always produce equal bytes.

    Sorted keys, no insignificant whitespace, no ASCII escaping. Use this
    anywhere a structure is interpolated into a prompt -- a bare
    `json.dumps(d)` re-orders keys across Python versions and across dicts
    built in a different order, which silently forks the cache.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def stable_tool_order(tools: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort tool definitions by name.

    Tools render at position 0 of the request. A tool list assembled from a
    dict, a set, or a plugin registry can come out in a different order on the
    next process, which invalidates the entire prompt -- system and messages
    included -- with no error and no obvious cause.
    """
    return [dict(tool) for tool in sorted(tools, key=lambda t: str(t.get("name", "")))]


@dataclass(frozen=True)
class Span:
    """A byte range of the rendered prefix, attributed to one block."""

    start: int
    end: int
    section: str
    label: str

    def __contains__(self, offset: int) -> bool:
        return self.start <= offset < self.end


@dataclass(frozen=True)
class RenderedPrefix:
    """The rendered request as bytes, with each byte attributable to a block."""

    data: bytes
    spans: tuple[Span, ...]

    def __len__(self) -> int:
        return len(self.data)

    def locate(self, offset: int) -> Span | None:
        """Return the span containing `offset`, or None if past the end."""
        for span in self.spans:
            if offset in span:
                return span
        return self.spans[-1] if self.spans and offset >= len(self.data) else None


def render_prefix(request: Mapping[str, Any]) -> RenderedPrefix:
    """Render `messages.create` kwargs into a canonical, span-annotated prefix.

    This is a faithful *proxy* for what the API hashes, not the API's own
    serializer: the section order (tools -> system -> messages) and the
    per-block boundaries match, the exact framing bytes do not. That is enough
    to answer the only question the linter asks -- "did the bytes before offset
    N change, and which block owns N" -- while staying honest that it cannot
    prove a cache hit.
    """
    chunks: list[bytes] = []
    spans: list[Span] = []
    cursor = 0

    def emit(payload: bytes, section: str, label: str) -> None:
        nonlocal cursor
        chunks.append(payload)
        spans.append(Span(cursor, cursor + len(payload), section, label))
        cursor += len(payload)

    model = request.get("model")
    if model is not None:
        emit(canonical_bytes({"model": model}) + b"\n", "model", "<model>")

    tools = request.get("tools") or []
    emit(b"\x1etools\x1e", "tools", "<section:tools>")
    for index, tool in enumerate(tools):
        name = str(dict(tool).get("name", index))
        emit(canonical_bytes(tool) + b"\n", "tools", f"tools[{index}]:{name}")

    system = request.get("system")
    emit(b"\x1esystem\x1e", "system", "<section:system>")
    if isinstance(system, str):
        emit(canonical_bytes({"type": "text", "text": system}) + b"\n", "system", "system[0]")
    else:
        for index, block in enumerate(system or []):
            emit(canonical_bytes(block) + b"\n", "system", f"system[{index}]")

    messages: Sequence[Mapping[str, Any]] = request.get("messages") or []
    emit(b"\x1emessages\x1e", "messages", "<section:messages>")
    for m_index, message in enumerate(messages):
        role = str(dict(message).get("role", "?"))
        content = dict(message).get("content")
        emit(
            canonical_bytes({"role": role}) + b"\n",
            "messages",
            f"messages[{m_index}].role={role}",
        )
        if isinstance(content, str):
            emit(
                canonical_bytes({"type": "text", "text": content}) + b"\n",
                "messages",
                f"messages[{m_index}].content[0] ({role})",
            )
        else:
            for c_index, block in enumerate(content or []):
                emit(
                    canonical_bytes(block) + b"\n",
                    "messages",
                    f"messages[{m_index}].content[{c_index}] ({role})",
                )

    return RenderedPrefix(b"".join(chunks), tuple(spans))
