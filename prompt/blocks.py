"""The block model: prompt content tagged with a stability level.

Every piece of a prompt -- a tool definition, a system paragraph, a message
content block -- is one `Block` carrying a `Stability`. Stability is the only
thing that determines where cache breakpoints can legally go, because caching
is a prefix match: a block may only be cached if *everything before it* is at
least as stable as it is.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Union

from .errors import PromptStructureError

__all__ = [
    "Stability",
    "ToolBlock",
    "SystemBlock",
    "MessageBlock",
    "Block",
    "SECTION_ORDER",
]

# The API renders tools, then system, then messages. Mirrors
# common.client.RENDER_ORDER; re-stated here so the block model is readable on
# its own.
SECTION_ORDER = ("tools", "system", "messages")


class Stability(IntEnum):
    """How often a block's rendered bytes change.

    Ordered from most stable to least. The numeric order is load-bearing:
    validation requires the rendered sequence to be non-decreasing in
    stability.
    """

    STATIC = 0
    """Never changes for the lifetime of the deployment (tool schemas, the
    frozen system prompt). Safe to sit at byte 0."""

    SESSION = 1
    """Changes per session, user, or document -- but not within a session
    (retrieved corpus, user profile, conversation history already sent)."""

    TURN = 2
    """Changes on every request (the new user question, a timestamp, a
    per-request ID). Nothing after it can ever be cached."""

    @property
    def display(self) -> str:
        return self.name


def _validate_stability(value: Any) -> Stability:
    try:
        return Stability(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise PromptStructureError(f"unknown stability level: {value!r}") from exc


@dataclass(frozen=True)
class ToolBlock:
    """One entry in the `tools` array.

    Tools render at byte 0. A tool set that varies per user or per turn is the
    single most destructive cache invalidator there is -- it moves every
    subsequent byte in the request.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    stability: Stability = Stability.STATIC
    label: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: bool = False

    section = "tools"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stability", _validate_stability(self.stability))
        if not self.name:
            raise PromptStructureError("ToolBlock requires a name")

    @property
    def display_label(self) -> str:
        return self.label or f"tool:{self.name}"

    def render(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": copy.deepcopy(dict(self.input_schema)),
        }
        payload.update(copy.deepcopy(dict(self.extra)))
        _reject_manual_cache_control(payload, self.display_label)
        return payload


@dataclass(frozen=True)
class SystemBlock:
    """One text block in the top-level `system` array."""

    text: str
    stability: Stability = Stability.STATIC
    label: str | None = None
    checkpoint: bool = False

    section = "system"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stability", _validate_stability(self.stability))

    @property
    def display_label(self) -> str:
        return self.label or f"system:{_preview(self.text)}"

    def render(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class MessageBlock:
    """One *content block* inside a message.

    Consecutive blocks that share a role are merged into a single message on
    render, which is what lets a breakpoint land on an individual content block
    rather than on a whole turn.
    """

    role: str
    content: Union[str, Mapping[str, Any]]
    stability: Stability = Stability.TURN
    label: str | None = None
    checkpoint: bool = False
    """Force a breakpoint candidate here even without a stability change.

    The rolling multi-turn pattern: mark the last block of each completed turn,
    so each request can read the longest prefix that still matches. Candidates
    still compete for the four available breakpoints."""

    section = "messages"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stability", _validate_stability(self.stability))
        if self.role not in ("user", "assistant", "system"):
            raise PromptStructureError(
                f"unsupported message role {self.role!r}; expected "
                "'user', 'assistant', or 'system'"
            )

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        if isinstance(self.content, str):
            return f"{self.role}:{_preview(self.content)}"
        kind = dict(self.content).get("type", "block")
        return f"{self.role}:{kind}"

    def render(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"type": "text", "text": self.content}
        payload = copy.deepcopy(dict(self.content))
        _reject_manual_cache_control(payload, self.display_label)
        return payload


Block = Union[ToolBlock, SystemBlock, MessageBlock]


def _preview(text: str, limit: int = 28) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _reject_manual_cache_control(payload: Mapping[str, Any], label: str) -> None:
    if "cache_control" in payload:
        raise PromptStructureError(
            f"block {label!r} sets cache_control by hand. Breakpoint placement "
            "is owned by PromptAssembler so the 4-breakpoint limit and the "
            "minimum-prefix rule can be enforced; drop the key and let the "
            "assembler place it."
        )
