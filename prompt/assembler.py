"""Cache-aware prompt assembly.

`PromptAssembler` owns three jobs that are easy to get wrong by hand:

1. It refuses to build a prompt whose rendered order contradicts its stability
   order, naming the block at fault.
2. It places `cache_control` breakpoints at stability boundaries, never more
   than four, preferring the boundaries that protect the most tokens.
3. It emits the exact kwargs dict for `client.messages.create(...)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from common.client import (
    CACHE_LOOKBACK_BLOCKS,
    MAX_CACHE_BREAKPOINTS,
    MIN_CACHEABLE_PREFIX_TOKENS,
    MODEL,
)

from .blocks import Block, MessageBlock, Stability, SystemBlock, ToolBlock
from .errors import PromptOrderingError, PromptStructureError
from .serialize import canonical_json

__all__ = ["PromptAssembler", "CachePlan", "PlacedBreakpoint", "estimate_tokens"]

#: Fallback when the model has no entry in MIN_CACHEABLE_PREFIX_TOKENS. The
#: minimum is not monotonic across model generations, so guessing low would
#: hand out breakpoints that silently never cache.
DEFAULT_MIN_PREFIX_TOKENS = max(MIN_CACHEABLE_PREFIX_TOKENS.values())

_VALID_TTLS = (None, "5m", "1h")


def estimate_tokens(text: str) -> int:
    """Rough offline token estimate (~4 characters per token).

    This is an *estimate*. It exists so breakpoint placement and the
    below-minimum warning work with no network access. For a real count, pass
    `token_counter=` a function backed by `common.client.count_tokens`, which
    calls the count_tokens endpoint.
    """
    return max(1, math.ceil(len(text) / 4))


@dataclass(frozen=True)
class PlacedBreakpoint:
    """One `cache_control` marker the assembler decided to emit."""

    index: int
    """Position in the assembler's flat, render-ordered block list."""

    section: str
    label: str
    stability: Stability
    prefix_tokens: int
    """Estimated tokens this breakpoint makes cacheable (everything up to and
    including its block)."""


@dataclass(frozen=True)
class CachePlan:
    """The outcome of breakpoint placement, including what was skipped."""

    model: str
    minimum_prefix_tokens: int
    total_estimated_tokens: int
    breakpoints: tuple[PlacedBreakpoint, ...] = ()
    warnings: tuple[str, ...] = ()
    candidates_considered: int = 0
    ttl: str | None = None

    def describe(self) -> str:
        lines = [
            f"model={self.model} "
            f"min_cacheable_prefix={self.minimum_prefix_tokens} tokens "
            f"est_prompt={self.total_estimated_tokens} tokens",
            f"breakpoints: {len(self.breakpoints)}/{MAX_CACHE_BREAKPOINTS} "
            f"(from {self.candidates_considered} stability boundaries)",
        ]
        for bp in self.breakpoints:
            ttl = self.ttl or "5m"
            lines.append(
                f"  - {bp.section}: {bp.label} [{bp.stability.display}] "
                f"protects ~{bp.prefix_tokens} tokens (ttl={ttl})"
            )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


@dataclass
class _Entry:
    block: Block
    index: int

    @property
    def section(self) -> str:
        return self.block.section

    @property
    def stability(self) -> Stability:
        return self.block.stability

    @property
    def label(self) -> str:
        return self.block.display_label


class PromptAssembler:
    """Build a cache-correct request out of stability-tagged blocks.

    >>> a = PromptAssembler(model="claude-opus-5")
    >>> _ = a.add_system("You are a careful assistant.", Stability.STATIC)
    >>> _ = a.add_message("user", "What changed in the deploy?")
    >>> kwargs = a.to_request_kwargs(max_tokens=1024)
    >>> kwargs["model"]
    'claude-opus-5'
    """

    def __init__(
        self,
        *,
        model: str = MODEL,
        ttl: str | None = None,
        sort_tools: bool = True,
        max_breakpoints: int = MAX_CACHE_BREAKPOINTS,
        cache_tail: bool = False,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if ttl not in _VALID_TTLS:
            raise PromptStructureError(
                f"ttl must be one of {_VALID_TTLS!r} (5m is the API default and "
                f"is sent by omitting the key); got {ttl!r}"
            )
        if not 1 <= max_breakpoints <= MAX_CACHE_BREAKPOINTS:
            raise PromptStructureError(
                f"max_breakpoints must be between 1 and {MAX_CACHE_BREAKPOINTS} "
                f"(the API's hard limit); got {max_breakpoints}"
            )
        self.model = model
        self.ttl = ttl
        self.sort_tools = sort_tools
        self.max_breakpoints = max_breakpoints
        self.cache_tail = cache_tail
        self._token_counter = token_counter or estimate_tokens
        self._tools: list[ToolBlock] = []
        self._system: list[SystemBlock] = []
        self._messages: list[MessageBlock] = []

    # -- construction ----------------------------------------------------

    def add_tool(
        self,
        name: str,
        description: str,
        input_schema: Mapping[str, Any],
        stability: Stability = Stability.STATIC,
        *,
        label: str | None = None,
        checkpoint: bool = False,
        **extra: Any,
    ) -> "PromptAssembler":
        self._tools.append(
            ToolBlock(
                name=name,
                description=description,
                input_schema=input_schema,
                stability=stability,
                label=label,
                extra=extra,
                checkpoint=checkpoint,
            )
        )
        return self

    def add_system(
        self,
        text: str,
        stability: Stability = Stability.STATIC,
        *,
        label: str | None = None,
        checkpoint: bool = False,
    ) -> "PromptAssembler":
        self._system.append(
            SystemBlock(
                text=text, stability=stability, label=label, checkpoint=checkpoint
            )
        )
        return self

    def add_message(
        self,
        role: str,
        content: Any,
        stability: Stability = Stability.TURN,
        *,
        label: str | None = None,
        checkpoint: bool = False,
    ) -> "PromptAssembler":
        self._messages.append(
            MessageBlock(
                role=role,
                content=content,
                stability=stability,
                label=label,
                checkpoint=checkpoint,
            )
        )
        return self

    def add_block(self, block: Block) -> "PromptAssembler":
        if isinstance(block, ToolBlock):
            self._tools.append(block)
        elif isinstance(block, SystemBlock):
            self._system.append(block)
        elif isinstance(block, MessageBlock):
            self._messages.append(block)
        else:  # pragma: no cover - defensive
            raise PromptStructureError(f"unknown block type: {type(block)!r}")
        return self

    def settle_history(self, stability: Stability = Stability.SESSION) -> "PromptAssembler":
        """Re-tag every existing TURN message block as `stability`.

        Call this at the top of an agentic loop, before appending the new turn.
        Content already sent is, by definition, no longer changing -- leaving it
        tagged TURN would make the new turn's blocks look like an ordering
        violation and would block every breakpoint inside the history.
        """
        self._messages = [
            replace(block, stability=stability)
            if block.stability is Stability.TURN
            else block
            for block in self._messages
        ]
        return self

    # -- inspection ------------------------------------------------------

    @property
    def blocks(self) -> tuple[Block, ...]:
        """All blocks in rendered order: tools, then system, then messages."""
        return tuple(entry.block for entry in self._entries())

    def _entries(self) -> list[_Entry]:
        tools: Sequence[ToolBlock] = self._tools
        if self.sort_tools:
            tools = sorted(self._tools, key=lambda t: t.name)
        ordered: list[Block] = [*tools, *self._system, *self._messages]
        return [_Entry(block, i) for i, block in enumerate(ordered)]

    # -- validation ------------------------------------------------------

    def validate(self) -> "PromptAssembler":
        """Raise if the prompt cannot cache by construction.

        Checks the ordering invariant (stability must be non-decreasing across
        the whole rendered byte stream) and the structural rules the Messages
        API enforces.
        """
        entries = self._entries()
        if not self._messages:
            raise PromptStructureError(
                "a request needs at least one message; add_message('user', ...)"
            )
        if self._messages[0].role != "user":
            raise PromptStructureError(
                f"the first message must have role 'user'; got "
                f"{self._messages[0].role!r}"
            )

        for previous, current in zip(entries, entries[1:]):
            if current.stability < previous.stability:
                raise PromptOrderingError(self._ordering_message(previous, current))
        return self

    def _ordering_message(self, previous: _Entry, current: _Entry) -> str:
        return (
            f"Ordering violation: {current.stability.display} block "
            f"{current.label!r} (position {current.index}, section "
            f"'{current.section}') is more stable than the "
            f"{previous.stability.display} block {previous.label!r} "
            f"(position {previous.index}, section '{previous.section}') that "
            "precedes it.\n"
            "Caching is a prefix match over the rendered bytes in the order "
            "tools -> system -> messages, so every byte after "
            f"{previous.label!r} changes whenever it changes -- "
            f"{current.label!r} can never be served from cache where it is.\n"
            f"Fix: move {current.label!r} before {previous.label!r}, move "
            f"{previous.label!r} to the end of the prompt, or re-tag one of "
            "them if the stability label is wrong."
        )

    # -- breakpoint placement --------------------------------------------

    @property
    def minimum_prefix_tokens(self) -> int:
        return MIN_CACHEABLE_PREFIX_TOKENS.get(self.model, DEFAULT_MIN_PREFIX_TOKENS)

    def _block_tokens(self, block: Block) -> int:
        return self._token_counter(canonical_json(block.render()))

    def plan(self) -> CachePlan:
        """Decide where the `cache_control` markers go.

        A candidate is any position where the next block is less stable than
        this one -- exactly the last byte that survives when the volatile part
        changes -- plus any block explicitly marked `checkpoint=True`, plus the
        final block when `cache_tail=True`. Candidates whose prefix is under the
        model's minimum are dropped (they would silently fail to cache), and if
        more than `max_breakpoints` survive, the ones protecting the most tokens
        win.
        """
        self.validate()
        entries = self._entries()
        minimum = self.minimum_prefix_tokens

        running = 0
        prefix_tokens: list[int] = []
        for entry in entries:
            running += self._block_tokens(entry.block)
            prefix_tokens.append(running)
        total = running

        candidates = {
            i
            for i in range(len(entries) - 1)
            if entries[i].stability < entries[i + 1].stability
        }
        candidates |= {
            entry.index for entry in entries if getattr(entry.block, "checkpoint", False)
        }
        if self.cache_tail and entries:
            candidates.add(len(entries) - 1)
        candidates = sorted(candidates)

        warnings: list[str] = []
        eligible: list[int] = []
        for i in candidates:
            if prefix_tokens[i] < minimum:
                warnings.append(
                    f"skipped boundary after {entries[i].label!r}: its prefix is "
                    f"~{prefix_tokens[i]} estimated tokens, below the "
                    f"{minimum}-token minimum for {self.model}. A breakpoint "
                    "there would be accepted by the API and then silently never "
                    "cache."
                )
            else:
                eligible.append(i)

        if len(eligible) > self.max_breakpoints:
            kept = sorted(
                sorted(eligible, key=lambda i: prefix_tokens[i], reverse=True)[
                    : self.max_breakpoints
                ]
            )
            dropped = [i for i in eligible if i not in kept]
            warnings.append(
                f"{len(eligible)} stability boundaries but only "
                f"{self.max_breakpoints} breakpoints allowed; kept the ones "
                "protecting the most tokens and dropped "
                + ", ".join(repr(entries[i].label) for i in dropped)
                + "."
            )
            eligible = kept

        if candidates and not eligible:
            warnings.append(
                "no cache breakpoints placed: every stability boundary is below "
                f"the {minimum}-token minimum for {self.model}. This prompt is "
                "too small to cache; adding cache_control would only cost the "
                "write premium."
            )
        if not candidates:
            warnings.append(
                "no breakpoint candidates: every block shares one stability "
                "level and none is marked checkpoint=True, so there is no prefix "
                "that outlives the rest of the prompt. Re-tag the volatile part, "
                "mark a checkpoint, or pass cache_tail=True to cache the whole "
                "thing."
            )

        if eligible:
            trailing = (len(entries) - 1) - eligible[-1]
            if trailing > CACHE_LOOKBACK_BLOCKS:
                warnings.append(
                    f"{trailing} content blocks follow the last breakpoint "
                    f"({entries[eligible[-1]].label!r}), more than the "
                    f"{CACHE_LOOKBACK_BLOCKS}-block lookback window. The next "
                    "request's breakpoint will not walk back far enough to find "
                    "this entry and will silently miss; add an intermediate "
                    "breakpoint inside the tail."
                )

        placed = tuple(
            PlacedBreakpoint(
                index=i,
                section=entries[i].section,
                label=entries[i].label,
                stability=entries[i].stability,
                prefix_tokens=prefix_tokens[i],
            )
            for i in eligible
        )
        return CachePlan(
            model=self.model,
            minimum_prefix_tokens=minimum,
            total_estimated_tokens=total,
            breakpoints=placed,
            warnings=tuple(warnings),
            candidates_considered=len(candidates),
            ttl=self.ttl,
        )

    # -- emission --------------------------------------------------------

    def cache_control(self) -> dict[str, str]:
        marker = {"type": "ephemeral"}
        if self.ttl == "1h":
            marker["ttl"] = "1h"
        return marker

    def to_request_kwargs(
        self, *, max_tokens: int = 16000, **extra: Any
    ) -> dict[str, Any]:
        """Return the exact kwargs dict for `client.messages.create(...)`.

        Anything in `extra` is passed straight through (`thinking`,
        `output_config`, `tool_choice`, ...). Note that `tool_choice` and
        `thinking` invalidate only the *messages* cache tier -- they are safe to
        vary per request. Changing `model` or the tool definitions is not.
        """
        plan = self.plan()
        marked = {bp.index for bp in plan.breakpoints}
        entries = self._entries()
        marker = self.cache_control()

        tools: list[dict[str, Any]] = []
        system: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []

        for entry in entries:
            payload = entry.block.render()
            if entry.index in marked:
                payload["cache_control"] = dict(marker)
            if entry.section == "tools":
                tools.append(payload)
            elif entry.section == "system":
                system.append(payload)
            else:
                block = entry.block
                assert isinstance(block, MessageBlock)
                if messages and messages[-1]["role"] == block.role:
                    messages[-1]["content"].append(payload)
                else:
                    messages.append({"role": block.role, "content": [payload]})

        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        kwargs["messages"] = messages
        kwargs.update(extra)
        return kwargs
