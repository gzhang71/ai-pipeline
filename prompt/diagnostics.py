"""Read a response `usage` and say whether the cache worked.

`usage.input_tokens` is the *uncached remainder*, not the prompt size. The full
prompt is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`
-- reading input_tokens alone badly understates a cache-warm request and makes
a total miss look like a normal small prompt.

A miss produces no error, so this module ranks the causes that are actually
checkable from the caller's own state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from common.client import (
    CACHE_LOOKBACK_BLOCKS,
    MIN_CACHEABLE_PREFIX_TOKENS,
    MODEL,
    usage_breakdown,
)

__all__ = ["CacheStatus", "MissReason", "CacheDiagnosis", "diagnose_usage"]


class CacheStatus(str, Enum):
    READ = "read"
    """Part or all of the prefix was served from cache."""

    WRITE = "write"
    """The cache was written but nothing was read -- normal on a cold first
    request, a bug if it repeats."""

    PARTIAL = "partial"
    """Read at an earlier breakpoint and wrote at a later one: the prefix is
    reusable up to some point and then diverges."""

    MISS = "miss"
    """Neither read nor written. No error was raised; this is the silent
    failure the whole package exists to surface."""


class MissReason(str, Enum):
    NO_BREAKPOINTS = "no_breakpoints"
    PREFIX_BELOW_MINIMUM = "prefix_below_minimum"
    LOOKBACK_EXCEEDED = "lookback_exceeded"
    CONCURRENT_WRITE_RACE = "concurrent_write_race"
    PREFIX_CHANGED = "prefix_changed"
    COLD_START = "cold_start"


_EXPLANATIONS = {
    MissReason.NO_BREAKPOINTS: (
        "no cache_control breakpoint was sent, so the API was never asked to "
        "cache anything"
    ),
    MissReason.PREFIX_BELOW_MINIMUM: (
        "the cached prefix is shorter than this model's minimum cacheable "
        "prefix; the API accepts the breakpoint and silently declines to cache"
    ),
    MissReason.LOOKBACK_EXCEEDED: (
        f"more than {CACHE_LOOKBACK_BLOCKS} content blocks were added since the "
        "previous request, so the breakpoint's backward walk never reached the "
        "prior entry"
    ),
    MissReason.CONCURRENT_WRITE_RACE: (
        "concurrent requests shared this prefix; an entry is only readable "
        "after the first response begins streaming, so parallel requests all "
        "pay full price"
    ),
    MissReason.PREFIX_CHANGED: (
        "the rendered prefix bytes differ from the previous request -- run "
        "prompt.find_silent_invalidator on your builder to locate the byte"
    ),
    MissReason.COLD_START: (
        "first request against this prefix (or the previous entry's TTL "
        "expired); a miss here is expected"
    ),
}


@dataclass(frozen=True)
class CacheDiagnosis:
    status: CacheStatus
    tokens: dict[str, int]
    likely_causes: tuple[MissReason, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def cache_hit(self) -> bool:
        return self.status in (CacheStatus.READ, CacheStatus.PARTIAL)

    @property
    def silently_missed(self) -> bool:
        return self.status is CacheStatus.MISS

    @property
    def total_prompt_tokens(self) -> int:
        return self.tokens["total_prompt_tokens"]

    def describe(self) -> str:
        t = self.tokens
        lines = [
            f"cache {self.status.value}: "
            f"read={t['cache_read_input_tokens']} "
            f"written={t['cache_creation_input_tokens']} "
            f"uncached={t['input_tokens']} "
            f"(total prompt {t['total_prompt_tokens']} tokens)"
        ]
        for reason in self.likely_causes:
            lines.append(f"  - {reason.value}: {_EXPLANATIONS[reason]}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.describe()


def diagnose_usage(
    usage: Any,
    *,
    model: str = MODEL,
    breakpoint_prefix_tokens: Sequence[int] | None = None,
    breakpoints_placed: int | None = None,
    blocks_added_since_last_request: int | None = None,
    concurrent_requests: int = 1,
    first_request: bool = False,
) -> CacheDiagnosis:
    """Classify a response's cache behaviour and, on a miss, rank the causes.

    Every keyword beyond `usage` is context the response cannot carry: how many
    breakpoints you sent, how large their prefixes were, how many blocks the
    last agentic turn appended, and whether you fanned out in parallel. Supply
    what you know; each one only ever adds a cause to the ranking.
    """
    tokens = usage_breakdown(usage)
    read = tokens["cache_read_input_tokens"]
    written = tokens["cache_creation_input_tokens"]

    if read and written:
        status = CacheStatus.PARTIAL
    elif read:
        status = CacheStatus.READ
    elif written:
        status = CacheStatus.WRITE
    else:
        status = CacheStatus.MISS

    notes: list[str] = []
    causes: list[MissReason] = []

    minimum = MIN_CACHEABLE_PREFIX_TOKENS.get(model)
    if status is not CacheStatus.MISS:
        if status is CacheStatus.WRITE and not first_request:
            notes.append(
                "the cache was written but never read. If this repeats on every "
                "request, the prefix is changing between requests or the entry "
                "is expiring before the next one arrives."
            )
        return CacheDiagnosis(status=status, tokens=tokens, notes=tuple(notes))

    if breakpoints_placed == 0:
        causes.append(MissReason.NO_BREAKPOINTS)

    if breakpoint_prefix_tokens and minimum is not None:
        if max(breakpoint_prefix_tokens) < minimum:
            causes.append(MissReason.PREFIX_BELOW_MINIMUM)
            notes.append(
                f"largest cached prefix is {max(breakpoint_prefix_tokens)} tokens; "
                f"{model} requires {minimum}. Note this minimum is not monotonic "
                "across model generations -- a prompt that cached on one model "
                "can silently stop caching on a newer one."
            )
    elif minimum is None:
        notes.append(
            f"no minimum cacheable prefix on record for model {model!r}; "
            "below-minimum could not be ruled out."
        )

    if (
        blocks_added_since_last_request is not None
        and blocks_added_since_last_request > CACHE_LOOKBACK_BLOCKS
    ):
        causes.append(MissReason.LOOKBACK_EXCEEDED)
        notes.append(
            f"{blocks_added_since_last_request} blocks were added since the last "
            f"request (limit {CACHE_LOOKBACK_BLOCKS}). Place an intermediate "
            "breakpoint inside long agentic turns."
        )

    if concurrent_requests > 1:
        causes.append(MissReason.CONCURRENT_WRITE_RACE)
        notes.append(
            f"{concurrent_requests} requests shared this prefix concurrently. "
            "Send one, wait for its first streamed token, then fan out the rest."
        )

    if first_request:
        causes.append(MissReason.COLD_START)
    else:
        causes.append(MissReason.PREFIX_CHANGED)

    return CacheDiagnosis(
        status=status,
        tokens=tokens,
        likely_causes=tuple(causes),
        notes=tuple(notes),
    )
