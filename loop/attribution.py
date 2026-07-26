"""Per-segment token attribution for a prompt.

WHY THIS IS NOT TRIVIAL
-----------------------
Token counts are not additive across an arbitrary split of a prompt. Counting
the system prompt alone and the messages alone and adding the two does not
reproduce the count of the whole, because the tokenizer merges across the
boundary and because the API wraps each part in framing the parts don't carry
on their own. Any decomposition that pretends otherwise is lying.

THE METHOD WE USE: ``incremental_prefix_delta``
-----------------------------------------------
We never count a segment in isolation. We count a strictly growing chain of
*prefixes* of the real request and attribute each segment the *delta* it
caused::

    p0  = count(messages=[PROBE])                          -> "framing"
    q1  = count(messages=prefix_through(group_1))          -> group_1 = q1 - p0
    q2  = count(messages=prefix_through(group_2))          -> group_2 = q2 - q1
    ...
    qN  = count(messages=all)
    t   = count(messages=all, tools=T)                     -> tools  = t - qN
    s   = count(messages=all, tools=T, system=S)           -> system = s - t

Two consequences, both deliberate:

1. **The decomposition is exactly additive by construction.** A telescoping sum
   of deltas equals the last term, so ``sum(segments) == count(full prompt)``
   exactly, always. ``decomposition_residual`` is therefore 0 for this method.
   That is not a claim of zero error -- see (2).

2. **Deltas are order-dependent, and that is the residual error.** The delta a
   segment gets is its *marginal* cost given everything measured before it, not
   an intrinsic cost. Measure the same segments in a different order and you
   get slightly different numbers. Where the boundary merges tokens, a segment
   can even come out negative; we report negatives as-is rather than clamping,
   and count them in ``negative_segments``. The measurement order is recorded
   on every run header so a consumer can reason about the bias.

The one genuinely approximate step is ``framing``: the API requires at least
one message, so no prefix exists that contains framing and nothing else. We
probe with a minimal one-character user message and subtract that probe from
the first real group. ``framing`` is therefore high, and the first message
group low, by the probe message's own cost (single-digit tokens). The
``framing`` segment is flagged ``approximate: true`` and listed in the header's
``approximate_segments``.

Counts always come from the ``count_tokens`` endpoint. Never from a
client-side estimator: tiktoken is OpenAI's tokenizer and undercounts Claude
badly, especially on code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from common.client import MODEL
from common.client import count_tokens as api_count_tokens

from .schema import SEGMENT_KINDS

METHOD = "incremental_prefix_delta"

#: The minimal legal message used to isolate request framing.
PROBE_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "."}]

#: Order in which prefixes are measured. Reported on the run header because
#: the deltas are order-dependent.
MEASUREMENT_ORDER = ("framing", "messages", "tool_schemas", "system_prompt")

_BLOCK_KIND_BY_ROLE = {
    ("user", "text"): "user_text",
    ("user", "tool_result"): "tool_result",
    ("assistant", "text"): "assistant_text",
    ("assistant", "thinking"): "thinking",
    ("assistant", "redacted_thinking"): "thinking",
    ("assistant", "tool_use"): "tool_use",
    ("assistant", "server_tool_use"): "tool_use",
}


# --------------------------------------------------------------------------
# Token counters
# --------------------------------------------------------------------------


class TokenCounter(Protocol):
    """Counts tokens for a fully-formed request.

    Implementations must be pure functions of their arguments, so results can
    be memoized across turns.
    """

    def __call__(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        system: Any = None,
        tools: Any = None,
        model: str = MODEL,
    ) -> int: ...


def api_token_counter(
    *,
    messages: Sequence[dict[str, Any]],
    system: Any = None,
    tools: Any = None,
    model: str = MODEL,
) -> int:
    """The real thing: ``POST /v1/messages/count_tokens``."""
    return api_count_tokens(messages, system=system, tools=tools, model=model)


def stable_key(obj: Any) -> str:
    """Deterministic JSON encoding, used as a memo key for prefixes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class CachingTokenCounter:
    """Memoize a counter across turns.

    Successive turns of an agent loop share long prompt prefixes, so the same
    prefix is re-counted on every turn without this. Cuts the ``count_tokens``
    call volume of a long run by roughly an order of magnitude.
    """

    def __init__(self, inner: TokenCounter):
        self.inner = inner
        self._cache: dict[str, int] = {}
        self.calls = 0  # cache misses, i.e. real counter invocations
        self.lookups = 0

    def __call__(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        system: Any = None,
        tools: Any = None,
        model: str = MODEL,
    ) -> int:
        self.lookups += 1
        key = stable_key([model, list(messages), system, tools])
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        value = self.inner(messages=messages, system=system, tools=tools, model=model)
        self._cache[key] = value
        return value


# --------------------------------------------------------------------------
# Normalizing SDK objects into plain JSON-able dicts
# --------------------------------------------------------------------------


def to_plain(obj: Any) -> Any:
    """Recursively convert SDK/pydantic objects into plain dicts and lists.

    ``response.content`` blocks are pydantic models; we append them straight
    back into ``messages`` for the next request, so everything downstream --
    counting, grouping, serializing to JSONL -- has to cope with both shapes.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return to_plain(dump(exclude_none=True))
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_plain(to_dict())
    return str(obj)


def normalize_message(message: Any) -> dict[str, Any]:
    """Return ``{"role": str, "content": [block, ...]}`` with plain blocks."""
    plain = to_plain(message)
    if not isinstance(plain, dict) or "role" not in plain:
        raise TypeError(f"not a message: {plain!r}")
    content = plain.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = [content]
    normalized = []
    for block in content:
        if isinstance(block, str):
            normalized.append({"type": "text", "text": block})
        else:
            normalized.append(block)
    return {**plain, "role": plain["role"], "content": normalized}


def normalize_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    return [normalize_message(m) for m in messages]


def block_kind(role: str, block: dict[str, Any]) -> str:
    return _BLOCK_KIND_BY_ROLE.get((role, block.get("type", "")), "other")


# --------------------------------------------------------------------------
# Segment model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockGroup:
    """A contiguous run of same-kind content blocks inside one message."""

    message_index: int
    role: str
    kind: str
    start: int  # inclusive block index
    end: int  # exclusive block index

    @property
    def segment_id(self) -> str:
        return f"m{self.message_index}:{self.start}-{self.end}:{self.kind}"


@dataclass(frozen=True)
class Segment:
    segment_id: str
    kind: str
    tokens: int
    message_index: int | None = None
    role: str | None = None
    block_span: tuple[int, int] | None = None
    approximate: bool = False
    note: str | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "tokens": self.tokens,
            "message_index": self.message_index,
            "role": self.role,
            "approximate": self.approximate,
        }
        if self.block_span is not None:
            record["block_span"] = list(self.block_span)
        if self.note:
            record["note"] = self.note
        return record


@dataclass
class Attribution:
    method: str
    granularity: str
    segments: list[Segment]
    counted_total: int
    counter_calls: int
    measurement_order: tuple[str, ...] = MEASUREMENT_ORDER
    approximate_segments: list[str] = field(default_factory=lambda: ["framing"])

    @property
    def segment_sum(self) -> int:
        return sum(s.tokens for s in self.segments)

    @property
    def decomposition_residual(self) -> int:
        """Always 0 for ``incremental_prefix_delta`` -- deltas telescope.

        Kept in the schema so a future method that is *not* exactly additive
        (measured-then-normalized, say) can report its true residual here
        without a schema break.
        """
        if not self.segments:
            return 0
        return self.counted_total - self.segment_sum

    @property
    def negative_segments(self) -> int:
        return sum(1 for s in self.segments if s.tokens < 0)

    def by_kind(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for segment in self.segments:
            totals[segment.kind] = totals.get(segment.kind, 0) + segment.tokens
        return {k: totals[k] for k in SEGMENT_KINDS if k in totals}

    def to_record(self) -> dict[str, Any]:
        return {
            "counted_total": self.counted_total,
            "segments": [s.to_record() for s in self.segments],
            "by_kind": self.by_kind(),
            "segment_sum": self.segment_sum,
            "decomposition_residual": self.decomposition_residual,
            "counter_calls": self.counter_calls,
            "negative_segments": self.negative_segments,
        }


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def block_groups(
    messages: Sequence[dict[str, Any]], granularity: str = "block_group"
) -> list[BlockGroup]:
    """Split messages into the units we attribute tokens to.

    ``block_group`` -- one group per contiguous run of same-kind blocks.
    ``message``     -- one group per message, kind taken from its first block.

    Groups are always *prefixes* of the message list, which is what makes the
    truncated prompts legal: a prefix can hold a ``tool_use`` whose
    ``tool_result`` has not arrived yet (the normal mid-turn state), but never
    the reverse.
    """
    groups: list[BlockGroup] = []
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        if not content:
            groups.append(BlockGroup(index, role, "other", 0, 0))
            continue
        if granularity == "message":
            groups.append(
                BlockGroup(index, role, block_kind(role, content[0]), 0, len(content))
            )
            continue
        start = 0
        current = block_kind(role, content[0])
        for position in range(1, len(content)):
            kind = block_kind(role, content[position])
            if kind != current:
                groups.append(BlockGroup(index, role, current, start, position))
                start, current = position, kind
        groups.append(BlockGroup(index, role, current, start, len(content)))
    return groups


def prefix_messages(
    messages: Sequence[dict[str, Any]], group: BlockGroup
) -> list[dict[str, Any]]:
    """Every message before ``group``, plus its own message truncated to it."""
    prefix = [dict(m) for m in messages[: group.message_index]]
    head = dict(messages[group.message_index])
    head["content"] = head["content"][: group.end]
    prefix.append(head)
    return prefix


# --------------------------------------------------------------------------
# The attribution itself
# --------------------------------------------------------------------------


def attribute(
    messages: Iterable[Any],
    *,
    system: Any = None,
    tools: Any = None,
    model: str = MODEL,
    counter: TokenCounter | None = None,
    granularity: str = "block_group",
) -> Attribution:
    """Decompose a prompt into per-segment token counts.

    ``granularity``:
      ``block_group`` -- per contiguous same-kind block run (default, ~1
        ``count_tokens`` call per new group, heavily memoized across turns)
      ``message``     -- per message
      ``coarse``      -- four calls total: framing / messages / tools / system
      ``off``         -- one call, total only, no segments
    """
    if granularity not in ("block_group", "message", "coarse", "off"):
        raise ValueError(f"unknown granularity {granularity!r}")

    # Reuse the caller's cache when they supply one, so prefixes shared
    # between turns are counted once for the whole run rather than once per
    # turn. Only the calls this invocation actually caused are reported.
    if isinstance(counter, CachingTokenCounter):
        count = counter
    else:
        count = CachingTokenCounter(counter or api_token_counter)
    calls_before = count.calls

    plain = normalize_messages(messages)
    if not plain:
        raise ValueError("cannot attribute an empty message list")

    def measure(msgs, *, with_tools=False, with_system=False) -> int:
        return count(
            messages=msgs,
            system=system if with_system else None,
            tools=tools if with_tools else None,
            model=model,
        )

    if granularity == "off":
        total = measure(plain, with_tools=True, with_system=True)
        return Attribution(
            method=METHOD,
            granularity="off",
            segments=[],
            counted_total=total,
            counter_calls=count.calls - calls_before,
            approximate_segments=[],
        )

    segments: list[Segment] = []

    # 1. Framing. No prefix contains framing alone -- the API requires at least
    #    one message -- so we probe with a minimal one and flag the result.
    framing = measure(PROBE_MESSAGES)
    segments.append(
        Segment(
            segment_id="framing",
            kind="framing",
            tokens=framing,
            approximate=True,
            note=(
                "measured with a 1-character probe message; includes that "
                "probe's own cost, which the first message segment is short by"
            ),
        )
    )

    # 2. Messages, as a growing prefix chain.
    previous = framing
    if granularity == "coarse":
        messages_total = measure(plain)
        segments.append(
            Segment(
                segment_id="messages",
                kind="messages_total",
                tokens=messages_total - previous,
                note="all messages as one segment (coarse granularity)",
            )
        )
        previous = messages_total
    else:
        for group in block_groups(plain, granularity):
            at_group = measure(prefix_messages(plain, group))
            segments.append(
                Segment(
                    segment_id=group.segment_id,
                    kind=group.kind,
                    tokens=at_group - previous,
                    message_index=group.message_index,
                    role=group.role,
                    block_span=(group.start, group.end),
                )
            )
            previous = at_group

    # 3. Tools, then system -- render order among themselves, measured on top
    #    of the full message list so the deltas are marginal costs against a
    #    realistic prompt rather than against a stub.
    with_tools = (
        measure(plain, with_tools=True) if tools else previous
    )
    segments.append(
        Segment(
            segment_id="tool_schemas",
            kind="tool_schemas",
            tokens=with_tools - previous,
        )
    )
    with_system = (
        measure(plain, with_tools=True, with_system=True) if system else with_tools
    )
    segments.append(
        Segment(
            segment_id="system_prompt",
            kind="system_prompt",
            tokens=with_system - with_tools,
        )
    )

    # Report in render order (tools -> system -> messages) even though we
    # measured messages first; the ordering used is on the run header.
    order = {"framing": 0, "tool_schemas": 1, "system_prompt": 2}
    segments.sort(key=lambda s: order.get(s.kind, 3))

    return Attribution(
        method=METHOD,
        granularity=granularity,
        segments=segments,
        counted_total=with_system,
        counter_calls=count.calls - calls_before,
    )


def reconcile(
    attribution: Attribution,
    usage: dict[str, int],
    *,
    tolerance_fraction: float = 0.02,
) -> dict[str, Any]:
    """Compare our decomposition against the response's authoritative usage.

    ``usage.input_tokens`` is the *uncached remainder*, not the prompt size.
    The real prompt is ``input + cache_creation + cache_read``; comparing
    against ``input_tokens`` alone will look catastrophically wrong on any
    cache-warm request.

    ``count_tokens`` and billed usage are computed by different code paths and
    routinely differ by a few tokens even on an identical prompt, so the
    residual is expected to be small but nonzero. We surface it rather than
    hiding it.
    """
    authoritative = int(usage.get("total_prompt_tokens", 0))
    counted = attribution.counted_total
    residual = counted - authoritative
    denominator = max(authoritative, 1)
    fraction = residual / denominator
    return {
        "counted_total": counted,
        "authoritative_total": authoritative,
        "residual_tokens": residual,
        "residual_fraction": round(fraction, 6),
        "within_tolerance": abs(fraction) <= tolerance_fraction,
        "tolerance_fraction": tolerance_fraction,
    }
