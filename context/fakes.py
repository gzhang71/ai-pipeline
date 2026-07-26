"""Offline doubles for the bench.

There are no API credentials in this environment, so the whole test suite and
the default bench run against these. They are *behavioural*, not mocks that
echo back what the caller wants:

* `FakeClient` answers the task's query by reading the context it was actually
  handed. If a strategy dropped the fact, the fake cannot answer — nothing in
  the fake knows which strategy is being tested, or what the right answer is.
  Success is therefore a measurement of information survival, not of the mock.
* usage is reported the way the real API reports it — `input_tokens` is only
  the *uncached remainder*, with the rest split across
  `cache_creation_input_tokens` and `cache_read_input_tokens`. Accounting code
  that reads `input_tokens` alone will report roughly a tenth of the truth,
  which is the point.
* the server-side compaction path is *simulated*: the fake summarizes the old
  span, returns a `compaction` block, and honours that block on the next call.
  Numbers for `ServerCompaction` are therefore a model of the feature, not a
  measurement of it. This is called out in the README and in the report.

A note on reuse: `loop/testing.py` (a sibling subproject) exposes similar
doubles. This package keeps its own because the two need different contracts —
the fake here has to simulate server-side compaction state and answer
fact-recall queries, neither of which the loop profiler's fake does — and
because a sibling's test double changing under us would silently move this
bench's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from common.client import MODEL

from .strategies import COMPACTION_EDIT
from .summarizers import FakeSummarizer, Summarizer, extract_facts
from .tasks import QUERY_PREFIX
from .tokens import HeuristicTokenCounter, TokenCounter
from .validation import Message, blocks, find_tail_start, history_text, sanitize


@dataclass
class FakeUsage:
    """Same attribute names the SDK's usage object exposes."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeResponse:
    content: list[dict[str, Any]]
    usage: FakeUsage
    stop_reason: str = "end_turn"
    model: str = MODEL


@dataclass
class CallRecord:
    prompt_tokens: int
    visible_messages: int
    compacted: bool
    overrides: dict[str, Any] = field(default_factory=dict)


class FakeClient:
    """A scripted stand-in for `client.messages.create`.

    Interface: ``create(messages=..., system=..., tools=..., **overrides)``.
    `overrides` carries whatever a strategy asked for (`betas`,
    `context_management`), exactly as the real call would receive it.
    """

    def __init__(
        self,
        counter: TokenCounter | None = None,
        *,
        compaction_threshold: int = 900,
        keep_recent: int = 4,
        compactor: Summarizer | None = None,
        cache_hit_ratio: float = 0.75,
    ):
        self.counter = counter or HeuristicTokenCounter()
        self.compaction_threshold = compaction_threshold
        self.keep_recent = keep_recent
        self.compactor = compactor or FakeSummarizer(keep_facts=3)
        self.cache_hit_ratio = cache_hit_ratio
        self.calls = 0
        self.compactions = 0
        self.calls_log: list[CallRecord] = []
        self._last_prompt = 0

    # -- server-side compaction simulation --------------------------------

    @staticmethod
    def _compaction_enabled(overrides: dict[str, Any]) -> bool:
        edits = (overrides.get("context_management") or {}).get("edits") or []
        return any(e.get("type") == COMPACTION_EDIT["type"] for e in edits)

    @staticmethod
    def _honour_existing_block(messages: Sequence[Message]) -> tuple[list[Message], str]:
        """Replace everything before the newest compaction block with its summary.

        This is the half of the contract the caller has to hold up: the block
        only exists in the history if the caller appended the *whole*
        `response.content`. Append only the text and this returns the raw
        history, and the server has to redo the work.
        """
        index: int | None = None
        summary = ""
        for i, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            for block in blocks(message):
                if block.get("type") == "compaction":
                    index, summary = i, str(block.get("summary", ""))
        if index is None:
            return list(messages), ""
        head = [{"role": "user", "content": [{"type": "text", "text": summary}]}]
        return sanitize(head + list(messages[index + 1 :])), summary

    def _visible(
        self, messages: Sequence[Message], overrides: dict[str, Any]
    ) -> tuple[list[Message], dict[str, Any] | None, int]:
        if not self._compaction_enabled(overrides):
            return list(messages), None, 0

        visible, prior = self._honour_existing_block(messages)
        if self.counter.count(visible) <= self.compaction_threshold:
            return visible, None, 0

        start = find_tail_start(
            visible, max(1, len(visible) - self.keep_recent), prefer="backward"
        )
        older, tail = visible[:start], visible[start:]
        if not older:
            return visible, None, 0

        summary, usage = self.compactor.summarize(older, prior)
        self.compactions += 1
        visible = sanitize(
            [{"role": "user", "content": [{"type": "text", "text": summary}]}] + tail
        )
        return visible, {"type": "compaction", "summary": summary}, usage.output_tokens

    # -- answering ---------------------------------------------------------

    def _respond(self, visible: Sequence[Message]) -> str:
        text = history_text(visible)
        facts: dict[str, str] = {}
        for key, value in extract_facts(text):
            facts[key] = value  # last write wins, as a reader would assume

        query_line = ""
        for line in text.splitlines():
            if QUERY_PREFIX in line:
                query_line = line
        if not query_line:
            return (
                "Acknowledged; continuing the checklist. "
                f"{len(facts)} durable values are currently in view."
            )

        keys = [
            k.strip()
            for k in query_line.split(QUERY_PREFIX, 1)[1].split(",")
            if k.strip()
        ]
        lines = [f"{key} = {facts.get(key, 'UNKNOWN')}" for key in keys]
        return "Final report.\n" + "\n".join(lines)

    # -- the call ----------------------------------------------------------

    def create(
        self,
        *,
        messages: Sequence[Message],
        system: Any = None,
        tools: Any = None,
        **overrides: Any,
    ) -> FakeResponse:
        self.calls += 1
        visible, compaction_block, compaction_output = self._visible(messages, overrides)

        prompt = self.counter.count(visible)
        text = self._respond(visible)
        output = max(1, len(text) // 4) + compaction_output

        # Split the prompt the way the API does: a cached prefix that is read
        # back cheaply, a newly written suffix, and an uncached remainder.
        cache_read = min(self._last_prompt, prompt)
        remainder = prompt - cache_read
        cache_creation = int(remainder * self.cache_hit_ratio)
        input_tokens = remainder - cache_creation
        self._last_prompt = prompt

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if compaction_block is not None:
            content.append(compaction_block)

        self.calls_log.append(
            CallRecord(
                prompt_tokens=prompt,
                visible_messages=len(visible),
                compacted=compaction_block is not None,
                overrides=dict(overrides),
            )
        )
        return FakeResponse(
            content=content,
            usage=FakeUsage(
                input_tokens=input_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                output_tokens=output,
            ),
        )


__all__ = ["FakeClient", "FakeResponse", "FakeUsage", "CallRecord"]
