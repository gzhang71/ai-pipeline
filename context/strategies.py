"""History-management strategies behind one interface.

    result = strategy.apply(messages, budget)
    request_messages = result.messages

`apply` returns a `StrategyResult` rather than a bare list because two things
have to come back with the edited history:

* `usage` — tokens the strategy spent on its *own* model calls. Return bare
  messages and this cost vanishes, and every summarizing strategy looks
  cheaper than it is.
* `request_overrides` — extra kwargs for the API call (`ServerCompaction`
  does its work server-side and has nothing to edit client-side).

`result.messages` is the whole interface for a caller that does not care about
either.

**When strategies fire.** On a token threshold *or* a turn count, whichever
trips first (`Budget.is_over`). Tokens are the real constraint — that is what
the context window and the bill are denominated in. But a pure token trigger
misses the case where each turn is individually small and there are hundreds
of them: block count grows, the 20-block cache lookback window is blown
through, and per-turn overhead accumulates while the character count stays
low. The turn count is a cheap backstop for that. Turn-only triggers are worse
than either: they fire on a conversation of ten one-line turns and do not fire
on two turns that each pasted a 200KB file.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from .summarizers import (
    NoteWriter,
    Summarizer,
    default_note_writer,
    default_summarizer,
)
from .tokens import HeuristicTokenCounter, TokenCounter
from .usage import Usage
from .validation import Message, blocks, find_tail_start, normalize, sanitize

# Beta headers / edit types, per the Claude API. These are two *different*
# server-side features and must not be conflated:
#   compaction      -> summarizes history
#   context editing -> clears old tool results / thinking blocks
COMPACTION_BETA = "compact-2026-01-12"
COMPACTION_EDIT = {"type": "compact_20260112"}
CONTEXT_EDITING_BETA = "context-management-2025-06-27"
CLEAR_TOOL_USES_EDIT = {"type": "clear_tool_uses_20250919"}
CLEAR_THINKING_EDIT = {"type": "clear_thinking_20251015"}


@dataclass
class Budget:
    """Everything a strategy needs to make a real decision."""

    max_tokens: int = 1200
    max_turns: int = 8
    keep_recent_messages: int = 4
    keep_recent_tool_results: int = 1
    turn_index: int = 0
    objective: str = ""
    workspace: Path | None = None
    counter: TokenCounter = field(default_factory=HeuristicTokenCounter)

    def count(self, messages: Sequence[Message]) -> int:
        return self.counter.count(messages)

    def is_over(self, messages: Sequence[Message]) -> bool:
        """Fire on tokens OR turns — see the module docstring."""
        return self.count(messages) > self.max_tokens or self.turn_index > self.max_turns

    def for_turn(self, turn_index: int) -> "Budget":
        return replace(self, turn_index=turn_index)


@dataclass
class StrategyResult:
    messages: list[Message]
    usage: Usage = Usage()
    request_overrides: dict[str, Any] = field(default_factory=dict)
    fired: bool = False
    note: str = ""

    def __iter__(self):  # lets callers do `messages, _ = result`
        return iter((self.messages, self.usage))


class Strategy:
    """Base class. Subclasses implement `_compact`."""

    name: str = "strategy"
    #: False for strategies that delegate the reduction to the server.
    client_side_reduction: bool = True

    def apply(self, messages: Sequence[Message], budget: Budget) -> StrategyResult:
        history = normalize(messages)
        if not budget.is_over(history):
            return StrategyResult(messages=history, fired=False)
        result = self._compact(history, budget)
        result.messages = sanitize(result.messages)
        result.fired = True
        return result

    def _compact(self, messages: list[Message], budget: Budget) -> StrategyResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-run state. The bench calls this between tasks."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------------


class TailTruncation(Strategy):
    """Drop the oldest messages. The baseline everything is measured against.

    Zero extra model calls, zero latency, and it deletes early information
    outright — no summary, no note, nothing. If a cleverer strategy cannot
    beat this on success rate, it is not paying for itself.
    """

    name = "TailTruncation"

    def _compact(self, messages: list[Message], budget: Budget) -> StrategyResult:
        keep = max(1, budget.keep_recent_messages)
        desired = max(0, len(messages) - keep)

        start = find_tail_start(messages, desired, prefer="forward")
        # Keep dropping until under budget, or until only the last legal
        # window is left. Never drop everything.
        while budget.count(messages[start:]) > budget.max_tokens:
            nxt = find_tail_start(messages, start + 1, prefer="forward")
            if nxt <= start or nxt >= len(messages):
                break
            start = nxt

        return StrategyResult(
            messages=messages[start:], note=f"dropped {start} oldest messages"
        )


class RecursiveSummarization(Strategy):
    """Client-side rolling summary: summarize old turns, keep a recent tail.

    Each compaction folds the previous summary plus the newly-evicted window
    into one new summary. Costs one model call per compaction, and every
    compaction re-reads its own previous output — so information is passed
    through the summarizer repeatedly and degrades a little each time.
    """

    name = "RecursiveSummarization"

    def __init__(self, summarizer: Summarizer | None = None, *, offline: bool = True):
        self.summarizer = summarizer or default_summarizer(offline=offline)
        self._summary = ""

    def reset(self) -> None:
        self._summary = ""

    def _split(self, messages: list[Message], budget: Budget) -> int:
        keep = max(1, budget.keep_recent_messages)
        return find_tail_start(messages, len(messages) - keep, prefer="backward")

    def _compact(self, messages: list[Message], budget: Budget) -> StrategyResult:
        start = self._split(messages, budget)
        older, tail = messages[:start], messages[start:]
        if not older:
            return StrategyResult(messages=messages, note="nothing old enough to summarize")

        summary, usage = self.summarizer.summarize(older, self._summary)
        self._summary = summary
        head = [{"role": "user", "content": [{"type": "text", "text": summary}]}]
        return StrategyResult(
            messages=head + tail,
            usage=usage,
            note=f"summarized {len(older)} messages",
        )


class AnchoredSummary(RecursiveSummarization):
    """Recursive summarization with the original objective pinned verbatim.

    The objective is the one thing a summarizer must never paraphrase: an
    agent that has drifted on *what it was asked to do* cannot recover from
    the transcript, because the transcript is gone. This costs a handful of
    tokens per request forever and removes an entire failure class.
    """

    name = "AnchoredSummary"

    def __init__(
        self,
        summarizer: Summarizer | None = None,
        *,
        offline: bool = True,
        objective: str = "",
    ):
        super().__init__(summarizer, offline=offline)
        self.objective = objective
        self._anchor: Message | None = None

    def reset(self) -> None:
        super().reset()
        self._anchor = None

    def _anchor_message(self, messages: list[Message], budget: Budget) -> Message:
        """The verbatim objective. Captured once and never rewritten."""
        if self._anchor is not None:
            return self._anchor
        objective = self.objective or budget.objective
        if objective:
            text = objective
        else:  # fall back to the opening user turn, verbatim
            text = "\n".join(
                str(b.get("text", ""))
                for b in blocks(messages[0])
                if b.get("type") == "text"
            )
        self._anchor = {
            "role": "user",
            "content": [{"type": "text", "text": f"[OBJECTIVE — verbatim]\n{text}"}],
        }
        return self._anchor

    def _compact(self, messages: list[Message], budget: Budget) -> StrategyResult:
        anchor = self._anchor_message(messages, budget)
        start = self._split(messages, budget)
        older, tail = messages[:start], messages[start:]
        if not older:
            return StrategyResult(messages=messages, note="nothing old enough to summarize")

        summary, usage = self.summarizer.summarize(older, self._summary)
        self._summary = summary
        head = [anchor, {"role": "user", "content": [{"type": "text", "text": summary}]}]
        return StrategyResult(
            messages=head + tail,
            usage=usage,
            note=f"summarized {len(older)} messages, objective pinned",
        )


class NoteTaking(Strategy):
    """Durable state on disk; context is rehydrated from the file.

    Two sources feed the notes file:

    1. `write_note` tool calls the agent made — harvested for free, no model
       call, because the agent already decided what was worth keeping.
    2. a scribe pass over the window about to be discarded, for everything the
       agent did not think to write down.

    Unlike a rolling summary, the file is keyed and merged rather than
    re-summarized, so a fact does not decay just because it is old — it
    survives until its key is overwritten or the file hits its capacity.
    Costs one model call per compaction plus the notes blob in every
    subsequent prompt.
    """

    name = "NoteTaking"
    NOTE_TOOL = "write_note"

    def __init__(
        self,
        note_writer: NoteWriter | None = None,
        *,
        offline: bool = True,
        workspace: Path | None = None,
        filename: str = "notes.md",
    ):
        self.note_writer = note_writer or default_note_writer(offline=offline)
        self.filename = filename
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._workspace = workspace
        if workspace is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="context-notes-")
            self._workspace = Path(self._tmp.name)

    @property
    def notes_path(self) -> Path:
        return Path(self._workspace) / self.filename

    def read_notes(self) -> str:
        path = self.notes_path
        return path.read_text() if path.exists() else ""

    def reset(self) -> None:
        if self.notes_path.exists():
            self.notes_path.unlink()

    def _harvest_tool_notes(self, messages: Sequence[Message]) -> list[str]:
        harvested: list[str] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for block in blocks(message):
                if block.get("type") == "tool_use" and block.get("name") == self.NOTE_TOOL:
                    payload = block.get("input") or {}
                    if isinstance(payload, dict):
                        note = payload.get("note") or payload.get("text")
                    else:
                        note = str(payload)
                    if note:
                        harvested.append(str(note))
        return harvested

    @staticmethod
    def _pinned_keys(harvested: Sequence[str]) -> list[str]:
        from .summarizers import extract_facts

        return [key for note in harvested for key, _ in extract_facts(note)]

    def _compact(self, messages: list[Message], budget: Budget) -> StrategyResult:
        keep = max(1, budget.keep_recent_messages)
        start = find_tail_start(messages, len(messages) - keep, prefer="backward")
        older, tail = messages[:start], messages[start:]
        if not older:
            return StrategyResult(messages=messages, note="nothing old enough to evict")

        notes = self.read_notes()
        # Facts the agent chose to write down with `write_note`. Their text is
        # already in the window, so harvesting is free (no model call); what it
        # buys is a list of keys the note writer must not evict.
        pinned = self._pinned_keys(self._harvest_tool_notes(older))

        notes, usage = self.note_writer.update(older, notes, pinned=pinned)
        path = self.notes_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(notes)

        head = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"[NOTES — durable state re-read from {self.filename}; the "
                            f"transcript before this point is gone]\n{notes}"
                        ),
                    }
                ],
            }
        ]
        return StrategyResult(
            messages=head + tail,
            usage=usage,
            note=f"evicted {len(older)} messages into {self.filename}",
        )


class ToolResultEviction(Strategy):
    """Keep `tool_use`, drop stale `tool_result` payloads, leave a placeholder.

    The client-side sibling of the API's `clear_tool_uses_20250919` context
    edit. The pairing is preserved exactly — the `tool_result` block stays,
    with the same `tool_use_id`, and only its payload is replaced — so the
    request stays legal. Zero model calls.

    Its blind spot is the point of the exercise: it protects conversational
    text perfectly and destroys everything that only ever existed inside a
    tool result.
    """

    name = "ToolResultEviction"

    def __init__(self, placeholder: str | None = None):
        self.placeholder = placeholder or (
            "[tool result evicted from context to save tokens; "
            "re-run the tool if this is needed again]"
        )

    def _placeholder_for(self, block: dict[str, Any], size: int) -> str:
        return f"{self.placeholder} (was {size} chars)"

    @staticmethod
    def _payload_size(block: dict[str, Any]) -> int:
        content = block.get("content")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            return sum(len(str(c)) for c in content)
        return 0

    def _compact(self, messages: list[Message], budget: Budget) -> StrategyResult:
        # Index every tool_result, newest last.
        positions: list[tuple[int, int]] = [
            (i, j)
            for i, message in enumerate(messages)
            if message.get("role") == "user"
            for j, block in enumerate(message["content"])
            if block.get("type") == "tool_result"
        ]
        protected = set(positions[-budget.keep_recent_tool_results :]) if (
            budget.keep_recent_tool_results > 0
        ) else set()

        out = [{**m, "content": list(m["content"])} for m in messages]
        evicted = 0
        for i, j in positions:
            if (i, j) in protected:
                continue
            block = out[i]["content"][j]
            size = self._payload_size(block)
            replacement = self._placeholder_for(block, size)
            if size <= len(replacement):
                continue  # evicting would make the request bigger
            out[i]["content"][j] = {
                "type": "tool_result",
                "tool_use_id": block["tool_use_id"],
                "content": replacement,
            }
            evicted += 1

        return StrategyResult(messages=out, note=f"evicted {evicted} tool results")


class ServerCompaction(Strategy):
    """Hand the problem to the API's own `compact_20260112`.

    The "do nothing clever" comparison point. Nothing is edited client-side —
    the strategy only attaches the beta header and the context-management
    edit, and the server summarizes the history when it approaches the
    trigger threshold.

    The contract that breaks people: the response carries a compaction block,
    and the *entire* `response.content` must be appended back to `messages`.
    Append only the extracted text and the block is lost, the server has no
    record of the compaction, and state silently resets. The bench runner
    always appends full content; `context.bench` has a switch to do it wrong,
    and a test that shows what that costs.

    Note this is compaction (summarize), not context editing (clear). Both are
    exposed here — `include_tool_clearing=True` adds
    `clear_tool_uses_20250919` — but they are different features with
    different beta headers and are not interchangeable.
    """

    name = "ServerCompaction"
    client_side_reduction = False

    def __init__(self, *, include_tool_clearing: bool = False):
        self.include_tool_clearing = include_tool_clearing

    def _overrides(self) -> dict[str, Any]:
        betas = [COMPACTION_BETA]
        edits: list[dict[str, Any]] = [dict(COMPACTION_EDIT)]
        if self.include_tool_clearing:
            betas.append(CONTEXT_EDITING_BETA)
            edits.append(dict(CLEAR_TOOL_USES_EDIT))
        return {"betas": betas, "context_management": {"edits": edits}}

    def apply(self, messages: Sequence[Message], budget: Budget) -> StrategyResult:
        # Always attach the overrides: the server decides when to compact, and
        # a request that only enables compaction once it is already over
        # budget has left the decision too late.
        history = normalize(messages)
        return StrategyResult(
            messages=history,
            request_overrides=self._overrides(),
            fired=budget.is_over(history),
            note="server-side compaction enabled",
        )


def all_strategies(*, offline: bool = True) -> list[Strategy]:
    """One fresh instance of every strategy, in baseline-first order."""
    return [
        TailTruncation(),
        RecursiveSummarization(offline=offline),
        AnchoredSummary(offline=offline),
        NoteTaking(offline=offline),
        ToolResultEviction(),
        ServerCompaction(),
    ]


__all__ = [
    "Budget",
    "StrategyResult",
    "Strategy",
    "TailTruncation",
    "RecursiveSummarization",
    "AnchoredSummary",
    "NoteTaking",
    "ToolResultEviction",
    "ServerCompaction",
    "all_strategies",
    "COMPACTION_BETA",
    "COMPACTION_EDIT",
    "CONTEXT_EDITING_BETA",
    "CLEAR_TOOL_USES_EDIT",
    "CLEAR_THINKING_EDIT",
]
