"""The model calls that history-management strategies make on their own.

Summarizing and note-writing are not free. They are extra inference, and if
their tokens are not counted the comparison is rigged in their favour. Every
callable here returns `(text, Usage)` and the caller folds that `Usage` into
the strategy's total.

Both a live (`Model*`) and an offline fake (`Fake*`) implementation are
provided. The fakes are deliberately *lossy* — a summarizer that preserved
everything would be a compression oracle, not a summarizer, and the bench
would tell you nothing.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

from common.client import MODEL, get_client

from .usage import Usage
from .validation import Message, history_text

FACT_RE = re.compile(r"\[FACT\]\s*([A-Z][A-Z0-9_]*)\s*=\s*(.+)")

SUMMARIZER_SYSTEM = (
    "You compress conversation history for an agent that must keep working "
    "after the raw transcript is gone. Preserve concrete durable state: "
    "identifiers, decisions, file paths, values, constraints, and anything "
    "marked [FACT]. Drop pleasantries, restatements, and the bodies of tool "
    "output you have already extracted the answer from. Write terse bullet "
    "lines, no preamble."
)

NOTE_WRITER_SYSTEM = (
    "You maintain a durable notes file for a long-running agent. You are given "
    "the current notes and a window of conversation that is about to be "
    "discarded. Return the complete updated notes file and nothing else. "
    "Keep one fact per line in the form '[FACT] KEY = value'. Merge rather "
    "than append: if a key already exists and the new information supersedes "
    "it, replace that line. Never invent facts."
)


class Summarizer(Protocol):
    def summarize(
        self, messages: Sequence[Message], previous_summary: str = ""
    ) -> tuple[str, Usage]: ...


class NoteWriter(Protocol):
    def update(
        self,
        messages: Sequence[Message],
        current_notes: str = "",
        pinned: Sequence[str] = (),
    ) -> tuple[str, Usage]: ...


# --------------------------------------------------------------------------
# Live implementations (require credentials)
# --------------------------------------------------------------------------


class ModelSummarizer:
    """Summarize a window of history with a real model call."""

    def __init__(self, *, model: str = MODEL, max_tokens: int = 2000, effort: str = "low"):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def _call(self, system: str, prompt: str) -> tuple[str, Usage]:
        response = get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return text, Usage.from_response_usage(response.usage)

    def summarize(
        self, messages: Sequence[Message], previous_summary: str = ""
    ) -> tuple[str, Usage]:
        prompt = (
            f"Existing summary of everything before this window:\n{previous_summary or '(none)'}\n\n"
            f"Conversation window to fold in:\n{history_text(messages)}\n\n"
            "Return the updated summary."
        )
        return self._call(SUMMARIZER_SYSTEM, prompt)


class ModelNoteWriter(ModelSummarizer):
    """Rewrite a durable notes file with a real model call."""

    def update(
        self,
        messages: Sequence[Message],
        current_notes: str = "",
        pinned: Sequence[str] = (),
    ) -> tuple[str, Usage]:
        pin_line = (
            f"The agent explicitly wrote these keys down; keep them even if you "
            f"must drop something else: {', '.join(pinned)}\n\n"
            if pinned
            else ""
        )
        prompt = (
            f"Current notes file:\n{current_notes or '(empty)'}\n\n"
            f"{pin_line}"
            f"Conversation window about to be discarded:\n{history_text(messages)}\n\n"
            "Return the complete updated notes file."
        )
        return self._call(NOTE_WRITER_SYSTEM, prompt)


# --------------------------------------------------------------------------
# Offline fakes
# --------------------------------------------------------------------------


def extract_facts(messages: Sequence[Message] | str) -> list[tuple[str, str]]:
    """Every `[FACT] KEY = value` line, in order of appearance."""
    text = messages if isinstance(messages, str) else history_text(messages)
    return [(m.group(1), m.group(2).strip()) for m in FACT_RE.finditer(text)]


#: Keys a summarizer would reasonably judge as routine telemetry — repetitive,
#: superseded by the next reading, not worth a line in a summary.
ROUTINE_KEY_RE = re.compile(r"^(METRIC|LATENCY|QPS|HEAP)_\d+$")


class FakeSummarizer:
    """A fixed-budget summarizer that keeps the K most salient recent facts.

    Real summarizers do not fail by forgetting at random; they fail because
    they compress to a budget and something has to fall out. What falls out is
    biased toward *old* and toward *whatever the summarizer judged
    unimportant*. This models both: facts are ranked salient-before-routine
    (a repeated `METRIC_07` reading loses to a `DEPLOY_TOKEN`) and recent-
    before-old within each band, then truncated to `keep_facts`.

    Giving it salience matters for fairness — a pure recency rule would let
    filler telemetry evict every identifier, and summarization would lose the
    bench by construction rather than on the merits. What it still cannot do
    is keep more than `keep_facts` things, which is the real cliff.

    Every summarizing strategy shares this object, so none of them is
    handicapped relative to the others.
    """

    def __init__(self, keep_facts: int = 5, tokens_per_char: float = 0.25):
        self.keep_facts = keep_facts
        self.tokens_per_char = tokens_per_char
        self.calls = 0

    def summarize(
        self, messages: Sequence[Message], previous_summary: str = ""
    ) -> tuple[str, Usage]:
        self.calls += 1
        window = history_text(messages)
        facts = extract_facts(previous_summary) + extract_facts(window)

        merged: dict[str, str] = {}
        for key, value in facts:
            merged.pop(key, None)  # re-stating a fact refreshes its recency
            merged[key] = value

        ordered = list(merged.items())
        salient = [kv for kv in ordered if not ROUTINE_KEY_RE.match(kv[0])]
        routine = [kv for kv in ordered if ROUTINE_KEY_RE.match(kv[0])]
        kept = salient[-self.keep_facts :]
        if len(kept) < self.keep_facts:
            kept = routine[-(self.keep_facts - len(kept)) :] + kept

        lines = [
            f"[SUMMARY] Compressed {len(messages)} earlier messages "
            f"({len(merged)} facts seen, {len(kept)} retained)."
        ]
        lines += [f"[FACT] {k} = {v}" for k, v in kept]
        text = "\n".join(lines)

        prompt_tokens = int(len(window + previous_summary) * self.tokens_per_char)
        usage = Usage(
            input_tokens=prompt_tokens,
            output_tokens=int(len(text) * self.tokens_per_char),
            model_calls=1,
        )
        return text, usage


class FakeNoteWriter:
    """A note writer with a bounded file.

    Keyed and merge-on-write, so a fact does not get squeezed out just because
    newer facts arrived — but the file has a hard capacity, so on a long
    enough run the oldest notes are still evicted. Note-taking is not magic;
    it trades a recency cliff for a capacity cliff, and the bench includes a
    task (`fact-flood`) built to fall off it.

    Salience handling is symmetric with `FakeSummarizer`: routine telemetry is
    dropped before durable identifiers, so neither side of the comparison is
    penalised for noise the other one filters.
    """

    def __init__(self, max_facts: int = 5, tokens_per_char: float = 0.25):
        self.max_facts = max_facts
        self.tokens_per_char = tokens_per_char
        self.calls = 0

    def update(
        self,
        messages: Sequence[Message],
        current_notes: str = "",
        pinned: Sequence[str] = (),
    ) -> tuple[str, Usage]:
        self.calls += 1
        window = history_text(messages)

        notes: dict[str, str] = {}
        for key, value in extract_facts(current_notes) + extract_facts(window):
            notes.pop(key, None)  # re-stating a fact refreshes its recency
            notes[key] = value

        # Over capacity, evict in this order: routine telemetry, then the
        # least-recently-updated ordinary fact. Keys the agent explicitly
        # wrote down with `write_note` are evicted last — that is the one
        # mechanism a rolling summary has no equivalent of, because only the
        # note-taking strategy gives the agent a say in what survives.
        pinned_set = set(pinned)
        overflow = len(notes) - self.max_facts
        if overflow > 0:
            routine = [
                k for k in notes if ROUTINE_KEY_RE.match(k) and k not in pinned_set
            ]
            evict = routine[:overflow]
            if len(evict) < overflow:
                ordinary = [
                    k for k in notes if k not in pinned_set and k not in set(evict)
                ]
                evict += ordinary[: overflow - len(evict)]
            if len(evict) < overflow:  # everything left is pinned
                still_pinned = [k for k in notes if k not in set(evict)]
                evict += still_pinned[: overflow - len(evict)]
            for key in evict:
                del notes[key]

        text = "\n".join(f"[FACT] {k} = {v}" for k, v in notes.items())
        prompt_tokens = int(len(window + current_notes) * self.tokens_per_char)
        usage = Usage(
            input_tokens=prompt_tokens,
            output_tokens=int(len(text) * self.tokens_per_char),
            model_calls=1,
        )
        return text, usage


def default_summarizer(*, offline: bool = True) -> Summarizer:
    return FakeSummarizer() if offline else ModelSummarizer()


def default_note_writer(*, offline: bool = True) -> NoteWriter:
    return FakeNoteWriter() if offline else ModelNoteWriter()


__all__ = [
    "Summarizer",
    "NoteWriter",
    "ModelSummarizer",
    "ModelNoteWriter",
    "FakeSummarizer",
    "FakeNoteWriter",
    "extract_facts",
    "default_summarizer",
    "default_note_writer",
    "FACT_RE",
]
