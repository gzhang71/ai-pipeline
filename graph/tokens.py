"""Token accounting for the head-to-head comparison.

Honest accounting is the whole point of the experiment, so the ledger counts
*everything that entered the context window on every request*: the tool schemas,
the system prompt, the question, and every tool result that came back. For a
multi-turn agent that means each request is counted in full, because the model
really does re-read the whole transcript on every turn -- reporting only the
first prompt would flatter JIT retrieval by an order of magnitude.

Two counters:

* `LiveTokenCounter` calls `common.client.count_tokens`, i.e. the real
  `/v1/messages/count_tokens` endpoint for the exact model being used. Used
  whenever credentials exist.
* `OfflineTokenCounter` is a deterministic character-per-token approximation,
  used when there are no credentials (tests, CI). It is *not* a substitute for
  the real tokenizer and is labelled as an estimate everywhere it surfaces.
  Both arms of the comparison always share one counter instance, so the ratio
  between them stays meaningful even when the absolute numbers are approximate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from common.client import MODEL, count_tokens, has_credentials

# Claude tokenizes code at roughly 3.4 characters per token. This is only used
# offline; never use it to make a costing claim.
OFFLINE_CHARS_PER_TOKEN = 3.4
OFFLINE_BLOCK_OVERHEAD = 4


class TokenCounter(Protocol):
    exact: bool

    def count(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Any = None,
        tools: Any = None,
    ) -> int: ...


def _flatten(value: Any) -> tuple[str, int]:
    """Return (concatenated text, number of content blocks) for any payload."""
    if value is None:
        return "", 0
    if isinstance(value, str):
        return value, 1
    if isinstance(value, dict):
        text_parts: list[str] = []
        blocks = 1
        for key, item in value.items():
            sub_text, sub_blocks = _flatten(item)
            text_parts.append(str(key))
            text_parts.append(sub_text)
            blocks += sub_blocks
        return " ".join(text_parts), blocks
    if isinstance(value, (list, tuple)):
        text_parts = []
        blocks = 0
        for item in value:
            sub_text, sub_blocks = _flatten(item)
            text_parts.append(sub_text)
            blocks += sub_blocks
        return " ".join(text_parts), blocks
    return json.dumps(value, default=str), 1


class OfflineTokenCounter:
    """Deterministic approximation used when no credentials are available."""

    exact = False

    def __init__(self, chars_per_token: float = OFFLINE_CHARS_PER_TOKEN) -> None:
        self.chars_per_token = chars_per_token

    def count(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Any = None,
        tools: Any = None,
    ) -> int:
        total_chars = 0
        total_blocks = 0
        for payload in (tools, system, messages):
            text, blocks = _flatten(payload)
            total_chars += len(text)
            total_blocks += blocks
        return int(total_chars / self.chars_per_token) + OFFLINE_BLOCK_OVERHEAD * total_blocks


class LiveTokenCounter:
    """Exact counts from the count_tokens endpoint, for the model in use."""

    exact = True

    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    def count(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Any = None,
        tools: Any = None,
    ) -> int:
        return count_tokens(messages, system=system, tools=tools, model=self.model)


def default_counter() -> TokenCounter:
    """Live counts when credentials exist, deterministic estimates otherwise."""
    return LiveTokenCounter() if has_credentials() else OfflineTokenCounter()


@dataclass
class TokenLedger:
    """Every request's prompt size, so nothing gets quietly left out."""

    counter: TokenCounter
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        label: str,
        messages: list[dict[str, Any]],
        *,
        system: Any = None,
        tools: Any = None,
    ) -> int:
        tokens = self.counter.count(messages, system=system, tools=tools)
        self.entries.append({"label": label, "prompt_tokens": tokens})
        return tokens

    @property
    def total_prompt_tokens(self) -> int:
        """Sum over every request -- what the model actually read, in total."""
        return sum(e["prompt_tokens"] for e in self.entries)

    @property
    def peak_prompt_tokens(self) -> int:
        """Largest single request -- how much context was needed at once."""
        return max((e["prompt_tokens"] for e in self.entries), default=0)

    @property
    def requests(self) -> int:
        return len(self.entries)

    @property
    def exact(self) -> bool:
        return bool(getattr(self.counter, "exact", False))
