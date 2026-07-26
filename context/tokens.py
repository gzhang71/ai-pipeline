"""Token counting for message histories.

Two implementations:

* `ApiTokenCounter` — exact, uses `common.client.count_tokens` (the
  `count_tokens` endpoint). Requires credentials. This is what you should use
  when running the bench for real.
* `HeuristicTokenCounter` — offline character-based approximation. The test
  suite and the default offline bench use this because there are no
  credentials in CI. It is *not* accurate; never use it to make a billing
  claim. It is monotonic in message size, which is all the strategies need in
  order to decide "am I over budget".

Do not reach for `tiktoken`: it is OpenAI's tokenizer and undercounts Claude
tokens, badly on code.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Protocol, runtime_checkable

from common.client import MODEL, count_tokens, has_credentials

Message = dict[str, Any]


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, messages: Iterable[Message]) -> int: ...


class HeuristicTokenCounter:
    """~4 characters per token, plus a fixed per-message and per-block cost.

    The per-block overhead matters: a strategy that turns one 2000-char tool
    result into ten 200-char ones has not actually saved anything, and a pure
    character count would say it had.
    """

    name = "heuristic"

    def __init__(
        self,
        chars_per_token: int = 4,
        per_message_overhead: int = 4,
        per_block_overhead: int = 3,
    ) -> None:
        self.chars_per_token = chars_per_token
        self.per_message_overhead = per_message_overhead
        self.per_block_overhead = per_block_overhead

    def count(self, messages: Iterable[Message]) -> int:
        total = 0
        for message in messages:
            total += self.per_message_overhead
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(content) // self.chars_per_token
                continue
            for block in content:
                total += self.per_block_overhead
                total += self._count_block(block)
        return total

    def _count_block(self, block: Any) -> int:
        if isinstance(block, str):
            return len(block) // self.chars_per_token
        if not isinstance(block, dict):  # SDK object
            block = getattr(block, "model_dump", lambda: {"repr": repr(block)})()
        chars = 0
        for key, value in block.items():
            if key == "type":
                continue
            if isinstance(value, str):
                chars += len(value)
            else:
                chars += len(json.dumps(value, default=str, sort_keys=True))
        return chars // self.chars_per_token


class ApiTokenCounter:
    """Exact counts via the count_tokens endpoint. Needs credentials."""

    name = "count_tokens_api"

    def __init__(self, *, model: str = MODEL, system: Any = None, tools: Any = None):
        self.model = model
        self.system = system
        self.tools = tools

    def count(self, messages: Iterable[Message]) -> int:
        return count_tokens(
            list(messages), system=self.system, tools=self.tools, model=self.model
        )


def default_counter() -> TokenCounter:
    """Exact counter when credentials exist, heuristic otherwise."""
    if has_credentials():
        return ApiTokenCounter()
    return HeuristicTokenCounter()
