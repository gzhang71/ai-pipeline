"""Token/usage accounting shared by strategies and the benchmark runner.

The single rule this module exists to enforce: *total prompt tokens are
`input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.*
Reading `input_tokens` alone understates a cache-warm request badly, and every
comparison in the bench would silently favour whichever strategy happened to
hit the cache more often.

The second rule: a strategy that calls a model to do its compaction (a
summarizer, a note writer) pays for those tokens. `Usage.__add__` is how that
cost gets folded into the strategy's total instead of disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.client import usage_breakdown


@dataclass(frozen=True)
class Usage:
    """Accumulated token spend, including the model calls a strategy makes."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0

    @property
    def total_prompt_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):  # pragma: no cover - defensive
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            output_tokens=self.output_tokens + other.output_tokens,
            model_calls=self.model_calls + other.model_calls,
        )

    __radd__ = __add__

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "output_tokens": self.output_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_tokens": self.total_tokens,
            "model_calls": self.model_calls,
        }

    @classmethod
    def from_response_usage(cls, usage: Any, *, model_calls: int = 1) -> "Usage":
        """Build a `Usage` from an SDK (or fake) `response.usage` object."""
        parts = usage_breakdown(usage)
        return cls(
            input_tokens=parts["input_tokens"],
            cache_creation_input_tokens=parts["cache_creation_input_tokens"],
            cache_read_input_tokens=parts["cache_read_input_tokens"],
            output_tokens=parts["output_tokens"],
            model_calls=model_calls,
        )


ZERO_USAGE = Usage()
