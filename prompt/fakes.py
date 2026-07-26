"""Client-shaped fakes for offline tests.

There are no API credentials in this environment, so nothing here talks to the
network. These stand in for the *shapes* the SDK returns -- a `usage` object
with the four token fields, and a client whose `messages.create` records the
kwargs it was handed -- so tests can assert on real behaviour rather than on
mock call counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FakeUsage", "FakeMessage", "FakeMessages", "FakeClient", "FakeCacheServer"]


@dataclass
class FakeUsage:
    """Mirrors the fields `common.client.usage_breakdown` reads."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeMessage:
    usage: FakeUsage
    content: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"


class FakeCacheServer:
    """A tiny model of the API's prefix-match cache.

    It implements only the rules this package encodes: an entry is keyed by the
    exact rendered bytes up to a breakpoint, an entry is only readable once
    written, and a prefix below the model's minimum is accepted and then
    ignored. It is deliberately not a simulator of the real service -- it exists
    so a test can show that assembler output actually round-trips through
    prefix-match semantics.
    """

    def __init__(self, *, minimum_prefix_tokens: int = 1024) -> None:
        self.minimum_prefix_tokens = minimum_prefix_tokens
        self.entries: dict[bytes, int] = {}

    def submit(self, prefixes: list[tuple[bytes, int]], tail_tokens: int) -> FakeUsage:
        """`prefixes` is [(rendered bytes up to breakpoint, tokens), ...]."""
        read = 0
        written = 0
        accounted = 0
        for data, tokens in prefixes:
            if tokens < self.minimum_prefix_tokens:
                continue  # silently not cacheable
            if data in self.entries:
                read = max(read, tokens)
            else:
                self.entries[data] = tokens
                written = max(written, tokens)
        accounted = max(read, written)
        uncached = max(0, tail_tokens - accounted) if accounted else tail_tokens
        return FakeUsage(
            input_tokens=uncached,
            cache_creation_input_tokens=max(0, written - read),
            cache_read_input_tokens=read,
            output_tokens=7,
        )


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeMessage:
        _validate_request_shape(kwargs)
        self.calls.append(kwargs)
        return FakeMessage(usage=FakeUsage(input_tokens=12, output_tokens=3))


class FakeClient:
    """Stands in for `anthropic.Anthropic()` with no network and no key."""

    def __init__(self) -> None:
        self.messages = FakeMessages()


def _validate_request_shape(kwargs: dict[str, Any]) -> None:
    """Enforce the request rules a real 400 would catch."""
    for required in ("model", "max_tokens", "messages"):
        if required not in kwargs:
            raise TypeError(f"messages.create() missing required kwarg {required!r}")
    if not kwargs["messages"]:
        raise ValueError("messages must not be empty")
    if kwargs["messages"][0]["role"] != "user":
        raise ValueError("first message must have role 'user'")

    breakpoints = 0
    for tool in kwargs.get("tools") or []:
        breakpoints += "cache_control" in tool
    system = kwargs.get("system") or []
    if isinstance(system, list):
        for block in system:
            breakpoints += "cache_control" in block
    for message in kwargs["messages"]:
        content = message["content"]
        if isinstance(content, list):
            for block in content:
                breakpoints += "cache_control" in block
    if breakpoints > 4:
        raise ValueError(
            f"a request may carry at most 4 cache_control breakpoints; got {breakpoints}"
        )
