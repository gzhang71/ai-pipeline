"""The silent-invalidator linter: finding the byte that broke the cache.

Nothing in this file needs credentials. The linter's whole premise is that
byte-level non-determinism in your own builder is observable offline -- you do
not need the API to tell you that two renders differ.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from prompt import (
    PromptAssembler,
    SilentInvalidatorError,
    Stability,
    assert_stable_prefix,
    canonical_json,
    diff_requests,
    find_silent_invalidator,
    stable_tool_order,
)

CORPUS = "corpus paragraph. " * 60


def _assembler() -> PromptAssembler:
    return PromptAssembler(model="claude-opus-5", token_counter=len)


def test_linter_catches_an_injected_datetime_now():
    """The archetypal bug: a clock read inside a block tagged STATIC.

    Ordering validation cannot catch this -- the block claims to be STATIC and
    sits in a legal position. Only a byte diff of two renders finds it.
    """

    def build() -> dict:
        time.sleep(0.001)  # guarantee the microsecond field advances
        now = datetime.now(timezone.utc).isoformat()
        a = _assembler()
        a.add_system(f"You are an agent. Current time: {now}", Stability.STATIC)
        a.add_system(CORPUS, Stability.SESSION)
        a.add_message("user", "What changed?", Stability.TURN)
        return a.to_request_kwargs()

    # The mis-tagged prompt passes ordering validation...
    build()

    # ...and the linter still catches it.
    diff = find_silent_invalidator(build)
    assert not diff.identical
    assert not diff  # PrefixDiff is falsy when the prefix is unstable
    assert diff.span is not None
    assert diff.span.section == "system"
    assert diff.offset is not None and diff.offset > 0

    report = diff.describe()
    assert f"byte {diff.offset}" in report
    assert "system" in report
    assert "different cache entry on every request" in report


def test_assert_stable_prefix_raises_on_the_same_bug():
    def unstable() -> dict:
        time.sleep(0.001)
        a = _assembler()
        a.add_system(
            f"boot={datetime.now(timezone.utc).isoformat()}", Stability.STATIC
        )
        a.add_message("user", "hello", Stability.TURN)
        return a.to_request_kwargs()

    with pytest.raises(SilentInvalidatorError) as excinfo:
        assert_stable_prefix(unstable)
    assert "prefix diverges at byte" in str(excinfo.value)


def test_assert_stable_prefix_passes_on_a_deterministic_builder():
    def stable() -> dict:
        a = _assembler()
        a.add_system("Frozen persona.", Stability.STATIC)
        a.add_system(CORPUS, Stability.SESSION)
        a.add_message("user", "What changed?", Stability.TURN)
        return a.to_request_kwargs()

    assert_stable_prefix(stable, runs=3)
    diff = find_silent_invalidator(stable, runs=3)
    assert diff.identical
    assert diff  # truthy when stable
    assert "byte-identical" in diff.describe()


def test_linter_catches_unsorted_json_dumps_in_the_prefix():
    """Same mapping, different insertion order, different bytes."""
    orders = iter([("alpha", "beta", "gamma"), ("gamma", "alpha", "beta")])

    def build_with(serializer) -> dict:
        keys = next(orders)
        payload = {key: f"value for {key}" for key in keys}
        a = _assembler()
        a.add_system(f"Configuration: {serializer(payload)}", Stability.STATIC)
        a.add_message("user", "go", Stability.TURN)
        return a.to_request_kwargs()

    naive = [build_with(json.dumps), build_with(json.dumps)]
    diff = diff_requests(*naive)
    assert not diff.identical
    assert diff.span is not None and diff.span.section == "system"

    orders = iter([("alpha", "beta", "gamma"), ("gamma", "alpha", "beta")])
    fixed = [build_with(canonical_json), build_with(canonical_json)]
    assert diff_requests(*fixed).identical


def test_varying_tool_order_is_caught_and_stable_tool_order_fixes_it():
    """A tool set that reorders invalidates the entire request from byte 0."""
    schema = {"type": "object", "properties": {}}
    tools_a = [
        {"name": "search", "description": "search", "input_schema": schema},
        {"name": "answer", "description": "answer", "input_schema": schema},
    ]
    tools_b = list(reversed(tools_a))
    base = {
        "model": "claude-opus-5",
        "system": [{"type": "text", "text": CORPUS}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }

    unsorted_diff = diff_requests({**base, "tools": tools_a}, {**base, "tools": tools_b})
    assert not unsorted_diff.identical
    assert unsorted_diff.span is not None
    assert unsorted_diff.span.section == "tools"
    # Tools render first, so almost nothing survives.
    assert unsorted_diff.cached_prefix_bytes < 120

    sorted_diff = diff_requests(
        {**base, "tools": stable_tool_order(tools_a)},
        {**base, "tools": stable_tool_order(tools_b)},
    )
    assert sorted_diff.identical

    # The assembler sorts tools by name for exactly this reason.
    def build(order):
        a = _assembler()
        for tool in order:
            a.add_tool(tool["name"], tool["description"], tool["input_schema"])
        a.add_message("user", "hi", Stability.TURN)
        return a.to_request_kwargs()

    assert diff_requests(build(tools_a), build(tools_b)).identical


def test_divergence_late_in_the_prompt_leaves_the_prefix_reusable():
    def build(question: str) -> dict:
        a = _assembler()
        a.add_system(CORPUS, Stability.STATIC)
        a.add_message("user", question, Stability.TURN)
        return a.to_request_kwargs()

    diff = diff_requests(build("first question"), build("second question"))
    assert not diff.identical
    assert diff.span is not None
    assert diff.span.section == "messages"
    # Everything before the volatile turn is still a valid cache prefix.
    assert diff.cached_prefix_bytes > len(CORPUS)


def test_find_silent_invalidator_requires_two_runs():
    with pytest.raises(ValueError, match="at least 2"):
        find_silent_invalidator(lambda: {"messages": []}, runs=1)
