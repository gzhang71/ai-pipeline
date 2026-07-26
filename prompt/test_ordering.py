"""Ordering validation: the check that catches the bug at construction time."""

from __future__ import annotations

import pytest

from prompt import (
    PromptAssembler,
    PromptOrderingError,
    PromptStructureError,
    Stability,
)


def _base() -> PromptAssembler:
    return PromptAssembler(model="claude-opus-5")


def test_stability_ordered_prompt_is_accepted():
    a = _base()
    a.add_tool("search", "Search the corpus.", {"type": "object", "properties": {}})
    a.add_system("You are a support agent.", Stability.STATIC)
    a.add_system("Account tier: enterprise.", Stability.SESSION)
    a.add_message("user", "What is my quota?", Stability.TURN)

    assert a.validate() is a
    assert [b.stability for b in a.blocks] == sorted(b.stability for b in a.blocks)


def test_timestamp_in_system_header_is_rejected_and_named():
    """The canonical failure: a per-request value ahead of the frozen prompt."""
    a = _base()
    a.add_system("Current time: 2026-07-25T18:00:00Z", Stability.TURN, label="clock")
    a.add_system("You are a support agent.", Stability.STATIC, label="persona")
    a.add_message("user", "hello")

    with pytest.raises(PromptOrderingError) as excinfo:
        a.validate()

    message = str(excinfo.value)
    assert "'persona'" in message, message
    assert "'clock'" in message, message
    assert "STATIC" in message and "TURN" in message
    assert "tools -> system -> messages" in message
    # The message must say what to do, not just that something is wrong.
    assert "Fix:" in message


def test_volatile_tool_before_static_system_is_rejected():
    """Tools render at byte 0, so a volatile tool poisons everything after it."""
    a = _base()
    a.add_tool(
        "a_dynamic",
        "Varies per user.",
        {"type": "object", "properties": {}},
        Stability.TURN,
    )
    a.add_system("Frozen persona.", Stability.STATIC)
    a.add_message("user", "hello")

    with pytest.raises(PromptOrderingError) as excinfo:
        a.validate()
    assert "tool:a_dynamic" in str(excinfo.value)
    assert "'tools'" in str(excinfo.value)


def test_session_block_after_turn_block_in_messages_is_rejected():
    a = _base()
    a.add_message("user", "new question", Stability.TURN, label="question")
    a.add_message("assistant", "old answer", Stability.SESSION, label="history")

    with pytest.raises(PromptOrderingError) as excinfo:
        a.validate()
    assert "'history'" in str(excinfo.value)


def test_plan_and_emit_run_validation():
    a = _base()
    a.add_system("volatile", Stability.TURN)
    a.add_system("frozen", Stability.STATIC)
    a.add_message("user", "hi")

    with pytest.raises(PromptOrderingError):
        a.plan()
    with pytest.raises(PromptOrderingError):
        a.to_request_kwargs()


def test_settle_history_makes_an_agentic_loop_legal():
    """Last turn's TURN blocks must be demoted before the next turn is added."""
    a = _base()
    a.add_system("Frozen persona.", Stability.STATIC)
    a.add_message("user", "turn one")
    a.add_message("assistant", "answer one")

    a.settle_history()
    a.add_message("user", "turn two")

    a.validate()
    stabilities = [b.stability for b in a.blocks]
    assert stabilities == [
        Stability.STATIC,
        Stability.SESSION,
        Stability.SESSION,
        Stability.TURN,
    ]


def test_settle_history_preserves_checkpoints():
    a = _base()
    a.add_message("user", "turn one", checkpoint=True)
    a.settle_history()
    assert a.blocks[0].checkpoint is True
    assert a.blocks[0].stability is Stability.SESSION


def test_structural_rules_are_enforced():
    empty = _base()
    empty.add_system("frozen")
    with pytest.raises(PromptStructureError, match="at least one message"):
        empty.validate()

    assistant_first = _base()
    assistant_first.add_message("assistant", "I go first?")
    with pytest.raises(PromptStructureError, match="role 'user'"):
        assistant_first.validate()

    with pytest.raises(PromptStructureError, match="unsupported message role"):
        _base().add_message("moderator", "nope")


def test_manual_cache_control_is_refused():
    a = _base()
    a.add_message(
        "user",
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
    )
    with pytest.raises(PromptStructureError, match="cache_control"):
        a.to_request_kwargs()


def test_invalid_ttl_and_breakpoint_limits_are_refused():
    with pytest.raises(PromptStructureError, match="ttl"):
        PromptAssembler(ttl="30m")
    with pytest.raises(PromptStructureError, match="max_breakpoints"):
        PromptAssembler(max_breakpoints=9)
