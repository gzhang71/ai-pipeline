"""Message-shape validation and repair.

A strategy that emits a 400 turns a cost problem into an outage, so the
validator is the load-bearing safety net for everything else in the package.
"""

from __future__ import annotations

import pytest

from context.validation import (
    InvalidMessageShape,
    assert_valid,
    blocks,
    find_tail_start,
    history_text,
    is_valid,
    message_text,
    normalize,
    sanitize,
    validate,
)


def user(*content):
    return {"role": "user", "content": list(content)}


def assistant(*content):
    return {"role": "assistant", "content": list(content)}


def text(t):
    return {"type": "text", "text": t}


def tool_use(tid, name="read_file", **inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp or {"path": "/x"}}


def tool_result(tid, content="ok"):
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


GOOD = [
    user(text("do the thing")),
    assistant(tool_use("t1")),
    user(tool_result("t1", "file contents")),
    assistant(text("done")),
    user(text("and now?")),
]


class TestValidate:
    def test_accepts_a_well_formed_history(self):
        assert validate(GOOD) == []
        assert is_valid(GOOD)

    def test_rejects_empty(self):
        assert validate([]) == ["history is empty"]

    def test_first_message_must_be_user(self):
        problems = validate([assistant(text("hi")), user(text("hello"))])
        assert any("first message must be 'user'" in p for p in problems)

    def test_trailing_assistant_is_a_prefill(self):
        problems = validate([user(text("hi")), assistant(text("sure"))])
        assert any("prefill" in p for p in problems)
        # ...but it is legal mid-loop when we are not about to send it.
        assert validate([user(text("hi")), assistant(text("sure"))], require_user_last=False) == []

    def test_orphaned_tool_result_is_caught(self):
        problems = validate([user(tool_result("gone")), user(text("hi"))])
        assert any("no preceding tool_use" in p for p in problems)

    def test_unanswered_tool_use_is_caught(self):
        problems = validate([user(text("hi")), assistant(tool_use("t1")), user(text("?"))])
        assert any("no matching tool_result" in p for p in problems)

    def test_duplicate_tool_use_id(self):
        history = [
            user(text("hi")),
            assistant(tool_use("t1"), tool_use("t1")),
            user(tool_result("t1")),
        ]
        assert any("duplicate tool_use id" in p for p in validate(history))

    def test_duplicate_tool_result(self):
        history = [
            user(text("hi")),
            assistant(tool_use("t1")),
            user(tool_result("t1"), tool_result("t1")),
        ]
        assert any("duplicate tool_result" in p for p in validate(history))

    def test_tool_blocks_in_the_wrong_role(self):
        problems = validate([user(tool_use("t1")), user(tool_result("t1"))])
        assert any("tool_use block in a 'user' message" in p for p in problems)

    def test_thinking_must_precede_other_blocks(self):
        history = [
            user(text("hi")),
            assistant(text("answer"), {"type": "thinking", "thinking": "hmm"}),
            user(text("ok")),
        ]
        assert any("must precede" in p for p in validate(history))

    def test_thinking_first_is_fine(self):
        history = [
            user(text("hi")),
            assistant({"type": "thinking", "thinking": "hmm"}, text("answer")),
            user(text("ok")),
        ]
        assert validate(history) == []

    def test_empty_text_block(self):
        assert any("empty text" in p for p in validate([user(text("   ")), user(text("x"))]))

    def test_empty_content(self):
        assert any("empty content" in p for p in validate([user(), user(text("x"))]))

    def test_system_message_may_not_be_first(self):
        history = [{"role": "system", "content": "op note"}, user(text("hi"))]
        assert any("cannot be first" in p for p in validate(history))

    def test_mid_conversation_system_message_is_allowed(self):
        history = [
            user(text("hi")),
            assistant(text("hello")),
            user(text("go")),
            {"role": "system", "content": "terse mode"},
        ]
        assert validate(history, require_user_last=False) == []

    def test_unknown_block_types_pass_through(self):
        """We must not reject the server's own compaction block."""
        history = [
            user(text("hi")),
            assistant(text("ok"), {"type": "compaction", "summary": "earlier stuff"}),
            user(text("next")),
        ]
        assert validate(history) == []

    def test_assert_valid_raises_with_detail(self):
        with pytest.raises(InvalidMessageShape) as excinfo:
            assert_valid([user(tool_result("gone"))], label="my-strategy")
        assert "my-strategy" in str(excinfo.value)
        assert "no preceding tool_use" in str(excinfo.value)


class TestSanitize:
    def test_repairs_a_head_slice(self):
        """Slicing off the head orphans the tool_result that survived."""
        sliced = GOOD[2:]
        assert not is_valid(sliced)
        repaired = sanitize(sliced)
        assert is_valid(repaired)
        assert repaired[0]["role"] == "user"

    def test_repairs_a_tail_slice(self):
        """Slicing off the tail leaves a tool_use nobody answered."""
        sliced = GOOD[:2] + [user(text("carry on"))]
        assert not is_valid(sliced)
        repaired = sanitize(sliced)
        assert is_valid(repaired)
        assert not any(
            b.get("type") == "tool_use" for m in repaired for b in blocks(m)
        )

    def test_keeps_intact_pairs(self):
        repaired = sanitize(GOOD)
        assert is_valid(repaired)
        assert len(repaired) == len(GOOD)

    def test_drops_thinking_only_messages(self):
        history = [
            user(text("hi")),
            assistant({"type": "thinking", "thinking": "hmm"}, tool_use("t1")),
            user(text("no result for t1")),
        ]
        repaired = sanitize(history)
        assert is_valid(repaired)
        assert all(m["role"] == "user" for m in repaired)

    def test_idempotent(self):
        once = sanitize(GOOD[2:])
        assert sanitize(once) == once


class TestHelpers:
    def test_normalize_expands_string_content(self):
        assert normalize([{"role": "user", "content": "hi"}])[0]["content"] == [
            {"type": "text", "text": "hi"}
        ]

    def test_message_text_includes_tool_payloads(self):
        combined = message_text(user(tool_result("t1", "the payload")))
        assert "the payload" in combined

    def test_message_text_does_not_leak_dict_repr(self):
        """A regression: `str(dict)` bled quotes and braces into fact values."""
        rendered = message_text(assistant(tool_use("t1", "write_note", note="[FACT] K = v")))
        assert rendered == "[FACT] K = v"
        assert "'" not in rendered and "}" not in rendered

    def test_history_text_spans_messages(self):
        combined = history_text(GOOD)
        assert "do the thing" in combined and "file contents" in combined

    def test_find_tail_start_avoids_splitting_a_pair(self):
        # index 2 is the user message holding the tool_result for t1.
        assert find_tail_start(GOOD, 2, prefer="backward") == 0
        assert find_tail_start(GOOD, 2, prefer="forward") == 4

    def test_find_tail_start_clamps(self):
        assert find_tail_start(GOOD, -5) == 0
        assert find_tail_start(GOOD, 99, prefer="backward") <= len(GOOD)
