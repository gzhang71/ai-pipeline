"""Message-shape validation and repair.

A history-management strategy that produces a 400 is worse than useless: it
turns a cost problem into an outage. Every strategy in this package runs its
output through `sanitize()` before returning it, and the bench runs
`assert_valid()` on the request it is about to send, on every single turn.

Rules enforced (the ones the Messages API actually rejects):

* the history is non-empty and the first message is `user`
* roles are `user` / `assistant` / `system`; a `system` message may not be
  first (mid-conversation system messages are an operator channel, not a
  system prompt)
* the last message is `user` — a trailing assistant turn is a prefill, which
  is rejected on Claude Opus 5
* `tool_use` blocks appear only in assistant messages, `tool_result` blocks
  only in user messages
* every `tool_use` has exactly one matching `tool_result` in a *later* user
  message, and every `tool_result` refers to an *earlier* `tool_use`
* `tool_use` ids are unique
* `thinking` blocks appear only in assistant messages and precede every
  non-thinking block in that message
* no empty content, no empty text blocks

Unknown block types (e.g. the server-side `compaction` block) are passed
through untouched — we never rewrite what we do not understand.
"""

from __future__ import annotations

from typing import Any, Iterable

Message = dict[str, Any]
Block = dict[str, Any]

_ROLES = {"user", "assistant", "system"}
_THINKING_TYPES = {"thinking", "redacted_thinking"}


class InvalidMessageShape(ValueError):
    """Raised by `assert_valid` when a history would be rejected by the API."""


def to_block(block: Any) -> Block:
    """Normalize an SDK content block (or string) into a plain dict."""
    if isinstance(block, dict):
        return block
    if isinstance(block, str):
        return {"type": "text", "text": block}
    dump = getattr(block, "model_dump", None)
    if dump is not None:
        return dump()
    return {"type": getattr(block, "type", "unknown")}  # pragma: no cover


def blocks(message: Message) -> list[Block]:
    """Content of a message as a list of block dicts."""
    content = message.get("content", "")
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [to_block(b) for b in content]


def normalize(messages: Iterable[Message]) -> list[Message]:
    """Copy a history with every message's content as a list of block dicts."""
    return [
        {**m, "content": blocks(m)}
        for m in messages
    ]


def message_text(message: Message) -> str:
    """All human-readable text in a message, including tool_result payloads."""
    parts: list[str] = []
    for block in blocks(message):
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "tool_result":
            content = block.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for sub in content:
                    sub = to_block(sub)
                    if sub.get("type") == "text":
                        parts.append(str(sub.get("text", "")))
        elif btype == "tool_use":
            # Render the input's string values, not `str(dict)` — repr quoting
            # and braces bleed into anything that reads this text back.
            payload = block.get("input")
            if isinstance(payload, dict):
                parts.extend(str(v) for v in payload.values() if isinstance(v, str))
            elif payload is not None:
                parts.append(str(payload))
        elif btype in _THINKING_TYPES:
            parts.append(str(block.get("thinking", "")))
        elif btype == "compaction":
            parts.append(str(block.get("summary", "")))
    return "\n".join(parts)


def history_text(messages: Iterable[Message]) -> str:
    return "\n".join(message_text(m) for m in messages)


def validate(messages: Iterable[Message], *, require_user_last: bool = True) -> list[str]:
    """Return a list of problems. Empty list means the history is legal."""
    messages = list(messages)
    problems: list[str] = []

    if not messages:
        return ["history is empty"]

    if messages[0].get("role") != "user":
        problems.append(f"first message must be 'user', got {messages[0].get('role')!r}")

    if require_user_last and messages[-1].get("role") == "assistant":
        problems.append(
            "last message is 'assistant' (assistant prefill is rejected on Claude Opus 5)"
        )

    seen_tool_use: dict[str, int] = {}
    answered: dict[str, int] = {}

    for i, message in enumerate(messages):
        role = message.get("role")
        if role not in _ROLES:
            problems.append(f"message {i}: unknown role {role!r}")
        if role == "system" and i == 0:
            problems.append("message 0: a 'system' role message cannot be first")

        content = message.get("content")
        if content is None or (isinstance(content, (str, list)) and len(content) == 0):
            problems.append(f"message {i}: empty content")
            continue

        message_blocks = blocks(message)
        seen_non_thinking = False
        for j, block in enumerate(message_blocks):
            btype = block.get("type")
            if not btype:
                problems.append(f"message {i} block {j}: missing 'type'")
                continue

            if btype in _THINKING_TYPES:
                if role != "assistant":
                    problems.append(
                        f"message {i} block {j}: {btype} block in a {role!r} message"
                    )
                if seen_non_thinking:
                    problems.append(
                        f"message {i} block {j}: thinking block must precede all "
                        "non-thinking blocks in the message"
                    )
                continue
            seen_non_thinking = True

            if btype == "text" and not str(block.get("text", "")).strip():
                problems.append(f"message {i} block {j}: empty text block")

            if btype == "tool_use":
                if role != "assistant":
                    problems.append(
                        f"message {i} block {j}: tool_use block in a {role!r} message"
                    )
                tid = block.get("id")
                if not tid:
                    problems.append(f"message {i} block {j}: tool_use without an id")
                elif tid in seen_tool_use:
                    problems.append(
                        f"message {i} block {j}: duplicate tool_use id {tid!r}"
                    )
                else:
                    seen_tool_use[tid] = i

            if btype == "tool_result":
                if role != "user":
                    problems.append(
                        f"message {i} block {j}: tool_result block in a {role!r} message"
                    )
                tid = block.get("tool_use_id")
                if not tid:
                    problems.append(
                        f"message {i} block {j}: tool_result without a tool_use_id"
                    )
                elif tid not in seen_tool_use:
                    problems.append(
                        f"message {i} block {j}: tool_result {tid!r} has no preceding "
                        "tool_use (orphaned by a history edit)"
                    )
                elif tid in answered:
                    problems.append(
                        f"message {i} block {j}: duplicate tool_result for {tid!r}"
                    )
                else:
                    answered[tid] = i

    for tid, index in seen_tool_use.items():
        if tid not in answered:
            problems.append(
                f"message {index}: tool_use {tid!r} has no matching tool_result"
            )

    return problems


def is_valid(messages: Iterable[Message], **kwargs: Any) -> bool:
    return not validate(messages, **kwargs)


def assert_valid(messages: Iterable[Message], *, label: str = "history", **kwargs: Any) -> None:
    problems = validate(messages, **kwargs)
    if problems:
        raise InvalidMessageShape(
            f"{label} would be rejected by the API:\n  - " + "\n  - ".join(problems)
        )


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def sanitize(messages: Iterable[Message], *, drop_leading_assistant: bool = True) -> list[Message]:
    """Make an edited history legal again, losing as little as possible.

    Strategies slice histories; slicing breaks tool_use/tool_result pairs. This
    repairs the three damage modes a slice can cause:

    1. a `tool_result` whose `tool_use` was cut away -> drop the block
    2. a `tool_use` whose `tool_result` was cut away -> drop the block
    3. the history now starts with an assistant message -> drop leading
       assistant messages

    Messages left with no content are dropped.
    """
    working = normalize(messages)

    # Pass 1: drop tool_results with no preceding tool_use.
    known_ids: set[str] = set()
    for message in working:
        if message["role"] == "assistant":
            for block in message["content"]:
                if block.get("type") == "tool_use" and block.get("id"):
                    known_ids.add(block["id"])
        elif message["role"] == "user":
            message["content"] = [
                b
                for b in message["content"]
                if b.get("type") != "tool_result" or b.get("tool_use_id") in known_ids
            ]

    # Pass 2: drop tool_uses that are never answered.
    answered: set[str] = {
        b["tool_use_id"]
        for m in working
        if m["role"] == "user"
        for b in m["content"]
        if b.get("type") == "tool_result" and b.get("tool_use_id")
    }
    for message in working:
        if message["role"] != "assistant":
            continue
        message["content"] = [
            b
            for b in message["content"]
            if b.get("type") != "tool_use" or b.get("id") in answered
        ]

    # Pass 3: drop empty messages, and messages that are now thinking-only
    # (a thinking block whose tool_use was removed carries no answer).
    cleaned: list[Message] = []
    for message in working:
        content = [
            b
            for b in message["content"]
            if not (b.get("type") == "text" and not str(b.get("text", "")).strip())
        ]
        if not content:
            continue
        if all(b.get("type") in _THINKING_TYPES for b in content):
            continue
        cleaned.append({**message, "content": content})

    if drop_leading_assistant:
        while cleaned and cleaned[0]["role"] != "user":
            cleaned.pop(0)

    return cleaned


def find_tail_start(
    messages: list[Message], desired: int, *, prefer: str = "backward"
) -> int:
    """Nearest index >= 0 where a tail may begin without orphaning a pair.

    A legal tail starts at a `user` message that contains no `tool_result`
    blocks. `prefer="backward"` keeps more history (used by the summarizing
    strategies, which would otherwise have to summarize a half-pair);
    `prefer="forward"` drops more (used by tail truncation, where dropping is
    the entire point).
    """
    if desired <= 0:
        return 0
    desired = min(desired, len(messages))

    def ok(i: int) -> bool:
        if i >= len(messages):
            return False
        message = messages[i]
        if message.get("role") != "user":
            return False
        return all(b.get("type") != "tool_result" for b in blocks(message))

    order = (
        list(range(desired, -1, -1)) + list(range(desired + 1, len(messages)))
        if prefer == "backward"
        else list(range(desired, len(messages))) + list(range(desired - 1, -1, -1))
    )
    for i in order:
        if i == 0 or ok(i):
            return i
    return len(messages)
