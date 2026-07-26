"""Content hashing helpers.

Everything the harness attributes a run to -- prompt bytes, task bytes, model
output -- is hashed the same way: sha256 over UTF-8 bytes, hex encoded. Hashes
are stored in full in run records and displayed truncated.
"""

from __future__ import annotations

import hashlib
from typing import Any

SHORT_LEN = 12


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


def short(digest: str, length: int = SHORT_LEN) -> str:
    """Truncate a digest for display. Never use this as a record key."""
    return digest[:length]


def hash_output(text: str, tool_calls: Any = ()) -> str:
    """Stable hash of a model output, used as part of the judge cache key.

    Tool calls participate because two outputs with identical text but
    different tool calls are not the same output.
    """
    parts = [text]
    for call in tool_calls:
        name = getattr(call, "name", None)
        payload = getattr(call, "input", None)
        if name is None and isinstance(call, dict):
            name, payload = call.get("name"), call.get("input")
        parts.append(f"\x00{name}\x00{_canonical(payload)}")
    return hash_text("".join(parts))


def _canonical(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
