"""Shared Anthropic client construction and model constants.

Every subproject imports from here so that model IDs, token counting, and
client construction stay consistent across the repo.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Iterable

import anthropic

# Default model for every subproject. Override per call site if a cheaper
# model is genuinely warranted -- do not change this constant.
MODEL = "claude-opus-5"

# Minimum cacheable prefix, in tokens, keyed by model. A prefix shorter than
# this silently will not cache: no error, just cache_creation_input_tokens == 0.
# Note this is NOT monotonic across generations.
MIN_CACHEABLE_PREFIX_TOKENS = {
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-haiku-4-5": 4096,
}

# The API renders a request in this order. Caching is a prefix match over the
# rendered bytes, so anything earlier in this order invalidates everything
# after it when it changes.
RENDER_ORDER = ("tools", "system", "messages")

# Hard limit on cache_control breakpoints in a single request.
MAX_CACHE_BREAKPOINTS = 4

# A cache breakpoint walks back at most this many content blocks looking for a
# prior entry. Long agentic turns that add more blocks than this between
# requests will silently miss.
CACHE_LOOKBACK_BLOCKS = 20


@functools.lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Return a process-wide Anthropic client.

    Credentials resolve from the environment: ANTHROPIC_API_KEY, then
    ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile. A bare client
    works in all three cases, so do not pass an explicit key.
    """
    return anthropic.Anthropic()


@functools.lru_cache(maxsize=1)
def get_async_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


def has_credentials() -> bool:
    """True if a live API call is plausible.

    Tests that would hit the network should skip when this is False.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = os.environ.get(
        "ANTHROPIC_CONFIG_DIR", os.path.expanduser("~/.config/anthropic")
    )
    return os.path.isdir(os.path.join(config_dir, "credentials"))


def count_tokens(
    messages: Iterable[dict[str, Any]],
    *,
    system: Any = None,
    tools: Any = None,
    model: str = MODEL,
) -> int:
    """Exact token count for a prompt, via the count_tokens endpoint.

    Counts are model-specific -- pass the same model you will use for
    inference. Never estimate with tiktoken; it is OpenAI's tokenizer and
    undercounts Claude tokens badly, especially on code.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": list(messages)}
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools
    return get_client().messages.count_tokens(**kwargs).input_tokens


def usage_breakdown(usage: Any) -> dict[str, int]:
    """Normalize a response `usage` object into a plain dict.

    `input_tokens` is the *uncached remainder* only. Total prompt size is
    input + cache_creation + cache_read -- reading input_tokens alone will
    badly understate a cache-warm request.
    """
    creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    uncached = getattr(usage, "input_tokens", 0) or 0
    return {
        "input_tokens": uncached,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": read,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "total_prompt_tokens": uncached + creation + read,
    }
