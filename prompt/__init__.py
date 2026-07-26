"""Cache-aware prompt assembly for the Anthropic Messages API.

Prompt caching is a prefix match over the rendered request bytes, in the order
tools -> system -> messages. One changed byte at position N invalidates every
breakpoint at or after N, and nothing about that failure is reported: the
request succeeds and `cache_read_input_tokens` is simply 0.

This package makes that correct by construction -- blocks carry a stability
level, assembly refuses orderings that cannot cache, breakpoints are placed at
stability boundaries within the API's limits, a linter finds the byte that
diverged, and a diagnostic reads a response's usage and names the likely cause
of a miss.
"""

from .assembler import (
    CachePlan,
    PlacedBreakpoint,
    PromptAssembler,
    estimate_tokens,
)
from .blocks import MessageBlock, Stability, SystemBlock, ToolBlock
from .diagnostics import CacheDiagnosis, CacheStatus, MissReason, diagnose_usage
from .errors import (
    PromptCacheError,
    PromptOrderingError,
    PromptStructureError,
    SilentInvalidatorError,
)
from .linter import (
    PrefixDiff,
    assert_stable_prefix,
    diff_requests,
    find_silent_invalidator,
)
from .serialize import (
    RenderedPrefix,
    Span,
    canonical_bytes,
    canonical_json,
    render_prefix,
    stable_tool_order,
)

__all__ = [
    # block model
    "Stability",
    "ToolBlock",
    "SystemBlock",
    "MessageBlock",
    # assembly
    "PromptAssembler",
    "CachePlan",
    "PlacedBreakpoint",
    "estimate_tokens",
    # deterministic serialization
    "canonical_json",
    "canonical_bytes",
    "stable_tool_order",
    "render_prefix",
    "RenderedPrefix",
    "Span",
    # linting
    "diff_requests",
    "find_silent_invalidator",
    "assert_stable_prefix",
    "PrefixDiff",
    # diagnostics
    "diagnose_usage",
    "CacheDiagnosis",
    "CacheStatus",
    "MissReason",
    # errors
    "PromptCacheError",
    "PromptOrderingError",
    "PromptStructureError",
    "SilentInvalidatorError",
]
