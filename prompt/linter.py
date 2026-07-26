"""Find the byte that silently broke your cache.

The failure mode this targets produces no error: the request succeeds, the
answer is fine, and `cache_read_input_tokens` is just always 0. The cause is
almost always one byte of per-request entropy sitting in the prefix -- a
`datetime.now()` in the system header, an unsorted `json.dumps`, a tool list
that comes out of a dict in a different order.

Give the linter two assembled requests (or a callable that assembles one, run
twice) and it reports the first divergent byte offset and which block owns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .errors import SilentInvalidatorError
from .serialize import RenderedPrefix, Span, render_prefix

__all__ = [
    "PrefixDiff",
    "diff_requests",
    "find_silent_invalidator",
    "assert_stable_prefix",
]

_WINDOW = 48


@dataclass(frozen=True)
class PrefixDiff:
    """Where two renders of the same prompt stopped agreeing."""

    identical: bool
    offset: int | None = None
    span: Span | None = None
    left: str = ""
    right: str = ""
    left_length: int = 0
    right_length: int = 0

    def __bool__(self) -> bool:
        """True when the prefixes match, i.e. the prompt is cache-stable."""
        return self.identical

    @property
    def cached_prefix_bytes(self) -> int:
        """Bytes that would still have been reusable before the divergence."""
        return self.offset if self.offset is not None else self.left_length

    def describe(self) -> str:
        if self.identical:
            return (
                f"prefixes are byte-identical ({self.left_length} bytes); no "
                "silent invalidator found in the rendered prefix"
            )
        label = self.span.label if self.span else "<past end of prefix>"
        section = self.span.section if self.span else "?"
        return (
            f"prefix diverges at byte {self.offset} of {self.left_length}, in "
            f"section '{section}', block {label!r}.\n"
            f"  A: {self.left!r}\n"
            f"  B: {self.right!r}\n"
            f"Everything from byte {self.offset} onward is a different cache "
            "entry on every request. Make that block deterministic, or move it "
            "after the last cache breakpoint."
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.describe()


def _first_difference(a: bytes, b: bytes) -> int | None:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return limit
    return None


def _snippet(data: bytes, offset: int) -> str:
    start = max(0, offset - _WINDOW)
    end = min(len(data), offset + _WINDOW)
    return data[start:end].decode("utf-8", errors="replace")


def _diff_rendered(left: RenderedPrefix, right: RenderedPrefix) -> PrefixDiff:
    offset = _first_difference(left.data, right.data)
    if offset is None:
        return PrefixDiff(
            identical=True,
            left_length=len(left),
            right_length=len(right),
        )
    return PrefixDiff(
        identical=False,
        offset=offset,
        span=left.locate(offset) or right.locate(offset),
        left=_snippet(left.data, offset),
        right=_snippet(right.data, offset),
        left_length=len(left),
        right_length=len(right),
    )


def diff_requests(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> PrefixDiff:
    """Diff two `messages.create` kwargs dicts at the byte level."""
    return _diff_rendered(render_prefix(left), render_prefix(right))


def find_silent_invalidator(
    builder: Callable[[], Mapping[str, Any]], *, runs: int = 2
) -> PrefixDiff:
    """Call `builder` `runs` times and diff the renders pairwise.

    `builder` should be the function your application actually uses to assemble
    a request -- the point is to exercise whatever non-determinism lives inside
    it. Returns the first divergence found, or an identical result.
    """
    if runs < 2:
        raise ValueError("runs must be at least 2 to have something to compare")
    rendered = [render_prefix(builder()) for _ in range(runs)]
    baseline = rendered[0]
    for candidate in rendered[1:]:
        diff = _diff_rendered(baseline, candidate)
        if not diff.identical:
            return diff
    return PrefixDiff(
        identical=True, left_length=len(baseline), right_length=len(baseline)
    )


def assert_stable_prefix(
    builder: Callable[[], Mapping[str, Any]], *, runs: int = 2
) -> None:
    """Raise `SilentInvalidatorError` if `builder` is not byte-deterministic.

    Suitable for a unit test in the caller's own suite -- it needs no network
    and no credentials.
    """
    diff = find_silent_invalidator(builder, runs=runs)
    if not diff.identical:
        raise SilentInvalidatorError(diff.describe())
