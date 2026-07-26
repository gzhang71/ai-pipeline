"""Exception types for the cache-aware prompt assembler."""

from __future__ import annotations


class PromptCacheError(Exception):
    """Base class for every error this package raises."""


class PromptOrderingError(PromptCacheError):
    """A less-stable block precedes a more-stable one in the rendered order.

    Prompt caching is a prefix match over the rendered request bytes in the
    order tools -> system -> messages. If a block that changes every request
    sits in front of a block that never changes, the stable block can never be
    served from cache: its bytes start at an offset that already moved.
    """


class PromptStructureError(PromptCacheError):
    """The assembled prompt would be rejected by the Messages API."""


class SilentInvalidatorError(PromptCacheError):
    """Two renders of the same prompt builder produced different bytes.

    Nothing about this fails at request time -- the API happily accepts both
    renders and simply never reports a cache read. That is what makes it worth
    an exception here.
    """
