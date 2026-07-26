"""Entry point for the sample repo, one package level away from `pkg`."""

from pkg.core import Widget, build

DEFAULT_NAME = "Hello World"
MAX_WIDGETS = 3


def main() -> str:
    """Build the default widget and return its loud slug."""
    widget = build(DEFAULT_NAME)  # SENTINEL_MAIN_BUILD_BODY
    return widget.loud()


def make_blank() -> Widget:
    """Construct a widget directly, bypassing the builder."""
    return Widget("")  # SENTINEL_MAIN_BLANK_BODY
