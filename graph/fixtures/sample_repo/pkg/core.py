"""Core objects for the sample package."""

from pkg import util
from pkg.util import slugify


class Widget:
    """A named widget."""

    def __init__(self, name: str) -> None:
        self.name = name

    def slug(self) -> str:
        """Return the widget's slug."""
        return slugify(self.name)  # SENTINEL_WIDGET_SLUG_BODY

    def loud(self) -> str:
        """Return a shouty version of the slug."""
        inner = self.slug()
        return util._shout(inner)  # SENTINEL_WIDGET_LOUD_BODY


class Gadget(Widget):
    """A widget subclass that overrides nothing interesting."""

    def describe(self) -> str:
        """Describe the gadget."""
        return "gadget:" + self.slug()  # SENTINEL_GADGET_BODY


def build(name: str) -> Widget:
    """Build a widget from a raw name."""

    def _clean(value: str) -> str:
        """Nested helper that trims whitespace."""
        return value.strip()  # SENTINEL_CLEAN_BODY

    return Widget(_clean(name))  # SENTINEL_BUILD_BODY
