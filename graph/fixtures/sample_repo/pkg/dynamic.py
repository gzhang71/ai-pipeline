"""Call shapes the static analysis deliberately cannot resolve.

Every function here is a documented blind spot of the `ast` pass. They exist so
the test suite can assert that the builder *misses* them rather than inventing
edges it cannot justify.
"""

import importlib


def load_and_run(module_name: str):
    """Dynamic import: the target is a runtime string, invisible to `ast`."""
    module = importlib.import_module(module_name)  # SENTINEL_DYNAMIC_IMPORT_BODY
    return module.run()


def call_by_name(obj, method_name: str):
    """getattr dispatch: the callee is not in the syntax tree at all."""
    return getattr(obj, method_name)()  # SENTINEL_GETATTR_BODY


def call_untyped(thing):
    """Attribute call on an untyped value: we cannot know what `thing` is."""
    return thing.slug()  # SENTINEL_UNTYPED_BODY


def call_via_alias():
    """A function bound to a variable, then called through it."""
    from pkg.util import slugify

    alias = slugify
    return alias("bound through a variable")  # SENTINEL_ALIAS_BODY
