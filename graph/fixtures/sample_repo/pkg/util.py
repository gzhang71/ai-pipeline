"""String helpers for the sample package."""

SEPARATOR = "-"
RETRY_LIMIT = 7


def slugify(text: str) -> str:
    """Lowercase a string and join its words with the separator."""
    cleaned = text.strip().lower()  # SENTINEL_SLUGIFY_BODY
    return SEPARATOR.join(cleaned.split())


def _shout(text: str) -> str:
    """Uppercase helper, private by convention."""
    return text.upper() + "!"  # SENTINEL_SHOUT_BODY
