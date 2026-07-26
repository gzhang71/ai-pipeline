"""Versioned prompts on disk.

Format: one file per prompt, `<id>.prompt.md`, with an optional TOML
frontmatter block delimited by `+++`. The body after the frontmatter is the
system prompt, verbatim.

    +++
    description = "Triage a support ticket into structured JSON"
    effort = "medium"
    +++
    You are a support triage assistant...

Why this format:

* Markdown body -- prompts are prose. Editing them in a `.md` file gives you
  syntax-free diffs and no escaping games. A prompt trapped inside a JSON
  string literal is a prompt nobody edits.
* TOML frontmatter -- `tomllib` is stdlib in 3.11+, so metadata costs no
  dependency, and it is typed (unlike YAML's surprises).
* The version lives in the *filename* (`triage.v1`, `triage.v2`), and identity
  lives in the *hash*. There is no `version:` field to forget to bump: the hash
  is taken over the entire raw file bytes, so any edit -- body, metadata, or a
  single trailing space -- produces a new hash automatically.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .hashing import hash_bytes, short

FRONTMATTER_DELIM = "+++"
PROMPT_SUFFIX = ".prompt.md"


class PromptError(Exception):
    pass


@dataclass(frozen=True)
class Prompt:
    """A prompt file plus the hash of its exact bytes."""

    id: str
    path: Path
    raw: bytes
    hash: str
    system: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def short_hash(self) -> str:
        return short(self.hash)

    @property
    def description(self) -> str:
        return str(self.meta.get("description", ""))

    @property
    def effort(self) -> str | None:
        value = self.meta.get("effort")
        return str(value) if value is not None else None

    @property
    def max_tokens(self) -> int | None:
        value = self.meta.get("max_tokens")
        return int(value) if value is not None else None

    def label(self) -> str:
        return f"{self.id}@{self.short_hash}"


def parse_prompt(raw: bytes, *, path: Path, prompt_id: str | None = None) -> Prompt:
    text = raw.decode("utf-8")
    meta: dict[str, Any] = {}
    body = text
    if text.startswith(FRONTMATTER_DELIM):
        lines = text.split("\n")
        try:
            end = next(
                i for i, line in enumerate(lines[1:], start=1)
                if line.strip() == FRONTMATTER_DELIM
            )
        except StopIteration:
            raise PromptError(f"{path}: unterminated +++ frontmatter block") from None
        try:
            meta = tomllib.loads("\n".join(lines[1:end]))
        except tomllib.TOMLDecodeError as exc:
            raise PromptError(f"{path}: bad TOML frontmatter: {exc}") from exc
        body = "\n".join(lines[end + 1:])

    resolved_id = str(meta.get("id") or prompt_id or _id_from_path(path))
    system = body.strip()
    if not system:
        raise PromptError(f"{path}: prompt body is empty")
    return Prompt(
        id=resolved_id,
        path=path,
        raw=raw,
        hash=hash_bytes(raw),
        system=system,
        meta=meta,
    )


def load_prompt_file(path: Path) -> Prompt:
    path = Path(path)
    if not path.is_file():
        raise PromptError(f"no such prompt file: {path}")
    return parse_prompt(path.read_bytes(), path=path)


def load_prompts(directory: Path) -> dict[str, Prompt]:
    """Load every `*.prompt.md` under `directory`, keyed by prompt id."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PromptError(f"no such prompt directory: {directory}")
    prompts: dict[str, Prompt] = {}
    for path in sorted(directory.glob(f"*{PROMPT_SUFFIX}")):
        prompt = load_prompt_file(path)
        if prompt.id in prompts:
            raise PromptError(
                f"duplicate prompt id {prompt.id!r}: {prompts[prompt.id].path} and {path}"
            )
        prompts[prompt.id] = prompt
    return prompts


def _id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(PROMPT_SUFFIX):
        return name[: -len(PROMPT_SUFFIX)]
    return path.stem


def default_prompt_dir() -> Path:
    return Path(__file__).parent / "data" / "prompts"
