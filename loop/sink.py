"""JSONL sink and reader for profiler records.

Records are validated against ``loop.schema`` *on write*, so a malformed
record fails at the producer rather than silently poisoning a consumer.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, IO, Iterable, Iterator

from .schema import validate_record, validate_run


class JsonlSink:
    """Append-only JSONL writer.

    ``JsonlSink(path)`` opens (and creates) the file; ``JsonlSink(handle)``
    wraps an already-open text handle and will not close it.
    """

    def __init__(self, target: str | os.PathLike[str] | IO[str], *, validate: bool = True):
        self.validate = validate
        self._owns_handle = not hasattr(target, "write")
        if self._owns_handle:
            path = Path(target)
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            self.path: Path | None = path
            self._handle: IO[str] = path.open("a", encoding="utf-8")
        else:
            self.path = None
            self._handle = target  # type: ignore[assignment]
        self.count = 0

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.validate:
            validate_record(record)
        self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()
        self.count += 1
        return record

    def write_all(self, records: Iterable[dict[str, Any]]) -> int:
        written = 0
        for record in records:
            self.write(record)
            written += 1
        return written

    def close(self) -> None:
        if self._owns_handle:
            self._handle.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextmanager
def open_sink(
    target: str | os.PathLike[str] | IO[str], *, validate: bool = True
) -> Iterator[JsonlSink]:
    sink = JsonlSink(target, validate=validate)
    try:
        yield sink
    finally:
        sink.close()


def read_jsonl(path: str | os.PathLike[str], *, validate: bool = True) -> list[dict[str, Any]]:
    """Read a JSONL file of profiler records.

    With ``validate=True`` the whole stream is checked for record order, turn
    contiguity and run_id agreement, not just per-record shape.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: not valid JSON: {exc}") from exc
    if validate:
        validate_run(records)
    return records


def write_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[dict[str, Any]],
    *,
    validate: bool = True,
) -> int:
    with open_sink(path, validate=validate) as sink:
        return sink.write_all(records)


class MemorySink:
    """Collect records in a list. Useful in tests and for piping to a renderer."""

    def __init__(self, *, validate: bool = True):
        self.validate = validate
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.validate:
            validate_record(record)
        self.records.append(record)
        return record

    def close(self) -> None:  # pragma: no cover - symmetry with JsonlSink
        pass
