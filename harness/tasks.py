"""The task set: an input plus assertions, one TOML file per task.

    # data/tasks/t01_simple_bug.toml
    description = "Plain bug report triages to category=bug"
    input = "The export button does nothing on Safari."
    max_tokens = 400

    [[assertions]]
    type = "json_schema"
    schema_json = '''{"type": "object", "required": ["category"]}'''

    [[assertions]]
    type = "no_tool_called"

Why TOML, one file per task:

* `tomllib` is stdlib, so no dependency, and it is strict about types.
* Multi-line basic/literal strings hold prose inputs and inline JSON schemas
  without escaping, which JSON cannot do and YAML does badly.
* One file per task keeps `git diff` on a 40-task set readable and makes merge
  conflicts local to the task somebody actually edited. Adding a task is
  `cp` + edit; nothing else in the repo changes.

Task ids come from the filename stem, so the id and the file cannot drift apart.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .assertions import AssertionSpecError, validate_spec
from .hashing import hash_bytes, hash_text, short

TASK_SUFFIX = ".toml"
DEFAULT_MAX_TOKENS = 1024


class TaskError(Exception):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    path: Path
    hash: str
    input: str
    assertions: tuple[Mapping[str, Any], ...]
    description: str = ""
    system_suffix: str = ""
    tools: tuple[Mapping[str, Any], ...] = ()
    max_tokens: int = DEFAULT_MAX_TOKENS
    tags: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def short_hash(self) -> str:
        return short(self.hash)

    @property
    def has_judge(self) -> bool:
        return any(spec.get("type") == "judge" for spec in self.assertions)


def parse_task(raw: bytes, *, path: Path, task_id: str | None = None) -> Task:
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise TaskError(f"{path}: bad TOML: {exc}") from exc

    resolved_id = str(data.get("id") or task_id or path.stem)
    if "input" not in data:
        raise TaskError(f"{path}: missing required key `input`")
    assertions = data.get("assertions") or []
    if not isinstance(assertions, list) or not assertions:
        raise TaskError(f"{path}: needs at least one [[assertions]] entry")
    for index, spec in enumerate(assertions):
        if not isinstance(spec, dict):
            raise TaskError(f"{path}: assertion {index} is not a table")
        try:
            validate_spec(spec, index, where=str(path))
        except AssertionSpecError as exc:
            raise TaskError(str(exc)) from exc

    tools = tuple(_normalize_tool(t, path) for t in data.get("tools", []))
    return Task(
        id=resolved_id,
        path=path,
        hash=hash_bytes(raw),
        input=str(data["input"]),
        assertions=tuple(assertions),
        description=str(data.get("description", "")),
        system_suffix=str(data.get("system_suffix", "")),
        tools=tools,
        max_tokens=int(data.get("max_tokens", DEFAULT_MAX_TOKENS)),
        tags=tuple(str(t) for t in data.get("tags", [])),
        meta={k: v for k, v in data.items() if k not in _RESERVED},
    )


_RESERVED = {
    "id",
    "input",
    "assertions",
    "description",
    "system_suffix",
    "tools",
    "max_tokens",
    "tags",
}


def load_task_file(path: Path) -> Task:
    path = Path(path)
    if not path.is_file():
        raise TaskError(f"no such task file: {path}")
    return parse_task(path.read_bytes(), path=path)


def load_tasks(directory: Path) -> dict[str, Task]:
    directory = Path(directory)
    if not directory.is_dir():
        raise TaskError(f"no such task directory: {directory}")
    tasks: dict[str, Task] = {}
    for path in sorted(directory.glob(f"*{TASK_SUFFIX}")):
        task = load_task_file(path)
        if task.id in tasks:
            raise TaskError(f"duplicate task id {task.id!r}: {tasks[task.id].path} and {path}")
        tasks[task.id] = task
    if not tasks:
        raise TaskError(f"no {TASK_SUFFIX} task files in {directory}")
    return tasks


def task_set_hash(tasks: Iterable[Task]) -> str:
    """Hash of the whole task set, so a diff can tell if the set itself moved.

    Comparing two runs over different task sets is a category error; the diff
    warns when these differ.
    """
    lines = sorted(f"{task.id}:{task.hash}" for task in tasks)
    return hash_text("\n".join(lines))


def _normalize_tool(tool: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(tool, dict):
        raise TaskError(f"{path}: [[tools]] entries must be tables")
    if "name" not in tool:
        raise TaskError(f"{path}: a [[tools]] entry is missing `name`")
    normalized = dict(tool)
    if "input_schema_json" in normalized:
        import json

        try:
            normalized["input_schema"] = json.loads(normalized.pop("input_schema_json"))
        except json.JSONDecodeError as exc:
            raise TaskError(f"{path}: tool {tool['name']}: bad input_schema_json: {exc}") from exc
    normalized.setdefault(
        "input_schema", {"type": "object", "properties": {}, "additionalProperties": True}
    )
    normalized.setdefault("description", "")
    return normalized


def default_task_dir() -> Path:
    return Path(__file__).parent / "data" / "tasks"
