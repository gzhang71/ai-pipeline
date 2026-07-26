"""Execute prompt x task and record durable results.

Run records are JSONL. The first line is a `run_meta` header (run id, prompt id
and hash, task-set hash, model, judge identity, concurrency); every line after
it is one `result` object for one task.

Durability and resumption
-------------------------
Results are appended and flushed as each task finishes, so a crash at task 37 of
50 loses one task, not the run. Re-invoking `run()` against the same file reads
back the completed task ids and skips them -- and refuses to append if the
header's `prompt_hash` does not match the prompt you are running now, because a
run file containing two prompt versions is worse than no run file at all.

Concurrency is a bounded thread pool. Each worker makes one blocking API call,
so threads are the right shape here, and the bound is what keeps a 50-task run
from tripping rate limits.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from common.client import MODEL

from . import assertions as assertions_mod
from .assertions import AssertionResult
from .hashing import hash_output
from .judge import Judge
from .model import EMPTY_USAGE, ModelClient, ModelOutput, ModelRequest, sum_usage
from .prompts import Prompt
from .tasks import Task, task_set_hash

MAX_STORED_OUTPUT_CHARS = 8000

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_INCOMPLETE = "incomplete"


class RunError(Exception):
    pass


@dataclass(frozen=True)
class TaskResult:
    run_id: str
    prompt_id: str
    prompt_hash: str
    task_id: str
    task_hash: str
    passed: bool
    status: str
    assertions: tuple[AssertionResult, ...]
    output_text: str
    output_hash: str
    output_truncated: bool
    tool_calls: tuple[Mapping[str, Any], ...]
    stop_reason: str | None
    usage: Mapping[str, int]
    judge_usage: Mapping[str, int]
    model: str
    duration_ms: int
    error: str | None
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "result",
            "run_id": self.run_id,
            "prompt_id": self.prompt_id,
            "prompt_hash": self.prompt_hash,
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "passed": self.passed,
            "status": self.status,
            "assertions": [a.to_dict() for a in self.assertions],
            "output_text": self.output_text,
            "output_hash": self.output_hash,
            "output_truncated": self.output_truncated,
            "tool_calls": [dict(c) for c in self.tool_calls],
            "stop_reason": self.stop_reason,
            "usage": dict(self.usage),
            "judge_usage": dict(self.judge_usage),
            "model": self.model,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskResult":
        return cls(
            run_id=data.get("run_id", ""),
            prompt_id=data.get("prompt_id", ""),
            prompt_hash=data.get("prompt_hash", ""),
            task_id=data["task_id"],
            task_hash=data.get("task_hash", ""),
            passed=bool(data["passed"]),
            status=data.get("status", STATUS_OK),
            assertions=tuple(
                AssertionResult.from_dict(a) for a in data.get("assertions", [])
            ),
            output_text=data.get("output_text", ""),
            output_hash=data.get("output_hash", ""),
            output_truncated=bool(data.get("output_truncated", False)),
            tool_calls=tuple(data.get("tool_calls", [])),
            stop_reason=data.get("stop_reason"),
            usage=dict(data.get("usage") or EMPTY_USAGE),
            judge_usage=dict(data.get("judge_usage") or {}),
            model=data.get("model", MODEL),
            duration_ms=int(data.get("duration_ms", 0)),
            error=data.get("error"),
            finished_at=data.get("finished_at", ""),
        )

    @property
    def failed_assertions(self) -> tuple[AssertionResult, ...]:
        return tuple(a for a in self.assertions if not a.passed)


@dataclass
class RunRecord:
    """A parsed run file: header plus results keyed by task id."""

    meta: Mapping[str, Any]
    results: dict[str, TaskResult] = field(default_factory=dict)
    path: Path | None = None

    @property
    def run_id(self) -> str:
        return str(self.meta.get("run_id", ""))

    @property
    def prompt_label(self) -> str:
        return f"{self.meta.get('prompt_id', '?')}@{str(self.meta.get('prompt_hash', ''))[:12]}"

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results.values() if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    def total_usage(self) -> dict[str, int]:
        return sum_usage(
            [r.usage for r in self.results.values()]
            + [r.judge_usage for r in self.results.values()]
        )


def load_run(path: Path) -> RunRecord:
    path = Path(path)
    if not path.is_file():
        raise RunError(f"no such run file: {path}")
    meta: dict[str, Any] = {}
    results: dict[str, TaskResult] = {}
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            # A crash mid-write can leave a torn final line. Everything before
            # it is still valid, so drop the tail rather than failing the read.
            if lineno == _line_count(path):
                break
            raise RunError(f"{path}:{lineno}: corrupt JSONL: {exc}") from exc
        if record.get("type") == "run_meta":
            meta = record
        elif record.get("type") == "result":
            result = TaskResult.from_dict(record)
            results[result.task_id] = result  # last write wins
    return RunRecord(meta=meta, results=results, path=path)


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


class RunRecorder:
    """Append-only JSONL writer, flushed per record and safe across threads."""

    def __init__(self, path: Path, meta: Mapping[str, Any], *, resume: bool = True) -> None:
        self.path = Path(path)
        self.meta = dict(meta)
        self.completed: set[str] = set()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.is_file() and self.path.stat().st_size > 0:
            existing = load_run(self.path)
            if not resume:
                raise RunError(
                    f"{self.path} already exists; pass resume=True to continue it "
                    "or choose another output path"
                )
            prior_hash = existing.meta.get("prompt_hash")
            if prior_hash and prior_hash != self.meta.get("prompt_hash"):
                raise RunError(
                    f"{self.path} was produced by prompt hash {str(prior_hash)[:12]}, "
                    f"but this run uses {str(self.meta.get('prompt_hash'))[:12]}. "
                    "Mixing prompt versions in one run file would corrupt the diff; "
                    "write to a different file."
                )
            self.completed = set(existing.results)
            self.meta = dict(existing.meta) or self.meta
            self._handle = self.path.open("a", encoding="utf-8")
            self.resumed = True
        else:
            self._handle = self.path.open("w", encoding="utf-8")
            self._write({"type": "run_meta", **self.meta})
            self.resumed = False

    def _write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def record(self, result: TaskResult) -> None:
        with self._lock:
            self._write(result.to_dict())
            self.completed.add(result.task_id)

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


@dataclass
class RunSummary:
    run_id: str
    path: Path
    prompt: Prompt
    results: list[TaskResult]
    skipped: list[str]
    meta: Mapping[str, Any]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def usage(self) -> dict[str, int]:
        return sum_usage([r.usage for r in self.results] + [r.judge_usage for r in self.results])


def build_request(prompt: Prompt, task: Task) -> ModelRequest:
    system = prompt.system
    if task.system_suffix:
        system = f"{system}\n\n{task.system_suffix.strip()}"
    return ModelRequest(
        system=system,
        messages=({"role": "user", "content": task.input},),
        max_tokens=task.max_tokens or prompt.max_tokens or 1024,
        tools=tuple(dict(t) for t in task.tools),
        effort=prompt.effort,
        prompt_id=prompt.id,
        prompt_hash=prompt.hash,
        task_id=task.id,
    )


def evaluate_task(
    *,
    prompt: Prompt,
    task: Task,
    output: ModelOutput,
    judge: Judge | None,
    run_id: str,
) -> TaskResult:
    """Score one already-produced output. Pure apart from the judge call."""
    results: list[AssertionResult] = []
    judge_usages: list[Mapping[str, int]] = []
    for index, spec in enumerate(task.assertions):
        if spec.get("type") == assertions_mod.JUDGE_TYPE:
            if judge is None:
                results.append(
                    AssertionResult(
                        assertions_mod.spec_id(spec, index),
                        "judge",
                        False,
                        "judge assertion not evaluated (no judge configured)",
                        skipped=True,
                    )
                )
                continue
            result = judge.evaluate(spec, task=task, prompt=prompt, output=output, index=index)
            judge_usages.append(result.meta.get("usage") or {})
            results.append(result)
        else:
            results.append(assertions_mod.evaluate(spec, output, index))

    skipped = any(a.skipped for a in results)
    if output.error:
        status = STATUS_ERROR
    elif skipped:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_OK

    text = output.text
    truncated = len(text) > MAX_STORED_OUTPUT_CHARS
    return TaskResult(
        run_id=run_id,
        prompt_id=prompt.id,
        prompt_hash=prompt.hash,
        task_id=task.id,
        task_hash=task.hash,
        passed=all(a.passed for a in results),
        status=status,
        assertions=tuple(results),
        output_text=text[:MAX_STORED_OUTPUT_CHARS],
        output_hash=hash_output(output.text, output.tool_calls),
        output_truncated=truncated,
        tool_calls=tuple(c.to_dict() for c in output.tool_calls),
        stop_reason=output.stop_reason,
        usage=dict(output.usage),
        judge_usage=sum_usage(judge_usages) if judge_usages else {},
        model=output.model,
        duration_ms=output.duration_ms,
        error=output.error,
        finished_at=_now(),
    )


def run(
    *,
    prompt: Prompt,
    tasks: Sequence[Task],
    client: ModelClient,
    out_path: Path,
    judge: Judge | None = None,
    concurrency: int = 4,
    resume: bool = True,
    run_id: str | None = None,
    model: str = MODEL,
    on_result: Callable[[TaskResult], None] | None = None,
) -> RunSummary:
    """Run `prompt` over `tasks`, appending results to `out_path`."""
    if concurrency < 1:
        raise RunError("concurrency must be >= 1")
    if not tasks:
        raise RunError("no tasks to run")

    run_id = run_id or uuid.uuid4().hex[:16]
    meta = {
        "run_id": run_id,
        "created_at": _now(),
        "prompt_id": prompt.id,
        "prompt_hash": prompt.hash,
        "prompt_path": str(prompt.path),
        "task_set_hash": task_set_hash(tasks),
        "task_count": len(tasks),
        "model": model,
        "concurrency": concurrency,
        "judge_prompt_id": judge.prompt.id if judge else None,
        "judge_hash": judge.hash if judge else None,
        "judge_samples": judge.samples if judge else 0,
        "harness_version": 1,
    }

    with RunRecorder(out_path, meta, resume=resume) as recorder:
        pending = [t for t in tasks if t.id not in recorder.completed]
        skipped = [t.id for t in tasks if t.id in recorder.completed]

        def execute(task: Task) -> TaskResult:
            output = client.complete(build_request(prompt, task))
            result = evaluate_task(
                prompt=prompt, task=task, output=output, judge=judge, run_id=run_id
            )
            recorder.record(result)
            if on_result is not None:
                on_result(result)
            return result

        fresh: list[TaskResult] = []
        if pending:
            workers = min(concurrency, len(pending))
            if workers == 1:
                fresh = [execute(task) for task in pending]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fresh = list(pool.map(execute, pending))

        final_meta = dict(recorder.meta)

    complete = load_run(out_path)
    ordered = [complete.results[t.id] for t in tasks if t.id in complete.results]
    _ = fresh
    return RunSummary(
        run_id=run_id,
        path=Path(out_path),
        prompt=prompt,
        results=ordered,
        skipped=skipped,
        meta=final_meta,
    )


def default_run_path(prompt: Prompt, directory: Path = Path("runs")) -> Path:
    return Path(directory) / f"{prompt.id}.{prompt.short_hash}.jsonl"


def iter_results(records: Iterable[RunRecord]) -> Iterable[TaskResult]:
    for record in records:
        yield from record.results.values()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
