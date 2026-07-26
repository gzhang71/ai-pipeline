"""A prompt regression harness.

Versioned prompts on disk, a fixed task set with two tiers of assertions, a
resumable runner, and a diff that puts regressions before aggregates.

    from harness import load_prompts, load_tasks, run, diff_files, format_report

    prompts = load_prompts(default_prompt_dir())
    tasks = list(load_tasks(default_task_dir()).values())
    summary = run(prompt=prompts["triage.v2"], tasks=tasks,
                  client=AnthropicClient(), out_path=Path("runs/v2.jsonl"))
    print(format_report(diff_files("runs/v1.jsonl", "runs/v2.jsonl")))
"""

from __future__ import annotations

from .assertions import (
    ALL_TYPES,
    STRUCTURAL_TYPES,
    AssertionResult,
    AssertionSpecError,
    evaluate,
    evaluate_all,
    extract_json,
    validate_spec,
)
from .diff import DiffReport, TaskDelta, diff_files, diff_runs, format_json, format_report
from .fakes import FixtureError, load_fixture
from .hashing import hash_bytes, hash_output, hash_text, short
from .judge import DEFAULT_JUDGE_ID, VERDICT_SCHEMA, Judge, JudgeCache, Verdict, cache_key
from .jsonschema import SchemaError, check_schema, validate
from .model import (
    AnthropicClient,
    ModelClient,
    ModelOutput,
    ModelRequest,
    ScriptedClient,
    ToolCall,
    make_usage,
    sum_usage,
)
from .prompts import Prompt, PromptError, default_prompt_dir, load_prompt_file, load_prompts
from .runner import (
    RunError,
    RunRecord,
    RunRecorder,
    RunSummary,
    TaskResult,
    build_request,
    default_run_path,
    evaluate_task,
    load_run,
    run,
)
from .stats import mcnemar_exact, min_detectable_flips, wilson_interval
from .tasks import Task, TaskError, default_task_dir, load_task_file, load_tasks, task_set_hash

__all__ = [
    # prompts
    "Prompt",
    "PromptError",
    "load_prompts",
    "load_prompt_file",
    "default_prompt_dir",
    # tasks
    "Task",
    "TaskError",
    "load_tasks",
    "load_task_file",
    "task_set_hash",
    "default_task_dir",
    # assertions
    "AssertionResult",
    "AssertionSpecError",
    "STRUCTURAL_TYPES",
    "ALL_TYPES",
    "evaluate",
    "evaluate_all",
    "validate_spec",
    "extract_json",
    # json schema subset
    "SchemaError",
    "check_schema",
    "validate",
    # model
    "ModelClient",
    "ModelRequest",
    "ModelOutput",
    "ToolCall",
    "AnthropicClient",
    "ScriptedClient",
    "make_usage",
    "sum_usage",
    # judge
    "Judge",
    "JudgeCache",
    "Verdict",
    "VERDICT_SCHEMA",
    "DEFAULT_JUDGE_ID",
    "cache_key",
    # runner
    "run",
    "RunSummary",
    "RunRecord",
    "RunRecorder",
    "RunError",
    "TaskResult",
    "load_run",
    "build_request",
    "evaluate_task",
    "default_run_path",
    # diff
    "diff_runs",
    "diff_files",
    "DiffReport",
    "TaskDelta",
    "format_report",
    "format_json",
    # stats
    "wilson_interval",
    "mcnemar_exact",
    "min_detectable_flips",
    # offline fixtures
    "load_fixture",
    "FixtureError",
    # hashing
    "hash_bytes",
    "hash_text",
    "hash_output",
    "short",
]

__version__ = "0.1.0"
