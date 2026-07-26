"""Context budget profiler: where did my context window actually go?

An instrumented tool-use loop that records, for every turn, a per-segment
breakdown of the prompt it sent -- system prompt, tool schemas, and each
message's contribution split by kind -- and reconciles that breakdown against
the authoritative ``usage`` on the response.

Quick start (offline, with the bundled fakes)::

    from loop import LoopConfig, MemorySink, run_loop, render_text
    from loop.testing import FakeAnthropicClient, text_block, echo_executor

    sink = MemorySink()
    result = run_loop(
        client=FakeAnthropicClient([...]),
        tools=[...],
        executor=echo_executor,
        prompt="...",
        config=LoopConfig(system="..."),
        sink=sink,
    )
    print(render_text(sink.records))

Against the live API, drop the fakes and pass ``common.client.get_client()``.
The JSONL schema is documented field by field in ``loop/README.md``.
"""

from __future__ import annotations

from .accuracy import (
    AccuracyReport,
    Bin,
    Observation,
    analyze_accuracy,
    observations_from_records,
    report_text,
    wilson_interval,
)
from .agent import (
    LIB_VERSION,
    Executor,
    LoopConfig,
    RunResult,
    ToolResult,
    run_loop,
)
from .attribution import (
    METHOD as ATTRIBUTION_METHOD,
    Attribution,
    BlockGroup,
    CachingTokenCounter,
    Segment,
    TokenCounter,
    api_token_counter,
    attribute,
    block_groups,
    reconcile,
)
from .render import KIND_COLORS, KIND_LABELS, KIND_ORDER, render_html, render_text
from .schema import (
    GRANULARITIES,
    RECORD_TYPES,
    RUN_STOP_REASONS,
    SCHEMA_ID,
    SCHEMA_VERSION,
    SEGMENT_KINDS,
    SchemaError,
    group_runs,
    validate_record,
    validate_run,
)
from .sink import JsonlSink, MemorySink, open_sink, read_jsonl, write_jsonl
from .tasks import SYNTHETIC_TASKS, TOOLS, Task, TaskRun, make_executor, run_task_set

__all__ = [
    # loop
    "run_loop",
    "LoopConfig",
    "RunResult",
    "ToolResult",
    "Executor",
    "LIB_VERSION",
    # attribution
    "attribute",
    "reconcile",
    "Attribution",
    "Segment",
    "BlockGroup",
    "block_groups",
    "TokenCounter",
    "CachingTokenCounter",
    "api_token_counter",
    "ATTRIBUTION_METHOD",
    # schema + sinks
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SEGMENT_KINDS",
    "RECORD_TYPES",
    "RUN_STOP_REASONS",
    "GRANULARITIES",
    "SchemaError",
    "validate_record",
    "validate_run",
    "group_runs",
    "JsonlSink",
    "MemorySink",
    "open_sink",
    "read_jsonl",
    "write_jsonl",
    # rendering
    "render_text",
    "render_html",
    "KIND_ORDER",
    "KIND_LABELS",
    "KIND_COLORS",
    # accuracy
    "Observation",
    "Bin",
    "AccuracyReport",
    "analyze_accuracy",
    "observations_from_records",
    "report_text",
    "wilson_interval",
    # synthetic task set
    "Task",
    "TaskRun",
    "TOOLS",
    "SYNTHETIC_TASKS",
    "make_executor",
    "run_task_set",
]
