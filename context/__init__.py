"""Compaction strategy bench.

Several history-management strategies behind one interface, run against the
same long-horizon task set, measured on task success against tokens spent.

    from context import all_strategies, run_bench, format_report

    print(format_report(run_bench()))

See `context/README.md` for what each strategy costs and how to read the
numbers.
"""

from .bench import (
    BenchReport,
    LiveClient,
    RunnerClient,
    StrategySummary,
    TaskResult,
    TurnRecord,
    format_report,
    run_bench,
    run_task,
)
from .fakes import FakeClient, FakeResponse, FakeUsage
from .strategies import (
    CLEAR_THINKING_EDIT,
    CLEAR_TOOL_USES_EDIT,
    COMPACTION_BETA,
    COMPACTION_EDIT,
    CONTEXT_EDITING_BETA,
    AnchoredSummary,
    Budget,
    NoteTaking,
    RecursiveSummarization,
    ServerCompaction,
    Strategy,
    StrategyResult,
    TailTruncation,
    ToolResultEviction,
    all_strategies,
)
from .summarizers import (
    FakeNoteWriter,
    FakeSummarizer,
    ModelNoteWriter,
    ModelSummarizer,
    NoteWriter,
    Summarizer,
    extract_facts,
)
from .tasks import TASKS, TOOLS, Task, ToolExchange, Turn, build_tasks
from .tokens import ApiTokenCounter, HeuristicTokenCounter, TokenCounter, default_counter
from .usage import Usage
from .validation import (
    InvalidMessageShape,
    assert_valid,
    is_valid,
    sanitize,
    validate,
)

__all__ = [
    # interface
    "Strategy",
    "StrategyResult",
    "Budget",
    # strategies
    "TailTruncation",
    "RecursiveSummarization",
    "AnchoredSummary",
    "NoteTaking",
    "ToolResultEviction",
    "ServerCompaction",
    "all_strategies",
    # validation
    "validate",
    "is_valid",
    "assert_valid",
    "sanitize",
    "InvalidMessageShape",
    # accounting
    "Usage",
    "TokenCounter",
    "HeuristicTokenCounter",
    "ApiTokenCounter",
    "default_counter",
    # summarizers
    "Summarizer",
    "NoteWriter",
    "ModelSummarizer",
    "ModelNoteWriter",
    "FakeSummarizer",
    "FakeNoteWriter",
    "extract_facts",
    # bench
    "run_bench",
    "run_task",
    "format_report",
    "BenchReport",
    "StrategySummary",
    "TaskResult",
    "TurnRecord",
    "RunnerClient",
    "LiveClient",
    "FakeClient",
    "FakeResponse",
    "FakeUsage",
    # tasks
    "Task",
    "Turn",
    "ToolExchange",
    "TASKS",
    "TOOLS",
    "build_tasks",
    # API constants
    "COMPACTION_BETA",
    "COMPACTION_EDIT",
    "CONTEXT_EDITING_BETA",
    "CLEAR_TOOL_USES_EDIT",
    "CLEAR_THINKING_EDIT",
]
