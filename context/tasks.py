"""A small synthetic long-horizon task set.

Every task has the same shape, because it is the shape that breaks context
management: a fact is introduced **early**, a long noisy middle blows the
budget, and the fact is needed **late**. A strategy that scores well here has
preserved information across a compaction boundary; a strategy that scores
badly has thrown it away.

The tasks are deliberately not all winnable the same way, so the report can
discriminate rather than produce one flat column:

* ``early-constant``   fact in ordinary user text
* ``early-tool-result``fact only ever present inside a tool result
* ``objective-drift``  the thing needed late is the original objective itself
* ``corrected-fact``   a value stated early and *corrected* early; the stale
                       value is still in the transcript and is the wrong answer
* ``fact-flood``       thirteen facts introduced early, the *first* one queried
                       — more durable state than any bounded store will hold
* ``late-fact``        control: the fact is two turns from the end, so any
                       strategy that fails this one is broken, not lossy

Facts are written as machine-checkable ``[FACT] KEY = value`` lines so that
success is a property of the *context the model was given*, not of the
model's prose. The filler turns carry their own ``[FACT] METRIC_n`` noise —
that noise is what pushes early facts out of a recency-biased summary, and
without it the task set would be far too easy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

QUERY_PREFIX = "[QUERY]"

LOREM = (
    "Reviewed the checklist against the runbook, re-ran the smoke suite, and "
    "confirmed the staging cutover window is still open. Nothing in the "
    "dashboards has moved outside its band since the last check, so the plan "
    "stands as written and the next item can proceed on schedule. "
)

LOG_LINES = (
    "INFO  scheduler: reconcile loop tick, 41 objects scanned, 0 drift\n"
    "INFO  billing-api: p99 118ms, p50 21ms, error-rate 0.02%\n"
    "WARN  cache: eviction rate elevated on shard 3, within tolerance\n"
    "INFO  worker: 1204 jobs drained, 0 retries, queue depth 7\n"
    "INFO  scheduler: reconcile loop tick, 41 objects scanned, 0 drift\n"
)


@dataclass(frozen=True)
class ToolExchange:
    """A scripted assistant tool_use + user tool_result pair."""

    name: str
    input: dict[str, Any]
    result: str


@dataclass(frozen=True)
class Turn:
    user: str
    tool: ToolExchange | None = None


@dataclass(frozen=True)
class Task:
    id: str
    objective: str
    turns: tuple[Turn, ...]
    question: str
    expected: dict[str, str]
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_early(self) -> bool:
        """True if the answer depends on information introduced early."""
        return "early" in self.tags


def _filler(i: int, *, tool: bool = False) -> Turn:
    text = (
        f"Step {i}: continue working the migration checklist. {LOREM}"
        f"[FACT] METRIC_{i:02d} = {40 + i * 3}ms"
    )
    exchange = None
    if tool:
        exchange = ToolExchange(
            name="read_logs",
            input={"service": "billing-api", "window": f"step-{i}"},
            result=f"{LOG_LINES}{LOG_LINES}INFO  step-{i} completed, no action required",
        )
    return Turn(user=text, tool=exchange)


def _question(*keys: str) -> str:
    listed = ", ".join(keys)
    return (
        "We are wrapping up. Report the current value of each of the following, "
        f"exactly as 'KEY = value', one per line: {listed}.\n"
        f"{QUERY_PREFIX} {listed}"
    )


def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    # 1. Early fact in plain user text.
    objective = (
        "Migrate the billing service to eu-west-2 and hand back a cutover report."
    )
    tasks.append(
        Task(
            id="early-constant",
            objective=objective,
            turns=(
                Turn(
                    user=(
                        f"{objective} The deploy token you will need at the very end "
                        "is issued once now and never repeated.\n"
                        "[FACT] DEPLOY_TOKEN = ZX-4417"
                    )
                ),
                _filler(1),
                _filler(2, tool=True),
                _filler(3),
                _filler(4, tool=True),
                _filler(5),
                _filler(6, tool=True),
                _filler(7),
            ),
            question=_question("DEPLOY_TOKEN"),
            expected={"DEPLOY_TOKEN": "ZX-4417"},
            tags=frozenset({"early", "text"}),
        )
    )

    # 2. Early fact that exists only inside a tool result.
    tasks.append(
        Task(
            id="early-tool-result",
            objective=objective,
            turns=(
                Turn(user=f"{objective} Start by reading the connection config."),
                Turn(
                    user="Pull the resolved config for the billing database.",
                    tool=ToolExchange(
                        name="read_file",
                        input={"path": "/etc/billing/resolved.conf"},
                        result=(
                            f"{LOG_LINES}"
                            "resolved connection string for the target region:\n"
                            "[FACT] DB_DSN = pg://eu-west-2.internal/billing\n"
                            f"{LOG_LINES}"
                        ),
                    ),
                ),
                _filler(2),
                _filler(3, tool=True),
                _filler(4),
                _filler(5, tool=True),
                _filler(6),
                _filler(7, tool=True),
            ),
            question=_question("DB_DSN"),
            expected={"DB_DSN": "pg://eu-west-2.internal/billing"},
            tags=frozenset({"early", "tool"}),
        )
    )

    # 3. The objective itself is what is needed late.
    drift_objective = (
        "Migrate the billing service to eu-west-2 WITHOUT taking a write outage; "
        "a read-only window is acceptable, a write outage is not.\n"
        "[FACT] OBJECTIVE = migrate-billing-eu-west-2-no-write-outage"
    )
    tasks.append(
        Task(
            id="objective-drift",
            objective=drift_objective,
            turns=(
                Turn(user=drift_objective),
                _filler(1, tool=True),
                _filler(2),
                _filler(3, tool=True),
                _filler(4),
                _filler(5, tool=True),
                _filler(6),
                _filler(7),
            ),
            question=_question("OBJECTIVE"),
            expected={"OBJECTIVE": "migrate-billing-eu-west-2-no-write-outage"},
            tags=frozenset({"early", "objective"}),
        )
    )

    # 4. Early value, corrected early. The stale value is still in the
    #    transcript, so "found something" is not the same as "found the truth".
    tasks.append(
        Task(
            id="corrected-fact",
            objective=objective,
            turns=(
                Turn(
                    user=(
                        f"{objective} The rollback plan is drafted.\n"
                        "[FACT] ROLLBACK_STEP = 3"
                    )
                ),
                Turn(
                    user=(
                        "Correction from the review: the rollback point moved after "
                        "the schema change was split in two.\n"
                        "[FACT] ROLLBACK_STEP = 5"
                    )
                ),
                _filler(2, tool=True),
                _filler(3),
                _filler(4, tool=True),
                _filler(5),
                _filler(6, tool=True),
                _filler(7),
            ),
            question=_question("ROLLBACK_STEP"),
            expected={"ROLLBACK_STEP": "5"},
            tags=frozenset({"early", "update"}),
        )
    )

    # 5. Thirteen early facts, the first one queried. Anything with a bounded
    #    durable store — summary budget or notes file — will evict it.
    flood_turns: list[Turn] = [
        Turn(
            user=(
                f"{objective} Here is the inventory of pinned versions.\n"
                "[FACT] PIN_00 = billing-api@1.9.3"
            ),
            tool=ToolExchange(
                name="write_note",
                input={"note": "[FACT] PIN_00 = billing-api@1.9.3"},
                result="note written",
            ),
        )
    ]
    for i in range(1, 13):
        flood_turns.append(
            Turn(
                user=(
                    f"Next pinned component. {LOREM}"
                    f"[FACT] PIN_{i:02d} = service-{i}@2.{i}.0"
                )
            )
        )
    flood_turns.append(_filler(13, tool=True))
    tasks.append(
        Task(
            id="fact-flood",
            objective=objective,
            turns=tuple(flood_turns),
            question=_question("PIN_00"),
            expected={"PIN_00": "billing-api@1.9.3"},
            tags=frozenset({"early", "flood"}),
        )
    )

    # 6. Moderate durable state: seven salient facts, the first one queried.
    #    Sits between the easy single-fact tasks and fact-flood, so the report
    #    shows where each bounded store starts to spill rather than only that
    #    it eventually does.
    accretion_turns: list[Turn] = [
        Turn(
            user=(
                f"{objective} Recording the cutover parameters as we go.\n"
                "[FACT] FREEZE_WINDOW = 02:00-04:00Z"
            )
        )
    ]
    accretion_params = [
        ("SHARD_COUNT", "24"),
        ("REPLICA_LAG_LIMIT", "900ms"),
        ("CUTOVER_LEAD", "harper"),
        ("DNS_TTL", "30s"),
        ("BACKOUT_BUCKET", "s3://billing-backout-eu"),
        ("PAGER_ROTA", "billing-primary"),
    ]
    for i, (key, value) in enumerate(accretion_params, start=1):
        accretion_turns.append(
            Turn(
                user=f"Next cutover parameter. {LOREM}[FACT] {key} = {value}",
                tool=ToolExchange(
                    name="write_note",
                    input={"note": f"[FACT] {key} = {value}"},
                    result="note written",
                )
                if i % 3 == 0
                else None,
            )
        )
    accretion_turns.append(_filler(8, tool=True))
    tasks.append(
        Task(
            id="state-accretion",
            objective=objective,
            turns=tuple(accretion_turns),
            question=_question("FREEZE_WINDOW"),
            expected={"FREEZE_WINDOW": "02:00-04:00Z"},
            tags=frozenset({"early", "accretion"}),
        )
    )

    # 7. Control: the fact is in the last turn before the question. Any
    #    strategy that fails this is broken, not lossy.
    tasks.append(
        Task(
            id="late-fact",
            objective=objective,
            turns=(
                Turn(user=objective),
                _filler(1, tool=True),
                _filler(2),
                _filler(3, tool=True),
                _filler(4),
                _filler(5, tool=True),
                _filler(6),
                Turn(
                    user=(
                        "Final sign-off just came in.\n"
                        "[FACT] APPROVER = sam.okafor"
                    )
                ),
            ),
            question=_question("APPROVER"),
            expected={"APPROVER": "sam.okafor"},
            tags=frozenset({"late", "control"}),
        )
    )

    return tasks


TASKS = build_tasks()

#: Tool definitions advertised to the model. `write_note` is the durable-state
#: tool the NoteTaking strategy harvests; the others exist so the transcripts
#: have realistic tool traffic.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "write_note",
        "description": (
            "Record a durable fact that must survive context compaction. "
            "Use for identifiers, decisions, and values you will need later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
    {
        "name": "read_logs",
        "description": "Read recent logs for a service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "window": {"type": "string"},
            },
            "required": ["service"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the deployment host.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]

SYSTEM = (
    "You are a long-running migration agent. Answer from the context you have. "
    "When asked to report values, output one 'KEY = value' line per key. If a "
    "value is not present in your context, output 'KEY = UNKNOWN' rather than "
    "guessing."
)

__all__ = [
    "Task",
    "Turn",
    "ToolExchange",
    "TASKS",
    "TOOLS",
    "SYSTEM",
    "QUERY_PREFIX",
    "build_tasks",
]
