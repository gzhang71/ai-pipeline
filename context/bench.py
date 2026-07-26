"""The benchmark runner and the comparison report.

Runs every strategy over every task and records, per strategy:

* task success (did the fact survive to the point where it was needed)
* total tokens — **`input + cache_creation + cache_read`, plus output**, and
  including the tokens the strategy spent on its own summarizer / note-writer
  calls. Both halves of that sentence matter: reading `input_tokens` alone
  understates a cache-warm run by an order of magnitude, and charging a
  summarizing strategy nothing for its summarizer rigs the comparison.
* number of model calls, split into agent calls and strategy calls
* wall-clock
* how often the strategy failed to get the request under budget at all

Run it:

    .venv/bin/python -m context.bench
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Protocol, Sequence

from common.client import MODEL, get_client, has_credentials

from .fakes import FakeClient
from .strategies import Budget, Strategy, all_strategies
from .tasks import SYSTEM, TASKS, TOOLS, Task
from .tokens import HeuristicTokenCounter, TokenCounter
from .usage import Usage
from .validation import Message, assert_valid, blocks, to_block


class RunnerClient(Protocol):
    def create(
        self,
        *,
        messages: Sequence[Message],
        system: Any = None,
        tools: Any = None,
        **overrides: Any,
    ) -> Any: ...


class LiveClient:
    """Adapter over the real SDK. Guarded by `has_credentials()`.

    Routes through `client.beta.messages.create` when a strategy asked for
    beta headers (compaction / context editing), and through the regular
    endpoint otherwise.
    """

    def __init__(self, *, model: str = MODEL, max_tokens: int = 2000):
        if not has_credentials():
            raise RuntimeError(
                "no credentials: run the bench with FakeClient, or authenticate"
            )
        self.model = model
        self.max_tokens = max_tokens

    def create(
        self,
        *,
        messages: Sequence[Message],
        system: Any = None,
        tools: Any = None,
        **overrides: Any,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": list(messages),
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        kwargs.update(overrides)
        client = get_client()
        if "betas" in kwargs:
            return client.beta.messages.create(**kwargs)
        return client.messages.create(**kwargs)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class TurnRecord:
    index: int
    fired: bool
    note: str
    tokens_before: int
    tokens_after: int
    prompt_tokens_billed: int
    over_budget: bool


@dataclass
class TaskResult:
    task_id: str
    strategy: str
    success: bool
    expected: dict[str, str]
    answer: str
    tags: frozenset[str]
    agent_usage: Usage
    strategy_usage: Usage
    wall_clock_s: float
    turns: list[TurnRecord] = field(default_factory=list)
    server_compactions: int = 0

    @property
    def total_usage(self) -> Usage:
        return self.agent_usage + self.strategy_usage

    @property
    def total_tokens(self) -> int:
        return self.total_usage.total_tokens

    @property
    def over_budget_turns(self) -> int:
        return sum(1 for t in self.turns if t.over_budget)

    @property
    def is_early(self) -> bool:
        return "early" in self.tags


@dataclass
class StrategySummary:
    strategy: str
    tasks: int
    successes: int
    total_tokens: int
    prompt_tokens: int
    output_tokens: int
    agent_calls: int
    strategy_calls: int
    wall_clock_s: float
    over_budget_turns: int
    failed_tasks: list[str]
    early_failures: list[str]
    early_tasks: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.tasks if self.tasks else 0.0

    @property
    def tokens_per_task(self) -> float:
        return self.total_tokens / self.tasks if self.tasks else 0.0

    @property
    def early_recall(self) -> float:
        if not self.early_tasks:
            return 0.0
        return (self.early_tasks - len(self.early_failures)) / self.early_tasks


@dataclass
class BenchReport:
    results: list[TaskResult]
    budget: Budget
    counter_name: str
    simulated: bool

    def summaries(self) -> list[StrategySummary]:
        by_strategy: dict[str, list[TaskResult]] = {}
        for result in self.results:
            by_strategy.setdefault(result.strategy, []).append(result)

        summaries: list[StrategySummary] = []
        for name, results in by_strategy.items():
            agent = sum((r.agent_usage for r in results), Usage())
            strat = sum((r.strategy_usage for r in results), Usage())
            total = agent + strat
            early = [r for r in results if r.is_early]
            summaries.append(
                StrategySummary(
                    strategy=name,
                    tasks=len(results),
                    successes=sum(1 for r in results if r.success),
                    total_tokens=total.total_tokens,
                    prompt_tokens=total.total_prompt_tokens,
                    output_tokens=total.output_tokens,
                    agent_calls=agent.model_calls,
                    strategy_calls=strat.model_calls,
                    wall_clock_s=sum(r.wall_clock_s for r in results),
                    over_budget_turns=sum(r.over_budget_turns for r in results),
                    failed_tasks=[r.task_id for r in results if not r.success],
                    early_failures=[r.task_id for r in early if not r.success],
                    early_tasks=len(early),
                )
            )
        # Rank by what the bench measures, in this order: success first
        # (a cheap strategy that loses the answer is not cheaper), then tokens.
        summaries.sort(key=lambda s: (-s.success_rate, s.total_tokens))
        return summaries

    def for_strategy(self, name: str) -> list[TaskResult]:
        return [r for r in self.results if r.strategy == name]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _user(text: str) -> Message:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _response_blocks(response: Any) -> list[dict[str, Any]]:
    return [to_block(b) for b in response.content]


def _response_text(response: Any) -> str:
    return "\n".join(
        str(b.get("text", ""))
        for b in _response_blocks(response)
        if b.get("type") == "text"
    )


def run_task(
    task: Task,
    strategy: Strategy,
    *,
    client: RunnerClient,
    budget: Budget,
    append_full_content: bool = True,
    validate_requests: bool = True,
    system: str = SYSTEM,
    tools: list[dict[str, Any]] | None = None,
) -> TaskResult:
    """Replay one long-horizon task under one strategy.

    The transcript is scripted (the same tool calls and tool results for every
    strategy, so the comparison is apples to apples); the model call at the end
    of each turn is real, and the final answer is whatever the model can say
    given the context the strategy left it.

    `append_full_content=False` reproduces the classic compaction bug — append
    only the extracted text and the server's `compaction` block is thrown away.
    """
    tools = TOOLS if tools is None else tools
    strategy.reset()
    task_budget = replace(budget, objective=task.objective)

    messages: list[Message] = []
    agent_usage = Usage()
    strategy_usage = Usage()
    records: list[TurnRecord] = []
    started = time.perf_counter()

    def step(index: int) -> Any:
        nonlocal agent_usage, strategy_usage, messages
        before = task_budget.count(messages)
        result = strategy.apply(messages, task_budget.for_turn(index))
        strategy_usage = strategy_usage + result.usage
        after = task_budget.count(result.messages)

        if validate_requests:
            assert_valid(
                result.messages,
                label=f"{strategy.name}/{task.id} turn {index}",
            )

        response = client.create(
            messages=result.messages,
            system=system,
            tools=tools,
            **result.request_overrides,
        )
        call_usage = Usage.from_response_usage(response.usage)
        agent_usage = agent_usage + call_usage

        records.append(
            TurnRecord(
                index=index,
                fired=result.fired,
                note=result.note,
                tokens_before=before,
                tokens_after=after,
                prompt_tokens_billed=call_usage.total_prompt_tokens,
                over_budget=call_usage.total_prompt_tokens > task_budget.max_tokens,
            )
        )

        content = _response_blocks(response)
        if not append_full_content:
            content = [b for b in content if b.get("type") == "text"]
        if content:
            messages.append({"role": "assistant", "content": content})
        return response

    for index, turn in enumerate(task.turns):
        messages.append(_user(turn.user))
        if turn.tool is not None:
            tool_id = f"toolu_{task.id}_{index}"
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": turn.tool.name,
                            "input": turn.tool.input,
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": turn.tool.result,
                        }
                    ],
                }
            )
        step(index)

    messages.append(_user(task.question))
    final = step(len(task.turns))
    answer = _response_text(final)

    success = all(f"{key} = {value}" in answer for key, value in task.expected.items())

    return TaskResult(
        task_id=task.id,
        strategy=strategy.name,
        success=success,
        expected=dict(task.expected),
        answer=answer,
        tags=task.tags,
        agent_usage=agent_usage,
        strategy_usage=strategy_usage,
        wall_clock_s=time.perf_counter() - started,
        turns=records,
        server_compactions=getattr(client, "compactions", 0),
    )


def run_bench(
    *,
    strategies: Iterable[Strategy] | None = None,
    tasks: Iterable[Task] | None = None,
    budget: Budget | None = None,
    client_factory: Callable[[], RunnerClient] | None = None,
    counter: TokenCounter | None = None,
    append_full_content: bool = True,
) -> BenchReport:
    """Every strategy x every task. A fresh client per task run."""
    counter = counter or HeuristicTokenCounter()
    budget = budget or Budget(counter=counter)
    strategies = list(strategies) if strategies is not None else all_strategies()
    tasks = list(tasks) if tasks is not None else TASKS
    simulated = client_factory is None
    # Align the simulated server-side trigger with the client-side budget, so
    # ServerCompaction is not flattered (or punished) by firing on a different
    # threshold than every other strategy.
    budget_tokens = budget.max_tokens
    client_factory = client_factory or (
        lambda: FakeClient(counter, compaction_threshold=budget_tokens)
    )

    results: list[TaskResult] = []
    for strategy in strategies:
        for task in tasks:
            results.append(
                run_task(
                    task,
                    strategy,
                    client=client_factory(),
                    budget=budget,
                    append_full_content=append_full_content,
                )
            )

    return BenchReport(
        results=results,
        budget=budget,
        counter_name=getattr(counter, "name", type(counter).__name__),
        simulated=simulated,
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def format_report(report: BenchReport) -> str:
    summaries = report.summaries()
    task_ids = sorted({r.task_id for r in report.results})

    lines: list[str] = []
    lines.append("compaction strategy bench")
    lines.append("=" * 96)
    lines.append(
        f"budget: {report.budget.max_tokens} tokens or {report.budget.max_turns} turns"
        f" (whichever trips first) | keep_recent={report.budget.keep_recent_messages}"
        f" | counter={report.counter_name}"
    )
    lines.append(
        f"{len(summaries)} strategies x {len(task_ids)} tasks"
        + ("  |  offline: fake client, simulated server compaction" if report.simulated else "")
    )
    lines.append("")

    header = (
        f"{'strategy':<24}{'success':>9}{'early':>8}{'tokens':>10}"
        f"{'tok/task':>10}{'calls':>7}{'strat':>7}{'over':>6}{'ms':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        lines.append(
            f"{s.strategy:<24}"
            f"{s.successes:>4}/{s.tasks:<4}"
            f"{s.early_recall * 100:>7.0f}%"
            f"{s.total_tokens:>10,}"
            f"{s.tokens_per_task:>10,.0f}"
            f"{s.agent_calls:>7}"
            f"{s.strategy_calls:>7}"
            f"{s.over_budget_turns:>6}"
            f"{s.wall_clock_s * 1000:>9.1f}"
        )
    lines.append("")
    lines.append(
        "success = tasks whose late question was answered correctly | "
        "early = recall on tasks whose answer was introduced early"
    )
    lines.append(
        "tokens  = input + cache_creation + cache_read + output, agent calls AND "
        "strategy's own summarizer/note-writer calls"
    )
    lines.append(
        "calls/strat = model calls made by the agent loop / by the strategy itself | "
        "over = turns whose billed prompt still exceeded the budget"
    )
    lines.append("")

    # Per-task grid: which strategy lost what.
    grid_header = f"{'strategy':<24}" + "".join(f"{tid[:16]:>18}" for tid in task_ids)
    lines.append("per-task outcome (. = solved, X = lost)")
    lines.append(grid_header)
    lines.append("-" * len(grid_header))
    for s in summaries:
        row = f"{s.strategy:<24}"
        for tid in task_ids:
            row += f"{('X' if tid in s.failed_tasks else '.'):>18}"
        lines.append(row)
    lines.append("")

    lines.append("early-information loss")
    lines.append("-" * 96)
    any_loss = False
    for s in summaries:
        if s.early_failures:
            any_loss = True
            lines.append(
                f"  {s.strategy:<24} lost {len(s.early_failures)}/{s.early_tasks}: "
                + ", ".join(s.early_failures)
            )
    for s in summaries:
        if not s.early_failures:
            lines.append(f"  {s.strategy:<24} kept every early fact in this task set")
    if not any_loss:
        lines.append("  (no strategy lost early information — the task set is too easy)")
    lines.append("")

    lines.append("read this with the caveats in context/README.md:")
    lines.append(
        "  - offline runs use a heuristic token counter and fake summarizer/note-writer;"
    )
    lines.append(
        "    the ranking is conditioned on their loss models (fixed fact budget,"
    )
    lines.append(
        "    salience-aware, no paraphrase drift). Re-run against the API before believing it."
    )
    if report.simulated:
        lines.append(
            "  - ServerCompaction's numbers are SIMULATED by the fake client, not measured."
        )
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - entry point
    report = run_bench()
    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
