"""A tiny synthetic task set with checkable outcomes.

Its only job is to exercise the accuracy-vs-length path end to end. It is
deliberately built so that **length is the only thing that varies**: the same
question, the same tool, the same correct answer, with a controlled amount of
irrelevant filler padded into the system prompt. That is what makes a bend in
the curve attributable to context length rather than to task difficulty.

Swap in a real task set for real conclusions. Six synthetic tasks prove the
plumbing, not a model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .accuracy import Observation
from .agent import LoopConfig, RunResult, run_loop

#: A ledger the model can only read through the tool -- the answer is nowhere
#: in the prompt, so a run that gets it right actually used the tool.
LEDGER: dict[str, dict[str, Any]] = {
    "INV-1001": {"customer": "Northwind", "amount_cents": 428_150, "status": "paid"},
    "INV-1002": {"customer": "Contoso", "amount_cents": 91_700, "status": "open"},
    "INV-1003": {"customer": "Fabrikam", "amount_cents": 1_204_000, "status": "open"},
    "INV-1004": {"customer": "Tailspin", "amount_cents": 55_025, "status": "void"},
}

LOOKUP_TOOL: dict[str, Any] = {
    "name": "lookup_invoice",
    "description": (
        "Look up one invoice in the ledger by its identifier. Call this "
        "whenever you need an invoice's customer, amount or status; the "
        "ledger is not present in the conversation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "invoice_id": {
                "type": "string",
                "description": "Invoice identifier, e.g. INV-1001",
            }
        },
        "required": ["invoice_id"],
        "additionalProperties": False,
    },
}

TOOLS: list[dict[str, Any]] = [LOOKUP_TOOL]

BASE_SYSTEM = (
    "You are a ledger assistant. Use the lookup_invoice tool to read invoice "
    "records; never guess. When you have the answer, reply with exactly the "
    "amount in cents as a bare integer and nothing else."
)

_FILLER_VOCAB = (
    "policy revision archive appendix clause schedule addendum exhibit "
    "memorandum retention disclosure indemnity covenant remittance "
    "reconciliation amortization provision"
).split()


def filler_text(approx_tokens: int, *, seed: str = "loop") -> str:
    """Deterministic irrelevant prose of roughly ``approx_tokens`` tokens.

    Padding, not a distractor: it never mentions an invoice id or an amount,
    so it cannot change what the correct answer is -- only how far away it is.
    """
    if approx_tokens <= 0:
        return ""
    digest = hashlib.sha256(seed.encode()).digest()
    words: list[str] = []
    # ~1.3 tokens per word for this vocabulary; overshoot then trim by chars.
    target_words = max(1, int(approx_tokens / 1.3))
    for index in range(target_words):
        pick = digest[index % len(digest)] ^ (index * 31 & 0xFF)
        words.append(_FILLER_VOCAB[pick % len(_FILLER_VOCAB)])
    return " ".join(words)


@dataclass(frozen=True)
class Task:
    """One checkable task at one context length."""

    task_id: str
    question: str
    expected: str
    filler_tokens: int = 0
    family: str = "invoice_amount"

    def system_prompt(self) -> str:
        pad = filler_text(self.filler_tokens, seed=self.family)
        if not pad:
            return BASE_SYSTEM
        return (
            f"{BASE_SYSTEM}\n\n<reference_material>\n{pad}\n"
            "</reference_material>\n\nIgnore the reference material unless the "
            "question refers to it."
        )

    def check(self, final_text: str) -> bool:
        """Outcome check: the last integer in the reply must be the answer."""
        numbers = re.findall(r"-?\d[\d,_]*", final_text or "")
        if not numbers:
            return False
        cleaned = numbers[-1].replace(",", "").replace("_", "")
        return cleaned == self.expected

    def to_metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "filler_tokens": self.filler_tokens,
            "expected": self.expected,
        }


#: The same question at six padding levels. Length is the independent variable.
SYNTHETIC_TASKS: list[Task] = [
    Task(
        task_id=f"invoice_amount@{pad}",
        question="What is the amount in cents on invoice INV-1003?",
        expected=str(LEDGER["INV-1003"]["amount_cents"]),
        filler_tokens=pad,
    )
    for pad in (0, 200, 800, 2_000, 6_000, 16_000)
]


def make_executor(ledger: dict[str, dict[str, Any]] | None = None) -> Callable[..., str]:
    """Build a tool executor over a ledger. Unknown ids return an error result."""
    table = ledger if ledger is not None else LEDGER

    def execute(name: str, tool_input: dict[str, Any], tool_use_id: str) -> str:
        if name != LOOKUP_TOOL["name"]:
            raise ValueError(f"unknown tool {name!r}")
        invoice_id = str(tool_input.get("invoice_id", "")).strip().upper()
        record = table.get(invoice_id)
        if record is None:
            raise KeyError(f"no such invoice {invoice_id!r}")
        return (
            f"invoice_id={invoice_id} customer={record['customer']} "
            f"amount_cents={record['amount_cents']} status={record['status']}"
        )

    return execute


@dataclass
class TaskRun:
    task: Task
    result: RunResult
    success: bool

    def observation(self) -> Observation:
        return Observation(
            run_id=self.result.run_id,
            task_id=self.task.task_id,
            prompt_tokens=self.result.peak_prompt_tokens,
            success=self.success,
            turns=self.result.turns,
            metadata=self.task.to_metadata(),
        )


def run_task_set(
    *,
    client: Any,
    tasks: Sequence[Task] | None = None,
    executor: Callable[..., Any] | None = None,
    config: LoopConfig | None = None,
    sink: Any = None,
    counter: Any = None,
    tools: Sequence[dict[str, Any]] | None = None,
) -> list[TaskRun]:
    """Run every task and check its outcome.

    Returns one ``TaskRun`` per task; feed ``[r.observation() for r in runs]``
    to ``loop.accuracy.analyze_accuracy``.
    """
    tasks = list(tasks if tasks is not None else SYNTHETIC_TASKS)
    executor = executor or make_executor()
    base = config or LoopConfig(max_iterations=6)
    runs: list[TaskRun] = []
    for task in tasks:
        task_config = LoopConfig(
            **{**base.__dict__, "system": task.system_prompt()}
        )
        result = run_loop(
            client=client,
            tools=list(tools if tools is not None else TOOLS),
            executor=executor,
            prompt=task.question,
            config=task_config,
            sink=sink,
            counter=counter,
            task=task.to_metadata(),
        )
        runs.append(TaskRun(task=task, result=result, success=task.check(result.final_text)))
    return runs
