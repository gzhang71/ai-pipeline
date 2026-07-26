#!/usr/bin/env python
"""Six ways to keep a conversation inside its context budget.

Shows the strategy interface on a single over-budget history, checks that
each strategy's output is still a legal request, then runs the full bench.

Runs offline against fakes. No API key, no network.

    .venv/bin/python examples/04_compaction_bench.py
"""

from __future__ import annotations

from context import (
    COMPACTION_BETA,
    COMPACTION_EDIT,
    CONTEXT_EDITING_BETA,
    CLEAR_TOOL_USES_EDIT,
    Budget,
    HeuristicTokenCounter,
    ToolResultEviction,
    all_strategies,
    format_report,
    is_valid,
    run_bench,
    validate,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ------------------------------------------------------------ 1. a history
rule("1. An over-budget history with a tool exchange in it")

counter = HeuristicTokenCounter()


def make_history() -> list[dict]:
    """A long-horizon conversation: the answer is introduced early and asked for late.

    Note the plain text turns between the tool exchanges. A history of nothing
    but back-to-back tool_use/tool_result pairs has almost no legal place to
    split -- a tail cannot begin on an orphaned tool_result -- so summarizing
    strategies would find "nothing old enough to summarize" and pass it through.
    """
    h = [{"role": "user", "content": "Track invoice INV-1003 to completion. " + "background. " * 60}]
    h += [{"role": "assistant", "content": "Understood. I'll start with the ledger."}]
    for i in range(3):
        h.append({"role": "user", "content": f"Check step {i}. " + "detail " * 40})
        h.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Checking step {i}."},
                    {"type": "tool_use", "id": f"toolu_{i}", "name": "lookup", "input": {"step": i}},
                ],
            }
        )
        h.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{i}",
                        "content": f"step {i}: " + "value " * 50,
                    }
                ],
            }
        )
        h.append({"role": "assistant", "content": f"Step {i} done."})
    h.append({"role": "user", "content": "Remind me what the invoice amount was?"})
    return h


history = make_history()
before = counter.count(history)
print(f"{len(history)} messages, ~{before:,} tokens")
print(f"valid request shape: {is_valid(history)}")


# ----------------------------------------------------- 2. every strategy runs
rule("2. Each strategy applied to that history")

budget = Budget(
    max_tokens=600,
    max_turns=20,
    turn_index=len(history),
    keep_recent_messages=4,
    objective="Track invoice INV-1003.",
    counter=counter,
)

strategies = all_strategies(offline=True)

print(f"{'strategy':<24} {'tokens':>8} {'delta':>8}  valid  calls  note")
print("-" * 70)
for strat in strategies:
    result = strat.apply(list(history), budget)
    after = counter.count(result.messages)
    # Server-side compaction edits nothing locally; its contribution is the
    # request override it asks for.
    override = "  (via request override)" if result.request_overrides else ""
    problems = validate(result.messages)
    print(
        f"{type(strat).__name__:<24} {after:>8,} {after - before:>+8,}"
        f"  {'ok' if not problems else 'INVALID':<6} {result.usage.model_calls:>4}   "
        f"{result.note}{override}"
    )

print(
    "\nEvery output above is still a legal request: first message is a user turn,"
    "\nevery tool_use has a matching tool_result, no trailing assistant prefill."
    "\nA strategy that produces a 400 is worse than useless, so the runner"
    "\nasserts validity on every request before sending it."
)


# --------------------------------------------------- 3. eviction keeps pairing
rule("3. The pairing detail that makes eviction safe")

evicted = ToolResultEviction().apply(list(history), budget)
for msg in evicted.messages:
    content = msg["content"]
    if isinstance(content, list):
        for block in content:
            if block.get("type") == "tool_result":
                body = str(block.get("content"))
                print(f"  tool_result for {block['tool_use_id']}: {body[:70]}")
            elif block.get("type") == "tool_use":
                print(f"  tool_use   {block['id']} -> {block['name']}")

print(
    "\nThe tool_use block survives and the payload is replaced with a placeholder."
    "\nDropping the tool_use instead would orphan the tool_result and the API"
    "\nwould reject the request."
)


# ------------------------------------------------------ 4. two distinct APIs
rule("4. Compaction and context editing are different features")

print(f"  compaction      beta={COMPACTION_BETA!r:28} edit={COMPACTION_EDIT!r}")
print(f"  context editing beta={CONTEXT_EDITING_BETA!r:28} edit={CLEAR_TOOL_USES_EDIT!r}")
print(
    "\nCompaction summarizes history; context editing clears tool results outright."
    "\nThey take different beta headers and are easy to conflate."
)


# ----------------------------------------------------------------- 5. the bench
rule("5. The full bench")

print(format_report(run_bench()))
