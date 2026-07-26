#!/usr/bin/env python
"""Where did the context window actually go?

Runs an instrumented tool-use loop, then breaks each turn's prompt down by
segment -- system, tool schemas, user text, assistant text, tool_use,
tool_result -- and reconciles that breakdown against the response's usage.

Runs offline against fakes. No API key, no network.

    .venv/bin/python examples/03_context_profile.py
"""

from __future__ import annotations

from pathlib import Path

from loop import (
    ATTRIBUTION_METHOD,
    JsonlSink,
    LoopConfig,
    MemorySink,
    analyze_accuracy,
    observations_from_records,
    render_html,
    read_jsonl,
    render_text,
    report_text,
    SYNTHETIC_TASKS,
    run_loop,
    run_task_set,
    validate_run,
)
from loop.testing import (
    FakeAnthropicClient,
    FakeMessage,
    echo_executor,
    heuristic_token_count,
    text_block,
    tool_use_block,
)

OUT = Path("runs/example-03")
TOOLS = [
    {
        "name": "lookup_invoice",
        "description": "Fetch an invoice by id. Call this whenever the user names an invoice.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    }
]


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ------------------------------------------------------------ 1. a single run
rule("1. One instrumented run: two turns, one tool call")

script = [
    FakeMessage(
        content=[
            text_block("Let me look that invoice up."),
            tool_use_block("lookup_invoice", {"id": "INV-1003"}, "toolu_1"),
        ],
        stop_reason="tool_use",
    ),
    FakeMessage(content=[text_block("Invoice INV-1003 is for 1204000 cents.")]),
]

sink = MemorySink()
result = run_loop(
    client=FakeAnthropicClient(script, cache_read_fraction=0.9),
    tools=TOOLS,
    executor=echo_executor,
    prompt="What is the amount in cents on invoice INV-1003?",
    config=LoopConfig(system="You are a billing assistant.", max_iterations=6),
    counter=heuristic_token_count,
    sink=sink,
)

print(f"turns: {result.turns}   stop reason: {result.stop_reason}")
print(f"final text: {result.final_text!r}")
print(f"attribution method: {ATTRIBUTION_METHOD}")


# -------------------------------------------------------- 2. the decomposition
rule("2. Per-turn segment breakdown, and its reconciliation")

turns = [r for r in sink.records if r["record_type"] == "turn"]
for rec in turns:
    pt = rec["prompt_tokens"]
    rc = rec["reconciliation"]
    print(f"\nturn {rec['turn_index']}  counted {pt['counted_total']:,} tokens")
    for seg in pt["segments"]:
        flag = "  ~approx" if seg.get("approximate") else ""
        print(f"    {seg['kind']:<16} {seg['tokens']:>7,}{flag}")
    print(f"    {'-' * 24}")
    print(f"    {'segment_sum':<16} {pt['segment_sum']:>7,}   residual={pt['decomposition_residual']}")
    print(
        f"    vs usage total {rc['authoritative_total']:>7,}  "
        f"residual={rc['residual_tokens']:+,} ({rc['residual_fraction']:.2%}) "
        f"within_tolerance={rc['within_tolerance']}"
    )

print(
    "\nThe decomposition is exactly additive by construction: segments are measured"
    "\nas deltas along a growing chain of prefixes, so they telescope. That is"
    "\narithmetic, not a claim that each number is a segment's 'true' cost -- the"
    "\nreal error is order-dependence, which is why measurement_order is recorded."
)
header = next(r for r in sink.records if r["record_type"] == "run_header")
print(f"\nmeasurement order: {header['attribution']['measurement_order']}")


# ----------------------------------------------------------------- 3. renderer
rule("3. Rendering, and the accuracy-vs-length analysis")

OUT.mkdir(parents=True, exist_ok=True)

print(render_text(sink.records))

# A fuller sweep: the same question asked with increasing filler, so prompt
# length varies while the task stays fixed.
sweep_path = OUT / "sweep.jsonl"
if sweep_path.exists():
    sweep_path.unlink()
sweep_sink = JsonlSink(sweep_path)


# One client for the whole sweep: the script is consumed across runs, so it
# needs one scripted reply per task.
answering_client = FakeAnthropicClient(
    [FakeMessage(content=[text_block("The amount is 1204000 cents.")]) for _ in SYNTHETIC_TASKS],
    cache_read_fraction=0.5,
)

runs = run_task_set(
    client=answering_client,
    executor=echo_executor,
    tools=TOOLS,
    counter=heuristic_token_count,
    sink=sweep_sink,
    config=LoopConfig(system="You are a billing assistant.", max_iterations=4),
)
sweep_sink.close()

print(f"\nswept {len(runs)} tasks across increasing filler lengths")
for r in runs:
    print(f"  {r.task.task_id:<24} filler={r.task.filler_tokens:>6,}  success={r.success}")

records = read_jsonl(sweep_path)
validate_run(records)
print(f"\nJSONL validates against the documented schema: {sweep_path}")

outcomes = {r.result.run_id: r.success for r in runs}
observations = observations_from_records(records, outcomes)
report = analyze_accuracy(observations)
print()
print(report_text(report))

html = OUT / "profile.html"
html.write_text(render_html(records, title="example 03", accuracy=report))
print(f"\nself-contained HTML written to {html}")
print("(no external requests -- inline SVG and CSS only)")
