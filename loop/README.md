# `loop` — context budget profiler

An instrumented agent loop that answers **"where did my context window actually go?"**

This is the measurement instrument the rest of the repo depends on. Everything
else in this repo is a *claim* about context; this is what tests those claims.
It prioritises accuracy of accounting over features.

For every turn of a tool-use loop it records a per-segment breakdown of the
prompt it sent — system prompt, tool schemas, and each message's contribution
split by kind — and reconciles that breakdown against the authoritative `usage`
on the response. Records go to JSONL with a versioned, documented schema.

---

## Contents

| Module | What it is |
|---|---|
| `agent.py` | `run_loop` — the instrumented request → tool_use → execute → tool_result cycle |
| `attribution.py` | The token-attribution engine and the `count_tokens` wrappers |
| `schema.py` | The public JSONL schema, its constants, and its validators |
| `sink.py` | `JsonlSink`, `MemorySink`, `read_jsonl`, `write_jsonl` |
| `render.py` | `render_text` and `render_html` — cumulative breakdown across turns |
| `accuracy.py` | Accuracy vs. context length: binning, Wilson intervals, bend detection |
| `tasks.py` | A tiny synthetic task set with checkable outcomes |
| `testing.py` | Offline doubles: a fake client and a fake token counter |

---

## Usage

### Offline (no credentials needed)

```python
from loop import LoopConfig, MemorySink, render_text, run_loop
from loop.testing import (
    FakeAnthropicClient, FakeMessage, echo_executor,
    heuristic_token_count, text_block, tool_use_block,
)

TOOLS = [{
    "name": "lookup",
    "description": "Look something up.",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}]

sink = MemorySink()
result = run_loop(
    client=FakeAnthropicClient([
        FakeMessage(content=[tool_use_block("lookup", {"q": "7781"}, "toolu_1")],
                    stop_reason="tool_use"),
        FakeMessage(content=[text_block("41250")], stop_reason="end_turn"),
    ]),
    tools=TOOLS,
    executor=echo_executor,
    prompt="What is on invoice 7781?",
    config=LoopConfig(system="You are a ledger assistant."),
    sink=sink,
    counter=heuristic_token_count,
)
print(render_text(sink.records))
```

### Against the live API

Identical, but pass the real client and drop the fake counter (the default
counter is the `count_tokens` endpoint):

```python
from common.client import get_client, has_credentials
from loop import JsonlSink, LoopConfig, run_loop

assert has_credentials()
with JsonlSink("runs/session.jsonl") as sink:
    result = run_loop(
        client=get_client(),
        tools=TOOLS,
        executor=my_executor,
        prompt="…",
        config=LoopConfig(system=MY_SYSTEM, effort="high", max_iterations=12),
        sink=sink,
    )
```

`client` needs only `client.messages.create(**kwargs)`; anything with that shape
works, which is how the offline tests drive it.

### Reading a run back

```python
from loop import analyze_accuracy, observations_from_records, read_jsonl, render_html

records = read_jsonl("runs/session.jsonl")          # validates the whole stream
open("report.html", "w").write(render_html(records))
```

### The tool executor

```python
def executor(name: str, tool_input: dict, tool_use_id: str) -> str | dict | list | ToolResult:
    ...
```

Return a string (or any JSON-able value) for success; return
`ToolResult(content=..., is_error=True)` to mark an error result. By default a
raised exception is converted into an error `tool_result` and handed back to the
model rather than aborting the run — set `LoopConfig(catch_tool_errors=False)`
to abort instead (the failure is still recorded on the run footer).

### Cost of instrumentation

Attribution costs `count_tokens` calls, not `messages` calls. At the default
`block_group` granularity a turn costs roughly one call per *new* block group,
because a single `CachingTokenCounter` is shared for the whole run and turn
N+1's prompt prefixes are turn N's. Drop to `attribution="coarse"` (4 calls per
turn) or `"off"` (1 call) if that matters.

---

## Token attribution

### The problem

Token counts are **not additive across an arbitrary split of a prompt**.
Counting the system prompt alone plus the messages alone does not reproduce the
count of the whole: the tokenizer merges across the boundary, and the API wraps
each part in framing that the parts don't carry on their own. Any decomposition
that pretends otherwise is lying, so this one doesn't.

All counts come from the `count_tokens` endpoint. Never from a client-side
estimator — tiktoken is OpenAI's tokenizer and undercounts Claude badly,
especially on code.

### The method: `incremental_prefix_delta`

No segment is ever counted in isolation. We count a strictly growing chain of
*prefixes of the real request* and attribute each segment the delta it caused:

```
p0  = count(messages=[PROBE])                     -> "framing"
q1  = count(messages=prefix_through(group_1))     -> group_1 = q1 - p0
q2  = count(messages=prefix_through(group_2))     -> group_2 = q2 - q1
…
qN  = count(messages=all)
t   = count(messages=all, tools=T)                -> tool_schemas  = t - qN
s   = count(messages=all, tools=T, system=S)      -> system_prompt = s - t
```

A "group" is a contiguous run of same-kind content blocks within one message
(`block_group` granularity), or a whole message (`message`). Prefix truncation
is always safe for tool pairing: a prefix can hold a `tool_use` whose
`tool_result` has not arrived yet — the normal mid-turn state — but never the
reverse.

Segments are **reported** in the API's render order (tools → system → messages);
the measurement order above is recorded on every run header as
`attribution.measurement_order`.

### Error characteristics — read this part

**1. The decomposition is exactly additive, by construction.** A telescoping sum
of deltas equals its last term, so `segment_sum == counted_total` *exactly*, and
`decomposition_residual` is always `0` for this method. The schema validator
enforces the identity on every record. This is a property of the arithmetic, not
a claim that each segment's number is the "true" cost of that segment.

**2. Deltas are marginal, and therefore order-dependent.** A segment's number is
its cost *given everything measured before it*, not an intrinsic cost. Measure
the same segments in a different order and you get slightly different numbers.
This is the real error term. It is now measurable rather than merely declared:
`attribute(..., measurement_order=...)` takes any permutation of `messages`,
`tool_schemas`, `system_prompt` (framing is always the base), and
`order_sensitivity()` sweeps all six and reports per-kind `min`/`max`/`spread`:

```python
from loop import order_sensitivity, order_sensitivity_text
print(order_sensitivity_text(order_sensitivity(messages, system=..., tools=...)))
```

The total is invariant under every order — only the split between segments
moves. Treat `max_spread_fraction` as the error bar on any single segment's
number.

⚠️ **Run it against `api_token_counter` or the result is meaningless.** Any
chars-per-token heuristic — including this package's offline test double — is
linear, and a linear counter is order-independent by construction, so it always
reports zero spread. That is a property of the counter, not evidence about the
real tokenizer. The report says which case you are in via
`counter_is_order_sensitive`.

The measurement order stays on the header rather than buried. Where the
tokenizer merges across a boundary a delta can come out **negative**; we report
negatives as-is rather than clamping, and count them in
`prompt_tokens.negative_segments`. A nonzero count there is a signal that a
boundary is merging, not a bug.

**3. `framing` is the one genuinely approximate segment.** The API requires at
least one message, so no prefix exists that contains framing and nothing else.
We probe with a minimal one-character user message. `framing` is therefore
**high**, and the first message segment **low**, by that probe's own cost
(single-digit tokens). The segment carries `approximate: true`, the header lists
`attribution.approximate_segments: ["framing"]`, and the segment's `note` field
says so in words. Nothing else is approximate.

**4. The residual that actually matters is against `usage`.** `count_tokens` and
billed usage are computed by different code paths and routinely differ by a few
tokens on an identical prompt. Every turn record therefore carries a
`reconciliation` block comparing our `counted_total` against the response's
authoritative total. **That total is `input_tokens + cache_creation_input_tokens
+ cache_read_input_tokens`** — `usage.input_tokens` alone is the *uncached
remainder*, and reading it as "the prompt size" will understate a cache-warm
request by an order of magnitude. `within_tolerance` compares
`|residual_fraction|` against `tolerance_fraction` (default 2%, configurable via
`LoopConfig.reconcile_tolerance`).

**Summary of what to trust.** Segment *shares* (which part of the window is
eating it, and how that changes across turns) are solid. Individual segment
counts are marginal costs with a small order-dependent bias, plus a single-digit
probe bias on `framing`. Totals are exact against `count_tokens` and within the
stated tolerance of billed usage.

---

## JSONL schema

`schema_id: "ai-pipeline/loop/context-budget"`, `schema_version: 1`.

A run is a JSONL file with exactly this record sequence:

```
run_header      exactly one, first
turn            zero or more, turn_index 0..N-1, contiguous, ascending
run_footer      exactly one, last
```

Several runs may be concatenated in one file, as long as each run's records are
adjacent and internally contiguous. `loop.schema.group_runs` splits a stream
back into `{header, turns, footer}` dicts; `validate_run` checks order, turn
numbering and `run_id` agreement, not just per-record shape.

Every record carries the same four envelope fields.

| Field | Type | Meaning |
|---|---|---|
| `schema_id` | str | Always `"ai-pipeline/loop/context-budget"`. Reject anything else. |
| `schema_version` | int | Currently `1`. Bumped only when a consumer must change. |
| `record_type` | str | `"run_header"` \| `"turn"` \| `"run_footer"` |
| `run_id` | str | Stable per run, e.g. `"run_1a2b3c4d5e6f"` |

### `run_header`

| Field | Type | Meaning |
|---|---|---|
| `started_at` | str | ISO 8601 UTC, millisecond precision |
| `lib_version` | str | Version of this package that produced the run |
| `model` | str | Model id passed to `messages.create` |
| `max_iterations` | int | The runaway guard's bound |
| `max_tokens` | int | `max_tokens` sent on every request |
| `effort` | str \| null | `output_config.effort`, or null if not sent |
| `thinking` | object \| null | The `thinking` param as sent, or null if omitted |
| `tool_names` | list[str] | Names of the tools offered, in the order sent |
| `system_fingerprint` | object | `{present: bool, sha256: str\|null, chars: int}` — identifies the system prompt without copying it |
| `attribution` | object | See below |
| `task` | object \| null | Caller-supplied task metadata. When present, `task_id` is required; the accuracy analysis also reads `family` |

`attribution`:

| Field | Type | Meaning |
|---|---|---|
| `method` | str | `"incremental_prefix_delta"` |
| `granularity` | str | `"block_group"` \| `"message"` \| `"coarse"` \| `"off"` |
| `counter` | str | Which token counter was used (`"api_token_counter"` on live runs) |
| `measurement_order` | list[str] | The order prefixes were measured in — deltas are order-dependent, so this is load-bearing |
| `approximate_segments` | list[str] | Segment ids whose value is approximate. Currently always `["framing"]` |
| `reconcile_tolerance_fraction` | float | Threshold used for `reconciliation.within_tolerance` |

### `turn`

One record per request/response round trip.

| Field | Type | Meaning |
|---|---|---|
| `turn_index` | int | 0-based, contiguous within the run |
| `started_at` / `ended_at` | str | ISO 8601 UTC |
| `duration_ms` | float | Wall clock for the `messages.create` call |
| `request` | object | `{model, n_messages, n_tools, max_tokens, effort}`. `n_messages` is the count *sent*, excluding the response |
| `prompt_tokens` | object | The decomposition. See below |
| `usage` | object | Normalized from the response. See below |
| `reconciliation` | object | Decomposition vs. authoritative usage. See below |
| `response` | object | `{stop_reason, n_tool_use, text_chars, model}` |
| `tool_calls` | list[object] | One per `tool_use` block executed this turn |

`prompt_tokens`:

| Field | Type | Meaning |
|---|---|---|
| `counted_total` | int | `count_tokens` over the full prompt as sent |
| `segments` | list[object] | The decomposition, in render order |
| `by_kind` | object | `{kind: tokens}` — `segments` aggregated by kind, ordered as `SEGMENT_KINDS` |
| `segment_sum` | int | Sum of `segments[].tokens` |
| `decomposition_residual` | int | `counted_total - segment_sum`. Always `0` for this method; present so a future non-additive method can report honestly without a schema break |
| `counter_calls` | int | `count_tokens` calls this turn actually caused (cache misses only) |
| `negative_segments` | int | How many segments came out negative (boundary token merging) |

`segments[]`:

| Field | Type | Meaning |
|---|---|---|
| `segment_id` | str | Unique within the turn. `"framing"`, `"tool_schemas"`, `"system_prompt"`, or `"m{msg}:{start}-{end}:{kind}"` |
| `kind` | str | One of `SEGMENT_KINDS` (below) |
| `tokens` | int | Marginal token cost. May be negative |
| `message_index` | int \| null | Index into the messages sent, for message segments |
| `role` | str \| null | `"user"` or `"assistant"`, for message segments |
| `block_span` | [int, int] | Half-open content-block range within that message |
| `approximate` | bool | True only for `framing` |
| `note` | str | Present when there is a caveat worth carrying |

`SEGMENT_KINDS` — `framing`, `tool_schemas`, `system_prompt`, `user_text`,
`assistant_text`, `thinking`, `tool_use`, `tool_result`, `messages_total`
(coarse granularity only), `other` (images, documents, anything unrecognized).

`usage` — normalized by `common.client.usage_breakdown`:

| Field | Type | Meaning |
|---|---|---|
| `input_tokens` | int | **The uncached remainder, not the prompt size** |
| `cache_creation_input_tokens` | int | Tokens written to cache |
| `cache_read_input_tokens` | int | Tokens served from cache |
| `output_tokens` | int | Tokens generated |
| `total_prompt_tokens` | int | The sum of the first three — **this is the prompt size** |

`reconciliation`:

| Field | Type | Meaning |
|---|---|---|
| `counted_total` | int | From `count_tokens` |
| `authoritative_total` | int | `usage.total_prompt_tokens` |
| `residual_tokens` | int | `counted_total - authoritative_total` |
| `residual_fraction` | float | Residual over authoritative total |
| `within_tolerance` | bool | `abs(residual_fraction) <= tolerance_fraction` |
| `tolerance_fraction` | float | The threshold in force |

`tool_calls[]`: `{tool_use_id, name, input_chars, result_chars, is_error, duration_ms}`.

### `run_footer`

| Field | Type | Meaning |
|---|---|---|
| `ended_at` | str | ISO 8601 UTC |
| `turns` | int | Number of `turn` records; the validator cross-checks this |
| `stop_reason` | str | `end_turn` \| `max_iterations` \| `max_tokens` \| `stop_sequence` \| `refusal` \| `error` |
| `final_text` | str | Last non-empty assistant text of the run |
| `totals` | object | See below |
| `counter_calls_total` | int | `count_tokens` calls for the whole run (cache misses) |
| `counter_lookups_total` | int | Counter lookups including cache hits |
| `error` | str \| null | `"TypeName: message"` when `stop_reason == "error"` |

`totals`: `prompt_tokens_total` (summed `counted_total` over turns — context
*re-sent*, not unique tokens), `output_tokens_total`, `cache_read_total`,
`cache_creation_total`, `peak_prompt_tokens` (largest single prompt, the number
to compare against the context window), and `by_kind_total` (`{kind: tokens}`
summed across turns).

### Compatibility promise

Fields may be **added** within `schema_version: 1`. Fields will not be removed,
renamed, or have their meaning changed without a version bump. `SEGMENT_KINDS`
and `RUN_STOP_REASONS` may gain members; consumers should tolerate unknown
members rather than crash. Validate what you read: `loop.schema.validate_run`.

---

## Rendering

`render_text(records)` — plain-text stacked bars, one row per turn, plus a
cumulative share table and the worst reconciliation residual. Good in CI logs.

`render_html(records, title=..., accuracy=...)` — a self-contained page: stat
tiles, a stacked bar chart in inline SVG (one bar per turn, segments stacked in
render order), a legend, a per-turn table repeating every number, and an
optional embedded accuracy report. No external requests of any kind — no CDN, no
fonts, no scripts; the tests assert this.

Both show **growth over turns**, not a final snapshot: turn N's bar is the whole
prompt sent on turn N, so the bars get taller as the conversation accumulates
and you can see which segment is doing the growing.

Colour comes from the repo's dataviz reference palette, one fixed categorical
slot per segment kind, assigned by identity and never cycled or reordered per
run. Light and dark are separate validated steps from the same ramps. Identity
is never carried by colour alone — there is always a legend, and the table view
repeats every value.

---

## Accuracy vs. context length

```python
from loop import analyze_accuracy, report_text, run_task_set

runs = run_task_set(client=client, sink=sink)
report = analyze_accuracy([r.observation() for r in runs], n_bins=5)
print(report_text(report))
```

`analyze_accuracy` bins runs by prompt length (quantile by default, or linear),
computes a success rate per bin with a **Wilson score interval** (well-behaved
at 0/n and n/n, unlike the normal approximation, which matters because eval
samples are usually tiny), and locates the **bend**: the largest single drop in
success rate between adjacent bins with at least `min_bin` observations each.

### What it proves and what it does not

- **It measures association, not causation.** In a real task set, longer prompts
  are usually also harder prompts, so a drop at high token counts may be
  difficulty rather than degradation. `analyze_accuracy` emits a caveat whenever
  the observations span more than one task *family*.
- **To make the claim causal, hold the task fixed and vary padding.** That is
  exactly how `loop.tasks.SYNTHETIC_TASKS` is built: one question, one correct
  answer, one tool, six levels of irrelevant filler in the system prompt. The
  filler never mentions an invoice id or an amount, so it cannot change what the
  right answer is — only how far away it is. A bend there is attributable to
  length because nothing else varies.
- **The bend estimator is coarse.** "Largest adjacent drop" is dominated by
  noise at small n. `AccuracyReport.significant` is `True` only when the two
  bins' Wilson intervals don't overlap; otherwise the report says in words that
  it's a hint, not a finding. Runs under 20 observations are labelled
  anecdote-scale.
- **The bundled task set is plumbing, not evidence.** Six synthetic tasks prove
  the path works end to end. Swap in a real task set before believing any curve.

`observations_from_records(records, outcomes, length=...)` rebuilds observations
from JSONL alone plus a `{run_id: success}` map, so outcome checking can live
anywhere. `length` selects `peak` (default — the largest prompt the model
actually saw), `first`, or `total`.

---

## Tests

```
.venv/bin/python -m pytest loop/ -q
```

105 pass, 1 skipped, with zero network access. The skipped test is the live
smoke test, guarded by `common.client.has_credentials()`.

The fakes in `loop/testing.py` are deliberately behavioural rather than
canned. `heuristic_token_count` is a real function of the prompt's content with
a small **deliberate non-additivity** (a merge discount that grows with block
count), so `test_naive_per_segment_counting_is_not_additive` fails loudly if the
counter ever becomes trivially additive and the additivity tests stop proving
anything. `FakeAnthropicClient` reports usage totals that differ slightly from
the counter's totals, exactly as the live API does — that skew is what the
reconciliation residual exists to surface, and the tests assert it lands inside
the stated tolerance rather than assuming it's zero.
