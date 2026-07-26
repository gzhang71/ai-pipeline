# context — a compaction strategy bench

Six history-management strategies behind one interface, run against the same
long-horizon task set, measured on **task success against tokens spent**.

The question this package exists to answer is not "which strategy makes the
context smallest" — tail truncation wins that trivially — but "which strategy
still has the answer when you need it, and what did that cost".

```bash
.venv/bin/python -m context              # run the bench, print the report
.venv/bin/python -m pytest context/ -q   # 122 tests (2 skip without credentials)
```

---

## The strategy interface

```python
result = strategy.apply(messages, budget)
request_messages = result.messages
```

`apply` returns a `StrategyResult`, not a bare list. Two things have to come
back alongside the edited history:

| field | why it exists |
|---|---|
| `messages` | the edited history — the whole interface for a caller that needs nothing else |
| `usage` | tokens the strategy spent on **its own** model calls (summarizer, note writer). Return bare messages and this cost disappears |
| `request_overrides` | extra kwargs for the API call. `ServerCompaction` edits nothing client-side; its entire contribution is `betas` + `context_management` |
| `fired` / `note` | whether it compacted this turn, and what it did — for the report |

`Budget` is the context passed in, and it carries enough to make a real
decision rather than a blind one: `max_tokens`, `max_turns`, `turn_index`,
`keep_recent_messages`, `keep_recent_tool_results`, the original `objective`,
a `workspace` path, and the `TokenCounter` itself (so a strategy can measure
rather than guess).

Every strategy runs its output through `sanitize()` before returning, and the
bench runs `assert_valid()` on every request before sending it.

---

## The strategies

**`TailTruncation`** — drop the oldest messages until the history fits. The
baseline. Costs nothing: no model calls, no latency, no state. It deletes
early information outright, with no summary and no trace, which is why it
scores 0% early recall here. Every other strategy has to justify its cost
against this one.

**`RecursiveSummarization`** — keep a recent tail, fold everything older into
a rolling summary via a model call. Each compaction re-reads its own previous
output, so information passes through the summarizer repeatedly and is
re-truncated to the summary budget every time. Costs one model call per
compaction plus the summary in every subsequent prompt.

**`AnchoredSummary`** — the same, with the original objective captured once
and pinned verbatim at the head, never paraphrased. Costs the same as
recursive summarization plus a few tokens per request forever. The objective
is the one thing a summarizer must never rewrite, because an agent that has
drifted on *what it was asked to do* cannot recover from a transcript that no
longer exists.

**`NoteTaking`** — durable state on disk; context is rehydrated from the file
rather than from transcript history. Two sources feed it: `write_note` tool
calls the agent already made (harvested for free — the text is in the window
already, so what harvesting buys is a list of keys the writer must not evict)
and a scribe pass over the window about to be discarded. Costs one model call
per compaction plus the notes blob in every subsequent prompt.

**`ToolResultEviction`** — keep every `tool_use` block, replace stale
`tool_result` payloads with a placeholder. The client-side sibling of the
API's `clear_tool_uses_20250919` context edit. Pairing is preserved exactly:
the `tool_result` block stays with the same `tool_use_id`, only the payload
changes, so the request stays legal. Zero model calls. Two limitations, both
visible in the report: it is blind to anything that only ever existed inside a
tool result, and on a text-heavy history it often cannot get under budget at
all (12 over-budget turns in the run below).

**`ServerCompaction`** — the API's own `compact_20260112`. The "do nothing
clever" comparison point. Nothing is edited client-side; the strategy attaches
the `compact-2026-01-12` beta and the context-management edit and lets the
server summarize as it approaches the trigger threshold. No client-side model
calls — the cost is real but billed inside the same request.

> **Compaction is not context editing.** They are two distinct server-side
> features and this package keeps them apart deliberately.
> Compaction *summarizes*: `{"type": "compact_20260112"}`, beta
> `compact-2026-01-12`. Context editing *clears* old tool results or thinking
> blocks without summarizing: `{"type": "clear_tool_uses_20250919"}` /
> `{"type": "clear_thinking_20251015"}`, beta `context-management-2025-06-27`.
> `ServerCompaction(include_tool_clearing=True)` enables both, with both
> headers.

> **The compaction footgun.** The response carries a compaction block, and the
> **entire `response.content`** must be appended back to `messages`. Append
> only the extracted text and the block is lost, the server has no record of
> the compaction, and state silently resets. The runner always appends full
> content; `run_task(..., append_full_content=False)` does it wrong on
> purpose, and `test_dropping_the_compaction_block_makes_the_server_redo_the_work`
> shows what that costs.

---

## Two judgment calls, stated explicitly

### 1. Summarizer tokens are charged to the strategy that spent them

A summarizing strategy makes extra inference calls. If those tokens are not
counted, the comparison is rigged in its favour — you are comparing one
strategy's full cost against another's partial cost. So `apply()` returns
`usage`, the runner folds it into the strategy's total, and the report breaks
model calls into `calls` (the agent loop) and `strat` (the strategy's own).
The effect is not small: summarizing strategies here spend roughly twice the
tokens of tail truncation, and about half of that gap is the summarizer.

The same discipline applies inside a single call. **Total prompt tokens are
`input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.**
`input_tokens` alone is the *uncached remainder* — on a warm cache it is under
a tenth of the truth. The fake client deliberately reports a ~90%-cached
split so that any code reading one field is caught by the tests.

`ServerCompaction` is the honest exception: its compaction cost is real but
arrives as output tokens inside the same request, so it shows as zero strategy
calls and a larger agent bill. That is what the API actually does; the report
does not pretend otherwise.

### 2. Strategies fire on a token threshold **or** a turn count

Whichever trips first (`Budget.is_over`).

Tokens are the real constraint — the context window and the bill are both
denominated in tokens, and a turn-count trigger fires uselessly on ten one-line
turns while sitting still through two turns that each pasted a 200KB file.

But a pure token trigger has a blind spot: many small turns. Block count grows
even when the character count does not, per-message overhead accumulates, and
the 20-block cache lookback window gets blown through — so a long conversation
of tiny turns silently stops hitting the cache while never tripping a token
threshold. The turn count is a cheap backstop for exactly that case.

Neither alone is sufficient, and the cost of the redundant check is one integer
comparison.

---

## Results (offline run, this checkout)

```
strategy                  success   early    tokens  tok/task  calls  strat  over
ToolResultEviction         6/7        83%    54,536     7,791     69      0    12
NoteTaking                 6/7        83%    59,017     8,431     69     24     0
ServerCompaction           5/7        67%    34,010     4,859     69      0     0
RecursiveSummarization     5/7        67%    59,611     8,516     69     24     0
AnchoredSummary            5/7        67%    60,451     8,636     69     24     0
TailTruncation             1/7         0%    29,284     4,183     69      0     0
```

### Which strategies lose early-introduced information

| strategy | early recall | what it lost |
|---|---|---|
| `ToolResultEviction` | 5/6 | `early-tool-result` — the fact only ever existed in a tool payload |
| `NoteTaking` | 5/6 | `state-accretion` — seven durable facts, notes capacity five, and the queried one was never `write_note`d |
| `ServerCompaction` | 4/6 | `fact-flood`, `state-accretion` — more durable state than the summary budget holds |
| `RecursiveSummarization` | 4/6 | same two |
| `AnchoredSummary` | 4/6 | same two |
| `TailTruncation` | 0/6 | **everything** — all six early tasks |

### Reading this honestly

- **The baseline loses every early fact.** That is the one unambiguous result:
  tail truncation is cheapest per token and useless for anything that needs
  memory. If your agent never refers back, it is fine; otherwise it is not a
  serious option.
- **Note-taking's advantage is narrower than the folklore.** The widely
  repeated claim that note-taking beats clever summarization does *not*
  cleanly reproduce here. Once the summarizer and the note writer are given
  the **same capacity** and the **same salience rule**, they behave almost
  identically — because they are almost the same algorithm. Note-taking wins
  exactly one task (`fact-flood`), and it wins it through the one mechanism a
  rolling summary has no equivalent of: the agent explicitly wrote the fact
  down with `write_note`, so it was pinned against eviction. **The honest
  conclusion is that note-taking only beats summarization for facts the agent
  chose to record.** An earlier draft of this bench showed note-taking sweeping
  every task — that was an artifact of giving the notes file a larger capacity
  than the summary budget and of an insertion-order bug in the note writer.
  Both are fixed; the result changed.
- **Anchoring did not separate from plain summarization on this task set.**
  `objective-drift` encodes the objective as a `[FACT]` line so success can be
  checked automatically, and the fake summarizer treats it as salient and keeps
  it — so both pass. Anchoring's real benefit is prose objectives surviving
  paraphrase, which this harness does not model (see limitations). The unit
  test `test_plain_recursive_summarization_can_lose_the_objective` shows the
  divergence directly; the bench does not.
- **The cheapest strategies are cheap because they do less.** `ServerCompaction`
  and `TailTruncation` are the two lowest token totals, but one of them scores
  5/7 and the other 1/7. Cost per task is only meaningful next to the success
  column.
- **`over` matters as much as `success`.** `ToolResultEviction` scores well but
  failed to get under budget on 12 turns. On a real context window that is not
  a lower score, it is a 400.

---

## Limitations — read before believing any of this

1. **Offline runs use fakes.** No credentials exist in this environment, so the
   default bench uses `FakeClient`, `FakeSummarizer`, and `FakeNoteWriter`.
   The fakes are behavioural, not oracles — the client answers from whatever
   context the strategy handed it and returns `UNKNOWN` when the fact is gone,
   and nothing in it knows which strategy is running or what the right answer
   is. But they are still models.
2. **The compressors do not model paraphrase drift.** Both fakes preserve fact
   values exactly and lose things only by capacity. Real summarization also
   degrades through repeated rewriting, which is very likely where the
   real-world "notes beat summaries" advantage comes from. This harness cannot
   see that effect, so it under-states the case for note-taking and for
   anchoring.
3. **The capacity numbers are chosen, and they drive the ranking.** Summary
   budget and notes capacity are both 5 facts. They are equal on purpose — an
   unequal pair would decide the outcome by fiat. Change them and the
   `fact-flood` / `state-accretion` results move.
4. **`ServerCompaction`'s numbers are simulated.** The fake client models the
   feature — summarize the old span, return a `compaction` block, honour that
   block next call. It is not a measurement of the real API. The report says so
   in its own output.
5. **The offline token counter is a heuristic** (~4 chars/token plus per-message
   and per-block overhead). It is monotonic, which is all the strategies need
   for a budget decision, but it is not a tokenizer. Pass
   `counter=ApiTokenCounter()` for exact counts via the `count_tokens`
   endpoint. Never substitute `tiktoken` — it is OpenAI's tokenizer and
   undercounts Claude tokens badly on code.
6. **Success is fact recall, not task quality.** These tasks check whether a
   specific value survived to the end. That is the failure mode compaction
   causes, but it is not the whole of "did the agent do a good job".

To run against the real API (needs credentials):

```python
from context import run_bench, format_report, LiveClient, ApiTokenCounter
print(format_report(run_bench(client_factory=LiveClient, counter=ApiTokenCounter())))
```

---

## The task set

Seven synthetic long-horizon tasks in `tasks.py`. Every one has the same
shape — a fact introduced **early**, a long noisy middle that blows the
budget, the fact needed **late** — because that is precisely what compaction
destroys. Filler turns carry their own `METRIC_nn` noise; without it the tasks
would be far too easy.

| task | what it probes |
|---|---|
| `early-constant` | fact in ordinary user text |
| `early-tool-result` | fact that only ever existed inside a tool result |
| `objective-drift` | the thing needed late is the original objective |
| `corrected-fact` | a value stated early and corrected early; the stale value is still in the transcript and is the wrong answer |
| `state-accretion` | seven durable facts, the first one queried — where bounded stores start to spill |
| `fact-flood` | thirteen durable facts, the first one queried — more than any bounded store holds |
| `late-fact` | **control.** The fact is in the last turn before the question. Any strategy that fails this is broken, not lossy — a test asserts all six pass it |

---

## Message-shape validity

`validation.py` enforces the rules the Messages API actually rejects: first
message `user`, no trailing assistant turn (a prefill, rejected on Claude
Opus 5), `tool_use` only in assistant messages and `tool_result` only in user
messages, every `tool_use` paired with exactly one later `tool_result` and
vice versa, unique tool ids, thinking blocks first within their message, no
empty content or empty text blocks, no `system` message in first position.
Unknown block types — the server's `compaction` block included — pass through
untouched.

`sanitize()` repairs the three ways slicing a history breaks it: orphaned
`tool_result`s, unanswered `tool_use`s, and a history that now starts with an
assistant message. Every strategy calls it; the tests assert every strategy's
output validates, including on a tool-heavy history and under repeated
application.

---

## Layout

| file | contents |
|---|---|
| `strategies.py` | `Strategy`, `Budget`, `StrategyResult`, the six strategies, API constants |
| `validation.py` | validator, `sanitize()`, block/text helpers |
| `summarizers.py` | `Summarizer` / `NoteWriter` protocols, live and fake implementations |
| `tokens.py` | `HeuristicTokenCounter`, `ApiTokenCounter` |
| `usage.py` | `Usage` — the three-field prompt sum, and strategy-call accounting |
| `tasks.py` | the seven-task long-horizon set, tool definitions, system prompt |
| `fakes.py` | `FakeClient` — answers from context, simulates server compaction |
| `bench.py` | `run_task`, `run_bench`, `format_report`, `LiveClient` |

A sibling subproject (`loop/testing.py`) exposes similar offline doubles.
This package keeps its own because the two need different contracts — the fake
here has to simulate server-side compaction state and answer fact-recall
queries, neither of which the loop profiler's fake does — and because a
sibling's test double changing underneath would silently move these numbers.
