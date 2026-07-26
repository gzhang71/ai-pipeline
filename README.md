# ai-pipeline

Applied context and prompt engineering against the Claude API.

Five subprojects. Two are **libraries** you'd use while building something else;
three are **benches** that measure a claim and report a number. All five share one
client layer and run offline against fakes, so the test suite doubles as the spec.

```
503 passed, 5 skipped     # the 5 skips are live-API smoke tests, gated on credentials
```

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

.venv/bin/python -m pytest        # whole repo, offline
bash examples/run_all.sh          # five runnable demos, offline
```

Start with [examples/README.md](examples/README.md). Each script prints what its subproject does
and why, without needing an API key.

---

## Architecture

Everything sits on one shared module, and each subproject is otherwise
independent — no subproject imports another.

```
                       common/client.py
        MODEL · get_client() · count_tokens() · usage_breakdown()
                              │
      ┌──────────┬────────────┼────────────┬──────────┐
   prompt/    harness/      loop/      context/    graph/
  (library)   (bench)     (library)     (bench)    (bench)
```

### `common/` — the shared floor

`client.py` holds the pieces every subproject would otherwise re-derive:

| Symbol | Purpose |
|---|---|
| `MODEL` | `claude-opus-5`. One constant, one place to change it. |
| `get_client()` / `get_async_client()` | Cached clients. Credentials resolve from env or an `ant` profile, so no key is ever passed explicitly. |
| `has_credentials()` | Guards every live path. Tests skip on it. |
| `count_tokens()` | Exact counts via the API endpoint. Never `tiktoken` — that's OpenAI's tokenizer and undercounts Claude badly on code. |
| `usage_breakdown()` | Normalizes a `usage` object and computes `total_prompt_tokens`. |
| `MIN_CACHEABLE_PREFIX_TOKENS`, `RENDER_ORDER`, `MAX_CACHE_BREAKPOINTS`, `CACHE_LOOKBACK_BLOCKS` | API constants that are easy to get wrong. |

The `usage_breakdown` helper exists because of one trap: **`usage.input_tokens` is
the uncached remainder only**. Total prompt is `input + cache_creation + cache_read`.
Read the single field on a cache-warm request and you'll understate it by 10×.

---

### `prompt/` — cache-aware prompt assembly *(library)*

**Problem.** Prompt caching is a prefix match over rendered bytes in the order
`tools → system → messages`. One changed byte at position N invalidates every
breakpoint at or after N — and nothing reports it. The request succeeds and
`cache_read_input_tokens` is silently 0.

**Approach.** Make it correct by construction.

| Module | Achieves |
|---|---|
| `blocks.py` | The block model. `Stability` (`STATIC`/`SESSION`/`TURN`) as an `IntEnum` so order is comparable, applied to `ToolBlock` / `SystemBlock` / `MessageBlock`. |
| `assembler.py` | `PromptAssembler.validate()` rejects any ordering where a less-stable block precedes a more-stable one. `plan()` places breakpoints at stability boundaries, drops candidates under the model's minimum, and ranks survivors by tokens protected when more than 4 qualify. `to_request_kwargs()` emits real `messages.create` kwargs. |
| `serialize.py` | Deterministic serialization (`canonical_json`, `stable_tool_order`) and a byte-level model of the rendered prefix. |
| `linter.py` | `find_silent_invalidator(builder)` calls a prompt builder twice and reports the first divergent byte offset plus the owning block. |
| `diagnostics.py` | `diagnose_usage()` classifies a response as READ / WRITE / PARTIAL / MISS with ranked causes. |

**The insight worth keeping:** a prefix that changes every request never shows up
as a MISS. It reports as a perpetual WRITE.

---

### `harness/` — prompt regression testing *(bench)*

**Problem.** "Is v2 better than v1?" is unanswerable without a fixed task set, and
a pass rate over ~30 tasks is mostly noise.

| Module | Achieves |
|---|---|
| `prompts.py` | Prompts are Markdown + TOML frontmatter files. Identity is a sha256 over raw bytes — no version field to forget to bump. |
| `tasks.py` | One TOML file per task; assertion specs validated at load, so a typo'd assertion type fails the run instead of never firing. |
| `assertions.py` | Nine structural types: JSON validity/schema, regex, contains/not_contains, length bounds, tool-called/no-tool-called, `stop_reason`. |
| `jsonschema.py` | A small JSON Schema subset validator that *raises* on unsupported keywords rather than ignoring them. |
| `judge.py` | Tier 2. The judge is itself a versioned prompt whose hash lands on every verdict; verdicts return via `output_config.format`, not prose. Caching keyed on `(judge_hash, prompt_hash, task_id, output_hash, criterion)`. |
| `runner.py` | Resumable JSONL run records, bounded concurrency, judge tokens tracked separately from task tokens. |
| `diff.py` | Regressions first and in full, then improvements, then aggregates. |
| `stats.py` | `wilson_interval`, `mcnemar_exact`, and `min_detectable_flips` — the noise floor. |

**The design decision that matters:** the diff leads with named regressions, not
the aggregate, because `min_detectable_flips` tells you a small task set often
*cannot* resolve the delta it just printed.

---

### `loop/` — context budget profiler *(library)*

**Problem.** "Where did my context window go?" Token counts aren't additive, so
counting segments separately won't sum to the whole.

| Module | Achieves |
|---|---|
| `agent.py` | `run_loop()` — an instrumented tool-use loop. Caller supplies tools and executor. Handles parallel tool calls, `pause_turn`, tool exceptions → `is_error` results, and a `max_iterations` guard. |
| `attribution.py` | The engine. Method `incremental_prefix_delta`: never count a segment alone; count a growing chain of prefixes of the real request and attribute each segment its delta. |
| `schema.py` | Versioned public JSONL schema (`schema_id`, `schema_version`) with `validate_record` / `validate_run`. |
| `sink.py` | `JsonlSink` validates on write, so malformed records fail at the producer. |
| `render.py` | `render_text` (stacked ASCII per turn) and `render_html` (self-contained page, inline SVG/CSS, no external requests). |
| `accuracy.py` | Accuracy vs. prompt length, with Wilson intervals and bend detection gated on non-overlapping intervals. |

**Why the deltas approach:** they telescope, so `segment_sum == counted_total`
always and the schema *enforces* that identity. That's arithmetic, not a claim
each number is a segment's true cost — the real error is order-dependence, which
is why `measurement_order` is on every run header. Negative deltas (boundary
token merging) are reported, never clamped. `framing` is flagged `approximate`
because no prefix can contain framing alone.

---

### `context/` — compaction strategy bench *(bench)*

**Problem.** Long conversations exceed the window. Which history-management
strategy loses the least, for what cost?

| Module | Achieves |
|---|---|
| `strategies.py` | One interface — `apply(messages, budget) -> StrategyResult` — and six implementations: `TailTruncation`, `RecursiveSummarization`, `AnchoredSummary`, `NoteTaking`, `ToolResultEviction`, `ServerCompaction`. |
| `validation.py` | `validate()` enforces what the API actually rejects (first-user, no trailing prefill, tool_use/tool_result pairing, unique ids). `sanitize()` repairs the ways slicing breaks a history. |
| `summarizers.py` | The model calls strategies make *on their own behalf*, plus offline fakes. |
| `usage.py` / `tokens.py` | Accounting. `StrategyResult` carries `usage` so a strategy is charged for its own summarizer calls. |
| `bench.py` | The runner and comparison report. Asserts request validity before every send. |

**Two accounting decisions** that decide the outcome: summarizer tokens are
charged to the strategy that spent them (otherwise summarizing strategies look
~2× cheaper than they are), and the trigger fires on **tokens OR turns** — turns
being a backstop for many-tiny-turns growth that never trips a token threshold.

`ServerCompaction` returns `request_overrides` rather than editing messages:
its entire contribution is the beta header plus `context_management`. Note that
compaction (`compact_20260112`) and context editing (`clear_tool_uses_20250919`)
are different features with different beta headers, and the code keeps them apart.

---

### `graph/` — just-in-time retrieval *(bench)*

**Problem.** Answering questions about a repo by stuffing retrieved chunks into
context is expensive. Can you load *identifiers* and fetch bodies on demand?

| Module | Achieves |
|---|---|
| `builder.py` | Walks a repo with stdlib `ast`. Nodes: modules, classes, functions, methods with signature and docline. Edges: `contains`, `imports`, `calls`. |
| `index.py` | Cheap JSON index with incremental refresh, so the graph reloads without re-walking. |
| `tools.py` | The four tools in Anthropic tool-use format: `search_symbols` (signatures, no bodies), `get_definition` (exactly one node), `get_neighbors` (callers/callees as identifiers), `read_lines` (bounded). Descriptions are prescriptive about *when* to call. |
| `agent.py` | Builds the outline and system prompt, runs the JIT loop. The outline auto-tiers to a module-level index above 250 symbols. |
| `baseline.py` | The opponent: 60-line chunks, idf-weighted lexical scoring, one model call. `sweep_k` / `recall_at_k` measure `k` rather than guessing it. |
| `compare.py` | Head-to-head over the same questions, counting tool schemas and every tool result that entered context. |

**Honesty constraint baked into the design:** the initial prompt provably contains
no file bodies (there's a test), and `ast` resolves only about a third of call
sites in this repo — a fixture of deliberate blind spots asserts each miss.

---

## Layout

```
common/     shared client, model constants, token counting
prompt/     cache-aware prompt assembler        (library)
harness/    prompt regression harness           (bench)
loop/       context budget profiler             (library)
context/    compaction strategy bench           (bench)
graph/      JIT retrieval over a code graph     (bench)
examples/   five runnable offline demos
skill/      reserved, empty
```

Three subprojects have a CLI:

```sh
.venv/bin/python -m harness demo     # two prompt versions, diffed
.venv/bin/python -m context          # compaction bench
.venv/bin/python -m graph --fake     # JIT vs. chunk-and-stuff
```

## Status: nothing here has run against the real API

Every suite and every example passes against fakes. That establishes internal
consistency and catches logic errors. It does not establish that the numbers are
true. Unvalidated until an `ANTHROPIC_API_KEY` or `ant auth login` profile exists:

- `loop/` attribution reconciles against real `usage` — precisely what a fake can't check
- `prompt/` cache behavior, including its assumption that the minimum-prefix rule
  applies per-breakpoint rather than per-prompt (the conservative reading)
- `context/` server-compaction round-trip is simulated
- every benchmark ranking

Each subproject README has its own limitations section. They're worth reading —
they're more interested in what the benches fail to model than in the headlines.
