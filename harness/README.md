# harness — a prompt regression harness

Versioned prompts, a fixed task set, and a runner that tells you whether a
prompt edit helped or hurt.

The deliverable is the diff:

```
$ python -m harness demo

triage.v1@3fd75f01f72a: 5/7 passed
triage.v2@1e4362f973ec: 5/7 passed

REGRESSIONS (1) -- pass -> fail
  - t06_ambiguous_request
      broke: asks_for_specifics
      asks_for_specifics: The summary says the complaint is vague but never
      states what information is needed to work it.

improvements (1) -- fail -> pass
  + t01_plain_bug  (fixed: schema)

pass rate  n=7
  before  5/7 (71%)  95% CI [36%, 92%]
  after   5/7 (71%)  95% CI [36%, 92%]
  change  +0%  (1 improved, 1 regressed)
  paired exact test (McNemar): p=1.000 -- NOT significant at alpha=0.05
  noise floor: with n=7, a one-directional change needs 6+ task flips to clear
  p=0.05. Read the regression list, not the rate.

tokens (task + judge)
  total_prompt_tokens              4,558 ->     4,810  (+252)
  output_tokens                      596 ->       376  (-220)
  est. cost                    $0.0377 -> $0.0335
```

The pass rate did not move. A task still regressed. That gap is the reason this
tool exists.

---

## Quick start

Everything below runs with **no API credentials** — `demo` replays a scripted
response fixture through the real runner, assertions, judge and diff.

```bash
python -m harness list prompts
python -m harness list tasks
python -m harness demo                       # both bundled prompt versions, offline

# live (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
python -m harness run --prompt triage.v1 --out runs/v1.jsonl
python -m harness run --prompt triage.v2 --out runs/v2.jsonl
python -m harness diff runs/v1.jsonl runs/v2.jsonl
```

`diff` exits **1** when there is at least one regression, so it drops into CI
without a wrapper. `--no-fail-on-regression` disables that; `--json` emits the
whole report as machine-readable JSON.

### `run` options worth knowing

| Flag | Why |
|---|---|
| `--concurrency N` | Bounded thread pool, default 4. Each worker makes one blocking call. |
| `--task ID` / `--tag TAG` | Run a subset while iterating; repeatable. |
| `--judge-samples N` | Majority-vote the judge over N samples instead of 1. |
| `--no-judge-cache` | Ask the judge fresh every time — use this to *measure* judge variance. |
| `--no-judge` | Skip tier-2 entirely; judged assertions record as unevaluated. |
| `--fake-responses FILE` | Replay a scripted fixture instead of calling the API. |
| `--no-resume` | Refuse to append to an existing run file. |

Re-running the same command against the same output file **resumes**: completed
tasks are read back and skipped. A crash at task 37 of 50 costs you one task.

---

## File formats

### Prompts — `data/prompts/<id>.prompt.md`

Markdown body, optional TOML frontmatter delimited by `+++`:

```markdown
+++
description = "Support ticket triage, v2: stricter output contract"
effort = "medium"
+++
You are a support triage assistant. Given a customer message, classify it and
return a JSON object with exactly these fields:
...
```

* **Markdown body** because prompts are prose. No escaping, no quoting, and a
  readable `git diff`. A prompt trapped in a JSON string literal is a prompt
  nobody edits.
* **TOML frontmatter** because `tomllib` is stdlib (no dependency) and typed.
  Recognised keys: `id`, `description`, `effort`, `max_tokens`.
* **The id is the filename** (`triage.v2.prompt.md` → `triage.v2`), so id and
  file cannot drift apart. Override with `id =` in frontmatter if you must.

**Versioning has no version field.** The hash is `sha256` over the *entire raw
file bytes*. Edit the body, edit the metadata, add a trailing space — the hash
changes and the run record points at exactly those bytes. There is nothing to
forget to bump. Human-facing versions live in the filename (`triage.v1`,
`triage.v2`) purely as labels; identity is the hash. Every run record stores
both, and `diff` warns if you compare a prompt to itself.

### Tasks — `data/tasks/<id>.toml`

One task per file. The id is the filename stem.

```toml
description = "A plain bug report classifies as a bug and needs no tool call"
input = """
The export button on the reports page does nothing when I click it in Safari.
"""
max_tokens = 400
tags = ["structural", "core"]

[[tools]]                       # optional: tool definitions for this task
name = "lookup_account"
description = "Look up a customer's plan by login email."
input_schema_json = '{"type": "object", "properties": {"email": {"type": "string"}}}'

[[assertions]]
id = "schema"                   # optional; defaults to "a0:json_schema"
type = "json_schema"
schema_json = '''
{"type": "object", "required": ["category"], "additionalProperties": false,
 "properties": {"category": {"type": "string", "enum": ["bug", "billing"]}}}
'''

[[assertions]]
type = "no_tool_called"
```

Keys: `input` (required), `assertions` (required, ≥1), `description`,
`system_suffix` (appended to the prompt's system for this task only), `tools`,
`max_tokens`, `tags`, `id`.

* **TOML** because `tomllib` is stdlib, it is strict about types, and its
  multi-line strings hold prose inputs and inline JSON schemas with no escaping.
  JSON can't do that; YAML does it badly.
* **One file per task** because the design target is 30–50 tasks. A single
  `tasks.yaml` of 40 entries produces unreadable diffs and merge conflicts on
  tasks nobody touched. One file per task means adding a task changes exactly
  one file, and `git blame` on a task means something.

Assertion specs are **validated at load time** — an unknown `type`, a bad regex,
a missing required field, or an unsupported JSON Schema keyword fails the run
immediately rather than silently never firing.

### Adding a task

```bash
cp harness/data/tasks/t01_plain_bug.toml harness/data/tasks/t08_my_case.toml
$EDITOR harness/data/tasks/t08_my_case.toml
python -m harness list tasks          # confirms it parses and shows the assertion types
python -m harness run --prompt triage.v2 --task t08_my_case --out /tmp/probe.jsonl
```

That is the whole procedure. Nothing registers the task anywhere; the directory
*is* the registry.

Note that adding a task changes the task-set hash, so a diff between a run made
before and a run made after will warn that the sets differ. Re-run both prompts
after changing the task set.

---

## Assertions

### Tier 1 — structural (the load-bearing tier)

Deterministic, exact, offline, microseconds. Every type below is covered by a
passing *and* a failing test.

| `type` | Fields | Checks |
|---|---|---|
| `contains` / `not_contains` | `text`, `case_sensitive` (default false) | Substring presence/absence |
| `regex` | `pattern`, `flags` (`imsx`), `negate` | Regex search over the output text |
| `json_valid` | `allow_code_fence` (default true) | Output parses as JSON |
| `json_schema` | `schema` (table) or `schema_json` (string) | Parses *and* validates |
| `length` | `min_chars`, `max_chars`, `min_words`, `max_words` | Output length bounds |
| `tool_called` | `name` (optional = any), `count`, `min_count` | "tool X was called" |
| `no_tool_called` | `name` (optional) | "no tool was called" |
| `stop_reason` | `equals` | `stop_reason == "end_turn"` / `"tool_use"` / … |

`json_valid` and `json_schema` unwrap a *surrounding* code fence by default,
because models fence JSON constantly. A fence with a preamble in front of it
does not parse — which is a real failure, and is exactly the v1→v2 improvement
the demo shows.

JSON Schema is a deliberately small **subset** implemented in `jsonschema.py`
(types, `required`, `properties`, `additionalProperties`, `items`, `enum`,
`const`, numeric/string bounds, `pattern`, `anyOf`/`allOf`/`oneOf`/`not`). Any
keyword outside that raises at load time. A validator that silently ignores a
keyword is worse than no validator, and the alternative was a dependency.

### Tier 2 — LLM judge

```toml
[[assertions]]
id = "asks_for_specifics"
type = "judge"
criterion = """
The summary states what specific information is missing before this ticket can
be worked -- for example which product area, which error, or when it last
worked.
"""
```

* **The judge is a prompt in the same directory, hashed the same way**
  (`data/prompts/judge.rubric.v1.prompt.md`). Its hash is written into the run
  header and into every judged assertion's metadata. This is not bookkeeping
  hygiene: an unversioned judge silently invalidates every historical
  comparison you have, because the same output scored by a quietly-edited judge
  produces a different verdict and you would read that as a prompt regression.
  `diff` warns when two runs used different judge hashes.
* **The verdict is structured**, via `output_config.format` with a JSON schema
  (`{verdict: "pass"|"fail", confidence: number, reasoning: string}`) — not
  prose to be regexed. Unparseable output fails the assertion loudly rather than
  being coerced into a pass.
* Judge assertions are scored **per criterion**, one call each. One criterion
  per assertion keeps a failure attributable.

---

## Two judgment calls

### 1. The noise floor, and why the pass rate is reported with an interval

A pass rate over a 30-task set is a noisy number, and reporting it alone invites
a specific mistake: shipping a prompt because 24/30 became 26/30.

What this harness does about it:

* **Every pass rate is printed with a 95% Wilson interval.** At n=30 that
  interval is roughly ±17 points. Seeing `73% [55%, 86%]` next to
  `80% [63%, 91%]` makes the overlap obvious without any statistical training.
* **The primary comparison is paired, not aggregate.** The two runs cover the
  same task set, so the informative quantity is *which tasks flipped*, not the
  totals. The diff runs a two-sided exact McNemar test over the discordant pairs
  and reports the p-value. Concordant tasks carry no information about direction
  and are ignored by design.
* **The report states its own resolution.** `min_detectable_flips` is printed
  every time: with any n, fewer than **6** one-directional flips cannot reach
  p=0.05 under an exact binomial. If your prompt edit moved 3 tasks, the honest
  reading is "this task set cannot resolve that change", and the report says so
  in those words.
* **Regressions are printed first and in full**, before any aggregate. A 1-for-1
  trade is a flat pass rate and is *not* a neutral change — it is two different
  prompts that happen to score the same, and the regression list is what tells
  you that.

What this harness deliberately does **not** do: it does not repeat-sample the
task model to estimate per-task flakiness. That would multiply cost by N for a
signal that per-task regression inspection already gives you more cheaply. If
you need it, `run` to different output files and diff them pairwise — two runs
of the *same* prompt hash tell you the flake rate directly, and the diff warns
you that you are doing exactly that.

### 2. Judge determinism: cache by default, with the tradeoff stated

Judge verdicts are cached on
`(judge_hash, prompt_hash, task_id, output_hash, criterion)`. All five
participate: change the judge, the prompt under test, the task, the sampled
output, or the criterion, and you get a fresh verdict. Re-running an unchanged
prompt over an unchanged task set costs zero judge tokens, and a cache hit
contributes zero tokens to the run totals so cost accounting stays honest.

**The cost:** the cache freezes one sample of a stochastic process. A judge that
would say "pass" 60% of the time is recorded as whatever it happened to say
first, and that verdict is then reused indefinitely. The cache makes judged
results *stable*, not *correct*.

**Why that is the right default here.** The harness's job is attributing a
difference between two runs to a prompt edit. An uncached judge adds a second
source of variance to every comparison, and a judged regression then can't be
distinguished from a judge coin-flip. A cached verdict at least holds the judge
constant across the two runs being compared — and because the key includes
`output_hash`, an unchanged output genuinely is the same question being asked
twice.

**The escape hatches, both first-class:**

* `--judge-samples 3` takes a best-of-N majority vote and stores the whole
  ballot in the record, so `["pass","fail","pass"]` and `["pass","pass","pass"]`
  are visibly different evidence for the same verdict. Cost scales with N; the
  ballot is cached as a unit.
* `--no-judge-cache` disables caching entirely. Use it when you want to measure
  judge variance rather than prompt variance — run the same prompt twice with
  the cache off and diff the results.

Judge **errors** are never cached, so a rate-limited or malformed judge call is
retried on the next run rather than being frozen into a permanent failure.

---

## Run records

JSONL, one file per run. Line 1 is a `run_meta` header; every line after is one
`result`.

```jsonc
{"type":"run_meta","run_id":"a1b2…","created_at":"2026-07-25T22:49:03+00:00",
 "prompt_id":"triage.v2","prompt_hash":"1e4362…","task_set_hash":"9f2c…",
 "task_count":7,"model":"claude-opus-5","concurrency":4,
 "judge_prompt_id":"judge.rubric.v1","judge_hash":"35b048…","judge_samples":1}

{"type":"result","task_id":"t06_ambiguous_request","passed":false,"status":"ok",
 "assertions":[{"id":"asks_for_specifics","type":"judge","passed":false,
                "detail":"…","meta":{"judge_hash":"35b048…","confidence":0.79,
                                     "cached":false,"usage":{…}}}],
 "output_text":"…","output_hash":"…","tool_calls":[],"stop_reason":"end_turn",
 "usage":{"input_tokens":532,"output_tokens":38,"total_prompt_tokens":532,…},
 "judge_usage":{…},"duration_ms":0,"error":null,"finished_at":"…"}
```

* Per-task usage comes from `common.client.usage_breakdown`, so
  `total_prompt_tokens` includes cache-creation and cache-read tokens rather
  than understating a cache-warm request. Judge tokens are recorded separately
  in `judge_usage` — you can see what tier 2 costs you.
* `status` is `ok`, `error` (the API call failed — recorded, never raised), or
  `incomplete` (a judged assertion was not evaluated).
* Results are appended and flushed per task. A torn final line from a crash is
  tolerated on read; everything before it is intact.
* Appending results from a *different prompt hash* to an existing run file is
  refused outright, because one run file containing two prompt versions is worse
  than no run file.

---

## Tests

```bash
.venv/bin/python -m pytest harness/ -q
```

154 passed, 1 skipped, offline, in ~0.2s. `conftest.py` severs `socket.connect`
for every test, so a change that accidentally reaches for the live client fails
loudly instead of hanging. The one skipped test is the live smoke test, gated on
`common.client.has_credentials()`.

The fakes are scripted `ModelClient`s — the same one-method protocol the live
`AnthropicClient` implements. Only the network call is replaced: request
construction, every assertion, the judge prompt and its JSON parsing, the judge
cache, the run record, resumption and the diff all execute for real. Token
accounting in the fakes routes through the real `usage_breakdown`, so a change
in how usage is summarized breaks a test instead of being mocked away.

---

## What this measures — and what it doesn't

**It measures:** whether a specific prompt edit, against a fixed task set, broke
things that used to work. That is a narrow question and this answers it well:
the regression list is exact, attributable to a specific assertion, and
reproducible from the recorded hashes.

**It does not measure quality.** A 100% pass rate means your assertions pass. If
the task set doesn't contain the failure mode you're about to ship, the harness
is silent about it. The task set is the actual product here; the code is
plumbing. Budget accordingly.

**Its ceiling is the task set's coverage.** Seven fixture tasks demonstrate the
mechanism; they do not evaluate a triage prompt. Real use starts at 30–50 tasks
drawn from production failures, and every new production failure should become a
task before it becomes a prompt edit.

**Single-sample per task.** One sample of a stochastic model per (prompt, task).
An individual task flip may be sampling noise rather than a prompt effect —
which is precisely why the diff reports the paired test and the minimum
resolvable flip count instead of just a delta. Treat a single unexplained
regression as a hypothesis; re-run it before you act on it.

**It cannot tell you *why*.** The diff tells you `t06` regressed and which
assertion broke. Reading the prompt diff and the two outputs is still your job.

**Judged assertions inherit the judge's blind spots**, and the judge is a
different model call with its own failure modes. Structural assertions are the
tier to reach for whenever a criterion can be expressed structurally at all;
tier 2 is for what genuinely cannot.

**Cost is real.** A 50-task set at Opus 5 rates with adaptive thinking is a
couple of dollars per run, and every judged assertion adds a second call. The
diff prints an estimated cost delta so a "better" prompt that doubled token
spend is visible rather than a surprise on the invoice.

---

## Deliberate omissions

* **No task-level retries or flake quarantine.** Both need a flake model this
  harness doesn't have; adding retries without one just hides variance.
* **No cross-model comparison.** `diff` warns if the models differ but makes no
  attempt to normalize; comparing prompts across models is a different question
  with different controls.
* **No prompt-caching optimization.** Runs are one-shot per task, so there is no
  prefix to warm. A larger harness would sort tasks to share a cached system
  prefix; at 50 tasks it isn't worth the complexity.
* **No web UI, no run database.** JSONL files and `git`. The run records are
  small, greppable, and diffable.
* **`max_tokens` is per task, not adaptive.** A task whose output hits the cap
  fails its `stop_reason` assertion, which is the correct signal.

## API notes / uncertainty

Written against the current API surface (`claude-opus-5`, adaptive thinking,
`output_config.effort`, `output_config.format`). Two calls worth flagging:

* **Adaptive thinking is always on for task calls.** Explicitly disabling
  thinking on Opus 5 can cause a tool call to be emitted as *plain text* — the
  turn succeeds, the call never happens, and no error is raised. A harness whose
  job includes asserting "tool X was called" cannot tolerate that failure mode,
  so `thinking: {"type": "adaptive"}` is not configurable per prompt. `effort`
  is (via prompt frontmatter), and it participates in the prompt hash, so an
  effort change is correctly treated as a new prompt version.
* **Task calls use `messages.stream()` + `get_final_message()`** rather than a
  plain create, so a task with a large `max_tokens` cannot hit an HTTP timeout.

Neither path has been exercised against the live API in this environment — there
are no credentials here, and every live call is gated behind
`common.client.has_credentials()`. The live smoke test in `test_cli.py` runs
automatically as soon as credentials exist.
