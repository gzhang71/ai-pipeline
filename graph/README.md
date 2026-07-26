# `graph/` — just-in-time retrieval over a code graph

Answer questions about a Python repo by loading **identifiers** into context and
fetching **bodies only on demand** — then measure that against a retrieval agent
that stuffs the prompt with chunks.

```bash
.venv/bin/python -m pytest graph/ -q          # 80 passed, 1 skipped (offline)
.venv/bin/python -m graph --sweep-k           # baseline recall@k, no API calls
.venv/bin/python -m graph --fake              # full harness, offline fakes
.venv/bin/python -m graph                     # live (needs credentials)
```

---

## Architecture

```
  repo ──ast──> FileAnalysis ──JSON──> .code_graph_index.json
                     │  (per file: nodes, import refs, call sites)
                     ▼
                 CodeGraph  ── resolution pass: cross-file imports + call edges
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
   GraphTools                  build_outline
   (4 tools, bodies            (identifiers +
    on demand)                  signatures, no bodies)
        └────────────┬─────────────┘
                     ▼
              run_jit_agent  ──vs──  run_baseline (chunk + stuff, 1 call)
                     └──────────┬───────────┘
                          run_comparison
```

| Module | Role |
|---|---|
| `builder.py` | `ast` walk → `Node`/`ImportRef`/`CallRef`; cross-file resolution into `CodeGraph` |
| `index.py` | JSON persistence, `discover_python_files`, incremental `refresh()` |
| `tools.py` | The four tools as Anthropic tool definitions, plus their implementation |
| `agent.py` | The outline builder and the tool-use loop |
| `baseline.py` | Chunking, idf lexical retrieval, `recall_at_k` / `choose_k` |
| `tokens.py` | Live and offline token counters, `TokenLedger` |
| `compare.py` | Head-to-head harness + CLI |
| `questions.py` | Question sets with checkable answers |
| `fake_client.py` | Offline client doubles (tests only) |
| `fixtures/sample_repo/` | Synthetic test corpus |

### Graph schema

**Nodes** — every node carries `id`, `kind`, `name`, `qualname`, `module`,
`path`, `lineno`, `end_lineno`, `signature`, `doc` (docstring first line),
`parent`.

| Kind | Id form | Example |
|---|---|---|
| `module` | dotted import path | `common.client` |
| `class` | `module:QualName` | `graph.builder:CodeGraph` |
| `function` | `module:QualName` | `common.client:has_credentials` |
| `method` | `module:Class.name` | `graph.builder:CodeGraph._resolve_call` |

Nested functions are nodes too, with a dotted qualname: `graph.baseline:build._clean`.
A `method` is a `def` whose immediate enclosing scope is a `class`; everything
else is a `function`.

Module nodes additionally carry `defines`: the **names** of module-level
assignments. Constants are deliberately *not* nodes — they have no body to
fetch — but recording the names is what lets `search_symbols("MIN_CACHEABLE_PREFIX_TOKENS")`
return the module that defines it. The names are indexed; the values are not
disclosed until you call `get_definition` on the module.

**Edges**

| Kind | Direction | Notes |
|---|---|---|
| `contains` | module → class → method; function → nested function | Complete. |
| `imports` | module → module | Only targets that resolve to a file in the walked tree. Everything else is recorded in `external_imports`, never invented as a node. |
| `calls` | node → node | **Incomplete by construction — see below.** |

---

## What the `ast` analysis can and cannot resolve

`ast` sees syntax, not values. The call graph is a **lower bound**: every edge it
reports is real, but many real edges are missing. Do not read it as a complete
call graph.

**Resolved:**

- `helper()` — a bare name, resolved outward through enclosing scopes
- `self.method()` / `cls.method()` — against the *same* class
- `module.func()` — through an `import x` / `import x as y` alias
- `func()` where `func` came from `from module import func`
- `Class()` — a constructor call resolves to the class node
- relative imports (`from . import util`, `from ..pkg import x`)

**Not resolved, and never guessed:**

| Blind spot | Example |
|---|---|
| Dynamic imports | `importlib.import_module(name)`, `__import__(s)` |
| `getattr` dispatch | `getattr(obj, name)()` |
| Attribute calls on untyped values | `thing.slug()` where `thing` is a parameter, a list element, or a return value |
| **Inheritance** | `self.method()` resolves only against the *same* class, never a base class |
| Functions bound to variables | `f = slugify; f()` |
| Decorators that replace a function, metaclasses, monkey-patching, `functools` wrappers | anything that rebinds at import time |
| Conditional definitions | `if X: def f(): ...` — the node exists, but which one runs does not |
| `from module import *` | skipped entirely |

The size of the blind spot is always visible:
`CodeGraph.stats()["unresolved_calls"]`. On this repository it is **3,066
unresolved call sites against 1,568 resolved edges** — roughly two in three
calls are not statically resolvable. `get_neighbors` repeats the caveat in every
response so the model does not over-trust an empty caller list.

`graph/fixtures/sample_repo/pkg/dynamic.py` exists purely to pin these blind
spots, and `test_builder.py` asserts that each one really is missed. A test that
the tool *fails* is worth more here than a claim that it succeeds.

---

## The tool surface

All four are declared in Anthropic tool-use format in `tools.py`; `GraphTools`
executes them.

| Tool | Returns | Bodies? |
|---|---|---|
| `search_symbols(query, kind?, limit?)` | matching identifiers + signatures + docstring heads | **no** |
| `get_definition(symbol_id)` | the source of exactly **one** node | yes, one node |
| `get_neighbors(symbol_id, direction?)` | callers / callees / imports / imported_by / children / parent, as identifiers | **no** |
| `read_lines(path, start, end)` | bounded raw read, capped at 400 lines | yes, bounded |

Descriptions are written **prescriptively** — they state *when* to call the tool,
not just what it does ("Call this FIRST for any question about the repository";
"do NOT guess what a function does from its name"; "This is the fallback, not the
default"). Recent models are conservative about reaching for tools, and trigger
conditions in the description measurably change call rate. `test_tools.py`
asserts every description contains a call-condition, so a future edit that
rewrites one into pure behaviour-description fails the suite.

### The initial prompt

`build_outline` has two levels and picks between them automatically:

| Level | Contents | Cost on this repo |
|---|---|---|
| `symbols` | every symbol, signature, docstring head | ~46,000 tokens |
| `modules` | module ids, paths, symbol counts, docstring heads | ~3,000 tokens |

`auto` uses `symbols` below 250 non-module symbols and `modules` above. The first
version of this shipped `symbols` unconditionally and the outline alone was 46k
tokens — a quarter of the whole corpus, re-read on every turn. That defeated the
entire point, and the token measurement is what caught it.

Either way the prompt contains **no file contents**. `test_prompt.py` proves it
two ways: no sentinel planted inside any fixture function body appears, and —
stronger — *no substantive line from inside any function or method* appears
anywhere in the rendered request, including the tool schemas. Module-level
literals (`RETRY_LIMIT = 7`) are checked too: names are indexed, values are not.

---

## The baseline

A fair opponent, not a strawman:

- **Same corpus.** It chunks exactly the files the graph indexed.
- **Overlapping windows.** 60-line chunks with 15 lines of overlap, so a
  definition straddling a boundary is still retrievable.
- **Same tokenizer** as `search_symbols`, with idf weighting on top.
- **`k` is measured, not guessed.** `recall_at_k` checks whether *every* gold
  symbol a question needs lands in the top-k. `choose_k` picks the smallest k
  reaching the retriever's best recall, and the CLI defaults to `--k auto`.
  On this repo's question set:

  ```
  k=4   0.33   k=16  0.50   k=48  0.83
  k=8   0.33   k=24  0.50   k=64  1.00
  k=12  0.33   k=32  0.50   k=96  1.00
  ```

  So the harness runs the baseline at **k=64** — ~3,800 lines of code in the
  prompt. Running it at the library default of 12 would have made the
  comparison look far better for JIT and meant nothing.

**It is lexical, not embedding-based, because no embedding model is available
offline** and calling a hosted one would make the token accounting
incomparable. Vector RAG would retrieve better on paraphrased questions. Treat
the head-to-head as **directional, not conclusive against a production retrieval
system**.

---

## Results

Offline (`python -m graph --fake`), 85 files / ~23,500 lines / ~190,000 tokens of
Python, `k=auto → 64`:

| | correct | total prompt tokens | peak prompt tokens | model calls |
|---|---|---|---|---|
| JIT graph | 4/6 | 236,872 | **15,789** | 30 |
| Baseline | 6/6 | 234,779 | 40,044 | 6 |
| ratio | | 1.01× | **0.39×** | 5.0× |

**Read these numbers carefully.**

- **Peak context is the real win: 0.39×.** The JIT arm never needed more than
  15.8k tokens in the window at once, against 40k for the baseline. That is the
  claim that survives scrutiny.
- **Total prompt tokens are a wash (1.01×).** A five-turn agent re-reads its
  whole transcript five times, and the ledger counts every one. Anyone quoting
  "a fraction of the context" from the initial prompt alone is inflating it.
- **Correctness here measures the harness, not the models.** The offline fakes
  are asymmetric on purpose: the baseline's fake is a *perfect reader* that
  echoes every retrieved chunk (so any failure is a retrieval failure), while
  the JIT fake runs a fixed three-step policy far dumber than a real model. The
  4/6 is the fake policy's ceiling, not the approach's. Run without `--fake` for
  a real answer-quality comparison.

Token counts are exact (`count_tokens`) when credentials exist and a documented
chars-per-token estimate otherwise; both arms always share one counter, and the
report labels which was used. The harness counts the payload it is about to
send rather than reading `usage.input_tokens` off responses — `input_tokens` is
the *uncached remainder* only, so with caching on it badly understates a
cache-warm request (total is `input + cache_creation + cache_read`).

---

## Where JIT retrieval loses

Stated plainly, because all three are real:

1. **Round trips, and latency is a real cost.** 5× the model calls here. Those
   are *sequential* — each depends on the previous tool result, so they cannot
   be parallelised. At ~2–5s per call that is a 10–25 second answer against one
   round trip for the baseline. For an interactive product that difference
   matters more than the token saving. Prompt caching helps the token side of
   repeated turns; it does nothing for wall-clock.

2. **Small corpora: it is strictly worse.** The four tool schemas plus the index
   are a fixed cost of roughly 3k tokens. Below that, stuffing the entire
   codebase is cheaper *and* one round trip.
   `test_on_a_tiny_corpus_stuffing_wins` pins this on the 120-line fixture,
   where the JIT arm costs more for the same answers. There is a crossover, and
   it is not near zero.

3. **Pinpoint keyword lookups are a fair fight the baseline often wins.** When a
   term is globally unique, idf ranks it first and k=4 suffices;
   `test_pinpoint_lookup_is_a_fair_fight_for_the_baseline` asserts the baseline
   is cheaper on exactly that shape. The graph wins on **relational, many-site**
   questions — "which functions call X" needs every call site at once, which
   chunk retrieval can only do by holding all of them in the window while
   `get_neighbors` returns an edge list. `test_relational_query_is_where_the_graph_wins`
   measures that case at >2× less peak context.

Two further honest caveats:

- **Retrieval quality is the ceiling.** `search_symbols` is lexical. Scoring now
  adds an exact-identifier bonus and a mild penalty for symbols in test modules,
  which fixed two concrete failures: `"MODEL"` used to return `ModelSummarizer`
  and three test functions while the module that actually defines the constant
  never appeared, and tests outranked the second real implementation of
  `wilson_interval`. Both now rank correctly. It is still token-overlap scoring
  underneath — no idf, no semantic similarity — so an unusual phrasing can still
  miss. A real model recovers by re-querying or walking edges; a fixed policy
  cannot. Tests are demoted, never excluded: sometimes the test *is* the answer.
- **The index is only as good as the last refresh.** `refresh()` re-analyzes on
  a size/mtime change and re-resolves *all* cross-file edges (a change in one
  file can create an edge in another), but a stale index silently answers about
  code that no longer exists. `load_or_build` refreshes by default.

---

## Testing

`.venv/bin/python -m pytest graph/ -q` → **80 passed, 1 skipped**, no network.
The skip is the live-API smoke test, guarded by `common.client.has_credentials()`.

Tests run against `graph/fixtures/sample_repo/` (5 files: classes, inheritance,
nested functions, module/from/relative imports, a package, and a module of
deliberate blind spots) and a generated 61-module corpus for the scaling claims.
Nothing is asserted against the live repo, which keeps results deterministic.

The fakes in `fake_client.py` are local to `graph/` rather than borrowed from a
sibling package: this subproject reports benchmark numbers, and a shared double
changing underneath could move them without a single test here failing. They are
also not mocks in the assert-on-calls sense — both drive the real retrieval code
and build answers from whatever it actually retrieved, so a passing test is
testing the retrieval, not the double.
