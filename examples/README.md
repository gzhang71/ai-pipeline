# Examples

Five runnable scripts, one per subproject. Every one runs **offline** against
fakes — no API key, no network — so you can read the output before deciding
whether the idea is worth spending tokens on.

```sh
bash examples/run_all.sh          # all five
.venv/bin/python examples/01_prompt_caching.py   # or one at a time
```

| Script | Subproject | What it demonstrates |
|---|---|---|
| `01_prompt_caching.py` | `prompt/` | Assembling a cacheable prompt, the ordering error, catching a `datetime.now()` in the prefix, reading a cache miss out of `usage` |
| `02_prompt_regression.py` | `harness/` | Two prompt versions over one task set, and a regression hiding under an unchanged pass rate |
| `03_context_profile.py` | `loop/` | Per-turn token decomposition, reconciliation against `usage`, the stacked renderer, accuracy vs. prompt length |
| `04_compaction_bench.py` | `context/` | Six strategies applied to one over-budget history, tool-pairing safety, the full bench |
| `05_jit_retrieval.py` | `graph/` | Building the code graph from this repo, the identifier-only prompt, the four tools, tuning the baseline's `k` |

## What each one is really showing

**01** ends on the failure mode worth internalizing: a prefix that changes every
request never reports as a cache MISS. It reports as a perpetual WRITE — you pay
the write premium forever and never read. The linter in step 3 is what catches
that before it ships.

**02** is the one to read if you only read one. Both prompt versions score 5/7.
The aggregate delta is +0%. There is still a real regression, named in the diff,
and the report tells you the task set is too small to detect it statistically
(7 tasks; you'd need 6 one-directional flips to clear p=0.05). Pass rates on
small task sets are theatre; the regression list is the product.

**03** prints a residual on every turn — the gap between the summed segments and
the authoritative `usage` total. Sometimes it exceeds tolerance and says so
rather than hiding it. The decomposition is exactly additive by construction,
which is arithmetic, not proof that any single segment number is "true".

**04** shows all six strategies keeping the request *legal* — a strategy that
produces a 400 is worse than useless. Note the history it builds has plain text
turns between the tool exchanges: a history of back-to-back tool pairs has almost
no legal place to split, and the summarizing strategies would find "nothing old
enough to summarize" and pass it straight through.

**05** builds a graph of this repository and reports that 67% of call sites can't
be resolved statically. That number is the honest framing for everything else the
graph claims.

## These are demonstrations, not evidence

Every script runs against fakes. Fakes prove the plumbing works and the code is
internally consistent. They do not prove the approaches are good:

- `03`'s reconciliation is checked against a *heuristic* counter, not the real
  `count_tokens` endpoint.
- `04`'s ranking is conditioned on the fake summarizer's loss model, which
  preserves values exactly and never paraphrases.
- `05`'s correctness figures measure the fakes — its baseline fake is a perfect
  reader and its JIT fake is a fixed three-step policy.

Set `ANTHROPIC_API_KEY` (or run `ant auth login`) and the live paths in each
subproject become reachable; each README says which ones.

## Artifacts

Scripts write to `runs/` (gitignored):

```
runs/example-02/    run records + judge cache
runs/example-03/    sweep.jsonl + profile.html
```

`runs/example-03/profile.html` is a self-contained page — inline SVG and CSS,
no external requests. Open it in a browser.
