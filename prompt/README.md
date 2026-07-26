# `prompt` — a cache-aware prompt assembler

Anthropic prompt caching fails silently. When it works you save ~90% on the
cached prefix; when it doesn't, the request still succeeds, the answer is still
correct, and the only symptom is that `cache_read_input_tokens` is always `0`.
Nothing raises. Nothing warns.

This package makes caching correct by construction instead of by luck: prompt
content is tagged with a stability level, assembly refuses orderings that
cannot cache, breakpoints are placed automatically inside the API's limits, a
linter finds the exact byte that broke a prefix, and a diagnostic reads a
response's `usage` and names the likely cause of a miss.

## The invariant, in plain language

**Prompt caching is a prefix match over the rendered request bytes.**

The API renders a request in a fixed order — `tools`, then `system`, then
`messages` — and hashes the bytes up to each `cache_control` breakpoint. If a
single byte at position N differs from last time, every breakpoint at or after
N is a different key, so all of it re-processes at full price.

Everything else follows from that one sentence:

- **Stable content must physically come first.** A `datetime.now()` in the
  system prompt header sits near byte 0, so it invalidates the entire request
  below it — tools, system, and the whole conversation — on every single call.
- **Tools are the most dangerous place to be non-deterministic.** They render at
  position 0. A tool list built from a `dict` or a per-user registry that comes
  out in a different order forks the cache for the whole prompt.
- **Structure isn't enough; bytes are what count.** Two structurally identical
  prompts that serialize differently (`json.dumps` without `sort_keys`, an
  iterated `set`) are two distinct cache entries.

## Failure modes this catches

| Failure | How it shows up in production | What this package does |
| --- | --- | --- |
| Volatile block ahead of a stable one | Cache never reads | `PromptOrderingError` at construction, naming the block |
| `datetime.now()` / `uuid4()` in the prefix | Cache never reads | `find_silent_invalidator` reports the byte offset and owning block |
| Unsorted `json.dumps`, varying tool order | Cache reads intermittently or never | `canonical_json`, `stable_tool_order`, and tools sorted by name on assembly |
| More than 4 breakpoints | HTTP 400 | Placement is capped at `MAX_CACHE_BREAKPOINTS`, keeping the ones that protect the most tokens |
| Prefix below the model minimum | Cache never reads, no error | Breakpoint is dropped and a warning explains why |
| Long agentic turn exceeds the 20-block lookback | Cache silently misses after a big turn | `CachePlan.warnings` flags it before you send |
| Concurrent fan-out with a shared prefix | Every parallel request pays full price | `diagnose_usage(..., concurrent_requests=N)` names it |
| Reading `usage.input_tokens` as prompt size | Cost dashboards understate cache-warm traffic | Diagnostics report `input + cache_creation + cache_read` |

## Usage

```python
from prompt import PromptAssembler, Stability, diagnose_usage, assert_stable_prefix

def build_request(question: str, account_summary: str) -> dict:
    a = PromptAssembler(model="claude-opus-5", ttl="1h")

    # tools render first -> they must be the most stable thing in the request
    a.add_tool("search_docs", "Search the product docs.", {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    })

    a.add_system(FROZEN_PERSONA, Stability.STATIC)      # never changes
    a.add_system(account_summary, Stability.SESSION)    # per user
    a.add_message("user", question, Stability.TURN)     # per request

    return a.to_request_kwargs(max_tokens=16000)

kwargs = build_request("What is my quota?", summary)
response = client.messages.create(**kwargs)            # exact kwargs, ready to send

print(diagnose_usage(response.usage, model="claude-opus-5").describe())
```

Inspect the placement decisions before sending:

```python
a = PromptAssembler(model="claude-opus-5")
...
print(a.plan().describe())
# model=claude-opus-5 min_cacheable_prefix=512 tokens est_prompt=1555 tokens
# breakpoints: 2/4 (from 2 stability boundaries)
#   - system: system:You are a support agent. Yo… [STATIC] protects ~757 tokens (ttl=5m)
#   - system: system:Account tier: enterprise. A… [SESSION] protects ~1544 tokens (ttl=5m)
```

`CachePlan.warnings` carries everything the planner declined to do and why —
boundaries skipped for being under the model minimum, breakpoints dropped at the
4-marker limit, and the 20-block lookback window being exceeded.

Wire the linter into your own test suite — it needs no credentials:

```python
def test_prompt_prefix_is_deterministic():
    assert_stable_prefix(lambda: build_request("fixed question", FIXED_SUMMARY))
```

If it isn't, you get the byte:

```
prefix diverges at byte 104 of 3047, in section 'system', block 'system[0]'.
  A: 'e an agent. Current time: 2026-07-26T05:45:37.246376+00:00","type":"text"}'
  B: 'e an agent. Current time: 2026-07-26T05:45:37.247808+00:00","type":"text"}'
Everything from byte 104 onward is a different cache entry on every request.
Make that block deterministic, or move it after the last cache breakpoint.
```

### Multi-turn / agentic loops

Content you already sent is, by definition, no longer changing. Demote it
before appending the next turn, or the new turn will look like an ordering
violation and nothing in the history can hold a breakpoint:

```python
a.settle_history()                                   # TURN -> SESSION
a.add_message("user", next_question, Stability.TURN)
```

Mark the last block of each completed turn with `checkpoint=True` for the
rolling multi-turn pattern; candidates then compete for the four available
breakpoints, and the ones protecting the most tokens win.

## Public surface

- **Block model** — `Stability` (`STATIC` / `SESSION` / `TURN`), `ToolBlock`,
  `SystemBlock`, `MessageBlock`.
- **Assembly** — `PromptAssembler` (`add_tool`, `add_system`, `add_message`,
  `add_block`, `settle_history`, `validate`, `plan`, `to_request_kwargs`),
  `CachePlan`, `PlacedBreakpoint`, `estimate_tokens`.
- **Deterministic serialization** — `canonical_json`, `canonical_bytes`,
  `stable_tool_order`, `render_prefix`, `RenderedPrefix`, `Span`.
- **Linting** — `diff_requests`, `find_silent_invalidator`,
  `assert_stable_prefix`, `PrefixDiff`.
- **Diagnostics** — `diagnose_usage`, `CacheDiagnosis`, `CacheStatus`,
  `MissReason`.
- **Errors** — `PromptCacheError`, `PromptOrderingError`,
  `PromptStructureError`, `SilentInvalidatorError`.

Model constants (`MODEL`, `MIN_CACHEABLE_PREFIX_TOKENS`, `MAX_CACHE_BREAKPOINTS`,
`CACHE_LOOKBACK_BLOCKS`, `RENDER_ORDER`) come from `common.client` and are not
re-exported here.

## What this does **not** verify

Being honest about the boundary matters more than the feature list, because
everything below is a place where this package can look green and the cache can
still miss.

- **It cannot prove a cache hit.** A hit is a fact about Anthropic's servers.
  The only evidence is `usage.cache_read_input_tokens` from a real response.
  Everything offline here proves the necessary conditions, not the outcome.
- **Token counts are estimates.** The default counter is ~4 characters per
  token. It is used for breakpoint ranking and the below-minimum warning, so a
  prompt sitting near the model's minimum can be misjudged in either direction.
  Pass `token_counter=` a function backed by `common.client.count_tokens` (which
  calls the `count_tokens` endpoint and therefore needs credentials) when the
  margin matters.
- **The rendered prefix is a faithful proxy, not the API's serializer.** The
  section order and per-block boundaries match; the framing bytes do not. A
  diff is reliable evidence that *your* inputs changed. Byte-identical output
  here does not guarantee byte-identical output on the wire.
- **Stability tags are claims, not proofs.** Tagging a block `STATIC` does not
  make it static. Ordering validation only checks that the claims are mutually
  consistent — the linter is what catches a lie, and only for non-determinism
  that reproduces across two calls in the same process.
- **Cross-process and cross-deploy drift is invisible.** A prompt that renders
  identically twice in one test run can still differ between machines,
  deployments, or library versions.
- **It does not model TTL expiry, eviction, or capacity.** A 5-minute entry that
  expired between requests is indistinguishable, offline, from a changed prefix
  — `diagnose_usage` reports it as a repeated write and says so.
- **It does not simulate the API.** `prompt.fakes` (`FakeClient`,
  `FakeCacheServer`) exists so the test suite can exercise prefix-match
  semantics offline. It implements only the rules encoded here and is not a
  substitute for a live call.

## Tests

```sh
.venv/bin/python -m pytest prompt/ -q
```

The suite passes with **zero network access** and no credentials. Anything that
would make a real API call is out of scope for it by design; there is no live
test to guard with `common.client.has_credentials()` because nothing in this
package calls the API.
