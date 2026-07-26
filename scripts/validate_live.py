#!/usr/bin/env python
"""Validate this repo's claims against the real API.

Everything in this repository passes offline against fakes. Fakes establish
internal consistency; they cannot establish that the numbers are true. This
script runs the specific checks that only a live API can settle.

    .venv/bin/python scripts/validate_live.py --list
    .venv/bin/python scripts/validate_live.py --check order-sensitivity
    .venv/bin/python scripts/validate_live.py --tier free

Checks are grouped by cost:

  free    only hits POST /v1/messages/count_tokens, which is not billed.
          Latency and rate limits are the only cost. Start here.
  cheap   a handful of small completions.
  full    multi-turn agent runs; the most expensive tier.

Nothing runs without credentials -- set ANTHROPIC_API_KEY, or run
`ant auth login` and the SDK picks the profile up automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.client import MODEL, get_client, has_credentials, usage_breakdown  # noqa: E402

OUT = Path("runs/live-validation")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def verdict(ok: bool, message: str) -> dict[str, Any]:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    return {"ok": ok, "message": message}


def _sample_prompt() -> dict[str, Any]:
    """A prompt with all three components substantial enough to measure."""
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize the invoice position for INV-1003. "
                    + "Background context that must be carried. " * 40
                ),
            },
            {"role": "assistant", "content": "Let me work through the ledger carefully."},
            {"role": "user", "content": "Please show the arithmetic step by step."},
        ],
        "system": "You are a precise billing assistant. Answer tersely. " * 20,
        "tools": [
            {
                "name": "lookup_invoice",
                "description": (
                    "Fetch an invoice by id. Call this whenever the user names "
                    "an invoice number, before answering from memory. "
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            }
        ],
    }


# --------------------------------------------------------------------------
# free tier -- count_tokens only, not billed
# --------------------------------------------------------------------------


def check_order_sensitivity() -> dict[str, Any]:
    """The headline: how much does measurement order move each segment?

    Offline this always reports zero, because any chars-per-token heuristic is
    linear and a linear counter cannot be order-sensitive. Against the real
    tokenizer the spread is the error bar on every per-segment number the
    profiler has ever printed.
    """
    rule("order-sensitivity of token attribution  [free: count_tokens only]")
    from loop.attribution import api_token_counter, order_sensitivity, order_sensitivity_text

    prompt = _sample_prompt()
    started = time.perf_counter()
    report = order_sensitivity(
        prompt["messages"],
        system=prompt["system"],
        tools=prompt["tools"],
        counter=api_token_counter,
    )
    elapsed = time.perf_counter() - started

    print(order_sensitivity_text(report))
    print(f"\n  wall clock: {elapsed:.1f}s")

    results = [
        verdict(report["total_is_invariant"], "total is invariant across all 6 orders"),
        verdict(
            report["counter_is_order_sensitive"],
            "the real tokenizer IS order-sensitive (offline heuristic reports 0)",
        ),
    ]
    print(
        f"\n  => error bar on any single segment: {report['max_spread_fraction']:.2%}\n"
        "     Put this number in loop/README.md; it is what the offline suite\n"
        "     structurally cannot produce."
    )
    return {"check": "order-sensitivity", "report": report, "results": results}


def check_prefix_minimums() -> dict[str, Any]:
    """Does the minimum-prefix rule apply per breakpoint, or per prompt?

    `prompt/` assumes per-breakpoint, the conservative reading, and says so.
    This measures a prompt whose total clears the model minimum but whose first
    breakpoint prefix does not, then reports what actually cached.
    """
    rule("minimum cacheable prefix: per-breakpoint or per-prompt?  [cheap]")
    from common.client import MIN_CACHEABLE_PREFIX_TOKENS

    minimum = MIN_CACHEABLE_PREFIX_TOKENS.get(MODEL, 1024)
    print(f"  documented minimum for {MODEL}: {minimum} tokens")
    print("  building a prompt whose FIRST breakpoint prefix is below it,")
    print("  and whose total is comfortably above it.\n")

    client = get_client()
    small = "Short stable preamble. " * 8            # deliberately under the minimum
    large = "Substantial shared instructions. " * 400

    system = [
        {"type": "text", "text": small, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": large, "cache_control": {"type": "ephemeral"}},
    ]
    messages = [{"role": "user", "content": "Reply with the single word OK."}]

    first = client.messages.create(
        model=MODEL, max_tokens=16, system=system, messages=messages
    )
    time.sleep(1.0)  # the entry is readable once the first response has begun
    second = client.messages.create(
        model=MODEL, max_tokens=16, system=system, messages=messages
    )

    u1, u2 = usage_breakdown(first.usage), usage_breakdown(second.usage)
    print(f"  first  request: {json.dumps(u1)}")
    print(f"  second request: {json.dumps(u2)}")

    read = u2["cache_read_input_tokens"]
    results = [verdict(read > 0, f"second request read {read:,} tokens from cache")]
    if read:
        # If only the large block cached, the read is ~large. If both cached,
        # the read includes the small one too.
        print(
            "\n  => compare the read against the size of the LARGE block alone.\n"
            "     A read matching only the large block supports the per-breakpoint\n"
            "     reading; a read covering both suggests per-prompt."
        )
    return {"check": "prefix-minimums", "usage": [u1, u2], "results": results}


# --------------------------------------------------------------------------
# cheap tier -- a few small completions
# --------------------------------------------------------------------------


def check_reconciliation() -> dict[str, Any]:
    """Does the decomposition agree with billed `usage` on a real request?"""
    rule("attribution reconciles against real usage  [cheap]")
    from loop.attribution import api_token_counter, attribute, reconcile

    prompt = _sample_prompt()
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        system=prompt["system"],
        tools=prompt["tools"],
        messages=prompt["messages"],
    )
    usage = usage_breakdown(response.usage)
    attribution = attribute(
        prompt["messages"],
        system=prompt["system"],
        tools=prompt["tools"],
        counter=api_token_counter,
    )
    recon = reconcile(attribution, usage)

    for segment in attribution.segments:
        flag = "  ~approx" if segment.approximate else ""
        print(f"    {segment.kind:<16} {segment.tokens:>7,}{flag}")
    print(f"\n  counted:       {recon['counted_total']:,}")
    print(f"  authoritative: {recon['authoritative_total']:,}  (input+creation+read)")
    print(f"  residual:      {recon['residual_tokens']:+,} ({recon['residual_fraction']:.2%})")

    results = [
        verdict(
            attribution.segment_sum == attribution.counted_total,
            "decomposition is exactly additive",
        ),
        verdict(recon["within_tolerance"], f"residual within {recon['tolerance_fraction']:.0%}"),
    ]
    negatives = [s.kind for s in attribution.segments if s.tokens < 0]
    if negatives:
        print(f"\n  note: negative segments (boundary merging): {negatives}")
    return {"check": "reconciliation", "reconciliation": recon, "results": results}


def check_cache_roundtrip() -> dict[str, Any]:
    """Does a prompt/ assembled request actually cache, and does the
    diagnostic read the result correctly?"""
    rule("prompt/ assembler produces a request that really caches  [cheap]")
    from prompt import PromptAssembler, Stability, diagnose_usage

    client = get_client()
    assembler = PromptAssembler(model=MODEL)
    assembler.add_system("Stable operating instructions. " * 400, Stability.STATIC, label="rules")
    assembler.add_message("user", "Reply with the single word OK.", Stability.TURN)
    kwargs = assembler.to_request_kwargs(max_tokens=16)

    plan = assembler.plan()
    print(f"  breakpoints placed: {len(plan.breakpoints)}")

    first = client.messages.create(**kwargs)
    time.sleep(1.0)
    second = client.messages.create(**kwargs)

    d1 = diagnose_usage(first.usage, model=MODEL, first_request=True)
    d2 = diagnose_usage(second.usage, model=MODEL)
    print(f"  first  -> {d1.status.name}  {usage_breakdown(first.usage)}")
    print(f"  second -> {d2.status.name}  {usage_breakdown(second.usage)}")

    return {
        "check": "cache-roundtrip",
        "results": [
            verdict(d1.status.name in ("WRITE", "PARTIAL"), "first request wrote the cache"),
            verdict(d2.status.name == "READ", "second request read it back"),
        ],
    }


def check_compaction_roundtrip() -> dict[str, Any]:
    """context/ simulates server compaction. Does the real contract hold?

    The trap: you must append the full `response.content` back into messages.
    Extracting only the text drops the compaction block and silently loses
    state, with no error.
    """
    rule("server-side compaction round-trip  [cheap]")
    from context import COMPACTION_BETA, COMPACTION_EDIT

    client = get_client()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Remember this number: 84217. Reply with just OK."}
    ]
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=32,
        betas=[COMPACTION_BETA],
        context_management={"edits": [COMPACTION_EDIT]},
        messages=messages,
    )
    kinds = [getattr(b, "type", "?") for b in response.content]
    print(f"  content block types: {kinds}")
    print(f"  beta accepted, stop_reason={response.stop_reason}")

    # Round-trip exactly as the docs require: full content, not extracted text.
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": "What number did I ask you to remember?"})
    follow = client.beta.messages.create(
        model=MODEL,
        max_tokens=32,
        betas=[COMPACTION_BETA],
        context_management={"edits": [COMPACTION_EDIT]},
        messages=messages,
    )
    text = "".join(getattr(b, "text", "") for b in follow.content)
    print(f"  follow-up answer: {text.strip()[:80]!r}")

    return {
        "check": "compaction-roundtrip",
        "results": [
            verdict(True, "compaction beta + edit accepted by the API"),
            verdict("84217" in text, "state survived the round-trip"),
        ],
    }


CHECKS: dict[str, tuple[str, str, Callable[[], dict[str, Any]]]] = {
    "order-sensitivity": ("free", "how much measurement order moves each segment", check_order_sensitivity),
    "reconciliation": ("cheap", "decomposition vs. billed usage", check_reconciliation),
    "prefix-minimums": ("cheap", "per-breakpoint or per-prompt minimum", check_prefix_minimums),
    "cache-roundtrip": ("cheap", "assembled request really caches", check_cache_roundtrip),
    "compaction-roundtrip": ("cheap", "server compaction contract holds", check_compaction_roundtrip),
}


def preflight() -> str | None:
    """Cheapest possible probe that the account can actually be used.

    Returns an actionable message, or None when the API is reachable.

    `count_tokens` is not billed per token, but it is still gated on the
    organization having a credit balance -- an unfunded org gets the same 400
    as any other endpoint. So there is no tier of this script that runs for
    free on an empty account, and it is worth finding that out in one call
    rather than five.
    """
    import anthropic

    try:
        get_client().messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": "."}]
        )
    except anthropic.BadRequestError as exc:
        text = str(exc)
        if "credit balance" in text.lower():
            return (
                "Authentication works, but the organization has no credit balance,\n"
                "so every endpoint returns 400 -- including count_tokens, which is\n"
                "free per token but still requires a funded account.\n\n"
                "  Add credits: https://console.anthropic.com/settings/billing\n\n"
                f"Active account: run `ant auth status` to confirm which org is in use.\n"
                f"API said: {text.strip()[:200]}"
            )
        return f"The API rejected a minimal request:\n  {text.strip()[:300]}"
    except anthropic.AuthenticationError as exc:
        return (
            "Credentials were found but rejected. Re-run `ant auth login`, or check\n"
            f"that no stale ANTHROPIC_API_KEY is shadowing the profile.\n  {exc}"
        )
    except Exception as exc:  # network, DNS, proxy
        return f"Could not reach the API: {type(exc).__name__}: {exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    parser.add_argument("--check", action="append", default=[], help="run one check (repeatable)")
    parser.add_argument("--tier", choices=["free", "cheap", "full"], help="run every check at or below a cost tier")
    parser.add_argument("--out", type=Path, default=OUT / "results.json")
    args = parser.parse_args()

    if args.list:
        print(f"{'check':<24} {'tier':<7} what it settles")
        print("-" * 72)
        for name, (tier, blurb, _) in CHECKS.items():
            print(f"{name:<24} {tier:<7} {blurb}")
        return 0

    if not has_credentials():
        print(
            "No credentials found, so nothing can be validated.\n\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...        (add to ~/.zshrc to persist)\n"
            "  or: brew install anthropics/tap/ant && ant auth login\n\n"
            "The SDK picks up either automatically -- no key is passed in code.",
            file=sys.stderr,
        )
        return 2

    problem = preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 3

    order = {"free": 0, "cheap": 1, "full": 2}
    selected = list(args.check)
    if args.tier:
        ceiling = order[args.tier]
        selected += [n for n, (tier, _, _) in CHECKS.items() if order[tier] <= ceiling]
    if not selected:
        selected = ["order-sensitivity"]  # the free, highest-information default

    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        if name not in CHECKS:
            print(f"unknown check {name!r}; --list shows the options", file=sys.stderr)
            return 2
        try:
            results.append(CHECKS[name][2]())
        except Exception as exc:  # a live API failure is data, not a crash
            rule(f"{name}  [ERROR]")
            print(f"  {type(exc).__name__}: {exc}")
            results.append({"check": name, "error": f"{type(exc).__name__}: {exc}"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))

    rule("summary")
    failures = 0
    for entry in results:
        if "error" in entry:
            print(f"  ERROR  {entry['check']}: {entry['error']}")
            failures += 1
            continue
        for result in entry.get("results", []):
            if not result["ok"]:
                failures += 1
    print(f"\n  {len(results)} check(s) run, {failures} failure(s)")
    print(f"  results written to {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
