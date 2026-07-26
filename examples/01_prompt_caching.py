#!/usr/bin/env python
"""Cache-aware prompt assembly, end to end.

Four things, in order:

  1. Assemble a prompt whose blocks are ordered so it can actually cache.
  2. Watch the assembler reject an ordering that cannot.
  3. Catch a `datetime.now()` in the prefix with the linter.
  4. Read a response's usage and name the reason for a cache miss.

Runs offline. No API key, no network.

    .venv/bin/python examples/01_prompt_caching.py
"""

from __future__ import annotations

import datetime

from prompt import (
    PromptAssembler,
    PromptOrderingError,
    Stability,
    diagnose_usage,
    find_silent_invalidator,
)

RULES = "You triage inbound support tickets. " * 60
ACCOUNT = "Account tier: enterprise. Region: us-east. " * 20


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------- 1. assemble
rule("1. A prompt ordered so it can cache")

a = PromptAssembler(model="claude-opus-5")
a.add_tool(
    "lookup_ticket",
    "Fetch a ticket by id. Call this whenever the user names a ticket number.",
    {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    Stability.STATIC,
    label="lookup_ticket",
)
a.add_system(RULES, Stability.STATIC, label="triage-rules")
a.add_system(ACCOUNT, Stability.SESSION, label="account-context")
a.add_message("user", "Ticket 4471 is still open. What next?", Stability.TURN, label="question")

plan = a.plan()
print(f"estimated tokens: {plan.total_estimated_tokens:,}")
print(f"boundaries considered: {plan.candidates_considered}")
print(f"breakpoints placed: {len(plan.breakpoints)} (API allows at most 4)\n")
for bp in plan.breakpoints:
    print(f"  after {bp.label!r:20} [{bp.section}] protects ~{bp.prefix_tokens:>6,} tokens")
for warning in plan.warnings:
    print(f"  ! {warning}")

kwargs = a.to_request_kwargs(max_tokens=1024)
print(f"\nready for client.messages.create(**kwargs): {sorted(kwargs)}")
marked = [i for i, b in enumerate(kwargs["system"]) if "cache_control" in b]
print(f"system blocks carrying cache_control: {marked}")


# ------------------------------------------------------------- 2. bad ordering
rule("2. An ordering that cannot cache is refused at construction")

bad = PromptAssembler(model="claude-opus-5")
bad.add_system(f"Session started {datetime.datetime.now()}", Stability.TURN, label="timestamp")
bad.add_system(RULES, Stability.STATIC, label="triage-rules")
bad.add_message("user", "Ticket 4471 is still open. What next?", label="question")

try:
    bad.validate()
    print("no error raised -- unexpected")
except PromptOrderingError as exc:
    print(exc)


# ----------------------------------------------------------------- 3. linting
rule("3. Finding the byte that changed between two identical-looking requests")


def build_leaky() -> dict:
    """A prompt builder with a timestamp buried in the system prompt."""
    asm = PromptAssembler(model="claude-opus-5")
    asm.add_system(f"{RULES}\nRendered at {datetime.datetime.now().isoformat()}", Stability.STATIC)
    asm.add_message("user", "Ticket 4471 is still open. What next?")
    return asm.to_request_kwargs()


diff = find_silent_invalidator(build_leaky)
print(f"prefix stable across two builds: {diff.identical}")
if not diff.identical:
    print(f"first divergent byte offset: {diff.offset}")
    if diff.span is not None:
        print(f"owning block: {diff.span.label!r}  (section: {diff.span.section})")
    print()
    print(diff.describe())


# ------------------------------------------------------------- 4. diagnostics
rule("4. Reading usage: was that a hit, a write, or a silent miss?")


class Usage:
    """Stand-in for a real response.usage."""

    def __init__(self, inp, creation, read):
        self.input_tokens = inp
        self.cache_creation_input_tokens = creation
        self.cache_read_input_tokens = read
        self.output_tokens = 120


scenarios = [
    ("warm cache, working as intended", Usage(40, 0, 9_600)),
    ("first request of the session", Usage(40, 9_600, 0)),
    ("prefix changed every request", Usage(40, 9_600, 0)),
]

for label, usage in scenarios:
    first = label.startswith("first")
    d = diagnose_usage(
        usage,
        model="claude-opus-5",
        breakpoints_placed=2,
        breakpoint_prefix_tokens=[4_800, 9_600],
        first_request=first,
    )
    print(f"\n{label}")
    print(f"  status: {d.status.name}   tokens: {d.tokens}")
    for cause in d.likely_causes:
        print(f"    likely cause: {cause.name}")
    for note in d.notes:
        print(f"    - {note}")

print(
    "\nNote the third case: a prefix that changes every request never shows up as a"
    "\nMISS. It reports as a perpetual WRITE -- you pay the write premium forever and"
    "\nnever read. That is the failure mode step 3 catches before you deploy it."
)
