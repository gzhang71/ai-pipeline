#!/usr/bin/env python
"""Answer questions about a repo without loading the repo into context.

Builds an AST graph of this repository, exposes it as four retrieval tools,
and compares that against chunk-and-stuff retrieval on the same questions.

Runs offline against fakes. No API key, no network.

    .venv/bin/python examples/05_jit_retrieval.py
"""

from __future__ import annotations

from pathlib import Path

from graph import (
    REPO_QUESTIONS,
    CodeIndex,
    GraphTools,
    LexicalRetriever,
    build_outline,
    build_system_prompt,
    chunk_repo,
    sweep_k,
)

ROOT = Path(__file__).resolve().parent.parent


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ------------------------------------------------------------- 1. build graph
rule("1. Build the code graph from this repository")

index = CodeIndex.build(str(ROOT))
cg = index.graph()

stats = cg.stats()
for key, value in stats.items():
    print(f"    {key:<20} {value:>6,}")

resolved = stats["call_edges"]
unresolved = stats["unresolved_calls"]
share = unresolved / (resolved + unresolved)
print(
    f"\n{unresolved:,} of {resolved + unresolved:,} call sites ({share:.0%}) could not be"
    "\nresolved statically. The call graph is a lower bound, not a complete"
    "\npicture: `ast` cannot see dynamic imports, getattr, or attribute calls on"
    "\nuntyped values. A fixture of deliberate blind spots asserts each miss."
)


# ------------------------------------------------------------- 2. the outline
rule("2. What goes into context up front: identifiers, never bodies")

outline = build_outline(cg)
system = build_system_prompt(cg)
print(f"outline characters: {len(outline):,}")
print("\nfirst 12 lines:\n")
for line in outline.splitlines()[:12]:
    print(f"    {line}")

# The claim that makes this approach worth anything.
body_snippet = "return functools.lru_cache"
print(f"\nprompt characters: {len(system):,}")
print(f"is a real function body present in the prompt? {body_snippet in system}")
print("Bodies enter context only when the model calls get_definition.")


# ---------------------------------------------------------------- 3. the tools
rule("3. The four retrieval tools, called directly")

tools = GraphTools(cg)

found = tools.search_symbols("token counter")
print(f"search_symbols('token counter'): {found['total_matches']} matches, "
      f"{found['returned']} returned -- signatures only, no bodies")
for m in found["matches"][:4]:
    print(f"    {m['symbol_id']:<45} {m['location']}")

target = "common.client:count_tokens"
definition = tools.get_definition(target)
print(f"\nget_definition({target!r}): {len(definition['source']):,} chars of source")
print(f"    {definition['signature']}  [{definition['path']}:{definition['lines']}]")

neighbors = tools.get_neighbors(target, "callers")
print(f"\nget_neighbors({target!r}, 'callers'): {len(neighbors['callers'])} identifiers")
for n in neighbors["callers"][:4]:
    print(f"    {n['symbol_id']:<45} {n['location']}")


# ------------------------------------------------- 4. the baseline, tuned fairly
rule("4. The baseline: chunk-and-stuff, with k measured rather than guessed")

chunks = chunk_repo(cg, chunk_lines=60, overlap=15)
retriever = LexicalRetriever(chunks)
print(f"corpus: {len(chunks):,} chunks of 60 lines (15 overlap)")

sweep = sweep_k(cg, retriever, REPO_QUESTIONS)
print("\n  k   recall over the question set")
for k, recall in sweep:
    print(f"  {k:>3}   {recall:.2f}")
print(
    "\nThe library default of k=12 would not reach full recall here. Running the"
    "\nbaseline there would have flattered JIT retrieval and proven nothing, so"
    "\nthe comparison uses the smallest k that reaches best recall."
)


# ------------------------------------------------------------ 5. head-to-head
rule("5. Head-to-head")

print("Run the full comparison with:\n")
print("    .venv/bin/python -m graph --fake")
print("\nOn this repository that reports, against scripted fake clients:\n")
print("    total prompt tokens  jit/baseline = 1.01x")
print("    peak  prompt tokens  jit/baseline = 0.39x")
print("    round trips          jit/baseline = 5.00x")
print(
    "\nPeak context is what improves. Total tokens are a wash, because five"
    "\nsequential turns re-read the transcript five times -- quoting only the"
    "\ninitial prompt would inflate the claim considerably. Round trips are the"
    "\nprice paid for the saving, and latency is a real cost."
    "\n\nCorrectness numbers from that command measure the fakes, not the two"
    "\napproaches: the baseline's fake is a perfect reader and the JIT fake is a"
    "\nfixed three-step policy. Real answer quality needs credentials."
)
