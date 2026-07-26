"""The just-in-time retrieval agent.

The initial prompt is an *outline*: module identifiers, the symbols they
contain, their signatures and docstring first lines. No file contents. Bodies
enter the context window only when the model calls `get_definition` or
`read_lines`, and then only the bytes it asked for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from common.client import MODEL

from .builder import CodeGraph
from .tokens import TokenCounter, TokenLedger
from .tools import TOOL_DEFINITIONS, GraphTools

DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_TOKENS = 8000

# Above this many non-module symbols, listing them all in the system prompt
# costs more than it saves -- the outline is re-read on every turn. See
# `build_outline`.
SYMBOL_BUDGET = 250

SYSTEM_PREAMBLE = """\
You answer questions about a Python repository.

You have NOT been given the repository's source code -- only the index below. \
It lists identifiers, and where the repository is small enough, signatures and \
docstring first lines too. It never contains function bodies, constants, or any \
other file contents. If it lists modules but not their symbols, use \
search_symbols or get_neighbors to enumerate them.

Therefore: you cannot answer a question about behaviour, a return value, a \
literal value, or the contents of anything by reading the outline. Use the \
outline to pick identifiers, then call the tools to fetch what you need. \
Reading two or three definitions before answering is the normal cost of a \
correct answer; guessing from a name is not acceptable.

When you have what you need, answer directly and concretely: name the exact \
symbols and quote the exact values you retrieved. If a tool result contradicts \
your expectation, trust the tool result.
"""


def _graph_note(graph: CodeGraph) -> str:
    return (
        "\n# graph stats: "
        + ", ".join(f"{k}={v}" for k, v in graph.stats().items())
        + "\n# call edges are statically resolved and incomplete "
        "(dynamic imports, getattr, and attribute calls on untyped values are missing)."
    )


def _module_outline(graph: CodeGraph, modules: list[Any], symbol_count: int) -> str:
    """Module identifiers only -- the cheapest prompt that still locates code."""
    lines = [
        "REPOSITORY MODULE INDEX (module identifiers only)",
        f"# {len(modules)} modules, {symbol_count} symbols. The individual symbols are",
        "# NOT listed here. To see a module's symbols call",
        '# get_neighbors(symbol_id="<module id>", direction="children"), or search',
        "# across the whole repository with search_symbols. Both return",
        "# identifiers and signatures without bodies.",
        "",
    ]
    for module in modules:
        kids = len(graph.children.get(module.id, []))
        line = f"{module.id}  ({module.path}, {kids} top-level symbols)"
        if module.doc:
            line += f"  -- {module.doc}"
        lines.append(line)
    lines.append(_graph_note(graph))
    return "\n".join(lines)


def _outline_line(indent: str, node: Any) -> str:
    """`identifier  signature  -- docstring first line`. Never a body."""
    line = f"{indent}{node.id}  {node.signature}"
    return f"{line}  -- {node.doc}" if node.doc else line


def build_outline(
    graph: CodeGraph,
    *,
    max_symbols_per_module: int = 40,
    include_private: bool = True,
    detail: str = "auto",
    symbol_budget: int = SYMBOL_BUDGET,
) -> str:
    """Identifiers + signatures only. Never file contents.

    Two levels, because the initial prompt is a *fixed cost paid on every turn*
    and a full symbol listing stops being cheap fast:

    ``symbols``  every symbol with its signature. Good up to a few hundred
                 symbols; lets the model skip the first search.
    ``modules``  module identifiers, paths, and symbol counts only. The model
                 reaches the symbols through ``search_symbols`` or
                 ``get_neighbors(direction="children")``. Costs one extra round
                 trip and roughly an order of magnitude fewer prompt tokens.

    ``auto`` (the default) picks ``symbols`` while the repo has fewer than
    ``symbol_budget`` non-module nodes, and ``modules`` above that. Measured on
    this repository, the difference is ~46k prompt tokens against ~3k.
    """
    modules = sorted(
        (n for n in graph.nodes.values() if n.kind == "module"), key=lambda n: n.path
    )
    symbol_count = len(graph.nodes) - len(modules)
    if detail == "auto":
        detail = "symbols" if symbol_count <= symbol_budget else "modules"
    if detail == "modules":
        return _module_outline(graph, modules, symbol_count)
    if detail != "symbols":
        raise ValueError(f"unknown detail level {detail!r}")

    lines: list[str] = ["REPOSITORY OUTLINE (identifiers and signatures only)"]
    for module in modules:
        header = f"\n# {module.id}  ({module.path})"
        if module.doc:
            header += f"  -- {module.doc}"
        lines.append(header)
        if module.defines:
            # Names only -- the values are file contents and must be fetched.
            lines.append(f"  # module-level names: {', '.join(module.defines)}")
        children = [graph.nodes[c] for c in graph.children.get(module.id, [])]
        children.sort(key=lambda n: n.lineno)
        shown = 0
        for child in children:
            if not include_private and child.name.startswith("_"):
                continue
            if shown >= max_symbols_per_module:
                lines.append(
                    f"  ... {len(children) - shown} more symbols "
                    f"(use search_symbols / get_neighbors to list them)"
                )
                break
            lines.append(_outline_line("  ", child))
            grandkids = [graph.nodes[g] for g in graph.children.get(child.id, [])]
            grandkids.sort(key=lambda n: n.lineno)
            for grandkid in grandkids[:max_symbols_per_module]:
                lines.append(_outline_line("    ", grandkid))
            shown += 1
    lines.append(_graph_note(graph))
    return "\n".join(lines)


def build_system_prompt(graph: CodeGraph, **outline_kwargs: Any) -> str:
    return SYSTEM_PREAMBLE + "\n" + build_outline(graph, **outline_kwargs)


@dataclass
class AgentRun:
    question: str
    answer: str
    model_calls: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    peak_prompt_tokens: int = 0
    wall_clock: float = 0.0
    stop_reason: str = ""
    exact_tokens: bool = False

    @property
    def fetched_symbols(self) -> list[str]:
        out = []
        for call in self.tool_calls:
            if call["name"] in ("get_definition", "get_neighbors"):
                sym = call["input"].get("symbol_id")
                if sym:
                    out.append(sym)
        return out


def _block_to_param(block: Any) -> dict[str, Any]:
    """Normalize an SDK content block to a plain dict we can send back.

    `model_dump(exclude_none=True)` keeps thinking-block signatures intact, which
    the API requires when echoing a turn back on the same model.
    """
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return {"type": getattr(block, "type", "text"), "text": getattr(block, "text", "")}


def run_jit_agent(
    question: str,
    tools: GraphTools,
    client: Any,
    *,
    counter: TokenCounter,
    model: str = MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: str | None = None,
) -> AgentRun:
    """Run the tool-use loop until the model answers or the turn budget runs out."""
    system_prompt = system if system is not None else build_system_prompt(tools.graph)
    ledger = TokenLedger(counter)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    recorded_calls: list[dict[str, Any]] = []
    answer = ""
    stop_reason = "max_turns"
    started = time.perf_counter()

    for turn in range(max_turns):
        ledger.record(
            f"jit-turn-{turn}", messages, system=system_prompt, tools=TOOL_DEFINITIONS
        )
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        blocks = list(response.content)
        stop_reason = getattr(response, "stop_reason", "") or ""

        if stop_reason == "refusal":
            answer = "[model refused]"
            break

        tool_uses = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
        texts = [
            getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
        ]
        if texts:
            answer = "\n".join(t for t in texts if t).strip()

        if not tool_uses:
            break

        messages.append(
            {"role": "assistant", "content": [_block_to_param(b) for b in blocks]}
        )
        results: list[dict[str, Any]] = []
        for use in tool_uses:
            arguments = dict(getattr(use, "input", {}) or {})
            payload = tools.call_as_text(use.name, arguments)
            recorded_calls.append(
                {"name": use.name, "input": arguments, "result_chars": len(payload)}
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": use.id,
                    "content": payload,
                }
            )
        messages.append({"role": "user", "content": results})

    return AgentRun(
        question=question,
        answer=answer,
        model_calls=ledger.requests,
        tool_calls=recorded_calls,
        total_prompt_tokens=ledger.total_prompt_tokens,
        peak_prompt_tokens=ledger.peak_prompt_tokens,
        wall_clock=time.perf_counter() - started,
        stop_reason=stop_reason,
        exact_tokens=ledger.exact,
    )
