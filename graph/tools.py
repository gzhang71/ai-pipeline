"""The JIT retrieval tool surface, in Anthropic tool-use format.

Design constraint: the initial prompt carries *identifiers and signatures only*.
No file contents reach the context window until the model asks for them, and
when it does it gets exactly one node's body, or a bounded line range.

The descriptions below are deliberately prescriptive -- they say *when* to call
each tool, not only what it does. Recent models are conservative about reaching
for tools, and trigger conditions written into the tool description measurably
raise the call rate compared with a description that only states behaviour.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .builder import CodeGraph, Node

MAX_READ_LINES = 400
DEFAULT_SEARCH_LIMIT = 15
MIN_SUBSTRING_TERM = 4

# Question words carry no signal about which symbol is wanted, and matching them
# ranks arbitrary code above the real hit. The baseline retriever gets the same
# effect for free from idf weighting.
STOPWORDS = frozenset(
    """
    a an and are as at be by do does doing done for from has have how i in is it
    its no not of on or that the their then there these this to two three four
    what when where which who why with you your value values name names return
    returns use uses used using
    """.split()
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_symbols",
        "description": (
            "Find symbols (modules, classes, functions, methods) in the repository "
            "by name, path, or docstring text. Returns identifiers and signatures "
            "ONLY -- never source bodies.\n"
            "\n"
            "Call this FIRST for any question about the repository, before you "
            "answer from the outline in the system prompt and before any other "
            "tool. The outline is partial; search covers every symbol. Call it "
            "again with different wording whenever the first query returns "
            "nothing that obviously matches -- a second query is far cheaper than "
            "a wrong answer. Prefer several short queries over one long one: "
            "query terms are matched independently."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Words to match: an identifier ('usage_breakdown'), a "
                        "concept ('token counting'), a path fragment "
                        "('common/client'), or the name of a module-level "
                        "constant ('MAX_RETRIES') -- constants are not symbols "
                        "of their own, so matching one returns the module that "
                        "defines it."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": ["module", "class", "function", "method", "any"],
                    "description": "Restrict to one node kind. Defaults to 'any'.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max matches to return (default {DEFAULT_SEARCH_LIMIT}).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_definition",
        "description": (
            "Return the full source of exactly one symbol, identified by the "
            "symbol_id that search_symbols or get_neighbors gave you.\n"
            "\n"
            "Call this as soon as a symbol's name or signature looks relevant -- "
            "do NOT guess what a function does from its name, and do NOT answer "
            "a question about behaviour, return values, or constants without "
            "reading the body first. One call returns one node, so call it once "
            "per symbol you need; calling it two or three times in a turn is "
            "normal and expected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol_id": {
                    "type": "string",
                    "description": (
                        "Exact identifier, e.g. 'common.client:usage_breakdown' "
                        "or 'graph.builder:CodeGraph._resolve_call'. Module ids "
                        "have no colon, e.g. 'common.client'."
                    ),
                }
            },
            "required": ["symbol_id"],
        },
    },
    {
        "name": "get_neighbors",
        "description": (
            "List the graph neighbours of one symbol as identifiers and "
            "signatures -- callers, callees, imports, importers, or contained "
            "children. No bodies are returned.\n"
            "\n"
            "Call this when the answer depends on how a symbol is used or wired "
            "up rather than on its own body: 'who calls X', 'what does X depend "
            "on', 'what is inside module M', 'which modules import M'. Also call "
            "it after get_definition when the body delegates to another symbol "
            "you have not read yet -- it is cheaper than searching blind. Call "
            "with direction 'children' on a module id to enumerate that module's "
            "top-level symbols."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol_id": {"type": "string", "description": "Exact identifier."},
                "direction": {
                    "type": "string",
                    "enum": [
                        "callers",
                        "callees",
                        "imports",
                        "imported_by",
                        "children",
                        "parent",
                        "all",
                    ],
                    "description": (
                        "Which relation to follow. 'imports'/'imported_by' apply "
                        "to modules. Defaults to 'all'."
                    ),
                },
            },
            "required": ["symbol_id"],
        },
    },
    {
        "name": "read_lines",
        "description": (
            "Read a bounded range of raw lines from one file.\n"
            "\n"
            "This is the fallback, not the default. Use get_definition for "
            "anything that is a symbol. Call this tool only for code that is "
            "not inside any function or class -- module-level constants, "
            "dunder blocks, decorators, configuration literals -- or to see the "
            "few lines surrounding a definition you already read. Keep the range "
            f"under {MAX_READ_LINES} lines; larger requests are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repository-relative path, e.g. 'common/client.py'.",
                },
                "start": {"type": "integer", "description": "First line, 1-based."},
                "end": {"type": "integer", "description": "Last line, inclusive."},
            },
            "required": ["path", "start", "end"],
        },
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOL_DEFINITIONS)


# --------------------------------------------------------------------------
# Lexical scoring shared by search_symbols and the baseline retriever
# --------------------------------------------------------------------------

_SPLIT = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with snake_case and camelCase split apart."""
    out: list[str] = []
    for chunk in _SPLIT.split(text):
        if not chunk:
            continue
        for piece in _CAMEL.split(chunk):
            piece = piece.lower()
            if piece:
                out.append(piece)
    return out


@dataclass(frozen=True)
class SearchHit:
    node: Node
    score: float


class GraphTools:
    """Executes the four tools against a `CodeGraph`."""

    def __init__(self, graph: CodeGraph) -> None:
        self.graph = graph
        self._haystack: dict[str, set[str]] = {}
        for node in graph.nodes.values():
            text = " ".join(
                [node.id, node.path, node.signature, node.doc, *node.defines]
            )
            self._haystack[node.id] = set(tokenize(text))

    # -- tool 1 ----------------------------------------------------------
    def search_symbols(
        self, query: str, kind: str = "any", limit: int = DEFAULT_SEARCH_LIMIT
    ) -> dict[str, Any]:
        terms = [t for t in tokenize(query) if t not in STOPWORDS]
        if not terms:
            return {"query": query, "matches": [], "note": "empty query"}
        wanted = None if kind in (None, "", "any") else {kind}
        hits: list[SearchHit] = []
        for node in self.graph.nodes.values():
            if wanted and node.kind not in wanted:
                continue
            score = self._score(node, terms)
            if score > 0:
                hits.append(SearchHit(node, score))
        hits.sort(key=lambda h: (-h.score, h.node.path, h.node.lineno))
        limit = max(1, min(int(limit or DEFAULT_SEARCH_LIMIT), 50))
        top = hits[:limit]
        return {
            "query": query,
            "total_matches": len(hits),
            "returned": len(top),
            "matches": [self._match(h.node) for h in top],
            "note": "signatures only -- call get_definition for a body",
        }

    def _match(self, node: Node) -> dict[str, Any]:
        match: dict[str, Any] = {
            "symbol_id": node.id,
            "kind": node.kind,
            "signature": node.signature,
            "doc": node.doc,
            "location": f"{node.path}:{node.lineno}-{node.end_lineno}",
        }
        if node.defines:
            # Names only. The values are file contents and are not disclosed
            # here -- call get_definition on the module to see them.
            match["defines_names"] = list(node.defines)
        return match

    def _score(self, node: Node, terms: Iterable[str]) -> float:
        bag = self._haystack[node.id]
        name_tokens = set(tokenize(node.name))
        # A module-level constant's name is an identifier too, so an exact hit
        # on one should rank its module near a name match, not near a docstring
        # match -- otherwise "MIN_CACHEABLE_PREFIX_TOKENS" loses to any function
        # whose prose happens to mention tokens.
        define_tokens = {t for name in node.defines for t in tokenize(name)}
        score = 0.0
        for term in terms:
            if term in name_tokens:
                score += 3.0
            elif term in define_tokens:
                score += 2.0
            elif term in bag:
                score += 1.0
            elif len(term) >= MIN_SUBSTRING_TERM and any(
                term in tok for tok in bag
            ):
                # Substring credit only for terms long enough to be meaningful;
                # short fragments match half the repo and drown the real hits.
                score += 0.35
        if score and node.kind in ("function", "method", "class"):
            score += 0.2  # mild preference for callable symbols over whole modules
        return score

    # -- tool 2 ----------------------------------------------------------
    def get_definition(self, symbol_id: str) -> dict[str, Any]:
        node = self.graph.node(symbol_id)
        if node is None:
            return {
                "error": f"unknown symbol_id {symbol_id!r}",
                "hint": "call search_symbols to get exact identifiers",
                "did_you_mean": [
                    m["symbol_id"]
                    for m in self.search_symbols(symbol_id, limit=5)["matches"]
                ],
            }
        source = self.graph.source_of(node)
        return {
            "symbol_id": node.id,
            "kind": node.kind,
            "signature": node.signature,
            "path": node.path,
            "lines": f"{node.lineno}-{node.end_lineno}",
            "source": source,
        }

    # -- tool 3 ----------------------------------------------------------
    def get_neighbors(self, symbol_id: str, direction: str = "all") -> dict[str, Any]:
        node = self.graph.node(symbol_id)
        if node is None:
            return {
                "error": f"unknown symbol_id {symbol_id!r}",
                "hint": "call search_symbols to get exact identifiers",
            }
        graph = self.graph
        buckets: dict[str, list[str]] = {
            "callers": list(graph.callers.get(symbol_id, [])),
            "callees": list(graph.callees.get(symbol_id, [])),
            "imports": list(graph.imports.get(symbol_id, [])),
            "imported_by": list(graph.imported_by.get(symbol_id, [])),
            "children": list(graph.children.get(symbol_id, [])),
            "parent": [node.parent] if node.parent else [],
        }
        if direction not in (None, "", "all"):
            if direction not in buckets:
                return {"error": f"unknown direction {direction!r}"}
            buckets = {direction: buckets[direction]}
        out: dict[str, Any] = {"symbol_id": symbol_id, "kind": node.kind}
        for name, ids in buckets.items():
            out[name] = [self._identify(i) for i in sorted(ids)]
        if node.kind == "module":
            out["external_imports"] = sorted(
                set(graph.external_imports.get(symbol_id, []))
            )
        out["note"] = (
            "identifiers only. Call edges are statically resolved and incomplete: "
            "dynamic imports, getattr, and attribute calls on untyped values are "
            "not represented."
        )
        return out

    def _identify(self, symbol_id: str) -> dict[str, str]:
        node = self.graph.node(symbol_id)
        if node is None:
            return {"symbol_id": symbol_id, "kind": "unknown", "signature": ""}
        return {
            "symbol_id": node.id,
            "kind": node.kind,
            "signature": node.signature,
            "location": f"{node.path}:{node.lineno}-{node.end_lineno}",
        }

    # -- tool 4 ----------------------------------------------------------
    def read_lines(self, path: str, start: int, end: int) -> dict[str, Any]:
        rel = path.replace(os.sep, "/").lstrip("/")
        abs_path = os.path.normpath(os.path.join(self.graph.root, rel))
        if not abs_path.startswith(self.graph.root):
            return {"error": "path escapes the repository root"}
        if not os.path.isfile(abs_path):
            return {"error": f"no such file {rel!r}"}
        with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
        start = max(1, int(start))
        end = min(len(lines), int(end))
        truncated = False
        if end - start + 1 > MAX_READ_LINES:
            end = start + MAX_READ_LINES - 1
            truncated = True
        if start > len(lines):
            return {"error": f"start {start} is past end of file ({len(lines)} lines)"}
        body = "\n".join(lines[start - 1 : end])
        return {
            "path": rel,
            "lines": f"{start}-{end}",
            "file_lines": len(lines),
            "truncated": truncated,
            "text": body,
        }

    # -- dispatch --------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_symbols":
            return self.search_symbols(
                arguments.get("query", ""),
                kind=arguments.get("kind", "any"),
                limit=arguments.get("limit", DEFAULT_SEARCH_LIMIT),
            )
        if name == "get_definition":
            return self.get_definition(arguments.get("symbol_id", ""))
        if name == "get_neighbors":
            return self.get_neighbors(
                arguments.get("symbol_id", ""),
                direction=arguments.get("direction", "all"),
            )
        if name == "read_lines":
            return self.read_lines(
                arguments.get("path", ""),
                arguments.get("start", 1),
                arguments.get("end", 1),
            )
        return {"error": f"unknown tool {name!r}", "available": list(TOOL_NAMES)}

    def call_as_text(self, name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(self.call(name, arguments), indent=1, sort_keys=False)
