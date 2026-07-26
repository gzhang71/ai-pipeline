"""The central claim: the initial prompt carries identifiers, never bodies.

If this file ever fails, the experiment is invalid -- the JIT arm would be
starting with the code already in context.
"""

from __future__ import annotations

import json

from graph.agent import build_outline, build_system_prompt
from graph.builder import CodeGraph
from graph.tools import TOOL_DEFINITIONS

SENTINELS = (
    "SENTINEL_SLUGIFY_BODY",
    "SENTINEL_SHOUT_BODY",
    "SENTINEL_WIDGET_SLUG_BODY",
    "SENTINEL_WIDGET_LOUD_BODY",
    "SENTINEL_GADGET_BODY",
    "SENTINEL_CLEAN_BODY",
    "SENTINEL_BUILD_BODY",
    "SENTINEL_MAIN_BUILD_BODY",
    "SENTINEL_MAIN_BLANK_BODY",
    "SENTINEL_DYNAMIC_IMPORT_BODY",
    "SENTINEL_GETATTR_BODY",
    "SENTINEL_UNTYPED_BODY",
    "SENTINEL_ALIAS_BODY",
)


def _body_lines(graph: CodeGraph) -> list[tuple[str, str]]:
    """Every substantive line strictly inside a callable, with its owner."""
    out: list[tuple[str, str]] = []
    for node in graph.nodes.values():
        if node.kind not in ("function", "method"):
            continue
        lines = graph.source_of(node).splitlines()
        for raw in lines[1:]:  # skip the `def` line itself
            line = raw.strip()
            if len(line) < 12:
                continue
            if line.startswith(('"""', "'''", "#")):
                continue  # docstrings and comments are summary, not body
            out.append((node.id, line))
    return out


def test_the_fixture_actually_has_body_lines_to_check(sample_graph: CodeGraph):
    """Guard against the assertions below passing vacuously."""
    bodies = _body_lines(sample_graph)
    assert len(bodies) >= 10


def test_initial_prompt_contains_no_sentinel_from_any_body(sample_graph: CodeGraph):
    prompt = build_system_prompt(sample_graph)
    for sentinel in SENTINELS:
        assert sentinel not in prompt, f"{sentinel} leaked into the initial prompt"


def test_initial_prompt_contains_no_body_line_at_all(sample_graph: CodeGraph):
    """Stronger than the sentinels: no substantive in-function line appears."""
    prompt = build_system_prompt(sample_graph)
    leaked = [
        (symbol_id, line)
        for symbol_id, line in _body_lines(sample_graph)
        if line in prompt
    ]
    assert leaked == [], f"body lines leaked into the initial prompt: {leaked[:3]}"


def test_initial_prompt_contains_no_module_level_constant_values(
    sample_graph: CodeGraph,
):
    """Module-level literals are file contents too -- they must be fetched."""
    prompt = build_system_prompt(sample_graph)
    for literal in ('SEPARATOR = "-"', "RETRY_LIMIT = 7", "MAX_WIDGETS = 3"):
        assert literal not in prompt


def test_full_first_request_payload_contains_no_bodies(sample_graph: CodeGraph):
    """Check the whole rendered request, not just the system prompt."""
    payload = json.dumps(
        {
            "system": build_system_prompt(sample_graph),
            "tools": TOOL_DEFINITIONS,
            "messages": [{"role": "user", "content": "what does slugify return?"}],
        }
    )
    for sentinel in SENTINELS:
        assert sentinel not in payload


def test_outline_does_contain_identifiers_and_signatures(sample_graph: CodeGraph):
    """The prompt must still be useful: identifiers and signatures are present."""
    outline = build_outline(sample_graph)
    assert "pkg.util:slugify  def slugify(text: str) -> str" in outline
    assert "pkg.core:Widget  class Widget" in outline
    assert "pkg.core:Widget.loud  def loud(self) -> str" in outline
    assert "pkg.core:build._clean" in outline  # nested symbols too
    assert "Lowercase a string and join its words" in outline  # docstring head


def test_outline_caps_and_discloses_truncation(sample_graph: CodeGraph):
    outline = build_outline(sample_graph, max_symbols_per_module=1)
    assert "more symbols" in outline
    assert "search_symbols" in outline


def test_outline_is_far_smaller_than_the_corpus(sample_graph: CodeGraph):
    corpus = sum(
        len(open(f"{sample_graph.root}/{rel}", encoding="utf-8").read())
        for rel in sample_graph.files
    )
    outline = build_outline(sample_graph)
    assert len(outline) < corpus
