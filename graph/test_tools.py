"""Each retrieval tool returns correct results, and the schemas are well formed."""

from __future__ import annotations

import json

import pytest

from graph.index import CodeIndex
from graph.tools import (
    MAX_READ_LINES,
    TOOL_DEFINITIONS,
    TOOL_NAMES,
    GraphTools,
    tokenize,
)


# -- tool definitions ---------------------------------------------------


def test_tool_definitions_are_valid_anthropic_tools():
    assert TOOL_NAMES == (
        "search_symbols",
        "get_definition",
        "get_neighbors",
        "read_lines",
    )
    for tool in TOOL_DEFINITIONS:
        assert set(tool) == {"name", "description", "input_schema"}
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert schema["required"]
        for name in schema["required"]:
            assert name in schema["properties"], f"{tool['name']}.{name}"
        for prop in schema["properties"].values():
            assert prop.get("description") or prop.get("enum")
        json.dumps(tool)  # must be serializable as-is


def test_descriptions_are_prescriptive_about_when_to_call():
    """Wording is load-bearing: models call tools more when told when to."""
    for tool in TOOL_DEFINITIONS:
        text = tool["description"].lower()
        assert "call this" in text or "call it" in text, tool["name"]


# -- tokenizer ----------------------------------------------------------


def test_tokenize_splits_snake_and_camel_case():
    assert tokenize("usage_breakdown") == ["usage", "breakdown"]
    assert tokenize("CodeGraph.build") == ["code", "graph", "build"]
    assert tokenize("read_lines(path, start)") == ["read", "lines", "path", "start"]


# -- search_symbols -----------------------------------------------------


def test_search_finds_a_symbol_by_name(sample_tools: GraphTools):
    result = sample_tools.search_symbols("slugify")
    ids = [m["symbol_id"] for m in result["matches"]]
    assert ids[0] == "pkg.util:slugify"
    assert result["total_matches"] >= 1


def test_search_matches_docstring_text(sample_tools: GraphTools):
    result = sample_tools.search_symbols("shouty version of the slug")
    assert "pkg.core:Widget.loud" in [m["symbol_id"] for m in result["matches"]]


def test_search_returns_signatures_and_never_bodies(sample_tools: GraphTools):
    payload = json.dumps(sample_tools.search_symbols("slugify widget build"))
    assert "def slugify(text: str) -> str" in payload
    for sentinel in (
        "SENTINEL_SLUGIFY_BODY",
        "SENTINEL_WIDGET_LOUD_BODY",
        "SENTINEL_BUILD_BODY",
    ):
        assert sentinel not in payload


def test_search_kind_filter_and_limit(sample_tools: GraphTools):
    only_classes = sample_tools.search_symbols("widget", kind="class")
    assert only_classes["matches"]
    assert {m["kind"] for m in only_classes["matches"]} == {"class"}

    limited = sample_tools.search_symbols("widget", limit=2)
    assert limited["returned"] <= 2
    assert limited["total_matches"] >= limited["returned"]


def test_search_on_nonsense_returns_empty_not_an_error(sample_tools: GraphTools):
    assert sample_tools.search_symbols("quixotic zebra apparatus")["matches"] == []


def test_search_ignores_question_words(sample_tools: GraphTools):
    """Stopwords must not rank arbitrary symbols above the real hit."""
    result = sample_tools.search_symbols("what does the slugify function return?")
    assert result["matches"][0]["symbol_id"] == "pkg.util:slugify"


# -- get_definition -----------------------------------------------------


def test_get_definition_returns_exactly_one_body(sample_tools: GraphTools):
    result = sample_tools.get_definition("pkg.core:Widget.slug")
    assert result["symbol_id"] == "pkg.core:Widget.slug"
    assert result["path"] == "pkg/core.py"
    assert "SENTINEL_WIDGET_SLUG_BODY" in result["source"]
    # exactly one node: the neighbouring method's body must not leak in
    assert "SENTINEL_WIDGET_LOUD_BODY" not in result["source"]
    assert result["source"].splitlines()[0].strip().startswith("def slug(")


def test_get_definition_of_a_module_returns_the_whole_file(sample_tools: GraphTools):
    result = sample_tools.get_definition("pkg.util")
    assert "SENTINEL_SLUGIFY_BODY" in result["source"]
    assert "RETRY_LIMIT = 7" in result["source"]


def test_get_definition_of_a_nested_function(sample_tools: GraphTools):
    result = sample_tools.get_definition("pkg.core:build._clean")
    assert "SENTINEL_CLEAN_BODY" in result["source"]
    assert "SENTINEL_BUILD_BODY" not in result["source"]


def test_get_definition_unknown_symbol_suggests_alternatives(sample_tools: GraphTools):
    result = sample_tools.get_definition("pkg.util:slugfy")
    assert "error" in result
    assert "pkg.util:slugify" in result["did_you_mean"]


# -- get_neighbors ------------------------------------------------------


def test_neighbors_callers_and_callees(sample_tools: GraphTools):
    callers = sample_tools.get_neighbors("pkg.util:slugify", direction="callers")
    assert "pkg.core:Widget.slug" in [c["symbol_id"] for c in callers["callers"]]

    callees = sample_tools.get_neighbors("pkg.core:Widget.loud", direction="callees")
    ids = {c["symbol_id"] for c in callees["callees"]}
    assert ids == {"pkg.core:Widget.slug", "pkg.util:_shout"}


def test_neighbors_imports_for_a_module(sample_tools: GraphTools):
    result = sample_tools.get_neighbors("pkg.core", direction="imports")
    assert "pkg.util" in [i["symbol_id"] for i in result["imports"]]

    back = sample_tools.get_neighbors("pkg.core", direction="imported_by")
    assert "app.main" in [i["symbol_id"] for i in back["imported_by"]]


def test_neighbors_children_enumerates_a_module(sample_tools: GraphTools):
    result = sample_tools.get_neighbors("pkg.core", direction="children")
    ids = {c["symbol_id"] for c in result["children"]}
    assert {"pkg.core:Widget", "pkg.core:Gadget", "pkg.core:build"} <= ids


def test_neighbors_returns_identifiers_only(sample_tools: GraphTools):
    payload = json.dumps(sample_tools.get_neighbors("pkg.core:Widget.loud"))
    assert "SENTINEL" not in payload
    assert "signature" in payload


def test_neighbors_disclaims_call_graph_completeness(sample_tools: GraphTools):
    note = sample_tools.get_neighbors("pkg.core:Widget.loud")["note"]
    assert "incomplete" in note and "getattr" in note


def test_neighbors_rejects_unknown_symbol_and_direction(sample_tools: GraphTools):
    assert "error" in sample_tools.get_neighbors("nope")
    assert "error" in sample_tools.get_neighbors("pkg.core", direction="sideways")


# -- read_lines ---------------------------------------------------------


def test_read_lines_returns_the_requested_range(sample_tools: GraphTools):
    result = sample_tools.read_lines("pkg/util.py", 1, 4)
    assert result["lines"] == "1-4"
    assert result["text"].splitlines()[0].startswith('"""String helpers')
    assert "SENTINEL_SLUGIFY_BODY" not in result["text"]


def test_read_lines_clamps_to_the_file(sample_tools: GraphTools):
    result = sample_tools.read_lines("pkg/util.py", 1, 10_000)
    assert int(result["lines"].split("-")[1]) == result["file_lines"]
    assert result["truncated"] is False


def test_read_lines_truncates_oversized_requests(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("\n".join(f"x{i} = {i}" for i in range(MAX_READ_LINES + 200)) + "\n")
    tools = GraphTools(CodeIndex.build(str(tmp_path)).graph())
    result = tools.read_lines("big.py", 1, MAX_READ_LINES + 200)
    assert result["truncated"] is True
    assert len(result["text"].splitlines()) == MAX_READ_LINES


def test_read_lines_refuses_to_escape_the_root(sample_tools: GraphTools):
    assert "error" in sample_tools.read_lines("../../../etc/passwd", 1, 5)
    assert "error" in sample_tools.read_lines("pkg/does_not_exist.py", 1, 5)


# -- dispatch -----------------------------------------------------------


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("search_symbols", {"query": "slugify"}),
        ("get_definition", {"symbol_id": "pkg.util:slugify"}),
        ("get_neighbors", {"symbol_id": "pkg.util:slugify"}),
        ("read_lines", {"path": "pkg/util.py", "start": 1, "end": 3}),
    ],
)
def test_dispatch_routes_every_tool(sample_tools: GraphTools, name, arguments):
    result = sample_tools.call(name, arguments)
    assert "error" not in result
    assert json.loads(sample_tools.call_as_text(name, arguments))


def test_dispatch_rejects_an_unknown_tool(sample_tools: GraphTools):
    result = sample_tools.call("rm_rf", {})
    assert "error" in result and result["available"] == list(TOOL_NAMES)
