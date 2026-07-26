"""Graph construction over the synthetic fixture repo."""

from __future__ import annotations

from graph.builder import NODE_KINDS, CodeGraph, analyze_source, module_name_for


# -- module naming ------------------------------------------------------


def test_module_name_for_handles_packages_and_init():
    assert module_name_for("common/client.py") == "common.client"
    assert module_name_for("pkg/__init__.py") == "pkg"
    assert module_name_for("main.py") == "main"


# -- node extraction ----------------------------------------------------


def test_every_expected_node_exists(sample_graph: CodeGraph):
    expected = {
        "pkg": "module",
        "pkg.util": "module",
        "pkg.core": "module",
        "app.main": "module",
        "pkg.util:slugify": "function",
        "pkg.util:_shout": "function",
        "pkg.core:Widget": "class",
        "pkg.core:Widget.__init__": "method",
        "pkg.core:Widget.slug": "method",
        "pkg.core:Widget.loud": "method",
        "pkg.core:Gadget": "class",
        "pkg.core:Gadget.describe": "method",
        "pkg.core:build": "function",
        "pkg.core:build._clean": "function",  # nested function
        "app.main:main": "function",
    }
    for symbol_id, kind in expected.items():
        node = sample_graph.node(symbol_id)
        assert node is not None, f"missing node {symbol_id}"
        assert node.kind == kind, f"{symbol_id} is {node.kind}, expected {kind}"
    assert set(NODE_KINDS) >= {n.kind for n in sample_graph.nodes.values()}


def test_nested_function_is_contained_by_its_parent(sample_graph: CodeGraph):
    nested = sample_graph.node("pkg.core:build._clean")
    assert nested.parent == "pkg.core:build"
    assert "pkg.core:build._clean" in sample_graph.children["pkg.core:build"]
    assert nested.qualname == "build._clean"


def test_containment_chain_module_class_method(sample_graph: CodeGraph):
    method = sample_graph.node("pkg.core:Widget.slug")
    assert method.parent == "pkg.core:Widget"
    assert sample_graph.node("pkg.core:Widget").parent == "pkg.core"
    assert "pkg.core:Widget" in sample_graph.children["pkg.core"]


def test_signature_and_docline_are_captured(sample_graph: CodeGraph):
    node = sample_graph.node("pkg.util:slugify")
    assert node.signature == "def slugify(text: str) -> str"
    assert node.doc == "Lowercase a string and join its words with the separator."
    widget = sample_graph.node("pkg.core:Gadget")
    assert widget.signature == "class Gadget(Widget)"


def test_line_spans_bracket_the_definition(sample_graph: CodeGraph):
    node = sample_graph.node("pkg.core:build")
    source = sample_graph.source_of(node)
    assert source.startswith("def build(")
    assert "SENTINEL_BUILD_BODY" in source
    assert node.end_lineno > node.lineno


# -- import edges -------------------------------------------------------


def test_import_edges_resolve_within_the_tree(sample_graph: CodeGraph):
    assert "pkg.util" in sample_graph.imports["pkg.core"]
    assert "pkg.core" in sample_graph.imports["app.main"]
    assert "pkg.core" in sample_graph.imported_by["pkg.util"]
    assert "app.main" in sample_graph.imported_by["pkg.core"]


def test_external_imports_are_recorded_not_invented(sample_graph: CodeGraph):
    external = sample_graph.external_imports["pkg.dynamic"]
    assert "importlib" in external
    assert "importlib" not in sample_graph.nodes
    assert "importlib" not in sample_graph.imports.get("pkg.dynamic", [])


def test_relative_imports_resolve_against_the_package():
    source = "from . import util\nfrom .util import slugify\n"
    _, imports, _, error = analyze_source(
        source, rel_path="pkg/core.py", module="pkg.core"
    )
    assert error is None
    assert {(i.module, i.name) for i in imports} == {
        ("pkg", "util"),
        ("pkg.util", "slugify"),
    }


# -- call edges ---------------------------------------------------------


def test_call_edge_through_a_from_import(sample_graph: CodeGraph):
    assert "pkg.util:slugify" in sample_graph.callees["pkg.core:Widget.slug"]
    assert "pkg.core:Widget.slug" in sample_graph.callers["pkg.util:slugify"]


def test_call_edge_through_self(sample_graph: CodeGraph):
    assert "pkg.core:Widget.slug" in sample_graph.callees["pkg.core:Widget.loud"]


def test_call_edge_through_a_module_alias(sample_graph: CodeGraph):
    assert "pkg.util:_shout" in sample_graph.callees["pkg.core:Widget.loud"]


def test_call_edge_to_a_nested_function_from_its_parent(sample_graph: CodeGraph):
    assert "pkg.core:build._clean" in sample_graph.callees["pkg.core:build"]


def test_call_edge_across_modules(sample_graph: CodeGraph):
    assert "pkg.core:build" in sample_graph.callees["app.main:main"]
    assert "app.main:main" in sample_graph.callers["pkg.core:build"]


def test_constructor_call_resolves_to_the_class(sample_graph: CodeGraph):
    assert "pkg.core:Widget" in sample_graph.callees["app.main:make_blank"]


# -- documented blind spots --------------------------------------------


def test_inherited_method_call_is_not_resolved(sample_graph: CodeGraph):
    """Gadget.describe calls self.slug(), inherited from Widget. We do not
    follow base classes, so no edge exists -- and we must not invent one."""
    assert "pkg.core:Gadget.describe" not in sample_graph.callees


def test_dynamic_and_untyped_calls_are_unresolved(sample_graph: CodeGraph):
    """Every blind spot the fixture documents really is a blind spot."""
    for caller in (
        "pkg.dynamic:load_and_run",  # importlib.import_module(name).run()
        "pkg.dynamic:call_by_name",  # getattr(obj, name)()
        "pkg.dynamic:call_untyped",  # thing.slug() on an untyped value
    ):
        assert caller not in sample_graph.callees, f"{caller} should be unresolved"
    # ...and they are counted, so the size of the blind spot stays visible.
    assert sample_graph.stats()["unresolved_calls"] > 0


def test_stats_are_self_consistent(sample_graph: CodeGraph):
    stats = sample_graph.stats()
    assert stats["modules"] == len(
        [n for n in sample_graph.nodes.values() if n.kind == "module"]
    )
    assert stats["call_edges"] == sum(len(v) for v in sample_graph.callees.values())
    assert stats["files"] == len(sample_graph.files)


def test_syntax_error_is_reported_not_raised():
    nodes, imports, calls, error = analyze_source(
        "def broken(:\n", rel_path="bad.py", module="bad"
    )
    assert error and "SyntaxError" in error
    assert nodes == [] and imports == [] and calls == []
