"""The renderer must produce something for any recorded run."""

from __future__ import annotations

import pytest

from loop.agent import LoopConfig, run_loop
from loop.render import KIND_COLORS, KIND_LABELS, KIND_ORDER, render_html, render_text
from loop.schema import SEGMENT_KINDS
from loop.sink import MemorySink
from loop.testing import (
    FakeAnthropicClient,
    FakeMessage,
    echo_executor,
    heuristic_token_count,
    text_block,
    thinking_block,
    tool_use_block,
)

TOOLS = [
    {
        "name": "lookup",
        "description": "Look something up.",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]


@pytest.fixture
def records():
    sink = MemorySink()
    run_loop(
        client=FakeAnthropicClient(
            [
                FakeMessage(
                    content=[
                        thinking_block("thinking about it"),
                        text_block("Looking up."),
                        tool_use_block("lookup", {"q": "7781"}, "toolu_1"),
                    ],
                    stop_reason="tool_use",
                ),
                FakeMessage(
                    content=[tool_use_block("lookup", {"q": "7782"}, "toolu_2")],
                    stop_reason="tool_use",
                ),
                FakeMessage(content=[text_block("41250")], stop_reason="end_turn"),
            ]
        ),
        tools=TOOLS,
        executor=echo_executor,
        prompt="What is on invoice 7781?",
        config=LoopConfig(system="You are a ledger assistant. " * 25),
        sink=sink,
        counter=heuristic_token_count,
    )
    return sink.records


# --------------------------------------------------------------------------
# Palette bookkeeping
# --------------------------------------------------------------------------


def test_every_schema_kind_has_a_colour_a_label_and_a_stack_position():
    for kind in SEGMENT_KINDS:
        assert kind in KIND_COLORS, kind
        assert kind in KIND_LABELS, kind
        assert kind in KIND_ORDER, kind


def test_categorical_hues_are_assigned_by_identity_and_never_repeated():
    """Colour follows the segment kind, not its rank in a given run."""
    light = [KIND_COLORS[k][0] for k in KIND_ORDER if k not in ("other", "messages_total")]
    assert len(set(light)) == len(light)


# --------------------------------------------------------------------------
# Plain text
# --------------------------------------------------------------------------


def test_render_text_produces_a_row_per_turn(records):
    out = render_text(records)
    assert "run " in out
    for index in range(3):
        assert f"t{index:>2} |" in out or f"t {index}" in out
    assert "legend:" in out
    assert "cumulative share" in out
    assert "worst reconciliation residual" in out


def test_render_text_shows_growth_over_turns_not_just_a_total(records):
    out = render_text(records)
    bars = [line for line in out.splitlines() if "|" in line and line.strip().startswith("t")]
    assert len(bars) == 3
    filled = [len(line) - line.count(" ") for line in bars]
    assert filled[-1] > filled[0], "later turns should have fuller bars"


def test_render_text_handles_a_run_with_no_turns():
    sink = MemorySink()
    run_loop(
        client=FakeAnthropicClient([FakeMessage(content=[text_block("x")])]),
        tools=[],
        executor=echo_executor,
        prompt="hi",
        config=LoopConfig(max_iterations=0),
        sink=sink,
        counter=heuristic_token_count,
    )
    out = render_text(sink.records)
    assert "no turns recorded" in out


def test_render_text_of_an_empty_stream():
    assert render_text([]) == "no runs"


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def test_render_html_is_a_complete_self_contained_document(records):
    html = render_html(records, title="demo run")
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<svg" in html and "</svg>" in html
    assert "<style>" in html
    # Self-contained: no external hosts, no remote assets, no script tags.
    for forbidden in ("http://", "https://", "<script", "src=", "@import"):
        assert forbidden not in html, forbidden


def test_render_html_draws_one_bar_stack_per_turn(records):
    html = render_html(records)
    turns = [r for r in records if r["record_type"] == "turn"]
    # Each stacked segment is a <rect> with a hover <title>.
    assert html.count("<rect") >= len(turns)
    for turn in turns:
        assert f"turn {turn['turn_index']} &middot;" in html or (
            f"turn {turn['turn_index']} ·" in html
        )


def test_render_html_carries_identity_beyond_colour(records):
    """Legend plus a table view, so colour is never the only channel."""
    html = render_html(records)
    assert 'class="legend"' in html
    assert "<table>" in html
    for label in ("system prompt", "tool schemas", "tool_result blocks"):
        assert label in html


def test_render_html_supports_both_themes(records):
    html = render_html(records)
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    assert '[data-theme="light"]' in html


def test_render_html_reports_the_reconciliation_residual_per_turn(records):
    html = render_html(records)
    assert "residual" in html
    assert "counted total" in html
    assert "usage total" in html
    turns = [r for r in records if r["record_type"] == "turn"]
    for turn in turns:
        assert f"{turn['prompt_tokens']['counted_total']:,}" in html


def test_render_html_of_an_empty_stream_still_renders():
    html = render_html([])
    assert "no runs in this record stream" in html
    assert html.startswith("<!doctype html>")


def test_render_html_renders_several_runs(records):
    import copy

    second = copy.deepcopy(records)
    for record in second:
        record["run_id"] = "run_two"
    html = render_html(records + second)
    assert html.count("<h2>run ") == 2


def test_render_html_can_embed_an_accuracy_report(records):
    from loop.accuracy import Observation, analyze_accuracy

    report = analyze_accuracy(
        [
            Observation(f"r{i}", "t", 500 + i * 500, success=i < 4)
            for i in range(8)
        ],
        n_bins=4,
    )
    html = render_html(records, accuracy=report)
    assert "accuracy vs. context length" in html
    assert "caveats" in html


def test_html_escapes_untrusted_text(records):
    import copy

    hostile = copy.deepcopy(records)
    hostile[0]["run_id"] = "<img onerror=x>"
    for record in hostile:
        record["run_id"] = "<img onerror=x>"
    html = render_html(hostile)
    assert "<img onerror" not in html
    assert "&lt;img onerror=x&gt;" in html
