"""The loop: does it terminate where it should, and bound itself where it must?"""

from __future__ import annotations

import pytest

from loop.agent import LoopConfig, ToolResult, run_loop
from loop.schema import validate_run
from loop.sink import MemorySink
from loop.testing import (
    FakeAnthropicClient,
    FakeMessage,
    echo_executor,
    text_block,
    thinking_block,
    tool_use_block,
)

TOOLS = [
    {
        "name": "lookup",
        "description": "Look something up.",
        "input_schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    }
]
SYSTEM = "You are a ledger assistant. " * 30


def _run(script, *, executor=echo_executor, config=None, **kwargs):
    sink = MemorySink()
    client = FakeAnthropicClient(script)
    result = run_loop(
        client=client,
        tools=TOOLS,
        executor=executor,
        prompt="What is on invoice 7781?",
        config=config or LoopConfig(system=SYSTEM, max_iterations=6),
        sink=sink,
        counter=_counter(),
        **kwargs,
    )
    return client, sink, result


def _counter():
    from loop.testing import heuristic_token_count

    return heuristic_token_count


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------


def test_terminates_immediately_on_end_turn():
    client, sink, result = _run(
        [FakeMessage(content=[text_block("41250")], stop_reason="end_turn")]
    )
    assert result.turns == 1
    assert result.stop_reason == "end_turn"
    assert result.final_text == "41250"
    assert len(client.requests) == 1
    assert [r["record_type"] for r in sink.records] == [
        "run_header",
        "turn",
        "run_footer",
    ]


def test_runs_tool_use_iterations_then_stops_on_end_turn():
    script = [
        FakeMessage(
            content=[
                thinking_block("I need the ledger."),
                text_block("Looking it up."),
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
    client, sink, result = _run(script)

    assert result.turns == 3
    assert result.stop_reason == "end_turn"
    assert len(client.requests) == 3

    turns = [r for r in sink.records if r["record_type"] == "turn"]
    assert [t["response"]["n_tool_use"] for t in turns] == [1, 1, 0]
    assert [len(t["tool_calls"]) for t in turns] == [1, 1, 0]

    # The transcript grows the way a real tool loop grows.
    roles = [m["role"] for m in result.messages]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]

    # And the prompt grows monotonically with it.
    totals = [t["prompt_tokens"]["counted_total"] for t in turns]
    assert totals == sorted(totals)
    assert totals[-1] > totals[0]


def test_tool_results_are_fed_back_with_matching_ids():
    script = [
        FakeMessage(
            content=[tool_use_block("lookup", {"q": "7781"}, "toolu_abc")],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[text_block("done")], stop_reason="end_turn"),
    ]
    client, _sink, _result = _run(script)
    second_request = client.requests[1]
    tool_result_message = second_request["messages"][-1]
    assert tool_result_message["role"] == "user"
    block = tool_result_message["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_abc"


def test_parallel_tool_calls_return_in_one_user_message():
    script = [
        FakeMessage(
            content=[
                tool_use_block("lookup", {"q": "a"}, "toolu_1"),
                tool_use_block("lookup", {"q": "b"}, "toolu_2"),
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[text_block("done")], stop_reason="end_turn"),
    ]
    client, sink, result = _run(script)
    follow_up = client.requests[1]["messages"][-1]
    assert follow_up["role"] == "user"
    assert [b["tool_use_id"] for b in follow_up["content"]] == ["toolu_1", "toolu_2"]
    turns = [r for r in sink.records if r["record_type"] == "turn"]
    assert turns[0]["response"]["n_tool_use"] == 2


@pytest.mark.parametrize("stop", ["max_tokens", "stop_sequence", "refusal"])
def test_other_terminal_stop_reasons_end_the_run(stop):
    _client, _sink, result = _run(
        [FakeMessage(content=[text_block("partial")], stop_reason=stop)]
    )
    assert result.turns == 1
    assert result.stop_reason == stop


def test_pause_turn_resends_without_injecting_a_message():
    script = [
        FakeMessage(content=[text_block("working")], stop_reason="pause_turn"),
        FakeMessage(content=[text_block("done")], stop_reason="end_turn"),
    ]
    client, _sink, result = _run(script)
    assert result.turns == 2
    assert result.stop_reason == "end_turn"
    # The resend carries the paused assistant turn and nothing fabricated.
    assert client.requests[1]["messages"][-1]["role"] == "assistant"


# --------------------------------------------------------------------------
# The runaway guard
# --------------------------------------------------------------------------


def test_max_iterations_bounds_a_loop_that_would_never_stop():
    """The script repeats its last entry forever; only the guard ends this."""
    forever = [
        FakeMessage(
            content=[tool_use_block("lookup", {"q": "again"}, "toolu_x")],
            stop_reason="tool_use",
        )
    ]
    client, sink, result = _run(
        forever, config=LoopConfig(system=SYSTEM, max_iterations=4)
    )
    assert result.turns == 4
    assert result.stop_reason == "max_iterations"
    assert len(client.requests) == 4
    turns = [r for r in sink.records if r["record_type"] == "turn"]
    assert [t["turn_index"] for t in turns] == [0, 1, 2, 3]
    assert sink.records[-1]["turns"] == 4


def test_max_iterations_of_zero_makes_no_requests_at_all():
    client, sink, result = _run(
        [FakeMessage(content=[text_block("hi")])],
        config=LoopConfig(system=SYSTEM, max_iterations=0),
    )
    assert client.requests == []
    assert result.turns == 0
    assert result.stop_reason == "max_iterations"
    assert [r["record_type"] for r in sink.records] == ["run_header", "run_footer"]


# --------------------------------------------------------------------------
# Tool execution
# --------------------------------------------------------------------------


def test_tool_exceptions_become_error_results_rather_than_killing_the_run():
    def explodes(name, tool_input, tool_use_id):
        raise RuntimeError("ledger offline")

    script = [
        FakeMessage(
            content=[tool_use_block("lookup", {"q": "x"}, "toolu_1")],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[text_block("sorry")], stop_reason="end_turn"),
    ]
    client, sink, result = _run(script, executor=explodes)
    assert result.stop_reason == "end_turn"
    turn = [r for r in sink.records if r["record_type"] == "turn"][0]
    assert turn["tool_calls"][0]["is_error"] is True
    block = client.requests[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "ledger offline" in block["content"]


def test_catch_tool_errors_false_records_the_failure_on_the_footer():
    def explodes(name, tool_input, tool_use_id):
        raise RuntimeError("boom")

    script = [
        FakeMessage(
            content=[tool_use_block("lookup", {"q": "x"}, "toolu_1")],
            stop_reason="tool_use",
        )
    ]
    _client, sink, result = _run(
        script,
        executor=explodes,
        config=LoopConfig(system=SYSTEM, catch_tool_errors=False),
    )
    assert result.stop_reason == "error"
    assert "boom" in (result.error or "")
    assert sink.records[-1]["error"] is not None
    validate_run(sink.records)


def test_executor_may_return_an_explicit_tool_result():
    def explicit(name, tool_input, tool_use_id):
        return ToolResult(content="not found", is_error=True)

    script = [
        FakeMessage(
            content=[tool_use_block("lookup", {"q": "x"}, "toolu_1")],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[text_block("ok")], stop_reason="end_turn"),
    ]
    _client, sink, _result = _run(script, executor=explicit)
    turn = [r for r in sink.records if r["record_type"] == "turn"][0]
    assert turn["tool_calls"][0]["is_error"] is True


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------


def test_request_carries_system_tools_and_optional_knobs():
    config = LoopConfig(
        system=SYSTEM,
        effort="medium",
        thinking={"type": "adaptive", "display": "summarized"},
        max_tokens=4096,
    )
    client, _sink, _result = _run(
        [FakeMessage(content=[text_block("ok")])], config=config
    )
    request = client.requests[0]
    assert request["system"] == SYSTEM
    assert request["tools"] == TOOLS
    assert request["max_tokens"] == 4096
    assert request["output_config"] == {"effort": "medium"}
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    # Sampling parameters are rejected by current models; we never send them.
    assert "temperature" not in request
    assert "top_p" not in request


def test_omitted_knobs_are_absent_rather_than_null():
    client, _sink, _result = _run([FakeMessage(content=[text_block("ok")])])
    request = client.requests[0]
    assert "output_config" not in request
    assert "thinking" not in request


def test_assistant_content_is_echoed_back_verbatim():
    """Thinking and tool_use blocks must survive the round trip unedited."""
    script = [
        FakeMessage(
            content=[
                thinking_block("reasoning"),
                tool_use_block("lookup", {"q": "x"}, "toolu_1"),
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[text_block("ok")], stop_reason="end_turn"),
    ]
    client, _sink, _result = _run(script)
    echoed = client.requests[1]["messages"][1]
    assert echoed["role"] == "assistant"
    assert [b["type"] for b in echoed["content"]] == ["thinking", "tool_use"]
    assert echoed["content"][0]["thinking"] == "reasoning"


def test_prompt_or_messages_is_required():
    with pytest.raises(ValueError):
        run_loop(
            client=FakeAnthropicClient([FakeMessage(content=[text_block("x")])]),
            tools=[],
            executor=echo_executor,
        )


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------


def test_every_turn_reconciles_within_tolerance_against_usage():
    """The fake client skews usage slightly, as the live API does. The
    decomposition must still land inside the stated tolerance."""
    script = [
        FakeMessage(
            content=[tool_use_block("lookup", {"q": "x"}, "toolu_1")],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[text_block("41250")], stop_reason="end_turn"),
    ]
    _client, sink, _result = _run(script)
    turns = [r for r in sink.records if r["record_type"] == "turn"]
    assert turns
    for turn in turns:
        recon = turn["reconciliation"]
        # The decomposition itself is exactly additive...
        assert turn["prompt_tokens"]["segment_sum"] == turn["prompt_tokens"]["counted_total"]
        assert turn["prompt_tokens"]["decomposition_residual"] == 0
        # ...and reconciles against authoritative usage within tolerance.
        assert recon["within_tolerance"] is True, recon
        assert abs(recon["residual_fraction"]) <= recon["tolerance_fraction"]
        assert recon["authoritative_total"] == turn["usage"]["total_prompt_tokens"]


def test_usage_total_is_the_sum_not_the_uncached_remainder():
    client = FakeAnthropicClient(
        [FakeMessage(content=[text_block("ok")])], cache_read_fraction=0.9
    )
    sink = MemorySink()
    run_loop(
        client=client,
        tools=TOOLS,
        executor=echo_executor,
        prompt="hello",
        config=LoopConfig(system=SYSTEM),
        sink=sink,
        counter=_counter(),
    )
    turn = [r for r in sink.records if r["record_type"] == "turn"][0]
    usage = turn["usage"]
    assert usage["cache_read_input_tokens"] > usage["input_tokens"]
    assert (
        usage["total_prompt_tokens"]
        == usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    # Reading input_tokens alone would badly understate the prompt.
    assert turn["reconciliation"]["within_tolerance"] is True


def test_footer_totals_agree_with_the_turn_records():
    script = [
        FakeMessage(
            content=[tool_use_block("lookup", {"q": "x"}, "toolu_1")],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[text_block("done")], stop_reason="end_turn"),
    ]
    _client, sink, _result = _run(script)
    turns = [r for r in sink.records if r["record_type"] == "turn"]
    footer = sink.records[-1]
    assert footer["totals"]["prompt_tokens_total"] == sum(
        t["prompt_tokens"]["counted_total"] for t in turns
    )
    assert footer["totals"]["peak_prompt_tokens"] == max(
        t["prompt_tokens"]["counted_total"] for t in turns
    )
    assert footer["totals"]["output_tokens_total"] == sum(
        t["usage"]["output_tokens"] for t in turns
    )
    by_kind = footer["totals"]["by_kind_total"]
    assert by_kind["system_prompt"] > 0
    assert sum(by_kind.values()) == footer["totals"]["prompt_tokens_total"]


def test_header_describes_the_attribution_method_it_used():
    _client, sink, _result = _run([FakeMessage(content=[text_block("ok")])])
    header = sink.records[0]
    assert header["attribution"]["method"] == "incremental_prefix_delta"
    assert header["attribution"]["granularity"] == "block_group"
    assert header["attribution"]["approximate_segments"] == ["framing"]
    assert header["attribution"]["measurement_order"] == [
        "framing",
        "messages",
        "tool_schemas",
        "system_prompt",
    ]
    assert header["tool_names"] == ["lookup"]
    assert header["system_fingerprint"]["present"] is True
