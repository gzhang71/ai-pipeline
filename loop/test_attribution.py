"""Attribution: does the decomposition actually add up, and to what?"""

from __future__ import annotations

import pytest

from common.client import has_credentials
from loop.attribution import (
    CachingTokenCounter,
    attribute,
    block_groups,
    block_kind,
    normalize_messages,
    reconcile,
    to_plain,
)
from loop.testing import (
    FakeAnthropicClient,
    heuristic_token_count,
    text_block,
    thinking_block,
    tool_use_block,
)

SYSTEM = "You are a careful assistant. " * 40
TOOLS = [
    {
        "name": "lookup",
        "description": "Look something up in the corpus.",
        "input_schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    }
]


def conversation():
    """A realistic multi-turn transcript covering every segment kind."""
    return [
        {"role": "user", "content": "Find the total for account 7781."},
        {
            "role": "assistant",
            "content": [
                thinking_block("The user wants an account total; I should look it up."),
                text_block("Looking that up."),
                tool_use_block("lookup", {"q": "account 7781"}, "toolu_1"),
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "account=7781 total=41250 currency=USD " * 30,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [text_block("The total is 41250 USD.")],
        },
        {"role": "user", "content": "And for 7782?"},
    ]


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def test_block_kind_depends_on_role():
    assert block_kind("user", {"type": "text"}) == "user_text"
    assert block_kind("assistant", {"type": "text"}) == "assistant_text"
    assert block_kind("assistant", {"type": "thinking"}) == "thinking"
    assert block_kind("assistant", {"type": "redacted_thinking"}) == "thinking"
    assert block_kind("assistant", {"type": "tool_use"}) == "tool_use"
    assert block_kind("user", {"type": "tool_result"}) == "tool_result"
    assert block_kind("user", {"type": "image"}) == "other"


def test_block_groups_split_by_kind_and_cover_every_block():
    plain = normalize_messages(conversation())
    groups = block_groups(plain)
    assert [g.kind for g in groups] == [
        "user_text",
        "thinking",
        "assistant_text",
        "tool_use",
        "tool_result",
        "assistant_text",
        "user_text",
    ]
    # Every block of every message belongs to exactly one group.
    for index, message in enumerate(plain):
        spans = [(g.start, g.end) for g in groups if g.message_index == index]
        covered = [i for start, end in spans for i in range(start, end)]
        assert sorted(covered) == list(range(len(message["content"])))


def test_message_granularity_makes_one_group_per_message():
    plain = normalize_messages(conversation())
    groups = block_groups(plain, granularity="message")
    assert len(groups) == len(plain)


# --------------------------------------------------------------------------
# The additivity property
# --------------------------------------------------------------------------


def test_naive_per_segment_counting_is_not_additive():
    """The premise of the whole module: counting parts separately is wrong.

    If this ever stops failing, the fake counter has become trivially additive
    and the additivity test below stops proving anything.
    """
    plain = normalize_messages(conversation())
    whole = heuristic_token_count(messages=plain, system=SYSTEM, tools=TOOLS)
    parts = sum(
        heuristic_token_count(messages=[m], system=None, tools=None) for m in plain
    )
    parts += heuristic_token_count(messages=[{"role": "user", "content": "."}], system=SYSTEM)
    parts += heuristic_token_count(messages=[{"role": "user", "content": "."}], tools=TOOLS)
    assert parts != whole


@pytest.mark.parametrize("granularity", ["block_group", "message", "coarse"])
def test_segments_sum_exactly_to_the_counted_total(granularity):
    """Prefix deltas telescope, so the decomposition is exactly additive."""
    result = attribute(
        conversation(),
        system=SYSTEM,
        tools=TOOLS,
        counter=heuristic_token_count,
        granularity=granularity,
    )
    assert result.segments
    assert result.segment_sum == result.counted_total
    assert result.decomposition_residual == 0
    assert sum(result.by_kind().values()) == result.counted_total


def test_counted_total_equals_a_single_count_of_the_whole_prompt():
    result = attribute(
        conversation(), system=SYSTEM, tools=TOOLS, counter=heuristic_token_count
    )
    direct = heuristic_token_count(
        messages=normalize_messages(conversation()), system=SYSTEM, tools=TOOLS
    )
    assert result.counted_total == direct


def test_every_kind_present_in_the_transcript_gets_attributed():
    result = attribute(
        conversation(), system=SYSTEM, tools=TOOLS, counter=heuristic_token_count
    )
    by_kind = result.by_kind()
    for kind in (
        "framing",
        "tool_schemas",
        "system_prompt",
        "user_text",
        "assistant_text",
        "thinking",
        "tool_use",
        "tool_result",
    ):
        assert kind in by_kind, kind
    # The bulky tool_result should dominate the message-side segments.
    assert by_kind["tool_result"] > by_kind["user_text"]
    assert by_kind["system_prompt"] > 0
    assert by_kind["tool_schemas"] > 0


def test_framing_is_flagged_approximate_and_nothing_else_is():
    result = attribute(
        conversation(), system=SYSTEM, tools=TOOLS, counter=heuristic_token_count
    )
    approximate = [s.segment_id for s in result.segments if s.approximate]
    assert approximate == ["framing"]
    assert result.approximate_segments == ["framing"]


def test_segments_are_reported_in_render_order():
    result = attribute(
        conversation(), system=SYSTEM, tools=TOOLS, counter=heuristic_token_count
    )
    kinds = [s.kind for s in result.segments]
    assert kinds[:3] == ["framing", "tool_schemas", "system_prompt"]


def test_no_tools_and_no_system_yield_zero_segments_not_missing_ones():
    result = attribute(
        [{"role": "user", "content": "hello"}], counter=heuristic_token_count
    )
    by_kind = result.by_kind()
    assert by_kind["tool_schemas"] == 0
    assert by_kind["system_prompt"] == 0
    assert result.segment_sum == result.counted_total


def test_off_granularity_reports_a_total_and_no_segments():
    result = attribute(
        conversation(),
        system=SYSTEM,
        tools=TOOLS,
        counter=heuristic_token_count,
        granularity="off",
    )
    assert result.segments == []
    assert result.counted_total > 0
    assert result.decomposition_residual == 0
    assert result.counter_calls == 1


def test_empty_message_list_is_rejected():
    with pytest.raises(ValueError):
        attribute([], counter=heuristic_token_count)


def test_unknown_granularity_is_rejected():
    with pytest.raises(ValueError):
        attribute(
            [{"role": "user", "content": "x"}],
            counter=heuristic_token_count,
            granularity="nonsense",
        )


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_caching_counter_reuses_prefixes_across_turns():
    """A shared cache should make turn 2 cost only its *new* prefixes."""
    cache = CachingTokenCounter(heuristic_token_count)
    first = conversation()[:3]
    attribute(first, system=SYSTEM, tools=TOOLS, counter=cache, granularity="block_group")
    calls_after_first = cache.calls

    second = conversation()
    result = attribute(
        second, system=SYSTEM, tools=TOOLS, counter=cache, granularity="block_group"
    )
    incremental = cache.calls - calls_after_first
    assert incremental < calls_after_first
    # Only the genuinely new prefixes should have been counted.
    assert result.counter_calls == incremental


def test_caching_counter_returns_the_same_answers_as_the_inner_counter():
    cache = CachingTokenCounter(heuristic_token_count)
    plain = normalize_messages(conversation())
    direct = heuristic_token_count(messages=plain, system=SYSTEM, tools=TOOLS)
    assert cache(messages=plain, system=SYSTEM, tools=TOOLS) == direct
    assert cache(messages=plain, system=SYSTEM, tools=TOOLS) == direct
    assert cache.calls == 1
    assert cache.lookups == 2


# --------------------------------------------------------------------------
# Reconciliation against authoritative usage
# --------------------------------------------------------------------------


def test_reconcile_uses_the_full_prompt_not_just_input_tokens():
    """`input_tokens` is the uncached remainder; the prompt is the sum."""
    result = attribute(
        conversation(), system=SYSTEM, tools=TOOLS, counter=heuristic_token_count
    )
    usage = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": result.counted_total - 10,
        "output_tokens": 5,
        "total_prompt_tokens": result.counted_total,
    }
    recon = reconcile(result, usage)
    assert recon["authoritative_total"] == result.counted_total
    assert recon["residual_tokens"] == 0
    assert recon["within_tolerance"] is True


def test_reconcile_reports_a_residual_rather_than_hiding_it():
    result = attribute(
        conversation(), system=SYSTEM, tools=TOOLS, counter=heuristic_token_count
    )
    usage = {
        "input_tokens": result.counted_total + 400,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
        "total_prompt_tokens": result.counted_total + 400,
    }
    recon = reconcile(result, usage, tolerance_fraction=0.02)
    assert recon["residual_tokens"] == -400
    assert recon["residual_fraction"] < 0
    assert recon["within_tolerance"] is False


def test_to_plain_flattens_sdk_style_blocks():
    blocks = [text_block("hi"), tool_use_block("t", {"a": 1}, "toolu_9")]
    plain = to_plain(blocks)
    assert plain == [
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "id": "toolu_9", "name": "t", "input": {"a": 1}},
    ]


# --------------------------------------------------------------------------
# The live path stays guarded
# --------------------------------------------------------------------------


@pytest.mark.skipif(has_credentials(), reason="only meaningful without credentials")
def test_default_counter_is_the_api_and_is_not_reached_offline():
    """Nothing in the default test path may touch the network."""
    client = FakeAnthropicClient([type("R", (), {"content": [text_block("x")], "stop_reason": "end_turn"})()])
    assert client.count_calls == 0
