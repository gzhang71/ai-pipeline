"""Automatic breakpoint placement, its limits, and its silent-failure guards."""

from __future__ import annotations

from common.client import CACHE_LOOKBACK_BLOCKS, MAX_CACHE_BREAKPOINTS

from prompt import PromptAssembler, Stability
from prompt.fakes import FakeClient

BIG = "x" * 600  # with token_counter=len, comfortably over every model minimum


def _sized() -> PromptAssembler:
    """Assembler whose 'tokens' are characters, so sizes are exact in tests."""
    return PromptAssembler(model="claude-opus-5", token_counter=len)


def test_breakpoints_land_on_stability_boundaries():
    a = _sized()
    a.add_tool("search", BIG, {"type": "object", "properties": {}})
    a.add_system(BIG, Stability.STATIC, label="persona")
    a.add_system(BIG, Stability.SESSION, label="account")
    a.add_message("user", BIG, Stability.TURN, label="question")

    plan = a.plan()
    assert [bp.index for bp in plan.breakpoints] == [1, 2]
    assert [bp.label for bp in plan.breakpoints] == ["persona", "account"]
    # Each breakpoint protects everything up to and including its own block.
    assert plan.breakpoints[0].prefix_tokens < plan.breakpoints[1].prefix_tokens
    assert plan.breakpoints[1].prefix_tokens < plan.total_estimated_tokens


def test_markers_are_emitted_on_the_planned_blocks_only():
    a = _sized()
    a.add_system(BIG, Stability.STATIC)
    a.add_system(BIG, Stability.SESSION)
    a.add_message("user", BIG, Stability.TURN)

    kwargs = a.to_request_kwargs(max_tokens=1024)
    # Both stability boundaries are marked: STATIC|SESSION and SESSION|TURN.
    # Marking the STATIC boundary is the point of having more than one
    # breakpoint -- it keeps the static prefix a cache read even on the
    # requests where the session block changes underneath it.
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert all(
        "cache_control" not in block
        for message in kwargs["messages"]
        for block in message["content"]
    )


def test_ttl_1h_variant():
    a = PromptAssembler(model="claude-opus-5", ttl="1h", token_counter=len)
    a.add_system(BIG, Stability.STATIC)
    a.add_message("user", BIG, Stability.TURN)

    kwargs = a.to_request_kwargs()
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    default_ttl = _sized()
    default_ttl.add_system(BIG, Stability.STATIC)
    default_ttl.add_message("user", BIG, Stability.TURN)
    # The 5-minute default is expressed by omitting the key, not by sending it.
    assert default_ttl.to_request_kwargs()["system"][0]["cache_control"] == {
        "type": "ephemeral"
    }


def test_breakpoint_count_never_exceeds_four():
    a = _sized()
    for i in range(8):
        a.add_message(
            "user" if i % 2 == 0 else "assistant",
            BIG,
            Stability.SESSION,
            label=f"turn{i}",
            checkpoint=True,
        )
    a.add_message("user", BIG, Stability.TURN, label="new")

    plan = a.plan()
    assert plan.candidates_considered == 8
    assert len(plan.breakpoints) == MAX_CACHE_BREAKPOINTS
    # The survivors are the four protecting the most tokens.
    assert [bp.index for bp in plan.breakpoints] == [4, 5, 6, 7]
    assert any("only 4 breakpoints allowed" in w for w in plan.warnings)

    kwargs = a.to_request_kwargs()
    markers = sum(
        "cache_control" in block
        for message in kwargs["messages"]
        for block in message["content"]
    )
    assert markers == MAX_CACHE_BREAKPOINTS
    # A request carrying a fifth breakpoint would be rejected; this one is not.
    FakeClient().messages.create(**kwargs)


def test_max_breakpoints_can_be_lowered():
    a = PromptAssembler(model="claude-opus-5", token_counter=len, max_breakpoints=2)
    for i in range(5):
        a.add_message(
            "user" if i % 2 == 0 else "assistant",
            BIG,
            Stability.SESSION,
            checkpoint=True,
        )
    a.add_message("user", BIG, Stability.TURN)
    assert len(a.plan().breakpoints) == 2


def test_prefix_below_model_minimum_is_flagged_and_not_marked():
    """Below the minimum the API accepts the marker and caches nothing."""
    a = PromptAssembler(model="claude-opus-5")  # real ~4 chars/token estimate
    a.add_system("You are terse.", Stability.STATIC, label="persona")
    a.add_message("user", "hi", Stability.TURN)

    plan = a.plan()
    assert plan.minimum_prefix_tokens == 512
    assert plan.breakpoints == ()
    assert any("below the 512-token minimum" in w for w in plan.warnings)
    assert any("'persona'" in w for w in plan.warnings)

    kwargs = a.to_request_kwargs()
    assert "cache_control" not in kwargs["system"][0]


def test_minimum_is_model_dependent_and_not_monotonic():
    """The same prompt caches on one model and silently will not on another."""

    def build(model: str) -> PromptAssembler:
        a = PromptAssembler(model=model, token_counter=len)
        a.add_system("y" * 700, Stability.STATIC)
        a.add_message("user", BIG, Stability.TURN)
        return a

    # 700 'tokens': over opus-5's 512 minimum, under opus-4-6's 4096.
    assert len(build("claude-opus-5").plan().breakpoints) == 1
    assert build("claude-opus-4-6").plan().breakpoints == ()
    # Newer generation, lower minimum -- the non-monotonic case.
    assert build("claude-opus-4-6").plan().minimum_prefix_tokens > build(
        "claude-opus-5"
    ).plan().minimum_prefix_tokens


def test_twenty_block_lookback_window_is_flagged():
    a = _sized()
    a.add_system(BIG, Stability.STATIC, label="persona")
    for i in range(CACHE_LOOKBACK_BLOCKS + 5):
        a.add_message(
            "user" if i % 2 == 0 else "assistant", BIG, Stability.TURN, label=f"b{i}"
        )

    plan = a.plan()
    assert [bp.label for bp in plan.breakpoints] == ["persona"]
    lookback = [w for w in plan.warnings if "lookback window" in w]
    assert lookback, plan.warnings
    assert f"{CACHE_LOOKBACK_BLOCKS}-block" in lookback[0]
    assert "25 content blocks follow the last breakpoint" in lookback[0]


def test_short_agentic_turn_does_not_trip_the_lookback_warning():
    a = _sized()
    a.add_system(BIG, Stability.STATIC)
    for i in range(4):
        a.add_message("user" if i % 2 == 0 else "assistant", BIG, Stability.TURN)
    assert not any("lookback" in w for w in a.plan().warnings)


def test_uniform_stability_gets_no_breakpoints_but_says_why():
    a = _sized()
    a.add_message("user", BIG, Stability.TURN)
    a.add_message("assistant", BIG, Stability.TURN)

    plan = a.plan()
    assert plan.breakpoints == ()
    assert any("no breakpoint candidates" in w for w in plan.warnings)


def test_cache_tail_caches_the_whole_prompt():
    a = PromptAssembler(model="claude-opus-5", token_counter=len, cache_tail=True)
    a.add_message("user", BIG, Stability.TURN)
    a.add_message("assistant", BIG, Stability.TURN)

    plan = a.plan()
    assert [bp.index for bp in plan.breakpoints] == [1]
    assert plan.breakpoints[0].prefix_tokens == plan.total_estimated_tokens


def test_plan_describe_is_human_readable():
    a = _sized()
    a.add_system(BIG, Stability.STATIC, label="persona")
    a.add_message("user", BIG, Stability.TURN)
    text = a.plan().describe()
    assert "claude-opus-5" in text
    assert "persona" in text
    assert "ttl=5m" in text
