"""Reading a response `usage` and naming the likely cause of a miss."""

from __future__ import annotations

from common.client import CACHE_LOOKBACK_BLOCKS

from prompt import (
    CacheStatus,
    MissReason,
    PromptAssembler,
    Stability,
    diagnose_usage,
    render_prefix,
)
from prompt.fakes import FakeCacheServer, FakeUsage

BIG = "x" * 600


def test_cache_read_is_reported_as_a_hit():
    usage = FakeUsage(input_tokens=120, cache_read_input_tokens=8000, output_tokens=50)
    result = diagnose_usage(usage, model="claude-opus-5")

    assert result.status is CacheStatus.READ
    assert result.cache_hit
    assert not result.silently_missed
    assert result.likely_causes == ()
    assert "cache read" in result.describe()


def test_cold_write_is_distinguished_from_a_repeated_write():
    usage = FakeUsage(input_tokens=120, cache_creation_input_tokens=8000)

    cold = diagnose_usage(usage, model="claude-opus-5", first_request=True)
    assert cold.status is CacheStatus.WRITE
    assert cold.notes == ()

    repeated = diagnose_usage(usage, model="claude-opus-5")
    assert repeated.status is CacheStatus.WRITE
    # A perpetual write is the real symptom of a changing prefix.
    assert any("written but never read" in note for note in repeated.notes)


def test_partial_hit_when_an_early_breakpoint_reads_and_a_later_one_writes():
    usage = FakeUsage(
        input_tokens=40, cache_creation_input_tokens=900, cache_read_input_tokens=6000
    )
    result = diagnose_usage(usage, model="claude-opus-5")
    assert result.status is CacheStatus.PARTIAL
    assert result.cache_hit


def test_total_prompt_tokens_is_the_sum_not_input_tokens():
    """input_tokens is the uncached remainder; reading it alone lies."""
    usage = FakeUsage(
        input_tokens=100, cache_creation_input_tokens=200, cache_read_input_tokens=9000
    )
    result = diagnose_usage(usage, model="claude-opus-5")
    assert result.total_prompt_tokens == 9300
    assert result.tokens["input_tokens"] == 100


def test_miss_below_model_minimum_is_ranked_first():
    usage = FakeUsage(input_tokens=300)
    result = diagnose_usage(
        usage,
        model="claude-opus-5",
        breakpoint_prefix_tokens=[300],
        breakpoints_placed=1,
    )
    assert result.status is CacheStatus.MISS
    assert result.silently_missed
    assert result.likely_causes[0] is MissReason.PREFIX_BELOW_MINIMUM
    assert any("512" in note for note in result.notes)
    assert any("not monotonic" in note for note in result.notes)


def test_miss_with_no_breakpoints_says_so():
    result = diagnose_usage(
        FakeUsage(input_tokens=9000), model="claude-opus-5", breakpoints_placed=0
    )
    assert result.likely_causes[0] is MissReason.NO_BREAKPOINTS


def test_twenty_block_lookback_is_offered_as_a_cause():
    result = diagnose_usage(
        FakeUsage(input_tokens=9000),
        model="claude-opus-5",
        breakpoint_prefix_tokens=[8000],
        breakpoints_placed=1,
        blocks_added_since_last_request=CACHE_LOOKBACK_BLOCKS + 5,
    )
    assert MissReason.LOOKBACK_EXCEEDED in result.likely_causes
    assert MissReason.PREFIX_BELOW_MINIMUM not in result.likely_causes
    assert any("25 blocks were added" in note for note in result.notes)


def test_lookback_within_the_window_is_not_offered():
    result = diagnose_usage(
        FakeUsage(input_tokens=9000),
        model="claude-opus-5",
        breakpoint_prefix_tokens=[8000],
        blocks_added_since_last_request=CACHE_LOOKBACK_BLOCKS,
    )
    assert MissReason.LOOKBACK_EXCEEDED not in result.likely_causes
    assert result.likely_causes[-1] is MissReason.PREFIX_CHANGED


def test_concurrent_fanout_race_is_offered_as_a_cause():
    result = diagnose_usage(
        FakeUsage(input_tokens=9000),
        model="claude-opus-5",
        breakpoint_prefix_tokens=[8000],
        concurrent_requests=8,
    )
    assert MissReason.CONCURRENT_WRITE_RACE in result.likely_causes
    assert any("8 requests shared this prefix" in note for note in result.notes)
    # An entry is only readable once the first response starts streaming.
    assert "begins streaming" in result.describe()


def test_first_request_miss_is_expected_not_a_bug():
    result = diagnose_usage(
        FakeUsage(input_tokens=9000), model="claude-opus-5", first_request=True
    )
    assert result.likely_causes[-1] is MissReason.COLD_START
    assert MissReason.PREFIX_CHANGED not in result.likely_causes


def test_prefix_changed_is_the_fallback_cause():
    result = diagnose_usage(FakeUsage(input_tokens=9000), model="claude-opus-5")
    assert result.likely_causes == (MissReason.PREFIX_CHANGED,)
    assert "find_silent_invalidator" in result.describe()


def test_unknown_model_minimum_is_admitted_rather_than_guessed():
    result = diagnose_usage(FakeUsage(input_tokens=10), model="claude-not-a-model")
    assert any("no minimum cacheable prefix on record" in n for n in result.notes)


# -- end-to-end against a prefix-match fake ------------------------------------


def _cached_prefixes(kwargs: dict) -> list[tuple[bytes, int]]:
    """Byte prefix (and its size) at each block carrying a cache_control marker."""
    rendered = render_prefix(kwargs)
    return [
        (rendered.data[: span.end], span.end)
        for span in rendered.spans
        if b'"cache_control"' in rendered.data[span.start : span.end]
    ]


def _submit(server: FakeCacheServer, kwargs: dict) -> FakeUsage:
    rendered = render_prefix(kwargs)
    return server.submit(_cached_prefixes(kwargs), tail_tokens=len(rendered))


def test_stable_prompt_writes_once_then_reads():
    server = FakeCacheServer(minimum_prefix_tokens=512)

    def build(question: str) -> dict:
        a = PromptAssembler(model="claude-opus-5", token_counter=len)
        a.add_system(BIG, Stability.STATIC)
        a.add_system(BIG, Stability.SESSION)
        a.add_message("user", question, Stability.TURN)
        return a.to_request_kwargs()

    first = diagnose_usage(
        _submit(server, build("q1")), model="claude-opus-5", first_request=True
    )
    assert first.status is CacheStatus.WRITE

    second = diagnose_usage(_submit(server, build("q2")), model="claude-opus-5")
    assert second.status is CacheStatus.READ
    assert second.tokens["cache_read_input_tokens"] > 0


def test_timestamp_in_the_prefix_shows_up_as_a_perpetual_write():
    """The silent failure, end to end: never an error, never a read."""
    server = FakeCacheServer(minimum_prefix_tokens=512)

    def build(clock: str) -> dict:
        a = PromptAssembler(model="claude-opus-5", token_counter=len)
        a.add_system(f"{BIG} now={clock}", Stability.STATIC)
        a.add_system(BIG, Stability.SESSION)
        a.add_message("user", "same question", Stability.TURN)
        return a.to_request_kwargs()

    statuses = [
        diagnose_usage(_submit(server, build(str(tick))), model="claude-opus-5").status
        for tick in range(3)
    ]
    assert statuses == [CacheStatus.WRITE, CacheStatus.WRITE, CacheStatus.WRITE]


def test_prompt_under_the_minimum_never_caches_at_all():
    server = FakeCacheServer(minimum_prefix_tokens=512)
    a = PromptAssembler(model="claude-opus-5", token_counter=len, cache_tail=True)
    a.add_system("tiny", Stability.STATIC)
    a.add_message("user", "hi", Stability.TURN)
    kwargs = a.to_request_kwargs()

    result = diagnose_usage(
        _submit(server, kwargs),
        model="claude-opus-5",
        breakpoint_prefix_tokens=[a.plan().total_estimated_tokens],
        breakpoints_placed=len(a.plan().breakpoints),
    )
    assert result.status is CacheStatus.MISS
    assert MissReason.PREFIX_BELOW_MINIMUM in result.likely_causes or (
        MissReason.NO_BREAKPOINTS in result.likely_causes
    )
