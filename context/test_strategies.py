"""Strategy behaviour: does it shrink the history, and is the result legal.

Every strategy is held to the same two contracts here — reduce the context and
emit a request the API would accept — plus the specific promise each one
makes on top.
"""

from __future__ import annotations

import pytest

from context.strategies import (
    CLEAR_TOOL_USES_EDIT,
    COMPACTION_BETA,
    COMPACTION_EDIT,
    CONTEXT_EDITING_BETA,
    AnchoredSummary,
    Budget,
    NoteTaking,
    RecursiveSummarization,
    ServerCompaction,
    TailTruncation,
    ToolResultEviction,
    all_strategies,
)
from context.summarizers import FakeNoteWriter, FakeSummarizer, extract_facts
from context.tokens import HeuristicTokenCounter
from context.validation import assert_valid, blocks, history_text, is_valid

OBJECTIVE = (
    "Migrate the billing service to eu-west-2 WITHOUT taking a write outage; "
    "a read-only window is acceptable."
)
FILLER = (
    "Reviewed the checklist against the runbook and re-ran the smoke suite. "
    "Nothing in the dashboards has moved outside its band since the last "
    "check, so the plan stands as written. " * 3
)
TOOL_PAYLOAD = (
    "INFO scheduler: reconcile tick, 41 objects scanned, 0 drift\n"
    "INFO billing-api: p99 118ms, p50 21ms, error-rate 0.02%\n" * 6
)


def make_history(turns: int = 8, *, tools: bool = True) -> list[dict]:
    """A long history that is well over any sane budget and ends on `user`."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{OBJECTIVE}\n[FACT] DEPLOY_TOKEN = ZX-4417",
                }
            ],
        }
    ]
    for i in range(turns):
        if tools and i % 2 == 0:
            tid = f"toolu_{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tid,
                            "name": "read_logs",
                            "input": {"service": "billing", "window": f"w{i}"},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tid, "content": TOOL_PAYLOAD}
                    ],
                }
            )
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": f"Step {i} done."}]}
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{FILLER}[FACT] METRIC_{i:02d} = {40 + i}ms"}
                ],
            }
        )
    return messages


COUNTER = HeuristicTokenCounter()


def budget(**kwargs) -> Budget:
    defaults = dict(max_tokens=1200, max_turns=8, keep_recent_messages=4, counter=COUNTER)
    defaults.update(kwargs)
    return Budget(**defaults)


@pytest.fixture(params=[s.name for s in all_strategies()])
def strategy(request):
    by_name = {s.name: s for s in all_strategies()}
    return by_name[request.param]


class TestUniversalContracts:
    """Held by every strategy, no exceptions."""

    def test_history_fixture_is_over_budget_and_legal(self):
        history = make_history()
        assert is_valid(history), "the fixture itself must be a legal request"
        assert COUNTER.count(history) > budget().max_tokens

    def test_reduces_context_or_delegates_to_the_server(self, strategy):
        history = make_history()
        before = COUNTER.count(history)
        result = strategy.apply(history, budget(turn_index=3))
        after = COUNTER.count(result.messages)

        if strategy.client_side_reduction:
            assert after < before, (
                f"{strategy.name} left the history at {after} tokens (was {before})"
            )
        else:
            # ServerCompaction cannot shrink anything client-side; what it must
            # do instead is actually ask the server to.
            assert after == before
            assert result.request_overrides, (
                f"{strategy.name} neither reduced the context nor asked the server to"
            )

    def test_output_passes_the_shape_validator(self, strategy):
        result = strategy.apply(make_history(), budget(turn_index=3))
        assert_valid(result.messages, label=strategy.name)

    def test_output_is_still_valid_on_a_tool_heavy_history(self, strategy):
        history = make_history(turns=10, tools=True)
        assert_valid(strategy.apply(history, budget(turn_index=9)).messages, label=strategy.name)

    def test_does_not_fire_when_under_budget(self, strategy):
        short = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = strategy.apply(short, budget(turn_index=0))
        assert result.fired is False
        assert result.messages == short

    def test_repeated_application_converges(self, strategy):
        """Applying twice must not corrupt the history or grow it."""
        b = budget(turn_index=3)
        first = strategy.apply(make_history(), b)
        second = strategy.apply(first.messages, b)
        assert_valid(second.messages, label=strategy.name)
        assert COUNTER.count(second.messages) <= COUNTER.count(first.messages)

    def test_reset_is_callable(self, strategy):
        strategy.apply(make_history(), budget(turn_index=3))
        strategy.reset()  # must not raise, and must leave the strategy usable
        assert_valid(strategy.apply(make_history(), budget(turn_index=3)).messages)


class TestTriggerPolicy:
    def test_fires_on_the_token_threshold(self):
        strategy = TailTruncation()
        assert strategy.apply(make_history(), budget(turn_index=0, max_turns=999)).fired

    def test_fires_on_the_turn_count_even_when_small(self):
        """The turn backstop: many tiny turns never trip a token threshold."""
        tiny = [
            {"role": "user", "content": [{"type": "text", "text": "ok"}]}
            for _ in range(3)
        ]
        assert COUNTER.count(tiny) < 1200
        strategy = TailTruncation()
        assert strategy.apply(tiny, budget(turn_index=0)).fired is False
        assert strategy.apply(tiny, budget(turn_index=9)).fired is True


class TestTailTruncation:
    def test_drops_the_oldest_and_loses_early_information(self):
        history = make_history()
        result = TailTruncation().apply(history, budget(turn_index=3))
        assert "DEPLOY_TOKEN = ZX-4417" not in history_text(result.messages)
        assert "Step 7 done." in history_text(result.messages)

    def test_costs_no_model_calls(self):
        result = TailTruncation().apply(make_history(), budget(turn_index=3))
        assert result.usage.model_calls == 0
        assert result.usage.total_tokens == 0

    def test_never_empties_the_history(self):
        result = TailTruncation().apply(make_history(), budget(max_tokens=1, turn_index=3))
        assert result.messages
        assert_valid(result.messages)


class TestToolResultEviction:
    def test_preserves_tool_use_result_pairing(self):
        history = make_history()
        result = ToolResultEviction().apply(history, budget(turn_index=3))
        assert_valid(result.messages, label="ToolResultEviction")

        def ids(messages, btype, key):
            return [
                b[key] for m in messages for b in blocks(m) if b.get("type") == btype
            ]

        assert ids(result.messages, "tool_use", "id") == ids(history, "tool_use", "id")
        assert ids(result.messages, "tool_result", "tool_use_id") == ids(
            history, "tool_result", "tool_use_id"
        )

    def test_leaves_a_placeholder_rather_than_removing_the_block(self):
        result = ToolResultEviction().apply(make_history(), budget(turn_index=3))
        payloads = [
            b["content"]
            for m in result.messages
            for b in blocks(m)
            if b.get("type") == "tool_result"
        ]
        assert any("evicted" in str(p) for p in payloads)
        assert all(p for p in payloads), "a tool_result must never become empty"

    def test_keeps_the_most_recent_tool_result_intact(self):
        result = ToolResultEviction().apply(
            make_history(), budget(turn_index=3, keep_recent_tool_results=1)
        )
        payloads = [
            b["content"]
            for m in result.messages
            for b in blocks(m)
            if b.get("type") == "tool_result"
        ]
        assert "INFO scheduler" in str(payloads[-1])

    def test_preserves_conversational_text_entirely(self):
        """Its blind spot is tool output, not user text."""
        result = ToolResultEviction().apply(make_history(), budget(turn_index=3))
        assert "DEPLOY_TOKEN = ZX-4417" in history_text(result.messages)

    def test_does_not_evict_when_that_would_grow_the_request(self):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "x", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
            {"role": "user", "content": [{"type": "text", "text": "next"}]},
        ]
        result = ToolResultEviction().apply(history, budget(max_tokens=0, turn_index=3))
        payload = [
            b["content"]
            for m in result.messages
            for b in blocks(m)
            if b.get("type") == "tool_result"
        ]
        assert payload == ["ok"], "a 2-char payload must not be replaced by a longer note"


class TestSummarizingStrategies:
    def test_recursive_summarization_calls_the_summarizer_and_reports_usage(self):
        summarizer = FakeSummarizer()
        result = RecursiveSummarization(summarizer).apply(make_history(), budget(turn_index=3))
        assert summarizer.calls == 1
        assert result.usage.model_calls == 1
        assert result.usage.total_tokens > 0, "summarizer tokens must not be free"

    def test_the_summary_is_rolling(self):
        summarizer = FakeSummarizer()
        strategy = RecursiveSummarization(summarizer)
        b = budget(turn_index=3)
        first = strategy.apply(make_history(), b)
        assert strategy._summary
        strategy.apply(first.messages + make_history(turns=4), b)
        assert summarizer.calls == 2

    def test_reset_clears_the_rolling_summary(self):
        strategy = RecursiveSummarization(FakeSummarizer())
        strategy.apply(make_history(), budget(turn_index=3))
        strategy.reset()
        assert strategy._summary == ""

    def test_anchored_summary_keeps_the_objective_verbatim(self):
        strategy = AnchoredSummary(FakeSummarizer(), objective=OBJECTIVE)
        result = strategy.apply(make_history(), budget(turn_index=3))
        assert OBJECTIVE in history_text(result.messages)

    def test_anchor_survives_repeated_compaction(self):
        strategy = AnchoredSummary(FakeSummarizer(), objective=OBJECTIVE)
        b = budget(turn_index=3)
        messages = make_history()
        for _ in range(4):
            messages = strategy.apply(messages + make_history(turns=3), b).messages
            assert OBJECTIVE in history_text(messages), "objective drifted"
        assert_valid(messages, label="AnchoredSummary")

    def test_anchor_falls_back_to_the_opening_turn(self):
        strategy = AnchoredSummary(FakeSummarizer())
        result = strategy.apply(make_history(), budget(turn_index=3))
        assert OBJECTIVE in history_text(result.messages)

    def test_plain_recursive_summarization_can_lose_the_objective(self):
        """The contrast that justifies anchoring's cost."""
        strategy = RecursiveSummarization(FakeSummarizer())
        result = strategy.apply(make_history(), budget(turn_index=3))
        assert OBJECTIVE not in history_text(result.messages)


class TestNoteTaking:
    def test_writes_a_notes_file_and_rehydrates_from_it(self, tmp_path):
        strategy = NoteTaking(FakeNoteWriter(), workspace=tmp_path)
        result = strategy.apply(make_history(), budget(turn_index=3))

        assert strategy.notes_path.exists()
        notes = strategy.notes_path.read_text()
        assert notes.strip()
        # The rehydrated context must be the file's contents, not a paraphrase.
        assert notes in history_text(result.messages)

    def test_notes_survive_across_compactions(self, tmp_path):
        strategy = NoteTaking(FakeNoteWriter(), workspace=tmp_path)
        b = budget(turn_index=3)
        messages = make_history()
        for _ in range(3):
            messages = strategy.apply(messages + make_history(turns=3), b).messages
        assert "DEPLOY_TOKEN" in strategy.read_notes()

    def test_pins_facts_the_agent_wrote_down(self, tmp_path):
        """The one channel a rolling summary has no equivalent of."""
        writer = FakeNoteWriter(max_facts=2)
        strategy = NoteTaking(writer, workspace=tmp_path)
        history = make_history(turns=6)
        history.insert(
            1,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "note1",
                        "name": "write_note",
                        "input": {"note": "[FACT] DEPLOY_TOKEN = ZX-4417"},
                    }
                ],
            },
        )
        history.insert(
            2,
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "note1", "content": "written"}
                ],
            },
        )
        strategy.apply(history, budget(turn_index=3))
        kept = dict(extract_facts(strategy.read_notes()))
        assert "DEPLOY_TOKEN" in kept, "an agent-written note was evicted before filler"

    def test_unpinned_early_facts_can_still_be_evicted(self, tmp_path):
        """Note-taking is not magic: capacity still bites."""
        strategy = NoteTaking(FakeNoteWriter(max_facts=2), workspace=tmp_path)
        strategy.apply(make_history(turns=8), budget(turn_index=3))
        assert len(extract_facts(strategy.read_notes())) <= 2

    def test_reset_clears_the_file(self, tmp_path):
        strategy = NoteTaking(FakeNoteWriter(), workspace=tmp_path)
        strategy.apply(make_history(), budget(turn_index=3))
        assert strategy.notes_path.exists()
        strategy.reset()
        assert not strategy.notes_path.exists()

    def test_reports_note_writer_usage(self, tmp_path):
        result = NoteTaking(FakeNoteWriter(), workspace=tmp_path).apply(
            make_history(), budget(turn_index=3)
        )
        assert result.usage.model_calls == 1
        assert result.usage.total_tokens > 0

    def test_default_workspace_is_self_managed(self):
        strategy = NoteTaking(FakeNoteWriter())
        strategy.apply(make_history(), budget(turn_index=3))
        assert strategy.notes_path.exists()


class TestServerCompaction:
    def test_sends_the_compaction_beta_and_edit(self):
        overrides = ServerCompaction().apply(make_history(), budget(turn_index=3)).request_overrides
        assert overrides["betas"] == [COMPACTION_BETA]
        assert overrides["context_management"]["edits"] == [COMPACTION_EDIT]

    def test_does_not_edit_the_history_itself(self):
        history = make_history()
        result = ServerCompaction().apply(history, budget(turn_index=3))
        assert history_text(result.messages) == history_text(history)

    def test_overrides_are_attached_even_before_the_budget_trips(self):
        """The server decides when to compact; enabling it late is too late."""
        short = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = ServerCompaction().apply(short, budget(turn_index=0))
        assert result.fired is False
        assert result.request_overrides["betas"] == [COMPACTION_BETA]

    def test_context_editing_is_a_separate_feature_with_its_own_header(self):
        """Compaction summarizes; context editing clears. Never conflate them."""
        overrides = (
            ServerCompaction(include_tool_clearing=True)
            .apply(make_history(), budget(turn_index=3))
            .request_overrides
        )
        assert overrides["betas"] == [COMPACTION_BETA, CONTEXT_EDITING_BETA]
        assert overrides["context_management"]["edits"] == [
            COMPACTION_EDIT,
            CLEAR_TOOL_USES_EDIT,
        ]
        assert COMPACTION_EDIT["type"] == "compact_20260112"
        assert CLEAR_TOOL_USES_EDIT["type"] == "clear_tool_uses_20250919"
        assert COMPACTION_BETA == "compact-2026-01-12"
        assert CONTEXT_EDITING_BETA == "context-management-2025-06-27"

    def test_costs_no_client_side_model_calls(self):
        result = ServerCompaction().apply(make_history(), budget(turn_index=3))
        assert result.usage.model_calls == 0
