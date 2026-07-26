"""The JSONL schema is a public interface; treat breaking it as a failure."""

from __future__ import annotations

import copy
import json

import pytest

from loop.agent import LoopConfig, run_loop
from loop.schema import (
    SCHEMA_ID,
    SCHEMA_VERSION,
    SEGMENT_KINDS,
    SchemaError,
    group_runs,
    validate_record,
    validate_run,
)
from loop.sink import JsonlSink, MemorySink, read_jsonl, write_jsonl
from loop.testing import (
    FakeAnthropicClient,
    FakeMessage,
    echo_executor,
    heuristic_token_count,
    text_block,
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
                    content=[tool_use_block("lookup", {"q": "x"}, "toolu_1")],
                    stop_reason="tool_use",
                ),
                FakeMessage(content=[text_block("41250")], stop_reason="end_turn"),
            ]
        ),
        tools=TOOLS,
        executor=echo_executor,
        prompt="What is on invoice 7781?",
        config=LoopConfig(system="You are a ledger assistant. " * 20),
        sink=sink,
        counter=heuristic_token_count,
        task={"task_id": "demo", "filler_tokens": 0},
    )
    return sink.records


# --------------------------------------------------------------------------
# Real runs validate
# --------------------------------------------------------------------------


def test_a_real_run_validates(records):
    assert validate_run(records) == records
    assert [r["record_type"] for r in records] == [
        "run_header",
        "turn",
        "turn",
        "run_footer",
    ]
    for record in records:
        assert record["schema_id"] == SCHEMA_ID
        assert record["schema_version"] == SCHEMA_VERSION
        assert record["run_id"] == records[0]["run_id"]


def test_records_are_json_serializable(records):
    for record in records:
        round_tripped = json.loads(json.dumps(record))
        validate_record(round_tripped)


def test_every_segment_kind_emitted_is_declared_in_the_schema(records):
    for record in records:
        if record["record_type"] != "turn":
            continue
        for segment in record["prompt_tokens"]["segments"]:
            assert segment["kind"] in SEGMENT_KINDS
        for kind in record["prompt_tokens"]["by_kind"]:
            assert kind in SEGMENT_KINDS


def test_group_runs_reassembles_the_run(records):
    runs = group_runs(records)
    assert len(runs) == 1
    assert runs[0]["header"]["record_type"] == "run_header"
    assert len(runs[0]["turns"]) == 2
    assert runs[0]["footer"]["record_type"] == "run_footer"


def test_two_concatenated_runs_validate_together(records):
    second = copy.deepcopy(records)
    for record in second:
        record["run_id"] = "run_second"
    combined = records + second
    validate_run(combined)
    assert len(group_runs(combined)) == 2


# --------------------------------------------------------------------------
# Malformed records are rejected
# --------------------------------------------------------------------------


def test_unknown_schema_id_is_rejected(records):
    bad = copy.deepcopy(records[0])
    bad["schema_id"] = "somebody-elses-schema"
    with pytest.raises(SchemaError, match="unknown schema_id"):
        validate_record(bad)


def test_future_schema_version_is_rejected(records):
    bad = copy.deepcopy(records[0])
    bad["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(SchemaError, match="not readable"):
        validate_record(bad)


def test_missing_required_field_is_rejected(records):
    bad = copy.deepcopy(records[1])
    del bad["reconciliation"]
    with pytest.raises(SchemaError, match="reconciliation"):
        validate_record(bad)


def test_segment_arithmetic_is_enforced(records):
    bad = copy.deepcopy(records[1])
    bad["prompt_tokens"]["segments"][0]["tokens"] += 100
    with pytest.raises(SchemaError, match="segment_sum"):
        validate_record(bad)


def test_segment_sum_plus_residual_must_equal_counted_total(records):
    bad = copy.deepcopy(records[1])
    bad["prompt_tokens"]["counted_total"] += 7
    with pytest.raises(SchemaError, match="decomposition_residual"):
        validate_record(bad)


def test_unknown_segment_kind_is_rejected(records):
    bad = copy.deepcopy(records[1])
    bad["prompt_tokens"]["segments"][0]["kind"] = "vibes"
    with pytest.raises(SchemaError, match="unknown segment kind"):
        validate_record(bad)


def test_duplicate_segment_ids_are_rejected(records):
    bad = copy.deepcopy(records[1])
    segments = bad["prompt_tokens"]["segments"]
    segments.append(copy.deepcopy(segments[0]))
    with pytest.raises(SchemaError, match="duplicate segment_id"):
        validate_record(bad)


def test_out_of_order_turns_are_rejected(records):
    bad = copy.deepcopy(records)
    bad[1], bad[2] = bad[2], bad[1]
    with pytest.raises(SchemaError, match="out of order"):
        validate_run(bad)


def test_a_run_without_a_footer_is_rejected(records):
    with pytest.raises(SchemaError, match="no run_footer"):
        validate_run(records[:-1])


def test_a_footer_that_miscounts_turns_is_rejected(records):
    bad = copy.deepcopy(records)
    bad[-1]["turns"] = 99
    with pytest.raises(SchemaError, match="turn records were seen"):
        validate_run(bad)


def test_turn_before_header_is_rejected(records):
    with pytest.raises(SchemaError, match="expected a run_header"):
        validate_run(records[1:])


def test_unknown_run_stop_reason_is_rejected(records):
    bad = copy.deepcopy(records[-1])
    bad["stop_reason"] = "gave_up"
    with pytest.raises(SchemaError, match="unknown run stop_reason"):
        validate_record(bad)


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


def test_jsonl_round_trip(tmp_path, records):
    path = tmp_path / "run.jsonl"
    assert write_jsonl(path, records) == len(records)
    loaded = read_jsonl(path)
    assert loaded == records
    validate_run(loaded)


def test_jsonl_sink_creates_parent_directories(tmp_path, records):
    path = tmp_path / "nested" / "deeper" / "run.jsonl"
    with JsonlSink(path) as sink:
        sink.write_all(records)
    assert path.exists()
    assert len(read_jsonl(path)) == len(records)


def test_sink_rejects_a_bad_record_at_the_producer(tmp_path, records):
    bad = copy.deepcopy(records[0])
    del bad["model"]
    with JsonlSink(tmp_path / "run.jsonl") as sink:
        with pytest.raises(SchemaError):
            sink.write(bad)
        assert sink.count == 0


def test_reader_reports_the_offending_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(path, validate=False)


def test_appending_two_runs_to_one_file_validates(tmp_path, records):
    path = tmp_path / "many.jsonl"
    second = copy.deepcopy(records)
    for record in second:
        record["run_id"] = "run_two"
    write_jsonl(path, records)
    write_jsonl(path, second)
    loaded = read_jsonl(path)
    assert len(group_runs(loaded)) == 2
