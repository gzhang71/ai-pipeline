"""Every structural assertion type, passing and failing.

Each type gets both directions. An assertion that can only pass is not an
assertion.
"""

from __future__ import annotations

import pytest

from harness import jsonschema
from harness.assertions import AssertionSpecError, evaluate, evaluate_all, extract_json


def check(spec, output) -> bool:
    return evaluate(spec, output).passed


# --- contains / not_contains ---------------------------------------------------


def test_contains(make_output):
    spec = {"type": "contains", "text": "escalate"}
    assert check(spec, make_output("please escalate this")) is True
    assert check(spec, make_output("resolved, no action")) is False


def test_contains_is_case_insensitive_by_default(make_output):
    spec = {"type": "contains", "text": "Escalate"}
    assert check(spec, make_output("ESCALATE now")) is True
    assert check({**spec, "case_sensitive": True}, make_output("ESCALATE now")) is False


def test_not_contains(make_output):
    spec = {"type": "not_contains", "text": "123-45-6789"}
    assert check(spec, make_output("ssn withheld")) is True
    assert check(spec, make_output("ssn is 123-45-6789")) is False


def test_failure_detail_names_the_string(make_output):
    result = evaluate({"type": "contains", "text": "zebra"}, make_output("giraffe"))
    assert "zebra" in result.detail


# --- regex ---------------------------------------------------------------------


def test_regex(make_output):
    spec = {"type": "regex", "pattern": r'"severity"\s*:\s*"high"'}
    assert check(spec, make_output('{"severity": "high"}')) is True
    assert check(spec, make_output('{"severity": "low"}')) is False


def test_regex_negate(make_output):
    spec = {"type": "regex", "pattern": r"\bTODO\b", "negate": True}
    assert check(spec, make_output("all done")) is True
    assert check(spec, make_output("TODO: finish")) is False


def test_regex_flags(make_output):
    spec = {"type": "regex", "pattern": "^done$", "flags": "im"}
    assert check(spec, make_output("first\nDONE\nlast")) is True
    assert check({"type": "regex", "pattern": "^done$"}, make_output("first\nDONE")) is False


def test_unknown_regex_flag_is_a_spec_error():
    with pytest.raises(AssertionSpecError):
        from harness.assertions import validate_spec

        validate_spec({"type": "regex", "pattern": "x", "flags": "z"}, 0)


# --- json_valid / json_schema ---------------------------------------------------


def test_json_valid(make_output):
    spec = {"type": "json_valid"}
    assert check(spec, make_output('{"a": 1}')) is True
    assert check(spec, make_output("here you go: {a: 1}")) is False
    assert check(spec, make_output("")) is False


def test_json_valid_unwraps_a_code_fence(make_output):
    spec = {"type": "json_valid"}
    assert check(spec, make_output('```json\n{"a": 1}\n```')) is True
    assert check({**spec, "allow_code_fence": False}, make_output('```json\n{"a": 1}\n```')) is False


def test_json_with_preamble_before_a_fence_does_not_parse(make_output):
    """This is exactly the v1 -> v2 improvement the demo shows."""
    assert check({"type": "json_valid"}, make_output('Here you go:\n\n```json\n{"a": 1}\n```')) is False


def test_json_schema(make_output):
    spec = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "required": ["category"],
            "properties": {"category": {"type": "string", "enum": ["bug", "billing"]}},
        },
    }
    assert check(spec, make_output('{"category": "bug"}')) is True
    assert check(spec, make_output('{"category": "weather"}')) is False
    assert check(spec, make_output('{"severity": "low"}')) is False


def test_json_schema_from_schema_json_string(make_output):
    spec = {"type": "json_schema", "schema_json": '{"type": "array", "minItems": 2}'}
    assert check(spec, make_output("[1, 2]")) is True
    assert check(spec, make_output("[1]")) is False


def test_json_schema_detail_lists_the_violation(make_output):
    spec = {"type": "json_schema", "schema": {"type": "object", "required": ["x"]}}
    result = evaluate(spec, make_output("{}"))
    assert "missing required property 'x'" in result.detail


def test_json_schema_without_a_schema_is_a_spec_error(make_output):
    with pytest.raises(AssertionSpecError):
        evaluate({"type": "json_schema"}, make_output("{}"))


# --- length --------------------------------------------------------------------


def test_length_chars(make_output):
    spec = {"type": "length", "min_chars": 3, "max_chars": 10}
    assert check(spec, make_output("hello")) is True
    assert check(spec, make_output("hi")) is False
    assert check(spec, make_output("x" * 11)) is False


def test_length_words(make_output):
    spec = {"type": "length", "max_words": 3}
    assert check(spec, make_output("one two three")) is True
    assert check(spec, make_output("one two three four")) is False


def test_length_detail_reports_the_measurement(make_output):
    result = evaluate({"type": "length", "max_words": 2}, make_output("a b c"))
    assert "words 3 > max 2" in result.detail


# --- tools ---------------------------------------------------------------------


def test_tool_called_by_name(make_output):
    spec = {"type": "tool_called", "name": "lookup_account"}
    called = make_output("", tool_calls=(("lookup_account", {"email": "a@b.c"}),))
    assert check(spec, called) is True
    assert check(spec, make_output("no tools here")) is False
    assert check(spec, make_output("", tool_calls=(("other_tool", {}),))) is False


def test_tool_called_any_tool(make_output):
    spec = {"type": "tool_called"}
    assert check(spec, make_output("", tool_calls=(("anything", {}),))) is True
    assert check(spec, make_output("text only")) is False


def test_tool_called_exact_count(make_output):
    spec = {"type": "tool_called", "name": "t", "count": 2}
    assert check(spec, make_output("", tool_calls=(("t", {}), ("t", {})))) is True
    assert check(spec, make_output("", tool_calls=(("t", {}),))) is False


def test_no_tool_called(make_output):
    spec = {"type": "no_tool_called"}
    assert check(spec, make_output("just text")) is True
    assert check(spec, make_output("", tool_calls=(("search", {}),))) is False


def test_tool_failure_detail_lists_what_was_called(make_output):
    result = evaluate(
        {"type": "tool_called", "name": "wanted"},
        make_output("", tool_calls=(("other", {}),)),
    )
    assert "other" in result.detail


# --- stop_reason ---------------------------------------------------------------


def test_stop_reason(make_output):
    spec = {"type": "stop_reason", "equals": "end_turn"}
    assert check(spec, make_output("done", stop_reason="end_turn")) is True
    assert check(spec, make_output("cut off", stop_reason="max_tokens")) is False


def test_stop_reason_tool_use(make_output):
    spec = {"type": "stop_reason", "equals": "tool_use"}
    assert check(spec, make_output("", stop_reason="tool_use")) is True
    assert check(spec, make_output("", stop_reason=None)) is False


# --- error handling and batching ------------------------------------------------


def test_model_error_fails_every_assertion(make_output):
    broken = make_output("", error="APIConnectionError: boom")
    for spec in (
        {"type": "contains", "text": "x"},
        {"type": "json_valid"},
        {"type": "no_tool_called"},
        {"type": "stop_reason", "equals": "end_turn"},
    ):
        result = evaluate(spec, broken)
        assert result.passed is False
        assert "model error" in result.detail


def test_evaluate_all_preserves_order_and_ids(make_output):
    specs = (
        {"id": "first", "type": "contains", "text": "a"},
        {"type": "json_valid"},
    )
    results = evaluate_all(specs, make_output("a"))
    assert [r.id for r in results] == ["first", "a1:json_valid"]
    assert [r.passed for r in results] == [True, False]


def test_judge_type_is_not_evaluated_here(make_output):
    with pytest.raises(AssertionSpecError):
        evaluate({"type": "judge", "criterion": "x"}, make_output("y"))


def test_assertion_result_round_trips(make_output):
    from harness.assertions import AssertionResult

    original = evaluate({"type": "contains", "text": "q"}, make_output("z"))
    assert AssertionResult.from_dict(original.to_dict()) == original


# --- extract_json --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('  {"a": 1}  ', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
    ],
)
def test_extract_json_accepts(text, expected):
    value, error = extract_json(text)
    assert error == ""
    assert value == expected


@pytest.mark.parametrize("text", ["", "not json", '{"a": ', 'prefix ```json\n{}\n```'])
def test_extract_json_rejects(text):
    value, error = extract_json(text)
    assert value is None
    assert error


# --- the bundled JSON Schema subset ---------------------------------------------


def test_schema_types():
    assert jsonschema.validate(1, {"type": "integer"}) == []
    assert jsonschema.validate(True, {"type": "integer"}) != []
    assert jsonschema.validate(1.5, {"type": "number"}) == []
    assert jsonschema.validate(None, {"type": "null"}) == []
    assert jsonschema.validate("s", {"type": ["string", "null"]}) == []


def test_schema_additional_properties_false():
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "additionalProperties": False}
    assert jsonschema.validate({"a": "x"}, schema) == []
    assert jsonschema.validate({"a": "x", "b": 1}, schema) != []


def test_schema_const_and_enum():
    assert jsonschema.validate("high", {"const": "high"}) == []
    assert jsonschema.validate("low", {"const": "high"}) != []
    assert jsonschema.validate("low", {"enum": ["low", "high"]}) == []


def test_schema_nested_and_arrays():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
    }
    assert jsonschema.validate({"items": [1, 2]}, schema) == []
    errors = jsonschema.validate({"items": [1, "x"]}, schema)
    assert errors and "$.items[1]" in errors[0]


def test_schema_bounds_and_pattern():
    assert jsonschema.validate("abc", {"minLength": 2, "maxLength": 3}) == []
    assert jsonschema.validate("abcd", {"maxLength": 3}) != []
    assert jsonschema.validate(5, {"minimum": 1, "maximum": 10}) == []
    assert jsonschema.validate(0, {"minimum": 1}) != []
    assert jsonschema.validate("a1", {"pattern": r"^[a-z]\d$"}) == []
    assert jsonschema.validate("11", {"pattern": r"^[a-z]\d$"}) != []


def test_schema_combinators():
    any_of = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert jsonschema.validate("s", any_of) == []
    assert jsonschema.validate(1.5, any_of) != []
    one_of = {"oneOf": [{"type": "integer"}, {"minimum": 10}]}
    assert jsonschema.validate(5, one_of) == []
    assert jsonschema.validate(50, one_of) != []


def test_check_schema_rejects_unsupported_keywords():
    with pytest.raises(jsonschema.SchemaError, match="unsupported schema keyword"):
        jsonschema.check_schema({"type": "object", "patternProperties": {}})
    with pytest.raises(jsonschema.SchemaError, match="unknown type"):
        jsonschema.check_schema({"type": "dictionary"})
    with pytest.raises(jsonschema.SchemaError):
        jsonschema.check_schema({"properties": {"a": {"minProperties": 1}}})
