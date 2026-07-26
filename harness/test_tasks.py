"""Task file loading, validation, and task-set hashing."""

from __future__ import annotations

import pytest

from harness.tasks import TaskError, default_task_dir, load_tasks, parse_task, task_set_hash


def test_bundled_tasks_load_and_ids_match_filenames():
    tasks = load_tasks(default_task_dir())
    assert len(tasks) >= 5, "the shipped fixture set should be 5-8 tasks"
    for task_id, task in tasks.items():
        assert task_id == task.path.stem
        assert task.assertions


def test_bundled_set_exercises_both_tiers():
    tasks = load_tasks(default_task_dir())
    types = {str(a.get("type")) for t in tasks.values() for a in t.assertions}
    assert "judge" in types
    assert {"json_schema", "regex", "tool_called", "stop_reason", "length"} <= types


def test_task_hash_changes_when_content_changes(write_task):
    before = write_task(
        "t", 'input = "a"\n[[assertions]]\ntype = "contains"\ntext = "x"\n'
    )
    after = write_task(
        "t", 'input = "b"\n[[assertions]]\ntype = "contains"\ntext = "x"\n'
    )
    assert before.hash != after.hash


def test_task_set_hash_is_order_independent_but_content_sensitive(write_task):
    a = write_task("ta", 'input = "a"\n[[assertions]]\ntype = "json_valid"\n')
    b = write_task("tb", 'input = "b"\n[[assertions]]\ntype = "json_valid"\n')
    assert task_set_hash([a, b]) == task_set_hash([b, a])
    c = write_task("tb", 'input = "b2"\n[[assertions]]\ntype = "json_valid"\n')
    assert task_set_hash([a, b]) != task_set_hash([a, c])


def test_missing_input_raises(tmp_path):
    with pytest.raises(TaskError, match="missing required key"):
        parse_task(b'[[assertions]]\ntype = "json_valid"\n', path=tmp_path / "t.toml")


def test_no_assertions_raises(tmp_path):
    with pytest.raises(TaskError, match="at least one"):
        parse_task(b'input = "x"\n', path=tmp_path / "t.toml")


def test_unknown_assertion_type_fails_at_load_time(tmp_path):
    """A typo must fail the run, not silently never fire."""
    with pytest.raises(TaskError, match="unknown type"):
        parse_task(
            b'input = "x"\n[[assertions]]\ntype = "containz"\ntext = "y"\n',
            path=tmp_path / "t.toml",
        )


def test_missing_required_assertion_field_fails_at_load_time(tmp_path):
    with pytest.raises(TaskError, match="missing 'text'"):
        parse_task(
            b'input = "x"\n[[assertions]]\ntype = "contains"\n', path=tmp_path / "t.toml"
        )


def test_bad_regex_fails_at_load_time(tmp_path):
    with pytest.raises(TaskError, match="bad regex"):
        parse_task(
            b'input = "x"\n[[assertions]]\ntype = "regex"\npattern = "([unclosed"\n',
            path=tmp_path / "t.toml",
        )


def test_unsupported_schema_keyword_fails_at_load_time(tmp_path):
    with pytest.raises(TaskError, match="unsupported schema keyword"):
        parse_task(
            b'input = "x"\n[[assertions]]\ntype = "json_schema"\n'
            b'schema_json = \'{"type": "object", "patternProperties": {}}\'\n',
            path=tmp_path / "t.toml",
        )


def test_length_assertion_needs_a_bound(tmp_path):
    with pytest.raises(TaskError, match="min_chars"):
        parse_task(
            b'input = "x"\n[[assertions]]\ntype = "length"\n', path=tmp_path / "t.toml"
        )


def test_tool_schema_json_is_normalized(write_task):
    task = write_task(
        "t",
        """
        input = "x"

        [[tools]]
        name = "lookup"
        description = "d"
        input_schema_json = '{"type": "object", "properties": {"q": {"type": "string"}}}'

        [[assertions]]
        type = "tool_called"
        name = "lookup"
        """,
    )
    assert task.tools[0]["input_schema"]["properties"]["q"]["type"] == "string"
    assert task.has_judge is False


def test_has_judge_detects_tier_two(write_task):
    task = write_task(
        "t",
        'input = "x"\n[[assertions]]\ntype = "judge"\ncriterion = "is it good"\n',
    )
    assert task.has_judge is True


def test_duplicate_ids_rejected(tmp_path):
    directory = tmp_path / "tasks"
    directory.mkdir()
    body = b'id = "same"\ninput = "x"\n[[assertions]]\ntype = "json_valid"\n'
    (directory / "a.toml").write_bytes(body)
    (directory / "b.toml").write_bytes(body)
    with pytest.raises(TaskError, match="duplicate task id"):
        load_tasks(directory)


def test_empty_directory_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(TaskError, match="no .toml task files"):
        load_tasks(tmp_path / "empty")
