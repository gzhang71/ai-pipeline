"""Prompt loading and, above all, hashing."""

from __future__ import annotations

import pytest

from harness.prompts import (
    Prompt,
    PromptError,
    default_prompt_dir,
    load_prompt_file,
    load_prompts,
    parse_prompt,
)


def test_hash_changes_when_body_changes(write_prompt):
    before = write_prompt("p.v1", "You are a helpful assistant.")
    after = write_prompt("p.v1", "You are a helpful, terse assistant.")
    assert before.hash != after.hash


def test_hash_changes_on_a_single_character_edit(write_prompt):
    before = write_prompt("p.v1", "Answer briefly.")
    after = write_prompt("p.v1", "Answer briefly!")
    assert before.hash != after.hash


def test_hash_changes_on_whitespace_only_edit(write_prompt):
    """Trailing whitespace is a byte change, and the model sees the bytes."""
    before = write_prompt("p.v1", "Answer briefly.")
    after = write_prompt("p.v1", "Answer briefly. ")
    assert before.hash != after.hash


def test_hash_changes_when_only_metadata_changes(write_prompt):
    before = write_prompt("p.v1", "Answer briefly.", meta='effort = "low"')
    after = write_prompt("p.v1", "Answer briefly.", meta='effort = "high"')
    assert before.system == after.system
    assert before.hash != after.hash, "effort is part of the run's identity"


def test_identical_bytes_hash_identically(tmp_path):
    body = b"+++\ndescription = \"x\"\n+++\nBe brief."
    one = parse_prompt(body, path=tmp_path / "a.prompt.md")
    two = parse_prompt(body, path=tmp_path / "b.prompt.md")
    assert one.hash == two.hash


def test_no_manual_version_bump_is_required(write_prompt, tmp_path):
    """Editing a prompt file in place produces a new hash with no bookkeeping."""
    original = write_prompt("p.v1", "Step one.")
    path = original.path
    path.write_text(path.read_text("utf-8") + "\nStep two.", "utf-8")
    reloaded = load_prompt_file(path)
    assert reloaded.id == original.id
    assert reloaded.hash != original.hash


def test_frontmatter_is_parsed_and_body_preserved(write_prompt):
    prompt = write_prompt(
        "p.v1",
        "Line one.\nLine two.",
        meta='description = "d"\neffort = "medium"\nmax_tokens = 512',
    )
    assert prompt.description == "d"
    assert prompt.effort == "medium"
    assert prompt.max_tokens == 512
    assert prompt.system == "Line one.\nLine two."


def test_id_comes_from_filename(write_prompt):
    assert write_prompt("triage.v9", "Body.").id == "triage.v9"


def test_frontmatter_id_overrides_filename(tmp_path):
    path = tmp_path / "whatever.prompt.md"
    prompt = parse_prompt(b'+++\nid = "explicit.id"\n+++\nBody.', path=path)
    assert prompt.id == "explicit.id"


def test_prompt_without_frontmatter_is_valid(tmp_path):
    prompt = parse_prompt(b"Just a system prompt.", path=tmp_path / "bare.prompt.md")
    assert prompt.system == "Just a system prompt."
    assert prompt.meta == {}


def test_unterminated_frontmatter_raises(tmp_path):
    with pytest.raises(PromptError, match="unterminated"):
        parse_prompt(b'+++\ndescription = "x"\nBody.', path=tmp_path / "x.prompt.md")


def test_bad_toml_frontmatter_raises(tmp_path):
    with pytest.raises(PromptError, match="bad TOML"):
        parse_prompt(b"+++\nnot = = toml\n+++\nBody.", path=tmp_path / "x.prompt.md")


def test_empty_body_raises(tmp_path):
    with pytest.raises(PromptError, match="empty"):
        parse_prompt(b'+++\ndescription = "x"\n+++\n   \n', path=tmp_path / "x.prompt.md")


def test_missing_directory_raises(tmp_path):
    with pytest.raises(PromptError, match="no such prompt directory"):
        load_prompts(tmp_path / "nope")


def test_bundled_prompts_load():
    prompts = load_prompts(default_prompt_dir())
    assert {"triage.v1", "triage.v2", "judge.rubric.v1"} <= set(prompts)
    assert all(isinstance(p, Prompt) and p.hash for p in prompts.values())


def test_bundled_prompt_versions_have_distinct_hashes():
    prompts = load_prompts(default_prompt_dir())
    assert prompts["triage.v1"].hash != prompts["triage.v2"].hash


def test_short_hash_is_a_prefix(write_prompt):
    prompt = write_prompt("p.v1", "Body.")
    assert prompt.hash.startswith(prompt.short_hash)
    assert len(prompt.short_hash) == 12
    assert prompt.label() == f"p.v1@{prompt.short_hash}"
