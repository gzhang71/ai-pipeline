"""Index persistence and incremental refresh."""

from __future__ import annotations

import json
import os
import time

from graph.index import CodeIndex, discover_python_files


def _touch_later(path: str) -> None:
    """Make the stat stamp differ even on a coarse-grained filesystem clock."""
    stat = os.stat(path)
    os.utime(path, (stat.st_atime + 2, stat.st_mtime + 2))


def test_discover_skips_dot_and_excluded_dirs(mutable_repo: str):
    os.makedirs(os.path.join(mutable_repo, ".venv", "lib"))
    with open(os.path.join(mutable_repo, ".venv", "lib", "junk.py"), "w") as handle:
        handle.write("x = 1\n")
    os.makedirs(os.path.join(mutable_repo, "__pycache__"))
    with open(os.path.join(mutable_repo, "__pycache__", "cached.py"), "w") as handle:
        handle.write("x = 1\n")

    found = discover_python_files(mutable_repo)
    assert "pkg/core.py" in found
    assert not any(p.startswith(".venv") for p in found)
    assert not any("__pycache__" in p for p in found)


def test_save_and_load_roundtrip(mutable_repo: str, tmp_path):
    index = CodeIndex.build(mutable_repo)
    target = str(tmp_path / "index.json")
    index.save(target)

    payload = json.loads(open(target).read())
    assert payload["version"] == 1
    assert "pkg/core.py" in payload["files"]

    reloaded = CodeIndex.load(target)
    assert set(reloaded.files) == set(index.files)
    # The reloaded index rebuilds an identical graph without re-walking.
    assert reloaded.graph().stats() == index.graph().stats()
    assert reloaded.content_hash() == index.content_hash()


def test_refresh_picks_up_a_changed_file(mutable_repo: str):
    index = CodeIndex.build(mutable_repo)
    graph = index.graph()
    assert graph.node("pkg.util:brand_new") is None
    before = index.content_hash()

    util = os.path.join(mutable_repo, "pkg", "util.py")
    with open(util, "a") as handle:
        handle.write('\n\ndef brand_new(value):\n    """Added later."""\n    return value\n')
    _touch_later(util)

    report = index.refresh()
    assert report.changed == ["pkg/util.py"]
    assert report.added == [] and report.removed == []
    assert report.unchanged >= 3
    assert index.content_hash() != before

    refreshed = index.graph()
    node = refreshed.node("pkg.util:brand_new")
    assert node is not None and node.doc == "Added later."


def test_refresh_recomputes_cross_file_edges(mutable_repo: str):
    """A change in one file must be able to create an edge in another."""
    index = CodeIndex.build(mutable_repo)
    assert "pkg.util:_shout" not in index.graph().callees.get("app.main:main", [])

    main = os.path.join(mutable_repo, "app", "main.py")
    source = open(main).read().replace(
        "from pkg.core import Widget, build",
        "from pkg.core import Widget, build\nfrom pkg.util import _shout",
    ).replace("return widget.loud()", "return _shout(widget.loud())")
    with open(main, "w") as handle:
        handle.write(source)
    _touch_later(main)

    report = index.refresh()
    assert report.changed == ["app/main.py"]
    graph = index.graph()
    assert "pkg.util:_shout" in graph.callees["app.main:main"]
    assert "app.main:main" in graph.callers["pkg.util:_shout"]


def test_refresh_detects_added_and_removed_files(mutable_repo: str):
    index = CodeIndex.build(mutable_repo)

    extra = os.path.join(mutable_repo, "pkg", "extra.py")
    with open(extra, "w") as handle:
        handle.write('"""Extra."""\n\n\ndef extra():\n    return 1\n')
    report = index.refresh()
    assert report.added == ["pkg/extra.py"]
    assert index.graph().node("pkg.extra:extra") is not None

    os.remove(extra)
    report = index.refresh()
    assert report.removed == ["pkg/extra.py"]
    assert index.graph().node("pkg.extra:extra") is None


def test_unchanged_files_are_not_reanalyzed(mutable_repo: str):
    index = CodeIndex.build(mutable_repo)
    identities = {rel: id(fa) for rel, fa in index.files.items()}
    report = index.refresh()
    assert not report.dirty
    assert report.unchanged == len(index.files)
    # Same objects: nothing was re-parsed.
    assert {rel: id(fa) for rel, fa in index.files.items()} == identities


def test_restat_without_content_change_is_not_a_change(mutable_repo: str):
    index = CodeIndex.build(mutable_repo)
    util = os.path.join(mutable_repo, "pkg", "util.py")
    source = open(util).read()
    time.sleep(0.01)
    with open(util, "w") as handle:  # rewrite identical bytes
        handle.write(source)
    _touch_later(util)

    report = index.refresh()
    assert report.changed == []
    assert report.unchanged == len(index.files)


def test_load_or_build_falls_back_when_the_index_is_corrupt(mutable_repo: str, tmp_path):
    target = str(tmp_path / "broken.json")
    with open(target, "w") as handle:
        handle.write("{not json")
    index = CodeIndex.load_or_build(mutable_repo, index_path=target)
    assert index.graph().node("pkg.core:Widget") is not None
