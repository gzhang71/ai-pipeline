"""A cheap on-disk index so the graph can be reloaded without re-walking.

The index stores the *unresolved* per-file analysis (nodes, import refs, call
sites) plus a stat/hash stamp per file. Cross-file resolution is redone on load,
which is microseconds of work and keeps incremental refresh correct: changing
one file can create or destroy call edges in other files, so edges must never be
cached across a refresh.

Format is plain JSON (stdlib only). It is written with sorted keys so the file
is stable under version control and diffs are readable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable

from .builder import CodeGraph, FileAnalysis, analyze_file, module_name_for

INDEX_VERSION = 1
DEFAULT_INDEX_NAME = ".code_graph_index.json"

DEFAULT_EXCLUDES = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    ".tox",
    ".eggs",
)


@dataclass
class RefreshReport:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0

    @property
    def dirty(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"RefreshReport(added={self.added}, changed={self.changed}, "
            f"removed={self.removed}, unchanged={self.unchanged})"
        )


def discover_python_files(
    root: str, *, excludes: Iterable[str] = DEFAULT_EXCLUDES
) -> list[str]:
    """Repo-relative posix paths of every .py file under `root`."""
    exclude = set(excludes)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in exclude and not d.startswith(".")
        )
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            found.append(rel.replace(os.sep, "/"))
    return found


class CodeIndex:
    """Load / build / refresh / persist the per-file analysis for a repo."""

    def __init__(
        self,
        root: str,
        files: dict[str, FileAnalysis] | None = None,
        *,
        excludes: Iterable[str] = DEFAULT_EXCLUDES,
    ) -> None:
        self.root = os.path.abspath(root)
        self.files: dict[str, FileAnalysis] = files or {}
        self.excludes = tuple(excludes)

    # -- construction ----------------------------------------------------
    @classmethod
    def build(
        cls, root: str, *, excludes: Iterable[str] = DEFAULT_EXCLUDES
    ) -> "CodeIndex":
        index = cls(root, {}, excludes=excludes)
        index.refresh()
        return index

    @classmethod
    def load(cls, path: str) -> "CodeIndex":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("version") != INDEX_VERSION:
            raise ValueError(
                f"index version {data.get('version')!r} != {INDEX_VERSION!r}"
            )
        files = {
            rel: FileAnalysis.from_dict(payload)
            for rel, payload in data["files"].items()
        }
        return cls(data["root"], files, excludes=tuple(data.get("excludes", DEFAULT_EXCLUDES)))

    @classmethod
    def load_or_build(
        cls, root: str, *, index_path: str | None = None, refresh: bool = True
    ) -> "CodeIndex":
        path = index_path or os.path.join(os.path.abspath(root), DEFAULT_INDEX_NAME)
        if os.path.exists(path):
            try:
                index = cls.load(path)
                index.root = os.path.abspath(root)
                if refresh:
                    index.refresh()
                return index
            except (ValueError, KeyError, json.JSONDecodeError):
                pass  # stale or corrupt index -- rebuild from scratch
        return cls.build(root)

    # -- persistence -----------------------------------------------------
    def save(self, path: str | None = None) -> str:
        target = path or os.path.join(self.root, DEFAULT_INDEX_NAME)
        payload = {
            "version": INDEX_VERSION,
            "root": self.root,
            "excludes": list(self.excludes),
            "files": {rel: fa.to_dict() for rel, fa in sorted(self.files.items())},
        }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=1)
            handle.write("\n")
        return target

    # -- incremental refresh ---------------------------------------------
    def refresh(self, paths: Iterable[str] | None = None) -> RefreshReport:
        """Re-analyze only files whose stamp changed. Returns what moved."""
        report = RefreshReport()
        if paths is None:
            present = discover_python_files(self.root, excludes=self.excludes)
        else:
            present = [p.replace(os.sep, "/") for p in paths]

        for rel in present:
            abs_path = os.path.join(self.root, rel)
            if not os.path.exists(abs_path):
                continue
            cached = self.files.get(rel)
            if cached is not None and not self._stale(cached, abs_path):
                report.unchanged += 1
                continue
            analysis = analyze_file(
                abs_path, root=self.root, module=module_name_for(rel)
            )
            if cached is None:
                report.added.append(rel)
            elif cached.sha256 == analysis.sha256:
                # mtime/size moved but content did not -- just restamp.
                cached.mtime = analysis.mtime
                cached.size = analysis.size
                report.unchanged += 1
                continue
            else:
                report.changed.append(rel)
            self.files[rel] = analysis

        if paths is None:
            for rel in sorted(set(self.files) - set(present)):
                del self.files[rel]
                report.removed.append(rel)
        return report

    @staticmethod
    def _stale(cached: FileAnalysis, abs_path: str) -> bool:
        stat = os.stat(abs_path)
        return cached.size != stat.st_size or cached.mtime != stat.st_mtime

    def content_hash(self) -> str:
        """Stable digest of the indexed content -- handy for cache keys."""
        digest = hashlib.sha256()
        for rel, analysis in sorted(self.files.items()):
            digest.update(rel.encode())
            digest.update(analysis.sha256.encode())
        return digest.hexdigest()

    # -- graph -----------------------------------------------------------
    def graph(self) -> CodeGraph:
        return CodeGraph(self.root, self.files)


def load_graph(root: str, *, index_path: str | None = None) -> CodeGraph:
    """Build the graph lazily at run time (never bake a snapshot into tests)."""
    return CodeIndex.load_or_build(root, index_path=index_path).graph()
