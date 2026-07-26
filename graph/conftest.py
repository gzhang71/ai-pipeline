"""Shared fixtures: the synthetic sample repo under `graph/fixtures/`."""

from __future__ import annotations

import os
import shutil

import pytest

from graph.builder import CodeGraph
from graph.index import CodeIndex
from graph.tools import GraphTools

FIXTURE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SAMPLE_REPO = os.path.join(FIXTURE_ROOT, "sample_repo")


@pytest.fixture(scope="session")
def sample_repo() -> str:
    """Path to the read-only synthetic repo."""
    assert os.path.isdir(SAMPLE_REPO), "fixture repo is missing"
    return SAMPLE_REPO


@pytest.fixture
def mutable_repo(tmp_path) -> str:
    """A throwaway copy of the sample repo, safe to edit in a test."""
    target = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, target)
    return str(target)


@pytest.fixture(scope="session")
def sample_index(sample_repo: str) -> CodeIndex:
    return CodeIndex.build(sample_repo)


@pytest.fixture(scope="session")
def sample_graph(sample_index: CodeIndex) -> CodeGraph:
    return sample_index.graph()


@pytest.fixture(scope="session")
def sample_tools(sample_graph: CodeGraph) -> GraphTools:
    return GraphTools(sample_graph)


LARGE_MODULES = 60
LARGE_FUNCS_PER_MODULE = 12
LARGE_CALLERS = 24  # modules whose first function calls big.util:shared_helper


@pytest.fixture(scope="session")
def large_repo(tmp_path_factory) -> str:
    """A generated corpus big enough for the scaling claim to be measurable.

    The 5-file `sample_repo` is *smaller* than the JIT arm's fixed overhead
    (tool schemas + index), so on it stuffing always wins. That crossover is
    real and documented; this fixture exists so the other side of it can be
    tested too, deterministically and without touching sibling packages.
    """
    root = tmp_path_factory.mktemp("large_repo")
    pkg = root / "big"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""Generated corpus."""\n')
    (pkg / "util.py").write_text(
        '"""Shared utilities."""\n'
        "\n"
        "TUNING_CONSTANT = 8675309\n"
        "\n"
        "\n"
        "def shared_helper(value):\n"
        '    """Normalize a value. Called from many modules."""\n'
        "    return str(value).strip().lower()\n"
    )
    for m in range(LARGE_MODULES):
        body = [f'"""Generated module {m}."""', ""]
        if m < LARGE_CALLERS:
            body += ["from big.util import shared_helper", ""]
        for f in range(LARGE_FUNCS_PER_MODULE):
            body += [
                "",
                f"def op_{m}_{f}(payload, factor=1):",
                f'    """Operation {f} of module {m}."""',
                f"    total = len(payload) * factor + {m * 100 + f}",
                "    for index, item in enumerate(payload):",
                "        total += index * len(str(item))",
            ]
            if m < LARGE_CALLERS and f == 0:
                body.append("    total += len(shared_helper(payload))")
            body.append(f"    return total, 'op_{m}_{f}'")
        (pkg / f"mod_{m:02d}.py").write_text("\n".join(body) + "\n")
    return str(root)
