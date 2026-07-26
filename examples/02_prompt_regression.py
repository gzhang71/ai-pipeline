#!/usr/bin/env python
"""Does prompt v2 actually beat prompt v1?

Runs two versions of a real prompt over the same task set, then diffs them.
The point of the example is the last section: the aggregate pass rate is
identical between the two versions, and there is still a regression hiding
underneath it.

Runs offline against a recorded fixture. No API key, no network.

    .venv/bin/python examples/02_prompt_regression.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from harness import (
    DEFAULT_JUDGE_ID,
    Judge,
    JudgeCache,
    default_prompt_dir,
    default_task_dir,
    diff_files,
    format_report,
    load_prompts,
    load_tasks,
    min_detectable_flips,
    run,
    wilson_interval,
)
from harness import fakes

OUT = Path("runs/example-02")


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------- what we have
rule("1. Versioned prompts and a task set, both on disk")

prompts = load_prompts(default_prompt_dir())
tasks = list(load_tasks(default_task_dir()).values())

for pid in sorted(prompts):
    p = prompts[pid]
    print(f"  {pid:22} hash={p.short_hash}")
print(f"\n{len(tasks)} tasks:")
for t in tasks:
    kinds = ", ".join(sorted({spec['type'] for spec in t.assertions}))
    print(f"  {t.id:20} {len(t.assertions)} assertion(s): {kinds}")

print(
    "\nIdentity is the content hash, not the filename. Edit a prompt by one"
    "\ncharacter and it becomes a different prompt with no version to bump."
)


# ------------------------------------------------------------------- run both
rule("2. Run both versions over the same tasks")

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

fixture = fakes.load_fixture()
cache = JudgeCache(OUT / "judge-cache.json")
paths = []

for pid in ("triage.v1", "triage.v2"):
    prompt = prompts[pid]
    path = OUT / f"{pid}.jsonl"
    judge = Judge(client=fakes.judge_client(fixture), prompt=prompts[DEFAULT_JUDGE_ID], cache=cache)
    summary = run(
        prompt=prompt,
        tasks=tasks,
        client=fakes.task_client(fixture),
        out_path=path,
        judge=judge,
        concurrency=4,
    )
    lo, hi = wilson_interval(summary.passed, summary.total)
    print(
        f"  {pid}@{prompt.short_hash}: {summary.passed}/{summary.total} passed "
        f"(95% CI {lo:.0%}-{hi:.0%})"
    )
    paths.append(path)

print(f"\njudge cache: {cache.hits} hit(s), {cache.misses} miss(es)")
print(
    "The judge is itself a versioned prompt, and its hash is recorded on every"
    "\nverdict. An unversioned judge silently invalidates historical comparisons."
)


# ----------------------------------------------------------------- the diff
rule("3. The diff -- regressions first, aggregates last")

report = diff_files(paths[0], paths[1])
print(format_report(report))


# ------------------------------------------------------------- the noise floor
rule("4. Can this task set even detect what it just reported?")

n = len(tasks)
need = min_detectable_flips(n)
print(f"tasks: {n}")
print(f"one-directional flips needed for p<0.05: {need}")
print(
    f"\nWith {n} tasks, fewer than {need} flips in one direction cannot reach"
    "\nsignificance at any sample size. That is a property of the test, not of"
    "\nthe prompts. A pass-rate delta smaller than that is not evidence -- which"
    "\nis exactly why the diff above leads with the named regression rather than"
    "\nwith the aggregate."
)
print(f"\n(run records in {OUT})")
