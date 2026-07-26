"""Diff two runs: what regressed, what improved, what it cost.

The deliverable is "prompt v1 vs v2 over the same task set". Regressions are
surfaced first and in full detail, because a change that trades three wins for
three losses shows a flat pass rate and is not a neutral change -- it is two
different prompts that happen to score the same.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runner import RunRecord, TaskResult, load_run
from .stats import mcnemar_exact, min_detectable_flips, wilson_interval

COST_PER_MTOK = {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25}


@dataclass(frozen=True)
class TaskDelta:
    task_id: str
    before: bool
    after: bool
    kind: str  # regressed | improved | unchanged_pass | unchanged_fail
    newly_failing: tuple[str, ...] = ()
    newly_passing: tuple[str, ...] = ()
    detail: str = ""
    token_delta: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "before": self.before,
            "after": self.after,
            "kind": self.kind,
            "newly_failing": list(self.newly_failing),
            "newly_passing": list(self.newly_passing),
            "detail": self.detail,
            "token_delta": self.token_delta,
        }


@dataclass
class DiffReport:
    before: RunRecord
    after: RunRecord
    deltas: list[TaskDelta]
    only_in_before: list[str] = field(default_factory=list)
    only_in_after: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # --- aggregates ---
    @property
    def regressions(self) -> list[TaskDelta]:
        return [d for d in self.deltas if d.kind == "regressed"]

    @property
    def improvements(self) -> list[TaskDelta]:
        return [d for d in self.deltas if d.kind == "improved"]

    @property
    def unchanged_pass(self) -> list[TaskDelta]:
        return [d for d in self.deltas if d.kind == "unchanged_pass"]

    @property
    def unchanged_fail(self) -> list[TaskDelta]:
        return [d for d in self.deltas if d.kind == "unchanged_fail"]

    @property
    def compared(self) -> int:
        return len(self.deltas)

    @property
    def before_passed(self) -> int:
        return sum(1 for d in self.deltas if d.before)

    @property
    def after_passed(self) -> int:
        return sum(1 for d in self.deltas if d.after)

    @property
    def before_rate(self) -> float:
        return self.before_passed / self.compared if self.compared else 0.0

    @property
    def after_rate(self) -> float:
        return self.after_passed / self.compared if self.compared else 0.0

    @property
    def rate_delta(self) -> float:
        return self.after_rate - self.before_rate

    @property
    def p_value(self) -> float:
        return mcnemar_exact(len(self.regressions), len(self.improvements))

    @property
    def significant(self) -> bool:
        return self.p_value <= 0.05

    def usage(self, which: str) -> dict[str, int]:
        record = self.before if which == "before" else self.after
        ids = {d.task_id for d in self.deltas}
        totals: dict[str, int] = {}
        for task_id in ids:
            result = record.results.get(task_id)
            if result is None:
                continue
            for source in (result.usage, result.judge_usage):
                for key, value in (source or {}).items():
                    totals[key] = totals.get(key, 0) + int(value or 0)
        return totals

    def token_delta(self) -> dict[str, int]:
        before, after = self.usage("before"), self.usage("after")
        keys = set(before) | set(after)
        return {key: after.get(key, 0) - before.get(key, 0) for key in sorted(keys)}

    def estimated_cost(self, which: str) -> float:
        usage = self.usage(which)
        return (
            usage.get("input_tokens", 0) * COST_PER_MTOK["input"]
            + usage.get("output_tokens", 0) * COST_PER_MTOK["output"]
            + usage.get("cache_read_input_tokens", 0) * COST_PER_MTOK["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * COST_PER_MTOK["cache_write"]
        ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": {
                "run_id": self.before.run_id,
                "prompt_id": self.before.meta.get("prompt_id"),
                "prompt_hash": self.before.meta.get("prompt_hash"),
                "passed": self.before_passed,
                "pass_rate": self.before_rate,
                "usage": self.usage("before"),
            },
            "after": {
                "run_id": self.after.run_id,
                "prompt_id": self.after.meta.get("prompt_id"),
                "prompt_hash": self.after.meta.get("prompt_hash"),
                "passed": self.after_passed,
                "pass_rate": self.after_rate,
                "usage": self.usage("after"),
            },
            "compared": self.compared,
            "regressions": [d.to_dict() for d in self.regressions],
            "improvements": [d.to_dict() for d in self.improvements],
            "unchanged_pass": [d.task_id for d in self.unchanged_pass],
            "unchanged_fail": [d.task_id for d in self.unchanged_fail],
            "only_in_before": self.only_in_before,
            "only_in_after": self.only_in_after,
            "rate_delta": self.rate_delta,
            "p_value": self.p_value,
            "significant": self.significant,
            "token_delta": self.token_delta(),
            "estimated_cost_usd": {
                "before": round(self.estimated_cost("before"), 6),
                "after": round(self.estimated_cost("after"), 6),
            },
            "warnings": self.warnings,
        }


def diff_runs(before: RunRecord, after: RunRecord) -> DiffReport:
    warnings: list[str] = []
    before_meta, after_meta = before.meta, after.meta
    if (
        before_meta.get("task_set_hash")
        and after_meta.get("task_set_hash")
        and before_meta["task_set_hash"] != after_meta["task_set_hash"]
    ):
        warnings.append(
            "task set differs between runs: the task files changed, so pass-rate "
            "deltas mix prompt effects with task-set effects"
        )
    if before_meta.get("prompt_hash") == after_meta.get("prompt_hash"):
        warnings.append(
            "both runs used the same prompt hash: any difference here is sampling "
            "noise, not a prompt effect"
        )
    if before_meta.get("judge_hash") != after_meta.get("judge_hash"):
        warnings.append(
            "judge prompt differs between runs: judged assertions are not comparable"
        )
    if before_meta.get("model") != after_meta.get("model"):
        warnings.append(
            f"model differs: {before_meta.get('model')} vs {after_meta.get('model')}"
        )

    shared = [t for t in before.results if t in after.results]
    deltas = [_delta(before.results[t], after.results[t]) for t in shared]
    only_before = sorted(set(before.results) - set(after.results))
    only_after = sorted(set(after.results) - set(before.results))
    if only_before or only_after:
        warnings.append(
            f"{len(only_before)} task(s) only in the before run, "
            f"{len(only_after)} only in the after run; comparison covers the "
            f"{len(shared)} tasks present in both"
        )
    incomplete = [
        t
        for t in shared
        if before.results[t].status == "incomplete" or after.results[t].status == "incomplete"
    ]
    if incomplete:
        warnings.append(
            f"{len(incomplete)} task(s) have unevaluated (skipped) assertions and are "
            "counted as failures: " + ", ".join(sorted(incomplete)[:5])
        )
    return DiffReport(
        before=before,
        after=after,
        deltas=deltas,
        only_in_before=only_before,
        only_in_after=only_after,
        warnings=warnings,
    )


def diff_files(before_path: Path, after_path: Path) -> DiffReport:
    return diff_runs(load_run(before_path), load_run(after_path))


def _delta(before: TaskResult, after: TaskResult) -> TaskDelta:
    if before.passed and not after.passed:
        kind = "regressed"
    elif not before.passed and after.passed:
        kind = "improved"
    elif before.passed:
        kind = "unchanged_pass"
    else:
        kind = "unchanged_fail"

    before_failed = {a.id for a in before.assertions if not a.passed}
    after_failed = {a.id for a in after.assertions if not a.passed}
    newly_failing = tuple(sorted(after_failed - before_failed))
    newly_passing = tuple(sorted(before_failed - after_failed))

    detail_parts = [
        f"{a.id}: {a.detail}"
        for a in after.assertions
        if not a.passed and a.id in newly_failing and a.detail
    ]
    if after.error and not before.error:
        detail_parts.insert(0, f"model error: {after.error}")
    return TaskDelta(
        task_id=after.task_id,
        before=before.passed,
        after=after.passed,
        kind=kind,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
        detail="; ".join(detail_parts),
        token_delta=_tokens(after) - _tokens(before),
    )


def _tokens(result: TaskResult) -> int:
    total = 0
    for source in (result.usage, result.judge_usage):
        source = source or {}
        total += int(source.get("total_prompt_tokens", 0) or 0)
        total += int(source.get("output_tokens", 0) or 0)
    return total


def format_report(report: DiffReport, *, verbose: bool = False) -> str:
    """Human-readable diff. Regressions first, always."""
    lines: list[str] = []
    before_label = report.before.prompt_label
    after_label = report.after.prompt_label
    lines.append(f"before  {before_label}  ({report.before.path})")
    lines.append(f"after   {after_label}  ({report.after.path})")
    lines.append("")

    if report.regressions:
        lines.append(f"REGRESSIONS ({len(report.regressions)}) -- pass -> fail")
        for delta in report.regressions:
            lines.append(f"  - {delta.task_id}")
            for assertion_id in delta.newly_failing:
                lines.append(f"      broke: {assertion_id}")
            if delta.detail:
                lines.append(f"      {_clip(delta.detail, 160)}")
    else:
        lines.append("REGRESSIONS (0) -- nothing that passed before now fails")
    lines.append("")

    if report.improvements:
        lines.append(f"improvements ({len(report.improvements)}) -- fail -> pass")
        for delta in report.improvements:
            fixed = ", ".join(delta.newly_passing) or "-"
            lines.append(f"  + {delta.task_id}  (fixed: {fixed})")
        lines.append("")

    if verbose and report.unchanged_fail:
        lines.append(f"still failing ({len(report.unchanged_fail)})")
        for delta in report.unchanged_fail:
            lines.append(f"  = {delta.task_id}")
        lines.append("")

    lo_b, hi_b = wilson_interval(report.before_passed, report.compared)
    lo_a, hi_a = wilson_interval(report.after_passed, report.compared)
    lines.append(f"pass rate  n={report.compared}")
    lines.append(
        f"  before  {report.before_passed}/{report.compared} "
        f"({report.before_rate:.0%})  95% CI [{lo_b:.0%}, {hi_b:.0%}]"
    )
    lines.append(
        f"  after   {report.after_passed}/{report.compared} "
        f"({report.after_rate:.0%})  95% CI [{lo_a:.0%}, {hi_a:.0%}]"
    )
    lines.append(
        f"  change  {report.rate_delta:+.0%}  "
        f"({len(report.improvements)} improved, {len(report.regressions)} regressed)"
    )
    needed = min_detectable_flips(report.compared)
    verdict = "significant" if report.significant else "NOT significant"
    lines.append(
        f"  paired exact test (McNemar): p={report.p_value:.3f} -- {verdict} at alpha=0.05"
    )
    lines.append(
        f"  noise floor: with n={report.compared}, a one-directional change needs "
        f"{needed}+ task flips to clear p=0.05. Read the regression list, not the rate."
    )
    lines.append("")

    delta = report.token_delta()
    total_delta = delta.get("total_prompt_tokens", 0) + delta.get("output_tokens", 0)
    lines.append("tokens (task + judge)")
    for key in ("total_prompt_tokens", "output_tokens", "cache_read_input_tokens"):
        if key in delta:
            before_val = report.usage("before").get(key, 0)
            after_val = report.usage("after").get(key, 0)
            lines.append(f"  {key:<28} {before_val:>9,} -> {after_val:>9,}  ({delta[key]:+,})")
    lines.append(f"  {'billable total':<28} {'':>9} {'':>13}  ({total_delta:+,})")
    lines.append(
        f"  est. cost                    ${report.estimated_cost('before'):.4f} -> "
        f"${report.estimated_cost('after'):.4f}"
    )

    if report.warnings:
        lines.append("")
        lines.append("warnings")
        for warning in report.warnings:
            lines.append(f"  ! {warning}")
    return "\n".join(lines)


def format_json(report: DiffReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize_runs(records: Sequence[RunRecord]) -> list[Mapping[str, Any]]:
    return [
        {
            "path": str(r.path),
            "run_id": r.run_id,
            "prompt": r.prompt_label,
            "passed": r.passed_count,
            "total": r.total,
            "pass_rate": r.pass_rate,
        }
        for r in records
    ]
