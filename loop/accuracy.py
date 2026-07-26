"""Accuracy vs. context length: the headline analysis.

Given runs over a task set with checkable outcomes, bin the runs by prompt
length, compute success rate per bin with a Wilson confidence interval, and
report where the curve bends.

WHAT THIS PROVES AND WHAT IT DOES NOT
-------------------------------------
It measures *association*, not causation. Longer prompts in a real task set
are usually also harder prompts, so a drop in success rate at high token
counts may be difficulty, not degradation. To make the claim causal you need
the same task at several lengths -- pad the context with irrelevant filler and
hold the question fixed. The synthetic set in ``loop.tasks`` is built that way
precisely so this path is exercised honestly; the bend it finds there is
attributable to length because nothing else varies.

The "bend point" is the largest single drop in success rate between adjacent
bins. That is a coarse estimator: with few runs per bin the drop is dominated
by noise, so ``AccuracyReport.significant`` is False whenever the two bins'
Wilson intervals overlap, and the caveats list says so in words.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .schema import group_runs

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Observation:
    """One run's outcome, keyed to the prompt length it ran at."""

    run_id: str
    task_id: str
    prompt_tokens: int
    success: bool
    turns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Bin:
    lo: int
    hi: int
    n: int
    successes: int

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.n)

    def to_record(self) -> dict[str, Any]:
        low, high = self.interval
        return {
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "successes": self.successes,
            "rate": round(self.rate, 4) if self.n else None,
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
        }


@dataclass
class AccuracyReport:
    bins: list[Bin]
    n: int
    overall_rate: float
    bend_tokens: int | None
    bend_drop: float
    bend_from_rate: float
    bend_to_rate: float
    significant: bool
    caveats: list[str]

    def to_record(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "overall_rate": round(self.overall_rate, 4) if self.n else None,
            "bins": [b.to_record() for b in self.bins],
            "bend_tokens": self.bend_tokens,
            "bend_drop": round(self.bend_drop, 4),
            "bend_from_rate": round(self.bend_from_rate, 4),
            "bend_to_rate": round(self.bend_to_rate, 4),
            "significant": self.significant,
            "caveats": list(self.caveats),
        }


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval. Well-behaved at 0/n and n/n, unlike the normal
    approximation, which is why we use it with the tiny samples that this kind
    of eval usually has."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denominator = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def _quantile_edges(values: Sequence[int], n_bins: int) -> list[int]:
    ordered = sorted(values)
    edges = [ordered[0]]
    for index in range(1, n_bins):
        position = min(len(ordered) - 1, int(round(index * len(ordered) / n_bins)))
        edges.append(ordered[position])
    edges.append(ordered[-1] + 1)
    # Collapse duplicates so heavily-tied lengths do not produce empty bins.
    deduped = [edges[0]]
    for edge in edges[1:]:
        if edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def _linear_edges(values: Sequence[int], n_bins: int) -> list[int]:
    low, high = min(values), max(values) + 1
    if high - low < n_bins:
        return list(range(low, high + 1))
    width = (high - low) / n_bins
    return [int(low + width * i) for i in range(n_bins)] + [high]


def analyze_accuracy(
    observations: Iterable[Observation],
    *,
    n_bins: int = 5,
    binning: str = "quantile",
    min_bin: int = 2,
) -> AccuracyReport:
    """Bin runs by prompt length and locate the bend in the success curve.

    ``min_bin`` is the smallest bin population that may participate in bend
    detection; smaller bins are still reported, just not trusted.
    """
    observations = list(observations)
    caveats: list[str] = []
    if not observations:
        return AccuracyReport([], 0, float("nan"), None, 0.0, 0.0, 0.0, False,
                              ["no observations"])

    lengths = [o.prompt_tokens for o in observations]
    if binning == "quantile":
        edges = _quantile_edges(lengths, n_bins)
    elif binning == "linear":
        edges = _linear_edges(lengths, n_bins)
    else:
        raise ValueError(f"unknown binning {binning!r}")

    bins: list[Bin] = []
    for lo, hi in zip(edges, edges[1:]):
        members = [o for o in observations if lo <= o.prompt_tokens < hi]
        bins.append(Bin(lo=lo, hi=hi, n=len(members), successes=sum(1 for o in members if o.success)))

    populated = [b for b in bins if b.n]
    overall = sum(o.success for o in observations) / len(observations)

    bend_tokens: int | None = None
    bend_drop = 0.0
    bend_from = bend_to = 0.0
    significant = False
    usable = [b for b in populated if b.n >= min_bin]
    for left, right in zip(usable, usable[1:]):
        drop = left.rate - right.rate
        if drop > bend_drop:
            bend_drop = drop
            bend_tokens = right.lo
            bend_from, bend_to = left.rate, right.rate
            left_low, left_high = left.interval
            right_low, right_high = right.interval
            significant = right_high < left_low

    if len(usable) < 2:
        caveats.append(
            f"fewer than two bins reached min_bin={min_bin}; no bend can be estimated"
        )
    if bend_tokens is None and len(usable) >= 2:
        caveats.append("success rate is flat or rising across bins; no bend detected")
    if bend_tokens is not None and not significant:
        caveats.append(
            "the drop at the bend is not significant at 95% -- the adjacent "
            "bins' Wilson intervals overlap; treat it as a hint, not a finding"
        )
    # Several *tasks* at several lengths confounds length with difficulty.
    # The same task family at several lengths does not -- that is the design
    # the synthetic set uses deliberately, so honour `metadata["family"]`
    # when the caller supplies it.
    families = {o.metadata.get("family", o.task_id) for o in observations}
    if len(families) > 1:
        caveats.append(
            "observations mix several task families, so length is confounded "
            "with difficulty; hold the task fixed and vary padding to isolate "
            "length"
        )
    if len(observations) < 20:
        caveats.append(f"only {len(observations)} observations; this is anecdote-scale")

    return AccuracyReport(
        bins=bins,
        n=len(observations),
        overall_rate=overall,
        bend_tokens=bend_tokens,
        bend_drop=bend_drop,
        bend_from_rate=bend_from,
        bend_to_rate=bend_to,
        significant=significant,
        caveats=caveats,
    )


def observations_from_records(
    records: Iterable[dict[str, Any]],
    outcomes: dict[str, bool],
    *,
    length: str = "peak",
) -> list[Observation]:
    """Build observations from a JSONL record stream plus a run_id -> success map.

    ``length`` picks which prompt length characterizes the run: ``peak`` (the
    largest prompt the model actually saw, the usual choice), ``first`` (the
    opening prompt) or ``total`` (summed across turns).
    """
    if length not in ("peak", "first", "total"):
        raise ValueError(f"unknown length selector {length!r}")
    observations: list[Observation] = []
    for run in group_runs(records):
        header, turns, footer = run["header"], run["turns"], run["footer"]
        run_id = header["run_id"]
        if run_id not in outcomes:
            continue
        totals = (footer or {}).get("totals", {})
        if length == "peak":
            tokens = totals.get("peak_prompt_tokens", 0)
        elif length == "first":
            tokens = turns[0]["prompt_tokens"]["counted_total"] if turns else 0
        else:  # "total"
            tokens = totals.get("prompt_tokens_total", 0)
        task = header.get("task") or {}
        observations.append(
            Observation(
                run_id=run_id,
                task_id=task.get("task_id", "unknown"),
                prompt_tokens=int(tokens),
                success=bool(outcomes[run_id]),
                turns=(footer or {}).get("turns", len(turns)),
                metadata=task,
            )
        )
    return observations


def report_text(report: AccuracyReport, *, width: int = 28) -> str:
    """Plain-text rendering of the accuracy-vs-length curve."""
    lines = [
        "accuracy vs. prompt length",
        "=" * 58,
        f"observations: {report.n}   overall success: "
        + (f"{report.overall_rate:.0%}" if report.n else "n/a"),
        "",
        f"{'prompt tokens':>20}  {'n':>3}  {'rate':>5}  curve",
    ]
    for bucket in report.bins:
        label = f"{bucket.lo:,}-{bucket.hi - 1:,}"
        if not bucket.n:
            lines.append(f"{label:>20}  {0:>3}  {'--':>5}  (empty)")
            continue
        filled = int(round(bucket.rate * width))
        low, high = bucket.interval
        bar = "#" * filled + "." * (width - filled)
        lines.append(
            f"{label:>20}  {bucket.n:>3}  {bucket.rate:>5.0%}  {bar} "
            f"[{low:.0%}-{high:.0%}]"
        )
    lines.append("")
    if report.bend_tokens is None:
        lines.append("bend: none detected")
    else:
        mark = "significant" if report.significant else "NOT significant at 95%"
        lines.append(
            f"bend: ~{report.bend_tokens:,} tokens "
            f"({report.bend_from_rate:.0%} -> {report.bend_to_rate:.0%}, "
            f"drop {report.bend_drop:.0%}, {mark})"
        )
    if report.caveats:
        lines.append("")
        lines.append("caveats:")
        lines.extend(f"  - {c}" for c in report.caveats)
    return "\n".join(lines)
