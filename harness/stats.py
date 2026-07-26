"""Small-sample statistics for pass-rate comparisons.

A bare pass rate over 30-50 tasks is a noisy number: 24/30 and 26/30 differ by
7 percentage points and are statistically indistinguishable. Two things help:

* `wilson_interval` gives an honest confidence interval on a single run's pass
  rate, so the reader sees the width of the noise floor next to the point
  estimate.
* `mcnemar_exact` compares two runs *pairwise* over the same tasks. Because the
  task set is fixed, the tasks that flipped are far more informative than the
  totals; this is the right test for that design and is what the diff reports.
"""

from __future__ import annotations

import math


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact(regressed: int, improved: int) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail outcomes.

    `regressed` and `improved` are the discordant counts: tasks that went
    pass->fail and fail->pass. Concordant tasks carry no information about the
    direction of change and are deliberately ignored.
    """
    n = regressed + improved
    if n == 0:
        return 1.0
    k = min(regressed, improved)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2**n))


def min_detectable_flips(total: int, alpha: float = 0.05) -> int:
    """Smallest one-sided flip count that would be significant at `alpha`.

    Reported by the diff so the reader knows how many task flips this task set
    can actually resolve. With 30 tasks and no improvements, 6 regressions are
    needed before the aggregate move is distinguishable from noise.
    """
    for flips in range(1, max(total, 1) + 1):
        if mcnemar_exact(flips, 0) <= alpha:
            return flips
    return max(total, 1)
