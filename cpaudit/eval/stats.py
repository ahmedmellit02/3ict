"""Statistical reporting -- one implementation, used everywhere.

Three things this module exists to get right, each of which was wrong somewhere
in the repository before it existed:

1. **Intervals use Student's t, not the normal quantile.** With 8 seeds the
   correct multiplier is t_7 = 2.365, not z = 1.96; using z understates every
   interval by 17%. `ci` takes the sample and its own dof.

2. **Families of comparisons are corrected.** Comparing one method against k
   baselines is k tests, and an uncorrected p = 0.03 among seven of them is not
   evidence at the 5% level. `holm` implements Holm-Bonferroni step-down, which
   is uniformly more powerful than Bonferroni and needs no independence
   assumption.

3. **A parametric test on a handful of seeds is reported beside a rank test.**
   `paired` returns both. Note the Wilcoxon floor: with n paired observations
   the smallest attainable two-sided p is 2/2^n (0.0078 at n=8), so a
   "p < 0.001" from a t-test on 8 seeds cannot be corroborated non-parametrically
   at all. Seeds are re-splits of one fixed sample, so these p-values describe
   variability over splits, not over datasets or patients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Paired:
    """One paired comparison, reported honestly."""
    name: str
    delta: float          # mean(a) - mean(b); negative = `a` has smaller sets
    ci_lo: float          # t-interval on the paired difference
    ci_hi: float
    p_t: float            # paired t-test
    p_w: float            # Wilcoxon signed-rank
    n: int
    p_holm: float = float("nan")   # filled in by holm()

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.p_t):
            return "n/a (need >=2 seeds)"
        p = self.p_holm if np.isfinite(self.p_holm) else self.p_t
        if p >= 0.05:
            return "tie"
        return "smaller" if self.delta < 0 else "LARGER"


def ci(x, level: float = 0.95) -> tuple[float, float]:
    """Two-sided t-interval for the mean of `x`. Falls back gracefully at n<2."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(x[0]), float(x[0]))
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.5 + level / 2, n - 1)
    return float(m - h), float(m + h)


def half_width(x, level: float = 0.95) -> float:
    """Half-width of the t-interval — what an error bar should be."""
    lo, hi = ci(x, level)
    return float((hi - lo) / 2) if np.isfinite(lo) else 0.0


def paired(a, b, name: str = "") -> Paired:
    """Paired comparison of `a` against `b` (matched by position, i.e. by seed)."""
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), f"unpaired lengths {len(a)} vs {len(b)}"
    d = a - b
    n = len(d)
    if n < 2 or np.allclose(d, 0):
        lo, hi = ci(d)
        return Paired(name, float(d.mean()), lo, hi, float("nan"), float("nan"), n)
    p_t = float(stats.ttest_rel(a, b).pvalue)
    try:
        p_w = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:                       # all differences zero
        p_w = float("nan")
    lo, hi = ci(d)
    return Paired(name, float(d.mean()), lo, hi, p_t, p_w, n)


def holm(results: list[Paired]) -> list[Paired]:
    """Holm-Bonferroni step-down over a family, writing `p_holm` in place.

    Sorts by raw p, multiplies the k-th smallest by (m-k), and enforces
    monotonicity so an adjusted p is never below one that precedes it.
    """
    finite = [r for r in results if np.isfinite(r.p_t)]
    m = len(finite)
    running = 0.0
    for rank, r in enumerate(sorted(finite, key=lambda r: r.p_t)):
        running = max(running, min(1.0, (m - rank) * r.p_t))
        r.p_holm = running
    return results


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p the signed-rank test can produce with n pairs."""
    return 2.0 / (2 ** n) if n > 0 else float("nan")


def table(results: list[Paired], label: str = "") -> str:
    """Render a family of comparisons as a fixed-width block."""
    holm(results)
    n = results[0].n if results else 0
    out = [f"\n  {label} (n={n} seeds; Wilcoxon floor p={wilcoxon_floor(n):.4f})",
           f"    {'vs':<14}{'delta':>8}{'95% CI':>18}{'t':>9}{'Holm':>9}{'Wilcox':>9}  verdict"]
    for r in results:
        out.append(f"    {r.name:<14}{r.delta:>+8.3f}"
                   f"{f'[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]':>18}"
                   f"{r.p_t:>9.4f}{r.p_holm:>9.4f}{r.p_w:>9.4f}  {r.verdict}")
    return "\n".join(out)
