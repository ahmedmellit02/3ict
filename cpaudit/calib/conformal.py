"""Split-conformal + Mondrian calibration.

Marginal split conformal:
    k    = ceil( (n+1)(1-alpha) )
    tau* = k-th smallest true-label nonconformity score on D_cal
    guarantee: P(Y in C(X)) >= 1 - alpha   (finite-sample, marginal)
    upper band: 1 - alpha + 1/(n+1)

Mondrian: the identical rule applied separately within each group g.
    guarantee: P(Y in C(X) | G=g) >= 1 - alpha  for every g with n_g >= ~1/alpha

The analyses pass jitter=0 (`cpaudit.bench.common.JITTER`), so the threshold
is an actual calibration score and ties are kept; the `jitter` argument
exists for the sensitivity arm of `cpaudit.bench.randomization`.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float, jitter: float = 1e-6,
                       rng: np.random.Generator | None = None) -> float:
    """The finite-sample conformal threshold tau* from true-label `scores`.

    Uses k = ceil((n+1)(1-alpha)); returns +inf when k > n (n too small to
    guarantee 1-alpha), which makes the set include everything (conservative).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.inf
    if jitter:
        rng = rng or np.random.default_rng(0)
        scores = scores + rng.uniform(-jitter, jitter, size=n)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf  # not enough calibration points for this alpha
    # k-th smallest (1-indexed) => index k-1 in sorted order.
    return float(np.partition(scores, k - 1)[k - 1])


def upper_band(n: int, alpha: float) -> float:
    """Coverage upper bound 1 - alpha + 1/(n+1)."""
    return 1.0 - alpha + 1.0 / (n + 1)


def mondrian_floor(alpha: float, mult: float = 1.0) -> int:
    """Minimum n_g for which the per-group quantile is defined.

    (6) needs ceil((n_g+1)(1-alpha)) <= n_g, i.e. n_g >= (1-alpha)/alpha -- 9 at
    alpha=0.10, not 10. The previous ceil(1/alpha) was conservative by one item
    and disagreed with eq. (7) of the paper; at n_g=75 neither binds, but the
    implementation should match the stated bound.
    """
    return int(np.ceil(mult * (1.0 - alpha) / alpha))


def calibrate_marginal(cal_true_scores: np.ndarray, alpha: float,
                       jitter: float = 1e-6, seed: int = 0) -> float:
    """Single marginal threshold tau* on D_cal true-label scores."""
    rng = np.random.default_rng(seed)
    return conformal_quantile(cal_true_scores, alpha, jitter, rng)


def calibrate_mondrian(cal_true_scores: np.ndarray, cal_groups: np.ndarray,
                       alpha: float, floor_mult: float = 1.0,
                       jitter: float = 1e-6, seed: int = 0) -> dict:
    """Per-group thresholds. Returns {group -> tau*_g}. Groups below the floor
    fall back to +inf if alone, but callers should pool/flag them.
    """
    rng = np.random.default_rng(seed)
    floor = mondrian_floor(alpha, floor_mult)
    taus: dict = {}
    for g in np.unique(cal_groups):
        sc = cal_true_scores[cal_groups == g]
        if len(sc) < floor:
            # Under-populated: mark; caller decides pool/drop. Conservative default.
            taus[g] = np.inf
        else:
            taus[g] = conformal_quantile(sc, alpha, jitter, rng)
    taus["__floor__"] = floor
    return taus


def predict_sets(option_scores: np.ndarray, mask: np.ndarray, tau,
                 groups: np.ndarray | None = None,
                 argmax_fallback: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Build prediction sets C(x) = {y : s(x,y) <= tau}.

    Parameters
    ----------
    option_scores : (N, Kmax) final scores s_theta (pad = +inf)
    mask          : (N, Kmax) bool, real options
    tau           : float (marginal) OR dict group->tau (Mondrian). If dict,
                    `groups` (N,) selects each row's threshold.
    argmax_fallback : if a set is empty, include the lowest-score option.

    Returns
    -------
    sets        : (N, Kmax) bool membership
    empty_before: (N,) bool — set was empty *before* fallback (diagnostic)
    """
    scores = np.where(mask, option_scores, np.inf)
    n, kmax = scores.shape
    if isinstance(tau, dict):
        if groups is None:
            raise ValueError("Mondrian tau dict requires per-row groups")
        tau_row = np.array([tau.get(g, np.inf) for g in groups], dtype=np.float64)
        thr = tau_row[:, None]
    else:
        thr = float(tau)
    sets = scores <= thr
    empty_before = ~sets.any(axis=1)
    if argmax_fallback and empty_before.any():
        best = np.argmin(scores[empty_before], axis=1)
        rows = np.where(empty_before)[0]
        sets[rows, best] = True
    return sets, empty_before
