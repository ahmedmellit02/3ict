"""Post-hoc nonconformity scores.

Each scorer returns per-option scores `S` (N, Kmax) with the convention that the
prediction set is {y : S(x,y) <= tau}, plus the true-label score used for
calibration. That uniform interface lets calib.conformal handle all of them
(marginal AND Mondrian) identically.

References: LAC=Sadinle'19, APS=Romano'20, RAPS=Angelopoulos'21.
"""
from __future__ import annotations

import numpy as np


def probs_from_base(s_base: np.ndarray, mask: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Recover per-option probabilities from base scores s_base = -log p.

    p(y) ∝ exp(-s_base(y)/T) over the real (unmasked) options; pad = 0.

    `temp` is a temperature-scaling knob: the frozen model's probabilities need
    not be calibrated, and T is the standard fix. It is tuned on D_thr only
    (`cpaudit.bench.common.tune_posthoc`).
    """
    neg = np.where(mask, -s_base.astype(np.float64) / float(temp), -np.inf)
    neg = neg - neg.max(axis=1, keepdims=True)      # stabilize
    e = np.exp(neg)
    e = np.where(mask, e, 0.0)
    return e / e.sum(axis=1, keepdims=True)


def lac_scores(s_base: np.ndarray, mask: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """LAC: S(y) = 1 - p(y). Pad = +inf so pads never enter a set."""
    p = probs_from_base(s_base, mask, temp)
    S = 1.0 - p
    return np.where(mask, S, np.inf)


def _order_desc(p_row: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Indices of real options sorted by descending prob."""
    idx = np.where(real)[0]
    return idx[np.argsort(-p_row[idx])]


def _cumulative_scores(p: np.ndarray, mask: np.ndarray, u: np.ndarray,
                       randomize: bool, gamma: float = 0.0,
                       k_reg: int = 1) -> np.ndarray:
    """APS/RAPS with externally supplied uniforms, vectorized.

    `u[i, y]` is the draw for candidate y of row i, indexed by the candidate's
    ORIGINAL option position, so the caller decides how draws are keyed. Ranks
    are a stable descending sort of p: equal probabilities are ranked by
    option index, lower first. The score of the rank-k candidate is

        S = C_k - (1 - u) p_k + gamma * max(0, k - k_reg),

    where C_k is the cumulative probability through rank k, normalised so the
    last real rank has C exactly 1. That is Romano et al.'s C_{k-1} + u p_k
    written so that the deterministic variant (u = 1) gives the lower-ranked
    candidate a score of exactly 1 rather than 1 +- one ulp of summation error,
    which would otherwise split a mathematically single atom.
    """
    n, kmax = p.shape
    key = np.where(mask, -p, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    ps = np.take_along_axis(p, order, axis=1)
    ms = np.take_along_axis(mask, order, axis=1)
    us = (np.take_along_axis(u, order, axis=1) if randomize
          else np.ones_like(ps))
    C = np.cumsum(np.where(ms, ps, 0.0), axis=1)
    total = C[np.arange(n), ms.sum(1) - 1][:, None]
    C = C / total
    rank = np.arange(1, kmax + 1)[None, :]
    Ss = C - (1.0 - us) * ps + gamma * np.maximum(0, rank - k_reg)
    Ss = np.where(ms, Ss, np.inf)
    S = np.full((n, kmax), np.inf, dtype=np.float64)
    np.put_along_axis(S, order, Ss, axis=1)
    return S


def aps_scores(s_base: np.ndarray, mask: np.ndarray,
               randomize: bool = True, seed: int = 0, temp: float = 1.0,
               u: np.ndarray | None = None) -> np.ndarray:
    """APS: S(y) = (cumulative prob of options at least as likely as y).

    With randomization, subtract u*p(y) so coverage is exact. Set = {S <= tau}.

    `u=None` is the original implementation: a generator seeded by `seed`,
    restarted on every call and consumed in row order, so row j of any two
    calls shares draws. Passing `u` (N, Kmax) uses those draws instead; the
    3ICT benchmark keys them by (re-split, item, option) so distinct items
    never share one.
    """
    p = probs_from_base(s_base, mask, temp)
    if u is not None:
        return _cumulative_scores(p, mask, u, randomize)
    rng = np.random.default_rng(seed)
    n, kmax = p.shape
    S = np.full((n, kmax), np.inf, dtype=np.float64)
    for i in range(n):
        order = _order_desc(p[i], mask[i])
        csum = 0.0
        for y in order:
            u = rng.uniform() if randomize else 1.0
            # cumulative up to and including y, minus a random fraction of p(y)
            S[i, y] = csum + u * p[i, y]
            csum += p[i, y]
    return S


def raps_scores(s_base: np.ndarray, mask: np.ndarray, k_reg: int = 1,
                gamma: float = 0.1, randomize: bool = True, seed: int = 0,
                temp: float = 1.0, u: np.ndarray | None = None) -> np.ndarray:
    """RAPS: APS + size regularizer gamma*max(0, rank - k_reg) (Angelopoulos'21).

    `u` as in `aps_scores`."""
    p = probs_from_base(s_base, mask, temp)
    if u is not None:
        return _cumulative_scores(p, mask, u, randomize, gamma, k_reg)
    rng = np.random.default_rng(seed)
    n, kmax = p.shape
    S = np.full((n, kmax), np.inf, dtype=np.float64)
    for i in range(n):
        order = _order_desc(p[i], mask[i])
        csum = 0.0
        for rank, y in enumerate(order, start=1):
            u = rng.uniform() if randomize else 1.0
            reg = gamma * max(0, rank - k_reg)
            S[i, y] = csum + u * p[i, y] + reg
            csum += p[i, y]
    return S


def softmax_threshold_sets(s_base_cal, mask_cal, true_idx_cal,
                           s_base_test, mask_test, alpha: float):
    """Naive confidence baseline (NO conformal guarantee).

    Pick t = alpha-quantile of true-class probs on cal (no finite-sample
    correction), then C(x) = {y : p(y) >= t}. Illustrates the calibration gap.
    Returns (sets, empty_before).
    """
    p_cal = probs_from_base(s_base_cal, mask_cal)
    p_true = p_cal[np.arange(len(true_idx_cal)), true_idx_cal]
    t = float(np.quantile(p_true, alpha))
    p_test = probs_from_base(s_base_test, mask_test)
    sets = (p_test >= t) & mask_test
    empty_before = ~sets.any(axis=1)
    # argmax fallback
    rows = np.where(empty_before)[0]
    if len(rows):
        best = np.argmax(np.where(mask_test[rows], p_test[rows], -np.inf), axis=1)
        sets[rows, best] = True
    return sets, empty_before


# Name -> callable producing per-option scores S (set = {S <= tau}).
BASELINE_SCORERS = {
    "LAC": lac_scores,
    "APS": aps_scores,
    "RAPS": raps_scores,
}
