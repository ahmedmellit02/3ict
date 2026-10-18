"""Evaluation metrics. All operate on boolean prediction-set arrays.

Reported: marginal coverage, avg set size + 95% CI, per-subgroup coverage/size,
size-stratified coverage (SSC), empty-set rate (before argmax fallback).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def coverage(sets: np.ndarray, answer_idx: np.ndarray) -> float:
    """Fraction of items whose set contains the true label."""
    hit = sets[np.arange(len(answer_idx)), answer_idx]
    return float(hit.mean())


def avg_set_size(sets: np.ndarray) -> float:
    return float(sets.sum(axis=1).mean())


def size_ci(sets: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Student-t 95% CI on the mean set size (see cpaudit.eval.stats)."""
    from cpaudit.eval.stats import ci
    return ci(sets.sum(axis=1).astype(float), level)


def empty_set_rate(empty_before: np.ndarray) -> float:
    """Rate of empty sets *before* the argmax fallback (diagnostic)."""
    return float(np.asarray(empty_before).mean())


def per_group_metrics(sets: np.ndarray, answer_idx: np.ndarray,
                      groups: np.ndarray, alpha: float) -> pd.DataFrame:
    """Coverage & size per subgroup -- the per-subgroup table."""
    hit = sets[np.arange(len(answer_idx)), answer_idx]
    size = sets.sum(axis=1)
    df = pd.DataFrame({"group": groups, "hit": hit, "size": size})
    out = df.groupby("group").agg(
        n=("hit", "size"),
        coverage=("hit", "mean"),
        avg_size=("size", "mean"),
    )
    out["target"] = 1.0 - alpha
    out["covered_ok"] = out["coverage"] >= out["target"]
    return out.sort_values("coverage")


def size_stratified_coverage(sets: np.ndarray, answer_idx: np.ndarray) -> pd.DataFrame:
    """Coverage conditioned on realized set size . Exposes methods that
    hit marginal coverage by over-covering easy items and under-covering hard ones.
    """
    hit = sets[np.arange(len(answer_idx)), answer_idx]
    size = sets.sum(axis=1)
    df = pd.DataFrame({"size": size, "hit": hit})
    return df.groupby("size").agg(n=("hit", "size"), coverage=("hit", "mean"))


def evaluate(sets: np.ndarray, empty_before: np.ndarray, answer_idx: np.ndarray,
             groups: np.ndarray, alpha: float) -> dict:
    """One-call summary bundle for a method on D_test."""
    lo, hi = size_ci(sets)
    return {
        "coverage": coverage(sets, answer_idx),
        "avg_size": avg_set_size(sets),
        "size_ci95": (lo, hi),
        "empty_rate": empty_set_rate(empty_before),
        "per_group": per_group_metrics(sets, answer_idx, groups, alpha),
        "ssc": size_stratified_coverage(sets, answer_idx),
        "worst_group_coverage": float(
            per_group_metrics(sets, answer_idx, groups, alpha)["coverage"].min()
        ),
    }
