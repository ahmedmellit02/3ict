"""A formal test for coverage heterogeneity across a partition.

`group_axes` reports an *excess spread* over a simulated reference. That is a
descriptive contrast, not a test: it has no null distribution attached, so it
cannot say whether an observed spread is larger than chance would produce, and
subtracting a simulated mean is not the same as rejecting a hypothesis.

This module supplies the test the contrast was standing in for.

    H0: whether an item is covered is independent of which level it belongs to.

Statistic: the Pearson chi-square of the G x 2 table (level) x (covered?).
Null: permute the level labels across test items, holding the coverage
indicators fixed. That construction preserves the realized marginal coverage
exactly -- only the association between level and coverage is destroyed -- which
is what makes it a test of heterogeneity rather than of the operating point. The
asymptotic chi-square reference is unreliable when levels are small, which is
precisely this setting, so the p-value is the permutation one.

Clustering. Items sharing an image are not independent, so on a corpus with
repeated questions per image the labels are permuted at the CLUSTER level: whole
images are relabelled together. On OmniMedVQA images are effectively unique and
the two coincide.

Seeds are re-splits of one corpus and are therefore NOT independent replicates.
We run the test within each seed and report the distribution of p-values rather
than pooling them, because pooling would treat 20 correlated re-splits as 20
experiments.

Run: python -m cpaudit.bench.heterogeneity --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import (ALPHA, POSTHOC, axis_labels, build_scorers,
                              image_of, load_items, split_indices)
from cpaudit.calib.conformal import predict_sets
from cpaudit.bench.common import calibrate_marginal
from cpaudit.bench.common import load_cache

AXES = ("modality", "question_type", "source", "position", "K")
N_PERM = 2000
MIN_GROUP = 25


def chi2_stat(labels, covered) -> float:
    """Pearson chi-square of the (level) x (covered?) table.

    Computed directly rather than via scipy so the permutation loop stays cheap;
    zero-expected cells are dropped, which only happens for an empty level.
    """
    levels, inv = np.unique(labels, return_inverse=True)
    n = len(covered)
    obs = np.zeros((len(levels), 2))
    np.add.at(obs, (inv, covered.astype(int)), 1.0)
    row = obs.sum(1, keepdims=True)
    col = obs.sum(0, keepdims=True)
    exp = row @ col / n
    ok = exp > 0
    return float((((obs - exp) ** 2 / np.where(ok, exp, 1.0))[ok]).sum())


def permutation_p(labels, covered, cluster=None, n_perm=N_PERM, seed=0):
    """Permute level labels (whole clusters together) and count exceedances."""
    rng = np.random.default_rng(seed)
    obs = chi2_stat(labels, covered)
    if cluster is None:
        ge = 0
        for _ in range(n_perm):
            ge += chi2_stat(rng.permutation(labels), covered) >= obs
    else:
        # One label per cluster, permuted as a unit, then broadcast back.
        cl, cinv = np.unique(cluster, return_inverse=True)
        # a cluster's level is constant in practice; take its first item's
        first = np.zeros(len(cl), dtype=int)
        seen = np.zeros(len(cl), dtype=bool)
        lev, linv = np.unique(labels, return_inverse=True)
        for i, c in enumerate(cinv):
            if not seen[c]:
                first[c], seen[c] = linv[i], True
        ge = 0
        for _ in range(n_perm):
            ge += chi2_stat(lev[rng.permutation(first)][cinv], covered) >= obs
    # Add-one estimator: never reports p = 0, which no finite permutation
    # sample can justify.
    return obs, (ge + 1) / (n_perm + 1)


def _one_per_image(cache, items):
    """Keep the first question of every image, so each image appears once.

    The row-wise permutation null treats test items as exchangeable units. On
    OmniMedVQA that is close to true -- 2,990 images behind 3,000 items, with 10
    images carrying two questions each -- but close is not the same as exact.
    Re-running on this subset removes the approximation entirely, at the cost of
    ten items.
    """
    img = image_of(cache, items)
    _, first = np.unique(img, return_index=True)
    return cache.subset(np.sort(first))


def run(cache_path, n_seeds=20, seed0=0, alpha=ALPHA, axes=AXES, n_perm=N_PERM,
        cluster=False, methods=POSTHOC, min_group=MIN_GROUP,
        unique_images=False):
    cache = load_cache(cache_path)
    items = load_items()
    if unique_images:
        before = cache.n
        cache = _one_per_image(cache, items)
        print(f"  [heterogeneity] one question per image: {before} -> {cache.n}")
    lab_all = {ax: axis_labels(cache, ax, items,
                               min_group=min_group if ax == "source" else 0)
               for ax in axes}
    lab_all = {ax: l for ax, l in lab_all.items() if len(np.unique(l)) >= 2}
    img = image_of(cache, items)
    rows = []
    t0 = time.time()
    for seed in range(seed0, seed0 + n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
        scorers, _ = build_scorers(cache, idx, seed, include=methods)
        cl = img[idx["test"]] if cluster else None
        for method, sc in scorers.items():
            S_c, S_t = sc.score(cal), sc.score(test)
            tau = calibrate_marginal(S_c[np.arange(cal.n), cal.answer_idx],
                                     alpha, seed=seed)
            sets, _ = predict_sets(S_t, test.mask, tau)
            covered = sets[np.arange(test.n), test.answer_idx]
            for ax, lab in lab_all.items():
                lt = lab[idx["test"]]
                stat, p = permutation_p(lt, covered, cl, n_perm, seed)
                rows.append(dict(seed=seed, method=method, axis=ax,
                                 chi2=stat, p_perm=p, n_perm=n_perm,
                                 n_levels=len(np.unique(lt)),
                                 marginal_cov=float(covered.mean())))
        print(f"  [heterogeneity] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    """Holm-correct within each (method) family of axes, per seed, then
    summarise across seeds."""
    from cpaudit.eval.stats import holm, Paired

    out = []
    for (seed, method), d in df.groupby(["seed", "method"]):
        fam = [Paired(name=r.axis, delta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                      p_t=r.p_perm, p_w=np.nan, n=0) for r in d.itertuples()]
        holm(fam)
        for r, f in zip(d.itertuples(), fam):
            out.append(dict(seed=seed, method=method, axis=r.axis,
                            chi2=r.chi2, p_perm=r.p_perm, p_holm=f.p_holm,
                            n_levels=r.n_levels))
    return pd.DataFrame(out)


def report(s, tag, n_perm=N_PERM):
    print(f"\n=== Coverage heterogeneity, permutation test on {tag} ===")
    print(f"statistic: Pearson chi-square of (level) x (covered?); null: level "
          f"labels permuted\nwith coverage fixed, {n_perm} permutations; Holm "
          f"across the {s.axis.nunique()} axes within each method and seed.")
    g = (s.groupby(["method", "axis"])
          .agg(chi2=("chi2", "mean"), median_p=("p_holm", "median"),
               frac_seeds_sig=("p_holm", lambda v: float((v < 0.05).mean())),
               levels=("n_levels", "mean"))
          .reset_index())
    piv = g.pivot(index="axis", columns="method",
                  values="frac_seeds_sig").reindex(AXES)
    print("\nFraction of the 20 re-splits where the axis shows significant "
          "heterogeneity\n(Holm-adjusted permutation p < 0.05). Seeds are "
          "re-splits of one corpus, so these\nare not 20 independent "
          "experiments; the column is a stability summary, not a count\nof "
          "successes.")
    print(piv.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nMean chi-square and median Holm-adjusted p:")
    print(g.pivot(index="axis", columns="method", values="chi2")
           .reindex(AXES).to_string(float_format=lambda x: f"{x:.1f}"))
    print(g.pivot(index="axis", columns="method", values="median_p")
           .reindex(AXES).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nHeterogeneity is an association, not a cause: these axes are "
          "themselves correlated,\nso a significant result for one does not "
          "isolate it from the others.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--unique-images", action="store_true",
                    help="keep one question per image before splitting, so the "
                         "row-wise permutation null is exact")
    args = ap.parse_args()
    df = run(args.cache, n_seeds=args.seeds, alpha=args.alpha,
             n_perm=args.perms, cluster=args.cluster,
             unique_images=args.unique_images)
    s = summarize(df)
    df.to_csv(P.metric(f"bench_heterogeneity_{args.tag}.csv"), index=False)
    s.to_csv(P.metric(f"bench_heterogeneity_holm_{args.tag}.csv"), index=False)
    report(s, args.tag, args.perms)
