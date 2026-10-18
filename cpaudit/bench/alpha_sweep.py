"""Coverage, set size and size-stratified coverage across target levels.

A post-hoc score separates cleanly from its level: the nonconformity score is
built once, and alpha enters only at the final quantile. Two arms run at every
level:

    matched  -- the score is re-tuned at the level it will be evaluated at
    transfer -- the score is tuned once at alpha=0.10 and recalibrated elsewhere

For post-hoc scores the two arms differ only through the tuned hyperparameters.
Table I is the matched arm at alpha=0.10. Size-stratified coverage is recorded
alongside, since it is the standard diagnostic separating APS/RAPS from LAC.

Run: python -m cpaudit.bench.alpha_sweep --cache cache/omnimedvqa_3000_mcq.npz --tag mcq --seeds 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import ALPHA, BASELINES, build_scorers, split_indices
from cpaudit.calib.conformal import predict_sets, upper_band
from cpaudit.bench.common import calibrate_marginal
from cpaudit.bench.common import load_cache
from cpaudit.eval.metrics import avg_set_size, coverage, size_stratified_coverage

ALPHAS = (0.01, 0.05, 0.10, 0.20)
MATCHED, TRANSFER = "matched", "transfer"


def _true(s, idx):
    return s[np.arange(len(idx)), idx]


def run(cache_path, n_seeds=20, seed0=0, alphas=ALPHAS, include=BASELINES,
        budget="paper", cluster=False):
    cache = load_cache(cache_path)
    rows, ssc_rows = [], []
    t0 = time.time()

    for seed in range(seed0, seed0 + n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])

        # The transfer arm is built once, at the reference level, and reused at
        # every evaluation level -- that is what "transfer" means.
        base_scorers, _ = build_scorers(cache, idx, seed, include=include,
                                        budget=budget, alpha=ALPHA)

        for a in alphas:
            arms = {TRANSFER: base_scorers}
            # At the reference level the two arms are the same object by
            # construction; rebuilding would only burn a GPU-hour to reproduce it.
            arms[MATCHED] = (base_scorers if a == ALPHA else
                             build_scorers(cache, idx, seed, include=include,
                                           budget=budget, alpha=a)[0])
            for arm, scorers in arms.items():
                for method, sc in scorers.items():
                    S_c, S_t = sc.score(cal), sc.score(test)
                    tau = calibrate_marginal(_true(S_c, cal.answer_idx), a, seed=seed)
                    sets, empty = predict_sets(S_t, test.mask, tau)
                    cov = coverage(sets, test.answer_idx)
                    rows.append(dict(
                        seed=seed, alpha=a, arm=arm, method=method,
                        coverage=cov, avg_size=avg_set_size(sets),
                        # Per item, then averaged: with mixed K, dividing the
                        # mean size by the mean K is a different statistic and
                        # not the one "fraction of the menu" names.
                        norm_size=float((sets.sum(1) / test.K).mean()),
                        empty_rate=float(empty.mean()),
                        lower=1 - a, upper=upper_band(cal.n, a),
                        in_band=bool(1 - a <= cov <= upper_band(cal.n, a)),
                        top1=float((np.where(test.mask, S_t, np.inf).argmin(1)
                                    == test.answer_idx).mean())))
                    if arm == MATCHED:
                        ssc = size_stratified_coverage(sets, test.answer_idx)
                        ssc_rows.append(ssc.reset_index().assign(
                            seed=seed, alpha=a, method=method))
        print(f"  [alpha_sweep] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    return pd.DataFrame(rows), pd.concat(ssc_rows, ignore_index=True)


def stats(df):
    """matched - transfer, per method and level, Holm-corrected per method."""
    from cpaudit.eval.stats import holm, paired

    out = []
    for method in df.method.unique():
        d = df[df.method == method]
        levels = [a for a in sorted(d.alpha.unique()) if a != ALPHA]
        fam = []
        for a in levels:
            x = d[(d.arm == MATCHED) & (d.alpha == a)].sort_values("seed")
            y = d[(d.arm == TRANSFER) & (d.alpha == a)].sort_values("seed")
            fam.append(paired(x.avg_size.values, y.avg_size.values, f"alpha={a}"))
        holm(fam)
        for r, a in zip(fam, levels):
            out.append(dict(method=method, alpha=a, delta=r.delta, ci_lo=r.ci_lo,
                            ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w,
                            p_holm=r.p_holm, n=r.n,
                            # every score here is post-hoc; the column keeps
                            # the CSV schema of the results shipped with it
                            trained=False))
    return pd.DataFrame(out)


def report(df, ssc, st, tag):
    print(f"\n=== Level sweep on {tag} ({df.seed.nunique()} seeds) ===")

    print("\nValidity holds everywhere -- that is the theorem, not a finding. "
          "Fraction of seeds inside [1-alpha, 1-alpha+1/(n+1)]:")
    band = df.pivot_table(index="method", columns="alpha", values="in_band",
                          aggfunc="mean")
    print(band.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n\nEfficiency, matched arm (each method rebuilt at its level):")
    m = df[df.arm == MATCHED].pivot_table(index="method", columns="alpha",
                                          values="avg_size", aggfunc="mean")
    print(m.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n\nCost of NOT rebuilding (transfer - matched set size). Post-hoc "
          "methods should be near zero; trained methods should not be:")
    piv = df.pivot_table(index=["method", "alpha"], columns="arm",
                         values="avg_size", aggfunc="mean")
    piv["cost"] = piv[TRANSFER] - piv[MATCHED]
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    if len(st):
        print("\n\nPaired matched - transfer, Holm-corrected within each method:")
        cols = ["method", "trained", "alpha", "delta", "ci_lo", "ci_hi",
                "p_t", "p_w", "p_holm"]
        print(st[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

        tr = st[st.trained]
        ph = st[~st.trained]
        if len(tr) and len(ph):
            print(f"\n  mean |cost| for trained methods  : "
                  f"{tr.delta.abs().mean():.4f}")
            print(f"  mean |cost| for post-hoc methods : "
                  f"{ph.delta.abs().mean():.4f}")

    print("\n\nSize-stratified coverage at alpha=0.10 (conditional on the size "
          "the method actually emitted):")
    s = ssc[ssc.alpha == ALPHA].pivot_table(index="method", columns="size",
                                            values="coverage", aggfunc="mean")
    print(s.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cluster", action="store_true",
                    help="image-disjoint split: every question about one "
                         "image stays in one partition")
    args = ap.parse_args()

    include = BASELINES
    df, ssc = run(args.cache, n_seeds=args.seeds, include=include,
                  cluster=args.cluster)
    st = stats(df)
    df.to_csv(P.metric(f"bench_alpha_{args.tag}.csv"), index=False)
    ssc.to_csv(P.metric(f"bench_ssc_{args.tag}.csv"), index=False)
    st.to_csv(P.metric(f"bench_stats_alpha_{args.tag}.csv"), index=False)
    report(df, ssc, st, args.tag)
