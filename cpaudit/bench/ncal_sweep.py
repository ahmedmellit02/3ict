"""How much calibration data does a conformal guarantee actually need, and what
does Mondrian charge for the group-conditional version?

Split conformal's coverage is exact in expectation at any n, but the realized
coverage on one calibration draw is a Beta random variable whose spread is set
by n alone. Mondrian pays that cost once per group: partitioning n_cal into G
levels leaves each level with roughly n_cal/G, so the same nominal guarantee is
bought on a fraction of the data, and levels that fall under the sample-size
floor ceil((1-alpha)/alpha) cannot be calibrated at all -- they take tau = inf
and emit the full option set.

So the practitioner's question is not "marginal or Mondrian?" but "at my
calibration budget, which one is still worth having?". We sweep n_cal by
subsampling D_cal, run both schemes for every baseline, and record realized
coverage, its spread across draws, worst-group coverage, set size, and how many
levels the floor disabled. The theoretical Beta spread is recorded beside the
empirical one, so a run that misses it is a bug rather than a finding.

Run: python -m cpaudit.bench.ncal_sweep --cache cache/omnimedvqa_3000_mcq.npz --tag mcq --seeds 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import ALPHA, BASELINES, axis_labels, build_scorers, split_indices
from cpaudit.calib.conformal import mondrian_floor, predict_sets
from cpaudit.bench.common import calibrate_marginal, calibrate_mondrian
from cpaudit.bench.common import load_cache
from cpaudit.eval.metrics import avg_set_size, coverage, per_group_metrics

#: Calibration budgets to sweep. `None` means "all of D_cal" (450 items on the
#: 3,000-item OmniMedVQA sample), which is the budget every other experiment uses.
N_CAL = (50, 100, 200, 300, None)
#: Independent subsamples per (seed, budget). Coverage at small n is a lottery;
#: one draw per seed would confound split variability with draw variability.
N_REPS = 3
GROUP_AXIS = "modality"


def _true(s, idx):
    return s[np.arange(len(idx)), idx]


def beta_sd(n: int, alpha: float = ALPHA) -> float:
    """Theoretical sd of realized coverage for split conformal at n calibration
    points. Coverage ~ Beta(n + 1 - l, l) with l = floor(alpha (n + 1)); this is
    the distribution-free prediction every run below should land on."""
    l = int(np.floor(alpha * (n + 1)))
    if l < 1:
        return float("nan")
    a, b = n + 1 - l, l
    return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))


def beta_sd_total(n: int, m: int, alpha: float = ALPHA) -> float:
    """sd of the coverage actually OBSERVED on m test points.

    Comparing an empirical spread against `beta_sd` alone understates the
    prediction: a finite test set adds binomial noise on top of the calibration
    draw,

        Var(observed) = Var(Q) + E[Q(1 - Q)] / m,

    with E[Q(1-Q)] = E[Q](1 - E[Q]) - Var(Q) for Q ~ Beta. Omitting the second
    term is what let an earlier version of this sweep claim agreement at n=50,
    where the calibration term dominates, while diverging badly at the full
    budget, where it does not.
    """
    l = int(np.floor(alpha * (n + 1)))
    if l < 1 or m <= 0:
        return float("nan")
    a, b = n + 1 - l, l
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return float(np.sqrt(var + (mean * (1 - mean) - var) / m))


def run(cache_path, n_seeds=20, seed0=0, alpha=ALPHA, include=BASELINES,
        budget="paper", n_cal=N_CAL, n_reps=N_REPS, axis=GROUP_AXIS, cluster=False):
    cache = load_cache(cache_path)
    floor = mondrian_floor(alpha)
    lab_all = axis_labels(cache, axis)
    rows = []
    t0 = time.time()

    for seed in range(seed0, seed0 + n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal_full, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
        lab_test = lab_all[idx["test"]]
        lab_cal_full = lab_all[idx["cal"]]
        scorers, _ = build_scorers(cache, idx, seed, include=include,
                                   budget=budget, alpha=alpha)

        for method, sc in scorers.items():
            S_cal_full, S_t = sc.score(cal_full), sc.score(test)

            for n in n_cal:
                size = cal_full.n if n is None else min(n, cal_full.n)
                reps = 1 if n is None else n_reps
                for rep in range(reps):
                    if n is None:
                        pick = np.arange(cal_full.n)
                    else:
                        rng = np.random.default_rng(7000 + 97 * seed + rep)
                        pick = rng.choice(cal_full.n, size=size, replace=False)
                    true_cal = _true(S_cal_full[pick],
                                     cal_full.answer_idx[pick])
                    lab_cal = lab_cal_full[pick]

                    tau = calibrate_marginal(true_cal, alpha, seed=seed)
                    sets, empty = predict_sets(S_t, test.mask, tau)
                    tau_g = calibrate_mondrian(true_cal, lab_cal, alpha, seed=seed)
                    sets_g, empty_g = predict_sets(S_t, test.mask, tau_g,
                                                   groups=lab_test)
                    below = sum(1 for g, v in tau_g.items()
                                if g != "__floor__" and not np.isfinite(v))
                    _, cnt = np.unique(lab_cal, return_counts=True)

                    pg = per_group_metrics(sets, test.answer_idx, lab_test, alpha)
                    pg_g = per_group_metrics(sets_g, test.answer_idx, lab_test, alpha)
                    rows.append(dict(
                        seed=seed, rep=rep, method=method, n_cal=size,
                        n_per_group=float(cnt.mean()), min_group_n=int(cnt.min()),
                        n_groups=len(cnt), n_below_floor=below, floor=floor,
                        beta_sd=beta_sd(size, alpha),
                        beta_sd_total=beta_sd_total(size, test.n, alpha),
                        n_test=test.n,
                        cov_marginal=coverage(sets, test.answer_idx),
                        size_marginal=avg_set_size(sets),
                        empty_marginal=float(empty.mean()),
                        worst_marginal=float(pg.coverage.min()),
                        cov_mondrian=coverage(sets_g, test.answer_idx),
                        size_mondrian=avg_set_size(sets_g),
                        empty_mondrian=float(empty_g.mean()),
                        worst_mondrian=float(pg_g.coverage.min())))
        print(f"  [ncal_sweep] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    return pd.DataFrame(rows)


def report(df, tag, alpha=ALPHA):
    n_seeds = df.seed.nunique()
    print(f"\n=== Calibration budget on {tag} (alpha={alpha}, "
          f"target {1 - alpha:.2f}, {n_seeds} seeds x {df.rep.nunique()} draws) ===")

    print("\nRealized coverage: mean, and spread across draws, against the "
          "distribution-free Beta prediction.")
    g = df.groupby("n_cal").agg(
        per_group=("n_per_group", "mean"), min_group=("min_group_n", "min"),
        disabled=("n_below_floor", "mean"),
        marg_cov=("cov_marginal", "mean"), marg_sd=("cov_marginal", "std"),
        beta_cal=("beta_sd", "mean"),
        beta_total=("beta_sd_total", "mean"),
        mond_cov=("cov_mondrian", "mean"), mond_sd=("cov_mondrian", "std"))
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nWhat Mondrian buys and what it costs, by budget "
          "(worst-group coverage / mean set size):")
    w = df.groupby("n_cal").agg(
        worst_marg=("worst_marginal", "mean"), worst_mond=("worst_mondrian", "mean"),
        size_marg=("size_marginal", "mean"), size_mond=("size_mondrian", "mean"))
    w["worst_gain"] = w.worst_mond - w.worst_marg
    w["size_cost"] = w.size_mond - w.size_marg
    print(w.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nPer method, at the smallest and largest budget:")
    lo, hi = df.n_cal.min(), df.n_cal.max()
    m = (df[df.n_cal.isin([lo, hi])]
         .pivot_table(index="method", columns="n_cal",
                      values=["cov_mondrian", "worst_mondrian", "size_mondrian"],
                      aggfunc="mean"))
    print(m.to_string(float_format=lambda x: f"{x:.3f}"))

    bad = w[(w.worst_gain <= 0)]
    if len(bad):
        print(f"\n\nBudgets at which Mondrian does NOT improve worst-group "
              f"coverage: {sorted(bad.index.tolist())}")
    print("\n(A level below the floor takes tau = inf, so its sets are the whole "
          "option list: Mondrian coverage rises toward 1 while set size blows up. "
          "Read the two columns together.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--cluster", action="store_true",
                    help="image-disjoint split: every question about one "
                         "image stays in one partition")
    ap.add_argument("--axis", default=GROUP_AXIS,
                    help="Mondrian partition; modality (G=8) by default")
    args = ap.parse_args()

    include = BASELINES
    df = run(args.cache, n_seeds=args.seeds, alpha=args.alpha, include=include,
             cluster=args.cluster, axis=args.axis)
    df.to_csv(P.metric(f"bench_ncal_{args.tag}.csv"), index=False)
    report(df, args.tag, args.alpha)
