"""What does it cost to tune a conformal baseline on the calibration set?

Split conformal's guarantee is conditional on the score being fixed before
D_cal is seen. Every post-hoc baseline here has hyperparameters -- a temperature,
RAPS's (gamma, k_reg) -- and they have to be chosen somehow. A benchmark that
picks them by "which setting gives the smallest sets" on the same split that
later supplies the conformal quantile has made the score a function of D_cal,
and the theorem no longer applies.

This is not a hypothetical failure mode; it is the path of least resistance when
running a comparison, because the calibration split is right there and is the
only labelled data left. So we run three arms on identical splits, differing
only in which split the hyperparameter search read, and in all three take the
final quantile from the full D_cal.

Sweeping the SIZE of the search turns this into a dose-response: at M = 1 there
is no choice to make, so the arms must coincide exactly -- that is the
experiment's internal control. As M grows, the leaky arms have more opportunity
to overfit D_cal, and their realized coverage should fall away from nominal
while the clean arm stays put. RAPS, with 60 configurations, is the most exposed
method in the benchmark for reasons that have nothing to do with its
nonconformity score.

Run: python -m cpaudit.bench.tuning_leak --cache cache/omnimedvqa_3000_mcq.npz --tag mcq --seeds 20
"""
from __future__ import annotations

import argparse
import time
import zlib

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import (ALPHA, POSTHOC, PostHoc, _POSTHOC_FN, posthoc_grid,
                              select_config, split_indices, thr_halves)
from cpaudit.calib.conformal import predict_sets
from cpaudit.bench.common import calibrate_marginal
from cpaudit.bench.common import load_cache
from cpaudit.eval.metrics import avg_set_size, coverage

#: Search sizes. Capped per method at its grid size and always extended to
#: include the full grid, so every method is also measured at the budget it
#: actually gets in the benchmark (LAC/APS 6, RAPS 60).
PRESSURE = (1, 2, 4, 8, 16, 32, 60)

#: Three protocols, identical but for which split the hyperparameter search read.
#:   clean      -- select on D_thr. What this repo does; D_cal stays untouched.
#:   leaky      -- select on a halved D_cal: tau on one half, set size on the
#:                 other. A careful-looking shortcut that is still a leak,
#:                 because both halves reappear in the final quantile.
#:   leaky-full -- select on all of D_cal, calibrating and measuring on the same
#:                 rows. The naive form, and the one a benchmark scripted in an
#:                 afternoon actually commits.
ARMS = ("clean", "leaky", "leaky-full")
LEAKY_ARMS = tuple(a for a in ARMS if a != "clean")


def _true(s, idx):
    return s[np.arange(len(idx)), idx]


def _halves(sub, seed):
    """Halve an arbitrary split the same way `thr_halves` halves D_thr."""
    perm = np.random.default_rng(1234 + seed).permutation(np.arange(sub.n))
    h = len(perm) // 2
    return sub.subset(perm[:h]), sub.subset(perm[h:])


def _subgrid_seed(method: str, seed: int, m: int) -> int:
    """A stable seed for the grid subsample.

    Python's `hash()` is salted per process for str/bytes, so `hash((method,
    seed, m))` would draw a different subgrid on every run and the experiment
    would not be reproducible. crc32 is stable across processes and versions.
    """
    return zlib.crc32(f"{method}|{seed}|{m}".encode()) & 0xFFFFFFFF


def _subgrid(grid, m, seed, method):
    """A size-m subset of the search space, drawn reproducibly.

    Random rather than the first m: the grids are ordered (temperature ascending,
    then gamma), so a prefix would confound search size with where in the space
    the search happens to start.
    """
    if m >= len(grid):
        return grid
    rng = np.random.default_rng(_subgrid_seed(method, seed, m))
    return [grid[i] for i in rng.choice(len(grid), size=m, replace=False)]


def run(cache_path, n_seeds=20, seed0=0, alpha=ALPHA, methods=POSTHOC,
        pressure=PRESSURE, cluster=False):
    cache = load_cache(cache_path)
    rows = []
    t0 = time.time()

    for seed in range(seed0, seed0 + n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
        # Selection data per arm. The clean arm reads D_thr; the leaky arms read
        # D_cal -- the very split that supplies the final quantile below.
        sel = {"clean": thr_halves(cache, idx["thr"], seed),
               "leaky": _halves(cal, seed),
               "leaky-full": (cal, cal)}

        for method in methods:
            grid = posthoc_grid(method)
            sizes = sorted({p for p in pressure if p <= len(grid)} | {len(grid)})
            for m in sizes:
                sub = _subgrid(grid, m, seed, method)
                for arm in ARMS:
                    sub_a, sub_b = sel[arm]
                    cfg, _ = select_config(method, sub, sub_a, sub_b, seed)
                    sc = PostHoc(_POSTHOC_FN[method], cfg, seed)
                    tau = calibrate_marginal(_true(sc.score(cal), cal.answer_idx),
                                             alpha, seed=seed)
                    sets, empty = predict_sets(sc.score(test), test.mask, tau)
                    rows.append(dict(
                        seed=seed, method=method, arm=arm, M=m,
                        grid_size=len(grid),
                        # Observations in each role. `n_thr_role` fits the
                        # provisional threshold, `n_obj_role` measures the set
                        # size being minimised, `n_cal_final` fits the reported
                        # threshold; `obj_is_thr` says whether the first two
                        # are the same rows (leaky-full only).
                        n_thr_role=sub_a.n, n_obj_role=sub_b.n,
                        n_cal_final=cal.n, n_test=test.n,
                        obj_is_thr=bool(sub_a is sub_b),
                        coverage=coverage(sets, test.answer_idx),
                        avg_size=avg_set_size(sets),
                        empty_rate=float(empty.mean()),
                        config=";".join(f"{k}={v}" for k, v in sorted(cfg.items()))))
        print(f"  [tuning_leak] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    return pd.DataFrame(rows)


def stats(df, alpha=ALPHA):
    """Paired leaky-vs-clean tests at each search size, plus a one-sample test
    of each arm against nominal coverage. Holm-corrected within each method."""
    from scipy import stats as sps

    from cpaudit.eval.stats import ci, holm, paired

    out = []
    for method in df.method.unique():
        d = df[df.method == method]
        ms = sorted(d.M.unique())
        keys = [(arm, m) for arm in LEAKY_ARMS for m in ms]
        fam = []
        for arm, m in keys:
            a = d[(d.arm == arm) & (d.M == m)].sort_values("seed").coverage.values
            b = d[(d.arm == "clean") & (d.M == m)].sort_values("seed").coverage.values
            fam.append(paired(a, b, f"{arm} M={m}"))
        holm(fam)
        for r, (arm, m) in zip(fam, keys):
            row = dict(method=method, arm=arm, M=m, delta=r.delta, ci_lo=r.ci_lo,
                       ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w, p_holm=r.p_holm,
                       n=r.n)
            for a_ in ARMS:
                v = d[(d.arm == a_) & (d.M == m)].coverage.values
                lo, hi = ci(v)
                row[f"{a_}_cov"] = float(v.mean())
                row[f"{a_}_ci_lo"], row[f"{a_}_ci_hi"] = lo, hi
                # Is this arm's realized coverage distinguishable from nominal?
                row[f"{a_}_p_vs_nominal"] = (
                    float(sps.ttest_1samp(v, 1 - alpha).pvalue) if len(v) > 1
                    else float("nan"))
            out.append(row)
    return pd.DataFrame(out)


def confirmatory(df, alpha=ALPHA):
    """The pre-specified test, kept separate from the exploratory M-sweep.

    The sweep over search sizes is a descriptive curve: it exists to show the
    shape of the dose-response, and correcting across all of its cells (2 arms x
    7 sizes per method) buries the one comparison anybody would pre-register.
    That comparison is: at the grid size the benchmark actually searches, does
    selecting on D_cal cost coverage? Three methods, three tests, Holm across
    those three and nothing else.
    """
    from cpaudit.eval.stats import holm, paired

    fam, keys = [], []
    for method in sorted(df.method.unique()):
        d = df[df.method == method]
        full = int(d.grid_size.iloc[0])
        a = d[(d.arm == "leaky-full") & (d.M == full)].sort_values("seed")
        b = d[(d.arm == "clean") & (d.M == full)].sort_values("seed")
        if len(a) == len(b) and len(a) > 1:
            r = paired(a.coverage.values, b.coverage.values, method)
            # A difference that is identically zero (binary LAC, whose sets
            # do not depend on temperature) has no test statistic. It stays in
            # the declared three-score family with p = 1 rather than silently
            # shrinking the family, which would make Holm less conservative.
            if not np.isfinite(r.p_t):
                r.p_t = r.p_w = 1.0
            fam.append(r)
            keys.append((method, full))
    holm(fam)
    out = []
    for r, (method, full) in zip(fam, keys):
        d = df[(df.method == method) & (df.M == full)]
        pivot = d.pivot_table(index="seed", columns="arm", values="config",
                              aggfunc="first")
        differ = float((pivot["leaky-full"] != pivot["clean"]).mean())
        out.append(dict(method=method, grid_size=full, delta=r.delta,
                        ci_lo=r.ci_lo, ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w,
                        p_holm=r.p_holm, n=r.n,
                        clean_cov=float(d[d.arm == "clean"].coverage.mean()),
                        leaky_cov=float(d[d.arm == "leaky-full"].coverage.mean()),
                        frac_seeds_different_config=differ))
    return pd.DataFrame(out)


def report(df, st, tag, alpha=ALPHA, conf=None):
    n = df.seed.nunique()
    print(f"\n=== Tuning leakage on {tag} (alpha={alpha}, target {1 - alpha:.2f}, "
          f"{n} seeds) ===")
    print("\nRealized coverage by search size M:")
    piv = df.pivot_table(index=["method", "M"], columns="arm", values="coverage",
                         aggfunc="mean").reindex(columns=list(ARMS))
    for a in LEAKY_ARMS:
        piv[f"d({a})"] = piv[a] - piv["clean"]
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nAt M=1 there is no choice to make, so the arms must coincide "
          "(internal control):")
    one = (df[df.M == 1].pivot_table(index="method", columns="arm",
                                     values="coverage", aggfunc="mean")
           .reindex(columns=list(ARMS)))
    one["identical"] = [bool(np.allclose(r[list(ARMS)].values, r["clean"]))
                        for _, r in one.iterrows()]
    print(one.to_string(float_format=lambda x: f"{x:.6f}"))

    if conf is not None and len(conf):
        print("\n\nCONFIRMATORY: at the grid size each method actually searches, "
              "does selecting on D_cal cost coverage? (Holm across these three "
              "tests only)")
        print(conf[["method", "grid_size", "clean_cov", "leaky_cov", "delta",
                    "ci_lo", "ci_hi", "p_t", "p_w", "p_holm",
                    "frac_seeds_different_config"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print("  `frac_seeds_different_config` is the mechanism: the leak bites "
              "only when it changes which configuration wins.")

    print("\n\nEXPLORATORY sweep (leaky arm - clean), Holm-corrected within each "
          "method across all arms and sizes:")
    cols = ["method", "arm", "M", "delta", "ci_lo", "ci_hi", "p_t", "p_w", "p_holm"]
    print(st[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print(f"\n\nIs each arm distinguishable from nominal {1 - alpha:.2f}? "
          "(one-sample t over seeds)")
    q = st[st.arm == LEAKY_ARMS[0]][
        ["method", "M"] + [f"{a}_{c}" for a in ARMS
                           for c in ("cov", "p_vs_nominal")]]
    print(q.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n\nEfficiency, to confirm the leak buys what it is supposed to buy "
          "(smaller sets) at the cost of validity:")
    sz = df.pivot_table(index=["method", "M"], columns="arm", values="avg_size",
                        aggfunc="mean").reindex(columns=list(ARMS))
    for a in LEAKY_ARMS:
        sz[f"d({a})"] = sz[a] - sz["clean"]
    print(sz.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--cluster", action="store_true",
                    help="image-disjoint split: every question about one "
                         "image stays in one partition")
    args = ap.parse_args()

    df = run(args.cache, n_seeds=args.seeds, alpha=args.alpha,
             cluster=args.cluster)
    st = stats(df, args.alpha)
    conf = confirmatory(df, args.alpha)
    df.to_csv(P.metric(f"bench_tuningleak_{args.tag}.csv"), index=False)
    st.to_csv(P.metric(f"bench_stats_tuningleak_{args.tag}.csv"), index=False)
    conf.to_csv(P.metric(f"bench_confirm_tuningleak_{args.tag}.csv"), index=False)
    report(df, st, args.tag, args.alpha, conf)
