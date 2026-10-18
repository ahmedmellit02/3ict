"""Two implementation choices that are not usually reported, and that matter
most exactly where medical VQA lives.

**Score randomization.** APS and RAPS are defined with a uniform draw inside the
score, `S = csum + u * p(y)`. Dropping it (`u = 1`) is tempting because it
makes the score deterministic and the predictions reproducible. It also makes
the score discrete, and a discrete score cannot hit the target level exactly:
the method over-covers, and pays for it in set size.

**Quantile jitter.** The earlier implementation (used for the submitted
version of the paper) added uniform noise of width 1e-6 to the calibration
scores. The primary protocol does
not (`cpaudit.bench.common.JITTER = 0`): the threshold is the plain order statistic
and ties are handled by the inclusive rule. The jittered arm is kept here only
as a sensitivity, since with K=2 the true-label score takes very few distinct
values and ties are routine.

Both effects scale with how atomic the score distribution is, so the experiment
is only meaningful run on both datasets: OmniMedVQA is mostly K=4, VQA-RAD is
entirely K=2 ("yes"/"no"), which is the worst case and also the clinically
common one.

Each arm re-tunes its own temperature on D_thr, because the tuning protocol is
part of the method: holding a temperature fixed that was chosen under
randomization would confound the ablation.

Run: python -m cpaudit.bench.randomization --cache cache/vqarad_mcq.npz --tag vqarad_mcq --seeds 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import (ALPHA, PostHoc, _POSTHOC_FN, posthoc_grid,
                              select_config, split_indices, thr_halves)
# The raw calibrator, because this module varies the jitter on purpose.
from cpaudit.calib.conformal import calibrate_marginal, predict_sets
from cpaudit.bench.common import load_cache
from cpaudit.eval.metrics import avg_set_size, coverage

#: LAC has no randomization: its score is 1 - p(y), with no tie-breaking draw.
#: It is kept as the control that isolates the jitter effect on its own.
METHODS = ("base-LAC", "APS", "RAPS")
RANDOMIZE = (True, False)
#: JITTER[0] is the primary protocol (none); JITTER[1] the earlier value.
JITTER = (0.0, 1e-6)


def _true(s, idx):
    return s[np.arange(len(idx)), idx]


def _tie_fraction(true_scores) -> float:
    """Share of calibration scores that share their value with another item."""
    _, cnt = np.unique(true_scores, return_counts=True)
    return float((cnt[cnt > 1].sum()) / len(true_scores)) if len(true_scores) else 0.0


def run(cache_path, n_seeds=20, seed0=0, alpha=ALPHA, methods=METHODS, cluster=False):
    cache = load_cache(cache_path)
    rows = []
    t0 = time.time()

    for seed in range(seed0, seed0 + n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
        sub_a, sub_b = thr_halves(cache, idx["thr"], seed)

        for method in methods:
            # LAC takes no `randomize`; running it twice would duplicate rows.
            rands = RANDOMIZE if method != "base-LAC" else (None,)
            for rand in rands:
                grid = [c if rand is None else {**c, "randomize": rand}
                        for c in posthoc_grid(method)]
                cfg, _ = select_config(method, grid, sub_a, sub_b, seed)
                sc = PostHoc(_POSTHOC_FN[method], cfg, seed)
                S_c, S_t = sc.score(cal), sc.score(test)
                true_cal = _true(S_c, cal.answer_idx)
                ties = _tie_fraction(true_cal)

                # Share of true-label calibration scores sitting exactly at the
                # score's maximum, 1: for deterministic APS that is every item
                # whose true label is ranked last. When it exceeds alpha the
                # quantile must land on it and the set admits every option.
                # Exact equality: the vectorized score gives the lower-ranked
                # candidate exactly 1.0 under u = 1 (no summation error).
                atom_one = float((true_cal == 1.0).mean())
                for jit in JITTER:
                    tau = calibrate_marginal(true_cal, alpha, jitter=jit, seed=seed)
                    sets, empty = predict_sets(S_t, test.mask, tau)
                    rows.append(dict(
                        seed=seed, method=method,
                        randomize="n/a" if rand is None else bool(rand),
                        jitter=jit, tie_frac=ties, atom_one=atom_one,
                        tau=float(tau),
                        all_options=float((sets.sum(1) == test.K).mean()),
                        temp=cfg.get("temp"), kmax=int(cache.kmax),
                        mean_K=float(cache.K.mean()),
                        # Chance for uniform guessing on a mixed-K corpus is
                        # E[1/K] over the evaluated items, not 1/E[K]: the two
                        # differ by a point on OmniMedVQA.
                        chance=float(np.mean(1.0 / test.K)),
                        coverage=coverage(sets, test.answer_idx),
                        avg_size=avg_set_size(sets),
                        empty_rate=float(empty.mean())))
        print(f"  [randomization] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    return pd.DataFrame(rows)


def stats(df, alpha=ALPHA):
    """Paired tests: deterministic vs randomized score, and no-jitter vs jitter."""
    from cpaudit.eval.stats import holm, paired

    out = []
    for metric in ("coverage", "avg_size"):
        fam = []
        keys = []
        for method in df.method.unique():
            d = df[df.method == method]
            if set(d.randomize.unique()) >= {True, False}:
                a = d[(d.randomize == False) & (d.jitter == JITTER[0])]  # noqa: E712
                b = d[(d.randomize == True) & (d.jitter == JITTER[0])]  # noqa: E712
                fam.append(paired(a.sort_values("seed")[metric].values,
                                  b.sort_values("seed")[metric].values,
                                  f"{method}: deterministic - randomized"))
                keys.append((method, "randomize", metric))
            # Jitter is tested inside EACH randomization arm. Pooling them, or
            # testing only the randomized arm, hides the effect entirely: with a
            # randomized score there are no ties for the jitter to break, so the
            # effect is identically zero there and non-zero only where it matters.
            for rand in d.randomize.unique():
                r0 = d[d.randomize == rand]
                a = r0[r0.jitter == JITTER[1]].sort_values("seed")[metric].values
                b = r0[r0.jitter == JITTER[0]].sort_values("seed")[metric].values
                if len(a) == len(b) and len(a):
                    fam.append(paired(a, b, f"{method}[rand={rand}]: "
                                            f"jitter - no jitter"))
                    keys.append((method, f"jitter|randomize={rand}", metric))
        holm(fam)
        for r, (method, effect, met) in zip(fam, keys):
            out.append(dict(method=method, effect=effect, metric=met,
                            delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                            p_t=r.p_t, p_w=r.p_w, p_holm=r.p_holm, n=r.n))
    return pd.DataFrame(out)


def report(df, st, tag, alpha=ALPHA):
    n = df.seed.nunique()
    print(f"\n=== Randomization and tie handling on {tag} "
          f"(alpha={alpha}, target {1 - alpha:.2f}, {n} seeds) ===")
    # Reported for the DETERMINISTIC arm only. Averaging over both arms halves
    # it for no reason: the randomized arm has no ties by construction, so the
    # pooled figure answers no question anyone asks.
    det = df[df.randomize.astype(str) == "False"]
    print(f"mean K = {df.mean_K.iloc[0]:.2f}, calibration-score ties without "
          f"the uniform draw = {det.tie_frac.mean():.1%} of items")

    print("\nCoverage and set size by (randomize, jitter):")
    piv = df.pivot_table(index=["method", "randomize"], columns="jitter",
                         values=["coverage", "avg_size"], aggfunc="mean")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nDropping the score's uniform draw (deterministic - randomized), "
          "and dropping the quantile jitter, Holm-corrected per metric:")
    cols = ["method", "effect", "metric", "delta", "ci_lo", "ci_hi",
            "p_t", "p_w", "p_holm"]
    print(st[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    over = df[(df.randomize == False) & (df.jitter == JITTER[0])]  # noqa: E712
    rand = df[(df.randomize == True) & (df.jitter == JITTER[0])]  # noqa: E712
    if len(over) and len(rand):
        print(f"\n\nDeterministic APS/RAPS over-cover by "
              f"{over.coverage.mean() - rand.coverage.mean():+.4f} and inflate "
              f"sets by {over.avg_size.mean() - rand.avg_size.mean():+.4f} "
              f"options on average.")


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
    df.to_csv(P.metric(f"bench_random_{args.tag}.csv"), index=False)
    st.to_csv(P.metric(f"bench_stats_random_{args.tag}.csv"), index=False)
    report(df, st, args.tag, args.alpha)
