"""Is LAC's temperature search doing anything, or is it vacuous at K=2?

The argument. `probs_from_base` sets p(y) proportional to exp(-s_base(y)/T) over
the unmasked options, so with z = -s_base and K = 2,

    p_1 = sigmoid((z_1 - z_2)/T),      s_LAC(y_1) = 1 - p_1 = sigmoid(-(z_1-z_2)/T).

Write d = z_1 - z_2. At T = 1, s = sigmoid(-d), hence -d = logit(s) and

    s'(y_1) = sigmoid(logit(s(y_1)) / T),

which is exactly the transform R1 names. It is strictly increasing in s and --
this is the part that matters -- it is the SAME function for every item, because
d enters only through s. A strictly increasing map g applied identically to all
scores commutes with the empirical quantile, so tau' = g(tau) and

    {y : s'(y) <= tau'} = {y : g(s(y)) <= g(tau)} = {y : s(y) <= tau}.

The prediction sets are therefore identical for every T > 0: at K = 2 the
temperature grid is a search over one point wearing six hats. Probability
VALUES change; the ranking, the calibrated threshold, and the sets do not.

At K >= 3 the argument fails. s_LAC(y) = 1 - p(y) depends on the whole logit
vector through the softmax denominator, so two items sharing a value of s(y) can
map to different s'(y) and no single g exists. The search is then real.

This module checks both halves empirically against the actual implementation
rather than trusting the algebra.

Run: python -m cpaudit.bench.lac_invariance
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.baselines.posthoc import lac_scores, probs_from_base
from cpaudit.bench.common import ALPHA, TEMPS, split_indices
from cpaudit.calib.conformal import predict_sets
from cpaudit.bench.common import calibrate_marginal
from cpaudit.bench.common import load_cache
from cpaudit.eval.metrics import avg_set_size, coverage


def check(cache_path, tag, n_seeds=20, alpha=ALPHA, temps=TEMPS, cluster=False):
    """Per seed, compare every temperature's LAC sets against T=1's."""
    cache = load_cache(cache_path)
    rows = []
    for seed in range(n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])

        def lac_sets(T):
            S_c = lac_scores(cal.s_base, cal.mask, temp=T)
            S_t = lac_scores(test.s_base, test.mask, temp=T)
            tau = calibrate_marginal(S_c[np.arange(cal.n), cal.answer_idx],
                                     alpha, seed=seed)
            return predict_sets(S_t, test.mask, tau)[0]

        # The reference is T = 1 itself, whatever order the grid lists.
        ref = lac_sets(1.0)
        for T in temps:
            sets = lac_sets(T)
            # Do the probabilities actually move? If they do not, set equality
            # is trivial and proves nothing about the transform.
            p1 = probs_from_base(test.s_base, test.mask, temp=1.0)
            pT = probs_from_base(test.s_base, test.mask, temp=T)
            rows.append(dict(
                tag=tag, seed=seed, temp=T,
                sets_identical_to_T1=bool(np.array_equal(sets, ref)),
                # Rounding can merge near-tied scores into exact ties, which the
                # inclusive rule then admits: sets may grow, never shrink.
                sets_contain_T1=bool((sets >= ref).all()),
                frac_items_set_differs=float((sets != ref).any(1).mean()),
                max_prob_shift=float(np.abs(pT - p1)[test.mask].max()),
                coverage=coverage(sets, test.answer_idx),
                avg_size=avg_set_size(sets)))
    return pd.DataFrame(rows)


def report(df):
    for tag, d in df.groupby("tag"):
        k = d.temp.nunique()
        ident = d.sets_identical_to_T1.all()
        worst = d.frac_items_set_differs.max()
        shift = d.max_prob_shift.max()
        print(f"\n=== {tag} ===")
        print(f"  temperatures compared : {k}  ({', '.join(str(t) for t in sorted(d.temp.unique()))})")
        print(f"  probabilities move    : max |p_T - p_1| = {shift:.4f}")
        print(f"  sets identical to T=1 : {'ALL' if ident else 'NO'}")
        print(f"  worst per-seed share of items whose set changes: {worst:.4f}")
        g = d.groupby("temp").agg(cov=("coverage", "mean"), size=("avg_size", "mean"))
        print(g.to_string(float_format=lambda x: f"{x:.4f}"))
        if ident:
            print("  => the temperature grid is vacuous for LAC here: probability")
            print("     values change but every temperature yields the same sets.")
        else:
            print("  => the temperature grid is a real search here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--backbone", default="",
                    help="cache suffix of another backbone, e.g. q3vl4b")
    args = ap.parse_args()
    sfx = f"_{args.backbone}" if args.backbone else ""
    out = []
    for tag, path, cl in (("VQA-RAD (K=2, image-disjoint)", "cache/vqarad_mcq.npz", True),
                          ("OmniMedVQA (mixed K)", "cache/omnimedvqa_3000_mcq.npz", False)):
        out.append(check(path.replace(".npz", f"{sfx}.npz"), tag, args.seeds,
                         args.alpha, cluster=cl))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(P.metric(f"bench_lac_invariance{sfx}.csv"), index=False)
    report(df)
