"""Is the order study's "item effect" just a difference between cohorts?

`order_effect` decomposes position-conditional coverage into a letter effect and
an item effect by averaging over the four cyclic presentations. The item effect
is a difference between the four cohorts defined by where the dataset's authors
originally placed the correct answer. Averaging over presentations does not make
those cohorts comparable: if the cohort whose answer sits at index 3 happens to
hold harder modalities, or shorter answer strings, or a different question mix,
the "item effect" is partly a composition difference and not a property of the
items' difficulty as such.

This module builds the diagnostic tables that decide how much of it is
composition. For each original-index cohort it reports size, modality and
question-type mix, option-count mix, answer-string length, and the frozen
model's base accuracy. It then asks whether the effect survives conditioning:
the same profile computed WITHIN a modality, which holds the most plausible
confounder fixed.

It does not attempt to prove the residual is causal. The point is to say how
much of the gap composition can account for, and to keep the manuscript's
wording matched to that.

Run: python -m cpaudit.bench.order_cohorts --tag mcq --seeds 20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import (ALPHA, POSTHOC, axis_labels, build_scorers,
                              load_items, split_indices)
from cpaudit.bench.order_effect import load_shifts
from cpaudit.calib.conformal import predict_sets
from cpaudit.bench.common import calibrate_marginal


def _tvd(p, q):
    """Total-variation distance between two categorical distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def cohort_composition(cache, items, shifts=None):
    """Per original-answer-index cohort, on the K=4 subset: what is in it.

    `shifts` is the dict of cyclically shifted caches. When given, the reported
    accuracy is averaged over all four rotations: measuring it at the original
    presentation alone would score each cohort under exactly the letter effect
    the section is separating out, so the unrotated number is not evidence about
    the items.
    """
    sel = cache.K == 4
    idx = np.flatnonzero(sel)
    rows = []

    def top1(c):
        return (np.where(c.mask, c.s_base, np.inf).argmin(1) == c.answer_idx)

    ok_orig = top1(cache)
    ok_rot = (np.mean([top1(c) for c in shifts.values()], axis=0)
              if shifts else None)
    src = axis_labels(cache, "source", items, min_group=0)
    pooled_mod = pd.Series(cache.modality[idx]).value_counts(normalize=True).to_dict()
    pooled_qt = pd.Series(cache.question_type[idx]).value_counts(normalize=True).to_dict()
    pooled_src = pd.Series(src[idx]).value_counts(normalize=True).to_dict()

    for a in range(4):
        m = idx[cache.answer_idx[idx] == a]
        mods = pd.Series(cache.modality[m]).value_counts(normalize=True)
        qts = pd.Series(cache.question_type[m]).value_counts(normalize=True)
        srcs = pd.Series(src[m]).value_counts(normalize=True)
        lens = [len(items[str(i)]["options"][cache.answer_idx[j]])
                for j, i in zip(m, cache.ids[m]) if str(i) in items]
        rows.append(dict(
            cohort=a, n=len(m),
            base_top1_orig=float(ok_orig[m].mean()),
            base_top1_rotavg=float(ok_rot[m].mean()) if ok_rot is not None else np.nan,
            answer_len=float(np.mean(lens)) if lens else np.nan,
            top_modality=mods.index[0], top_modality_share=float(mods.iloc[0]),
            modality_entropy=float(-(mods * np.log(mods)).sum()),
            modality_tvd=_tvd(mods.to_dict(), pooled_mod),
            top_qtype=qts.index[0], top_qtype_share=float(qts.iloc[0]),
            qtype_tvd=_tvd(qts.to_dict(), pooled_qt),
            n_sources=int(srcs.size), top_source_share=float(srcs.iloc[0]),
            source_tvd=_tvd(srcs.to_dict(), pooled_src),
        ))
    return pd.DataFrame(rows)


def conditional_item_profile(n_seeds=20, alpha=ALPHA, methods=POSTHOC):
    """Item profile overall and WITHIN each modality.

    Averaging a cohort's coverage over the four shifts removes the letter
    effect. Doing the same inside one modality additionally removes any
    composition difference in modality, which is the confounder with the
    largest cohort imbalance.
    """
    shifts = load_shifts()
    rows = []
    for seed in range(n_seeds):
        for s, cache in sorted(shifts.items()):
            idx = split_indices(cache, seed)
            cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
            scorers, _ = build_scorers(cache, idx, seed, include=methods)
            sel = test.K == 4
            for method, sc in scorers.items():
                S_c, S_t = sc.score(cal), sc.score(test)
                tau = calibrate_marginal(S_c[np.arange(cal.n), cal.answer_idx],
                                         alpha, seed=seed)
                sets, _ = predict_sets(S_t, test.mask, tau)
                cov = sets[np.arange(test.n), test.answer_idx]
                for a in range(4):
                    m = sel & (test.answer_idx == a)
                    if m.sum():
                        rows.append(dict(seed=seed, shift=s, method=method,
                                         cohort=a, stratum="all",
                                         n=int(m.sum()),
                                         coverage=float(cov[m].mean())))
                    for mod in np.unique(test.modality):
                        mm = m & (test.modality == mod)
                        if mm.sum() >= 10:
                            rows.append(dict(seed=seed, shift=s, method=method,
                                             cohort=a, stratum=mod,
                                             n=int(mm.sum()),
                                             coverage=float(cov[mm].mean())))
        print(f"  [order_cohorts] seed {seed + 1}/{n_seeds}", flush=True)
    return pd.DataFrame(rows)


def report(comp, prof):
    print("\n=== Cohort composition (K=4 items, by original answer index) ===")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    so = comp.base_top1_orig.max() - comp.base_top1_orig.min()
    sr = comp.base_top1_rotavg.max() - comp.base_top1_rotavg.min()
    print(f"\nBase-score top-1 across cohorts: spread {so:.3f} at the original "
          f"presentation,\n{sr:.3f} averaged over all four rotations. Only the "
          f"second is free of the letter\neffect, so only the second is "
          f"evidence about the items.")
    print(f"\nComposition distances to the pooled K=4 distribution (TVD): "
          f"modality <= {comp.modality_tvd.max():.3f}, "
          f"question type <= {comp.qtype_tvd.max():.3f}, "
          f"source <= {comp.source_tvd.max():.3f}.")

    print("\n\n=== Item profile, averaged over all four presentations ===")
    overall = (prof[prof.stratum == "all"]
               .groupby(["method", "cohort"]).coverage.mean().unstack())
    overall["spread"] = overall.max(axis=1) - overall.min(axis=1)
    print(overall.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== The same profile WITHIN modality (composition held fixed) ===")
    within = prof[prof.stratum != "all"]
    # average the per-modality cohort means, weighting modalities equally so a
    # large modality cannot reintroduce the composition difference
    per = (within.groupby(["method", "stratum", "cohort"]).coverage.mean()
                 .groupby(["method", "cohort"]).mean().unstack())
    per["spread"] = per.max(axis=1) - per.min(axis=1)
    print(per.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nIf the item effect were composition, conditioning on modality "
          "would collapse the\nspread. Compare the two `spread` columns.")
    for m in overall.index:
        a, b = overall.loc[m, "spread"], per.loc[m, "spread"]
        print(f"  {m:<9s} overall {a:.4f} -> within-modality {b:.4f} "
              f"({'shrinks' if b < a else 'does not shrink'}, "
              f"{100*(a-b)/a:+.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mcq")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    shifts = load_shifts()
    comp = cohort_composition(shifts[min(shifts)], load_items(), shifts)
    prof = conditional_item_profile(args.seeds)
    comp.to_csv(P.metric(f"bench_cohort_composition_{args.tag}.csv"), index=False)
    prof.to_csv(P.metric(f"bench_cohort_profile_{args.tag}.csv"), index=False)
    report(comp, prof)
