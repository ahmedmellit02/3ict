"""Which partition should you Mondrianize on, and does any of them buy
conditional validity?

Mondrian calibration is always described against ONE grouping -- here, imaging
modality. That framing hides the question a practitioner actually faces: the
partition is a modelling choice, and a guarantee conditional on modality says
nothing about any other way of slicing the data. So we cross the axis used to
CALIBRATE against the axis used to EVALUATE, for every baseline:

    calibrate on {marginal, modality, question type, source dataset, position}
    evaluate on  {modality, question type, source dataset, position, K}

The diagonal reproduces the usual claim (Mondrian repairs the axis it was given).
The off-diagonal is the finding: a guarantee bought on one axis does not transfer
to another, and in particular answer position -- a pure prompt-formatting
variable, nothing to do with clinical content -- stays broken no matter which
clinically-motivated partition you calibrate on.

Also records, for every method, the distribution of predicted answer position
against the true one: the MCQ letter protocol's own bias, measured six ways.

Run: python -m cpaudit.bench.group_axes --cache cache/omnimedvqa_3000_mcq.npz --tag mcq --seeds 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import (ALPHA, BASELINES, axis_labels, build_scorers,
                              load_items, split_indices)
from cpaudit.calib.conformal import mondrian_floor, predict_sets
from cpaudit.bench.common import calibrate_marginal, calibrate_mondrian
from cpaudit.bench.common import load_cache
from cpaudit.eval.metrics import per_group_metrics

MARGINAL = "marginal"
#: Partitions a practitioner could plausibly calibrate on. `K` is excluded: the
#: option count is known at inference but is not a population a clinician cares
#: about, and the 3,000-item sample has only 7 items at K=3.
CALIB_AXES = (MARGINAL, "modality", "question_type", "source", "position")
EVAL_AXES = ("modality", "question_type", "source", "position", "K")
#: Levels rarer than this are pooled into "other" before calibrating on `source`
#: (42 source datasets, most below the Mondrian floor).
MIN_GROUP = 25


def _true(s, idx):
    return s[np.arange(len(idx)), idx]


def all_labels(cache, axes, items, min_group=MIN_GROUP):
    """Label the WHOLE cache once, then slice per split.

    Labelling each split separately would let `source` pool differently on D_cal
    and D_test: a dataset rare in one and common in the other would be "other"
    on one side and its own level on the other, so at prediction time it would
    look like an unseen group and silently collect tau = inf. Pooling on the full
    sample uses nothing but item ids -- no labels, no scores -- so it is a
    dataset constant, not a leak.
    """
    return {ax: axis_labels(cache, ax, items,
                            min_group=min_group if ax == "source" else 0)
            for ax in axes}


def _calib_group_stats(labels, tau):
    """How the Mondrian partition behaved on D_cal: how many levels it has, how
    many the sample-size floor disabled, and the smallest level."""
    _, cnt = np.unique(labels, return_counts=True)
    below = sum(1 for g, v in tau.items()
                if g != "__floor__" and not np.isfinite(v))
    return dict(n_calib_groups=len(cnt), n_below_floor=below,
                min_calib_group_n=int(cnt.min()))


def _position_rows(S_t, test, method, seed):
    """Predicted vs true answer position, on the items of one option count.

    Mixing option counts would manufacture a deficit at the tail that has
    nothing to do with position bias: on a K=2 item the letters C and D do not
    exist, so pooling it with K=4 items guarantees an apparent shortfall at C
    and D. We therefore restrict to the dataset's modal K -- 4 on OmniMedVQA,
    2 on the VQA-RAD yes/no subset.
    """
    counts = np.bincount(test.K)
    if not counts.size:
        return []
    k = int(counts.argmax())
    sel = test.K == k
    if k < 2 or not sel.any():
        return []
    S = np.where(test.mask[sel], S_t[sel], np.inf)
    pred, true = S.argmin(1), test.answer_idx[sel]
    n = int(sel.sum())
    return [dict(seed=seed, method=method, position=p, n=n, K=k,
                 pred_frac=float((pred == p).mean()),
                 true_frac=float((true == p).mean())) for p in range(k)]


def run(cache_path, n_seeds=20, seed0=0, alpha=ALPHA, include=BASELINES,
        budget="paper", calib_axes=CALIB_AXES, eval_axes=EVAL_AXES,
        min_group=MIN_GROUP, cluster=False):
    cache = load_cache(cache_path)
    items = load_items()
    floor = mondrian_floor(alpha)
    axes = sorted(set(eval_axes) | {a for a in calib_axes if a != MARGINAL})
    lab_all = all_labels(cache, axes, items, min_group)

    # Drop axes this dataset cannot support. On VQA-RAD every item is a closed
    # yes/no question from one source with K=2, so question_type, source and K
    # are single-level: Mondrian on them is just marginal calibration, and
    # reporting them as if they were partitions would pad the tables with
    # tautologies.
    degenerate = {ax for ax, lab in lab_all.items() if len(np.unique(lab)) < 2}
    if degenerate:
        print(f"  [group_axes] skipping single-level axes on this dataset: "
              f"{', '.join(sorted(degenerate))}", flush=True)
        lab_all = {ax: lab for ax, lab in lab_all.items() if ax not in degenerate}
        calib_axes = tuple(a for a in calib_axes if a not in degenerate)
        eval_axes = tuple(a for a in eval_axes if a not in degenerate)
    if not eval_axes:
        raise ValueError("no usable grouping axis on this dataset")
    rows, pos_rows = [], []
    t0 = time.time()

    for seed in range(seed0, seed0 + n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
        lab_cal = {ax: lab[idx["cal"]] for ax, lab in lab_all.items()}
        lab_test = {ax: lab[idx["test"]] for ax, lab in lab_all.items()}
        scorers, _ = build_scorers(cache, idx, seed, include=include,
                                   budget=budget, alpha=alpha)

        for method, sc in scorers.items():
            S_c, S_t = sc.score(cal), sc.score(test)
            true_cal = _true(S_c, cal.answer_idx)
            pos_rows += _position_rows(S_t, test, method, seed)

            for ca in calib_axes:
                if ca == MARGINAL:
                    tau = calibrate_marginal(true_cal, alpha, seed=seed)
                    groups = None
                    stats = dict(n_calib_groups=1, n_below_floor=0,
                                 min_calib_group_n=cal.n)
                else:
                    tau = calibrate_mondrian(true_cal, lab_cal[ca], alpha, seed=seed)
                    groups = lab_test[ca]
                    stats = _calib_group_stats(lab_cal[ca], tau)
                sets, empty = predict_sets(S_t, test.mask, tau, groups=groups)

                for ea in eval_axes:
                    pg = per_group_metrics(sets, test.answer_idx, lab_test[ea], alpha)
                    rows.append(pg.reset_index().assign(
                        method=method, calib_axis=ca, eval_axis=ea, seed=seed,
                        empty_rate=float(empty.mean()), floor=floor, **stats))
        print(f"  [group_axes] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    return pd.concat(rows, ignore_index=True), pd.DataFrame(pos_rows)


def null_worst(sizes, alpha=ALPHA, n_sim=4000, seed=0):
    """Worst-group coverage a PERFECTLY conditionally valid method would show.

    Worst-group coverage is the minimum of G noisy per-group estimates, so it
    sits below the target even when the guarantee holds exactly, and it sinks
    further the more groups there are. Comparing `source` (25 levels) against
    `position` (4) on raw worst-group would therefore reward having few groups,
    which is an artefact, not a property of the calibration.

    So we simulate the null: each group's coverage is Binomial(n_g, 1-alpha)/n_g,
    and we take the minimum. The gap between this and the observed worst group is
    the part that needs explaining.

    Returns the expected worst group, its 5th percentile, and the expected
    spread (max - min), which is the null for the second statistic: worst-of-G
    answers "how bad is the worst group", spread answers "does coverage vary
    across groups at all". They are not the same question, and an axis whose
    effect is a systematic ordered trend rather than one bad level shows up in
    the second and barely in the first.
    """
    sizes = np.asarray([s for s in sizes if s > 0])
    if not len(sizes):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.binomial(sizes[None, :], 1 - alpha,
                         size=(n_sim, len(sizes))) / sizes[None, :]
    m = draws.min(axis=1)
    sp = draws.max(axis=1) - m
    return float(m.mean()), float(np.quantile(m, 0.05)), float(sp.mean())


def null_worst_shared(sizes, n_cal, alpha=ALPHA, n_sim=4000, seed=0):
    """`null_worst`, with the calibration threshold allowed to vary.

    `null_worst` gives every group coverage 1 - alpha exactly in expectation,
    independently. Under marginal calibration all groups share one threshold,
    and the coverage that threshold delivers varies across calibration draws:
    for n exchangeable continuous scores it is Q ~ Beta(k, n + 1 - k) with
    k = ceil((n + 1)(1 - alpha)). Here each simulation draws one Q and gives
    every group Binomial(n_g, Q) / n_g, so the groups move together. It is
    still a reference, not the exact law of a stratified pipeline with ties.
    """
    sizes = np.asarray([s for s in sizes if s > 0])
    if not len(sizes) or not n_cal:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    if k > n_cal:
        return 1.0, 1.0, 0.0
    q = rng.beta(k, n_cal + 1 - k, size=n_sim)
    draws = rng.binomial(sizes[None, :], q[:, None],
                         size=(n_sim, len(sizes))) / sizes[None, :]
    worst = draws.min(1)
    return (float(worst.mean()), float(np.quantile(worst, 0.05)),
            float((draws.max(1) - worst).mean()))


def add_shared_reference(summary, pg, alpha=ALPHA):
    """Excesses against `null_worst_shared`, for marginal calibration only
    (the reference describes one shared threshold). Adds `*_shared` columns."""
    marg = summary[summary.calib_axis == "marginal"].copy()
    n_cal = (marg.groupby("seed").min_calib_group_n.first())
    keys = pg[pg.calib_axis == "marginal"][["seed", "eval_axis", "group", "n"]] \
        .drop_duplicates()
    rows = []
    for (seed, ax), d in keys.groupby(["seed", "eval_axis"]):
        mean, p05, sp = null_worst_shared(d.n.values, int(n_cal.loc[seed]), alpha,
                                          seed=int(seed))
        rows.append(dict(seed=seed, eval_axis=ax, shared_worst=mean,
                         shared_worst_p05=p05, shared_spread=sp))
    out = marg.merge(pd.DataFrame(rows), on=["seed", "eval_axis"], how="left")
    out["excess_deficit_shared"] = out.shared_worst - out.worst_cov
    out["excess_spread_shared"] = out.spread - out.shared_spread
    return out


def add_null_baseline(summary, pg, alpha=ALPHA):
    """Attach the null worst-group baseline and the excess deficit.

    Group sizes depend only on (seed, eval_axis), not on the method or on what
    was calibrated, so the simulation runs once per pair and is merged on.
    """
    keys = pg[["seed", "eval_axis", "group", "n"]].drop_duplicates()
    rows = []
    for (seed, ax), d in keys.groupby(["seed", "eval_axis"]):
        mean, p05, sp = null_worst(d.n.values, alpha, seed=int(seed))
        rows.append(dict(seed=seed, eval_axis=ax, null_worst=mean,
                         null_worst_p05=p05, null_spread=sp,
                         n_groups_eval=len(d)))
    out = summary.merge(pd.DataFrame(rows), on=["seed", "eval_axis"], how="left")
    # Positive = worse than a perfectly conditionally valid method would look.
    out["excess_deficit"] = out.null_worst - out.worst_cov
    # Second statistic: does coverage vary across the levels at all, beyond the
    # variation G finite groups would show anyway?
    out["excess_spread"] = out.spread - out.null_spread
    return out


def summarize(pg):
    """Collapse per-group rows to the per-(method, calib_axis, eval_axis) numbers
    the paper quotes: worst group, spread, and what the floor disabled."""
    g = (pg.groupby(["seed", "method", "calib_axis", "eval_axis"])
           .agg(worst_cov=("coverage", "min"), best_cov=("coverage", "max"),
                mean_cov=("coverage", "mean"), avg_size=("avg_size", "mean"),
                n_eval_groups=("group", "nunique"), min_eval_group_n=("n", "min"),
                groups_below_target=("covered_ok", lambda s: int((~s).sum())),
                n_calib_groups=("n_calib_groups", "first"),
                min_calib_group_n=("min_calib_group_n", "first"),
                n_below_floor=("n_below_floor", "first"),
                floor=("floor", "first"))
           .reset_index())
    g["spread"] = g.best_cov - g.worst_cov
    return g


def report(summary, tag, alpha=ALPHA):
    from cpaudit.eval.stats import half_width

    n_seeds = summary.seed.nunique()
    print(f"\n=== Grouping axes on {tag} (alpha={alpha}, target {1 - alpha:.2f}, "
          f"{n_seeds} seeds) ===")
    print("\nWorst-group coverage. Rows = axis calibrated on, "
          "columns = axis evaluated on.")
    for method in summary.method.unique():
        m = summary[summary.method == method]
        piv = m.pivot_table(index="calib_axis", columns="eval_axis",
                            values="worst_cov", aggfunc="mean")
        piv = piv.reindex(index=[a for a in CALIB_AXES if a in piv.index],
                          columns=[a for a in EVAL_AXES if a in piv.columns])
        print(f"\n  {method}")
        print(piv.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n\nDoes a guarantee transfer across axes? "
          "(worst-group coverage, mean over methods and axes, paired by seed)")
    mg = summary[summary.calib_axis == MARGINAL]
    diag = summary[(summary.calib_axis == summary.eval_axis)]
    off = summary[(summary.calib_axis != summary.eval_axis)
                  & (summary.calib_axis != MARGINAL)]
    for label, frame in (("marginal calibration", mg),
                         ("Mondrian, matched axis", diag),
                         ("Mondrian, mismatched axis", off)):
        if not len(frame):
            continue
        w = frame.groupby("seed").worst_cov.mean().values
        s = frame.groupby("seed").avg_size.mean().values
        print(f"  {label:<28s} worst-group {w.mean():.3f} +/- {half_width(w):.3f}"
              f"   set size {s.mean():.3f} +/- {half_width(s):.3f}   (n={len(w)})")

    if "excess_deficit" in summary:
        print("\n\nWorst-group coverage is the minimum of G noisy estimates, so "
              "it sits below target even under a perfect guarantee, and sinks "
              "further the more groups there are. Against the simulated null:")
        nb = (summary.groupby("eval_axis")
              .agg(groups=("n_groups_eval", "mean"),
                   null_worst=("null_worst", "mean"),
                   worst=("worst_cov", "mean"),
                   excess_worst=("excess_deficit", "mean"),
                   null_spread=("null_spread", "mean"),
                   spread=("spread", "mean"),
                   excess_spread=("excess_spread", "mean")))
        print(nb.to_string(float_format=lambda x: f"{x:.3f}"))
        print("  `excess_worst`  -- how bad the worst level is, beyond chance.")
        print("  `excess_spread` -- whether coverage varies across levels at "
              "all, beyond chance.")
        print("  An axis whose effect is one bad level scores on the first; an")
        print("  axis whose effect is a systematic trend across levels scores")
        print("  on the second. Compare these, not the raw columns.")

    print("\n\nPosition is the axis nobody calibrates on. Worst-group coverage "
          "when evaluated on position:")
    vals = ["worst_cov"] + (["excess_deficit", "excess_spread"]
                            if "excess_deficit" in summary else [])
    pos = (summary[summary.eval_axis == "position"]
           .pivot_table(index="method", columns="calib_axis", values=vals,
                        aggfunc="mean"))
    print(pos.to_string(float_format=lambda x: f"{x:.3f}"))

    floor = int(summary.floor.iloc[0]) if len(summary) else 0
    print(f"\n\nThe Mondrian partition on D_cal (a level needs n_g >= {floor} at "
          f"alpha={alpha}, else tau = inf and the set is everything):")
    fl = (summary[summary.calib_axis != MARGINAL]
          .groupby("calib_axis")
          .agg(levels=("n_calib_groups", "mean"),
               disabled_by_floor=("n_below_floor", "mean"),
               smallest_level_n=("min_calib_group_n", "min")))
    print(fl.to_string(float_format=lambda x: f"{x:.2f}"))


def position_report(pos, tag):
    if not len(pos):
        print(f"\n=== Predicted vs true answer position on {tag}: no usable "
              f"option count (needs K >= 2) ===")
        return
    k = int(pos.K.iloc[0])
    print(f"\n=== Predicted vs true answer position on {tag} (K={k} items) ===")
    piv = pos.pivot_table(index="method", columns="position",
                          values=["pred_frac", "true_frac"], aggfunc="mean")
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    tvd = (pos.groupby(["method", "seed"])[["pred_frac", "true_frac"]]
              .apply(lambda d: 0.5 * (d.pred_frac - d.true_frac).abs().sum())
              .groupby("method").mean())
    print("\n  total variation distance from the true position distribution:")
    print(tvd.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--cluster", action="store_true",
                    help="image-disjoint split: every question about one "
                         "image stays in one partition")
    ap.add_argument("--shared-reference", action="store_true",
                    help="only add the shared-threshold reference to this "
                         "tag's existing CSVs -> bench_axisshared_{tag}.csv")
    args = ap.parse_args()

    if args.shared_reference:
        pg = pd.read_csv(P.metric(f"bench_groupaxes_{args.tag}.csv"))
        summary = pd.read_csv(P.metric(f"bench_axissummary_{args.tag}.csv"))
        add_shared_reference(summary, pg, args.alpha).to_csv(
            P.metric(f"bench_axisshared_{args.tag}.csv"), index=False)
        raise SystemExit(0)

    include = BASELINES
    pg, pos = run(args.cache, n_seeds=args.seeds, alpha=args.alpha,
                  include=include, cluster=args.cluster)
    summary = add_null_baseline(summarize(pg), pg, args.alpha)
    pg.to_csv(P.metric(f"bench_groupaxes_{args.tag}.csv"), index=False)
    summary.to_csv(P.metric(f"bench_axissummary_{args.tag}.csv"), index=False)
    pos.to_csv(P.metric(f"bench_position_dist_{args.tag}.csv"), index=False)
    report(summary, args.tag, args.alpha)
    position_report(pos, args.tag)
