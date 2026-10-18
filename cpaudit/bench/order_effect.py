"""Is the position effect a property of the item, or of the letter it sat behind?

Under marginal calibration the MCQ base score's coverage varies sharply with
which letter the correct answer was printed against. That is a correlation: the
items whose answer happens to land at D might simply be harder. Observational
data cannot separate the two, so we re-ran the extraction.

The same 3,000 items are re-scored with the option list cyclically shifted, so
that each option appears behind each letter in turn. Everything else -- image,
question, option text, model, prompt template -- is held fixed, and the cache
stays in the ORIGINAL option indexing, so a shifted cache pairs 1:1 with the
unshifted one on item id. The manipulation is presentation order and nothing
else.

Three questions:

  (a) Does presentation order change a method's aggregate behaviour at all?
      Paired over seeds, per method, across shifts.
  (b) Does the coverage gap follow the LETTER or the ITEM? If it follows the
      letter, then conditioning on display position reproduces the same profile
      under every shift, even though a different set of items occupies each
      position. If it follows the item, the profile moves with the items.
  (c) Does averaging over presentations fix it? The order-ensembled score
      averages s_base over the K distinct cyclic shifts of each item -- exactly
      position-balanced, since every option then appears in every position once.
      A practitioner pays K forward passes for it, and it requires no
      retraining, no extra labels, and no change to the conformal machinery.

Run: python -m cpaudit.bench.order_effect --tag mcq --seeds 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import ALPHA, BASELINES, build_scorers, split_indices
from cpaudit.calib.conformal import predict_sets
from cpaudit.bench.common import calibrate_marginal, calibrate_mondrian
from cpaudit.bench.common import load_cache
from cpaudit.data.cache import FeatureCache
from cpaudit.eval.metrics import avg_set_size, coverage, per_group_metrics

#: shift -> cache. Shift 0 is the frozen cache every other experiment uses.
DEFAULT_CACHES = {
    0: "cache/omnimedvqa_3000_mcq.npz",
    1: "cache/omnimedvqa_3000_mcq_s1.npz",
    2: "cache/omnimedvqa_3000_mcq_s2.npz",
    3: "cache/omnimedvqa_3000_mcq_s3.npz",
}
#: VQA-RAD: two orders of {yes, no}. With K = 2 the two cyclic shifts are all
#: the permutations there are, so the decomposition is complete.
VQARAD_CACHES = {
    0: "cache/vqarad_mcq.npz",
    1: "cache/vqarad_mcq_s1.npz",
}
ENSEMBLE = "ensemble"
LETTERS = "ABCDE"


def _true(s, idx):
    return s[np.arange(len(idx)), idx]


def load_shifts(paths=DEFAULT_CACHES):
    """Load every available shifted cache and assert they are item-aligned.

    If the ids or labels disagree the pairing is void and every paired test
    below would be comparing different items, so this fails loudly.
    """
    out = {}
    for s, p in sorted(paths.items()):
        if Path(p).exists():
            out[s] = load_cache(p)
    if not out:
        raise FileNotFoundError(f"no shifted caches found among {list(paths.values())}")
    ref = out[min(out)]
    for s, c in out.items():
        if not np.array_equal(c.ids, ref.ids):
            raise ValueError(f"shift {s} is not item-aligned with shift {min(out)}: "
                             "the caches cannot be paired")
        if not np.array_equal(c.answer_idx, ref.answer_idx):
            raise ValueError(f"shift {s} has different answer_idx -- the shifted "
                             "cache was not written in original option indexing")
    return out


def display_position(K, answer_idx, shift):
    """Which letter (0=A) the true answer was printed behind, at this shift."""
    return (np.asarray(answer_idx) - shift) % np.asarray(K)


def ensemble_cache(shifts: dict) -> FeatureCache:
    """Average s_base over the K DISTINCT cyclic shifts of each item.

    Distinct matters: for a K=2 item, shifts 0 and 2 present the same order, so
    averaging over all four would weight one presentation twice and the ensemble
    would not be position-balanced. Averaging is done on s_base = -log p(letter),
    i.e. a geometric mean of the letter probabilities, which is the usual way to
    pool log-scores.
    """
    ref = shifts[min(shifts)]
    avail = sorted(shifts)
    acc = np.zeros_like(ref.s_base, dtype=np.float64)
    cnt = np.zeros_like(ref.s_base, dtype=np.float64)
    for s in avail:
        c = shifts[s]
        use = (s % np.maximum(ref.K, 1)) == s          # s < K: a distinct order
        use = use[:, None] & ref.mask
        vals = np.where(np.isfinite(c.s_base), c.s_base, 0.0)
        acc += np.where(use, vals, 0.0)
        cnt += use
    s_base = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.inf).astype(np.float32)
    s_base = np.where(ref.mask, s_base, np.inf).astype(np.float32)
    return FeatureCache(ids=ref.ids, K=ref.K, mask=ref.mask, s_base=s_base,
                        z_x=ref.z_x, e_y=ref.e_y, answer_idx=ref.answer_idx,
                        modality=ref.modality, question_type=ref.question_type)


def _eval(sc, cal, test, alpha, seed, shift, k=4):
    """Marginal calibration, then overall and per-display-position metrics.

    The position profile uses the items with exactly `k` options, where every
    letter hosts every original index once across the k cyclic shifts."""
    S_c, S_t = sc.score(cal), sc.score(test)
    tau = calibrate_marginal(_true(S_c, cal.answer_idx), alpha, seed=seed)
    sets, empty = predict_sets(S_t, test.mask, tau)
    top1 = float((np.where(test.mask, S_t, np.inf).argmin(1)
                  == test.answer_idx).mean())
    overall = dict(coverage=coverage(sets, test.answer_idx),
                   avg_size=avg_set_size(sets), empty_rate=float(empty.mean()),
                   top1=top1)
    sel = test.K == k
    pos = display_position(test.K, test.answer_idx, shift)
    pg = per_group_metrics(sets[sel], test.answer_idx[sel],
                           np.asarray([f"pos-{LETTERS[p]}" for p in pos[sel]]),
                           alpha)
    # Within a display-position cell every item has the same ORIGINAL option
    # index, since p = (a - shift) mod K. Recording it is what makes the
    # two-way decomposition in `decompose` possible: averaging cells over
    # shifts holds the letter fixed and rotates the items through it, and
    # averaging over letters does the reverse.
    pg = pg.reset_index()
    letter = pg["group"].str[-1].map({c: i for i, c in enumerate(LETTERS)})
    pg["letter"] = letter
    pg["orig_index"] = (letter + shift) % k
    return overall, pg


def decompose(pos, alpha=ALPHA, k=4):
    """Separate the letter effect from the item effect.

    With every cyclic shift present the design is balanced: each letter hosts
    each original option index exactly once across the four shifts. So

        averaging a letter's cells over shifts   -> item differences cancel,
                                                    what remains is the letter
        averaging an index's cells over shifts   -> letter effects cancel,
                                                    what remains is the items

    Only the single-presentation arms take part; the ensemble has no display
    order, so it cannot contribute a letter profile.
    """
    d = pos[pos.arm.isin([str(s) for s in range(k)])]
    if d.arm.nunique() < k:
        return None, None
    letter = (d.groupby(["method", "seed", "letter"]).coverage.mean()
                .rename("coverage").reset_index())
    item = (d.groupby(["method", "seed", "orig_index"]).coverage.mean()
              .rename("coverage").reset_index())
    return letter, item


def run(n_seeds=20, seed0=0, alpha=ALPHA, include=BASELINES, budget="paper",
        caches=DEFAULT_CACHES, k=4, cluster=False):
    shifts = load_shifts(caches)
    print(f"[order_effect] shifts available: {sorted(shifts)}  "
          f"(n={shifts[min(shifts)].n} items, aligned)")
    arms = dict(shifts)
    if len(shifts) > 1:
        arms[ENSEMBLE] = ensemble_cache(shifts)

    rows, pos_rows = [], []
    t0 = time.time()
    for seed in range(seed0, seed0 + n_seeds):
        for arm, cache in arms.items():
            shift = 0 if arm == ENSEMBLE else arm
            idx = split_indices(cache, seed, cluster=cluster)
            cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
            scorers, _ = build_scorers(cache, idx, seed, include=include,
                                       budget=budget, alpha=alpha)
            for method, sc in scorers.items():
                overall, pg = _eval(sc, cal, test, alpha, seed, shift, k=k)
                rows.append(dict(seed=seed, arm=str(arm), method=method, **overall))
                pos_rows.append(pg.assign(seed=seed, arm=str(arm),
                                          method=method))
        print(f"  [order_effect] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    return pd.DataFrame(rows), pd.concat(pos_rows, ignore_index=True)


def stats(df, pos):
    """Each shift against shift 0, and the ensemble against shift 0."""
    from cpaudit.eval.stats import holm, paired

    spread = (pos.groupby(["seed", "arm", "method"])
                 .coverage.agg(lambda s: s.max() - s.min())
                 .rename("spread").reset_index())
    out, fam, keys = [], [], []
    for method in df.method.unique():
        d = df[df.method == method]
        sp = spread[spread.method == method]
        base = "0"
        for arm in [a for a in d.arm.unique() if a != base]:
            for metric, frame in (("coverage", d), ("avg_size", d),
                                  ("pos_spread", sp)):
                col = "spread" if metric == "pos_spread" else metric
                a = frame[frame.arm == arm].sort_values("seed")[col].values
                b = frame[frame.arm == base].sort_values("seed")[col].values
                if len(a) == len(b) and len(a) > 1:
                    fam.append(paired(a, b, f"{method} {arm}-vs-0 {metric}"))
                    keys.append((method, arm, metric))
    holm(fam)
    for r, (method, arm, metric) in zip(fam, keys):
        out.append(dict(method=method, arm=arm, metric=metric, delta=r.delta,
                        ci_lo=r.ci_lo, ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w,
                        p_holm=r.p_holm, n=r.n))
    return pd.DataFrame(out), spread


def report(df, pos, st, spread, tag, alpha=ALPHA):
    print(f"\n=== Option order on {tag} (alpha={alpha}, target {1 - alpha:.2f}, "
          f"{df.seed.nunique()} seeds) ===")

    print("\n(a) Aggregate behaviour per presentation order:")
    g = df.pivot_table(index="method", columns="arm",
                       values=["coverage", "avg_size", "top1"], aggfunc="mean")
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\n(b) Coverage by the LETTER the answer was printed behind "
          "(K=4 items). If the effect were a property of the items, the profile "
          "would move with the shift; if it is the letter, it stays put.")
    piv = pos.pivot_table(index=["method", "arm"], columns="group",
                          values="coverage", aggfunc="mean")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\n(c) Spread of position-conditional coverage (max - min over the "
          "four letters). Lower is better; the ensemble is the remedy.")
    sp = spread.pivot_table(index="method", columns="arm", values="spread",
                            aggfunc="mean")
    print(sp.to_string(float_format=lambda x: f"{x:.4f}"))

    letter, item = decompose(pos, alpha)
    if letter is not None:
        print("\n\n(b2) The two-way decomposition. With all four shifts the "
              "design is balanced, so averaging a letter's cells over shifts "
              "cancels item differences and vice versa.")
        lp = letter.pivot_table(index="method", columns="letter",
                                values="coverage", aggfunc="mean")
        lp.columns = [f"letter-{'ABCD'[c]}" for c in lp.columns]
        ip = item.pivot_table(index="method", columns="orig_index",
                              values="coverage", aggfunc="mean")
        ip.columns = [f"origidx-{c}" for c in ip.columns]
        both = lp.join(ip)
        both["letter_spread"] = lp.max(axis=1) - lp.min(axis=1)
        both["item_spread"] = ip.max(axis=1) - ip.min(axis=1)
        print(both.to_string(float_format=lambda x: f"{x:.4f}"))
        print("  A letter effect survives averaging over items; an item effect "
              "survives averaging over letters.")
        print("  Compare `letter_spread` against `item_spread`: whichever is "
              "larger is what the")
        print("  original single-presentation gap was mostly made of.")

    if ENSEMBLE in set(st.arm):
        print("\n\nOrder-ensembled vs single presentation (shift 0), "
              "Holm-corrected:")
        e = st[st.arm == ENSEMBLE][["method", "metric", "delta", "ci_lo",
                                    "ci_hi", "p_t", "p_w", "p_holm"]]
        print(e.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def confirmatory(df, pos, alpha=ALPHA):
    """The pre-specified test, kept separate from the exploratory arm sweep.

    `stats` corrects across every arm and metric, which is right for reading
    that table but buries the one comparison worth pre-registering: does
    averaging over presentation orders reduce the spread of position-conditional
    coverage, against the single presentation everyone actually runs? One test
    per score -- three of them, since this paper reports three post-hoc scores --
    Holm across those and nothing else.
    """
    from cpaudit.eval.stats import holm, paired

    spread = (pos.groupby(["seed", "arm", "method"])
                 .coverage.agg(lambda s: s.max() - s.min())
                 .rename("spread").reset_index())
    fam, keys = [], []
    for method in sorted(spread.method.unique()):
        d = spread[spread.method == method]
        a = d[d.arm == ENSEMBLE].sort_values("seed").spread.values
        b = d[d.arm == "0"].sort_values("seed").spread.values
        if len(a) == len(b) and len(a) > 1:
            fam.append(paired(a, b, method))
            keys.append(method)
    holm(fam)
    return pd.DataFrame([
        dict(method=m, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
             p_t=r.p_t, p_w=r.p_w, p_holm=r.p_holm, n=r.n,
             spread_shift0=float(spread[(spread.method == m)
                                        & (spread.arm == "0")].spread.mean()),
             spread_ensemble=float(spread[(spread.method == m)
                                          & (spread.arm == ENSEMBLE)].spread.mean()))
        for r, m in zip(fam, keys)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mcq")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--vqarad", action="store_true",
                    help="VQA-RAD: yes/no in both orders (K=2, every "
                         "permutation), image-disjoint split")
    ap.add_argument("--backbone", default="",
                    help="cache suffix of another backbone, e.g. q3vl4b reads "
                         "cache/omnimedvqa_3000_mcq_s1_q3vl4b.npz for shift 1")
    args = ap.parse_args()

    include = BASELINES
    caches = VQARAD_CACHES if args.vqarad else DEFAULT_CACHES
    if args.backbone:
        caches = {s: p.replace(".npz", f"_{args.backbone}.npz")
                  for s, p in caches.items()}
    if args.vqarad:
        df, pos = run(n_seeds=args.seeds, alpha=args.alpha, include=include,
                      caches=caches, k=2,
                      cluster=True)
    else:
        df, pos = run(n_seeds=args.seeds, alpha=args.alpha, include=include,
                      caches=caches)
    st, spread = stats(df, pos)
    df.to_csv(P.metric(f"bench_order_{args.tag}.csv"), index=False)
    pos.to_csv(P.metric(f"bench_order_position_{args.tag}.csv"), index=False)
    st.to_csv(P.metric(f"bench_stats_order_{args.tag}.csv"), index=False)
    if args.vqarad or args.backbone:
        confirmatory(df, pos, args.alpha).to_csv(
            P.metric(f"bench_confirm_order_{args.tag}.csv"), index=False)
    report(df, pos, st, spread, args.tag, args.alpha)
