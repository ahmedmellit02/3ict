"""How much does the split UNIT matter, before any score is computed?

The setup section quotes several numbers about VQA-RAD's clustering -- how many
questions share an image, how much of the test set has its image on the other
side of the boundary, what the cluster-disjoint rule costs in proportion
accuracy, and why clusters are placed in shuffled rather than size-descending
order. Those were measured ad hoc while the split was being written, which is
exactly the situation `claims.py` exists to prevent: a number in the manuscript
with no CSV behind it cannot be re-checked after the code changes.

This module regenerates all of them.

Three assignment rules are compared on identical strata and seeds:
  item      -- `four_way_split`, items placed independently (the default)
  cluster   -- `grouped_four_way_split`, whole images, shuffled order (used)
  cluster-desc -- the same, largest cluster first

The last one is the tempting variant: taking big clusters first tracks the item
quotas more tightly. It also sends every big image to whichever split has the
largest remaining quota, which is systematically the training split, so the
partitions end up differing in questions per image. That is a composition
difference between the sets being compared, and it is worse than the quota
error it fixes.

Run: python -m cpaudit.bench.split_unit --cache cache/vqarad_mcq.npz --tag vqarad
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import SPLIT_FRACS, image_of, load_items
from cpaudit.bench.common import load_cache
from cpaudit.data.split import (SPLIT_NAMES, cluster_overlap, four_way_split,
                            grouped_four_way_split)


def _desc_split(rows, fractions, group_key, cluster_key, seed):
    """`grouped_four_way_split` with clusters taken largest-first.

    Kept here rather than as a flag on the real splitter: it is the rejected
    alternative, and the only thing it is used for is measuring what it costs.
    """
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(rows):
        clusters[str(it.get(cluster_key, i))].append(i)
    stratum = {c: str(rows[ix[0]].get(group_key, "Unknown"))
               for c, ix in clusters.items()}
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for c in clusters:
        by_stratum[stratum[c]].append(c)

    out: dict[str, list[int]] = {s: [] for s in SPLIT_NAMES}
    for g, cs in sorted(by_stratum.items()):
        cs = sorted(cs, key=lambda c: -len(clusters[c]))
        n_items = sum(len(clusters[c]) for c in cs)
        quota = {s: fractions[s] * n_items for s in SPLIT_NAMES}
        filled = {s: 0 for s in SPLIT_NAMES}
        for c in cs:
            s = max(SPLIT_NAMES, key=lambda s: quota[s] - filled[s])
            out[s].extend(clusters[c])
            filled[s] += len(clusters[c])
    return out


def _rows(cache, img):
    return [{"modality": m, "image_path": p} for m, p in zip(cache.modality, img)]


def run(cache_path: str, n_seeds: int = 20) -> pd.DataFrame:
    cache = load_cache(cache_path)
    items = load_items()
    img = image_of(cache, items)
    rows = _rows(cache, img)
    rules = {
        "item": lambda s: four_way_split([{"modality": m} for m in cache.modality],
                                         SPLIT_FRACS, "modality", seed=s),
        "cluster": lambda s: grouped_four_way_split(rows, SPLIT_FRACS, "modality",
                                                    "image_path", seed=s),
        "cluster-desc": lambda s: _desc_split(rows, SPLIT_FRACS, "modality",
                                              "image_path", s),
    }
    out = []
    for rule, fn in rules.items():
        for seed in range(n_seeds):
            sp = {s: np.array(v) for s, v in fn(seed).items()}
            ov = cluster_overlap(sp, img)
            qpi = {s: len(sp[s]) / len(np.unique(img[sp[s]])) for s in SPLIT_NAMES}
            r = dict(rule=rule, seed=seed,
                     test_share=len(sp["test"]) / cache.n,
                     max_frac_err=max(abs(len(sp[s]) / cache.n - SPLIT_FRACS[s])
                                      for s in SPLIT_NAMES),
                     test_items_sharing_image=ov["test_items_sharing_cluster"],
                     images_spanning_splits=ov["clusters_spanning_splits"],
                     qpi_spread=max(qpi.values()) - min(qpi.values()))
            r.update({f"qpi_{s}": qpi[s] for s in SPLIT_NAMES})
            out.append(r)
    return pd.DataFrame(out)


def one_per_cluster(cache_path: str, n_seeds: int = 20, alpha: float = 0.10,
                    methods=None) -> pd.DataFrame:
    """Sample one question per image, then split those items in the ordinary way.

    This is the design the ordinary split-conformal theorem covers without
    qualification: every unit appears once, the partition is an unrestricted
    uniform one over those units, and the test metric weights each image
    equally. The price is the corpus shrinking to its number of images.

    The sampled subset changes with the seed, so variation here mixes the split
    with the choice of representative question. That is intended -- both are
    part of the procedure being evaluated -- but it is not the same estimand as
    the main run.
    """
    from cpaudit.bench.common import POSTHOC, build_scorers
    from cpaudit.calib.conformal import predict_sets
    from cpaudit.bench.common import calibrate_marginal
    from cpaudit.eval.metrics import avg_set_size, coverage

    methods = methods or POSTHOC
    cache = load_cache(cache_path)
    items = load_items()
    img = image_of(cache, items)
    order = np.argsort(img, kind="stable")
    _, first = np.unique(img[order], return_index=True)
    groups = np.split(order, first[1:])

    out = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        pick = np.array(sorted(int(rng.choice(g)) for g in groups))
        sub = cache.subset(pick)
        # Unstratified: one stratum for every item, so this is a uniformly
        # random permutation cut at the split fractions. Stratifying on organ,
        # as an earlier version did, fixes per-stratum quotas and the pooled
        # calibration and test scores are then not exactly exchangeable --
        # which is the whole point of this design.
        sp = four_way_split([{} for _ in range(sub.n)], SPLIT_FRACS,
                            "_none", seed=seed)
        idx = {s: np.array(v) for s, v in sp.items()}
        cal, test = sub.subset(idx["cal"]), sub.subset(idx["test"])
        scorers, _ = build_scorers(sub, idx, seed, include=methods)
        for m, sc in scorers.items():
            S_c, S_t = sc.score(cal), sc.score(test)
            tau = calibrate_marginal(S_c[np.arange(cal.n), cal.answer_idx],
                                     alpha, seed=seed)
            sets, empty = predict_sets(S_t, test.mask, tau)
            # Pre-fallback too: the finite-sample upper bound describes the
            # unmodified set {y : s <= tau}. The argmax fallback only enlarges
            # sets, so post-fallback coverage can sit above that bound without
            # any violation, and only the pre-fallback number is comparable.
            raw = sets & ~empty[:, None]
            out.append(dict(seed=seed, method=m, n_items=int(sub.n),
                            n_cal=int(cal.n), n_test=int(test.n),
                            coverage=coverage(sets, test.answer_idx),
                            coverage_prefallback=float(
                                raw[np.arange(test.n), test.answer_idx].mean()),
                            avg_size=avg_set_size(sets),
                            empty_rate=float(empty.mean())))
    return pd.DataFrame(out)


def paired_stats(item_tag: str, cluster_tag: str, alpha: float = 0.10):
    """Item-level against image-disjoint on the same seeds.

    Reads the two `bench_alpha_*` runs rather than recomputing them, so the
    comparison is between exactly the numbers the paper reports. Holm runs
    across all nine tests at once (three scores x three metrics): they are one
    family asking a single question -- does the split unit change the answer --
    and correcting within it is what keeps the negative result honest.
    """
    from cpaudit.eval.stats import holm, paired

    def load(tag):
        d = pd.read_csv(P.metric(f"bench_alpha_{tag}.csv"))
        return d[(d.alpha == alpha) & (d.arm == "matched")]

    a, b = load(item_tag), load(cluster_tag)
    fam, meta = [], []
    for m in sorted(set(a.method) & set(b.method)):
        x = a[a.method == m].sort_values("seed")
        y = b[b.method == m].sort_values("seed")
        if list(x.seed) != list(y.seed):
            raise ValueError(f"{m}: the two runs are not on the same seeds")
        for col in ("coverage", "avg_size", "top1"):
            fam.append(paired(y[col].values, x[col].values, name=f"{m}/{col}"))
            meta.append((m, col, float(x[col].mean()), float(y[col].mean())))
    holm(fam)
    return pd.DataFrame([
        dict(method=m, metric=col, item_level=xi, image_disjoint=yi,
             delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p_t=r.p_t,
             p_w=r.p_w, p_holm=r.p_holm, n=r.n)
        for r, (m, col, xi, yi) in zip(fam, meta)])


def leak_did(item_tag: str, cluster_tag: str):
    """Difference-in-differences for the tuning leak across split policies.

    A leak that is significant under one split rule and not under the other is
    not, by itself, evidence that the rules differ: the comparison the claim
    needs is

        [(leaky-full - clean) | image-disjoint] - [(leaky-full - clean) | item],

    paired on seed, at each score's own grid size. Even this estimates the total
    change between two POLICIES. They also change which items sit in which
    partition, the calibration count and the search problem itself, so a
    difference here would still not isolate image sharing as the mechanism.
    """
    from cpaudit.eval.stats import holm, paired

    def leak(tag):
        d = pd.read_csv(P.metric(f"bench_tuningleak_{tag}.csv"))
        d = d[d.M == d.grid_size]
        return {m: (g.pivot_table(index="seed", columns="arm", values="coverage")
                    .pipe(lambda x: x["leaky-full"] - x["clean"]).sort_index())
                for m, g in d.groupby("method")}

    a, b = leak(item_tag), leak(cluster_tag)
    fam, meta = [], []
    for m in sorted(set(a) & set(b)):
        if list(a[m].index) != list(b[m].index):
            raise ValueError(f"{m}: the two runs are not on the same seeds")
        fam.append(paired(b[m].values, a[m].values, name=m))
        meta.append((m, float(a[m].mean()), float(b[m].mean())))
    holm(fam)
    return pd.DataFrame([
        dict(method=m, leak_item=li, leak_cluster=lc, did=r.delta,
             ci_lo=r.ci_lo, ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w,
             p_holm=r.p_holm, n=r.n)
        for r, (m, li, lc) in zip(fam, meta)])


def report(df, cache_path, n_images, n_items):
    print(f"\n=== Split unit on {cache_path} ===")
    print(f"{n_items} items over {n_images} images "
          f"({n_items / n_images:.2f} questions per image)\n")
    g = df.groupby("rule").mean(numeric_only=True).drop(columns="seed")
    show = ["test_items_sharing_image", "images_spanning_splits", "test_share",
            "max_frac_err", "qpi_train", "qpi_thr", "qpi_cal", "qpi_test",
            "qpi_spread"]
    print(g[show].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\ntest_items_sharing_image: share of test items whose image also "
          "appears in another\npartition -- 0 by construction for both cluster "
          "rules.\nqpi_*: questions per image inside each partition; a spread "
          "means the partitions differ\nin composition, which confounds any "
          "comparison across them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--paired", nargs=2, metavar=("ITEM_TAG", "CLUSTER_TAG"),
                    help="also pair two existing bench_alpha_* runs, e.g. "
                         "--paired vqarad_mcq vqarad_img")
    ap.add_argument("--one-per-image", action="store_true",
                    help="also run the one-question-per-image sensitivity check")
    args = ap.parse_args()
    df = run(args.cache, args.seeds)
    df.to_csv(P.metric(f"bench_splitunit_{args.tag}.csv"), index=False)
    c = load_cache(args.cache)
    report(df, args.cache, len(np.unique(image_of(c, load_items()))), c.n)
    if args.paired:
        did = leak_did(*args.paired)
        did.to_csv(P.metric(f"bench_stats_leakdid_{args.tag}.csv"), index=False)
        print("\n=== Does the split policy change the SIZE of the leak? ===")
        print(did.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n`did` is the extra coverage the leak costs under image-disjoint "
              "splitting.\nA change in significance between the two rules is not "
              "a significant change.")
    if args.one_per_image:
        from cpaudit.eval.stats import ci
        o = one_per_cluster(args.cache, args.seeds)
        o.to_csv(P.metric(f"bench_oneperimage_{args.tag}.csv"), index=False)
        print("\n=== One question per image, ordinary item-level split ===")
        print(f"{int(o.n_items.iloc[0])} items, {int(o.n_cal.iloc[0])} "
              f"calibration and {int(o.n_test.iloc[0])} test on average")
        for m, g in o.groupby("method"):
            lo, hi = ci(g.sort_values("seed").coverage.values)
            print(f"  {m:<9s} coverage {g.coverage.mean():.4f} "
                  f"[{lo:.4f},{hi:.4f}]   size {g.avg_size.mean():.4f}")
        print("\nEvery unit appears once and the partition is an ordinary "
              "uniform one, so this is\nthe design the split-conformal theorem "
              "covers directly. It is also a much smaller\ncorpus, and it "
              "targets a one-question-per-image distribution rather than the\n"
              "corpus's own question mix.")
    if args.paired:
        st = paired_stats(*args.paired)
        st.to_csv(P.metric(f"bench_stats_splitunit_{args.tag}.csv"), index=False)
        print("\n=== Same seeds, item-level vs image-disjoint "
              "(Holm across all nine) ===")
        print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nNothing surviving Holm here means the split unit did not move "
              "the headline\nnumbers; it does not mean the two rules are "
              "equivalent, only that 20 seeds\ncannot separate them.")
