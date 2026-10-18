"""Group-stratified, disjoint four-way split.

    D = D_train  u  D_thr  u  D_cal  u  D_test        (all disjoint)

Stratifying by the subgroup key keeps every group populated in D_cal and
D_test, which the Mondrian floor (n_g >= 1/alpha) needs. D_cal is produced here
but is *sacred*: nothing in training may read it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

SPLIT_NAMES = ("train", "thr", "cal", "test")


def four_way_split(
    items: list[dict],
    fractions: dict[str, float],
    group_key: str = "modality",
    seed: int = 0,
) -> dict[str, list[int]]:
    """Return disjoint index lists keyed by split name.

    Parameters
    ----------
    items       : item dicts carrying `group_key`
    fractions   : {'train','thr','cal','test'} -> float, summing to ~1.0
    group_key   : field to stratify on ('modality' or 'question_type')
    seed        : RNG seed
    """
    missing = set(SPLIT_NAMES) - set(fractions)
    if missing:
        raise ValueError(f"fractions missing splits: {missing}")
    total = sum(fractions[s] for s in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    # Bucket indices by group.
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_group[str(it.get(group_key, "Unknown"))].append(i)

    out: dict[str, list[int]] = {s: [] for s in SPLIT_NAMES}
    # Cumulative cut points in [0,1].
    cuts = np.cumsum([fractions[s] for s in SPLIT_NAMES])  # -> train,thr,cal,test edges
    for _, idxs in sorted(by_group.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        # Integer boundaries so each group contributes proportionally.
        b = (cuts * n).round().astype(int)
        b[-1] = n  # guard rounding
        prev = 0
        for s, edge in zip(SPLIT_NAMES, b):
            out[s].extend(idxs[prev:edge].tolist())
            prev = edge

    # Sanity: disjoint + complete.
    _assert_partition(out, len(items))
    return out


def _assert_partition(out: dict[str, list[int]], n: int) -> None:
    seen = set()
    for s in SPLIT_NAMES:
        cur = set(out[s])
        if cur & seen:
            raise AssertionError(f"split {s} overlaps another split")
        seen |= cur
    if len(seen) != n:
        raise AssertionError(f"split covers {len(seen)} of {n} items")


def grouped_four_way_split(
    items: list[dict],
    fractions: dict[str, float],
    group_key: str = "modality",
    cluster_key: str = "image_path",
    seed: int = 0,
) -> dict[str, list[int]]:
    """Four-way split in which a whole CLUSTER lands in exactly one partition.

    `four_way_split` assigns items independently, which is fine when items are
    the unit of interest but not when several items share an underlying object.
    VQA-RAD asks about 4.05 questions per image, so an item-level split puts
    questions about the same image on both sides of the calibration/test
    boundary. The conformal guarantee survives that -- a random partition of a
    fixed item set is exchangeable however the items are correlated -- but the
    independence that binomial and Beta references assume does not, and neither
    does any reading of the result as generalization to an unseen image.

    Clusters are shuffled within each stratum and then assigned greedily to
    whichever split is furthest below its target ITEM quota, so the realized
    item proportions stay close to `fractions` even though cluster sizes vary.
    Exact proportions are unattainable in general: with 294 clusters and a
    0.15 quota, one oversized cluster moves the realized share by ~1%.
    """
    missing = set(SPLIT_NAMES) - set(fractions)
    if missing:
        raise ValueError(f"fractions missing splits: {missing}")

    rng = np.random.default_rng(seed)
    # cluster -> (its item indices, its stratum)
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        clusters[str(it.get(cluster_key, i))].append(i)
    stratum = {c: str(items[idx[0]].get(group_key, "Unknown"))
               for c, idx in clusters.items()}

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for c in clusters:
        by_stratum[stratum[c]].append(c)

    out: dict[str, list[int]] = {s: [] for s in SPLIT_NAMES}
    for g, cs in sorted(by_stratum.items()):
        cs = list(cs)
        rng.shuffle(cs)
        n_items = sum(len(clusters[c]) for c in cs)
        quota = {s: fractions[s] * n_items for s in SPLIT_NAMES}
        filled = {s: 0 for s in SPLIT_NAMES}
        # Shuffled order, NOT largest-first. Largest-first tracks the item
        # quotas more tightly (max proportion error 0.0007 vs 0.0041 on
        # VQA-RAD, stratified by released organ) but sends the biggest
        # clusters to whichever split has the largest quota, so train ends up
        # holding images with many more questions than the other partitions:
        # 5.23 against 3.18 in thr, a spread of 2.05 questions per image. That
        # is a composition difference between the sets being compared, and it
        # is worse than the quota error it fixes. Shuffled order holds the
        # spread to 0.61 and still lands within 0.5% of the target proportions. Regenerate both with
        # `python -m cpaudit.bench.split_unit --cache cache/vqarad_mcq.npz
        # --tag vqarad`.
        for c in cs:
            s = max(SPLIT_NAMES, key=lambda s: quota[s] - filled[s])
            out[s].extend(clusters[c])
            filled[s] += len(clusters[c])

    _assert_partition(out, len(items))
    return out


def cluster_overlap(out: dict[str, list[int]], cluster_of) -> dict[str, float]:
    """Diagnostic: how much cluster leakage a split actually has.

    `cluster_of` maps item index -> cluster id; either a callable or an array
    indexed by item. Returns the share of test items whose cluster also appears
    in train/thr/cal, and the share of clusters spanning more than one split.
    Both are 0 for a cluster-disjoint split.

    Index containers may be lists or numpy arrays -- `split_indices` returns the
    latter, and `if array` raises rather than testing emptiness.
    """
    cid = cluster_of if callable(cluster_of) else (lambda i: cluster_of[i])
    where: dict[str, set] = defaultdict(set)
    for s in SPLIT_NAMES:
        for i in out[s]:
            where[cid(i)].add(s)
    others = {cid(i) for s in ("train", "thr", "cal") for i in out[s]}
    test = list(out["test"])
    return {
        "test_items_sharing_cluster": (
            float(np.mean([cid(i) in others for i in test])) if len(test) else 0.0),
        "clusters_spanning_splits": float(
            np.mean([len(v) > 1 for v in where.values()])) if where else 0.0,
    }
