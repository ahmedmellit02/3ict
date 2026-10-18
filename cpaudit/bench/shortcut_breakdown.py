"""How sure, and how widespread, is the OmniMedVQA answer-string shortcut?

`supervised_control` reports the answer-only head's mean top-1 (0.953) beside
the head with every feature (0.955). This module adds what a reader needs to
weigh that comparison:

  * a paired interval for answer-only minus full-feature top-1 over re-splits;
  * the shortcut broken down by modality and by source dataset, for the
    answer-only head, the string lookup and the frozen base score.

The heads are the paper's: `supervised_control.fit_ce` seeds torch and numpy at
entry, so calling it with the same split, seed and flags trains the same head.
That is asserted against `bench_supervised_mcq.csv` seed by seed before any
new number is written. Both heads are trained separately per re-split with the
same architecture, optimizer, batch size and early-stopping rule; only the
zeroed inputs differ.

The lookup is `answer_shortcut`'s, recomputed per item here, and checked
against `bench_answershortcut_mcq.csv` the same way.

Breakdown rows pool item-split occurrences over the 20 re-splits; an item that
lands in D_test on several re-splits counts each time, so `n` is not a count of
distinct questions.

Run (torch): <conda-py> -m cpaudit.bench.shortcut_breakdown --device cpu
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import load_cache, load_items, source_dataset, split_indices

CACHE = "cache/omnimedvqa_3000_mcq.npz"
TAG = "mcq"


def _options(cache, items):
    out = []
    for i, k in zip(cache.ids, cache.K):
        it = items.get(str(i))
        opts = it["options"] if it else []
        out.append([str(o).strip().lower() for o in opts[:int(k)]])
    return out


def _lookup_hits(cache, opts, idx, seed):
    """Per-test-item correctness of `answer_shortcut`'s lookup, same rng."""
    ans = cache.answer_idx
    seen, right = defaultdict(int), defaultdict(int)
    for i in idx["train"]:
        for k, s in enumerate(opts[i]):
            seen[s] += 1
            right[s] += (k == ans[i])
    prior = sum(right.values()) / sum(seen.values())
    hits = []
    for i in idx["test"]:
        o = opts[i]
        rate = [right[s] / seen[s] if seen.get(s) else prior for s in o]
        best = max(rate)
        winners = [k for k, r in enumerate(rate) if r == best]
        pick = int(np.random.default_rng(seed * 100_003 + i).choice(winners))
        hits.append(pick == ans[i])
    return np.array(hits)


def run(n_seeds=20, device="cpu"):
    from cpaudit.bench.supervised_control import fit_ce

    cache = load_cache(CACHE)
    items = load_items()
    opts = _options(cache, items)
    src = np.array([source_dataset(i) for i in cache.ids], dtype=object)
    ref_sup = pd.read_csv(P.metric(f"bench_supervised_{TAG}.csv"))
    ref_lk = pd.read_csv(P.metric(f"bench_answershortcut_{TAG}.csv"))

    seeds, items_rows = [], []
    for seed in range(n_seeds):
        idx = split_indices(cache, seed)
        test = cache.subset(idx["test"])
        hit = {}
        for name, blind, zb in (("full", False, False),
                                ("answer-only", True, True)):
            f, _ = fit_ce(cache, idx["train"], idx["thr"], seed=seed,
                          image_blind=blind, zero_base=zb, device=device)
            S = f(test)
            hit[name] = np.where(test.mask, S, np.inf).argmin(1) == test.answer_idx
        hit["base"] = (np.where(test.mask, test.s_base, np.inf).argmin(1)
                       == test.answer_idx)
        hit["lookup"] = _lookup_hits(cache, opts, idx, seed)

        # Guards: these are the paper's heads and the paper's lookup.
        r = ref_sup[ref_sup.seed == seed].set_index("method").top1
        for name, method in (("full", "CE+LAC"),
                             ("answer-only", "CE+LAC (answer text only)")):
            assert abs(hit[name].mean() - r[method]) < 1e-9, (seed, name)
        assert abs(hit["lookup"].mean()
                   - ref_lk[ref_lk.seed == seed].lookup_top1.iloc[0]) < 1e-9, seed

        seeds.append(dict(seed=seed, **{f"top1_{k}": float(v.mean())
                                        for k, v in hit.items()}))
        for j, i in enumerate(idx["test"]):
            items_rows.append(dict(seed=seed, id=str(cache.ids[i]),
                                   modality=str(cache.modality[i]),
                                   source=src[i],
                                   **{k: bool(v[j]) for k, v in hit.items()}))
        print(f"  [shortcut_breakdown] seed {seed + 1}/{n_seeds}", flush=True)
    return pd.DataFrame(seeds), pd.DataFrame(items_rows)


def summarize(per_seed, per_item):
    from cpaudit.eval.stats import paired
    r = paired(per_seed["top1_answer-only"].values, per_seed["top1_full"].values,
               "answer-only - full")
    stats = pd.DataFrame([dict(comparison=r.name, delta=r.delta, ci_lo=r.ci_lo,
                               ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w, n=r.n)])
    rows = []
    for axis in ("modality", "source"):
        g = per_item.groupby(axis)
        for grp, d in g:
            rows.append(dict(axis=axis, group=grp, n=len(d),
                             **{f"top1_{m}": float(d[m].mean())
                                for m in ("answer-only", "lookup", "base", "full")}))
    return stats, pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    per_seed, per_item = run(args.seeds, args.device)
    # Raw results first: a failure in the summary must not cost the training run.
    per_seed.to_csv(P.metric(f"bench_shortcutbreakdown_seeds_{TAG}.csv"), index=False)
    per_item.to_csv(P.metric(f"bench_shortcutbreakdown_items_{TAG}.csv"), index=False)
    stats, groups = summarize(per_seed, per_item)
    stats.to_csv(P.metric(f"bench_stats_shortcutbreakdown_{TAG}.csv"), index=False)
    groups.to_csv(P.metric(f"bench_shortcutbreakdown_{TAG}.csv"), index=False)
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    m = groups[groups.axis == "modality"]
    print(m.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    s = groups[groups.axis == "source"]
    print("source datasets:", len(s), "answer-only top-1 range",
          round(s["top1_answer-only"].min(), 3), "-", round(s["top1_answer-only"].max(), 3))
