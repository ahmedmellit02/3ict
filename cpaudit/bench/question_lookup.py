"""A text-prior control for VQA-RAD that goes beyond the yes/no base rate.

`answer_shortcut` asks how far answer strings alone go. On VQA-RAD that
question is empty -- every item offers "yes" and "no" -- so its lookup can only
recover the class split. VQA-RAD does reuse QUESTION strings, though: "is the
image normal?" or "is this an mri?" recur across images. A question-string
lookup is the analogous shortcut there: for each normalised question seen in
training, predict the answer it most often had; for an unseen question, the
training majority. No image, no model.

Ties between "yes" and "no" are broken uniformly at random, seeded per item,
as `answer_shortcut` does, so the baseline gets no position or order prior.
Splits are the paper's image-disjoint ones.

Run: python -m cpaudit.bench.question_lookup --cache cache/vqarad_mcq.npz --tag vqarad_img --cluster
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import load_cache, load_items, split_indices


def _norm(q: str) -> str:
    q = re.sub(r"\s+", " ", str(q).strip().lower())
    return re.sub(r"\s+([?.,!])", r"\1", q)


def run(cache_path: str, n_seeds: int = 20, cluster: bool = True) -> pd.DataFrame:
    cache = load_cache(cache_path)
    items = load_items()
    qs = np.array([_norm(items[str(i)]["question"]) for i in cache.ids], dtype=object)
    ans = cache.answer_idx
    base_hit = (np.where(cache.mask, cache.s_base, np.inf).argmin(1) == ans)
    rows = []
    for seed in range(n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
        for i in idx["train"]:
            counts[qs[i]][ans[i]] += 1
        prior = np.bincount(ans[idx["train"]], minlength=2).astype(float)
        hit, seen_hit, seen_n, base_seen = [], [], 0, []
        for i in idx["test"]:
            c = counts.get(qs[i], prior)
            is_seen = qs[i] in counts
            winners = np.flatnonzero(c == c.max())
            pick = int(np.random.default_rng(seed * 100_003 + int(i)).choice(winners))
            hit.append(pick == ans[i])
            if is_seen:
                seen_n += 1
                seen_hit.append(pick == ans[i])
                base_seen.append(base_hit[i])
        n = len(idx["test"])
        rows.append(dict(seed=seed, n_test=n,
                         lookup_top1=float(np.mean(hit)),
                         question_seen_in_train=seen_n / n,
                         lookup_top1_seen=float(np.mean(seen_hit)) if seen_hit
                         else float("nan"),
                         base_top1_seen=float(np.mean(base_seen)) if base_seen
                         else float("nan"),
                         n_seen=seen_n,
                         base_top1=float(base_hit[idx["test"]].mean()),
                         n_distinct_questions=len(counts)))
    return pd.DataFrame(rows)


def report(df, tag):
    print(f"\n=== Question-string lookup on {tag} ({df.seed.nunique()} seeds) ===")
    m = df.mean(numeric_only=True)
    print(f"top-1, question lookup (no image, no model): {m.lookup_top1:.4f}")
    print(f"top-1, frozen base score on the same items:  {m.base_top1:.4f}")
    print(f"test questions whose string occurs in train:  {m.question_seen_in_train:.4f}")
    print(f"on those: lookup {m.lookup_top1_seen:.4f} vs base {m.base_top1_seen:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cluster", action="store_true")
    args = ap.parse_args()
    df = run(args.cache, args.seeds, args.cluster)
    df.to_csv(P.metric(f"bench_questionlookup_{args.tag}.csv"), index=False)
    report(df, args.tag)
