"""What exactly is the answer-text shortcut made of?

`supervised_control` shows that an MLP given nothing but the frozen input
embeddings of the option strings reaches top-1 0.953 on OmniMedVQA, against
0.955 for a head with every feature. That is the paper's scope limit, and a
reader is entitled to ask what the head is exploiting.

This module answers it without a model. For every option string it counts how
often that string is the correct answer, on the TRAINING split only, and at test
time picks the option with the highest such rate. No embeddings, no gradients,
no image, no question -- a lookup table over answer text.

If the lookup alone approaches the learned head, the shortcut is answer-string
identity: the corpus reuses option strings, and some of them are usually right.
If the lookup falls well short, the head is generalizing over answer *text*
rather than memorizing strings, which is a different (and less alarming) claim.
Both outcomes are worth reporting, and the gap between them is the part of the
shortcut that needs an encoder.

Reported alongside: how much answer vocabulary test items share with train,
which bounds how far a pure lookup could ever get.

Run: python -m cpaudit.bench.answer_shortcut --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import load_items, split_indices
from cpaudit.bench.common import load_cache


def _options(cache, items):
    """Per item, the list of option strings in cache indexing (lowercased)."""
    out = []
    for i, k in zip(cache.ids, cache.K):
        it = items.get(str(i))
        opts = it["options"] if it else []
        out.append([str(o).strip().lower() for o in opts[:int(k)]])
    return out


def run(cache_path: str, n_seeds: int = 20, cluster: bool = False) -> pd.DataFrame:
    cache = load_cache(cache_path)
    items = load_items()
    opts = _options(cache, items)
    ans = cache.answer_idx
    rows = []
    for seed in range(n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        seen, right = defaultdict(int), defaultdict(int)
        for i in idx["train"]:
            for k, s in enumerate(opts[i]):
                seen[s] += 1
                right[s] += (k == ans[i])
        prior = (sum(right.values()) / sum(seen.values())) if seen else 0.0

        hit = cover = tie = 0
        n = 0
        hit_seen = hit_unseen = n_unseen = 0
        for i in idx["test"]:
            o = opts[i]
            if not o:
                continue
            n += 1
            # Rate, not count: times correct / times offered among training
            # options, so a string that is offered often is not favoured for
            # that alone. Unseen strings get the pooled rate.
            rate = [right[s] / seen[s] if seen.get(s) else prior for s in o]
            best = max(rate)
            winners = [k for k, r in enumerate(rate) if r == best]
            # A tie is broken uniformly rather than by option order: breaking by
            # index would smuggle a position prior into a text-only baseline.
            pick = int(np.random.default_rng(seed * 100_003 + i).choice(winners))
            hit += pick == ans[i]
            tie += len(winners) > 1
            was_seen = seen.get(o[ans[i]], 0) > 0
            cover += was_seen
            if was_seen:
                hit_seen += pick == ans[i]
            else:
                hit_unseen += pick == ans[i]
                n_unseen += 1
        rows.append(dict(seed=seed, n_test=n,
                         lookup_top1=hit / n,
                         answer_seen_in_train=cover / n,
                         lookup_top1_seen=hit_seen / max(cover, 1),
                         lookup_top1_unseen=(hit_unseen / n_unseen
                                             if n_unseen else float("nan")),
                         n_unseen=n_unseen,
                         tied_fraction=tie / n,
                         n_distinct_strings=len(seen)))
    return pd.DataFrame(rows)


def report(df, tag):
    print(f"\n=== Answer-string lookup baseline on {tag} "
          f"({df.seed.nunique()} seeds) ===")
    print(f"top-1 from answer text alone, no model: "
          f"{df.lookup_top1.mean():.4f}")
    print(f"test items whose correct answer string was seen in train: "
          f"{df.answer_seen_in_train.mean():.4f}")
    print(f"test items where the lookup had to break a tie: "
          f"{df.tied_fraction.mean():.4f}")
    print(f"distinct option strings in train: "
          f"{df.n_distinct_strings.mean():.0f}")
    print("\nCompare against bench_supervised_*.csv. A lookup close to the "
          "learned head means the\nshortcut is string identity; a large gap "
          "means the head generalizes over answer text.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cluster", action="store_true")
    args = ap.parse_args()
    df = run(args.cache, args.seeds, args.cluster)
    df.to_csv(P.metric(f"bench_answershortcut_{args.tag}.csv"), index=False)
    report(df, args.tag)
