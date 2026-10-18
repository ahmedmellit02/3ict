"""Letter versus answer: the option-order profiles of both backbones.

VQA-RAD (K = 2). Presenting each question as (yes, no) and as (no, yes) covers
every permutation, so letter and answer separate exactly. Per re-split, each
level's coverage is averaged with equal weight over the other factor's two
levels (as in `order_effect.decompose`), and the two factors are compared
pairwise across re-splits:

    answer: coverage of questions whose answer is "no" minus "yes"
    letter: coverage of answers shown at B minus at A

OmniMedVQA (K = 4, the four cyclic rotations). As in Fig. 3: the range of the
mean-over-splits coverage profile by letter in the original presentation, by
letter averaged over rotations, and by original answer index.

Reads bench_order_position_{tag}.csv (written by `order_effect`); writes
bench_order_decomp.csv. Intervals and p-values come from `cpaudit.eval.stats`.

Run: python -m cpaudit.bench.order_decomp
"""
from __future__ import annotations

import pandas as pd

from cpaudit import paths as P
from cpaudit.eval.stats import paired

#: (backbone, corpus) -> order_effect tag. Option index 0 is "yes" in every
#: VQA-RAD item (options are stored as ["yes", "no"]).
TAGS = {
    ("Qwen2.5-VL-3B", "VQA-RAD"): "vqarad_img",
    ("Qwen3-VL-4B", "VQA-RAD"): "vqarad_img_q3vl4b",
    ("Qwen2.5-VL-3B", "OmniMedVQA"): "mcq",
    ("Qwen3-VL-4B", "OmniMedVQA"): "mcq_q3vl4b",
}


def _single(d, k):
    return d[d.arm.astype(str).isin([str(s) for s in range(k)])]


def binary(d: pd.DataFrame) -> list[dict]:
    rows = []
    s = _single(d, 2)
    for m, g in s.groupby("method"):
        for factor, key, hi, lo in (("answer (no - yes)", "orig_index", 1, 0),
                                    ("letter (B - A)", "letter", 1, 0)):
            w = g.groupby(["seed", key]).coverage.mean().unstack()
            r = paired(w[hi].values, w[lo].values, factor)
            # cov_0: "yes" or letter A; cov_1: "no" or letter B.
            rows.append(dict(method=m, factor=factor, cov_0=w[lo].mean(),
                             cov_1=w[hi].mean(), delta=r.delta,
                             ci_lo=r.ci_lo, ci_hi=r.ci_hi, p_t=r.p_t,
                             p_w=r.p_w, n=r.n))
    return rows


def rotations(d: pd.DataFrame) -> list[dict]:
    rows = []
    s = _single(d, 4)
    for m, g in s.groupby("method"):
        orig = g[g.arm.astype(str) == "0"].groupby("letter").coverage.mean()
        for factor, prof in (("original presentation range", orig),
                             ("letter profile range",
                              g.groupby("letter").coverage.mean()),
                             ("original-index profile range",
                              g.groupby("orig_index").coverage.mean())):
            rows.append(dict(method=m, factor=factor, cov_min=prof.min(),
                             cov_max=prof.max(), delta=prof.max() - prof.min()))
    return rows


def build() -> pd.DataFrame:
    out = []
    for (backbone, corpus), tag in TAGS.items():
        p = P.metric(f"bench_order_position_{tag}.csv")
        if not p.exists():
            print(f"  skip {backbone} / {corpus}: {p.name} not found")
            continue
        d = pd.read_csv(p)
        rows = binary(d) if corpus == "VQA-RAD" else rotations(d)
        out += [dict(backbone=backbone, corpus=corpus, **r) for r in rows]
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = build()
    df.to_csv(P.metric("bench_order_decomp.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda x: f"{x:.4f}"):
        print(df[df.method == "base-LAC"].drop(columns="method").to_string(index=False))
