"""One table of the protocol effects the 3ICT paper reports (reviewer R2.4).

Every row is a paired contrast over the 20 re-splits: an intervention against
its reference protocol, same splits, same score. Coverage and set-size changes
are means of per-split differences with Student-t 95% intervals
(`cpaudit.eval.stats.paired`). The p-value is the Holm-adjusted paired-t p of the
row's primary contrast, taken from the family that contrast belongs to:

  reuse, full      leakage confirmatory family (3 scores, per dataset)
  reuse, matched   the exploratory M-sweep family (per score)
  no randomization randomization family (per metric)
  order ensemble   ensembling confirmatory family (position spread)
  image split      split-unit family (nine comparisons)

Nothing here is recomputed from rounded published values; everything is read
from the per-split CSVs.

Run: python -m cpaudit.bench.effects_table
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.eval.stats import paired

DATA = {"mcq": "OmniMedVQA", "vqarad_img": "VQA-RAD"}
SCORES = ("base-LAC", "APS", "RAPS")
NAME = {"base-LAC": "LAC", "APS": "APS", "RAPS": "RAPS"}


def _csv(name):
    return pd.read_csv(P.metric(name))


def _pair(a, b):
    r = paired(np.asarray(a), np.asarray(b))
    return r.delta, r.ci_lo, r.ci_hi


def reuse_rows():
    out = []
    for tag, ds in DATA.items():
        t = _csv(f"bench_tuningleak_{tag}.csv")
        conf = _csv(f"bench_confirm_tuningleak_{tag}.csv").set_index("method")
        sweep = _csv(f"bench_stats_tuningleak_{tag}.csv")
        for m in SCORES:
            d = t[(t.method == m) & (t.M == t.grid_size)]
            cov = d.pivot_table(index="seed", columns="arm", values="coverage")
            size = d.pivot_table(index="seed", columns="arm", values="avg_size")
            for arm, block in (("leaky-full", "reuse, full"),
                               ("leaky", "reuse, matched")):
                dc = _pair(cov[arm], cov["clean"])
                dz = _pair(size[arm], size["clean"])
                if block == "reuse, full":
                    # The confirmatory CSV is the source the text quotes; the
                    # same mean computed in another order can land on the
                    # other side of a rounding tie (-0.00725).
                    dc = (conf.loc[m, "delta"], conf.loc[m, "ci_lo"],
                          conf.loc[m, "ci_hi"])
                if block == "reuse, full":
                    p = conf.loc[m, "p_holm"]
                else:
                    s = sweep[(sweep.method == m) & (sweep.arm == "leaky")
                              & (sweep.M == d.grid_size.iloc[0])]
                    p = s.p_holm.iloc[0] if len(s) else np.nan
                out.append(dict(block=block, dataset=ds, score=NAME[m],
                                d_cov=dc[0], cov_lo=dc[1], cov_hi=dc[2],
                                d_size=dz[0], size_lo=dz[1], size_hi=dz[2],
                                p_holm=p))
    return out


def randomization_rows():
    out = []
    for tag, ds in DATA.items():
        r = _csv(f"bench_random_{tag}.csv")
        r = r[(r.jitter == 0) & (r.method == "APS")]
        st = _csv(f"bench_stats_random_{tag}.csv")
        p = st[(st.method == "APS") & (st.effect == "randomize")
               & (st.metric == "coverage")].p_holm.iloc[0]
        det = r[r.randomize.astype(str) == "False"].sort_values("seed")
        ran = r[r.randomize.astype(str) == "True"].sort_values("seed")
        dc = _pair(det.coverage.values, ran.coverage.values)
        dz = _pair(det.avg_size.values, ran.avg_size.values)
        out.append(dict(block="no randomization", dataset=ds, score="APS",
                        d_cov=dc[0], cov_lo=dc[1], cov_hi=dc[2],
                        d_size=dz[0], size_lo=dz[1], size_hi=dz[2], p_holm=p))
    return out


def ensemble_rows():
    out = []
    o = _csv("bench_order_mcq.csv")
    pos = _csv("bench_order_position_mcq.csv")
    conf = _csv("bench_confirm_order_mcq.csv").set_index("method")
    spread = (pos.groupby(["seed", "arm", "method"]).coverage
                 .agg(lambda s: s.max() - s.min()).rename("spread").reset_index())
    for m in SCORES:
        sp = spread[spread.method == m].pivot_table(index="seed", columns="arm",
                                                    values="spread")
        sz = o[o.method == m].pivot_table(index="seed", columns="arm",
                                          values="avg_size")
        dc = _pair(sp["ensemble"], sp["0"])
        dz = _pair(sz["ensemble"], sz["0"])
        out.append(dict(block="order ensemble", dataset="OmniMedVQA",
                        score=NAME[m], d_cov=dc[0], cov_lo=dc[1], cov_hi=dc[2],
                        d_size=dz[0], size_lo=dz[1], size_hi=dz[2],
                        p_holm=conf.loc[m, "p_holm"], cov_is_spread=True))
    return out


def split_rows():
    out = []
    st = _csv("bench_stats_splitunit_vqarad.csv").set_index(["method", "metric"])
    for m in SCORES:
        c, z = st.loc[(m, "coverage")], st.loc[(m, "avg_size")]
        out.append(dict(block="image split", dataset="VQA-RAD", score=NAME[m],
                        d_cov=c.delta, cov_lo=c.ci_lo, cov_hi=c.ci_hi,
                        d_size=z.delta, size_lo=z.ci_lo, size_hi=z.ci_hi,
                        p_holm=c.p_holm))
    return out


def build():
    rows = reuse_rows() + randomization_rows() + ensemble_rows() + split_rows()
    df = pd.DataFrame(rows)
    df["cov_is_spread"] = df.get("cov_is_spread", False).fillna(False)
    return df


def _f(x, dp=4):
    s = f"{x:+.{dp}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


BLOCKS = ("reuse, full", "reuse, matched", "no randomization", "order ensemble",
          "image split")


def _p(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def latex(df):
    """Rows grouped by intervention, a rule between groups; an identically
    zero contrast (binary LAC under reuse) is written as such."""
    df = df.assign(_b=df.block.map(BLOCKS.index),
                   _d=df.dataset.map({"OmniMedVQA": 0, "VQA-RAD": 1}),
                   _s=df.score.map({"LAC": 0, "APS": 1, "RAPS": 2}))
    df = df.sort_values(["_b", "_d", "_s"])
    lines, prev = [], None
    for _, r in df.iterrows():
        if prev is not None and r.block != prev:
            lines.append("\\midrule")
        prev = r.block
        zero = (r.d_cov == 0 and r.cov_lo == 0 and r.cov_hi == 0
                and r.d_size == 0 and r.size_lo == 0 and r.size_hi == 0)
        cov = ("0 exactly" if zero else
               f"{_f(r.d_cov)} [{_f(r.cov_lo)}, {_f(r.cov_hi)}]")
        size = ("0 exactly" if zero else
                f"{_f(r.d_size, 3)} [{_f(r.size_lo, 3)}, {_f(r.size_hi, 3)}]")
        lines.append(f"{r.block} & {r.dataset} & {r.score} & {cov} & {size} "
                     f"& {_p(r.p_holm)} \\\\")
    return "\n".join(lines)


if __name__ == "__main__":
    df = build()
    df.to_csv(P.metric("bench_effects_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print()
    print(latex(df))
