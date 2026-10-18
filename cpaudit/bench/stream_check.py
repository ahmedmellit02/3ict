"""How much did the old uniform stream matter?

APS and RAPS need a uniform draw per candidate. The earlier implementation
(used for the submitted version of the paper) took it from
`np.random.default_rng(seed)`, restarted on every call to the score and
consumed row by row, so calibration row j and test row j received the SAME
draws. The exchangeability argument wants the test point's auxiliary draw
independent of every calibration point's, and positional reuse does not give
that. The primary protocol therefore keys the draws by
(re-split, item, option) instead (`cpaudit.bench.common.keyed_uniforms`): distinct
items never share one, and the same item-option pair keeps its draw across
configurations, arms and rotations.

This module reruns the whole Table I pipeline -- temperature and RAPS search on
D_thr, calibration on D_cal, evaluation on D_test -- under both streams, with
the same splits and no quantile jitter in either, and pairs them on seed. It is
a sensitivity note on the old implementation, not evidence that the old one was
valid: a small mean difference is a failure to detect one.

Run: python -m cpaudit.bench.stream_check --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import (ALPHA, PostHoc, _POSTHOC_FN, calibrate_marginal,
                              load_cache, posthoc_grid, split_indices,
                              thr_halves)
from cpaudit.calib.conformal import predict_sets
from cpaudit.eval.metrics import avg_set_size, coverage

METHODS = ("APS", "RAPS")
STREAMS = ("keyed", "reset")


def _select(method, stream, sub_a, sub_b, seed, alpha):
    """`select_config`, with the stream made explicit."""
    from cpaudit.bench.common import sel_size
    best, best_cfg = np.inf, None
    for cfg in posthoc_grid(method):
        sc = PostHoc(_POSTHOC_FN[method], cfg, seed, stream=stream)
        v = sel_size(sc.score(sub_a), sub_a, sc.score(sub_b), sub_b, seed, alpha)
        if v < best:
            best, best_cfg = v, cfg
    return best_cfg


def run(cache_path, n_seeds=20, alpha=ALPHA, cluster=False):
    cache = load_cache(cache_path)
    rows = []
    for seed in range(n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
        sub_a, sub_b = thr_halves(cache, idx["thr"], seed)
        for m in METHODS:
            for stream in STREAMS:
                cfg = _select(m, stream, sub_a, sub_b, seed, alpha)
                sc = PostHoc(_POSTHOC_FN[m], cfg, seed, stream=stream)
                S_c, S_t = sc.score(cal), sc.score(test)
                tau = calibrate_marginal(S_c[np.arange(cal.n), cal.answer_idx],
                                         alpha, seed=seed)
                sets, empty = predict_sets(S_t, test.mask, tau)
                rows.append(dict(seed=seed, method=m, stream=stream,
                                 config=";".join(f"{k}={v}" for k, v in
                                                 sorted(cfg.items())),
                                 coverage=coverage(sets, test.answer_idx),
                                 avg_size=avg_set_size(sets),
                                 empty_rate=float(empty.mean())))
    return pd.DataFrame(rows)


def stats(df):
    """Reset minus keyed, paired on seed, Holm over the four tests."""
    from cpaudit.eval.stats import holm, paired
    fam, keys = [], []
    for m in METHODS:
        for metric in ("coverage", "avg_size"):
            d = df[df.method == m]
            a = d[d.stream == "reset"].sort_values("seed")[metric].values
            b = d[d.stream == "keyed"].sort_values("seed")[metric].values
            fam.append(paired(a, b, f"{m} {metric}"))
            keys.append((m, metric))
    holm(fam)
    out = []
    for r, (m, k) in zip(fam, keys):
        d = df[df.method == m].pivot_table(index="seed", columns="stream",
                                           values="config", aggfunc="first")
        out.append(dict(method=m, metric=k, delta=r.delta, ci_lo=r.ci_lo,
                        ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w, p_holm=r.p_holm,
                        n=r.n, frac_config_differs=float(
                            (d["reset"] != d["keyed"]).mean())))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cluster", action="store_true")
    args = ap.parse_args()
    df = run(args.cache, args.seeds, cluster=args.cluster)
    st = stats(df)
    df.to_csv(P.metric(f"bench_streamcheck_{args.tag}.csv"), index=False)
    st.to_csv(P.metric(f"bench_stats_streamcheck_{args.tag}.csv"), index=False)
    print(df.pivot_table(index="method", columns="stream",
                         values=["coverage", "avg_size"]).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
