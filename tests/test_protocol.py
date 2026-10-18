"""Protocol tests: the implementation does what the paper says it does.

  * keyed uniforms: one draw per (re-split, item, option), shared by no two items;
  * the vectorized APS/RAPS scores equal a per-candidate loop given the same
    draws, and deterministic APS gives the lower-ranked candidate exactly 1.0;
  * the threshold is an actual calibration score (no jitter);
  * every grouping axis labels every item;
  * re-split 0 reproduces the recorded configurations
    (`paper/materials/selected_configs.csv`) and Table I's per-split rows
    (`results/metrics/bench_alpha_mcq.csv`) exactly.

Needs the OmniMedVQA score cache in `cache/`:

    mkdir cache && cp paper/materials/scores/*.npz cache/

Run:  python -m tests.test_protocol
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench import common as C

CACHE = P.ROOT / "cache" / "omnimedvqa_3000_mcq.npz"
SEED = 0


def test_keyed_uniforms():
    """Draws are fixed by (seed, item, option), not by row position."""
    ids = np.array([f"item_{i}" for i in range(400)])
    u = C.keyed_uniforms(ids, 3, 4)
    assert u.shape == (400, 4) and (u >= 0).all() and (u < 1).all()
    # the same item keeps its draws in any subset and order
    perm = np.random.default_rng(0).permutation(400)[:150]
    assert np.array_equal(C.keyed_uniforms(ids[perm], 3, 4), u[perm])
    # distinct items: no shared draw
    assert len(np.unique(u)) == u.size
    # another re-split draws afresh
    assert not np.array_equal(C.keyed_uniforms(ids, 4, 4), u)
    # roughly uniform
    assert abs(u.mean() - 0.5) < 0.02 and abs(u.var() - 1 / 12) < 0.01
    print("keyed uniforms: subset-invariant, unshared across items, fresh per seed")


def test_vectorized_scores_match_loop():
    """Given the same draws per candidate, the vectorized APS/RAPS equal a
    per-candidate loop; deterministic APS gives the last rank exactly 1.0."""
    from cpaudit.baselines.posthoc import aps_scores, probs_from_base, raps_scores
    rng = np.random.default_rng(1)
    n, k = 300, 4
    s_base = rng.normal(size=(n, k))
    mask = np.ones((n, k), bool)
    mask[:50, 2:] = False                      # some K=2 rows
    s_base = np.where(mask, s_base, np.inf)
    # Reproduce the loop's draws in candidate indexing: it consumes one draw
    # per real candidate in descending-probability order, row by row.
    p = probs_from_base(s_base, mask, 1.0)
    g = np.random.default_rng(7)
    u = np.zeros((n, k))
    for i in range(n):
        idx = np.where(mask[i])[0]
        for y in idx[np.argsort(-p[i, idx])]:
            u[i, y] = g.uniform()
    for fn, kw in ((aps_scores, {}), (raps_scores, dict(gamma=0.05, k_reg=2))):
        loop = fn(s_base, mask, seed=7, **kw)
        vec = fn(s_base, mask, seed=7, u=u, **kw)
        assert np.allclose(np.where(mask, loop, 0), np.where(mask, vec, 0),
                           atol=1e-12), fn.__name__
        assert np.isinf(vec[~mask]).all()
    det = aps_scores(s_base, mask, randomize=False, u=u)
    last = np.where(mask, det, -np.inf).max(1)
    assert (last == 1.0).all(), "deterministic APS top score is not exactly 1"
    print("vectorized APS/RAPS match the loop; deterministic APS top is exactly 1.0")


def test_quantile_has_no_jitter():
    """The threshold is an actual calibration score."""
    s = np.array([0.2, 0.5, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0])
    tau = C.calibrate_marginal(s, 0.3)
    k = int(np.ceil((len(s) + 1) * 0.7))
    assert tau == np.sort(s)[k - 1] and tau in s
    print(f"threshold is the k-th order statistic ({tau}), no jitter")


def test_axis_labels_partition_the_sample():
    """Every axis must label every item exactly once, with no empty level."""
    cache = C.load_cache(CACHE)
    sub = cache.subset(np.arange(min(500, cache.n)))
    for axis in C.AXES:
        lab = C.axis_labels(sub, axis)
        assert len(lab) == sub.n, f"{axis}: labelled {len(lab)} of {sub.n}"
        assert not any(v is None or v == "" for v in lab), f"{axis}: empty label"
        uniq, cnt = np.unique(lab, return_counts=True)
        print(f"  {axis:<14s} {len(uniq):>3d} levels, "
              f"n per level {cnt.min()}-{cnt.max()}")
    pooled = C.axis_labels(sub, "source", min_group=25)
    assert "other" in set(pooled.tolist()), \
        "min_group pooling produced no 'other' level"
    print(f"  source (min_group=25) {len(set(pooled.tolist())):>3d} levels")


def test_selected_configs_reproduce():
    """Re-split 0's tuning on D_thr picks the recorded configurations."""
    rec = pd.read_csv(P.CONF_PAPER / "materials" / "selected_configs.csv")
    rec = rec[(rec.corpus == "OmniMedVQA") & (rec.seed == SEED)].set_index("score")
    cache = C.load_cache(CACHE)
    idx = C.split_indices(cache, SEED)
    got = C.tune_posthoc(cache, idx["thr"], SEED)
    for m, cfg in got.items():
        for key, v in cfg.items():
            assert float(v) == float(rec.loc[m, key]), (m, key, v, rec.loc[m, key])
    print(f"selected configurations match selected_configs.csv: {got}")


def test_table_one_rows_reproduce():
    """Re-split 0 at alpha = 0.10 reproduces bench_alpha_mcq.csv exactly."""
    from cpaudit.bench import alpha_sweep
    df, _ = alpha_sweep.run(CACHE, n_seeds=1, seed0=SEED, alphas=(0.10,),
                            include=C.POSTHOC)
    ref = pd.read_csv(P.metric("bench_alpha_mcq.csv"))
    ref = ref[(ref.seed == SEED) & (ref.alpha == 0.10)]
    cols = ["coverage", "avg_size", "empty_rate", "top1"]
    for _, r in df.iterrows():
        want = ref[(ref.arm == r.arm) & (ref.method == r.method)].iloc[0]
        for c in cols:
            assert abs(r[c] - want[c]) < 1e-12, (r.arm, r.method, c, r[c], want[c])
    print(f"re-split {SEED} reproduces bench_alpha_mcq.csv ({len(df)} rows)")


if __name__ == "__main__":
    if not CACHE.exists():
        sys.exit(f"missing score cache: {CACHE} "
                 "(mkdir cache && cp paper/materials/scores/*.npz cache/)")
    test_keyed_uniforms()
    test_vectorized_scores_match_loop()
    test_quantile_has_no_jitter()
    print("axis labels:")
    test_axis_labels_partition_the_sample()
    test_selected_configs_reproduce()
    test_table_one_rows_reproduce()
    print("\nProtocol tests passed.")
