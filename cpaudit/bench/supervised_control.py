"""The learned-score controls of Section III: how much does the benchmark
reward reading the answer strings?

**The full-feature head.** An MLP on the cached features,
s = s_base + g([z_x; e_y]), trained by cross-entropy and conformalized with
plain LAC. Early stopping uses D_thr top-1.

**The answer-only head.** The same head with z_x and s_base zeroed, leaving
g([0; e_y]): it sees nothing but each option string's frozen input embedding.
Whatever top-1 it reaches is the share of the benchmark answerable from the
option text alone; on OmniMedVQA that share is large, so every learned-head
result there is partly a measurement of the shortcut. Both heads are trained
separately on every re-split with the same architecture, optimizer, batch size,
seed and stopping rule.

Run: python -m cpaudit.bench.supervised_control --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.baselines.posthoc import lac_scores
from cpaudit.bench.common import (ALPHA, PostHoc, TEMPS, split_indices, thr_halves)
from cpaudit.calib.conformal import predict_sets
from cpaudit.bench.common import calibrate_marginal
from cpaudit.bench.common import load_cache
from cpaudit.eval.metrics import avg_set_size, coverage

#: Long enough to converge (the diagnostic showed D_thr top-1 plateauing by
#: ~epoch 20 at lr=1e-3), with patience so a plateau stops it early.
EPOCHS, PATIENCE, BATCH, LR = 60, 10, 64, 1e-3


def _top1(s, sub):
    return float((np.where(sub.mask, s, np.inf).argmin(1) == sub.answer_idx).mean())


def fit_ce(cache, train_idx, thr_idx, seed=0, image_blind=False, device=None,
           zero_base=False):
    """Cross-entropy head, early-stopped on D_thr top-1. D_cal is never read.

    `image_blind` zeroes z_x only, which does NOT give a text-only predictor:
    ScoringHead computes s = s_base + g(concat(z_x, e_y)), and s_base is the
    frozen VLM's own score, already encoding image and question. `zero_base`
    additionally zeroes s_base, leaving g(e_y) alone -- a predictor that has
    seen nothing but the answer strings. Only that one bounds the answer-text
    shortcut.
    """
    import torch

    from cpaudit.model import ScoringHead
    from cpaudit.model.head import _pack

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, thr = cache.subset(train_idx), cache.subset(thr_idx)

    def pack(sub):
        d = _pack(sub, dev, image_blind=image_blind)
        if zero_base:
            d["s_base"] = torch.zeros_like(d["s_base"])
        return d

    tr_t, thr_t = pack(tr), pack(thr)

    head = ScoringHead(cache.z_x.shape[1], cache.e_y.shape[2],
                       hidden=(256, 128), device=dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)

    def thr_top1():
        with torch.no_grad():
            s = head.scores(thr_t["s_base"], thr_t["z_x"], thr_t["e_y"],
                            thr_t["mask"]).cpu().numpy()
        return _top1(s, thr)

    best, best_state, since = -np.inf, None, 0
    for epoch in range(EPOCHS):
        perm = np.random.default_rng(seed + epoch).permutation(tr.n)
        for b0 in range(0, tr.n, BATCH):
            bi = perm[b0:b0 + BATCH]
            if len(bi) < 4:
                continue
            s = head.scores(tr_t["s_base"][bi], tr_t["z_x"][bi],
                            tr_t["e_y"][bi], tr_t["mask"][bi])
            # Low nonconformity = plausible, so logits are -s; pads are pushed
            # far below any real option. -1e4 is already total underflow after
            # exp() in float32 and, unlike a near-limit constant such as -1e30,
            # cannot turn cross_entropy's logsumexp into inf/NaN.
            logits = (-s).masked_fill(~tr_t["mask"][bi], -1e4)
            loss = torch.nn.functional.cross_entropy(logits, tr_t["true"][bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = thr_top1()
        if acc > best + 1e-4:
            best, since = acc, 0
            best_state = {k: v.detach().clone()
                          for k, v in head.g.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state is not None:
        head.g.load_state_dict(best_state)

    def score(sub):
        t = pack(sub)
        with torch.no_grad():
            s = head.scores(t["s_base"], t["z_x"], t["e_y"],
                            t["mask"]).cpu().numpy().astype(np.float64)
        # Batched float32 arithmetic makes a row's score depend on the batch it
        # was computed in, by ~1e-8: the answer-only head on VQA-RAD, whose
        # scores are mathematically one value per option, came out with two
        # values in D_cal and one in D_test. With the threshold taken as an
        # unperturbed order statistic that splits an exact tie and moves
        # coverage. Rounding to 1e-5 restores the ties; it is far below any
        # difference the head means.
        return np.where(np.isfinite(s), np.round(s, 5), s)
    return score, best


def _tune_temp(score_fn, sub_a, sub_b, seed, alpha):
    """Pick LAC's temperature on D_thr halves, the same protocol the post-hoc
    baselines get. The head's raw scores are wildly scaled after cross-entropy
    training, so this step is not optional."""
    from cpaudit.bench.common import sel_size
    best, best_T = np.inf, 1.0
    for T in TEMPS + [10.0, 100.0, 1000.0]:
        Sa = lac_scores(score_fn(sub_a), sub_a.mask, temp=T)
        Sb = lac_scores(score_fn(sub_b), sub_b.mask, temp=T)
        v = sel_size(Sa, sub_a, Sb, sub_b, seed, alpha)
        if v < best:
            best, best_T = v, T
    return best_T


def run(cache_path, n_seeds=20, seed0=0, alpha=ALPHA, device=None, cluster=False):
    cache = load_cache(cache_path)
    rows = []
    t0 = time.time()
    for seed in range(seed0, seed0 + n_seeds):
        idx = split_indices(cache, seed, cluster=cluster)
        cal, test = cache.subset(idx["cal"]), cache.subset(idx["test"])
        sub_a, sub_b = thr_halves(cache, idx["thr"], seed)

        # reference: the frozen base score, tuned exactly as in the main study
        from cpaudit.bench.common import posthoc_grid, select_config
        cfg, _ = select_config("base-LAC", posthoc_grid("base-LAC"),
                               sub_a, sub_b, seed, alpha)
        arms = {"base-LAC": (PostHoc(lac_scores, cfg, seed).score, np.nan)}

        for name, blind, zb in (("CE+LAC", False, False),
                                ("CE+LAC (no image feats)", True, False),
                                ("CE+LAC (answer text only)", True, True)):
            score_fn, thr_acc = fit_ce(cache, idx["train"], idx["thr"],
                                       seed=seed, image_blind=blind,
                                       device=device, zero_base=zb)
            T = _tune_temp(score_fn, sub_a, sub_b, seed, alpha)
            arms[name] = (lambda sub, f=score_fn, T=T:
                          lac_scores(f(sub), sub.mask, temp=T), thr_acc)

        # Was each test item's correct answer string offered anywhere in
        # training? Splits the answer-only head's accuracy into what a string
        # lookup could reach and what it could not.
        from cpaudit.bench.answer_shortcut import _options
        from cpaudit.bench.common import load_items
        opts = _options(cache, load_items())
        train_strings = {s for i in idx["train"] for s in opts[i]}
        seen = np.array([opts[i][cache.answer_idx[i]] in train_strings
                         for i in idx["test"]])

        for name, (fn, thr_acc) in arms.items():
            S_c, S_t = fn(cal), fn(test)
            hit = (np.where(test.mask, S_t, np.inf).argmin(1) == test.answer_idx)
            tau = calibrate_marginal(S_c[np.arange(cal.n), cal.answer_idx],
                                     alpha, seed=seed)
            sets, empty = predict_sets(S_t, test.mask, tau)
            rows.append(dict(seed=seed, method=name,
                             coverage=coverage(sets, test.answer_idx),
                             avg_size=avg_set_size(sets),
                             empty_rate=float(empty.mean()),
                             top1=_top1(S_t, test), thr_top1=thr_acc,
                             top1_seen=float(hit[seen].mean()) if seen.any()
                             else float("nan"),
                             top1_unseen=float(hit[~seen].mean()) if (~seen).any()
                             else float("nan"),
                             n_unseen=int((~seen).sum())))
        print(f"  [supervised_control] seed {seed - seed0 + 1}/{n_seeds}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows)


def stats(df):
    from cpaudit.eval.stats import holm, paired
    fam, keys = [], []
    for m in ("CE+LAC", "CE+LAC (no image feats)",
              "CE+LAC (answer text only)"):
        for metric in ("avg_size", "top1"):
            a = df[df.method == m].sort_values("seed")[metric].values
            b = df[df.method == "base-LAC"].sort_values("seed")[metric].values
            if len(a) == len(b) and len(a) > 1:
                fam.append(paired(a, b, f"{m} - base-LAC ({metric})"))
                keys.append((m, metric))
    holm(fam)
    return pd.DataFrame([
        dict(method=m, metric=metric, delta=r.delta, ci_lo=r.ci_lo,
             ci_hi=r.ci_hi, p_t=r.p_t, p_w=r.p_w, p_holm=r.p_holm, n=r.n)
        for r, (m, metric) in zip(fam, keys)])


def report(df, st, tag, alpha=ALPHA):
    print(f"\n=== Supervised and text-only controls on {tag} "
          f"(alpha={alpha}, {df.seed.nunique()} seeds) ===")
    g = df.groupby("method").agg(cov=("coverage", "mean"),
                                 size=("avg_size", "mean"),
                                 top1=("top1", "mean"),
                                 thr_top1=("thr_top1", "mean"))
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nPaired against base-LAC, Holm-corrected:")
    print(st[["method", "metric", "delta", "ci_lo", "ci_hi", "p_t", "p_w",
              "p_holm"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    key = "CE+LAC (answer text only)"
    if key in g.index:
        print()
        print(f"A predictor given nothing but the answer strings reaches "
              f"top-1 {g.loc[key, 'top1']:.3f}.")
        print("That is the answer-text shortcut: it bounds how much of any "
              "learned score's")
        print("efficiency on this corpus can be credited to the image at all.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--device", default=None,
                    help="cpu or cuda; the head is ~1M params so cpu is fine")
    ap.add_argument("--cluster", action="store_true",
                    help="image-disjoint split: every question about one "
                         "image stays in one partition")
    args = ap.parse_args()
    df = run(args.cache, n_seeds=args.seeds, alpha=args.alpha,
             device=args.device, cluster=args.cluster)
    st = stats(df)
    df.to_csv(P.metric(f"bench_supervised_{args.tag}.csv"), index=False)
    st.to_csv(P.metric(f"bench_stats_supervised_{args.tag}.csv"), index=False)
    report(df, st, args.tag, args.alpha)
