"""Pinned claims: each headline number recomputed from the CSVs and required to
appear, verbatim, in main.tex.

Why this exists. `verify.check_numbers` asks "does this numeral appear somewhere
in the space of quotable values?". With thousands of legitimate aggregates that
space saturates the three-decimal grid, so the check waves through fabrications:
it missed a planted 0.054 against a true 0.055. It is a net for gross errors,
not a proof.

This module is the proof. Every claim recomputes one number the paper asserts
and checks the formatted string is printed. Matching is on numeric tokens rather
than substrings, and only at the claim's own precision or higher -- accepting
fewer digits let "0.055" pass on the strength of "0.05", which the paper prints
as a significance level.

Known limit. This checks that each value is printed SOMEWHERE, not that every
printed instance is correct. A value appearing in both a table and the prose can
be corrupted in one place and still pass. Catching that needs position-aware
claims, which would be brittle against ordinary rewriting.

Run:  python paper/claims.py
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
MET = HERE.parent / "results" / "metrics"
TEX = HERE / "main.tex"

_cache: dict[str, pd.DataFrame] = {}


def csv(name: str) -> pd.DataFrame:
    if name not in _cache:
        _cache[name] = pd.read_csv(MET / name)
    return _cache[name]


#: VQA-RAD is evaluated image-disjoint, so every headline VQA-RAD number is
#: pinned against the `vqarad_img` run. The item-level `vqarad_mcq` run is used
#: only for the split-unit comparison in the discussion. Both are stratified on
#: the released organ (`cpaudit.data.vqarad_organ`), not the retired keyword guess.
VQA = "vqarad_img"
VQA_ITEM = "vqarad_mcq"


def alpha10(tag):
    d = csv(f"bench_alpha_{tag}.csv")
    return d[(d.alpha == 0.10) & (d.arm == "matched")]


def paired_split_unit(col):
    """Item-level vs image-disjoint on VQA-RAD, paired on seed."""
    from cpaudit.eval.stats import holm, paired
    a, b = alpha10(VQA_ITEM), alpha10(VQA)
    fam = []
    for m in ("base-LAC", "APS", "RAPS"):
        x = a[a.method == m].sort_values("seed")
        y = b[b.method == m].sort_values("seed")
        for c in ("coverage", "avg_size", "top1"):
            fam.append(paired(y[c].values, x[c].values, name=f"{m}/{c}"))
    holm(fam)
    return {r.name: r for r in fam}[col]


def claims():
    """(label, value, decimals, source) for every pinned number.

    Grouped in the order the six-section manuscript prints them.
    """
    import math

    out = []
    POSTHOC = ("base-LAC", "APS", "RAPS")

    # --- II. setup --------------------------------------------------------
    for tag, lbl in (("mcq", "Omni"), (VQA, "VQA-RAD")):
        out.append((f"mean K ({lbl})", float(csv(f"bench_random_{tag}.csv").mean_K.iloc[0]),
                    2, f"bench_random_{tag}"))
    # From the exported manifest, so the check runs without the raw VQA-RAD
    # release (the artifact ships identifiers, not data).
    ids = pd.read_csv(HERE / "materials" / "item_ids.csv")
    org = collections.Counter(ids[ids.corpus == "VQA-RAD"].stratum)
    vq = ids[ids.corpus == "VQA-RAD"]
    out.append(("VQA-RAD questions per image", len(vq) / vq.image.nunique(), 2,
                "materials/item_ids.csv"))
    out += [(f"VQA-RAD questions, {o}", org[o], 0, "data/vqarad organ match")
            for o in ("head", "chest", "abdomen")]
    om = pd.read_csv(HERE / "materials" / "vqarad_organ_map.csv")
    out.append(("VQA-RAD images matched uniquely by question set",
                int((om.matched_by == "question set, unique").sum()), 0,
                "materials/vqarad_organ_map.csv"))
    assert (om.matched_by == "inspection").sum() == 2
    assert (om.matched_by == "question set, same-organ candidates").sum() == 2
    su = csv("bench_splitunit_vqarad.csv")
    item, clus = su[su.rule == "item"], su[su.rule == "cluster"]
    out += [("test items sharing image pct, item rule",
             item.test_items_sharing_image.mean() * 100, 1, "bench_splitunit_vqarad"),
            ("test share, cluster rule", clus.test_share.mean(), 3,
             "bench_splitunit_vqarad")]
    so = csv("bench_splitunit_mcq.csv")
    out.append(("test items sharing image pct, Omni item rule",
                so[so.rule == "item"].test_items_sharing_image.mean() * 100, 1,
                "bench_splitunit_mcq"))
    # The shared-stream side effect, measured.
    st = pd.concat([csv(f"bench_stats_streamcheck_{t}.csv") for t in ("mcq", VQA)])
    out += [("stream check, largest coverage change",
             st[st.metric == "coverage"].delta.abs().max(), 4,
             "bench_stats_streamcheck_*"),
            ("stream check, smallest p (floor quoted)",
             math.floor(st[st.metric == "coverage"].p_t.min() * 100) / 100, 2,
             "bench_stats_streamcheck_*")]

    # --- III. the benchmark control ---------------------------------------
    sup = csv("bench_supervised_mcq.csv")
    for m, short in (("CE+LAC", "CE all-features"),
                     ("CE+LAC (answer text only)", "answer-text-only"),
                     ("base-LAC", "base score")):
        out.append((f"{short} top-1 (Omni)", sup[sup.method == m].top1.mean(), 3,
                    "bench_supervised_mcq"))
    ao = sup[sup.method == "CE+LAC (answer text only)"]
    out.append(("answer-text-only top-1, unseen answers (Omni)",
                (ao.top1_unseen * ao.n_unseen).sum() / ao.n_unseen.sum(), 3,
                "bench_supervised_mcq"))
    sv = csv(f"bench_supervised_{VQA}.csv")
    out.append(("answer-text-only top-1 (VQA-RAD)",
                sv[sv.method == "CE+LAC (answer text only)"].top1.mean(), 3,
                f"bench_supervised_{VQA}"))
    s = csv("bench_answershortcut_mcq.csv")
    out += [("lookup top-1 (Omni)", s.lookup_top1.mean(), 3, "bench_answershortcut_mcq"),
            ("answer seen in train (Omni)", s.answer_seen_in_train.mean(), 3,
             "bench_answershortcut_mcq"),
            ("distinct option strings (Omni)", s.n_distinct_strings.mean(), 0,
             "bench_answershortcut_mcq"),
            ("unseen test items per split (Omni)", s.n_unseen.mean(), 0,
             "bench_answershortcut_mcq"),
            # Pooled over splits, so each unseen item counts once.
            ("lookup top-1, unseen answers (Omni)",
             (s.lookup_top1_unseen * s.n_unseen).sum() / s.n_unseen.sum(), 3,
             "bench_answershortcut_mcq")]
    sv = csv(f"bench_answershortcut_{VQA}.csv")
    out.append(("lookup top-1 (VQA-RAD)", sv.lookup_top1.mean(), 3,
                f"bench_answershortcut_{VQA}"))
    out.append(("chance (Omni)", csv("bench_random_mcq.csv").chance.mean(), 3,
                "bench_random_mcq"))

    # --- Table I ------------------------------------------------------------
    empties = []
    for tag, lbl in (("mcq", "Omni"), (VQA, "VQA-RAD")):
        a = alpha10(tag)
        s = csv(f"bench_axissummary_{tag}.csv")
        s = s[(s.calib_axis == "marginal") & (s.eval_axis == "modality")]
        c = csv(f"bench_confirm_tuningleak_{tag}.csv")
        for m in POSTHOC:
            am = a[a.method == m]
            out += [(f"{m} coverage ({lbl})", am.coverage.mean(), 3, f"bench_alpha_{tag}"),
                    (f"{m} size ({lbl})", am.avg_size.mean(), 3, f"bench_alpha_{tag}"),
                    (f"{m} worst group ({lbl})", s[s.method == m].worst_cov.mean(), 3,
                     f"bench_axissummary_{tag}")]
            empties.append(am.empty_rate.mean())
            r = c[c.method == m].iloc[0]
            if r.delta != 0:
                out += [(f"{m} leak delta ({lbl})", r.delta, 4,
                         f"bench_confirm_tuningleak_{tag}"),
                        (f"{m} leak ci_lo ({lbl})", r.ci_lo, 4,
                         f"bench_confirm_tuningleak_{tag}"),
                        (f"{m} leak ci_hi ({lbl})", r.ci_hi, 4,
                         f"bench_confirm_tuningleak_{tag}")]
        out.append((f"shared top-1 ({lbl})", a[a.method == "base-LAC"].top1.mean(), 3,
                    f"bench_alpha_{tag}"))
    # Quoted as a bound ("at most"), so pinned rounded up.
    out.append(("max pre-fallback empty rate, bound",
                math.ceil(max(empties) * 1000) / 1000, 3, "bench_alpha_*"))

    # --- IV-A. calibration reuse ------------------------------------------
    cov = pd.concat([alpha10(t).groupby("method").coverage.mean() for t in ("mcq", VQA)])
    out += [("lowest mean coverage", cov.min(), 3, "bench_alpha_*"),
            ("highest mean coverage", cov.max(), 3, "bench_alpha_*")]
    gaps = [alpha10(t).groupby("method").avg_size.mean() for t in ("mcq", VQA)]
    covs = [alpha10(t).groupby("method").coverage.mean() for t in ("mcq", VQA)]
    out += [("largest set-size gap", max(g.max() - g.min() for g in gaps), 3,
             "bench_alpha_*"),
            ("largest within-corpus coverage gap",
             max(c.max() - c.min() for c in covs), 3, "bench_alpha_*")]
    for tag in ("mcq", VQA):
        t = csv(f"bench_tuningleak_{tag}.csv")
        g = t.groupby("arm")[["n_thr_role", "n_obj_role", "n_cal_final"]].mean()
        out += [(f"role size {arm}/{col} ({tag})", g.loc[arm, col], 0,
                 f"bench_tuningleak_{tag}")
                for arm in ("clean", "leaky-full") for col in g.columns]
        out.append((f"test size ({tag})", t.n_test.mean(), 0,
                    f"bench_tuningleak_{tag}"))
    for tag, lbl in (("mcq", "Omni"), (VQA, "VQA-RAD")):
        c = csv(f"bench_confirm_tuningleak_{tag}.csv")
        for _, r in c[c.delta != 0].iterrows():
            out.append((f"{r.method} leak p_holm ({lbl})", r.p_holm,
                        4 if r.p_holm < 0.01 else 3, f"bench_confirm_tuningleak_{tag}"))
            if r.p_holm < 0.05:
                out.append((f"{r.method} leak p_w ({lbl})", r.p_w,
                            4 if r.p_w < 0.01 else 3, f"bench_confirm_tuningleak_{tag}"))
        if tag == "mcq":
            for _, r in c.iterrows():
                out.append((f"{r.method} pct selection changed (Omni)",
                            r.frac_seeds_different_config * 100, 0,
                            "bench_confirm_tuningleak_mcq"))
    tl = csv("bench_tuningleak_mcq.csv")
    piv = tl.pivot_table(index=["method", "M"], columns="arm", values="coverage")
    loss = piv["leaky-full"] - piv["clean"]
    for M in (2, 4, 6):
        out.append((f"LAC leak loss at M={M}", abs(loss[("base-LAC", M)]), 4,
                    "bench_tuningleak_mcq"))
    # Size-matched arm: leaky vs clean at the full grid, paired on seed. The
    # text prints every interval that excludes zero and says the rest contain
    # it, so the rest are asserted.
    from cpaudit.eval.stats import ci
    excl = 0
    for tag, lbl in (("mcq", "Omni"), (VQA, "VQA-RAD")):
        tt = csv(f"bench_tuningleak_{tag}.csv")
        for m in POSTHOC:
            d = tt[(tt.method == m) & (tt.M == tt.grid_size)]
            p = d.pivot_table(index="seed", columns="arm", values="coverage")
            x = (p["leaky"] - p["clean"]).values
            lo, hi = ci(x)
            if lo > 0 or hi < 0:
                excl += 1
                out += [(f"size-matched {m} change ({lbl})", x.mean(), 4,
                         f"bench_tuningleak_{tag}"),
                        (f"size-matched {m} ci_lo ({lbl})", lo, 4,
                         f"bench_tuningleak_{tag}"),
                        (f"size-matched {m} ci_hi ({lbl})", hi, 4,
                         f"bench_tuningleak_{tag}")]
    out.append(("size-matched intervals excluding zero", excl, 0,
                "bench_tuningleak_*"))

    # --- IV-A. the binary LAC exception -----------------------------------
    li = csv("bench_lac_invariance.csv")
    vq = li[li.tag.str.startswith("VQA-RAD")]
    om = li[li.tag.str.startswith("OmniMedVQA")]
    out += [("invariance max prob shift (VQA-RAD)", vq.max_prob_shift.max(), 3,
             "bench_lac_invariance"),
            ("max set change pct vs T=1 (Omni)", om.frac_items_set_differs.max() * 100, 1,
             "bench_lac_invariance")]

    # --- IV-B. randomization ----------------------------------------------
    for tag, lbl in (("mcq", "Omni"), (VQA, "VQA-RAD")):
        r = csv(f"bench_random_{tag}.csv")
        # Primary protocol: no quantile jitter (jitter == 0).
        det = r[(r.randomize.astype(str) == "False") & (r.jitter == 0)
                & (r.method == "APS")]
        ran = r[(r.randomize.astype(str) == "True") & (r.jitter == 0)
                & (r.method == "APS")]
        out.append((f"tie fraction pct, no draw ({lbl})",
                    r[(r.randomize.astype(str) == "False")
                      & (r.jitter == 0)].tie_frac.mean() * 100,
                    1, f"bench_random_{tag}"))
        if lbl == "Omni":
            out.append(("deterministic APS size penalty (Omni)",
                        det.avg_size.mean() - ran.avg_size.mean(), 3,
                        "bench_random_mcq"))
        else:
            out += [("deterministic APS coverage (VQA-RAD)", det.coverage.mean(), 4,
                     f"bench_random_{tag}"),
                    ("deterministic APS size (VQA-RAD)", det.avg_size.mean(), 4,
                     f"bench_random_{tag}"),
                    ("atom at 1, pct of D_cal (VQA-RAD)", det.atom_one.mean() * 100, 1,
                     f"bench_random_{tag}")]
            # "tau equals 1 on every re-split" and "every test question receives
            # both answers" are printed as words; assert them instead of pinning.
            # tau == 1 exactly is what makes the inclusive rule admit the
            # lower-ranked candidate, whose score is exactly 1.
            assert (det.tau == 1.0).all(), "deterministic APS tau is not exactly 1"
            assert (det.all_options == 1.0).all(), "deterministic APS not all-label"

    # --- V-A. subgroups -----------------------------------------------------
    pg = csv("bench_groupaxes_mcq.csv")
    q = pg[(pg.eval_axis == "modality") & (pg.calib_axis == "marginal")
           & (pg.method == "base-LAC")]
    mon = pg[(pg.eval_axis == "modality") & (pg.calib_axis == "modality")
             & (pg.method == "base-LAC")]
    g = mon.groupby("group").coverage.mean()
    out += [("ultrasound, LAC marginal", q[q.group == "ultrasound"].coverage.mean(), 3,
             "bench_groupaxes_mcq"),
            ("fundus, LAC marginal", q[q.group == "Fundus Photography"].coverage.mean(),
             3, "bench_groupaxes_mcq"),
            ("LAC Mondrian modality min", g.min(), 3, "bench_groupaxes_mcq"),
            ("LAC Mondrian modality max", g.max(), 3, "bench_groupaxes_mcq"),
            ("ultrasound size, LAC marginal",
             q[q.group == "ultrasound"].avg_size.mean(), 3, "bench_groupaxes_mcq"),
            ("ultrasound size, LAC Mondrian",
             mon[mon.group == "ultrasound"].avg_size.mean(), 3, "bench_groupaxes_mcq")]
    s = csv("bench_axissummary_mcq.csv")
    s = s[s.calib_axis == "marginal"]
    for axis in ("modality", "question_type", "position", "K", "source"):
        d = s[s.eval_axis == axis]
        G = d.n_eval_groups.mean()
        out += [(f"G, {axis}", G, 0 if float(G).is_integer() else 1,
                 "bench_axissummary_mcq"),
                (f"worst, {axis}", d.worst_cov.mean(), 3, "bench_axissummary_mcq"),
                (f"excess deficit, {axis}", d.excess_deficit.mean(), 3,
                 "bench_axissummary_mcq"),
                (f"excess spread, {axis}", d.excess_spread.mean(), 3,
                 "bench_axissummary_mcq")]
    src = s[s.eval_axis == "source"]
    out += [("smallest source level, mean over splits", src.min_eval_group_n.mean(), 0,
             "bench_axissummary_mcq")]
    s = csv(f"bench_axissummary_{VQA}.csv")
    d = s[(s.calib_axis == "marginal") & (s.eval_axis == "modality")]
    out += [("G, VQA-RAD organ", d.n_eval_groups.mean(), 0, f"bench_axissummary_{VQA}"),
            ("worst, VQA-RAD organ", d.worst_cov.mean(), 3, f"bench_axissummary_{VQA}"),
            ("excess deficit, VQA-RAD organ", d.excess_deficit.mean(), 3,
             f"bench_axissummary_{VQA}"),
            ("excess spread, VQA-RAD organ", d.excess_spread.mean(), 3,
             f"bench_axissummary_{VQA}")]
    assert abs(d.excess_deficit.mean()) < 0.01 and abs(d.excess_spread.mean()) < 0.01, \
        "the text says both organ excesses are within 0.01 of zero"
    vg = csv(f"bench_groupaxes_{VQA}.csv")
    vg = vg[(vg.eval_axis == "modality") & (vg.calib_axis == "marginal")]
    lac = vg[vg.method == "base-LAC"].groupby("group").coverage.mean()
    sizes = vg[vg.method == "base-LAC"].groupby("group").n.mean()
    out += [("VQA-RAD organ, LAC abdomen", lac["abdomen"], 3, f"bench_groupaxes_{VQA}"),
            ("VQA-RAD organ, LAC chest", lac["chest"], 3, f"bench_groupaxes_{VQA}"),
            ("VQA-RAD organ test size min", sizes.min(), 0, f"bench_groupaxes_{VQA}"),
            ("VQA-RAD organ test size max", sizes.max(), 0, f"bench_groupaxes_{VQA}")]
    # The permutation test, now summarized in prose.
    h = csv("bench_heterogeneity_holm_mcq.csv")
    u = csv("bench_heterogeneity_holm_mcq_uniqimg.csv")
    frac = {a: float((h[h.axis == a].p_holm < 0.05).mean()) for a in h.axis.unique()}
    out += [("significant pct, modality", frac["modality"] * 100, 0,
             "bench_heterogeneity_holm_mcq"),
            ("significant pct, question type", frac["question_type"] * 100, 0,
             "bench_heterogeneity_holm_mcq"),
            ("significant pct, largest other axis",
             max(frac[a] for a in ("position", "K", "source")) * 100, 0,
             "bench_heterogeneity_holm_mcq")]
    runs = h.groupby("axis").size().iloc[0]
    shift = max(abs(int((h[h.axis == a].p_holm < 0.05).sum())
                    - int((u[u.axis == a].p_holm < 0.05).sum())) for a in frac)
    assert runs == 60, runs
    out.append(("largest unique-image shift, runs of sixty", shift, 0,
                "bench_heterogeneity_holm_mcq_uniqimg"))

    # --- V-B. Mondrian budget ---------------------------------------------
    nc = csv("bench_ncal_mcq.csv")
    lo = nc[nc.n_cal == 50]
    out += [("points per level at n=50", lo.n_per_group.mean(), 2, "bench_ncal_mcq"),
            ("levels below floor at n=50", lo.n_below_floor.mean(), 1, "bench_ncal_mcq"),
            ("Mondrian coverage at n=50", lo.cov_mondrian.mean(), 3, "bench_ncal_mcq"),
            ("Mondrian size at n=50", lo.size_mondrian.mean(), 2, "bench_ncal_mcq")]
    g = nc.groupby("n_cal").agg(per=("n_per_group", "mean"),
                                sm=("size_marginal", "mean"),
                                so=("size_mondrian", "mean"))
    for n_cal, r in g.iterrows():
        if n_cal == 300:        # in the sweep, not quoted
            continue
        out += [(f"points per level at n={n_cal}", r.per,
                 0 if float(r.per).is_integer() else 1, "bench_ncal_mcq"),
                (f"Mondrian size penalty at n={n_cal}", r.so - r.sm, 3,
                 "bench_ncal_mcq")]

    # --- V-C. cyclic presentations ----------------------------------------
    from cpaudit.bench.order_effect import decompose
    pos = csv("bench_order_position_mcq.csv")
    letter, item = decompose(pos)
    for frame, key, lbl in ((letter, "letter", "letter"),
                            (item, "orig_index", "item")):
        p = frame[frame.method == "base-LAC"].groupby(key).coverage.mean()
        out.append((f"LAC {lbl} profile range", p.max() - p.min(), 3,
                    "bench_order_position_mcq"))
    for arm, lbl in (("0", "original presentation"), ("ensemble", "ensemble")):
        p = pos[(pos.arm == arm) & (pos.method == "base-LAC")].groupby("group").coverage.mean()
        out.append((f"LAC {lbl} profile range", p.max() - p.min(), 3,
                    "bench_order_position_mcq"))
        if arm == "0":
            out += [("LAC single-presentation at A", p["pos-A"], 3,
                     "bench_order_position_mcq"),
                    ("LAC single-presentation at D", p["pos-D"], 3,
                     "bench_order_position_mcq")]
    out.append(("K=4 test items per split",
                pos[(pos.arm == "0") & (pos.method == "base-LAC")]
                .groupby("seed").n.sum().mean(), 0, "bench_order_position_mcq"))
    co = csv("bench_confirm_order_mcq.csv")
    for m in ("base-LAC", "APS", "RAPS"):
        r = co[co.method == m].iloc[0]
        out.append((f"{m} ensemble spread reduction", abs(r.delta), 3,
                    "bench_confirm_order_mcq"))
        # Its Holm p is printed in Table II and pinned with the table.
    cc = csv("bench_cohort_composition_mcq.csv")
    out += [("cohort modality TVD max", cc.modality_tvd.max(), 3,
             "bench_cohort_composition_mcq"),
            ("cohort qtype TVD max", cc.qtype_tvd.max(), 3,
             "bench_cohort_composition_mcq"),
            ("cohort source TVD max", cc.source_tvd.max(), 3,
             "bench_cohort_composition_mcq"),
            ("cohort top-1 rotation-averaged max", cc.base_top1_rotavg.max(), 3,
             "bench_cohort_composition_mcq"),
            ("cohort top-1 rotation-averaged min", cc.base_top1_rotavg.min(), 3,
             "bench_cohort_composition_mcq")]
    pr = csv("bench_cohort_profile_mcq.csv")
    allm = (pr[pr.stratum == "all"].groupby(["method", "cohort"])
              .coverage.mean().unstack())
    win = (pr[pr.stratum != "all"].groupby(["method", "stratum", "cohort"])
             .coverage.mean().groupby(["method", "cohort"]).mean().unstack())
    for frame, lbl in ((allm, "overall"), (win, "within-modality")):
        v = frame.loc["base-LAC"]
        out.append((f"LAC cohort spread, {lbl}", v.max() - v.min(), 4,
                    "bench_cohort_profile_mcq"))

    # --- VI. split unit and one question per image ------------------------
    ps = [paired_split_unit(f"{m}/{c}") for m in POSTHOC
          for c in ("coverage", "avg_size", "top1")]
    assert all(r.p_holm >= 0.05 for r in ps), "a split-unit comparison survives Holm"
    out.append(("split-unit smallest raw p (floor quoted)",
                math.floor(min(r.p_t for r in ps) * 100) / 100, 2,
                "bench_alpha_vqarad_* paired"))
    dd = csv("bench_stats_leakdid_vqarad.csv")
    for _, r in dd[dd.method != "base-LAC"].iterrows():
        out += [(f"leak DiD {r.method}", r.did, 4, "bench_stats_leakdid_vqarad"),
                (f"leak DiD ci_lo {r.method}", r.ci_lo, 4, "bench_stats_leakdid_vqarad"),
                (f"leak DiD ci_hi {r.method}", r.ci_hi, 4, "bench_stats_leakdid_vqarad")]
    op = csv("bench_oneperimage_vqarad.csv")
    for m in POSTHOC:
        g = op[op.method == m].sort_values("seed")
        out.append((f"one-per-image pre-fallback coverage {m}",
                    g.coverage_prefallback.mean(), 3, "bench_oneperimage_vqarad"))
        lo, hi = ci(g.coverage_prefallback.values)
        band_hi = 0.90 + 1.0 / (op.n_cal.mean() + 1)
        assert lo <= 0.90 and hi >= band_hi, f"{m} interval does not contain the band"
    out += [("one-per-image calibration points", op.n_cal.mean(), 0,
             "bench_oneperimage_vqarad"),
            ("one-per-image upper band", 0.90 + 1.0 / (op.n_cal.mean() + 1), 3,
             "bench_oneperimage_vqarad")]

    # --- revision after conditional acceptance (Oct 2026) -----------------
    from cpaudit.eval.stats import ci as _ci
    # Table I: coverage intervals.
    for tag, lbl in (("mcq", "Omni"), (VQA, "VQA-RAD")):
        a = alpha10(tag)
        for m in POSTHOC:
            lo, hi = _ci(a[a.method == m].sort_values("seed").coverage.values)
            out += [(f"{m} coverage CI lo ({lbl})", lo, 3, f"bench_alpha_{tag}"),
                    (f"{m} coverage CI hi ({lbl})", hi, 3, f"bench_alpha_{tag}")]
    # Table II: every printed protocol effect, from the same CSV the table is
    # generated from.
    from cpaudit.bench.effects_table import build as _effects
    for _, r in _effects().iterrows():
        key = f"{r.block}/{r.dataset}/{r.score}"
        if r.d_cov == 0 and r.cov_lo == 0 and r.cov_hi == 0:
            continue                      # printed as "0 exactly"
        for col, dp in (("d_cov", 4), ("cov_lo", 4), ("cov_hi", 4),
                        ("d_size", 3), ("size_lo", 3), ("size_hi", 3)):
            out.append((f"effects {key} {col}", r[col], dp, "bench_effects_summary"))
        if np.isfinite(r.p_holm) and r.p_holm >= 0.001:
            out.append((f"effects {key} p_holm", r.p_holm,
                        4 if r.p_holm < 0.01 else 3, "bench_effects_summary"))
    # Section III: shortcut uncertainty and breakdown.
    sb = csv("bench_stats_shortcutbreakdown_mcq.csv").iloc[0]
    out += [("answer-only minus full", sb.delta, 4, "bench_stats_shortcutbreakdown_mcq"),
            ("answer-only minus full ci_lo", sb.ci_lo, 4, "bench_stats_shortcutbreakdown_mcq"),
            ("answer-only minus full ci_hi", sb.ci_hi, 4, "bench_stats_shortcutbreakdown_mcq")]
    gb = csv("bench_shortcutbreakdown_mcq.csv")
    gm = gb[gb.axis == "modality"].set_index("group")
    out += [("answer-only top-1, lowest modality", gm["top1_answer-only"].min(), 3,
             "bench_shortcutbreakdown_mcq"),
            ("answer-only top-1, highest modality", gm["top1_answer-only"].max(), 3,
             "bench_shortcutbreakdown_mcq"),
            ("answer-only top-1, ultrasound", gm.loc["ultrasound", "top1_answer-only"], 3,
             "bench_shortcutbreakdown_mcq"),
            ("lookup top-1, ultrasound", gm.loc["ultrasound", "top1_lookup"], 3,
             "bench_shortcutbreakdown_mcq"),
            ("base top-1, ultrasound", gm.loc["ultrasound", "top1_base"], 3,
             "bench_shortcutbreakdown_mcq")]
    assert gm["top1_answer-only"].idxmin() == "X-Ray"
    assert gm["top1_answer-only"].idxmax().startswith("OCT")
    # Section III: VQA-RAD question-string lookup.
    q = csv(f"bench_questionlookup_{VQA}.csv").mean(numeric_only=True)
    out += [("question lookup top-1", q.lookup_top1, 3, f"bench_questionlookup_{VQA}"),
            ("questions seen in train pct", q.question_seen_in_train * 100, 1,
             f"bench_questionlookup_{VQA}"),
            ("question lookup top-1, seen", q.lookup_top1_seen, 3,
             f"bench_questionlookup_{VQA}"),
            ("base top-1, seen questions", q.base_top1_seen, 3,
             f"bench_questionlookup_{VQA}")]
    # Section V-A: shared-threshold reference.
    sh = pd.concat([csv(f"bench_axisshared_{t}.csv") for t in ("mcq", VQA)])
    g = sh.groupby("eval_axis")
    shift = max((g.excess_deficit_shared.mean() - g.excess_deficit.mean()).abs().max(),
                (g.excess_spread_shared.mean() - g.excess_spread.mean()).abs().max())
    out.append(("shared reference, largest excess change (bound)",
                math.ceil(shift * 1000) / 1000, 3, "bench_axisshared_*"))
    for tag in ("mcq", VQA):
        d = csv(f"bench_axisshared_{tag}.csv").groupby("eval_axis").mean(numeric_only=True)
        for a, b in (("excess_deficit", "excess_deficit_shared"),
                     ("excess_spread", "excess_spread_shared")):
            assert list(d[a].sort_values().index) == list(d[b].sort_values().index), \
                f"axis ordering changes under the shared reference ({tag}, {a})"
    # Table III: VQA-RAD answer (yes/no) row.
    s = csv(f"bench_axissummary_{VQA}.csv")
    d = s[(s.calib_axis == "marginal") & (s.eval_axis == "position")]
    out += [("worst, VQA-RAD answer", d.worst_cov.mean(), 3, f"bench_axissummary_{VQA}"),
            ("excess deficit, VQA-RAD answer", d.excess_deficit.mean(), 3,
             f"bench_axissummary_{VQA}"),
            ("excess spread, VQA-RAD answer", d.excess_spread.mean(), 3,
             f"bench_axissummary_{VQA}")]
    # Section V-B: the knee across alpha and partition.
    def pen(tag, n_cal):
        n = csv(f"bench_ncal_{tag}.csv")
        n = n[n.n_cal == n_cal]
        return float((n.size_mondrian - n.size_marginal).mean())
    out += [("floor at alpha 0.05", int(csv("bench_ncal_mcq_a005.csv").floor.iloc[0]), 0,
             "bench_ncal_mcq_a005"),
            ("floor at alpha 0.20", int(csv("bench_ncal_mcq_a020.csv").floor.iloc[0]), 0,
             "bench_ncal_mcq_a020"),
            ("penalty a005 at 12.5/level", pen("mcq_a005", 100), 3, "bench_ncal_mcq_a005"),
            ("penalty a005 at 25/level", pen("mcq_a005", 200), 3, "bench_ncal_mcq_a005"),
            ("penalty a020 at 12.5/level", pen("mcq_a020", 100), 3, "bench_ncal_mcq_a020"),
            ("penalty a020 at 25/level", pen("mcq_a020", 200), 3, "bench_ncal_mcq_a020"),
            ("penalty question type at 40/level", pen("mcq_qtype", 200), 3,
             "bench_ncal_mcq_qtype")]
    qt = csv("bench_ncal_mcq_qtype.csv")
    out.append(("question type points per level at n=200",
                qt[qt.n_cal == 200].n_per_group.mean(), 0, "bench_ncal_mcq_qtype"))
    # Section V-C: VQA-RAD, both orders.
    from cpaudit.bench.order_effect import decompose as _dec
    vp = csv(f"bench_order_position_{VQA}.csv")
    L, I = _dec(vp, k=2)
    li = L[L.method == "base-LAC"].groupby("letter").coverage.mean()
    ii = I[I.method == "base-LAC"].groupby("orig_index").coverage.mean()
    out += [("VQA-RAD LAC coverage, answer yes", ii.loc[0], 3, f"bench_order_position_{VQA}"),
            ("VQA-RAD LAC coverage, answer no", ii.loc[1], 3, f"bench_order_position_{VQA}"),
            ("VQA-RAD LAC coverage, letter A", li.loc[0], 3, f"bench_order_position_{VQA}"),
            ("VQA-RAD LAC coverage, letter B", li.loc[1], 3, f"bench_order_position_{VQA}")]
    vc = csv(f"bench_confirm_order_{VQA}.csv").set_index("method").loc["base-LAC"]
    out += [("VQA-RAD ensemble spread change", vc.delta, 3, f"bench_confirm_order_{VQA}"),
            ("VQA-RAD ensemble spread ci_lo", vc.ci_lo, 3, f"bench_confirm_order_{VQA}"),
            ("VQA-RAD ensemble spread ci_hi", vc.ci_hi, 3, f"bench_confirm_order_{VQA}"),
            ("VQA-RAD ensemble Holm p", vc.p_holm, 3, f"bench_confirm_order_{VQA}")]
    out += backbone2()
    return out


def backbone2():
    """Section VI and Table IV: the second backbone (Qwen3-VL-4B)."""
    POSTHOC = ("base-LAC", "APS", "RAPS")
    out = []
    B2 = {"Omni": "mcq_q3vl4b", "VQA-RAD": "vqarad_img_q3vl4b"}
    B1 = {"Omni": "mcq", "VQA-RAD": VQA}
    stars = {}
    nonzero = []
    for lbl, tag in B2.items():
        a = alpha10(tag)
        lac = a[a.method == "base-LAC"]
        s = csv(f"bench_axissummary_{tag}.csv")
        s = s[s.calib_axis == "marginal"]
        worst = s[(s.eval_axis == "modality") & (s.method == "base-LAC")].worst_cov
        out += [(f"B2 top-1 ({lbl})", lac.top1.mean(), 3, f"bench_alpha_{tag}"),
                (f"B2 LAC coverage ({lbl})", lac.coverage.mean(), 3, f"bench_alpha_{tag}"),
                (f"B2 LAC size ({lbl})", lac.avg_size.mean(), 3, f"bench_alpha_{tag}"),
                (f"B2 LAC worst group ({lbl})", worst.mean(), 3,
                 f"bench_axissummary_{tag}")]
        # Excess deficit: modality on OmniMedVQA, the yes/no answer (the
        # position axis at K = 2) on VQA-RAD, averaged over the three scores.
        axis = "modality" if lbl == "Omni" else "position"
        out.append((f"B2 excess deficit, {axis} ({lbl})",
                    s[s.eval_axis == axis].excess_deficit.mean(), 3,
                    f"bench_axissummary_{tag}"))
        r = csv(f"bench_random_{tag}.csv")
        det = r[(r.randomize.astype(str) == "False") & (r.jitter == 0) & (r.method == "APS")]
        ran = r[(r.randomize.astype(str) == "True") & (r.jitter == 0) & (r.method == "APS")]
        out.append((f"B2 deterministic APS size penalty ({lbl})",
                    det.avg_size.mean() - ran.avg_size.mean(), 3, f"bench_random_{tag}"))
        if lbl == "VQA-RAD":
            assert (det.tau == 1.0).all(), "B2: deterministic APS tau is not exactly 1"
            assert (det.all_options == 1.0).all(), "B2: deterministic APS not all-label"
            out.append(("B2 atom at 1, pct of D_cal (VQA-RAD)", det.atom_one.mean() * 100,
                        1, f"bench_random_{tag}"))
        for bb, t in (("B1", B1[lbl]), ("B2", tag)):
            c = csv(f"bench_confirm_tuningleak_{t}.csv").set_index("method")
            for m in POSTHOC:
                stars[(bb, lbl, m)] = bool(c.loc[m, "p_holm"] < 0.05)
                if c.loc[m, "delta"] != 0:
                    nonzero.append(c.loc[m, "delta"])
                elif bb == "B2":
                    # Printed as "0": binary LAC, coverage identical in both arms.
                    assert m == "base-LAC" and lbl == "VQA-RAD"
                    assert c.loc[m, "clean_cov"] == c.loc[m, "leaky_cov"]
                if bb == "B2" and c.loc[m, "delta"] != 0:
                    out.append((f"B2 {m} leak delta ({lbl})", c.loc[m, "delta"], 4,
                                f"bench_confirm_tuningleak_{t}"))
    # Table IV's stars, and "on Qwen3-VL-4B none on OmniMedVQA, and APS and
    # RAPS on VQA-RAD".
    assert [k for k, v in stars.items() if v] == [
        ("B1", "Omni", "base-LAC"), ("B1", "Omni", "RAPS"),
        ("B2", "VQA-RAD", "APS"), ("B2", "VQA-RAD", "RAPS")], stars
    assert len(nonzero) == 10 and all(d < 0 for d in nonzero), \
        "the text says all ten non-zero reuse contrasts lower coverage"
    out.append(("largest reuse loss, percentage points", -min(nonzero) * 100, 1,
                "bench_confirm_tuningleak_*"))
    # Worst and recovered modality on OmniMedVQA.
    pg = csv("bench_groupaxes_mcq_q3vl4b.csv")
    q = pg[(pg.eval_axis == "modality") & (pg.calib_axis == "marginal")
           & (pg.method == "base-LAC")].groupby("group").coverage.mean()
    assert q.idxmin().startswith("OCT"), q
    out += [("B2 OCT, LAC marginal", q.min(), 3, "bench_groupaxes_mcq_q3vl4b"),
            ("B2 ultrasound, LAC marginal", q["ultrasound"], 3, "bench_groupaxes_mcq_q3vl4b")]
    # Binary LAC across temperatures (reference T = 1).
    li = csv("bench_lac_invariance_q3vl4b.csv")
    v = li[li.tag.str.startswith("VQA-RAD")]
    changed = v[v.frac_items_set_differs > 0]
    assert set(changed.temp) == {0.25}, "the text says only T = 0.25 changes sets"
    assert v.sets_contain_T1.all(), "the text says rounding only added options"
    assert (v.groupby("seed").coverage.nunique() == 1).all(), \
        "the text says coverage never changed"
    out += [("B2 re-splits with a set change", changed.seed.nunique(), 0,
             "bench_lac_invariance_q3vl4b"),
            ("B2 max pct of test items changed", v.frac_items_set_differs.max() * 100, 1,
             "bench_lac_invariance_q3vl4b")]
    # Option order (bench_order_decomp, LAC).
    od = csv("bench_order_decomp.csv")
    od = od[od.method == "base-LAC"].set_index(["backbone", "corpus", "factor"])
    for bb in ("Qwen2.5-VL-3B", "Qwen3-VL-4B"):
        let = od.loc[(bb, "VQA-RAD", "letter (B - A)")]
        ans = od.loc[(bb, "VQA-RAD", "answer (no - yes)")]
        out += [(f"{bb} VQA-RAD letter range", abs(let.delta), 3, "bench_order_decomp"),
                (f"{bb} VQA-RAD no minus yes", ans.delta, 3, "bench_order_decomp")]
        if bb == "Qwen3-VL-4B":
            out += [("B2 VQA-RAD no minus yes ci_lo", ans.ci_lo, 3, "bench_order_decomp"),
                    ("B2 VQA-RAD no minus yes ci_hi", ans.ci_hi, 3, "bench_order_decomp"),
                    ("B2 VQA-RAD coverage, answer yes", ans.cov_0, 3, "bench_order_decomp"),
                    ("B2 VQA-RAD coverage, answer no", ans.cov_1, 3, "bench_order_decomp"),
                    ("B2 VQA-RAD coverage, letter A", let.cov_0, 3, "bench_order_decomp"),
                    ("B2 VQA-RAD coverage, letter B", let.cov_1, 3, "bench_order_decomp")]
        key = (bb, "OmniMedVQA", "letter profile range")
        if key in od.index:
            out.append((f"{bb} OmniMedVQA letter range", od.loc[key].delta, 3,
                        "bench_order_decomp"))
    vc = csv("bench_confirm_order_vqarad_img_q3vl4b.csv").set_index("method").loc["base-LAC"]
    out += [("B2 VQA-RAD ensemble spread change", vc.delta, 3,
             "bench_confirm_order_vqarad_img_q3vl4b"),
            ("B2 VQA-RAD ensemble Holm p", vc.p_holm, 3,
             "bench_confirm_order_vqarad_img_q3vl4b")]
    # OmniMedVQA cyclic orders on the second backbone.
    out.append(("B2 OmniMedVQA original presentation range",
                od.loc[("Qwen3-VL-4B", "OmniMedVQA", "original presentation range")].delta,
                3, "bench_order_decomp"))
    mc = csv("bench_confirm_order_mcq_q3vl4b.csv").set_index("method")
    assert (mc.p_holm >= 0.05).all(), "the text says the ensemble no longer narrows it"
    out += [("B2 OmniMedVQA ensemble spread change", mc.loc["base-LAC", "delta"], 3,
             "bench_confirm_order_mcq_q3vl4b"),
            ("B2 OmniMedVQA ensemble Holm p", mc.loc["base-LAC", "p_holm"], 3,
             "bench_confirm_order_mcq_q3vl4b")]
    return out


def tex_number_tokens(tex: str) -> set[str]:
    """Every numeric literal in the manuscript, as a whole token, signed and not.

    Substring matching is not safe: a percentage variant like "5" occurs inside
    almost every other number on the page, so every claim would pass vacuously.

    Digit grouping is stripped first. IEEE style writes four-digit numbers as
    `1{,}244`, which the tokenizer would otherwise read as "1" and "244" --- so
    no pin on such a value could ever match, and the check would silently report
    it missing forever.
    """
    tex = re.sub(r"(?<=\d)\{,\}(?=\d)", "", tex)
    toks = set(re.findall(r"(-?)(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", tex))
    return {s + n for s, n in toks} | {n for _, n in toks}


def main():
    tex = TEX.read_text(encoding="utf-8")
    # TikZ coordinates are layout; left in, a value like 0.62 could satisfy a
    # pin by coincidence.
    tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", tex,
                 flags=re.S)
    # Reviewer tags (\rtag{R1.2}) and URLs are labels, not numbers.
    tex = re.sub(r"\\(rtag|url)\{[^}]*\}", " ", tex)
    tokens = tex_number_tokens(tex)
    counts = collections.Counter(
        m.group(0) for m in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", tex))
    bad, ok, multi = [], 0, []
    for label, value, dp, src in claims():
        if value != value:              # NaN: nothing to pin
            continue
        s = f"{value:.{dp}f}"
        # Both signs: a negative table entry is written "$-$0.011", where the
        # minus is LaTeX markup and never reaches the number token.
        variants = {f"{v:.{d}f}" for d in range(dp, dp + 3)
                    for v in (value, abs(value))}
        # A proportion printed as a percentage is the same claim. At least one
        # decimal is kept so that, say, 0.0799 cannot be satisfied by a bare "8"
        # sitting in an unrelated sentence.
        # Not for tiny values (p-values): 0.00004 as a percentage is "0.0",
        # which any stray zero in the text would satisfy.
        if 0.001 <= abs(value) < 1:
            variants |= {f"{v * 100:.{d}f}" for d in range(1, 3)
                         for v in (value, abs(value))}
        hit = variants & tokens
        if hit:
            ok += 1
            # A value printed more than once is only partly protected:
            # corrupting one instance leaves the claim satisfied by the others.
            # Counting them is more useful than asserting the limitation and
            # moving on.
            n = max(counts.get(h, 0) for h in hit)
            if n > 1:
                multi.append((label, s, n))
        else:
            bad.append(f"  {label:<40s} computed {s:>10s}   "
                       f"none of {sorted(variants)} printed   [{src}]")
    print(f"pinned claims: {ok} present, {len(bad)} missing or drifted")
    for b in bad:
        print(b)
    if multi:
        print()
        print(f"{len(multi)} of {ok} are printed more than once, so "
              "corrupting a single instance would still pass.")
        print("Most exposed:")
        for label, s, n in sorted(multi, key=lambda r: -r[2])[:6]:
            print(f"  {label:<40s} {s:>9s}  appears {n}x")
    return not bad


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
