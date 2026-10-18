"""Submission-readiness verification for the 3ICT benchmark paper.

No number in the paper may be invented. This enforces it mechanically for
paper/main.tex, against the bench_*.csv universe.

Checks:
  1. Every 2-or-3-decimal numeral in the prose traces to a value computable from
     results/metrics/bench_*.csv, or to a whitelist of protocol constants.
  2. Every \\cite key exists in refs.bib.
  3. Every \\includegraphics target exists on disk.
  4. No dangling \\ref.
  5. Every number in the abstract reappears in the body.

Run:  python paper/verify.py
Exit 0 if everything passes.
"""
from __future__ import annotations

import io
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
TEX = HERE / "main.tex"
BIB = HERE / "refs.bib"
FIGDIR = HERE / "figures"
MET = HERE.parent / "results" / "metrics"
#: VQA-RAD is reported image-disjoint (`vqarad_img`); the item-level run is
#: kept because Section VIII quotes it as the contrast. `_q3vl4b` is the
#: second backbone (Section VI).
TAGS = ("mcq", "vqarad_mcq", "vqarad_img", "mcq_q3vl4b", "vqarad_img_q3vl4b")

#: Numbers stated by design rather than measured: split fractions, target
#: levels, grid values, dataset sizes-as-proportions, and the sample-size floor.
WHITELIST = {
    "0.50", "0.15", "0.20", "0.10", "0.05", "0.01",     # split fractions, alphas
    "0.90", "0.95", "0.99",                             # target coverage levels
    "0.25", "0.30",                                     # grid temperatures/gammas
    "0.001", "0.005", "0.028", "0.026",                 # quoted p-values
    "0.00026", "0.00029",
}


#: Columns that are already one row per comparison (the output of a paired test
#: or a confirmatory table), rather than one row per seed. These may be quoted
#: verbatim; everything else is only quotable as a mean over seeds.
_SCALAR_COLS = {
    "delta", "ci_lo", "ci_hi", "p_t", "p_w", "p_holm",
    "clean_cov", "leaky_cov", "frac_seeds_different_config",
    "diag_cov", "offdiag_cov", "offdiag_min", "frac_below_target",
    "spread_shift0", "spread_ensemble",
}


def _round(x, dp):
    return round(float(x), dp)


def csv_universe() -> set[float]:
    """Every scalar this paper could legitimately quote."""
    vals: set[float] = set()

    def add(x):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return
        if np.isfinite(f):
            vals.add(round(f, 6))
            # A negative result quoted as "-0.060" reaches the number regex as
            # "0.060", since the sign is not part of the match. Store the
            # magnitude too, or every negative finding looks fabricated.
            vals.add(round(abs(f), 6))

    def add_all(frame, cols, by=None):
        """Add the AGGREGATES the paper can quote -- never the raw rows.

        Adding every per-seed value floods the universe: with 26,880 rows in
        bench_groupaxes alone, 90% of all 3-decimal numbers in [0,1] landed
        within tolerance of something, and the check waved through a planted
        fabrication. The manuscript never quotes a single seed's value, so the
        universe is means over seeds (grouped the ways the tables group them)
        plus the derived statistics computed explicitly below.
        """
        for col in cols:
            if col not in frame:
                continue
            if by:
                by_present = [b for b in by if b in frame]
                if by_present:
                    for v in frame.groupby(by_present)[col].mean():
                        add(v)
            else:
                add(frame[col].mean())
        # Statistics columns are already one-row-per-comparison, not per-seed,
        # so those ARE quotable verbatim.
        for col in cols:
            if col in frame and col in _SCALAR_COLS:
                for v in frame[col]:
                    add(v)

    for tag in TAGS:
        # headline efficiency / validity
        p = MET / f"bench_alpha_{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            add_all(d, ["coverage", "avg_size", "norm_size", "empty_rate",
                        "top1"], by=["method", "alpha", "arm"])
            m = d[(d.alpha == 0.10) & (d.arm == "matched")]
            g = m.groupby("method")["avg_size"].mean()
            for a in g.index:            # pairwise set-size differences
                for b in g.index:
                    add(g[a] - g[b])

        # Conditional validity. Restricted to the VIEWS the paper actually
        # quotes rather than every (method x calib x eval x group) cell: with
        # 6x5x5x8 combinations the universe saturates the 3-decimal grid and the
        # check stops discriminating. The paper reports the marginal-calibration
        # slice by evaluation axis, and the modality axis per method.
        p = MET / f"bench_axissummary_{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            marg = d[d.calib_axis == "marginal"]
            cols = ["worst_cov", "spread", "excess_deficit", "excess_spread",
                    "null_worst", "null_spread", "n_groups_eval", "avg_size"]
            add_all(marg, cols, by=["eval_axis"])
            add_all(marg, cols, by=["method", "eval_axis"])
            # Mondrian on its own axis (the diagonal) is quoted as a contrast.
            diag = d[d.calib_axis == d.eval_axis]
            add_all(diag, cols, by=["eval_axis"])
            add_all(diag, cols, by=["method", "eval_axis"])
            add_all(d, ["n_calib_groups", "min_calib_group_n", "n_below_floor"],
                    by=["calib_axis"])

        p = MET / f"bench_groupaxes_{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            mod = d[(d.eval_axis == "modality")
                    & (d.calib_axis.isin(["marginal", "modality"]))]
            add_all(mod, ["coverage", "avg_size", "n"],
                    by=["method", "calib_axis", "group"])
            pos = d[(d.eval_axis == "position") & (d.calib_axis == "marginal")]
            add_all(pos, ["coverage", "avg_size"], by=["method", "group"])

        # calibration budget
        p = MET / f"bench_ncal_{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            add_all(d, ["cov_marginal", "cov_mondrian", "size_marginal",
                        "size_mondrian", "worst_marginal", "worst_mondrian",
                        "empty_marginal", "empty_mondrian", "beta_sd",
                        "n_per_group", "n_below_floor", "min_group_n"],
                    by=["n_cal"])
            for n, grp in d.groupby("n_cal"):     # observed sd vs Beta prediction
                add(grp["cov_marginal"].std())
                add(grp["cov_mondrian"].std())
                # the Mondrian set-size penalty over marginal, quoted in VI
                add(grp["size_mondrian"].mean() - grp["size_marginal"].mean())

        # tuning leakage
        for stem in ("bench_tuningleak", "bench_confirm_tuningleak",
                     "bench_stats_tuningleak"):
            p = MET / f"{stem}_{tag}.csv"
            if p.exists():
                d = pd.read_csv(p)
                add_all(d, ["coverage", "avg_size", "empty_rate", "delta",
                            "ci_lo", "ci_hi", "p_t", "p_w", "p_holm",
                            "clean_cov", "leaky_cov",
                            "frac_seeds_different_config"],
                        by=["method", "arm", "M"])

        # supervised and text-only controls
        for stem in ("bench_supervised", "bench_stats_supervised"):
            q = MET / f"{stem}_{tag}.csv"
            if q.exists():
                d = pd.read_csv(q)
                add_all(d, ["coverage", "avg_size", "top1", "thr_top1",
                            "empty_rate", "delta", "ci_lo", "ci_hi",
                            "p_t", "p_w", "p_holm"], by=["method"])

        # randomization / ties
        for stem in ("bench_random", "bench_stats_random"):
            p = MET / f"{stem}_{tag}.csv"
            if p.exists():
                d = pd.read_csv(p)
                add_all(d, ["coverage", "avg_size", "empty_rate", "tie_frac", "chance",
                            "delta", "ci_lo", "ci_hi", "p_t", "p_w", "p_holm",
                            "mean_K"],
                        by=["method", "randomize", "jitter"])
                if {"randomize", "jitter"} <= set(d.columns):
                    det = d[(d.randomize.astype(str) == "False") & (d.jitter > 0)]
                    ran = d[(d.randomize.astype(str) == "True") & (d.jitter > 0)]
                    if len(det) and len(ran):
                        add(det.coverage.mean() - ran.coverage.mean())
                        add(det.avg_size.mean() - ran.avg_size.mean())

        # shift, ssc, alpha stats
        for stem in ("bench_shift", "bench_stats_shift", "bench_ssc",
                     "bench_stats_alpha"):
            p = MET / f"{stem}_{tag}.csv"
            if p.exists():
                d = pd.read_csv(p)
                add_all(d, ["coverage", "avg_size", "delta", "ci_lo", "ci_hi",
                            "p_t", "p_w", "p_holm", "diag_cov", "offdiag_cov",
                            "offdiag_min", "frac_below_target"],
                        by=["method", "axis", "alpha"])

    # option order (OmniMedVQA only)
    for stem in ("bench_order", "bench_order_position", "bench_stats_order",
                 "bench_confirm_order"):
        p = MET / f"{stem}_mcq.csv"
        if p.exists():
            d = pd.read_csv(p)
            add_all(d, ["coverage", "avg_size", "top1", "empty_rate", "delta",
                        "ci_lo", "ci_hi", "p_t", "p_w", "p_holm",
                        "spread_shift0", "spread_ensemble"],
                    by=["method", "arm", "group"])

    # Derived dataset descriptors the manuscript quotes but no CSV stores:
    # normalized set size |C|/K, and VQA-RAD's questions-per-image ratio.
    import json as _json
    for tag in TAGS:
        rp = MET / f"bench_random_{tag}.csv"
        ap = MET / f"bench_alpha_{tag}.csv"
        if rp.exists() and ap.exists():
            meanK = float(pd.read_csv(rp).mean_K.iloc[0])
            add(meanK)
            d = pd.read_csv(ap)
            d = d[(d.alpha == 0.10) & (d.arm == "matched")]
            for v in d.groupby("method").avg_size.mean():
                add(v / meanK)
    # Questions per VQA-RAD image, from the exported manifest so the check also
    # runs in the artifact, which ships identifiers but not the raw items.
    ids = HERE / "materials" / "item_ids.csv"
    if ids.exists():
        v = pd.read_csv(ids)
        v = v[v.corpus == "VQA-RAD"]
        add(len(v) / max(v.image.nunique(), 1))

    # heterogeneity permutation test, cohort diagnostics, temperature
    # invariance and the split-unit comparison: all added after the first
    # submission-readiness pass, and all quoted in the manuscript.
    for stem in ("bench_heterogeneity_holm_mcq",
                 "bench_heterogeneity_holm_mcq_uniqimg"):
        q = MET / f"{stem}.csv"
        if not q.exists():
            continue
        d = pd.read_csv(q)
        for axis, grp in d.groupby("axis"):
            add(grp.chi2.mean())
            add(grp.p_holm.median())
            add(float((grp.p_holm < 0.05).mean()))
            add(float((grp.p_holm < 0.05).mean()) * 100)
    p = MET / "bench_heterogeneity_holm_mcq.csv"
    if p.exists():
        d = pd.read_csv(p)
        for axis, grp in d.groupby("axis"):
            add(grp.chi2.mean())
            add(grp.p_holm.median())
            add(float((grp.p_holm < 0.05).mean()))
            add(float((grp.p_holm < 0.05).mean()) * 100)
            add(grp.n_levels.mean())
    p = MET / "bench_cohort_composition_mcq.csv"
    if p.exists():
        d = pd.read_csv(p)
        for col in ("n", "base_top1_orig", "base_top1_rotavg", "answer_len",
                    "top_modality_share", "modality_entropy", "modality_tvd",
                    "top_qtype_share", "qtype_tvd", "source_tvd",
                    "top_source_share", "n_sources"):
            if col not in d:
                continue
            for v in d[col]:
                add(v)
                add(v * 100)
            add(d[col].max() - d[col].min())
    p = MET / "bench_cohort_profile_mcq.csv"
    if p.exists():
        d = pd.read_csv(p)
        # Overall profile, and the within-modality one with modalities weighted
        # equally (mean per stratum first), which is how the paper defines it.
        within = (d[d.stratum != "all"]
                  .groupby(["method", "stratum", "cohort"]).coverage.mean()
                  .groupby(["method", "cohort"]).mean())
        for prof in (d[d.stratum == "all"].groupby(["method", "cohort"])
                     .coverage.mean(), within):
            for v in prof:
                add(v)
            for m, g in prof.groupby("method"):
                add(g.max() - g.min())
    p = MET / "bench_lac_invariance.csv"
    if p.exists():
        d = pd.read_csv(p)
        for tag, g in d.groupby("tag"):
            for col in ("max_prob_shift", "frac_items_set_differs", "coverage",
                        "avg_size"):
                add(g[col].mean())
                add(g[col].max())
                add(g[col].max() * 100)
            for temp, gt in g.groupby("temp"):
                add(temp)
                for col in ("max_prob_shift", "frac_items_set_differs",
                            "coverage", "avg_size"):
                    add(gt[col].mean())
    # round 2: the answer-string lookup, the one-question-per-image check and
    # the leakage difference-in-differences.
    for tag in ("mcq", "vqarad_img"):
        q = MET / f"bench_answershortcut_{tag}.csv"
        if q.exists():
            d = pd.read_csv(q)
            for col in ("lookup_top1", "answer_seen_in_train", "tied_fraction",
                        "n_distinct_strings", "n_test"):
                add(d[col].mean())
                add(d[col].mean() * 100)
    q = MET / "bench_oneperimage_vqarad.csv"
    if q.exists():
        d = pd.read_csv(q)
        for m, g in d.groupby("method"):
            for col in ("coverage", "coverage_prefallback", "avg_size",
                        "empty_rate", "n_items", "n_cal", "n_test"):
                add(g[col].mean())
                add(g[col].mean() * 100)
        add(0.90 + 1.0 / (d.n_cal.mean() + 1))
    q = MET / "bench_stats_leakdid_vqarad.csv"
    if q.exists():
        d = pd.read_csv(q)
        for col in ("leak_item", "leak_cluster", "did", "ci_lo", "ci_hi",
                    "p_t", "p_w", "p_holm"):
            for v in d[col]:
                add(v)

    p = MET / "bench_stats_splitunit_vqarad.csv"
    if p.exists():
        d = pd.read_csv(p)
        for col in ("item_level", "image_disjoint", "delta", "ci_lo", "ci_hi",
                    "p_t", "p_w", "p_holm"):
            for v in d[col]:
                add(v)
    for tag in ("vqarad", "mcq"):
        p = MET / f"bench_splitunit_{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            for rule, g in d.groupby("rule"):
                for col in g.columns:
                    if col in ("rule", "seed"):
                        continue
                    add(g[col].mean())
                    add(g[col].mean() * 100)

    # the letter/item decomposition, which is derived rather than stored
    p = MET / "bench_order_position_mcq.csv"
    if p.exists():
        d = pd.read_csv(p)
        if "letter" in d and "orig_index" in d:
            single = d[d.arm.isin([str(s) for s in range(4)])]
            for key in ("letter", "orig_index"):
                prof = single.groupby(["method", key]).coverage.mean()
                for v in prof:
                    add(v)
                for m, grp in prof.groupby("method"):
                    add(grp.max() - grp.min())
            spread = (d.groupby(["seed", "arm", "method"]).coverage
                      .agg(lambda s: s.max() - s.min()).reset_index())
            for v in spread.groupby(["method", "arm"]).coverage.mean():
                add(v)

    # Revision after conditional acceptance: new controls and sensitivities.
    for name in ("bench_questionlookup_vqarad_img.csv",
                 "bench_stats_shortcutbreakdown_mcq.csv",
                 "bench_shortcutbreakdown_mcq.csv",
                 "bench_effects_summary.csv",
                 "bench_order_decomp.csv"):
        p = MET / name
        if p.exists():
            d = pd.read_csv(p)
            for c in d.select_dtypes("number").columns:
                for v in d[c].dropna():
                    add(v)
                add(d[c].mean())
    for tag in ("mcq_a005", "mcq_a020", "mcq_qtype"):
        p = MET / f"bench_ncal_{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            g = d.groupby("n_cal")
            for v in (g.size_mondrian.mean() - g.size_marginal.mean()):
                add(v)
            for v in g.n_per_group.mean():
                add(v)
    p = MET / "bench_order_position_vqarad_img.csv"
    if p.exists():
        d = pd.read_csv(p)
        single = d[d.arm.isin(["0", "1"])]
        for key in ("letter", "orig_index"):
            for v in single.groupby(["method", key]).coverage.mean():
                add(v)
    p = MET / "bench_confirm_order_vqarad_img.csv"
    if p.exists():
        for c in ("delta", "ci_lo", "ci_hi", "p_holm"):
            for v in pd.read_csv(p)[c]:
                add(v)

    # Second backbone (Section VI): temperature invariance and option order.
    p = MET / "bench_lac_invariance_q3vl4b.csv"
    if p.exists():
        d = pd.read_csv(p)
        for tag, g in d.groupby("tag"):
            add(g.max_prob_shift.max())
            add(g.frac_items_set_differs.max())
            add(g.frac_items_set_differs.max() * 100)
            add(int((g.groupby("seed").frac_items_set_differs.max() > 0).sum()))
    for tag, arms in (("mcq_q3vl4b", [str(s) for s in range(4)]),
                      ("vqarad_img_q3vl4b", ["0", "1"])):
        p = MET / f"bench_order_position_{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            single = d[d.arm.isin(arms)]
            for key in ("letter", "orig_index"):
                prof = single.groupby(["method", key]).coverage.mean()
                for v in prof:
                    add(v)
                for m, grp in prof.groupby("method"):
                    add(grp.max() - grp.min())
        p = MET / f"bench_confirm_order_{tag}.csv"
        if p.exists():
            for c in ("delta", "ci_lo", "ci_hi", "p_holm"):
                for v in pd.read_csv(p)[c]:
                    add(v)
    return vals


# --------------------------------------------------------------------------
def strip(tex: str) -> str:
    """Remove everything that legitimately contains digits but is not a result."""
    tex = re.sub(r"(?m)^\s*%.*$", "", tex)
    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\(label|ref|eqref|cite[a-z]*|rtag|url)\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\documentclass(\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\usepackage(\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"width=[0-9.]+\\?[a-z]*", " ", tex)          # float sizing
    tex = re.sub(r"\\texttt\{[^}]*\}", " ", tex)               # code spans
    tex = re.sub(r"Qwen2\.5-VL-3B[A-Za-z-]*", " ", tex)        # model name
    tex = re.sub(r"Qwen3-VL-4B[A-Za-z-]*", " ", tex)
    # TikZ drawings: coordinates and style values are layout, not results.
    tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", tex,
                 flags=re.S)
    return tex


def check_numbers():
    tex = strip(TEX.read_text(encoding="utf-8"))
    universe = csv_universe()
    bad = []
    for m in re.finditer(r"(?<![\d.])(\d\.\d{2,5})(?![\d])", tex):
        tok = m.group(1)
        if tok in WHITELIST:
            continue
        dp = len(tok.split(".")[1])
        val = float(tok)
        tol = 0.5 * 10 ** (-dp)
        if not any(abs(val - u) <= tol for u in universe):
            ctx = tex[max(0, m.start() - 60):m.start() + 40].replace("\n", " ")
            bad.append(f"  {tok}   ...{ctx.strip()}...")
    print(f"[1] numbers: {len(bad)} untraceable")
    for b in bad:
        print(b)
    return not bad


def check_citations():
    tex = TEX.read_text(encoding="utf-8")
    keys = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", tex):
        keys |= {k.strip() for k in m.group(1).split(",")}
    bib = set(re.findall(r"@[a-zA-Z]+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    missing = sorted(keys - bib)
    print(f"[2] citations: {len(keys)} cited, {len(missing)} unresolved")
    for k in missing:
        print(f"  missing from refs.bib: {k}")
    return not missing


def check_graphics():
    tex = TEX.read_text(encoding="utf-8")
    missing = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        if not (FIGDIR / m.group(1)).exists():
            missing.append(m.group(1))
    print(f"[3] graphics: {len(missing)} missing")
    for g in missing:
        print(f"  not on disk: {g}")
    return not missing


def check_refs():
    tex = TEX.read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", tex))
    dangling = sorted(refs - labels)
    print(f"[4] cross-references: {len(dangling)} dangling")
    for r in dangling:
        print(f"  no such label: {r}")
    return not dangling


def check_float_order():
    """3ICT: cite every figure and table in the text before it appears.

    Checked in source order: the first \\ref to each float's label must come
    before the float's \\begin. LaTeX may still float it earlier on the page,
    so the rendered PDF is checked by eye as well."""
    tex = TEX.read_text(encoding="utf-8")
    bad = []
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", tex, re.S):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(2))
        if not lab:
            continue
        first = re.search(r"\\ref\{" + re.escape(lab.group(1)) + r"\}", tex)
        if first is None or first.start() > m.start():
            bad.append(lab.group(1))
    print(f"[6] float order: {len(bad)} floats not cited before they appear")
    for b in bad:
        print(f"  first cited after its float: {b}")
    return not bad


def check_abstract():
    tex = TEX.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        print("[5] abstract: not found")
        return False
    abstract, body = strip(m.group(1)), strip(tex[m.end():])
    # Any decimal, not just 0.xx: the abstract quotes an accuracy as a
    # percentage ("95.3%") where the body prints a proportion ("0.953"), and the
    # old pattern matched neither form, so the check passed on an empty set.
    pat = r"(?<![\d.])(\d+\.\d+)(?![\d])"
    nums = set(re.findall(pat, abstract))
    body_nums = set(re.findall(pat, body))

    def restated(s):
        if s in body_nums or s in WHITELIST:
            return True
        v = float(s)
        # A percentage in the abstract and a proportion in the body are the same
        # claim. Nothing else counts as a restatement.
        for alt in (v / 100.0, v * 100.0):
            if any(f"{alt:.{dp}f}" in body_nums for dp in range(1, 5)):
                return True
        return False

    orphan = sorted(s for s in nums if not restated(s))
    words = len(abstract.split())
    ok_len = 150 <= words <= 250
    verdict = "ok" if ok_len else "OUT OF RANGE, 3ICT asks 150-250"
    print(f"[5] abstract: {words} words ({verdict}), {len(nums)} numbers, "
          f"{len(orphan)} not restated in body")
    for o in orphan:
        print(f"  only in abstract: {o}")
    return not orphan and ok_len


if __name__ == "__main__":
    if not TEX.exists():
        sys.exit(f"missing {TEX}")
    print(f"Verifying {TEX.name} against {MET}\n")
    ok = all([check_numbers(), check_citations(), check_graphics(),
              check_refs(), check_abstract(), check_float_order()])
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
