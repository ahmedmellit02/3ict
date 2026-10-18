"""Figures for the paper.

Post-hoc scores on the frozen models' cached scores. Figures go to
paper/figures/ via `paths.conf_figure`.

Every number traces to results/metrics/bench_*.csv.

Regenerate all:  python -m cpaudit.figures.bench_figs
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpaudit import paths as P

MET = P.METRICS

#: Post-hoc scores only. Learned scores are not ranked in this paper: on
#: OmniMedVQA a predictor given nothing but the answer strings reaches top-1
#: 0.953, so any learned head's efficiency there is measuring a benchmark
#: shortcut rather than a method. See `bench_supervised_*.csv`.
ORDER = ["base-LAC", "APS", "RAPS"]
COL = {"base-LAC": "#4c72b0", "APS": "#6f9bd1", "RAPS": "#93b8e0"}
#: Controls, drawn apart from the scores they are controls for.
CTRL = {"base-LAC": "#4c72b0", "CE+LAC": "#55a868",
        "CE+LAC (no image feats)": "#8fc7a5", "CE+LAC (answer text only)": "#c44e52"}
TARGET = 0.90
#: IEEEtran a4paper two-column: \columnwidth is 252 pt = 3.49 in, and the
#: manuscript prints these at 0.95 of that. Figures are authored at exactly that
#: width so the font sizes below are the sizes that reach the page -- drawing
#: wide and scaling down is what made the old figures unreadable.
COL_IN = 3.31
#: Point sizes on paper. IEEE asks for text in figures to be no smaller than the
#: caption (8 pt); 7 pt is the floor we allow, for tick labels and legends.
FS_TICK, FS_LABEL, FS_LEGEND = 7.0, 8.0, 7.0
plt.rcParams.update({
    "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND, "axes.titlesize": FS_LABEL,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "figure.dpi": 400, "savefig.dpi": 400,
    "lines.linewidth": 0.9, "lines.markersize": 2.6,
    "legend.handletextpad": 0.4, "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.25,
})
#: VQA-RAD is evaluated image-disjoint (see data.split.grouped_four_way_split);
#: the item-level tag `vqarad_mcq` is kept on disk only for the split-unit
#: comparison in the paper's Section VIII.
DATASETS = {"mcq": "OmniMedVQA (K mostly 4)",
            "vqarad_img": "VQA-RAD (K=2, image-disjoint)"}
#: Modality names in the cache are verbose and one is misspelled upstream.
_SHORT = {"CT(Computed Tomography)": "CT", "Dermoscopy": "Derm",
          "Fundus Photography": "Fundus", "MR (Mag-netic Resonance Imaging)": "MRI",
          "Microscopy Images": "Micro", "OCT (Optical Coherence Tomography": "OCT",
          "X-Ray": "X-Ray", "ultrasound": "US"}


def _ci(x):
    """Student-t half-width, never the normal quantile (see cpaudit.eval.stats)."""
    from cpaudit.eval.stats import half_width
    return half_width(np.asarray(x, float))


def _order(present):
    return [m for m in ORDER if m in set(present)]


def _save(fig, name):
    path = P.conf_figure(name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------
def fig_shortcut(alpha=0.10):
    """B1 - why this paper does not rank learned scores.

    Three arms per corpus: the frozen base score, a cross-entropy head on the
    same features, and a predictor handed nothing but the answer strings. On
    OmniMedVQA the last one matches the first two, which means the corpus is
    answerable without the image; on VQA-RAD it collapses to chance, which is
    what a corpus without an answer-text prior looks like.
    """
    order = ["base-LAC", "CE+LAC", "CE+LAC (no image feats)",
             "CE+LAC (answer text only)"]
    short = {"base-LAC": "frozen base", "CE+LAC": "CE head (all)",
             "CE+LAC (no image feats)": "CE (no img feats)",
             "CE+LAC (answer text only)": "answer text only"}
    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 1.55))
    for ax, (tag, title) in zip(axes, DATASETS.items()):
        d = pd.read_csv(MET / f"bench_supervised_{tag}.csv")
        present = [m for m in order if m in set(d.method)]
        x = np.arange(len(present))
        g = d.groupby("method")["top1"].agg(["mean", _ci]).reindex(present)
        ax.bar(x, g["mean"], yerr=g["_ci"], capsize=3,
               color=[CTRL[m] for m in present], edgecolor="black", linewidth=.5)
        r = pd.read_csv(MET / f"bench_random_{tag}.csv")
        # E[1/K] over the evaluated items. Reciprocal-of-the-mean would be
        # 1/E[K], which is smaller and is not what uniform guessing achieves.
        chance = float(r.chance.mean()) if "chance" in r else 1.0 / r.mean_K.iloc[0]
        ax.axhline(chance, color="black", ls=":", lw=1,
                   label=f"chance ({chance:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([short[m] for m in present], rotation=32,
                           ha="right", fontsize=FS_TICK)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=FS_LABEL)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("top-1 accuracy")
    fig.tight_layout(pad=0.2, w_pad=0.6)
    return _save(fig, "B1_shortcut.png")


def fig_per_group(tag="mcq", alpha=0.10):
    """B2 - per-modality coverage, marginal vs Mondrian, every method."""
    d = pd.read_csv(MET / f"bench_groupaxes_{tag}.csv")
    d = d[d.eval_axis == "modality"]
    marg = d[d.calib_axis == "marginal"]
    mond = d[d.calib_axis == "modality"]
    groups = sorted(marg.group.unique())
    order = _order(d.method)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, frame, title in [
            (axes[0], marg, "marginal calibration"),
            (axes[1], mond, "Mondrian, calibrated per modality")]:
        piv = frame.pivot_table(index="group", columns="method",
                                values="coverage", aggfunc="mean").reindex(
            index=groups, columns=order)
        x = np.arange(len(groups))
        w = 0.8 / len(order)
        for k, m in enumerate(order):
            ax.bar(x + k * w - 0.4 + w / 2, piv[m], w, label=m, color=COL[m],
                   edgecolor="black", linewidth=.3)
        ax.axhline(TARGET, color="black", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([_SHORT.get(g, g) for g in groups], rotation=30,
                           ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.6, 1.02)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("coverage", fontsize=9)
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Marginal calibration abandons subgroups; Mondrian repairs it "
                 "unevenly", fontsize=11)
    fig.tight_layout()
    return _save(fig, "B2_per_group.png")


def fig_axes(tag="mcq"):
    """B3 - which grouping axis actually fails, measured against a null.

    Raw worst-group coverage is the minimum of G noisy estimates, so it falls
    with G whether or not anything is wrong. Plotting the raw number beside the
    excess over a simulated Binomial null is the point of the figure: `source`
    looks like the worst axis, but most of that is what 25 small levels produce
    under the reference; its remaining excess (0.028) is small and rests on
    levels of a few test items.
    """
    s = pd.read_csv(MET / f"bench_axissummary_{tag}.csv")
    s = s[s.calib_axis == "marginal"]
    g = s.groupby("eval_axis").agg(
        G=("n_groups_eval", "mean"), worst=("worst_cov", "mean"),
        null=("null_worst", "mean"), excess=("excess_spread", "mean"))
    g = g.sort_values("excess", ascending=False)
    x = np.arange(len(g))

    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 1.65))
    ax = axes[0]
    ax.bar(x - 0.2, g["worst"], 0.4, label="observed",
           color="#4c72b0", edgecolor="black", linewidth=.3)
    ax.bar(x + 0.2, g["null"], 0.4, label="null",
           color="#c9c9c9", edgecolor="black", linewidth=.3)
    ax.axhline(TARGET, color="black", ls="--", lw=1)
    ax.set_ylabel("worst-group coverage")
    # Start above zero: the whole point is the gap between the pairs, which a
    # zero-based axis compresses into invisibility.
    ax.set_ylim(0.6, 0.96)
    ax.legend(loc="lower left", handlelength=1.0, ncol=2, columnspacing=0.8)

    ax = axes[1]
    colors = ["#c44e52" if v > 0.02 else "#b8b8b8" for v in g["excess"]]
    ax.bar(x, g["excess"], 0.6, color=colors, edgecolor="black", linewidth=.3)
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("excess spread over null")

    short = {"modality": "mod", "question_type": "qtype",
             "position": "pos", "source": "src", "K": "K"}
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{short.get(a, a)}\n{v:.0f}" if abs(v - round(v)) < 0.05
             else f"{short.get(a, a)}\n{v:.1f}"
             for a, v in zip(g.index, g["G"])], fontsize=FS_TICK)
        ax.set_xlabel("evaluation axis, and its $G$")
    fig.tight_layout(pad=0.2, w_pad=0.5)
    return _save(fig, "B3_axes.png")


def fig_ncal(tag="mcq"):
    """B4 - what Mondrian costs at a realistic calibration budget."""
    d = pd.read_csv(MET / f"bench_ncal_{tag}.csv")
    g = d.groupby("n_cal").agg(
        marg=("cov_marginal", "mean"), mond=("cov_mondrian", "mean"),
        marg_sd=("cov_marginal", "std"), beta=("beta_sd", "mean"),
        beta_total=("beta_sd_total", "mean"),
        worst_marg=("worst_marginal", "mean"), worst_mond=("worst_mondrian", "mean"),
        size_marg=("size_marginal", "mean"), size_mond=("size_mondrian", "mean"),
        disabled=("n_below_floor", "mean"))
    x = g.index.values

    fig, axes = plt.subplots(1, 3, figsize=(COL_IN, 1.45))
    ax = axes[0]
    ax.plot(x, g["marg_sd"], "o-", color="#4c72b0", label="observed")
    ax.plot(x, g["beta_total"], "s--", color="#c44e52", label="Beta + test")
    ax.plot(x, g["beta"], "^:", color="#888", label="Beta only")
    ax.set_ylabel("sd of coverage")
    ax.legend(handlelength=1.4, loc="upper right")

    ax = axes[1]
    ax.plot(x, g["worst_marg"], "o-", color="#4c72b0", label="marginal")
    ax.plot(x, g["worst_mond"], "s-", color="#c44e52", label="Mondrian")
    ax.axhline(TARGET, color="black", ls="--", lw=1)
    ax.set_ylabel("worst-group coverage")
    ax.legend(handlelength=1.2)

    ax = axes[2]
    ax.plot(x, g["size_marg"], "o-", color="#4c72b0", label="marginal")
    ax.plot(x, g["size_mond"], "s-", color="#c44e52", label="Mondrian")
    ax.set_ylabel("mean set size")
    ax.legend(handlelength=1.2)
    for n, dis in zip(x, g["disabled"]):
        if dis >= 1:
            ax.annotate(f"{dis:.1f} below\nfloor", (n, g["size_mond"][n]),
                        textcoords="offset points", xytext=(3, -7),
                        fontsize=5.0, color="#c44e52")

    axes[1].set_xlabel("calibration items")
    fig.tight_layout(pad=0.2, w_pad=0.5)
    return _save(fig, "B4_ncal.png")


def fig_order(tag="mcq"):
    """B5 - the letter effect and the item effect, separated by re-extraction."""
    pos = pd.read_csv(MET / f"bench_order_position_{tag}.csv")
    from cpaudit.bench.order_effect import decompose
    letter, item = decompose(pos)
    if letter is None:
        print("  B5 skipped: needs all four cyclic shifts")
        return None
    order = _order(pos.method)

    # Two panels, column width. The third panel used to show the ensembling
    # remedy; at three panels the figure only fitted full width, which cost
    # more than a column of page for a result the text already states
    # numerically (the per-split spread reductions and their adjusted p).
    # Each point is the mean over 20 re-splits of a per-split profile value;
    # bars are the 95% Student-t interval of that mean. Scores are dodged
    # horizontally so the bars do not overprint.
    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 1.6), sharey=True)
    dodge = dict(zip(order, np.linspace(-0.14, 0.14, len(order))))
    for ax, frame, key in [(axes[0], letter, "letter"),
                           (axes[1], item, "orig_index")]:
        for m in order:
            g = frame[frame.method == m].groupby(key).coverage
            mu, hw = g.mean(), g.agg(_ci)
            ax.errorbar(mu.index + dodge[m], mu.values, yerr=hw.values,
                        fmt="o-", color=COL[m], label=m.replace("base-", ""), ms=3, capsize=1.2,
                        elinewidth=0.6)
        ax.axhline(TARGET, color="black", ls="--", lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(list("ABCD") if key == "letter" else
                           ["1st", "2nd", "3rd", "4th"], fontsize=FS_TICK)
        ax.set_xlabel("letter shown" if key == "letter" else
                      "original-index cohort")
    axes[0].set_ylabel("coverage")
    axes[0].legend(ncol=1, handlelength=1.2, loc="lower left")
    fig.tight_layout(pad=0.2, w_pad=0.4)
    return _save(fig, "B5_order.png")


def fig_subgroups(tag="mcq", method="base-LAC"):
    """B7 - what near-nominal marginal coverage hides, and what fixing it costs.

    One row per modality. Left: coverage under marginal calibration (open) and
    per-modality Mondrian calibration (filled). Right: the matching mean set
    size. Points are means over re-splits of each split's per-modality value;
    bars are 95% Student-t intervals of that mean. The dashed line is the
    nominal level 1 - alpha, drawn as a target, not as a guarantee: these
    splits are stratified, so the theorem does not apply to them exactly.
    """
    d = pd.read_csv(MET / f"bench_groupaxes_{tag}.csv")
    d = d[(d.eval_axis == "modality") & (d.method == method)
          & (d.calib_axis.isin(["marginal", "modality"]))]
    marg = d[d.calib_axis == "marginal"].groupby("group")
    order = marg.coverage.mean().sort_values().index.tolist()   # worst on top
    y = np.arange(len(order))[::-1]
    style = {"marginal": dict(mfc="white", mec="#555555", color="#555555",
                              marker="o", label="marginal"),
             "modality": dict(mfc=COL["base-LAC"], mec=COL["base-LAC"],
                              color=COL["base-LAC"], marker="s",
                              label="per-modality (Mondrian)")}
    off = {"marginal": 0.14, "modality": -0.14}
    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 1.9), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax, col, xl in ((axes[0], "coverage", "coverage"),
                        (axes[1], "avg_size", "mean set size (options)")):
        for cal, st in style.items():
            g = d[d.calib_axis == cal].groupby("group")[col]
            mu = g.mean().reindex(order).values
            hw = g.agg(_ci).reindex(order).values
            ax.errorbar(mu, y + off[cal], xerr=hw, fmt=st["marker"], ms=3,
                        mfc=st["mfc"], mec=st["mec"], color=st["color"],
                        elinewidth=0.6, capsize=1.0, lw=0, label=st["label"])
        ax.set_xlabel(xl)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].axvline(TARGET, color="black", ls="--", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_SHORT.get(g, g) for g in order], fontsize=FS_TICK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, handlelength=1.0,
               bbox_to_anchor=(0.55, 1.0), columnspacing=1.2)
    fig.tight_layout(pad=0.2, w_pad=0.5, rect=(0, 0, 1, 0.9))
    return _save(fig, "B7_subgroups.png")


def fig_tuning_leak(tag="mcq"):
    """B6 - what selecting hyperparameters on the calibration set costs."""
    d = pd.read_csv(MET / f"bench_tuningleak_{tag}.csv")
    fig, ax = plt.subplots(figsize=(0.84 / 0.95 * COL_IN, 1.5))
    style = {"base-LAC": "o-", "APS": "s-", "RAPS": "^-"}
    for m in d.method.unique():
        g = d[d.method == m].pivot_table(index="M", columns="arm",
                                         values="coverage", aggfunc="mean")
        ax.plot(g.index, g["leaky-full"] - g["clean"], style.get(m, "o-"),
                color=COL.get(m, "#333"), label=m, ms=4)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xscale("log")
    ax.set_xlabel("configurations searched ($M$)")
    ax.set_ylabel("coverage lost vs clean")
    ax.legend(handlelength=1.5)
    fig.tight_layout(pad=0.2)
    return _save(fig, "B6_tuning_leak.png")


#: The manuscript prints B7 (per-modality coverage and set size) and B5
#: (cyclic presentations); its Fig. 1, the search-role diagram, is TikZ in
#: main.tex. The others say in a picture what the text says in numbers or
#: tables -- the answer-only control (B1), the axis table (B3), the older
#: per-group bar chart (B2), the budget sweep (B4) and the leak's dose-response
#: (B6) -- so `--extra` writes them for working use only.
EXTRA = ("shortcut", "axes", "per_group", "ncal", "tuning_leak")


def main(extra=False):
    print(f"Writing benchmark figures -> {P.CONF_FIGURES}")
    fig_subgroups()
    fig_order()
    if extra:
        fig_shortcut()
        fig_axes()
        fig_per_group()
        fig_ncal()
        fig_tuning_leak()
    print("done.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true",
                    help="also write B1-B4 and B6, which the manuscript omits")
    main(ap.parse_args().extra)
