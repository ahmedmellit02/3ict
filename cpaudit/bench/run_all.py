"""Run the post-hoc suite on one score cache and write its bench_*.csv files.

    python -m cpaudit.bench.run_all --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
    python -m cpaudit.bench.run_all --cache cache/vqarad_mcq.npz --tag vqarad_img --cluster

`--only` runs a subset by name. Every experiment writes through `cpaudit.paths`,
so `CPAUDIT_RESULTS=/tmp/scratch` gives an isolated run that cannot touch the
shipped results.
"""
from __future__ import annotations

import argparse
import time
import traceback

from cpaudit import paths as P
from cpaudit.bench.common import BASELINES

#: name -> (module, needs_torch, runs_on_a_named_cache)
EXPERIMENTS = {
    "group_axes": ("cpaudit.bench.group_axes", False, True),
    "tuning_leak": ("cpaudit.bench.tuning_leak", False, True),
    "ncal_sweep": ("cpaudit.bench.ncal_sweep", False, True),
    "randomization": ("cpaudit.bench.randomization", False, True),
    "alpha_sweep": ("cpaudit.bench.alpha_sweep", False, True),
    "heterogeneity": ("cpaudit.bench.heterogeneity", False, True),
    "split_unit": ("cpaudit.bench.split_unit", False, True),
    # order_effect reads the shifted caches itself rather than one --cache.
    "order_effect": ("cpaudit.bench.order_effect", False, False),
}
#: Experiments whose results the paper's core claims rest on. Run these first,
#: so a suite that dies halfway still leaves the paper standing.
PRIORITY = ("group_axes", "tuning_leak", "ncal_sweep", "randomization",
            "alpha_sweep", "heterogeneity", "split_unit",
            "order_effect")


def _write(name, tag, frames):
    for suffix, frame in frames.items():
        path = P.metric(f"bench_{suffix}_{tag}.csv" if suffix != name
                        else f"bench_{name}_{tag}.csv")
        frame.to_csv(path, index=False)
        print(f"    wrote {path.name}  ({len(frame)} rows)")


def run_one(name, cache, tag, seeds, include, budget, cluster=False):
    import importlib

    mod = importlib.import_module(EXPERIMENTS[name][0])
    takes_cache = EXPERIMENTS[name][2]
    kw = dict(n_seeds=seeds, include=include, budget=budget,
              cluster=cluster)

    if name == "group_axes":
        pg, pos = mod.run(cache, **kw)
        summary = mod.add_null_baseline(mod.summarize(pg), pg)
        _write(name, tag, {"groupaxes": pg, "axissummary": summary,
                           "position_dist": pos})
        mod.report(summary, tag)
        mod.position_report(pos, tag)
    elif name == "tuning_leak":
        df = mod.run(cache, n_seeds=seeds, cluster=cluster)
        st, conf = mod.stats(df), mod.confirmatory(df)
        _write(name, tag, {"tuningleak": df, "stats_tuningleak": st,
                           "confirm_tuningleak": conf})
        mod.report(df, st, tag, conf=conf)
    elif name == "ncal_sweep":
        df = mod.run(cache, **kw)
        _write(name, tag, {"ncal": df})
        mod.report(df, tag)
    elif name == "randomization":
        df = mod.run(cache, n_seeds=seeds, cluster=cluster)
        st = mod.stats(df)
        _write(name, tag, {"random": df, "stats_random": st})
        mod.report(df, st, tag)
    elif name == "alpha_sweep":
        df, ssc = mod.run(cache, n_seeds=seeds, include=include,
                      budget=budget, cluster=cluster)
        st = mod.stats(df)
        _write(name, tag, {"alpha": df, "ssc": ssc, "stats_alpha": st})
        mod.report(df, ssc, st, tag)
    elif name == "heterogeneity":
        # Permutation labels are resampled at the image level whenever the
        # split is image-disjoint, so the null respects the same unit.
        df = mod.run(cache, n_seeds=seeds, cluster=cluster,
                     methods=tuple(m for m in include if m in mod.POSTHOC))
        s = mod.summarize(df)
        _write(name, tag, {"heterogeneity": df, "heterogeneity_holm": s})
        mod.report(s, tag)
    elif name == "split_unit":
        df = mod.run(cache, n_seeds=seeds)
        _write(name, tag, {"splitunit": df})
        import numpy as np
        from cpaudit.bench.common import image_of, load_items
        from cpaudit.bench.common import load_cache
        c = load_cache(cache)
        mod.report(df, cache, len(np.unique(image_of(c, load_items()))), c.n)
    elif name == "order_effect":
        df, pos = mod.run(n_seeds=seeds, include=include, budget=budget)
        st, spread = mod.stats(df, pos)
        conf = mod.confirmatory(df, pos)
        _write(name, tag, {"order": df, "order_position": pos,
                           "stats_order": st, "confirm_order": conf})
        mod.report(df, pos, st, spread, tag)
        print("\nCONFIRMATORY: does order-ensembling reduce position-conditional "
              "spread? (Holm across these six tests only)")
        print(conf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    else:
        raise KeyError(name)
    assert takes_cache or name == "order_effect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/omnimedvqa_3000_mcq.npz")
    ap.add_argument("--tag", default="mcq")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS),
                    help="run just these experiments")
    ap.add_argument("--cluster", action="store_true",
                    help="image-disjoint split (see data.split."
                         "grouped_four_way_split)")
    ap.add_argument("--keep-going", action="store_true",
                    help="report a failing experiment and continue with the rest")
    args = ap.parse_args()

    include = BASELINES
    todo = [n for n in PRIORITY if not args.only or n in args.only]
    print(f"[run_all] {len(todo)} experiments on {args.tag}: {', '.join(todo)}")
    print(f"[run_all] scores: {', '.join(include)}   seeds: {args.seeds}")
    print(f"[run_all] metrics -> {P.METRICS}")

    t0, failed = time.time(), []
    for name in todo:
        print(f"\n{'=' * 70}\n[run_all] {name}\n{'=' * 70}", flush=True)
        try:
            run_one(name, args.cache, args.tag, args.seeds, include,
                    "paper", cluster=args.cluster)
        except Exception:
            failed.append(name)
            print(f"[run_all] {name} FAILED:\n{traceback.format_exc()}", flush=True)
            if not args.keep_going:
                raise
        print(f"[run_all] {name} done, {(time.time() - t0) / 60:.1f} min elapsed",
              flush=True)

    print(f"\n[run_all] finished in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"[run_all] FAILED: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
