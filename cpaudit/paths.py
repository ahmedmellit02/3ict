"""Canonical output locations.

Every experiment and figure script writes through this module rather than
through a path literal. Paths are anchored to the repository root (derived from
this file's location), not to `os.getcwd()`: a relative literal like
`"results/metrics"` silently creates a second, empty results tree when a script
is run from anywhere other than the repository root.

Layout::

    results/metrics/   metric CSVs -- the source of every number the paper cites
    results/logs/      run logs
    paper/figures/     the paper's figures, beside main.tex

Override. `CPAUDIT_RESULTS` relocates the run-artefact tree, and the figures
with it, so a scratch run is self-contained and cannot touch the shipped
results::

    CPAUDIT_RESULTS=/tmp/scratch python -m cpaudit.bench.lac_invariance

`ensure()` creates a directory on demand and returns it.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Repository root -- the directory containing `cpaudit/`, `paper/`, `results/`.
ROOT = Path(__file__).resolve().parent.parent

_results_env = os.environ.get("CPAUDIT_RESULTS")

#: Root of the run-artefact tree (metrics, logs).
RESULTS = Path(_results_env) if _results_env else ROOT / "results"

METRICS = RESULTS / "metrics"     # metric CSVs
LOGS = RESULTS / "logs"           # run logs

#: The paper and its figures. Follows CPAUDIT_RESULTS so a scratch run keeps its
#: figures with its metrics.
CONF_PAPER = ROOT / "paper"
CONF_FIGURES = (RESULTS / "conf_figures") if _results_env else CONF_PAPER / "figures"


def ensure(p: Path) -> Path:
    """Create `p` (and parents) if absent, then return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def metric(name: str) -> Path:
    """Path to a metric CSV, with the directory created."""
    return ensure(METRICS) / name


def conf_figure(name: str) -> Path:
    """Path to a figure of the paper, with the directory created."""
    return ensure(CONF_FIGURES) / name
