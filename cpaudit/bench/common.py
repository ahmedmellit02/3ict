"""Shared machinery for every analysis: one place that builds each score, one
place that splits, tunes and calibrates, and one place that names a grouping axis.

**Scorers, not score matrices.** `build_scorers` returns objects exposing
`.score(sub) -> (N, Kmax)` rather than precomputed `(S_cal, S_test)` pairs,
because many analyses need a score evaluated on subsets that are not D_cal or
D_test: a subsample of D_cal (calibration budget), the other half of D_thr
(tuning leakage), one option order (presentation).

**The protocol.** Splits are 0.50/0.15/0.15/0.20, stratified on the group
field (`split_indices`); hyperparameters are chosen on D_thr alone, halved so a
provisional threshold is fitted on one half and mean set size measured on the
other (`thr_halves`, `select_config`); APS/RAPS uniforms are keyed by
(re-split, item, option) (`keyed_uniforms`); and the threshold is the plain
order statistic with no jitter (`JITTER`). `tests/test_protocol.py` checks each
of these.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cpaudit.data import four_way_split, SPLIT_NAMES
from cpaudit.data import load_cache as _load_cache_raw
from cpaudit.baselines.posthoc import lac_scores, aps_scores, raps_scores
from cpaudit import paths as P

SPLIT_FRACS = {"train": .5, "thr": .15, "cal": .15, "test": .2}

#: The nominal miscoverage level of every headline result.
ALPHA = 0.10
#: Search grids, in search order (temperature outermost); ties in the
#: selection objective go to the first entry.
TEMPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RAPS_GAMMAS = [0.0, 0.01, 0.05, 0.1, 0.3]
RAPS_KREG = [1, 2]

#: The threshold is the plain k-th order statistic of the calibration scores,
#: compared with test scores by the inclusive rule s <= tau. The earlier
#: implementation (used for the submitted version of the paper) added
#: U(-1e-6, 1e-6) to every calibration score first; that perturbs the numerical
#: threshold, is applied to one side of the comparison only, and does not
#: commute with a temperature change, so it is not used. Ties are left as ties;
#: `s <= tau` already handles them. `randomization.py` keeps a jittered arm as a
#: sensitivity only.
JITTER = 0.0


def calibrate_marginal(cal_true_scores, alpha, seed: int = 0):
    """`cpaudit.calib.conformal.calibrate_marginal` without jitter (see JITTER)."""
    from cpaudit.calib.conformal import calibrate_marginal as _cm
    return _cm(cal_true_scores, alpha, jitter=JITTER, seed=seed)


def calibrate_mondrian(cal_true_scores, cal_groups, alpha, floor_mult=1.0,
                       seed: int = 0):
    """`cpaudit.calib.conformal.calibrate_mondrian` without jitter (see JITTER)."""
    from cpaudit.calib.conformal import calibrate_mondrian as _cmo
    return _cmo(cal_true_scores, cal_groups, alpha, floor_mult=floor_mult,
                jitter=JITTER, seed=seed)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser on uint64 arrays (wrapping arithmetic)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@functools.lru_cache(maxsize=None)
def _id_key(item_id: str) -> int:
    """A stable 64-bit key for an item id (not Python's salted `hash`)."""
    return int.from_bytes(hashlib.blake2b(item_id.encode(), digest_size=8).digest(),
                          "little")


def keyed_uniforms(ids, seed: int, kmax: int) -> np.ndarray:
    """U(0,1) draws keyed by (re-split, item id, original option index).

    Distinct items get independent draws, wherever they sit: calibration row j
    and test row j no longer share one, which the reset-per-call generator of
    the earlier implementation made them do. The same item and option get the SAME draw
    under every configuration, arm, calibration subset and cyclic rotation
    (caches keep original option indexing), which is what pairs those
    comparisons. APS and RAPS see identical draws.
    """
    with np.errstate(over="ignore"):
        idk = np.array([_id_key(str(i)) for i in ids], dtype=np.uint64)
        z = _splitmix64(_splitmix64(np.full(len(idk), seed, dtype=np.uint64)) ^ idk)
        k = np.arange(1, kmax + 1, dtype=np.uint64)
        z = _splitmix64(z[:, None] ^ _splitmix64(k)[None, :])
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def load_cache(path):
    """`cpaudit.data.load_cache`, with VQA-RAD's group labels replaced by the
    organ the original release files each image under.

    The VQA-RAD cache was extracted with a keyword guess from the question
    text in its `modality` slot, and that field stratifies every split and
    defines every organ group. The guess is not a released annotation, so every
    analysis loads through here and gets the organ the original release
    records for each image (`cpaudit.data.vqarad_organ.organ_by_id()`).
    """
    cache = _load_cache_raw(path)
    if cache.n and str(cache.ids[0]).startswith("vqarad_"):
        from cpaudit.data.vqarad_organ import organ_by_id
        org = organ_by_id()
        cache = dataclasses.replace(
            cache, modality=np.array([org[str(i)] for i in cache.ids]))
    return cache

#: The three post-hoc scores, in the order the paper's tables use.
POSTHOC = ("base-LAC", "APS", "RAPS")
BASELINES = POSTHOC

RAPS_EXTRA = [dict(gamma=g, k_reg=k) for g in RAPS_GAMMAS for k in RAPS_KREG]


def posthoc_grid(name: str) -> list[dict]:
    """Every configuration the tuner searches for one post-hoc baseline.

    Enumerated rather than nested-looped so an experiment can subsample the grid
    and sweep the SIZE of the search -- which is what turns tuning leakage from
    a yes/no question into a dose-response.
    """
    if name == "RAPS":
        return [dict(temp=T, **ex) for T in TEMPS for ex in RAPS_EXTRA]
    if name in POSTHOC:
        return [dict(temp=T) for T in TEMPS]
    raise KeyError(f"{name!r} is not a post-hoc baseline")


def sel_size(S_a, sub_a, S_b, sub_b, seed, alpha=ALPHA, jitter=JITTER):
    """The selection objective: mean set size on half B after calibrating
    tau on half A at level `alpha`."""
    from cpaudit.calib.conformal import conformal_quantile, predict_sets
    from cpaudit.eval.metrics import avg_set_size

    true_a = S_a[np.arange(len(sub_a.answer_idx)), sub_a.answer_idx]
    tau = conformal_quantile(true_a, alpha, jitter, np.random.default_rng(seed))
    sets, _ = predict_sets(S_b, sub_b.mask, tau)
    return avg_set_size(sets)


def select_config(name, configs, sub_a, sub_b, seed, alpha=ALPHA):
    """Pick the configuration with the smallest mean set size, measured the
    project's way: calibrate tau on half A, measure on half B.

    Which halves those are is the caller's choice, and is exactly what the
    leakage experiment varies -- D_thr (legal) or D_cal (the leak). Ties in the
    objective go to the first configuration in `configs` order (strict <).
    """
    fn = _POSTHOC_FN[name]
    best, best_cfg = np.inf, None
    for cfg in configs:
        sc = PostHoc(fn, cfg, seed)
        v = sel_size(sc.score(sub_a), sub_a, sc.score(sub_b), sub_b, seed, alpha)
        if v < best:
            best, best_cfg = v, cfg
    return best_cfg, best


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------
@dataclass
class PostHoc:
    """A post-hoc scorer bound to its tuned hyperparameters.

    `stream` sets where APS/RAPS take their uniform draws:
      "keyed" -- `keyed_uniforms(sub.ids, seed)`: one draw per (re-split,
                 item, option), independent across items. The benchmark default.
      "reset" -- the earlier implementation's generator, seeded by `seed`
                 and restarted on every call, so row j of the calibration and
                 test calls shares draws. Kept only as the sensitivity arm of
                 `stream_check`.
    """
    fn: object
    kwargs: dict = field(default_factory=dict)
    seed: int = 0
    stream: str = "keyed"

    @property
    def _takes_seed(self) -> bool:
        return "seed" in self.fn.__code__.co_varnames

    def score(self, sub) -> np.ndarray:
        if not self._takes_seed:
            return self.fn(sub.s_base, sub.mask, **self.kwargs)
        if self.stream == "reset":
            return self.fn(sub.s_base, sub.mask, seed=self.seed, **self.kwargs)
        u = keyed_uniforms(sub.ids, self.seed, sub.s_base.shape[1])
        return self.fn(sub.s_base, sub.mask, seed=self.seed, u=u, **self.kwargs)


_POSTHOC_FN = {"base-LAC": lac_scores, "APS": aps_scores, "RAPS": raps_scores}


def image_of(cache, items: dict | None = None) -> np.ndarray:
    """Per-item image identifier, for cluster-disjoint splitting."""
    items = items if items is not None else load_items()
    return np.array([items[str(i)]["image_path"] if str(i) in items else str(i)
                     for i in cache.ids], dtype=object)


def split_indices(cache, seed: int, cluster: bool = False,
                  items: dict | None = None) -> dict[str, np.ndarray]:
    """The project's split: 0.50/0.15/0.15/0.20, stratified on modality.

    `cluster=True` additionally keeps every question about one image inside a
    single partition. That is the honest unit for VQA-RAD, which asks ~4
    questions per image; on OmniMedVQA it is close to a no-op because images
    are effectively unique.
    """
    if cluster:
        from cpaudit.data.split import grouped_four_way_split
        img = image_of(cache, items)
        rows = [{"modality": m, "image_path": p}
                for m, p in zip(cache.modality, img)]
        sp = grouped_four_way_split(rows, SPLIT_FRACS, "modality",
                                    "image_path", seed=seed)
    else:
        sp = four_way_split([{"modality": m} for m in cache.modality],
                            SPLIT_FRACS, "modality", seed=seed)
    return {s: np.array(sp[s]) for s in SPLIT_NAMES}


def thr_halves(cache, thr, seed: int):
    """Halve D_thr for hyperparameter selection: tau is calibrated on A, mean
    set size measured on B. D_cal is never read."""
    perm = np.random.default_rng(1234 + seed).permutation(thr)
    h = len(perm) // 2
    return cache.subset(perm[:h]), cache.subset(perm[h:])


def tune_posthoc(cache, thr, seed: int, sub_a=None, sub_b=None,
                 alpha: float = ALPHA) -> dict[str, dict]:
    """Tuned hyperparameters for LAC/APS/RAPS, selected on D_thr only, with
    keyed uniforms and no jitter."""
    if sub_a is None:
        sub_a, sub_b = thr_halves(cache, thr, seed)
    return {m: select_config(m, posthoc_grid(m), sub_a, sub_b, seed, alpha)[0]
            for m in POSTHOC}


def build_scorers(cache, idx, seed: int, include=BASELINES, tune: bool = True,
                  budget: str = "paper", alpha: float = ALPHA,
                  verbose: bool = False):
    """Build every requested score's scorer for one seed.

    Returns `(scorers, hparams)`. `scorers[name].score(sub)` gives the (N, Kmax)
    nonconformity matrix under the convention C(x) = {y : S <= tau}.
    Hyperparameters are selected at `alpha` on the halves of D_thr. `budget`
    is accepted for interface stability; the post-hoc grids are the same under
    every budget.
    """
    thr = idx["thr"]
    scorers: dict[str, object] = {}
    hparams: dict[str, dict] = {}
    sub_a, sub_b = thr_halves(cache, thr, seed)
    wanted = [m for m in include if m in POSTHOC]
    assert len(wanted) == len(include), f"unknown scores in {include}"
    hp = (tune_posthoc(cache, thr, seed, sub_a, sub_b, alpha)
          if tune else {m: {} for m in POSTHOC})
    for m in wanted:
        scorers[m] = PostHoc(_POSTHOC_FN[m], hp[m], seed)
        hparams[m] = hp[m]
    if verbose:
        print("    " + " | ".join(f"{m} {hparams[m]}" for m in scorers), flush=True)
    return scorers, hparams


# --------------------------------------------------------------------------
# grouping axes
# --------------------------------------------------------------------------
#: Candidate Mondrian partitions. `modality` is the primary one; the rest are
#: the alternatives the subgroup analysis compares against it.
AXES = ("modality", "question_type", "source", "position", "K")

_ITEM_FILES = (
    P.ROOT / "data" / "omnimedvqa" / "sample_items.jsonl",
    P.ROOT / "data" / "vqarad" / "yesno_items.jsonl",
)


def load_items(paths=_ITEM_FILES) -> dict[str, dict]:
    """id -> raw dataset record, across every items file present on disk."""
    out: dict[str, dict] = {}
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[str(rec["id"])] = rec
    return out


def source_dataset(item_id: str) -> str:
    """OmniMedVQA ids are '<source dataset>_<n>'; VQA-RAD ids are 'vqarad_<n>'."""
    return str(item_id).rsplit("_", 1)[0]


def axis_labels(sub, axis: str, items: dict | None = None,
                min_group: int = 0, other: str = "other") -> np.ndarray:
    """Per-item group label for one candidate Mondrian partition.

    `modality` and `question_type` ride on the cache. `position` (which letter
    the answer sat behind) and `K` are derived from the labels. `source` needs
    the raw items file, because the cache does not carry it -- note that
    OmniMedVQA's `anatomy` field is "Unknown" for every item in the sample and
    is therefore NOT offered as an axis.

    `min_group` pools rare levels into `other`, which matters for `source`: the
    3,000-item sample spans 42 source datasets, most far below the Mondrian
    sample-size floor.
    """
    if axis == "modality":
        lab = np.asarray(sub.modality, dtype=object)
    elif axis == "question_type":
        lab = np.asarray(sub.question_type, dtype=object)
    elif axis == "position":
        letters = "ABCDE"
        lab = np.array([f"pos-{letters[i]}" for i in sub.answer_idx], dtype=object)
    elif axis == "K":
        lab = np.array([f"K={k}" for k in sub.K], dtype=object)
    elif axis == "source":
        lab = np.array([source_dataset(i) for i in sub.ids], dtype=object)
    else:
        raise KeyError(f"unknown axis {axis!r}; expected one of {AXES}")

    if min_group > 0:
        uniq, cnt = np.unique(lab, return_counts=True)
        rare = set(uniq[cnt < min_group].tolist())
        if rare:
            lab = np.array([other if v in rare else v for v in lab], dtype=object)
    return lab
