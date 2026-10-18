"""Feature cache. Downstream training/calibration reads ONLY this;
the VLM is never re-run.

K varies per item (2-5), so ragged arrays are padded to K_max with a boolean
`mask`. Padded score slots are +inf, so they can never enter a prediction set.

Per-item cached record:
    id, s_base[K], z_x[d], e_y[K, d_e], answer_idx, modality, question_type, K
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INF = np.float32(np.inf)


@dataclass
class FeatureCache:
    ids: np.ndarray            # (N,)   str
    K: np.ndarray              # (N,)   int
    mask: np.ndarray           # (N, Kmax) bool   True = real option
    s_base: np.ndarray         # (N, Kmax) float32, +inf on pad
    z_x: np.ndarray            # (N, d)  float32   (shared across options)
    e_y: np.ndarray            # (N, Kmax, d_e) float32, 0 on pad
    answer_idx: np.ndarray     # (N,)   int
    modality: np.ndarray       # (N,)   str
    question_type: np.ndarray  # (N,)   str

    def __post_init__(self):
        n, kmax = self.s_base.shape
        assert self.mask.shape == (n, kmax)
        assert self.e_y.shape[:2] == (n, kmax)
        assert len(self.ids) == n == len(self.answer_idx)
        # True label must be a real (unmasked) option.
        assert self.mask[np.arange(n), self.answer_idx].all(), "answer_idx points at a pad slot"

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def kmax(self) -> int:
        return self.s_base.shape[1]

    def subset(self, idx) -> "FeatureCache":
        idx = np.asarray(idx)
        return FeatureCache(
            ids=self.ids[idx], K=self.K[idx], mask=self.mask[idx],
            s_base=self.s_base[idx], z_x=self.z_x[idx], e_y=self.e_y[idx],
            answer_idx=self.answer_idx[idx], modality=self.modality[idx],
            question_type=self.question_type[idx],
        )

    def true_base_scores(self) -> np.ndarray:
        """s_base at the true-label position, (N,)."""
        return self.s_base[np.arange(self.n), self.answer_idx]

    def group(self, key: str) -> np.ndarray:
        if key == "modality":
            return self.modality
        if key == "question_type":
            return self.question_type
        raise KeyError(key)


def save_cache(cache: FeatureCache, path: str | Path) -> None:
    """Persist to a compressed .npz shard (parquet is awkward for the 3-D e_y)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=cache.ids, K=cache.K, mask=cache.mask, s_base=cache.s_base,
        z_x=cache.z_x, e_y=cache.e_y, answer_idx=cache.answer_idx,
        modality=cache.modality, question_type=cache.question_type,
    )


def load_cache(path: str | Path) -> FeatureCache:
    d = np.load(Path(path), allow_pickle=True)
    return FeatureCache(
        ids=d["ids"], K=d["K"], mask=d["mask"], s_base=d["s_base"],
        z_x=d["z_x"], e_y=d["e_y"], answer_idx=d["answer_idx"],
        modality=d["modality"], question_type=d["question_type"],
    )


def cache_report(cache: FeatureCache) -> pd.DataFrame:
    """Per-modality counts — quick sanity table."""
    df = pd.DataFrame({"modality": cache.modality, "K": cache.K})
    return df.groupby("modality").agg(n=("K", "size"), meanK=("K", "mean"))
