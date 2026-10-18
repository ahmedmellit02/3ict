"""Post-hoc conformal scores on the frozen models' cached letter scores.

LAC, APS, RAPS and a naive softmax threshold -- pure numpy, no torch.
"""
from .posthoc import (
    probs_from_base,
    lac_scores,
    aps_scores,
    raps_scores,
    softmax_threshold_sets,
    BASELINE_SCORERS,
)

__all__ = [
    "probs_from_base", "lac_scores", "aps_scores", "raps_scores",
    "softmax_threshold_sets", "BASELINE_SCORERS",
]
