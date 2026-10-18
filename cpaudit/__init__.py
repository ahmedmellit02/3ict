"""cpaudit -- auditing split-conformal protocols on frozen medical VLM scores.

Code for *Beyond Marginal Coverage: Auditing Conformal Prediction for Medical
VQA* (3ICT 2026). The conformal core (data, calib, baselines, eval, bench,
figures) is pure numpy and runs without a GPU. Score extraction from the frozen
models (extract/) and the learned-head controls of Section III
(bench.supervised_control, bench.shortcut_breakdown, model/) need torch.
"""

__version__ = "1.0.0"
