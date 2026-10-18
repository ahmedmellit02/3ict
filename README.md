# Beyond Marginal Coverage: Auditing Conformal Prediction for Medical VQA

Code, identifiers, split assignments, score caches and per-split results for the
3ICT 2026 paper *Beyond Marginal Coverage: Auditing Conformal Prediction for
Medical VQA* (A. Mellit, A. Cohen, H. Benaboud).

The paper varies the protocol around three post-hoc conformal scores (LAC, APS,
RAPS) on OmniMedVQA and VQA-RAD, using cached letter scores of two frozen
vision-language models (Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct), over
20 re-splits per corpus.

No dataset content (questions, options, images) and no model weights or image
features are included. The data files under `data/` hold only each item's id
and image path, which the image-disjoint split and the image-sharing statistics
read.

## Layout

| Path | Contents |
|---|---|
| `cpaudit/` | the code: scores (`baselines/`), calibration (`calib/`), splits and caches (`data/`), statistics (`eval/`), every analysis (`bench/`), figures (`figures/`), score extraction (`extract/`), the learned-head controls (`model/`) |
| `paper/` | manuscript source, references, figures, `claims.py`, `verify.py` |
| `paper/materials/` | item identifiers, split assignments, selected configurations, VQA-RAD organ map, search grids, extraction record, score caches |
| `results/metrics/bench_*.csv` | per-split results behind every number in the paper |
| `tests/test_protocol.py` | protocol tests |

## Check the paper against the results (no data or GPU needed)

    pip install -r requirements.txt
    python paper/claims.py    # recomputes every pinned number from the CSVs
    python paper/verify.py    # numbers, citations, figures, references, float order

## Re-run analyses from the score caches

The caches in `paper/materials/scores/` hold the frozen models' letter scores
for every item: both backbones, OmniMedVQA in its four cyclic option orders
(`_s1`..`_s3`) and VQA-RAD in both yes/no orders (`_s1`); `_q3vl4b` is
Qwen3-VL-4B. Copy them to `cache/`, then run the tests or any post-hoc analysis:

    mkdir cache
    cp paper/materials/scores/*.npz cache/
    python -m tests.test_protocol
    python -m cpaudit.bench.run_all --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
    python -m cpaudit.bench.run_all --cache cache/vqarad_mcq.npz --tag vqarad_img --cluster

Each analysis rewrites its `results/metrics/bench_*.csv`; the shipped copies are
in git, so `git diff` shows any difference. The commands behind every CSV:

    # suites (Tables I and III, Sections IV-V); --cluster = image-disjoint split
    python -m cpaudit.bench.run_all --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
    python -m cpaudit.bench.run_all --cache cache/vqarad_mcq.npz --tag vqarad_img --cluster --only group_axes tuning_leak ncal_sweep randomization alpha_sweep heterogeneity
    python -m cpaudit.bench.run_all --cache cache/vqarad_mcq.npz --tag vqarad_mcq --only tuning_leak alpha_sweep randomization
    python -m cpaudit.bench.heterogeneity --cache cache/omnimedvqa_3000_mcq.npz --tag mcq_uniqimg --unique-images
    python -m cpaudit.bench.split_unit --cache cache/vqarad_mcq.npz --tag vqarad --paired vqarad_mcq vqarad_img --one-per-image
    python -m cpaudit.bench.split_unit --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
    python -m cpaudit.bench.stream_check --cache cache/omnimedvqa_3000_mcq.npz --tag mcq
    python -m cpaudit.bench.stream_check --cache cache/vqarad_mcq.npz --tag vqarad_img --cluster
    python -m cpaudit.bench.group_axes --cache - --tag mcq --shared-reference
    python -m cpaudit.bench.group_axes --cache - --tag vqarad_img --shared-reference
    python -m cpaudit.bench.ncal_sweep --cache cache/omnimedvqa_3000_mcq.npz --tag mcq_a005 --alpha 0.05
    python -m cpaudit.bench.ncal_sweep --cache cache/omnimedvqa_3000_mcq.npz --tag mcq_a020 --alpha 0.20
    python -m cpaudit.bench.ncal_sweep --cache cache/omnimedvqa_3000_mcq.npz --tag mcq_qtype --axis question_type
    python -m cpaudit.bench.lac_invariance
    # option order (Section V-C)
    python -m cpaudit.bench.run_all --cache cache/omnimedvqa_3000_mcq.npz --tag mcq --only order_effect
    python -m cpaudit.bench.order_effect --tag vqarad_img --vqarad
    # second backbone (Section VI, Table IV)
    python -m cpaudit.bench.run_all --cache cache/omnimedvqa_3000_mcq_q3vl4b.npz --tag mcq_q3vl4b --only alpha_sweep group_axes tuning_leak randomization
    python -m cpaudit.bench.run_all --cache cache/vqarad_mcq_q3vl4b.npz --tag vqarad_img_q3vl4b --cluster --only alpha_sweep group_axes tuning_leak randomization
    python -m cpaudit.bench.lac_invariance --backbone q3vl4b
    python -m cpaudit.bench.order_effect --tag mcq_q3vl4b --backbone q3vl4b
    python -m cpaudit.bench.order_effect --tag vqarad_img_q3vl4b --backbone q3vl4b --vqarad
    # derived tables, run last
    python -m cpaudit.bench.order_decomp
    python -m cpaudit.bench.effects_table
    python -m cpaudit.figures.bench_figs

`split_unit --paired` reads the `vqarad_mcq` and `vqarad_img` CSVs, so run it
after both suites. Results are deterministic given the seed.

The controls of Section III (`answer_shortcut`, `question_lookup`,
`supervised_control`, `shortcut_breakdown`, `order_cohorts`) read the option
and question text, and the learned heads also read the image features, so they
need the public datasets and the extraction step below.

## Regenerating the score caches

`cpaudit/extract/extract_mcq.py` runs a frozen model once per item under the
letter-logit protocol: image, question and the options as lines "A. ...", and
s_base(x, y_k) = -log p(letter_k | x) at the first generated position, over the
full vocabulary. `--shift s` shows option (k + s) mod K behind letter k;
`--model` selects the backbone. Model revisions, precision, processor limits
and the prompt are in `paper/materials/extraction.json`. The VQA-RAD organs come
from the original release on OSF (`cpaudit/data/vqarad_organ.py`, which pins
the file's sha256); without that file the mapping is read from
`paper/materials/item_ids.csv`. Extraction needs `requirements-gpu.txt`.

## Where each protocol detail lives

| Item | Location |
|---|---|
| Item identifiers | `paper/materials/item_ids.csv` (3,000 OmniMedVQA open-access items, 375 per modality; 1,191 VQA-RAD closed yes/no questions with image and released organ) |
| VQA-RAD organ mapping, match counts, ambiguity handling | `paper/materials/vqarad_organ_map.csv`; `cpaudit/data/vqarad_organ.py` |
| Split assignments | `paper/materials/splits_*.csv` (every item, re-splits 0-19) |
| Grids, search order, tie-breaking | `paper/materials/grids.json`; `cpaudit/bench/common.py` (`posthoc_grid`, `select_config`: strict <, first entry wins) |
| Selected configuration per split | `paper/materials/selected_configs.csv` |
| Keyed randomness | `cpaudit/bench/common.py::keyed_uniforms`: key = SplitMix64(SplitMix64(re-split) XOR blake2b-64(item id)), then XOR SplitMix64(option index + 1), finalised by SplitMix64; uniform = top 53 bits / 2^53. Options are indexed by their original position, so a rotated presentation keeps each option's draw. |
| APS/RAPS scores | `cpaudit/baselines/posthoc.py::_cumulative_scores` (C_k - (1 - u) p_k [+ RAPS penalty], C normalised to exactly 1 at the last rank; stable ranking) |
| Threshold and fallback | `cpaudit/calib/conformal.py::conformal_quantile` with `cpaudit.bench.common.JITTER = 0` (plain k-th order statistic); `predict_sets` (inclusive rule, argmax fallback) |
| Calibration reuse (Fig. 1 roles) | `cpaudit/bench/tuning_leak.py` |
| Protocol-effects table (Table II) | `cpaudit/bench/effects_table.py` -> `bench_effects_summary.csv` |
| Subgroup references (Table III) | `cpaudit/bench/group_axes.py` (`null_worst`, `null_worst_shared`); permutation test in `cpaudit/bench/heterogeneity.py` |
| Option order (Sections V-C, VI) | `cpaudit/bench/order_effect.py`, `cpaudit/bench/order_decomp.py` |
| Intervals, paired tests, Holm | `cpaudit/eval/stats.py`; confirmatory families in `tuning_leak.confirmatory` and `order_effect.confirmatory` |

## License

No license has been chosen yet; until one is added, all rights are reserved by
the authors.
