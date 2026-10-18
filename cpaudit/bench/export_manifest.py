"""Write the reproduction manifest for the 3ICT paper.

Everything under paper/materials/ is regenerated from the caches, the
split code and the organ match, never edited by hand:

  item_ids.csv            every evaluated item: corpus, id, stratum (modality or
                          released organ), K, and for VQA-RAD its image.
  splits_<design>.csv     the partition of every item on every re-split
                          (columns s00..s19: train / thr / cal / test), for each
                          split design the paper uses.
  selected_configs.csv    the hyperparameters each score selected on D_thr, per
                          corpus and re-split (primary protocol).
  vqarad_organ_map.csv    each VQA-RAD image: our file, its OSF match, the
                          released organ, how it was matched, and every
                          question-set candidate.
  grids.json              the search grids, in the order searched (ties go to
                          the first entry).
  extraction.json         model, revision, preprocessing, prompt template and
                          score definition used to build the feature caches.

Per-split results, paired differences and rotation-cell counts are the per-seed
rows of results/metrics/bench_*.csv (see README.md for the map).

Run: python -m cpaudit.bench.export_manifest
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpaudit import paths as P
from cpaudit.bench.common import (POSTHOC, RAPS_EXTRA, SPLIT_FRACS, image_of,
                              load_cache, load_items, posthoc_grid,
                              split_indices, tune_posthoc)
from cpaudit.data.split import four_way_split

OUT = P.ROOT / "paper" / "materials"
CACHES = {"OmniMedVQA": "cache/omnimedvqa_3000_mcq.npz",
          "VQA-RAD": "cache/vqarad_mcq.npz"}
SEEDS = range(20)


def item_table() -> pd.DataFrame:
    items = load_items()
    rows = []
    for corpus, path in CACHES.items():
        c = load_cache(P.ROOT / path)
        img = image_of(c, items)
        for i, g, k, im in zip(c.ids, c.modality, c.K, img):
            rows.append(dict(corpus=corpus, id=str(i), stratum=str(g), K=int(k),
                             image=Path(str(im)).name if corpus == "VQA-RAD" else ""))
    return pd.DataFrame(rows)


def split_table(cache, cluster: bool) -> pd.DataFrame:
    out = pd.DataFrame({"id": [str(i) for i in cache.ids]})
    for s in SEEDS:
        lab = np.empty(cache.n, dtype=object)
        for part, ix in split_indices(cache, s, cluster=cluster).items():
            lab[ix] = part
        out[f"s{s:02d}"] = lab
    return out


def one_per_image_table(cache) -> pd.DataFrame:
    """Mirrors `split_unit.one_per_cluster`: which question represents each
    image on each re-split, and where it went (blank = not sampled)."""
    img = image_of(cache, load_items())
    order = np.argsort(img, kind="stable")
    _, first = np.unique(img[order], return_index=True)
    groups = np.split(order, first[1:])
    out = pd.DataFrame({"id": [str(i) for i in cache.ids]})
    for s in SEEDS:
        rng = np.random.default_rng(10_000 + s)
        pick = np.array(sorted(int(rng.choice(g)) for g in groups))
        sp = four_way_split([{} for _ in range(len(pick))], SPLIT_FRACS,
                            "_none", seed=s)
        lab = np.full(cache.n, "", dtype=object)
        for part, ix in sp.items():
            lab[pick[np.array(ix, dtype=int)]] = part
        out[f"s{s:02d}"] = lab
    return out


def config_table() -> pd.DataFrame:
    rows = []
    for corpus, path, cluster in (("OmniMedVQA", CACHES["OmniMedVQA"], False),
                                  ("VQA-RAD", CACHES["VQA-RAD"], True)):
        c = load_cache(P.ROOT / path)
        for s in SEEDS:
            idx = split_indices(c, s, cluster=cluster)
            for m, cfg in tune_posthoc(c, idx["thr"], s).items():
                rows.append(dict(corpus=corpus, seed=s, score=m, **cfg))
    return pd.DataFrame(rows)


def organ_table() -> pd.DataFrame:
    from cpaudit.data.vqarad_organ import (ORGAN_NAME, RESOLVED_BY_INSPECTION,
                                       _osf_signatures, _our_signatures, match)
    m, audit = match()
    osf_sig, osf_organ = _osf_signatures()
    by_sig: dict = {}
    for name, sig in osf_sig.items():
        by_sig.setdefault(sig, []).append(name)
    ours = _our_signatures()
    rows = []
    for img, (osf_name, organ) in sorted(m.items()):
        name = Path(img).name
        cands = sorted(by_sig[ours[img]])
        how = ("inspection" if name in RESOLVED_BY_INSPECTION else
               "question set, unique" if len(cands) == 1 else
               "question set, same-organ candidates")
        rows.append(dict(image=name, osf_image_name=osf_name,
                         organ=ORGAN_NAME[organ], matched_by=how,
                         n_questions=len(ours[img]), n_candidates=len(cands),
                         candidates="|".join(f"{c}:{osf_organ[c]}" for c in cands)))
    df = pd.DataFrame(rows)
    assert len(df) == 294 and audit["missing"] == 0
    # Single matches must be distinct OSF images; a same-organ pair records
    # both candidates (the organ is determined, the pairing is not), and
    # `match` has already checked the classes are one-to-one.
    single = df[~df.osf_image_name.str.contains("|", regex=False)]
    assert single.osf_image_name.is_unique, "two images mapped to one OSF image"
    return df


#: Score caches shipped with the artifact: every post-hoc analysis in the paper
#: reads only s_base, the option mask, K, the label and the group fields.
SCORE_CACHES = {
    "omnimedvqa_3000_mcq": "cache/omnimedvqa_3000_mcq.npz",
    "omnimedvqa_3000_mcq_s1": "cache/omnimedvqa_3000_mcq_s1.npz",
    "omnimedvqa_3000_mcq_s2": "cache/omnimedvqa_3000_mcq_s2.npz",
    "omnimedvqa_3000_mcq_s3": "cache/omnimedvqa_3000_mcq_s3.npz",
    "vqarad_mcq": "cache/vqarad_mcq.npz",
    "vqarad_mcq_s1": "cache/vqarad_mcq_s1.npz",
    "omnimedvqa_3000_mcq_q3vl4b": "cache/omnimedvqa_3000_mcq_q3vl4b.npz",
    "omnimedvqa_3000_mcq_s1_q3vl4b": "cache/omnimedvqa_3000_mcq_s1_q3vl4b.npz",
    "omnimedvqa_3000_mcq_s2_q3vl4b": "cache/omnimedvqa_3000_mcq_s2_q3vl4b.npz",
    "omnimedvqa_3000_mcq_s3_q3vl4b": "cache/omnimedvqa_3000_mcq_s3_q3vl4b.npz",
    "vqarad_mcq_q3vl4b": "cache/vqarad_mcq_q3vl4b.npz",
    "vqarad_mcq_s1_q3vl4b": "cache/vqarad_mcq_s1_q3vl4b.npz",
}


def score_caches(out_dir: Path) -> list[str]:
    """Write score-only copies of the caches the post-hoc analyses use.

    `z_x` and `e_y` (the image state and option embeddings) are replaced by
    one-column zero placeholders: they are model features of the images, not
    needed by any post-hoc score, and the loaders still find the keys they
    expect. `modality` is written as stored in the cache; the VQA-RAD organ is
    substituted at load time by `cpaudit.bench.common.load_cache` as usual.
    """
    from cpaudit.data.cache import FeatureCache, load_cache as raw, save_cache
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, path in SCORE_CACHES.items():
        p = P.ROOT / path
        if not p.exists():
            continue
        c = raw(p)
        slim = FeatureCache(ids=c.ids, K=c.K, mask=c.mask, s_base=c.s_base,
                            z_x=np.zeros((c.n, 1), np.float32),
                            e_y=np.zeros((c.n, c.s_base.shape[1], 1), np.float32),
                            answer_idx=c.answer_idx, modality=c.modality,
                            question_type=c.question_type)
        save_cache(slim, out_dir / f"{name}.npz")
        written.append(name)
    return written


def grids() -> dict:
    return {m: posthoc_grid(m) for m in POSTHOC} | {
        "order": "temperature ascending, then gamma, then k_reg; ties in the "
                 "selection objective go to the first entry",
        "raps_extra": RAPS_EXTRA}


def extraction() -> dict:
    from cpaudit.extract import extract_mcq as E
    from cpaudit.extract.extract_qwen3b import _load_model
    import inspect
    sig = inspect.signature(_load_model)
    return {
        "model": E.MODEL_NAME,
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3 (the only snapshot "
                    "in the local HF cache used for extraction)",
        "dtype": "float16",
        "processor": {"min_pixels": 256 * 256,
                      "max_pixels": sig.parameters["max_pixels"].default},
        "prompt": "chat template with one user turn: [image, text]; text = "
                  "question + '\\n' + lines 'A. <option>' ... + "
                  "'\\nAnswer with the option's letter from the given choices "
                  "directly.'; add_generation_prompt=True",
        "score": "s_base(x, y_k) = -log softmax(logits at the first generated "
                 "position, full vocabulary)[first token of bare letter k]",
        "rotations": "shift s shows option (k + s) mod K behind letter k; caches "
                     "stay in original option indexing",
        "environment_now": "transformers 5.14.1, torch 2.11.0+cu128; the "
                           "versions at extraction time (2026-08-03, rotations "
                           "2026-09-11) were not logged",
        "vqarad_swap": "VQA-RAD re-scored with options in the order (no, yes) "
                       "(--shift 1, 2026-10-18, transformers 5.14.1)",
        "second_backbone": {
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "dtype": "bfloat16 (its native precision)",
            "image_input": "raw PIL image handed to the processor, which "
                           "resizes it itself (qwen_vl_utils is Qwen2.5-specific)",
            "prompt_and_score": "identical to the first backbone",
            "orders": "OmniMedVQA shifts 0-3 and VQA-RAD (yes, no) and (no, yes), "
                      "as for the first backbone",
            "extracted": "2026-10-18, transformers 5.14.1, torch 2.11.0+cu128",
        },
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    it, org = item_table(), organ_table()
    it.to_csv(OUT / "item_ids.csv", index=False)
    org.to_csv(OUT / "vqarad_organ_map.csv", index=False)
    om, vq = (load_cache(P.ROOT / CACHES[c]) for c in ("OmniMedVQA", "VQA-RAD"))
    split_table(om, False).to_csv(OUT / "splits_omnimedvqa.csv", index=False)
    split_table(vq, True).to_csv(OUT / "splits_vqarad_image_disjoint.csv", index=False)
    split_table(vq, False).to_csv(OUT / "splits_vqarad_item_level.csv", index=False)
    one_per_image_table(vq).to_csv(OUT / "splits_vqarad_one_per_image.csv", index=False)
    config_table().to_csv(OUT / "selected_configs.csv", index=False)
    (OUT / "grids.json").write_text(json.dumps(grids(), indent=1))
    (OUT / "extraction.json").write_text(json.dumps(extraction(), indent=1))
    print("score caches:", score_caches(OUT / "scores"))
    print(it.groupby("corpus").size().to_string())
    print(org.matched_by.value_counts().to_string())
    print("written:", sorted(p.name for p in OUT.iterdir()))
