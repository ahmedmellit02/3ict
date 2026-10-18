"""Letter-option (MCQ-prompt) base score, the protocol of medical-VQA
leaderboards (Med-R1, Lai et al. 2025): every option is listed in the prompt as
a line "A. ...", and the score is read at the first generated position.

    s_base(x, y_k) = -log p(letter_k | image, question, all options listed)

Only s_base is computed; z_x, e_y, the mask and the labels are reused from a
base cache built by `extract_qwen3b`. One forward pass per item. `--shift s`
shows option (k + s) mod K behind letter k (caches stay in original option
indexing); `--model` selects the backbone.

Run:
  python -m cpaudit.extract.extract_mcq --items data/omnimedvqa/sample_items.jsonl \\
      --base-cache cache/omnimedvqa_full_3000.npz --out cache/omnimedvqa_3000_mcq.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
LETTERS = "ABCDE"


def _letter_token_ids(tokenizer):
    """Token id for each bare letter (the answer is the first generated token)."""
    ids = {}
    for L in LETTERS:
        toks = tokenizer.encode(L, add_special_tokens=False)
        ids[L] = toks[0]
    return ids


def _build_prompt(processor, image, question, options, order=None,
                  vision_utils=True):
    """Med-R1 style: list the options, ask for a single letter.

    `order[k]` is the index (in the item's own option list) of the option shown
    behind letter k. Defaults to identity, i.e. the order the dataset gives.
    `vision_utils=False` hands the raw image to the processor, which resizes it
    itself; `qwen_vl_utils` resizes for Qwen2.5-VL's 14-pixel patches and is
    not used for other models.
    """
    if order is None:
        order = range(len(options))
    lines = [f"{LETTERS[i]}. {options[j]}" for i, j in enumerate(order)]
    text = (f"{question}\n" + "\n".join(lines)
            + "\nAnswer with the option's letter from the given choices directly.")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    if not vision_utils:
        return prompt, [image], None
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except Exception:
        image_inputs, video_inputs = [image], None
    return prompt, image_inputs, video_inputs


def option_order(K: int, shift: int) -> list[int]:
    """Display position k shows the item's option `(k + shift) % K`.

    shift=0 reproduces the dataset order exactly. Because the cache keeps every
    array in the ORIGINAL option indexing, `answer_idx`, `e_y` and `mask` stay
    valid across shifts and two caches pair 1:1 on `ids` -- the only changed
    variable is which letter the model saw each option behind.
    """
    return [(k + shift) % K for k in range(K)]


def answer_letter_position(K, answer_idx, shift: int):
    """Which display position (0=A) the true answer sat at, given the shift."""
    return (np.asarray(answer_idx) - shift) % np.asarray(K)


def extract(items, base_cache_path, out_path, model_name=MODEL_NAME, limit=None,
            option_shift: int = 0):
    import torch
    from PIL import Image
    from cpaudit.data.cache import load_cache, FeatureCache, save_cache
    from cpaudit.extract.extract_qwen3b import _load_model

    base = load_cache(base_cache_path)
    id_to_row = {str(i): r for r, i in enumerate(base.ids.tolist())}
    items = [it for it in items if str(it["id"]) in id_to_row]
    if limit:
        items = items[:limit]
    print(f"[mcq] {len(items)} items aligned to base cache ({base.n} rows)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_model(model_name, device)
    letter_ids = _letter_token_ids(processor.tokenizer)
    print(f"[mcq] letter token ids: {letter_ids}")

    n, kmax = base.s_base.shape
    s_new = np.full((n, kmax), np.inf, dtype=np.float32)
    done = np.zeros(n, dtype=bool)

    t0 = time.time()
    for i, it in enumerate(items):
        row = id_to_row[str(it["id"])]
        try:
            img = Image.open(it["image_path"]).convert("RGB")
            order = option_order(len(it["options"]), option_shift)
            prompt, image_inputs, video_inputs = _build_prompt(
                processor, img, it["question"], it["options"], order,
                vision_utils="Qwen2.5-VL" in model_name)
            inputs = processor(text=[prompt], images=image_inputs,
                               videos=video_inputs, padding=True,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0, -1].float()            # next-token distribution
            logp = torch.log_softmax(logits, dim=-1)
            for k, j in enumerate(order):
                s_new[row, j] = float(-logp[letter_ids[LETTERS[k]]].item())
            done[row] = True
        except Exception as e:
            print(f"  [skip {i}] {e}")
            continue
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(items) - i - 1) / rate / 60
            acc = _running_top1(s_new, base, done)
            print(f"  [mcq] {i+1}/{len(items)}  {rate:.2f} it/s  ETA {eta:.1f} min "
                  f"| running top-1 = {acc:.3f}", flush=True)

    keep = np.where(done)[0]
    cache = FeatureCache(
        ids=base.ids[keep], K=base.K[keep], mask=base.mask[keep],
        s_base=s_new[keep], z_x=base.z_x[keep], e_y=base.e_y[keep],
        answer_idx=base.answer_idx[keep], modality=base.modality[keep],
        question_type=base.question_type[keep],
    )
    save_cache(cache, out_path)
    top1 = _top1(cache)
    print(f"[mcq] saved {cache.n} items -> {out_path}")
    print(f"[mcq] FINAL top-1 accuracy (MCQ letter protocol) = {top1:.4f}")
    print(f"[mcq]   (independent-likelihood protocol was 0.373; Med-R1 reports 0.5424)")
    return cache


def _running_top1(s_new, base, done):
    idx = np.where(done)[0]
    if len(idx) == 0:
        return float("nan")
    s = np.where(base.mask[idx], s_new[idx], np.inf)
    return float((s.argmin(1) == base.answer_idx[idx]).mean())


def _top1(cache):
    s = np.where(cache.mask, cache.s_base, np.inf)
    return float((s.argmin(1) == cache.answer_idx).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/omnimedvqa/sample_items.jsonl")
    ap.add_argument("--base-cache", default="cache/omnimedvqa_full_3000.npz")
    ap.add_argument("--out", default="cache/omnimedvqa_3000_mcq.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shift", type=int, default=0,
                    help="cyclic option-order shift; 0 = dataset order (default)")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="HF checkpoint; default Qwen2.5-VL-3B (the 3ICT second "
                         "backbone is Qwen/Qwen3-VL-4B-Instruct)")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    extract(items, args.base_cache, args.out, model_name=args.model,
            limit=args.limit, option_shift=args.shift)


if __name__ == "__main__":
    main()
