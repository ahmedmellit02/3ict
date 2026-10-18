"""Frozen-model loading, and the per-option feature pass that built the base caches.

`_load_model` loads either backbone (Qwen2.5-VL-3B in float16, Qwen3-VL-4B in
bfloat16); `extract_mcq` uses it for the letter scores the paper reports.

`extract` is the per-option pass that built the base caches. For each
(image x, option y_k) it records the image state and the option embedding that
the Section III controls read, and a length-normalized option log-likelihood
that the paper does not use:

  z_x           = final-layer hidden state at the answer position (shared)
  e_y_k         = mean-pooled frozen input embedding of the option string
  s_base(x,y_k) = -mean_t log p(option_token_t | image, question, prefix)

3B in fp16 is ~6-7 GB VRAM, so a 12 GB card needs no 4-bit quantization.

Usage (torch environment):
    python -m cpaudit.extract.extract_qwen3b --items items.jsonl --out cache/real.npz --limit 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def _load_model(model_name: str, device: str, max_pixels: int = 768 * 768):
    import torch
    if "Qwen2.5-VL" in model_name:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        except ImportError:                   # older/newer naming
            from transformers import AutoModelForImageTextToText as VLModel
        dtype = torch.float16
    else:
        # Any other image-text-to-text checkpoint (the 3ICT second backbone is
        # Qwen3-VL-4B). bfloat16 is its native precision; float16 risks
        # overflow in a model that was never trained in it.
        from transformers import AutoModelForImageTextToText as VLModel
        dtype = torch.bfloat16
    from transformers import AutoProcessor
    # Cap image resolution -> bounds vision-token count -> bounds VRAM on 12 GB.
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=256 * 256, max_pixels=max_pixels)
    # transformers>=5 renamed torch_dtype -> dtype; try new kwarg, fall back.
    try:
        model = VLModel.from_pretrained(model_name, dtype=dtype,
                                        device_map={"": 0} if device == "cuda" else None)
    except TypeError:
        model = VLModel.from_pretrained(model_name, torch_dtype=dtype,
                                        device_map={"": 0} if device == "cuda" else None)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def _prep_inputs(processor, image, text):
    """Build model inputs for one (image, text) using Qwen vision utils."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except Exception:
        image_inputs, video_inputs = [image], None
    return prompt, image_inputs, video_inputs


def _option_logprob(model, processor, torch, image, question, option, device):
    """Real length-normalized -log p(option | image, question) + z_x."""
    prefix_text, image_inputs, video_inputs = _prep_inputs(processor, image, question)
    full_text = prefix_text + option

    prefix = processor(text=[prefix_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    full = processor(text=[full_text], images=image_inputs, videos=video_inputs,
                     padding=True, return_tensors="pt").to(device)

    len_prefix = prefix["input_ids"].shape[1]
    len_full = full["input_ids"].shape[1]
    n_opt = max(1, len_full - len_prefix)

    with torch.no_grad():
        out = model(**full, output_hidden_states=True)
    logits = out.logits[0]                                   # (T, V)
    target = full["input_ids"][0, len_prefix:len_full]       # option tokens
    pred = logits[len_prefix - 1: len_full - 1]              # aligned predictions
    logp = torch.log_softmax(pred.float(), dim=-1)
    tok_logp = logp[torch.arange(target.shape[0]), target]
    s_base = float(-tok_logp.mean().item())                  # length-normalized
    z_x = out.hidden_states[-1][0, len_prefix - 1].float().cpu().numpy().astype(np.float32)
    return s_base, z_x


def _text_embedding(model, processor, torch, option, device):
    ids = processor.tokenizer(option, return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        emb = model.get_input_embeddings()(ids)[0]           # (n_tok, d_e)
    return emb.mean(0).float().cpu().numpy().astype(np.float32)


def _build_cache(items, keep, s_base, z_x, e_y, mask):
    from cpaudit.data.cache import FeatureCache
    keep = np.array(keep)
    return FeatureCache(
        ids=np.array([items[i]["id"] for i in keep]),
        K=np.array([items[i]["K"] for i in keep]),
        mask=mask[keep], s_base=s_base[keep], z_x=z_x[keep], e_y=e_y[keep],
        answer_idx=np.array([items[i]["answer_idx"] for i in keep]),
        modality=np.array([items[i]["modality"] for i in keep]),
        question_type=np.array([items[i]["question_type"] for i in keep]),
    )


def extract(items: list[dict], out_path: str | Path, model_name: str = MODEL_NAME,
            limit: int | None = None, checkpoint_every: int = 100,
            resume: bool = True):
    """Extract features for `items`, saving progressively so a crash mid-run is
    recoverable. If `resume` and a cache already exists at out_path, items whose
    id is already cached are skipped.
    """
    import torch
    from PIL import Image
    from cpaudit.data.cache import save_cache, load_cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(out_path)
    if limit:
        items = items[:limit]

    done_ids: set = set()
    if resume and out_path.exists():
        try:
            prev = load_cache(out_path)
            done_ids = set(prev.ids.tolist())
            print(f"[resume] found existing cache with {len(done_ids)} items")
        except Exception:
            done_ids = set()
    todo = [it for it in items if it["id"] not in done_ids]
    if done_ids:
        print(f"[resume] {len(items) - len(todo)} of {len(items)} requested items "
              f"already cached; {len(todo)} to extract")
    if not todo:
        print("[resume] nothing to do; cache already complete")
        return load_cache(out_path)

    model, processor = _load_model(model_name, device)

    kmax = max(it["K"] for it in items)
    probe_img = Image.open(todo[0]["image_path"]).convert("RGB")
    _, z0 = _option_logprob(model, processor, torch, probe_img,
                            todo[0]["question"], todo[0]["options"][0], device)
    e0 = _text_embedding(model, processor, torch, todo[0]["options"][0], device)
    d, d_e = z0.shape[0], e0.shape[0]

    n = len(todo)
    s_base = np.full((n, kmax), np.inf, dtype=np.float32)
    z_x = np.zeros((n, d), dtype=np.float32)
    e_y = np.zeros((n, kmax, d_e), dtype=np.float32)
    mask = np.zeros((n, kmax), dtype=bool)
    keep = []

    def _flush():
        """Merge newly-extracted items with any prior cache and save."""
        cache = _build_cache(todo, keep, s_base, z_x, e_y, mask)
        if done_ids and out_path.exists():
            prev = load_cache(out_path)
            cache = _concat_caches(prev, cache)
        save_cache(cache, out_path)
        return cache

    import time
    t0 = time.time()
    for i, it in enumerate(todo):
        try:
            img = Image.open(it["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  [skip {i}] image load failed: {e}")
            continue
        zx = None
        try:
            for k, opt in enumerate(it["options"]):
                sb, zx = _option_logprob(model, processor, torch, img,
                                         it["question"], opt, device)
                s_base[i, k] = sb
                e_y[i, k] = _text_embedding(model, processor, torch, opt, device)
                mask[i, k] = True
        except Exception as e:                      # OOM or bad sample -> skip, keep going
            print(f"  [skip {i}] extraction failed: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        if zx is not None:
            z_x[i] = zx
            keep.append(i)
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            print(f"  extracted {i + 1}/{n}  ({rate:.2f} it/s, ETA {eta:.1f} min)")
        if (i + 1) % checkpoint_every == 0:
            _flush()
            print(f"  [checkpoint] saved {len(keep)} new items -> {out_path.name}")

    cache = _flush()
    print(f"Saved cache -> {out_path}  (n={cache.n}, d={d}, d_e={d_e}, kmax={cache.kmax})")
    return cache


def _concat_caches(a, b):
    """Concatenate two FeatureCaches, padding to the larger kmax."""
    from cpaudit.data.cache import FeatureCache
    kmax = max(a.kmax, b.kmax)

    def _pad(c):
        if c.kmax == kmax:
            return c.mask, c.s_base, c.e_y
        pad = kmax - c.kmax
        m = np.pad(c.mask, ((0, 0), (0, pad)))
        s = np.pad(c.s_base, ((0, 0), (0, pad)), constant_values=np.inf)
        e = np.pad(c.e_y, ((0, 0), (0, pad), (0, 0)))
        return m, s, e

    ma, sa, ea = _pad(a)
    mb, sb, eb = _pad(b)
    return FeatureCache(
        ids=np.concatenate([a.ids, b.ids]), K=np.concatenate([a.K, b.K]),
        mask=np.concatenate([ma, mb]), s_base=np.concatenate([sa, sb]),
        z_x=np.concatenate([a.z_x, b.z_x]), e_y=np.concatenate([ea, eb]),
        answer_idx=np.concatenate([a.answer_idx, b.answer_idx]),
        modality=np.concatenate([a.modality, b.modality]),
        question_type=np.concatenate([a.question_type, b.question_type]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="JSONL of normalized items")
    ap.add_argument("--out", default="cache/omnimedvqa_qwen3b.npz")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    extract(items, args.out, args.model, args.limit)


if __name__ == "__main__":
    main()
