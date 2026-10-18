"""Attach VQA-RAD's released `image_organ` to our yes/no items.

Why this exists. The HuggingFace copy the loader reads
(`flaviagiammarino/vqa-rad`) carries image, question and answer only. The
original release on OSF (Lau et al. 2018; `VQA_RAD Dataset Public.json`) also
records, per image, the organ the radiologist filed it under: HEAD, CHEST or
ABD. An earlier version of this repo had no access to that field and grouped
questions by keywords in their text instead (`vqarad._organ_guess`). That
grouping was per question, so 146 of the 294 images had questions in more than
one group, and 70% of questions fell in a catch-all `other` that is not a
category of the release at all. It is retired; this module replaces it.

Matching. The HF images carry no file name, so each of our images is matched
to an OSF `image_name` by the multiset of (question, answer) pairs asked about
it, after the same normalisation on both sides (lowercase, trimmed, internal
whitespace collapsed; answers restricted to yes/no; duplicate
(image, question, answer) triples dropped). Question sets must match exactly.
290 of our 294 images match exactly one OSF image; four share a one-question
set with a second OSF image. For two of those both candidates carry the same
organ, so the organ is determined anyway; the other two are resolved by
inspection (`RESOLVED_BY_INSPECTION`). `attach` fails loudly on anything else:
a silent near-match would put the wrong organ on an image and every grouped
result downstream would inherit it. The result reproduces the release's own
counts for this subset: 1,191 questions on 294 images, 308 head, 475 chest and
408 abdomen.

Run: python -m cpaudit.data.vqarad_organ   (prints the match audit)
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from cpaudit import paths as P

OSF_JSON = P.ROOT / "data" / "vqarad" / "VQA_RAD_Dataset_Public.json"
#: sha256 of the file the audit below was run against (1,419,295 bytes,
#: fetched from https://osf.io/download/6qdas/ on 2026-09-23).
OSF_SHA256 = "948b7156059b864cf72344fb10669233526b175c243e5a046d20d4201d297a95"
ITEMS = P.ROOT / "data" / "vqarad" / "yesno_items.jsonl"
#: id -> organ as exported by `cpaudit.bench.export_manifest` (see `organ_by_id`).
MANIFEST = P.ROOT / "paper" / "materials" / "item_ids.csv"
#: The release's three organ codes, renamed for printing.
ORGAN_NAME = {"HEAD": "head", "CHEST": "chest", "ABD": "abdomen"}

#: Two of our images each carry the single question "is the image normal?"
#: answered "no", and so do two OSF images of DIFFERENT organs (synpic17693,
#: CHEST; synpic54802, HEAD). Question sets cannot tell them apart, so they
#: were resolved by looking at the images: 21be... is a frontal chest
#: radiograph, 37bb... an axial brain MRI. Every other image is matched by its
#: question set alone.
RESOLVED_BY_INSPECTION = {
    "21be90484c82c193.png": "synpic17693.jpg",
    "37bb2730a5feb443.png": "synpic54802.jpg",
}


def _norm(s: str) -> str:
    """Lowercase, trim, collapse whitespace, and drop whitespace before
    punctuation (the release has 'well-defined ?' where HF has 'well-defined?')."""
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return re.sub(r"\s+([?.,!])", r"\1", s)


def _osf_signatures(path=OSF_JSON) -> tuple[dict, dict]:
    """image_name -> (sorted (q, a) tuple, organ), yes/no questions only."""
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    qa: dict[str, set] = defaultdict(set)
    organ: dict[str, set] = defaultdict(set)
    for r in recs:
        a = _norm(r["answer"])
        if a not in ("yes", "no"):
            continue
        qa[r["image_name"]].add((_norm(r["question"]), a))
        organ[r["image_name"]].add(str(r["image_organ"]).strip().upper())
    bad = {k: v for k, v in organ.items() if len(v) != 1}
    if bad:
        raise ValueError(f"OSF images with more than one organ: {bad}")
    return ({k: tuple(sorted(v)) for k, v in qa.items()},
            {k: next(iter(v)) for k, v in organ.items()})


def _our_signatures(items=ITEMS) -> dict:
    """image_path -> sorted (q, a) tuple over our yes/no items. `items` is the
    items file or, during a rebuild, the in-memory list about to be written."""
    if not isinstance(items, list):
        items = [json.loads(l) for l in Path(items).read_text(
            encoding="utf-8").splitlines() if l.strip()]
    qa: dict[str, set] = defaultdict(set)
    for it in items:
        a = it["options"][it["answer_idx"]]
        qa[it["image_path"]].add((_norm(it["question"]), a))
    return {k: tuple(sorted(v)) for k, v in qa.items()}


def match(osf=OSF_JSON, items=ITEMS) -> tuple[dict, dict]:
    """Our image_path -> (OSF image_name, organ), plus an audit dict."""
    osf_sig, osf_organ = _osf_signatures(osf)
    ours = _our_signatures(items)
    by_sig: dict[tuple, list[str]] = defaultdict(list)
    for name, sig in osf_sig.items():
        by_sig[sig].append(name)

    out, ambiguous, missing = {}, {}, []
    for img, sig in ours.items():
        cands = by_sig.get(sig, [])
        if not cands:
            missing.append(img)
            continue
        organs = {osf_organ[c] for c in cands}
        if len(cands) > 1:
            ambiguous[img] = (cands, sorted(organs))
        forced = RESOLVED_BY_INSPECTION.get(Path(img).name)
        if forced is not None:
            if forced not in cands:
                raise ValueError(f"{img}: inspected match {forced} is not among "
                                 f"its question-set candidates {cands}")
            out[img] = (forced, osf_organ[forced])
        # An ambiguous signature is still usable when every candidate carries
        # the same organ: the organ, which is all we need, is determined. The
        # image-level pairing is not, so all candidates are recorded rather
        # than one being picked arbitrarily.
        elif len(organs) == 1:
            out[img] = ("|".join(sorted(cands)), organs.pop())
    # One-to-one at the level of question-set classes: each signature must be
    # carried by as many of our images as OSF images, or some image has no
    # counterpart (or two images claim one).
    our_by_sig: dict[tuple, int] = defaultdict(int)
    for sig in ours.values():
        our_by_sig[sig] += 1
    unbalanced = {s: (n, len(by_sig.get(s, []))) for s, n in our_by_sig.items()
                  if n != len(by_sig.get(s, []))}
    if unbalanced:
        raise ValueError(f"question-set classes not one-to-one: {unbalanced}")
    forced_names = list(RESOLVED_BY_INSPECTION.values())
    if len(set(forced_names)) != len(forced_names):
        raise ValueError("two images resolved by inspection to one OSF image")
    audit = dict(n_ours=len(ours), n_osf_yesno_images=len(osf_sig),
                 matched=len(out), missing=len(missing),
                 ambiguous=len(ambiguous),
                 ambiguous_organ_conflict=sum(len(o) > 1
                                              for _, o in ambiguous.values()),
                 resolved_by_inspection=len(RESOLVED_BY_INSPECTION),
                 missing_list=missing, ambiguous_detail=ambiguous)
    return out, audit


@functools.lru_cache(maxsize=1)
def organ_by_id() -> dict[str, str]:
    """Item id -> released organ ('head' / 'chest' / 'abdomen').

    Checks the OSF file's hash first: the match was audited against one
    specific file, and a different release could re-label images silently.

    Without the OSF file, the mapping is read from the manifest that
    `cpaudit.bench.export_manifest` wrote from this function's own output
    (`paper/materials/item_ids.csv`), so the released artifact can re-run
    the post-hoc analyses from its score caches without the raw data.
    """
    if not OSF_JSON.exists() and MANIFEST.exists():
        import csv
        with MANIFEST.open(encoding="utf-8", newline="") as f:
            org = {r["id"]: r["stratum"] for r in csv.DictReader(f)
                   if r["corpus"] == "VQA-RAD"}
        assert len(org) == 1191 and set(org.values()) == {"head", "chest", "abdomen"}
        return org
    if not OSF_JSON.exists():
        raise FileNotFoundError(
            f"{OSF_JSON} is missing. VQA-RAD's organ labels come from the "
            "original release (https://osf.io/download/6qdas/); the HF copy "
            "has none.")
    digest = hashlib.sha256(OSF_JSON.read_bytes()).hexdigest()
    if digest != OSF_SHA256:
        raise ValueError(f"{OSF_JSON.name} has sha256 {digest}, not the audited "
                         f"{OSF_SHA256}; re-run the match audit before using it")
    items = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    return {it["id"]: it["anatomy"] for it in attach(items)}


def attach(items: list[dict], osf=OSF_JSON) -> list[dict]:
    """Return copies of `items` with `anatomy` (and `modality`, which the
    cache and the split read) set to the released organ. Fails on any
    unmatched image rather than guessing."""
    m, audit = match(osf, items)
    unmatched = {it["image_path"] for it in items} - set(m)
    if unmatched:
        raise ValueError(f"{len(unmatched)} images have no unique organ match: "
                         f"{sorted(unmatched)[:5]} ... audit={ {k: v for k, v in audit.items() if not k.endswith(('list', 'detail'))} }")
    out = []
    for it in items:
        o = ORGAN_NAME[m[it["image_path"]][1]]
        out.append({**it, "anatomy": o, "modality": o,
                    "osf_image_name": m[it["image_path"]][0]})
    return out


if __name__ == "__main__":
    m, audit = match()
    print({k: v for k, v in audit.items() if not k.endswith(("list", "detail"))})
    for img, (cands, organs) in list(audit["ambiguous_detail"].items())[:10]:
        print("  ambiguous:", Path(img).name, cands, organs)
    items = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if not audit["missing"]:
        new = attach(items)
        print("questions per organ:", dict(Counter(i["anatomy"] for i in new)))
        print("images per organ:", dict(Counter(
            m[p][1] for p in {i["image_path"] for i in items})))
        old = Counter((i["anatomy"], j["anatomy"]) for i, j in zip(items, new))
        print("keyword guess -> released organ:", dict(old))
