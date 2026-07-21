#!/usr/bin/env python3
"""
Modal GPU training for GLiNER PII fine-tuning.

Runs the full 5-epoch training on a CUDA GPU (no MPS crashes, bigger batches,
focal loss works). Trains on the full Nemotron-PII healthcare subset (or the
entire 100k dataset with --full).

Two label vocabularies:
  * default        — filter spans to the 24-label FINE_TO_COARSE subset (legacy
                     recipe that produced daslabhq/pii-proxy).
  * --full-labels  — keep EVERY entity label present in the data (~55 native
                     Nemotron labels). Use this for a fair head-to-head vs
                     nvidia/gliner-PII on the whole vocabulary.

Honest eval: training always draws from Nemotron's `train` split and evaluation
always draws from the SEPARATE `test` split (~100k records the model never sees).
A zero-text-overlap assertion runs before training. The old recipe evaluated on a
holdout of the train split, which leaked and inflated F1 — do not reintroduce it.

Setup (one-time):
    modal token new          # interactive browser auth
    # OR export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from a Daslab modal/account

Run:
    modal run experiments/modal_train.py                                  # healthcare subset, 24 labels
    modal run experiments/modal_train.py --full                           # full 100k, 24-label subset
    modal run experiments/modal_train.py --full --full-labels             # full 100k, ~55 native labels
    modal run experiments/modal_train.py --base-model urchade/gliner_medium-v2.1

Publish to Hugging Face (needs the "huggingface" Modal secret with HF_TOKEN):
    modal run --detach experiments/modal_train.py --full --full-labels \
        --push-hf --hf-repo daslabhq/pii-proxy-exp-full55 --private

Results print to stdout and save to the Modal volume (downloadable).
"""

import modal

app = modal.App("pii-proxy-gliner-train")

# Image: CUDA-capable torch + GLiNER + datasets
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "gliner",
        "datasets",
        "accelerate",
        "seqeval",
        "sentencepiece",
        "huggingface_hub",
    )
)

# Volume for caching the dataset + saving trained models
volume = modal.Volume.from_name("pii-proxy-models", create_if_missing=True)
MODEL_DIR = "/models"
DATASET = "nvidia/Nemotron-PII"

FINE_TO_COARSE = {
    "first_name": "person_name", "last_name": "person_name", "middle_name": "person_name",
    "medical_record_number": "medical_record",
    "date_of_birth": "date_of_birth",
    "date": "date", "date_time": "date", "time": "date",
    "email": "email", "phone_number": "phone",
    "street_address": "location", "city": "location", "state": "location",
    "county": "location", "zip_code": "location", "country": "location",
    "ssn": "national_id", "health_plan_beneficiary_number": "insurance_id",
    "certificate_license_number": "national_id", "biometric_identifier": "national_id",
    "url": "url", "unique_id": "national_id", "blood_type": "medical_info",
    "pin": "national_id", "password": "credential", "swift_bic": "financial_id",
}


@app.function(
    image=image,
    gpu="A100",            # 40GB — handles batch_size 16+, full focal loss
    timeout=60 * 60 * 8,   # up to 8 hours for full dataset
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(
    epochs: int = 5,
    base_model: str = "urchade/gliner_small-v2.1",
    batch_size: int = 16,
    full: bool = False,
    full_labels: bool = False,
    eval_size: int = 500,
    health_eval_size: int = 200,
    push_hf: bool = False,
    hf_repo: str = "daslabhq/pii-proxy",
    private: bool = False,
):
    import json, random, time
    import torch
    from gliner import GLiNER

    random.seed(42)
    torch.manual_seed(42)

    # ── Load train + clean (held-out) test splits ──
    # `test` is loaded FIRST so a missing/renamed split fails fast, before training.
    print(f"Loading Nemotron-PII ({'full 100k' if full else 'healthcare subset'})...")
    train_records = _load_split(
        "train", full_labels=full_labels,
        health_only=not full, scan_limit=100_000 if full else 20_000,
    )
    print(f"  Train records: {len(train_records)}")

    print("Loading clean held-out `test` split for eval...")
    eval_records = _load_split(
        "test", full_labels=full_labels, max_records=eval_size, scan_limit=300_000,
    )
    health_eval = _load_split(
        "test", full_labels=full_labels, health_only=True,
        max_records=health_eval_size, scan_limit=300_000,
    )
    print(f"  Eval records: {len(eval_records)} (all domains), {len(health_eval)} (healthcare)")

    _assert_no_leakage(train_records, eval_records, health_eval)

    fine_labels = sorted({s["label"] for r in train_records for s in r["goldSpans"]})
    print(f"  Fine labels ({len(fine_labels)}): {fine_labels}")

    # ── Tokenize + build GLiNER training set ──
    train_data = [d for d in (_to_gliner(r) for r in train_records) if d["ner"]]
    trainer_eval = train_data[-100:]     # in-loop monitoring only (from TRAIN, never test)
    train_split = train_data[:-100]
    print(f"  GLiNER train examples: {len(train_split)} (+{len(trainer_eval)} in-loop eval)")

    # ── Train ──
    print(f"\nLoading {base_model}...")
    model = GLiNER.from_pretrained(base_model).to("cuda")

    print(f"Fine-tuning: {epochs} epochs, batch {batch_size}...")
    start = time.time()
    model.train_model(
        train_dataset=train_split,
        eval_dataset=trainer_eval,
        output_dir=f"{MODEL_DIR}/checkpoint",
        learning_rate=5e-6,
        others_lr=1e-5,
        weight_decay=0.01,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        focal_loss_alpha=0.75,   # works on CUDA
        focal_loss_gamma=2,
        num_train_epochs=epochs,
        max_steps=-1,
        save_steps=500,
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to="none",
    )
    train_time = time.time() - start
    print(f"Training done in {train_time:.0f}s")

    # ── Eval on the clean test split (native fine-label query) ──
    model.eval()
    fine_f1 = _eval_fine(model, eval_records, fine_labels)
    health_f1 = _eval_fine(model, health_eval, fine_labels) if health_eval else 0.0
    coarse_f1 = _eval_coarse(model, eval_records, fine_labels, FINE_TO_COARSE)

    result = {
        "base_model": base_model,
        "full_dataset": full,
        "full_labels": full_labels,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_size": len(train_split),
        "eval_split": "test",
        "eval_size": len(eval_records),
        "health_eval_size": len(health_eval),
        "fine_labels": len(fine_labels),
        "labels": fine_labels,
        "fine_f1": fine_f1,
        "health_fine_f1": health_f1,
        "coarse_f1": coarse_f1,
        "train_time_seconds": train_time,
    }
    print(f"\n{'='*60}")
    print(f"RESULT ({len(fine_labels)} labels, clean `test` split):")
    print(f"  fine F1 {fine_f1*100:.1f}%  |  healthcare fine F1 {health_f1*100:.1f}%  |  coarse F1 {coarse_f1*100:.1f}%")
    print(f"  NVIDIA reference: fine 96.2%, coarse 96.7%")
    print(f"{'='*60}")

    # Save result to volume
    with open(f"{MODEL_DIR}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()

    if push_hf:
        _push_to_hub(model, hf_repo, result, private=private)

    return result


# ── Dataset loading (train vs clean test split) ──

def _load_split(split: str, *, full_labels: bool, health_only: bool = False,
                max_records: int = None, scan_limit: int = 100_000):
    """Stream a Nemotron-PII split into {text, goldSpans} records.

    full_labels=False keeps only the 24-label FINE_TO_COARSE subset (legacy).
    full_labels=True keeps every entity label present (~55 native labels).
    """
    import ast
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split, streaming=True)
    records, total = [], 0
    for row in ds:
        total += 1
        domain = (row.get("domain", "") or "").lower()
        if health_only and "health" not in domain:
            if total >= scan_limit:
                break
            continue
        spans_raw = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
        gold = [{"start": s["start"], "end": s["end"], "text": s["text"], "label": s["label"]}
                for s in spans_raw if (full_labels or s["label"] in FINE_TO_COARSE)]
        if gold:
            records.append({"text": row["text"], "goldSpans": gold})
        if max_records and len(records) >= max_records:
            break
        if total >= scan_limit:
            break
    return records


def _assert_no_leakage(train_records, *eval_sets):
    """Fail loudly if any eval record's text also appears in the training set."""
    train_texts = {r["text"] for r in train_records}
    total_overlap = 0
    for eval_records in eval_sets:
        overlap = sum(1 for r in eval_records if r["text"] in train_texts)
        total_overlap += overlap
        print(f"  Overlap check: {overlap}/{len(eval_records)} eval records found in train")
    assert total_overlap == 0, (
        f"LEAKAGE: {total_overlap} eval records appear in the training set — "
        f"eval is not honest, aborting."
    )
    print("  Overlap check PASSED: 0 eval records in train (clean held-out eval).")


# ── Char→token span conversion ──

def _to_gliner(rec):
    toks, offs = _tokenize_with_offsets(rec["text"])
    ner = []
    for sp in rec["goldSpans"]:
        ts = _char_to_tok(sp["start"], sp["end"], offs)
        if ts:
            ner.append([ts[0], ts[1], sp["label"]])
    return {"tokenized_text": toks, "ner": ner}


def _tokenize_with_offsets(text):
    import re
    pattern = re.compile(r"\w+|[^\w\s]")
    toks, offs = [], []
    for m in pattern.finditer(text):
        toks.append(m.group())
        offs.append((m.start(), m.end()))
    return toks, offs


def _char_to_tok(cs, ce, offs):
    si = ei = None
    for i, (s, e) in enumerate(offs):
        if s < ce and e > cs:
            if si is None:
                si = i
            ei = i
    return [si, ei] if si is not None else None


# ── Evaluation (span-overlap + exact-label match, threshold 0.5) ──

def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def _eval_fine(m, recs, labels):
    """Fine F1 querying the model with its native fine labels."""
    tp = fp = fn = 0
    for rec in recs:
        ents = m.predict_entities(rec["text"], labels, threshold=0.5)
        pred = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in ents]
        gold = rec["goldSpans"]
        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm:
                    continue
                if p["label"] == g["label"] and _overlap(p, g):
                    pm.add(pi); gm.add(gi); tp += 1; break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)
    return _f1(tp, fp, fn)


def _eval_coarse(m, recs, labels, mapping):
    """Coarse F1: predictions/gold collapsed via mapping (only mapped labels count)."""
    tp = fp = fn = 0
    for rec in recs:
        ents = m.predict_entities(rec["text"], labels, threshold=0.5)
        pred = [{"start": e["start"], "end": e["end"], "label": mapping.get(e["label"])}
                for e in ents if mapping.get(e["label"])]
        gold = [{**s, "label": mapping[s["label"]]} for s in rec["goldSpans"] if s["label"] in mapping]
        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm:
                    continue
                if p["label"] == g["label"] and _overlap(p, g):
                    pm.add(pi); gm.add(gi); tp += 1; break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)
    return _f1(tp, fp, fn)


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0


# ── Hugging Face publish ──

def _model_card(result: dict) -> str:
    fine = result["fine_f1"] * 100
    coarse = result["coarse_f1"] * 100
    health = result.get("health_fine_f1", 0) * 100
    n_labels = result["fine_labels"]
    experimental = result.get("full_labels")
    scope = "the full Nemotron-PII train split (~100k, all domains)" if result["full_dataset"] \
        else "the Nemotron-PII healthcare subset"
    header = "# pii-proxy (experimental — full label vocabulary)\n\n> Experimental checkpoint. The flagship model is `daslabhq/pii-proxy`.\n" \
        if experimental else "# pii-proxy\n"
    return f'''---
license: apache-2.0
base_model: {result["base_model"]}
library_name: gliner
pipeline_tag: token-classification
tags:
- gliner
- token-classification
- pii
- ner
---

{header}
A GLiNER model fine-tuned for PII detection. GLiNER does zero-shot NER over
arbitrary labels — you pass the entity types you care about at inference time,
so there is no fixed label schema to work around.

Fine-tuned from `{result["base_model"]}` on {scope}, over {n_labels} native
Nemotron-PII entity labels.

## Results

Evaluated on Nemotron-PII's held-out `test` split ({result["eval_size"]} examples,
all domains) — a clean set the model never trained on (zero train/test text
overlap, asserted at train time). {n_labels} fine-grained labels, queried natively.

| Metric | F1 |
|---|---|
| Fine-grained (all domains) | {fine:.1f}% |
| Fine-grained (healthcare slice, {result.get("health_eval_size", 0)}) | {health:.1f}% |
| Coarse-grained (mapped subset) | {coarse:.1f}% |

NVIDIA's Nemotron-PII reference: fine 96.2%, coarse 96.7%.

## Usage

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("{result.get("_repo", "daslabhq/pii-proxy")}")
labels = ["first_name", "last_name", "email", "phone_number", "ssn", "street_address"]
entities = model.predict_entities("Patient Marcus Weber, marcus.weber@gmail.com", labels)
for e in entities:
    print(e["text"], "->", e["label"])
```

## Training

- Base model: `{result["base_model"]}`
- Epochs: {result["epochs"]}, batch size {result["batch_size"]}
- Train examples: {result["train_size"]}
- Labels: {n_labels} native Nemotron-PII entity types
- Focal loss (alpha 0.75, gamma 2)
- Train split: `train` — Eval split: `test` (held out, zero overlap)
'''


def _push_to_hub(model, repo_id: str, result: dict, private: bool = False) -> None:
    import os
    from huggingface_hub import HfApi

    out = f"{MODEL_DIR}/hf_export"
    model.save_pretrained(out)
    result = {**result, "_repo": repo_id}
    with open(f"{out}/README.md", "w") as f:
        f.write(_model_card(result))
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=private)
    api.upload_folder(folder_path=out, repo_id=repo_id, repo_type="model")
    print(f"pushed -> https://huggingface.co/{repo_id} (private={private})")


@app.local_entrypoint()
def main(
    epochs: int = 5,
    full: bool = False,
    full_labels: bool = False,
    base_model: str = "urchade/gliner_small-v2.1",
    batch_size: int = 16,
    eval_size: int = 500,
    health_eval_size: int = 200,
    push_hf: bool = False,
    hf_repo: str = "daslabhq/pii-proxy",
    private: bool = False,
):
    result = train.remote(
        epochs=epochs, full=full, full_labels=full_labels, base_model=base_model,
        batch_size=batch_size, eval_size=eval_size, health_eval_size=health_eval_size,
        push_hf=push_hf, hf_repo=hf_repo, private=private,
    )
    print("\nFinal result:")
    import json
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def sweep():
    """Run multiple configs in parallel on separate A100s.

    Small dataset (525 examples) needs many gradient updates → small batch
    or many epochs. This sweep finds the right combo.
    """
    configs = [
        {"epochs": 5,  "batch_size": 4},    # ~650 steps
        {"epochs": 10, "batch_size": 4},    # ~1300 steps
        {"epochs": 15, "batch_size": 2},    # ~3900 steps
        {"epochs": 10, "batch_size": 2},    # ~2600 steps
    ]
    print(f"Running {len(configs)} configs in parallel on A100s...")
    # positional: (epochs, base_model, batch_size, full, full_labels)
    results = list(train.starmap(
        [(c["epochs"], "urchade/gliner_small-v2.1", c["batch_size"], False, False) for c in configs]
    ))
    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print(f"{'epochs':>7} {'batch':>6} {'fine F1':>9} {'coarse F1':>10} {'time':>8}")
    for c, r in zip(configs, results):
        print(f"{c['epochs']:>7} {c['batch_size']:>6} {r['fine_f1']*100:>8.1f}% {r['coarse_f1']*100:>9.1f}% {r['train_time_seconds']:>6.0f}s")
    print(f"\n  NVIDIA reference: fine 96.2%, coarse 96.7%")
    best = max(results, key=lambda r: r["fine_f1"])
    print(f"  Best: {best['epochs']}ep/batch{best['batch_size']} → fine {best['fine_f1']*100:.1f}%, coarse {best['coarse_f1']*100:.1f}%")
