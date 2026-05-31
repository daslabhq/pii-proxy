#!/usr/bin/env python3
"""
Modal GPU training for GLiNER PII fine-tuning.

Runs the full 5-epoch training on a CUDA GPU (no MPS crashes, bigger batches,
focal loss works). Trains on the full Nemotron-PII healthcare subset (or the
entire 100k dataset with --full).

Setup (one-time):
    modal token new          # interactive browser auth
    # OR export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from a Daslab modal/account

Run:
    modal run experiments/modal_train.py                    # healthcare subset, fine labels
    modal run experiments/modal_train.py --full             # full 100k Nemotron
    modal run experiments/modal_train.py --base-model urchade/gliner_medium-v2.1

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
    )
)

# Volume for caching the dataset + saving trained models
volume = modal.Volume.from_name("pii-proxy-models", create_if_missing=True)
MODEL_DIR = "/models"

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
)
def train(
    epochs: int = 5,
    base_model: str = "urchade/gliner_small-v2.1",
    batch_size: int = 16,
    full: bool = False,
    test_size: int = 100,
):
    import ast, json, os, random, re, time
    import torch
    from datasets import load_dataset
    from gliner import GLiNER

    random.seed(42)
    torch.manual_seed(42)

    # ── Load Nemotron-PII ──
    print(f"Loading Nemotron-PII ({'full 100k' if full else 'healthcare subset'})...")
    ds = load_dataset("nvidia/Nemotron-PII", split="train", streaming=True)
    records = []
    total = 0
    scan_limit = 100_000 if full else 20_000
    for row in ds:
        total += 1
        domain = (row.get("domain", "") or "").lower()
        if not full and "health" not in domain:
            if total >= scan_limit:
                break
            continue
        spans_raw = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
        fine_spans = [{"start": s["start"], "end": s["end"], "text": s["text"], "label": s["label"]}
                      for s in spans_raw if s["label"] in FINE_TO_COARSE]
        if fine_spans:
            records.append({"text": row["text"], "goldSpans": fine_spans})
        if total >= scan_limit:
            break
    print(f"  Loaded {len(records)} records from {total} scanned")

    fine_labels = sorted({s["label"] for r in records for s in r["goldSpans"]})
    print(f"  Fine labels: {len(fine_labels)}")

    random.shuffle(records)
    test_records = records[:test_size]
    train_records = records[test_size:]
    print(f"  Train: {len(train_records)}, Test: {len(test_records)}")

    # ── Tokenize + char→token spans ──
    def tokenize_with_offsets(text):
        pattern = re.compile(r"\w+|[^\w\s]")
        toks, offs = [], []
        for m in pattern.finditer(text):
            toks.append(m.group()); offs.append((m.start(), m.end()))
        return toks, offs

    def char_to_tok(cs, ce, offs):
        si = ei = None
        for i, (s, e) in enumerate(offs):
            if s < ce and e > cs:
                if si is None:
                    si = i
                ei = i
        return [si, ei] if si is not None else None

    def to_gliner(rec):
        toks, offs = tokenize_with_offsets(rec["text"])
        ner = []
        for sp in rec["goldSpans"]:
            ts = char_to_tok(sp["start"], sp["end"], offs)
            if ts:
                ner.append([ts[0], ts[1], sp["label"]])
        return {"tokenized_text": toks, "ner": ner}

    train_data = [to_gliner(r) for r in train_records]
    train_data = [d for d in train_data if d["ner"]]
    eval_split = train_data[-100:]
    train_split = train_data[:-100]

    # ── Train ──
    print(f"\nLoading {base_model}...")
    model = GLiNER.from_pretrained(base_model).to("cuda")

    print(f"Fine-tuning: {epochs} epochs, batch {batch_size}...")
    start = time.time()
    model.train_model(
        train_dataset=train_split,
        eval_dataset=eval_split,
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

    # ── Eval (fine + coarse) ──
    def overlap(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]

    def eval_fine(m, recs, labels):
        tp = fp = fn = 0
        for rec in recs:
            ents = m.predict_entities(rec["text"], labels, threshold=0.5)
            pred = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in ents]
            gold = rec["goldSpans"]
            gm, pm = set(), set()
            for pi, p in enumerate(pred):
                for gi, g in enumerate(gold):
                    if gi in gm: continue
                    if p["label"] == g["label"] and overlap(p, g):
                        pm.add(pi); gm.add(gi); tp += 1; break
            fp += sum(1 for pi in range(len(pred)) if pi not in pm)
            fn += sum(1 for gi in range(len(gold)) if gi not in gm)
        prec = tp/(tp+fp) if tp+fp else 0
        rec_ = tp/(tp+fn) if tp+fn else 0
        return 2*prec*rec_/(prec+rec_) if prec+rec_ else 0

    def eval_coarse(m, recs, labels, mapping):
        tp = fp = fn = 0
        for rec in recs:
            ents = m.predict_entities(rec["text"], labels, threshold=0.5)
            pred = [{"start": e["start"], "end": e["end"], "label": mapping.get(e["label"])} for e in ents if mapping.get(e["label"])]
            gold = [{**s, "label": mapping[s["label"]]} for s in rec["goldSpans"] if s["label"] in mapping]
            gm, pm = set(), set()
            for pi, p in enumerate(pred):
                for gi, g in enumerate(gold):
                    if gi in gm: continue
                    if p["label"] == g["label"] and overlap(p, g):
                        pm.add(pi); gm.add(gi); tp += 1; break
            fp += sum(1 for pi in range(len(pred)) if pi not in pm)
            fn += sum(1 for gi in range(len(gold)) if gi not in gm)
        prec = tp/(tp+fp) if tp+fp else 0
        rec_ = tp/(tp+fn) if tp+fn else 0
        return 2*prec*rec_/(prec+rec_) if prec+rec_ else 0

    model.eval()
    fine_f1 = eval_fine(model, test_records, fine_labels)
    coarse_f1 = eval_coarse(model, test_records, fine_labels, FINE_TO_COARSE)

    result = {
        "base_model": base_model,
        "full_dataset": full,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_size": len(train_split),
        "test_size": len(test_records),
        "fine_labels": len(fine_labels),
        "fine_f1": fine_f1,
        "coarse_f1": coarse_f1,
        "train_time_seconds": train_time,
    }
    print(f"\n{'='*60}")
    print(f"RESULT: fine F1 {fine_f1*100:.1f}%, coarse F1 {coarse_f1*100:.1f}%")
    print(f"  NVIDIA reference: fine 96.2%, coarse 96.7%")
    print(f"{'='*60}")

    # Save result to volume
    with open(f"{MODEL_DIR}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()

    return result


@app.local_entrypoint()
def main(epochs: int = 5, full: bool = False, base_model: str = "urchade/gliner_small-v2.1", batch_size: int = 16):
    result = train.remote(epochs=epochs, full=full, base_model=base_model, batch_size=batch_size)
    print("\nFinal result:")
    import json
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def sweep():
    """Run multiple configs in parallel on separate A100s.

    Small dataset (525 examples) needs many gradient updates → small batch
    or many epochs. This sweep finds the right combo.
    """
    import json
    configs = [
        {"epochs": 5,  "batch_size": 4},    # ~650 steps
        {"epochs": 10, "batch_size": 4},    # ~1300 steps
        {"epochs": 15, "batch_size": 2},    # ~3900 steps
        {"epochs": 10, "batch_size": 2},    # ~2600 steps
    ]
    print(f"Running {len(configs)} configs in parallel on A100s...")
    results = list(train.starmap(
        [(c["epochs"], "urchade/gliner_small-v2.1", c["batch_size"], False, 100) for c in configs]
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
