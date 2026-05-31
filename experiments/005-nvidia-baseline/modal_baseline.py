#!/usr/bin/env python3
"""
Experiment 005: Replicate NVIDIA's gliner-PII as a baseline in our pipeline.

NVIDIA's recipe (from their model card):
  - Base: urchade/gliner_large-v2.1 (459M params)
  - Data: full Nemotron-PII, ~100k records, 50+ industries, 55+ entity types
  - All FINE labels (no coarsening)

We replicate it (optionally on gliner_small to compare base-model effect),
then evaluate against nvidia/gliner-PII on a held-out slice of the SAME data.

Goal: prove we can reproduce a general-purpose PII model from scratch — the
reference CONTROL that every domain-specialized model gets measured against.
NVIDIA shipped weights but no recipe; this is the recipe.

Run:
    modal run experiments/005-nvidia-baseline/modal_baseline.py                    # gliner_small, full data
    modal run experiments/005-nvidia-baseline/modal_baseline.py --base-model urchade/gliner_large-v2.1
    modal run experiments/005-nvidia-baseline/modal_baseline.py --max-records 30000  # smaller for speed
"""

import modal

app = modal.App("pii-proxy-nvidia-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "gliner", "datasets", "accelerate", "seqeval", "sentencepiece")
)

volume = modal.Volume.from_name("pii-proxy-models", create_if_missing=True)
MODEL_DIR = "/models"


@app.function(
    image=image,
    gpu="A100:4",
    timeout=60 * 60 * 4,
    volumes={MODEL_DIR: volume},
)
def train_baseline(
    base_model: str = "urchade/gliner_large-v2.1",
    max_records: int = 100_000,
    epochs: int = 1,
    batch_size: int = 32,
    test_size: int = 500,
):
    import ast, json, random, re, time, sys
    import torch
    from datasets import load_dataset
    from gliner import GLiNER

    sys.stdout.reconfigure(line_buffering=True)  # live progress in Modal logs

    random.seed(42)
    torch.manual_seed(42)

    # ── Load FULL Nemotron-PII with ALL fine labels (no coarsening) ──
    print(f"Loading full Nemotron-PII (up to {max_records} records, ALL entity types)...")
    ds = load_dataset("nvidia/Nemotron-PII", split="train", streaming=True)
    records = []
    for row in ds:
        spans_raw = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
        spans = [{"start": s["start"], "end": s["end"], "text": s["text"], "label": s["label"]}
                 for s in spans_raw]
        if spans:
            records.append({"text": row["text"], "goldSpans": spans, "domain": row.get("domain", "")})
        if len(records) >= max_records:
            break
    print(f"  Loaded {len(records)} records")

    all_labels = sorted({s["label"] for r in records for s in r["goldSpans"]})
    print(f"  Entity types (full Nemotron taxonomy): {len(all_labels)}")
    print(f"  Domains: {len(set(r['domain'] for r in records))}")

    random.shuffle(records)
    test_records = records[:test_size]
    train_records = records[test_size:]
    print(f"  Train: {len(train_records)}, Test: {len(test_records)}")
    n_steps = (len(train_records) // batch_size) * epochs
    print(f"  Planned steps: ~{n_steps}")

    # ── tokenize + char→token ──
    def tok(text):
        p = re.compile(r"\w+|[^\w\s]")
        toks, offs = [], []
        for m in p.finditer(text):
            toks.append(m.group()); offs.append((m.start(), m.end()))
        return toks, offs

    def c2t(cs, ce, offs):
        si = ei = None
        for i, (s, e) in enumerate(offs):
            if s < ce and e > cs:
                if si is None: si = i
                ei = i
        return [si, ei] if si is not None else None

    def to_gliner(rec):
        toks, offs = tok(rec["text"])
        ner = []
        for sp in rec["goldSpans"]:
            ts = c2t(sp["start"], sp["end"], offs)
            if ts:
                ner.append([ts[0], ts[1], sp["label"]])
        return {"tokenized_text": toks, "ner": ner}

    train_data = [to_gliner(r) for r in train_records]
    train_data = [d for d in train_data if d["ner"]]
    eval_split = train_data[-500:]
    train_split = train_data[:-500]
    print(f"  Train examples: {len(train_split)}, eval: {len(eval_split)}")

    # ── train (NVIDIA-style: focal loss, full data) ──
    print(f"\nLoading {base_model}...")
    model = GLiNER.from_pretrained(base_model).to("cuda")

    print(f"Training: {epochs} epochs, batch {batch_size} (NVIDIA recipe: focal loss on)...")
    start = time.time()
    model.train_model(
        train_dataset=train_split,
        eval_dataset=eval_split,
        output_dir=f"{MODEL_DIR}/baseline-checkpoint",
        learning_rate=2e-5,        # scaled up for larger effective batch (32 x 4 GPU = 128)
        others_lr=4e-5,            # scaled proportionally
        weight_decay=0.01,
        others_weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        bf16=True,                 # A100 native bf16 tensor cores -> ~1.5-2x throughput
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=epochs,
        max_steps=-1,
        save_steps=2000,
        save_total_limit=1,
        dataloader_num_workers=8,
        report_to="none",
    )
    train_time = time.time() - start
    print(f"Training done in {train_time:.0f}s ({train_time/60:.1f} min)")

    # ── eval our baseline + nvidia on same test set ──
    def overlap(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]

    def evaluate(m, recs, labels):
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
        return {"f1": 2*prec*rec_/(prec+rec_) if prec+rec_ else 0, "p": prec, "r": rec_,
                "tp": tp, "fp": fp, "fn": fn}

    import gc

    def free_gpu():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model.eval()
    print("\nEvaluating our baseline on held-out test...")
    ours = evaluate(model, test_records, all_labels)
    print(f"  Ours: F1 {ours['f1']*100:.1f}% (P {ours['p']*100:.1f}%, R {ours['r']*100:.1f}%)")

    # Save OUR result immediately so a downstream OOM can't lose it
    with open(f"{MODEL_DIR}/baseline_ours_partial.json", "w") as f:
        json.dump({"base_model": base_model, "ours": ours, "train_time_seconds": train_time,
                   "train_records": len(train_split), "entity_types": len(all_labels)}, f, indent=2)
    volume.commit()

    # Aggressively free the trained model + optimizer state before loading NVIDIA's 1.7GB model
    del model
    free_gpu()
    print(f"  GPU freed. Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    print("Loading nvidia/gliner-PII for comparison...")
    nv = GLiNER.from_pretrained("nvidia/gliner-PII").to("cuda")
    nvidia = evaluate(nv, test_records, all_labels)
    print(f"  NVIDIA: F1 {nvidia['f1']*100:.1f}% (P {nvidia['p']*100:.1f}%, R {nvidia['r']*100:.1f}%)")

    result = {
        "experiment": "005-nvidia-baseline",
        "base_model": base_model,
        "train_records": len(train_split),
        "test_records": len(test_records),
        "entity_types": len(all_labels),
        "epochs": epochs,
        "batch_size": batch_size,
        "ours": ours,
        "nvidia": nvidia,
        "delta_f1_pp": (ours["f1"] - nvidia["f1"]) * 100,
        "train_time_seconds": train_time,
    }
    print(f"\n{'='*60}")
    print(f"BASELINE REPLICATION RESULT")
    print(f"  Our baseline ({base_model.split('/')[-1]}): {ours['f1']*100:.1f}% F1")
    print(f"  nvidia/gliner-PII:                          {nvidia['f1']*100:.1f}% F1")
    print(f"  Delta: {result['delta_f1_pp']:+.1f}pp")
    print(f"  Trained on {len(train_split)} records, {len(all_labels)} entity types")
    print(f"{'='*60}")

    with open(f"{MODEL_DIR}/baseline_result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return result


@app.local_entrypoint()
def main(base_model: str = "urchade/gliner_small-v2.1", max_records: int = 100_000, epochs: int = 3, batch_size: int = 16):
    import json
    r = train_baseline.remote(base_model=base_model, max_records=max_records, epochs=epochs, batch_size=batch_size)
    print("\nFinal:")
    print(json.dumps(r, indent=2))
