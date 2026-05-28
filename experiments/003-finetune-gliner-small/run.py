#!/usr/bin/env python3
"""
Fine-tune urchade/gliner_small-v2.1 on Nemotron-PII healthcare subset.

Uses GLiNER's official training API (model.train_model) — not a custom loop.
Reference: https://github.com/urchade/GLiNER/blob/main/examples/finetune.ipynb

Same held-out test set as experiment 002 (100 records).
Compares against:
  - nvidia/gliner-PII zero-shot (experiment 002): 90.3% F1, 333ms
  - Our BERT classifier (experiment 002): 93.9% F1, 26ms

Usage:
  python3 experiments/003-finetune-gliner-small/run.py
"""

import json
import os
import time
import random
import re

import torch
from gliner import GLiNER

# ─── Config ──────────────────────────────────────────────────────

SEED = 42
EPOCHS = 5          # reduced from 8 — MPS dies near completion; loss plateaus by ep 4
BATCH_SIZE = 4      # reduced from 8 — MPS OOM at batch=8 with focal loss
LR = 5e-6           # encoder LR
OTHERS_LR = 5e-6    # span/scorer head LR — reduced from 1e-5 (grad explosion)
TEST_SIZE = 100
BASE_MODEL = "urchade/gliner_small-v2.1"
OUTPUT_DIR = "experiments/003-finetune-gliner-small"
MODEL_OUT = os.path.join(OUTPUT_DIR, "model")

random.seed(SEED)
torch.manual_seed(SEED)

# GLiNER v2.1 wants Capitalized labels — map our snake_case → Title Case
LABEL_MAP = {
    "person_name": "Person Name",
    "date_of_birth": "Date of Birth",
    "date": "Date",
    "email": "Email",
    "phone": "Phone",
    "location": "Location",
    "medical_record": "Medical Record",
    "national_id": "National ID",
    "insurance_id": "Insurance ID",
    "medical_info": "Medical Info",
    "credential": "Credential",
    "financial_id": "Financial ID",
    "url": "URL",
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
ENTITY_TYPES_CAPITALIZED = list(LABEL_MAP.values())


# ─── Data loading ─────────────────────────────────────────────────

def load_nemotron():
    cache = "experiments/output/nemotron_healthcare_cache.jsonl"
    records = []
    with open(cache) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ─── Char → token mapping (GLiNER expects inclusive token indices) ────

def tokenize_with_offsets(text):
    pattern = re.compile(r"\w+|[^\w\s]")
    tokens = []
    offsets = []
    for m in pattern.finditer(text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def char_span_to_token_span(char_start, char_end, offsets):
    """Convert character span [char_start, char_end) to INCLUSIVE token span."""
    start_idx = None
    end_idx = None
    for i, (s, e) in enumerate(offsets):
        if s < char_end and e > char_start:
            if start_idx is None:
                start_idx = i
            end_idx = i
    if start_idx is None:
        return None
    return [start_idx, end_idx]


def record_to_gliner_format(record):
    """Convert {text, goldSpans} to {tokenized_text, ner} with Capitalized labels."""
    tokens, offsets = tokenize_with_offsets(record["text"])
    ner = []
    for span in record["goldSpans"]:
        tok_span = char_span_to_token_span(span["start"], span["end"], offsets)
        if tok_span is None:
            continue
        cap_label = LABEL_MAP.get(span["label"])
        if cap_label is None:
            continue
        ner.append([tok_span[0], tok_span[1], cap_label])
    return {"tokenized_text": tokens, "ner": ner}


# ─── Eval ────────────────────────────────────────────────────────

def spans_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def evaluate_model(model, test_records, threshold=0.5):
    tp, fp, fn = 0, 0, 0
    times = []

    for rec in test_records:
        text = rec["text"]
        gold = rec["goldSpans"]

        t0 = time.perf_counter()
        entities = model.predict_entities(text, ENTITY_TYPES_CAPITALIZED, threshold=threshold)
        times.append((time.perf_counter() - t0) * 1000)

        # Map predictions back to snake_case for comparison with gold
        pred = []
        for e in entities:
            snake_label = REVERSE_LABEL_MAP.get(e["label"])
            if snake_label is None:
                continue
            pred.append({
                "start": e["start"],
                "end": e["end"],
                "text": e["text"],
                "label": snake_label,
            })

        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm:
                    continue
                if p["label"] == g["label"] and spans_overlap(p, g):
                    pm.add(pi)
                    gm.add(gi)
                    tp += 1
                    break

        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    avg_ms = sum(times) / len(times)
    return {
        "f1": f1, "precision": p, "recall": r,
        "tp": tp, "fp": fp, "fn": fn, "avg_ms": avg_ms,
    }


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"Experiment 003: Fine-tune {BASE_MODEL} on Nemotron-PII healthcare")
    print("=" * 70)

    # Load and split (same seed as experiment 002 → same test set)
    all_records = load_nemotron()
    random.shuffle(all_records)
    test_records = all_records[:TEST_SIZE]
    train_records = all_records[TEST_SIZE:]
    print(f"Train: {len(train_records)}, Test: {len(test_records)}")

    # Convert to GLiNER format
    print("\nConverting to GLiNER format (Capitalized labels)...")
    train_data = [record_to_gliner_format(r) for r in train_records]
    train_data = [d for d in train_data if d["ner"]]
    print(f"Train examples with NER: {len(train_data)}")
    print(f"Total train spans: {sum(len(d['ner']) for d in train_data)}")

    # Eval split (small subset for in-loop eval during training)
    eval_split = train_data[-50:]
    train_split = train_data[:-50]
    print(f"Train: {len(train_split)}, In-loop eval: {len(eval_split)}")

    # Load model
    print(f"\nLoading {BASE_MODEL}...")
    model = GLiNER.from_pretrained(BASE_MODEL)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    # Baseline: zero-shot BEFORE fine-tuning
    print("\nBaseline: zero-shot evaluation BEFORE fine-tuning (gliner_small-v2.1)...")
    baseline_results = evaluate_model(model, test_records, threshold=0.5)
    print(f"  F1: {baseline_results['f1']*100:.1f}% (P: {baseline_results['precision']*100:.1f}%, R: {baseline_results['recall']*100:.1f}%)")
    print(f"  Latency: {baseline_results['avg_ms']:.1f}ms/record")

    # Fine-tune using official API
    print(f"\nFine-tuning via model.train_model() — {EPOCHS} epochs, batch {BATCH_SIZE}...")
    start_time = time.time()

    trainer = model.train_model(
        train_dataset=train_split,
        eval_dataset=eval_split,
        output_dir=MODEL_OUT,
        learning_rate=LR,
        others_lr=OTHERS_LR,
        weight_decay=0.01,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        focal_loss_alpha=-1,  # disable focal loss (unstable on MPS, huge grad norms)
        focal_loss_gamma=0,
        num_train_epochs=EPOCHS,
        max_steps=-1,  # force epochs to take priority (default may be 10000)
        save_steps=200,
        save_total_limit=2,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
    )

    train_time = time.time() - start_time
    print(f"\nTraining done in {train_time:.0f}s")

    # Eval AFTER fine-tuning
    print("\nFine-tuned evaluation on held-out test...")
    model.eval()
    results = evaluate_model(model, test_records, threshold=0.5)
    print(f"  F1: {results['f1']*100:.1f}% (P: {results['precision']*100:.1f}%, R: {results['recall']*100:.1f}%)")
    print(f"  Latency: {results['avg_ms']:.1f}ms/record")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Base model:               {BASE_MODEL}")
    print(f"  Train size:               {len(train_split)} examples")
    print(f"  Test size:                {len(test_records)} records, {sum(len(r['goldSpans']) for r in test_records)} gold spans")
    print()
    print(f"  Zero-shot baseline F1:    {baseline_results['f1']*100:.1f}%  ({baseline_results['avg_ms']:.1f}ms)")
    print(f"  Fine-tuned F1:            {results['f1']*100:.1f}%  ({results['avg_ms']:.1f}ms)")
    print(f"  Improvement:              {(results['f1'] - baseline_results['f1'])*100:+.1f}pp F1")
    print(f"  Training time:            {train_time:.0f}s")
    print()
    print(f"  Comparison (from experiment 002):")
    print(f"    nvidia/gliner-PII zero-shot:        90.3% F1, 333ms")
    print(f"    Our BERT classifier (fine-tuned):   93.9% F1,  26ms")

    output = {
        "experiment": "003-finetune-gliner-small",
        "model": BASE_MODEL,
        "train_size": len(train_split),
        "eval_size": len(eval_split),
        "test_size": len(test_records),
        "epochs": EPOCHS,
        "lr": LR,
        "others_lr": OTHERS_LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "baseline_zero_shot": baseline_results,
        "fine_tuned": results,
        "train_time_seconds": train_time,
    }
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
