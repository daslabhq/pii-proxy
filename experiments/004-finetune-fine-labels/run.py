#!/usr/bin/env python3
"""
Experiment 004: Fine-tune gliner_small-v2.1 on Nemotron-PII with FINE labels
(same vocabulary as NVIDIA's gliner-PII, not collapsed).

This enables a truly fair comparison with nvidia/gliner-PII:
  - Same training data
  - Same label granularity (~26 healthcare-relevant fine labels from Nemotron)
  - Same query vocabulary at inference

Coarse PII categories can still be derived post-hoc via label mapping.

Usage:
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 experiments/004-finetune-fine-labels/run.py
"""

import json, os, time, random, re
import ast
import torch
from gliner import GLiNER

SEED = 42
EPOCHS = 5
BATCH_SIZE = 2  # reduced from 4 — 26 fine labels ~2x memory vs 13 coarse
LR = 5e-6
OTHERS_LR = 5e-6
TEST_SIZE = 100
BASE_MODEL = "urchade/gliner_small-v2.1"
OUTPUT_DIR = "experiments/004-finetune-fine-labels"
MODEL_OUT = os.path.join(OUTPUT_DIR, "model")

random.seed(SEED)
torch.manual_seed(SEED)

# Fine-grained Nemotron labels present in healthcare records
# (sourced from the raw goldSpans before any collapsing)
# Coarse mapping for downstream pii-proxy use (NOT used during training)
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


def load_nemotron_raw():
    """Load with ORIGINAL fine labels (not the collapsed cache)."""
    from datasets import load_dataset
    ds = load_dataset("nvidia/Nemotron-PII", split="train", streaming=True)
    records = []
    total = 0
    for row in ds:
        total += 1
        domain = row.get("domain", "") or ""
        if "health" not in domain.lower():
            if total >= 20000: break
            continue
        spans_raw = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
        # Keep ALL fine labels — no coarsening
        fine_spans = [
            {"start": s["start"], "end": s["end"], "text": s["text"], "label": s["label"]}
            for s in spans_raw if s["label"] in FINE_TO_COARSE
        ]
        if fine_spans:
            records.append({"text": row["text"], "goldSpans": fine_spans})
        if total >= 20000: break
    return records


def tokenize_with_offsets(text):
    pattern = re.compile(r"\w+|[^\w\s]")
    tokens, offsets = [], []
    for m in pattern.finditer(text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def char_span_to_token_span(char_start, char_end, offsets):
    start_idx, end_idx = None, None
    for i, (s, e) in enumerate(offsets):
        if s < char_end and e > char_start:
            if start_idx is None:
                start_idx = i
            end_idx = i
    if start_idx is None:
        return None
    return [start_idx, end_idx]


def record_to_gliner_format(record):
    tokens, offsets = tokenize_with_offsets(record["text"])
    ner = []
    for span in record["goldSpans"]:
        tok_span = char_span_to_token_span(span["start"], span["end"], offsets)
        if tok_span is None:
            continue
        ner.append([tok_span[0], tok_span[1], span["label"]])
    return {"tokenized_text": tokens, "ner": ner}


def spans_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def evaluate_fine(model, test_records, fine_labels, threshold=0.5):
    """Evaluate at fine granularity (no collapsing)."""
    tp, fp, fn = 0, 0, 0
    times = []
    for rec in test_records:
        text = rec["text"]
        gold = rec["goldSpans"]
        t0 = time.perf_counter()
        entities = model.predict_entities(text, fine_labels, threshold=threshold)
        times.append((time.perf_counter() - t0) * 1000)
        pred = [{"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"]}
                for e in entities if e["label"] in fine_labels]
        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm: continue
                if p["label"] == g["label"] and spans_overlap(p, g):
                    pm.add(pi); gm.add(gi); tp += 1; break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return {"f1": f1, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn,
            "avg_ms": sum(times)/len(times)}


def evaluate_coarse(model, test_records, fine_labels, fine_to_coarse, threshold=0.5):
    """Evaluate at coarse granularity (collapsed post-hoc)."""
    tp, fp, fn = 0, 0, 0
    times = []
    for rec in test_records:
        text = rec["text"]
        # Collapse gold to coarse for fair comparison
        gold = [{**s, "label": fine_to_coarse[s["label"]]}
                for s in rec["goldSpans"] if s["label"] in fine_to_coarse]
        t0 = time.perf_counter()
        entities = model.predict_entities(text, fine_labels, threshold=threshold)
        times.append((time.perf_counter() - t0) * 1000)
        # Collapse predictions to coarse
        pred = [{"start": e["start"], "end": e["end"], "text": e["text"],
                 "label": fine_to_coarse.get(e["label"])}
                for e in entities if e["label"] in fine_to_coarse]
        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm: continue
                if p["label"] == g["label"] and spans_overlap(p, g):
                    pm.add(pi); gm.add(gi); tp += 1; break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return {"f1": f1, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn,
            "avg_ms": sum(times)/len(times)}


def main():
    print("=" * 70)
    print(f"Experiment 004: Fine-tune {BASE_MODEL} with FINE Nemotron labels")
    print("=" * 70)

    # Load raw Nemotron with fine labels
    print("\nLoading Nemotron-PII healthcare records with FINE labels...")
    all_records = load_nemotron_raw()
    print(f"  Records: {len(all_records)}")
    fine_label_set = sorted({s["label"] for r in all_records for s in r["goldSpans"]})
    print(f"  Fine labels in healthcare subset: {len(fine_label_set)}")
    print(f"  Labels: {fine_label_set}")

    random.shuffle(all_records)
    test_records = all_records[:TEST_SIZE]
    train_records = all_records[TEST_SIZE:]
    print(f"\nTrain: {len(train_records)}, Test: {len(test_records)}")

    # Convert
    print("\nConverting to GLiNER format with fine labels...")
    train_data = [record_to_gliner_format(r) for r in train_records]
    train_data = [d for d in train_data if d["ner"]]
    eval_split = train_data[-50:]
    train_split = train_data[:-50]
    print(f"  Train examples: {len(train_split)}, In-loop eval: {len(eval_split)}")

    # Load model
    print(f"\nLoading {BASE_MODEL}...")
    model = GLiNER.from_pretrained(BASE_MODEL)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    # Baseline (zero-shot with fine labels)
    print("\nBaseline: zero-shot with fine labels...")
    baseline = evaluate_fine(model, test_records, fine_label_set)
    print(f"  Fine F1: {baseline['f1']*100:.1f}%")
    coarse_baseline = evaluate_coarse(model, test_records, fine_label_set, FINE_TO_COARSE)
    print(f"  Coarse F1 (post-hoc collapsed): {coarse_baseline['f1']*100:.1f}%")

    # Fine-tune
    print(f"\nFine-tuning {EPOCHS} epochs, batch {BATCH_SIZE}...")
    start = time.time()
    model.train_model(
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
        focal_loss_alpha=-1,
        focal_loss_gamma=0,
        num_train_epochs=EPOCHS,
        max_steps=-1,
        save_steps=200,
        save_total_limit=2,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
    )
    train_time = time.time() - start
    print(f"Training done in {train_time:.0f}s")

    # Evaluate at fine + coarse granularity
    model.eval()
    print("\nFine-grained evaluation (for direct comparison with NVIDIA):")
    fine_results = evaluate_fine(model, test_records, fine_label_set)
    print(f"  F1: {fine_results['f1']*100:.1f}% (P: {fine_results['p']*100:.1f}%, R: {fine_results['r']*100:.1f}%)")
    print(f"  Latency: {fine_results['avg_ms']:.1f}ms/record")

    print("\nCoarse-grained evaluation (for pii-proxy use case):")
    coarse_results = evaluate_coarse(model, test_records, fine_label_set, FINE_TO_COARSE)
    print(f"  F1: {coarse_results['f1']*100:.1f}% (P: {coarse_results['p']*100:.1f}%, R: {coarse_results['r']*100:.1f}%)")
    print(f"  Latency: {coarse_results['avg_ms']:.1f}ms/record")

    # Compare with NVIDIA at the same fine granularity
    print("\nLoading nvidia/gliner-PII for direct comparison...")
    del model
    nvidia_model = GLiNER.from_pretrained("nvidia/gliner-PII").to(device)
    print("nvidia/gliner-PII at fine granularity (same label set):")
    nvidia_fine = evaluate_fine(nvidia_model, test_records, fine_label_set)
    print(f"  F1: {nvidia_fine['f1']*100:.1f}% (P: {nvidia_fine['p']*100:.1f}%, R: {nvidia_fine['r']*100:.1f}%)")
    print(f"  Latency: {nvidia_fine['avg_ms']:.1f}ms/record")

    print("\n" + "=" * 70)
    print("SUMMARY — TRUE apples-to-apples")
    print("=" * 70)
    print(f"  Test set: {len(test_records)} records, fine labels: {len(fine_label_set)}")
    print()
    print(f"  Fine-grained F1 (same labels both models trained on):")
    print(f"    nvidia/gliner-PII:        {nvidia_fine['f1']*100:.1f}%  ({nvidia_fine['avg_ms']:.0f}ms)")
    print(f"    Our fine-tuned (this exp): {fine_results['f1']*100:.1f}%  ({fine_results['avg_ms']:.0f}ms)")
    delta_fine = (fine_results['f1'] - nvidia_fine['f1']) * 100
    print(f"    Delta: {delta_fine:+.1f}pp")
    print()
    print(f"  Coarse-grained F1 (collapsed to 13 types for pii-proxy):")
    print(f"    Our fine-tuned (collapsed): {coarse_results['f1']*100:.1f}%")
    print(f"    Exp 003 (trained on coarse): 95.5%")

    # Save
    output = {
        "experiment": "004-finetune-fine-labels",
        "model": BASE_MODEL,
        "fine_labels": fine_label_set,
        "train_size": len(train_split),
        "test_size": len(test_records),
        "epochs": EPOCHS,
        "baseline_zero_shot_fine": baseline,
        "baseline_zero_shot_coarse": coarse_baseline,
        "fine_tuned_fine": fine_results,
        "fine_tuned_coarse": coarse_results,
        "nvidia_fine": nvidia_fine,
        "train_time_seconds": train_time,
    }
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
