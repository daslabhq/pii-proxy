#!/usr/bin/env python3
"""
Evaluate the checkpoint from the crashed training run.
Reuses the same logic as run.py but skips training.
"""
import json, os, random, time, ast, re
import torch
from gliner import GLiNER

SEED = 42
TEST_SIZE = 100
CHECKPOINT = "experiments/004-finetune-fine-labels/model/checkpoint-1000"
OUTPUT_DIR = "experiments/004-finetune-fine-labels"

random.seed(SEED)
torch.manual_seed(SEED)

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
        fine_spans = [{"start": s["start"], "end": s["end"], "text": s["text"], "label": s["label"]}
                      for s in spans_raw if s["label"] in FINE_TO_COARSE]
        if fine_spans:
            records.append({"text": row["text"], "goldSpans": fine_spans})
        if total >= 20000: break
    return records


def spans_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def evaluate_fine(model, test_records, fine_labels, threshold=0.5):
    tp, fp, fn = 0, 0, 0
    times = []
    for rec in test_records:
        text, gold = rec["text"], rec["goldSpans"]
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
    return {"f1": 2*p*r/(p+r) if p+r else 0, "p": p, "r": r,
            "tp": tp, "fp": fp, "fn": fn, "avg_ms": sum(times)/len(times)}


def evaluate_coarse(model, test_records, fine_labels, fine_to_coarse, threshold=0.5):
    tp, fp, fn = 0, 0, 0
    times = []
    for rec in test_records:
        text = rec["text"]
        gold = [{**s, "label": fine_to_coarse[s["label"]]}
                for s in rec["goldSpans"] if s["label"] in fine_to_coarse]
        t0 = time.perf_counter()
        entities = model.predict_entities(text, fine_labels, threshold=threshold)
        times.append((time.perf_counter() - t0) * 1000)
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
    return {"f1": 2*p*r/(p+r) if p+r else 0, "p": p, "r": r,
            "tp": tp, "fp": fp, "fn": fn, "avg_ms": sum(times)/len(times)}


def main():
    print("=" * 70)
    print("Eval-only: checkpoint-1000 from crashed run 004")
    print("=" * 70)

    all_records = load_nemotron_raw()
    fine_labels = sorted({s["label"] for r in all_records for s in r["goldSpans"]})
    print(f"\nFine labels: {len(fine_labels)} — {fine_labels}")
    random.shuffle(all_records)
    test_records = all_records[:TEST_SIZE]
    gold_count = sum(len(r["goldSpans"]) for r in test_records)
    print(f"Test: {len(test_records)} records, {gold_count} gold entities\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading checkpoint: {CHECKPOINT}")
    our_model = GLiNER.from_pretrained(CHECKPOINT, load_tokenizer=True).to(device)

    print("\n[1] Our model (fine labels, fine eval) — direct vs NVIDIA")
    ours_fine = evaluate_fine(our_model, test_records, fine_labels)
    print(f"    F1: {ours_fine['f1']*100:.1f}% (P: {ours_fine['p']*100:.1f}%, R: {ours_fine['r']*100:.1f}%)")
    print(f"    Latency: {ours_fine['avg_ms']:.1f}ms/record")
    print(f"    TP/FP/FN: {ours_fine['tp']}/{ours_fine['fp']}/{ours_fine['fn']}")

    print("\n[2] Our model (fine labels → coarse mapping) — for pii-proxy use")
    ours_coarse = evaluate_coarse(our_model, test_records, fine_labels, FINE_TO_COARSE)
    print(f"    F1: {ours_coarse['f1']*100:.1f}% (P: {ours_coarse['p']*100:.1f}%, R: {ours_coarse['r']*100:.1f}%)")
    print(f"    Latency: {ours_coarse['avg_ms']:.1f}ms/record")
    del our_model

    print("\nLoading nvidia/gliner-PII...")
    nvidia_model = GLiNER.from_pretrained("nvidia/gliner-PII").to(device)
    print("\n[3] nvidia/gliner-PII (same fine labels)")
    nvidia_fine = evaluate_fine(nvidia_model, test_records, fine_labels)
    print(f"    F1: {nvidia_fine['f1']*100:.1f}% (P: {nvidia_fine['p']*100:.1f}%, R: {nvidia_fine['r']*100:.1f}%)")
    print(f"    Latency: {nvidia_fine['avg_ms']:.1f}ms/record")
    print(f"    TP/FP/FN: {nvidia_fine['tp']}/{nvidia_fine['fp']}/{nvidia_fine['fn']}")

    print("\n[4] nvidia/gliner-PII (fine → coarse mapping)")
    nvidia_coarse = evaluate_coarse(nvidia_model, test_records, fine_labels, FINE_TO_COARSE)
    print(f"    F1: {nvidia_coarse['f1']*100:.1f}% (P: {nvidia_coarse['p']*100:.1f}%, R: {nvidia_coarse['r']*100:.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY — same labels, same test set, fair comparison")
    print("=" * 70)
    print(f"  Fine-grained F1 (26 native Nemotron labels):")
    print(f"    nvidia/gliner-PII:  {nvidia_fine['f1']*100:.1f}%  ({nvidia_fine['avg_ms']:.0f}ms)")
    print(f"    Ours (ckpt-1000):    {ours_fine['f1']*100:.1f}%  ({ours_fine['avg_ms']:.0f}ms)")
    delta = (ours_fine['f1'] - nvidia_fine['f1']) * 100
    print(f"    Delta: {delta:+.1f}pp")
    print()
    print(f"  Coarse-grained F1 (13 types via post-hoc collapse):")
    print(f"    nvidia/gliner-PII:  {nvidia_coarse['f1']*100:.1f}%")
    print(f"    Ours (ckpt-1000):    {ours_coarse['f1']*100:.1f}%")
    print(f"    (Exp 003 — trained directly on coarse: 95.5%)")

    output = {
        "experiment": "004-finetune-fine-labels (eval-only from ckpt-1000)",
        "fine_labels": fine_labels,
        "ours_fine": ours_fine, "ours_coarse": ours_coarse,
        "nvidia_fine": nvidia_fine, "nvidia_coarse": nvidia_coarse,
        "delta_fine_pp": delta,
    }
    out_path = os.path.join(OUTPUT_DIR, "eval_only_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
