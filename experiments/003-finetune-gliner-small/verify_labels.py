#!/usr/bin/env python3
"""
Verify NVIDIA's gliner-PII isn't being unfairly handicapped by our
choice of query label strings.

Tests 3 label configurations for NVIDIA's model:
  A. Natural language ("person name") — what we used
  B. Snake_case Nemotron labels ("first_name", "last_name") — its training labels
  C. The original 55+ Nemotron labels (full set)

If NVIDIA's score jumps significantly with B or C, our comparison was biased.
"""

import json
import random
import time

import torch
from gliner import GLiNER

SEED = 42
TEST_SIZE = 100
CACHE = "experiments/output/nemotron_healthcare_cache.jsonl"


def get_test_records():
    random.seed(SEED)
    torch.manual_seed(SEED)
    records = []
    with open(CACHE) as f:
        for line in f:
            records.append(json.loads(line))
    random.shuffle(records)
    return records[:TEST_SIZE]


def spans_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


# Configuration A: Natural language (what we used before)
LABELS_A = [
    "person name", "date of birth", "date", "email address",
    "phone number", "street address", "city", "state", "country",
    "medical record number", "social security number",
    "health insurance id", "blood type", "url", "password",
    "biometric id", "organization",
]
MAP_A = {
    "person name": "person_name", "date of birth": "date_of_birth",
    "date": "date", "email address": "email", "phone number": "phone",
    "street address": "location", "city": "location", "state": "location",
    "country": "location", "medical record number": "medical_record",
    "social security number": "national_id", "health insurance id": "insurance_id",
    "blood type": "medical_info", "url": "url", "password": "credential",
    "biometric id": "national_id", "organization": "organization",
}

# Configuration B: Exact Nemotron training label strings
LABELS_B = [
    "first_name", "last_name", "middle_name",
    "medical_record_number", "date_of_birth", "date", "date_time", "time",
    "email", "phone_number",
    "street_address", "city", "state", "county", "zip_code", "country",
    "ssn", "health_plan_beneficiary_number", "certificate_license_number",
    "biometric_identifier", "url", "unique_id", "blood_type",
    "pin", "password", "swift_bic",
]
MAP_B = {
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


def eval_with_labels(model, test_records, query_labels, label_map):
    tp, fp, fn = 0, 0, 0
    times = []

    for rec in test_records:
        text = rec["text"]
        gold = rec["goldSpans"]

        t0 = time.perf_counter()
        entities = model.predict_entities(text, query_labels, threshold=0.5)
        times.append((time.perf_counter() - t0) * 1000)

        pred = []
        for e in entities:
            mapped = label_map.get(e["label"])
            if mapped is None:
                continue
            pred.append({"start": e["start"], "end": e["end"],
                         "text": e["text"], "label": mapped})

        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm:
                    continue
                if p["label"] == g["label"] and spans_overlap(p, g):
                    pm.add(pi); gm.add(gi); tp += 1; break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return {"f1": f1, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn,
            "avg_ms": sum(times)/len(times)}


def main():
    print("=" * 70)
    print("LABEL-SENSITIVITY CHECK — nvidia/gliner-PII")
    print("=" * 70)
    print("Testing if NVIDIA's score depends on query label string choice")
    print()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    test_records = get_test_records()
    print(f"Test set: {len(test_records)} records, "
          f"{sum(len(r['goldSpans']) for r in test_records)} gold entities\n")

    model = GLiNER.from_pretrained("nvidia/gliner-PII").to(device)

    print("[A] Natural language labels ('person name', 'medical record number', ...)")
    a = eval_with_labels(model, test_records, LABELS_A, MAP_A)
    print(f"    F1: {a['f1']*100:.1f}% (P: {a['p']*100:.1f}%, R: {a['r']*100:.1f}%)")
    print(f"    TP/FP/FN: {a['tp']}/{a['fp']}/{a['fn']}")
    print()

    print("[B] Snake_case Nemotron labels ('first_name', 'medical_record_number', ...)")
    b = eval_with_labels(model, test_records, LABELS_B, MAP_B)
    print(f"    F1: {b['f1']*100:.1f}% (P: {b['p']*100:.1f}%, R: {b['r']*100:.1f}%)")
    print(f"    TP/FP/FN: {b['tp']}/{b['fp']}/{b['fn']}")
    print()

    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    delta = (b["f1"] - a["f1"]) * 100
    print(f"  Natural language (A):  {a['f1']*100:.1f}% F1")
    print(f"  Native labels (B):     {b['f1']*100:.1f}% F1")
    print(f"  Delta:                 {delta:+.1f}pp")
    print()
    if abs(delta) < 1.0:
        print("  → Label choice doesn't matter. Our 90.4% number is fair.")
    elif delta > 0:
        print(f"  → NVIDIA scored {delta:.1f}pp HIGHER with its native labels.")
        print(f"  → Our previous comparison underrated their model.")
        print(f"  → Best NVIDIA score: {b['f1']*100:.1f}% vs our 95.5% = {(0.955 - b['f1'])*100:+.1f}pp")
    else:
        print(f"  → Natural language was actually slightly better. NVIDIA's score holds.")

    with open("experiments/003-finetune-gliner-small/label_sensitivity.json", "w") as f:
        json.dump({"natural_language": a, "snake_case_native": b, "delta_pp": delta}, f, indent=2)


if __name__ == "__main__":
    main()
