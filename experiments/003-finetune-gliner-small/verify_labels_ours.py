#!/usr/bin/env python3
"""
Test label sensitivity on OUR fine-tuned model (symmetry with verify_labels.py).

Our model was trained with Title Case labels ('Person Name', etc.).
But maybe it inherited GLiNER's pre-training for snake_case labels too,
or generalizes to natural-language.

Test all three label formats on the same test set.
"""

import json
import os
import random
import time

import torch
from gliner import GLiNER

SEED = 42
TEST_SIZE = 100
CACHE = "experiments/output/nemotron_healthcare_cache.jsonl"
MODEL_PATH = "experiments/003-finetune-gliner-small/model"


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


# Three label configurations to test our model with

# A: Title Case (what we TRAINED on)
LABELS_TITLE = [
    "Person Name", "Date of Birth", "Date", "Email", "Phone",
    "Location", "Medical Record", "National ID", "Insurance ID",
    "Medical Info", "Credential", "Financial ID", "URL",
]
MAP_TITLE = {
    "Person Name": "person_name", "Date of Birth": "date_of_birth",
    "Date": "date", "Email": "email", "Phone": "phone",
    "Location": "location", "Medical Record": "medical_record",
    "National ID": "national_id", "Insurance ID": "insurance_id",
    "Medical Info": "medical_info", "Credential": "credential",
    "Financial ID": "financial_id", "URL": "url",
}

# B: snake_case (Nemotron native, what NVIDIA used)
LABELS_SNAKE = [
    "person_name", "date_of_birth", "date", "email", "phone",
    "location", "medical_record", "national_id", "insurance_id",
    "medical_info", "credential", "financial_id", "url",
]
MAP_SNAKE = {label: label for label in LABELS_SNAKE}

# C: Natural language
LABELS_NATURAL = [
    "person name", "date of birth", "date", "email address", "phone number",
    "location", "medical record", "national id", "insurance id",
    "medical information", "credential", "financial id", "url",
]
MAP_NATURAL = {
    "person name": "person_name", "date of birth": "date_of_birth",
    "date": "date", "email address": "email", "phone number": "phone",
    "location": "location", "medical record": "medical_record",
    "national id": "national_id", "insurance id": "insurance_id",
    "medical information": "medical_info", "credential": "credential",
    "financial id": "financial_id", "url": "url",
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
    print("LABEL-SENSITIVITY CHECK — our fine-tuned model")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    test_records = get_test_records()

    # Find latest checkpoint
    checkpoints = sorted([d for d in os.listdir(MODEL_PATH) if d.startswith("checkpoint-")],
                        key=lambda d: int(d.split("-")[1]))
    latest = os.path.join(MODEL_PATH, checkpoints[-1])
    print(f"Model: {latest}")
    print(f"Test set: {len(test_records)} records\n")

    model = GLiNER.from_pretrained(latest, load_tokenizer=True).to(device)

    print("[A] Title Case ('Person Name', 'Medical Record') — what we TRAINED on")
    a = eval_with_labels(model, test_records, LABELS_TITLE, MAP_TITLE)
    print(f"    F1: {a['f1']*100:.1f}% (P: {a['p']*100:.1f}%, R: {a['r']*100:.1f}%) — TP/FP/FN: {a['tp']}/{a['fp']}/{a['fn']}")
    print()

    print("[B] snake_case ('person_name', 'medical_record') — Nemotron native")
    b = eval_with_labels(model, test_records, LABELS_SNAKE, MAP_SNAKE)
    print(f"    F1: {b['f1']*100:.1f}% (P: {b['p']*100:.1f}%, R: {b['r']*100:.1f}%) — TP/FP/FN: {b['tp']}/{b['fp']}/{b['fn']}")
    print()

    print("[C] Natural language ('person name', 'medical record')")
    c = eval_with_labels(model, test_records, LABELS_NATURAL, MAP_NATURAL)
    print(f"    F1: {c['f1']*100:.1f}% (P: {c['p']*100:.1f}%, R: {c['r']*100:.1f}%) — TP/FP/FN: {c['tp']}/{c['fp']}/{c['fn']}")
    print()

    print("=" * 70)
    print("SUMMARY — our model")
    print("=" * 70)
    print(f"  Title Case (trained on):  {a['f1']*100:.1f}% F1  ← what we previously reported")
    print(f"  snake_case (Nemotron):    {b['f1']*100:.1f}% F1")
    print(f"  Natural language:         {c['f1']*100:.1f}% F1")

    best = max([("Title", a), ("snake", b), ("Natural", c)], key=lambda x: x[1]["f1"])
    print(f"\n  Best: {best[0]}-case at {best[1]['f1']*100:.1f}%")
    print(f"\n  For comparison:")
    print(f"  nvidia/gliner-PII best (snake_case):  97.2% F1")

    with open("experiments/003-finetune-gliner-small/our_label_sensitivity.json", "w") as f:
        json.dump({"title_case": a, "snake_case": b, "natural_language": c}, f, indent=2)


if __name__ == "__main__":
    main()
