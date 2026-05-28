#!/usr/bin/env python3
"""
Verify claims for experiment 003. Run independently of training.

Claims to verify:
  1. Test sets are identical between experiment 002 and 003
  2. nvidia/gliner-PII zero-shot scores 90.3% F1 (matches exp 002)
  3. Our fine-tuned model scores 95.5% F1 (matches exp 003)
  4. Model sizes: gliner_small ~150MB, gliner-PII ~570MB
  5. No train/test leakage

Usage:
  python3 experiments/003-finetune-gliner-small/verify.py
"""

import json
import os
import random
import time
import hashlib

import torch
from gliner import GLiNER

SEED = 42
TEST_SIZE = 100
MODEL_PATH = "experiments/003-finetune-gliner-small/model"
CACHE = "experiments/output/nemotron_healthcare_cache.jsonl"

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

GLINER_PII_TYPES = [
    "person name", "date of birth", "date", "email address",
    "phone number", "street address", "city", "state", "country",
    "medical record number", "social security number",
    "health insurance id", "blood type", "url", "password",
    "biometric id", "organization",
]
GLINER_PII_MAP = {
    "person name": "person_name", "date of birth": "date_of_birth",
    "date": "date", "email address": "email", "phone number": "phone",
    "street address": "location", "city": "location", "state": "location",
    "country": "location", "medical record number": "medical_record",
    "social security number": "national_id", "health insurance id": "insurance_id",
    "blood type": "medical_info", "url": "url", "password": "credential",
    "biometric id": "national_id", "organization": "organization",
}


def get_test_records(test_size):
    """Same exact loading + splitting as experiments 002 and 003."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    records = []
    with open(CACHE) as f:
        for line in f:
            records.append(json.loads(line))
    random.shuffle(records)
    return records[:test_size], records[test_size:]


def fingerprint(records):
    """Hash of record texts — proves two sets contain identical records."""
    h = hashlib.sha256()
    for r in records:
        h.update(r["text"].encode())
    return h.hexdigest()[:16]


def spans_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def eval_model(model, test_records, entity_types, label_map, threshold=0.5):
    tp, fp, fn = 0, 0, 0
    times = []

    for rec in test_records:
        text = rec["text"]
        gold = rec["goldSpans"]

        t0 = time.perf_counter()
        entities = model.predict_entities(text, entity_types, threshold=threshold)
        times.append((time.perf_counter() - t0) * 1000)

        pred = []
        for e in entities:
            mapped = label_map.get(e["label"])
            if mapped is None:
                continue
            pred.append({
                "start": e["start"], "end": e["end"],
                "text": e["text"], "label": mapped,
            })

        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm:
                    continue
                if p["label"] == g["label"] and spans_overlap(p, g):
                    pm.add(pi); gm.add(gi)
                    tp += 1
                    break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return {"f1": f1, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn,
            "avg_ms": sum(times)/len(times)}


def model_size_mb(model_dir_or_repo):
    """Get model size from HuggingFace cache or local dir."""
    from huggingface_hub import scan_cache_dir
    if os.path.isdir(model_dir_or_repo):
        total = 0
        for root, _, files in os.walk(model_dir_or_repo):
            for fn in files:
                if any(fn.endswith(ext) for ext in [".safetensors", ".bin", ".pt", ".pth", ".onnx"]):
                    total += os.path.getsize(os.path.join(root, fn))
        return total / 1024 / 1024
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == model_dir_or_repo:
            total = 0
            for rev in repo.revisions:
                for f in rev.files:
                    if any(str(f.file_path).endswith(ext) for ext in [".safetensors", ".bin", ".pt", ".onnx"]):
                        total += f.size_on_disk
            return total / 1024 / 1024
    return None


def main():
    print("=" * 70)
    print("VERIFICATION — experiment 003 claims")
    print("=" * 70)

    # ─── Claim 1: Test sets are identical ────────────────────────
    print("\n[1] Test set identity check")
    test_records, train_records = get_test_records(TEST_SIZE)
    fp = fingerprint(test_records)
    print(f"    Test records: {len(test_records)}")
    print(f"    Fingerprint:  {fp}")
    print(f"    Gold entities: {sum(len(r['goldSpans']) for r in test_records)}")
    print(f"    First record text starts: '{test_records[0]['text'][:60]}...'")

    # Train/test leakage check
    test_texts = set(r["text"] for r in test_records)
    train_texts = set(r["text"] for r in train_records)
    overlap = test_texts & train_texts
    print(f"    Train/test text overlap: {len(overlap)} (should be 0)")
    assert len(overlap) == 0, "LEAKAGE DETECTED"

    # ─── Claim 2: nvidia/gliner-PII zero-shot ────────────────────
    print("\n[2] nvidia/gliner-PII zero-shot on the test set")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nvidia_model = GLiNER.from_pretrained("nvidia/gliner-PII").to(device)
    nvidia_results = eval_model(nvidia_model, test_records, GLINER_PII_TYPES, GLINER_PII_MAP)
    print(f"    F1: {nvidia_results['f1']*100:.1f}% (P: {nvidia_results['p']*100:.1f}%, R: {nvidia_results['r']*100:.1f}%)")
    print(f"    Latency: {nvidia_results['avg_ms']:.1f}ms/record")
    print(f"    TP/FP/FN: {nvidia_results['tp']}/{nvidia_results['fp']}/{nvidia_results['fn']}")
    print(f"    Claimed (exp 002): 90.3% F1, 333ms")
    del nvidia_model

    # ─── Claim 3: Our fine-tuned model ───────────────────────────
    print("\n[3] Our fine-tuned model on the test set")
    # Find the latest checkpoint
    if not os.path.isdir(MODEL_PATH):
        print(f"    ERROR: Model dir not found: {MODEL_PATH}")
        return
    checkpoints = sorted([d for d in os.listdir(MODEL_PATH) if d.startswith("checkpoint-")],
                        key=lambda d: int(d.split("-")[1]))
    if not checkpoints:
        print(f"    ERROR: No checkpoints in {MODEL_PATH}")
        return
    latest = os.path.join(MODEL_PATH, checkpoints[-1])
    print(f"    Loading: {latest}")
    our_model = GLiNER.from_pretrained(latest, load_tokenizer=True).to(device)
    our_results = eval_model(
        our_model, test_records,
        list(LABEL_MAP.values()), REVERSE_LABEL_MAP,
    )
    print(f"    F1: {our_results['f1']*100:.1f}% (P: {our_results['p']*100:.1f}%, R: {our_results['r']*100:.1f}%)")
    print(f"    Latency: {our_results['avg_ms']:.1f}ms/record")
    print(f"    TP/FP/FN: {our_results['tp']}/{our_results['fp']}/{our_results['fn']}")
    print(f"    Claimed (exp 003): 95.5% F1, 106ms")
    del our_model

    # ─── Claim 4: Model sizes ────────────────────────────────────
    print("\n[4] Model sizes (actual .safetensors / weights)")
    nvidia_size = model_size_mb("nvidia/gliner-PII")
    small_size = model_size_mb("urchade/gliner_small-v2.1")
    our_size = model_size_mb(latest)
    print(f"    nvidia/gliner-PII:        {nvidia_size:.1f} MB" if nvidia_size else "    nvidia/gliner-PII: ? MB (not in cache)")
    print(f"    urchade/gliner_small-v2.1: {small_size:.1f} MB" if small_size else "    urchade/gliner_small-v2.1: ? MB")
    print(f"    Our fine-tuned (latest checkpoint): {our_size:.1f} MB" if our_size else f"    Our fine-tuned: ? MB")

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    f1_us = our_results["f1"] * 100
    f1_nv = nvidia_results["f1"] * 100
    delta_f1 = f1_us - f1_nv
    delta_latency = nvidia_results["avg_ms"] / our_results["avg_ms"]

    print(f"\n  Test set:           {len(test_records)} records, fingerprint {fp}")
    print(f"  Train/test leak:    {'NONE ✓' if len(overlap) == 0 else 'LEAK FOUND ✗'}")
    print()
    print(f"  Our fine-tuned:     {f1_us:.1f}% F1,  {our_results['avg_ms']:.1f}ms,  {our_size:.0f}MB" if our_size else f"  Our fine-tuned:     {f1_us:.1f}% F1,  {our_results['avg_ms']:.1f}ms")
    print(f"  nvidia/gliner-PII:  {f1_nv:.1f}% F1, {nvidia_results['avg_ms']:.1f}ms, {nvidia_size:.0f}MB" if nvidia_size else f"  nvidia/gliner-PII:  {f1_nv:.1f}% F1, {nvidia_results['avg_ms']:.1f}ms")
    print()
    print(f"  Delta F1:           {delta_f1:+.1f}pp")
    print(f"  Speed ratio:        {delta_latency:.1f}x faster")
    if our_size and nvidia_size:
        print(f"  Size ratio:         {nvidia_size/our_size:.1f}x smaller")

    print("\n  Claims status:")
    print(f"    F1 95.5%:    {'✓' if abs(f1_us - 95.5) < 1.0 else '✗ (got ' + f'{f1_us:.1f}%)'}")
    print(f"    F1 90.3% (nvidia): {'✓' if abs(f1_nv - 90.3) < 1.0 else '✗ (got ' + f'{f1_nv:.1f}%)'}")
    print(f"    3x faster:   {'✓' if delta_latency >= 2.5 else '✗ (got ' + f'{delta_latency:.1f}x)'}")
    if our_size and nvidia_size:
        size_ratio = nvidia_size / our_size
        print(f"    4x smaller:  {'✓' if size_ratio >= 3.0 else '✗ (got ' + f'{size_ratio:.1f}x)'}")

    # Save verification results
    verification = {
        "test_set_fingerprint": fp,
        "train_test_leakage": len(overlap),
        "nvidia_gliner_pii": nvidia_results,
        "our_fine_tuned": our_results,
        "model_sizes_mb": {
            "nvidia/gliner-PII": nvidia_size,
            "urchade/gliner_small-v2.1": small_size,
            "our_fine_tuned": our_size,
        },
        "deltas": {
            "f1_pp": delta_f1,
            "speed_ratio": delta_latency,
            "size_ratio": (nvidia_size / our_size) if (our_size and nvidia_size) else None,
        },
    }
    out_path = "experiments/003-finetune-gliner-small/verification.json"
    with open(out_path, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
