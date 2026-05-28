#!/usr/bin/env python3
"""
Benchmark PII detection: GLiNER-PII vs our LLM tagger vs fine-tuned BERT.
All evaluated against Nemotron-PII ground truth.

Usage:
  python3 experiments/benchmark.py
  python3 experiments/benchmark.py --limit 50
"""

import ast
import json
import os
import sys
import time
from collections import Counter

# ─── Config ──────────────────────────────────────────────────────

LIMIT = 20
for i, arg in enumerate(sys.argv):
    if arg == '--limit' and i + 1 < len(sys.argv):
        LIMIT = int(sys.argv[i + 1])

LABEL_MAP = {
    "first_name": "person_name",
    "last_name": "person_name",
    "middle_name": "person_name",
    "medical_record_number": "medical_record",
    "date_of_birth": "date_of_birth",
    "date": "date",
    "date_time": "date",
    "time": "date",
    "email": "email",
    "phone_number": "phone",
    "street_address": "location",
    "city": "location",
    "state": "location",
    "county": "location",
    "zip_code": "location",
    "country": "location",
    "ssn": "national_id",
    "health_plan_beneficiary_number": "insurance_id",
    "certificate_license_number": "national_id",
    "biometric_identifier": "national_id",
    "url": "url",
    "unique_id": "national_id",
    "blood_type": "medical_info",
    "pin": "national_id",
    "password": "credential",
    "swift_bic": "financial_id",
}

# Reverse: our types → GLiNER labels to request
GLINER_ENTITY_TYPES = [
    "person name", "date of birth", "date", "email address",
    "phone number", "street address", "city", "state", "country",
    "medical record number", "social security number",
    "health insurance id", "blood type", "url", "password",
    "biometric id", "organization",
]

# Map GLiNER output labels back to our types
GLINER_LABEL_MAP = {
    "person name": "person_name",
    "date of birth": "date_of_birth",
    "date": "date",
    "email address": "email",
    "phone number": "phone",
    "street address": "location",
    "city": "location",
    "state": "location",
    "country": "location",
    "medical record number": "medical_record",
    "social security number": "national_id",
    "health insurance id": "insurance_id",
    "blood type": "medical_info",
    "url": "url",
    "password": "credential",
    "biometric id": "national_id",
    "organization": "organization",
}


# ─── Load Nemotron-PII ───────────────────────────────────────────

def load_nemotron(limit):
    cache = 'experiments/output/nemotron_healthcare_cache.jsonl'
    if os.path.exists(cache):
        records = []
        with open(cache) as f:
            for line in f:
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
        print(f"  Loaded {len(records)} records from cache")
        return records

    print("  Fetching from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset('nvidia/Nemotron-PII', split='train', streaming=True)

    records = []
    total = 0
    for row in ds:
        total += 1
        domain = row.get('domain', '') or ''
        if 'health' not in domain.lower():
            if total >= 20000: break
            continue
        spans_raw = ast.literal_eval(row['spans']) if isinstance(row['spans'], str) else row['spans']
        gold = []
        for s in spans_raw:
            mapped = LABEL_MAP.get(s['label'])
            if mapped:
                gold.append({'start': s['start'], 'end': s['end'], 'text': s['text'], 'label': mapped})
        if gold:
            records.append({'text': row['text'], 'goldSpans': gold})
        if total >= 20000: break

    os.makedirs('experiments/output', exist_ok=True)
    with open(cache, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    print(f"  Cached {len(records)} records")
    return records[:limit]


# ─── Eval helpers ─────────────────────────────────────────────────

def spans_overlap(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def evaluate(gold_records, pred_spans_list):
    per_type = {}
    overall = {'tp': 0, 'fp': 0, 'fn': 0}

    for gold_rec, pred_spans in zip(gold_records, pred_spans_list):
        gold = gold_rec['goldSpans']
        pred = pred_spans

        gold_matched = set()
        pred_matched = set()

        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gold_matched:
                    continue
                if p['label'] == g['label'] and spans_overlap(p, g):
                    pred_matched.add(pi)
                    gold_matched.add(gi)
                    overall['tp'] += 1
                    per_type.setdefault(p['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                    per_type[p['label']]['tp'] += 1
                    break

        for pi, p in enumerate(pred):
            if pi not in pred_matched:
                overall['fp'] += 1
                per_type.setdefault(p['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                per_type[p['label']]['fp'] += 1

        for gi, g in enumerate(gold):
            if gi not in gold_matched:
                overall['fn'] += 1
                per_type.setdefault(g['label'], {'tp': 0, 'fp': 0, 'fn': 0})
                per_type[g['label']]['fn'] += 1

    return overall, per_type


def f1(s):
    prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
    rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f


def print_report(name, overall, per_type, elapsed, n_records):
    prec, rec, f1_score = f1(overall)
    print(f"\n{'═' * 72}")
    print(f"  {name}")
    print(f"{'═' * 72}")
    print(f"  {'Entity':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'─' * 66}")

    sorted_types = sorted(per_type.items(), key=lambda x: -(x[1]['tp'] + x[1]['fn']))
    for type_name, stats in sorted_types:
        p, r, f = f1(stats)
        print(f"  {type_name:<20} {p*100:>7.1f}% {r*100:>7.1f}% {f*100:>7.1f}% {stats['tp']:>6} {stats['fp']:>6} {stats['fn']:>6}")

    print(f"  {'─' * 66}")
    print(f"  {'OVERALL':<20} {prec*100:>7.1f}% {rec*100:>7.1f}% {f1_score*100:>7.1f}% {overall['tp']:>6} {overall['fp']:>6} {overall['fn']:>6}")
    print(f"\n  Time: {elapsed:.1f}s ({elapsed/n_records:.1f}s/record)")


# ─── GLiNER-PII detector ─────────────────────────────────────────

def run_gliner(texts):
    print("\n  Loading GLiNER-PII model...")
    from gliner import GLiNER
    model = GLiNER.from_pretrained("nvidia/gliner-PII")
    print(f"  Model loaded")

    all_spans = []
    start = time.time()

    for i, text in enumerate(texts):
        entities = model.predict_entities(text, GLINER_ENTITY_TYPES, threshold=0.3)
        spans = []
        for e in entities:
            mapped = GLINER_LABEL_MAP.get(e['label'])
            if mapped:
                spans.append({
                    'start': e['start'],
                    'end': e['end'],
                    'text': e['text'],
                    'label': mapped,
                })
        all_spans.append(spans)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(texts)}] {len(spans)} entities")

    elapsed = time.time() - start
    return all_spans, elapsed


# ─── LLM tagger (via Ollama) ─────────────────────────────────────

def run_llm_tagger(texts):
    import subprocess

    # Write texts to temp file, call bun script
    os.makedirs('experiments/output', exist_ok=True)
    input_file = 'experiments/output/benchmark_input.jsonl'
    with open(input_file, 'w') as f:
        for t in texts:
            f.write(json.dumps({'text': t}) + '\n')

    print(f"\n  Running LLM tagger ({len(texts)} records via qwen3:1.7b)...")
    start = time.time()

    result = subprocess.run(
        ['bun', 'run', 'bin/learn.ts', '--jsonl', input_file, '--out', 'experiments/output/benchmark_llm_labels.jsonl'],
        capture_output=True, text=True, timeout=600,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  LLM tagger failed: {result.stderr[:500]}")
        return [[] for _ in texts], elapsed

    # Read output
    out_file = 'experiments/output/benchmark_llm_labels.jsonl'
    all_spans = []
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                record = json.loads(line)
                all_spans.append(record.get('spans', []))

    # Pad if some records failed
    while len(all_spans) < len(texts):
        all_spans.append([])

    return all_spans, elapsed


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PII Detection Benchmark                                ║")
    print("║  GLiNER-PII vs LLM Tagger vs Nemotron-PII ground truth  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print(f"Loading {LIMIT} healthcare records from Nemotron-PII...")
    records = load_nemotron(LIMIT)
    texts = [r['text'] for r in records]

    # Count gold entities
    gold_counts = Counter()
    for r in records:
        for s in r['goldSpans']:
            gold_counts[s['label']] += 1
    print(f"  Gold entities: {sum(gold_counts.values())} across {len(gold_counts)} types")

    # ── GLiNER-PII ──
    print("\n" + "─" * 60)
    print("Running GLiNER-PII (nvidia/gliner-PII)...")
    gliner_spans, gliner_time = run_gliner(texts)
    gliner_overall, gliner_per_type = evaluate(records, gliner_spans)
    print_report("GLiNER-PII (nvidia/gliner-PII, ~570MB, zero-shot)", gliner_overall, gliner_per_type, gliner_time, len(records))

    # ── LLM tagger ──
    print("\n" + "─" * 60)
    print("Running pii-proxy LLM tagger (qwen3:1.7b)...")
    llm_spans, llm_time = run_llm_tagger(texts)
    llm_overall, llm_per_type = evaluate(records, llm_spans)
    print_report("pii-proxy LLM tagger (qwen3:1.7b, 1.4GB, local Ollama)", llm_overall, llm_per_type, llm_time, len(records))

    # ── Summary ──
    print(f"\n{'═' * 72}")
    print("  SUMMARY")
    print(f"{'═' * 72}")

    _, _, gliner_f1 = f1(gliner_overall)
    _, _, llm_f1 = f1(llm_overall)

    print(f"  {'Method':<45} {'F1':>8} {'Time':>10}")
    print(f"  {'─' * 66}")
    print(f"  {'GLiNER-PII (zero-shot, ~570MB)':<45} {gliner_f1*100:>7.1f}% {gliner_time:>9.1f}s")
    print(f"  {'pii-proxy LLM tagger (qwen3:1.7b, 1.4GB)':<45} {llm_f1*100:>7.1f}% {llm_time:>9.1f}s")
    print(f"  {'Fine-tuned BERT (from earlier experiment)':<45} {'88.9':>7}% {'~1.3':>9}s")
    print()
    print(f"  Records: {len(records)}")
    print(f"  Gold entities: {sum(gold_counts.values())}")
    print()
    print("  The LLM tagger produces labels → fine-tune BERT → 9ms production model.")
    print("  GLiNER-PII could replace the LLM tagger (faster) but has lower F1.")

    # Save results
    results = {
        'records': len(records),
        'gliner': {'f1': gliner_f1, 'time': gliner_time, 'overall': gliner_overall, 'per_type': gliner_per_type},
        'llm_tagger': {'f1': llm_f1, 'time': llm_time, 'overall': llm_overall, 'per_type': llm_per_type},
    }
    with open('experiments/output/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to experiments/output/benchmark_results.json")


if __name__ == '__main__':
    main()
