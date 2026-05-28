#!/usr/bin/env python3
"""
Methodologically clean benchmark.

Same held-out test set across all methods:
  1. BERT trained on GOLD labels (upper bound)
  2. BERT trained on GLINER labels (real learn→compile pipeline)
  3. GLiNER-PII zero-shot (baseline)
  4. Claude Sonnet zero-shot (cloud baseline, if API key)

This is the honest comparison. Gap between #1 and #2 shows the cost
of automated labeling. Gap between #2 and #3 shows the value of
compilation (and #2 should match or beat #3).

Usage:
  python3 experiments/clean_benchmark.py
"""

import ast, json, os, sys, time, random
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import f1_score

# ─── Config ──────────────────────────────────────────────────────

SEED = 42
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 8
LR = 3e-5
TEST_SIZE = 100  # held-out test set used for ALL methods

random.seed(SEED)
torch.manual_seed(SEED)

LABEL_MAP = {
    "first_name": "person_name", "last_name": "person_name", "middle_name": "person_name",
    "medical_record_number": "medical_record", "date_of_birth": "date_of_birth",
    "date": "date", "date_time": "date", "time": "date",
    "email": "email", "phone_number": "phone",
    "street_address": "location", "city": "location", "state": "location",
    "county": "location", "zip_code": "location", "country": "location",
    "ssn": "national_id", "health_plan_beneficiary_number": "insurance_id",
    "certificate_license_number": "national_id", "biometric_identifier": "national_id",
    "url": "url", "unique_id": "national_id", "blood_type": "medical_info",
    "pin": "national_id", "password": "credential", "swift_bic": "financial_id",
}

GLINER_ENTITY_TYPES = [
    "person name", "date of birth", "date", "email address",
    "phone number", "street address", "city", "state", "country",
    "medical record number", "social security number",
    "health insurance id", "blood type", "url", "password",
    "biometric id", "organization",
]

GLINER_LABEL_MAP = {
    "person name": "person_name", "date of birth": "date_of_birth",
    "date": "date", "email address": "email", "phone number": "phone",
    "street address": "location", "city": "location", "state": "location",
    "country": "location", "medical record number": "medical_record",
    "social security number": "national_id", "health insurance id": "insurance_id",
    "blood type": "medical_info", "url": "url", "password": "credential",
    "biometric id": "national_id", "organization": "organization",
}


# ─── Data loading ─────────────────────────────────────────────────

def load_nemotron():
    cache = 'experiments/output/nemotron_healthcare_cache.jsonl'
    records = []
    with open(cache) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ─── BIO conversion + Dataset ────────────────────────────────────

def spans_to_bio(text, spans, tokenizer, max_len=MAX_LEN):
    encoding = tokenizer(text, max_length=max_len, truncation=True, padding="max_length",
                         return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding["offset_mapping"][0].tolist()
    labels = ["O"] * len(offsets)
    for span in sorted(spans, key=lambda s: s['start']):
        first = True
        for i, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0: continue
            if te <= span['start']: continue
            if ts >= span['end']: break
            labels[i] = f"B-{span['label']}" if first else f"I-{span['label']}"
            first = False
    return encoding, labels


class NERDataset(Dataset):
    def __init__(self, records, tokenizer, label2id, spans_key='spans'):
        self.items = []
        for rec in records:
            encoding, bio_labels = spans_to_bio(rec["text"], rec[spans_key], tokenizer)
            label_ids = [label2id.get(l, label2id["O"]) for l in bio_labels]
            self.items.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "labels": torch.tensor(label_ids, dtype=torch.long),
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


# ─── GLiNER tagger ─────────────────────────────────────────────────

def gliner_tag(records, text_key='text'):
    """Tag records with GLiNER-PII. Returns list of [{start, end, text, label}]."""
    from gliner import GLiNER
    print("  Loading GLiNER-PII...")
    model = GLiNER.from_pretrained("nvidia/gliner-PII")

    all_spans = []
    for i, rec in enumerate(records):
        text = rec[text_key]
        entities = model.predict_entities(text, GLINER_ENTITY_TYPES, threshold=0.3)
        spans = []
        for e in entities:
            mapped = GLINER_LABEL_MAP.get(e['label'])
            if mapped:
                spans.append({'start': e['start'], 'end': e['end'], 'text': e['text'], 'label': mapped})
        all_spans.append(spans)
        if (i + 1) % 50 == 0:
            print(f"  Tagged {i+1}/{len(records)}")

    return all_spans


# ─── BERT train + eval ────────────────────────────────────────────

def train_and_eval_bert(model_name, train_records, test_records, entity_types,
                         train_spans_key='spans', test_spans_key='spans'):
    bio_labels = ["O"]
    for et in entity_types:
        bio_labels += [f"B-{et}", f"I-{et}"]
    label2id = {l: i for i, l in enumerate(bio_labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NERDataset(train_records, tokenizer, label2id, train_spans_key)
    test_dataset = NERDataset(test_records, tokenizer, label2id, test_spans_key)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(bio_labels), id2label=id2label, label2id=label2id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    train_time = time.time() - start

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    inf_times = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            t0 = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if device.type == "mps": torch.mps.synchronize()
            inf_times.append((time.perf_counter() - t0) * 1000 / input_ids.shape[0])

            preds = torch.argmax(outputs.logits, dim=-1).cpu()
            for i in range(preds.shape[0]):
                mask = attention_mask[i].cpu()
                pred_seq, label_seq = [], []
                for j in range(len(mask)):
                    if mask[j] == 0: continue
                    pred_seq.append(id2label[preds[i][j].item()])
                    label_seq.append(id2label[batch["labels"][i][j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    f1 = f1_score(all_labels, all_preds)
    avg_ms = sum(inf_times) / len(inf_times)
    return f1, train_time, avg_ms


# ─── Span-level eval (for non-BERT methods) ──────────────────────

def spans_overlap(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def eval_spans(gold_records, pred_spans_list):
    tp, fp, fn = 0, 0, 0
    for rec, pred in zip(gold_records, pred_spans_list):
        gold = rec['goldSpans']
        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm: continue
                if p['label'] == g['label'] and spans_overlap(p, g):
                    pm.add(pi); gm.add(gi)
                    tp += 1
                    break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f = 2 * p * r / (p + r) if p + r else 0
    return f, p, r


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CLEAN Benchmark — All methods on the SAME held-out set     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Load and split — IDENTICAL test set for all methods
    all_records = load_nemotron()
    random.shuffle(all_records)

    test_records = all_records[:TEST_SIZE]
    train_records = all_records[TEST_SIZE:]

    test_records_for_bert = [{'text': r['text'], 'spans': r['goldSpans']} for r in test_records]
    train_records_gold = [{'text': r['text'], 'spans': r['goldSpans']} for r in train_records]

    print(f"Test set:  {len(test_records)} records (same for all methods)")
    print(f"Train set: {len(train_records)} records")

    gold_count_test = sum(len(r['goldSpans']) for r in test_records)
    print(f"Test gold entities: {gold_count_test}\n")

    entity_types = sorted(set(s['label'] for r in all_records for s in r['goldSpans']))
    print(f"Entity types: {len(entity_types)}\n")

    results = {}

    # ═══════════════════════════════════════════════════════════
    # 1. BERT trained on GOLD labels (upper bound)
    # ═══════════════════════════════════════════════════════════
    print("─" * 72)
    print("1. BERT trained on GOLD labels (upper bound)")
    print("   (trains on ground truth, tests on held-out ground truth)")
    print("─" * 72)
    f1, train_time, avg_ms = train_and_eval_bert(
        "bert-base-uncased",
        train_records_gold,
        test_records_for_bert,
        entity_types,
    )
    print(f"  F1: {f1*100:.1f}%, train: {train_time:.0f}s, inference: {avg_ms:.1f}ms\n")
    results['bert_on_gold'] = {'f1': f1, 'train_time': train_time, 'inference_ms': avg_ms}

    # ═══════════════════════════════════════════════════════════
    # 2. BERT trained on GLINER labels (real learn→compile pipeline)
    # ═══════════════════════════════════════════════════════════
    print("─" * 72)
    print("2. BERT trained on GLINER labels (real pipeline)")
    print("   (GLiNER labels training data, BERT trains on those, tests on gold)")
    print("─" * 72)

    print("  Step 1: GLiNER tagging training set...")
    train_gliner_spans = gliner_tag(train_records)
    train_records_gliner = [
        {'text': r['text'], 'spans': spans}
        for r, spans in zip(train_records, train_gliner_spans)
    ]

    print("  Step 2: Fine-tune BERT on GLiNER labels...")
    f1, train_time, avg_ms = train_and_eval_bert(
        "bert-base-uncased",
        train_records_gliner,
        test_records_for_bert,
        entity_types,
    )
    print(f"  F1: {f1*100:.1f}%, train: {train_time:.0f}s, inference: {avg_ms:.1f}ms\n")
    results['bert_on_gliner'] = {'f1': f1, 'train_time': train_time, 'inference_ms': avg_ms}

    # ═══════════════════════════════════════════════════════════
    # 3. GLiNER-PII zero-shot (baseline)
    # ═══════════════════════════════════════════════════════════
    print("─" * 72)
    print("3. GLiNER-PII zero-shot (baseline)")
    print("   (no training, just predict on held-out test set)")
    print("─" * 72)

    start = time.time()
    test_gliner_spans = gliner_tag(test_records)
    gliner_time = time.time() - start
    gliner_f1, _, _ = eval_spans(test_records, test_gliner_spans)
    print(f"  F1: {gliner_f1*100:.1f}%, time: {gliner_time:.1f}s ({gliner_time/len(test_records)*1000:.0f}ms/record)\n")
    results['gliner_zero_shot'] = {'f1': gliner_f1, 'time': gliner_time, 'ms_per_record': gliner_time/len(test_records)*1000}

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("═" * 72)
    print("CLEAN BENCHMARK SUMMARY (same test set, same conditions)")
    print("═" * 72)
    print(f"\n  Test set: {len(test_records)} records, {gold_count_test} gold entities\n")

    print(f"  {'Method':<45} {'F1':>8}  {'Latency':>10}")
    print(f"  {'─' * 68}")
    print(f"  {'1. BERT (trained on GOLD labels)':<45} {results['bert_on_gold']['f1']*100:>7.1f}%  {results['bert_on_gold']['inference_ms']:>8.1f}ms")
    print(f"  {'2. BERT (trained on GLiNER labels)':<45} {results['bert_on_gliner']['f1']*100:>7.1f}%  {results['bert_on_gliner']['inference_ms']:>8.1f}ms")
    print(f"  {'3. GLiNER-PII (zero-shot)':<45} {results['gliner_zero_shot']['f1']*100:>7.1f}%  {results['gliner_zero_shot']['ms_per_record']:>8.0f}ms")
    print()

    gap_gold_vs_gliner = (results['bert_on_gold']['f1'] - results['bert_on_gliner']['f1']) * 100
    gap_gliner_vs_zero = (results['bert_on_gliner']['f1'] - results['gliner_zero_shot']['f1']) * 100

    print(f"  Cost of automated labeling (gold vs GLiNER train):  {gap_gold_vs_gliner:+.1f} F1 points")
    print(f"  Value of compilation (BERT-on-GLiNER vs GLiNER):    {gap_gliner_vs_zero:+.1f} F1 points")
    print()
    print(f"  → If BERT-on-GLiNER beats GLiNER zero-shot, the compile step adds value.")
    print(f"  → If close to BERT-on-gold, the automated pipeline is nearly as good as human labels.")

    with open('experiments/output/clean_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: experiments/output/clean_benchmark_results.json")


if __name__ == '__main__':
    main()
