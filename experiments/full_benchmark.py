#!/usr/bin/env python3
"""
Full PII detection benchmark matrix.

Taggers (label quality):
  - GLiNER-PII (local, zero-shot, 570MB)
  - qwen3:1.7b (local Ollama)
  - qwen3.7-max (OpenRouter cloud)
  - Claude Sonnet (Anthropic cloud)

Fine-tuned production models (trained on GLiNER labels):
  - bert-base-uncased (generic, 110M params)
  - dmis-lab/biobert-base-cased-v1.2 (biomedical, 110M params)
  - medicalai/ClinicalBERT (clinical notes, 110M params)

All evaluated against Nemotron-PII ground truth.

Usage:
  source server/.env  # for OPENROUTER_API_KEY, ANTHROPIC_API_KEY
  python3 experiments/full_benchmark.py --limit 50
"""

import ast
import json
import os
import sys
import time
import random
import subprocess
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report, f1_score

# ─── Config ──────────────────────────────────────────────────────

LIMIT = 50
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 8
LR = 3e-5

for i, arg in enumerate(sys.argv):
    if arg == '--limit' and i + 1 < len(sys.argv):
        LIMIT = int(sys.argv[i + 1])

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

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

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

LLM_SYSTEM_PROMPT = """You are a PII detection engine. Extract all personally identifiable information from text.

Return JSON: {"entities": [{"type": "...", "value": "..."}]}

Entity types: person_name, organization, location, date_of_birth, date, national_id, medical_record, insurance_id, email, phone, medical_info, credential, financial_id

Rules:
- "value" must be copied EXACTLY from the input (verbatim)
- Find ALL people, dates, IDs, locations, medical info
- Do NOT include: medical/scientific terms, field labels, public knowledge
- If no PII found, return {"entities": []}"""


# ─── Data loading ─────────────────────────────────────────────────

def load_nemotron(limit):
    cache = 'experiments/output/nemotron_healthcare_cache.jsonl'
    records = []
    with open(cache) as f:
        for line in f:
            records.append(json.loads(line))
    random.shuffle(records)
    return records[:limit]


# ─── Eval helpers ─────────────────────────────────────────────────

def spans_overlap(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def evaluate_spans(gold_records, pred_spans_list):
    per_type = {}
    overall = {'tp': 0, 'fp': 0, 'fn': 0}

    for gold_rec, pred_spans in zip(gold_records, pred_spans_list):
        gold = gold_rec['goldSpans']

        gold_matched = set()
        pred_matched = set()

        for pi, p in enumerate(pred_spans):
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

        for pi, p in enumerate(pred_spans):
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


def calc_f1(s):
    prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
    rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f


def print_report(name, overall, per_type, elapsed, n_records):
    prec, rec, f1 = calc_f1(overall)
    print(f"\n  {name}")
    print(f"  {'─' * 68}")
    print(f"  {'Entity':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")

    sorted_types = sorted(per_type.items(), key=lambda x: -(x[1]['tp'] + x[1]['fn']))
    for type_name, stats in sorted_types[:8]:
        p, r, f = calc_f1(stats)
        print(f"  {type_name:<20} {p*100:>7.1f}% {r*100:>7.1f}% {f*100:>7.1f}% {stats['tp']:>5} {stats['fp']:>5} {stats['fn']:>5}")

    print(f"  {'─' * 68}")
    print(f"  {'OVERALL':<20} {prec*100:>7.1f}% {rec*100:>7.1f}% {f1*100:>7.1f}% {overall['tp']:>5} {overall['fp']:>5} {overall['fn']:>5}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_records:.2f}s/record)")
    return prec, rec, f1


# ─── Tagger: GLiNER-PII ──────────────────────────────────────────

def run_gliner(texts):
    from gliner import GLiNER
    print("  Loading model...")
    model = GLiNER.from_pretrained("nvidia/gliner-PII")

    all_spans = []
    start = time.time()

    for i, text in enumerate(texts):
        entities = model.predict_entities(text, GLINER_ENTITY_TYPES, threshold=0.3)
        spans = []
        for e in entities:
            mapped = GLINER_LABEL_MAP.get(e['label'])
            if mapped:
                spans.append({'start': e['start'], 'end': e['end'], 'text': e['text'], 'label': mapped})
        all_spans.append(spans)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(texts)}]")

    return all_spans, time.time() - start


# ─── Tagger: Cloud LLM (OpenRouter / Anthropic) ──────────────────

def run_cloud_llm(texts, model_id, api_key, endpoint, is_anthropic=False):
    import urllib.request

    all_spans = []
    start = time.time()
    cost = 0.0

    for i, text in enumerate(texts):
        user_msg = f"Entity types: person_name, organization, location, date_of_birth, date, national_id, medical_record, insurance_id, email, phone, medical_info, credential, financial_id\n\nText:\n\"\"\"\n{text}\n\"\"\""

        if is_anthropic:
            body = json.dumps({
                "model": model_id,
                "max_tokens": 2048,
                "messages": [
                    {"role": "user", "content": LLM_SYSTEM_PROMPT + "\n\n" + user_msg}
                ],
            }).encode()
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        else:
            body = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }).encode()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            if is_anthropic:
                content = data.get('content', [{}])[0].get('text', '[]')
                usage = data.get('usage', {})
                # rough cost estimate for sonnet
                cost += (usage.get('input_tokens', 0) * 3 + usage.get('output_tokens', 0) * 15) / 1_000_000
            else:
                content = data['choices'][0]['message']['content']
                usage = data.get('usage', {})
                cost += usage.get('total_cost', 0) or 0

            # Parse entities
            try:
                # Try to extract JSON from content (might have markdown wrapping)
                json_str = content
                if '```' in content:
                    json_str = content.split('```')[1]
                    if json_str.startswith('json'):
                        json_str = json_str[4:]
                parsed = json.loads(json_str)
                entities = parsed if isinstance(parsed, list) else parsed.get('entities', parsed.get('results', []))
            except:
                entities = []

            spans = []
            for e in (entities if isinstance(entities, list) else []):
                if not isinstance(e, dict) or 'value' not in e or 'type' not in e:
                    continue
                val = e['value']
                etype = e['type']
                search_from = 0
                while True:
                    idx = text.find(val, search_from)
                    if idx == -1:
                        break
                    spans.append({'start': idx, 'end': idx + len(val), 'text': val, 'label': etype})
                    search_from = idx + len(val)
            all_spans.append(spans)

        except Exception as ex:
            print(f"  [{i+1}] ERROR: {ex}")
            all_spans.append([])

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(texts)}]")

    elapsed = time.time() - start
    return all_spans, elapsed, cost


# ─── Fine-tune NER ────────────────────────────────────────────────

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
    def __init__(self, records, tokenizer, label2id):
        self.items = []
        for rec in records:
            encoding, bio_labels = spans_to_bio(rec["text"], rec["spans"], tokenizer)
            label_ids = [label2id.get(l, label2id["O"]) for l in bio_labels]
            self.items.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "labels": torch.tensor(label_ids, dtype=torch.long),
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def finetune_and_eval(model_name, train_records, test_records, entity_types):
    bio_labels = ["O"]
    for et in entity_types:
        bio_labels += [f"B-{et}", f"I-{et}"]
    label2id = {l: i for i, l in enumerate(bio_labels)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NERDataset(train_records, tokenizer, label2id)
    test_dataset = NERDataset(test_records, tokenizer, label2id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(bio_labels), id2label=id2label, label2id=label2id,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count/1e6:.0f}M params, device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
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
            total_loss += outputs.loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} loss: {avg_loss:.4f}")

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
    report = classification_report(all_labels, all_preds, digits=4)
    avg_ms = sum(inf_times) / len(inf_times)

    return f1, report, train_time, avg_ms, param_count


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Full PII Detection Benchmark Matrix                        ║")
    print("║  Taggers × Fine-tuned Models × Nemotron-PII Ground Truth    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Load all records
    all_records = load_nemotron(9999)
    print(f"Total healthcare records: {len(all_records)}")

    # Use LIMIT records for tagger eval
    eval_records = all_records[:LIMIT]
    texts = [r['text'] for r in eval_records]

    gold_count = sum(len(r['goldSpans']) for r in eval_records)
    print(f"Tagger eval: {len(eval_records)} records, {gold_count} gold entities")

    # For fine-tuning: use ALL records, 80/20 split
    random.shuffle(all_records)
    split = int(len(all_records) * 0.8)
    ft_train = [{'text': r['text'], 'spans': r['goldSpans']} for r in all_records[:split]]
    ft_test = [{'text': r['text'], 'spans': r['goldSpans']} for r in all_records[split:]]
    print(f"Fine-tune: {len(ft_train)} train, {len(ft_test)} test")

    entity_types = sorted(set(s['label'] for r in all_records for s in r['goldSpans']))
    print(f"Entity types: {len(entity_types)}")

    results = {}

    # ════════════════════════════════════════════════════════════
    # PART 1: TAGGER BENCHMARK
    # ════════════════════════════════════════════════════════════

    print(f"\n{'═' * 72}")
    print("PART 1: TAGGER BENCHMARK (label quality for fine-tuning)")
    print(f"{'═' * 72}")

    # ── GLiNER-PII ──
    print(f"\n{'─' * 72}")
    print("1. GLiNER-PII (nvidia/gliner-PII, zero-shot, local)")
    gliner_spans, gliner_time = run_gliner(texts)
    gliner_overall, gliner_per = evaluate_spans(eval_records, gliner_spans)
    _, _, gliner_f1 = print_report("GLiNER-PII", gliner_overall, gliner_per, gliner_time, len(texts))
    results['gliner'] = {'f1': gliner_f1, 'time': gliner_time, 'cost': 0}

    # ── qwen3.7-max via OpenRouter ──
    if OPENROUTER_API_KEY:
        print(f"\n{'─' * 72}")
        print("2. qwen3.7-max (OpenRouter cloud)")
        qwen_spans, qwen_time, qwen_cost = run_cloud_llm(
            texts, "qwen/qwen3.7-max", OPENROUTER_API_KEY,
            "https://openrouter.ai/api/v1/chat/completions"
        )
        qwen_overall, qwen_per = evaluate_spans(eval_records, qwen_spans)
        _, _, qwen_f1 = print_report("qwen3.7-max (OpenRouter)", qwen_overall, qwen_per, qwen_time, len(texts))
        results['qwen37max'] = {'f1': qwen_f1, 'time': qwen_time, 'cost': qwen_cost}
        print(f"  Cost: ${qwen_cost:.4f}")
    else:
        print("\n  Skipping qwen3.7-max (no OPENROUTER_API_KEY)")

    # ── Claude Sonnet via Anthropic ──
    if ANTHROPIC_API_KEY:
        print(f"\n{'─' * 72}")
        print("3. Claude Sonnet (Anthropic cloud)")
        claude_spans, claude_time, claude_cost = run_cloud_llm(
            texts, "claude-sonnet-4-20250514", ANTHROPIC_API_KEY,
            "https://api.anthropic.com/v1/messages", is_anthropic=True
        )
        claude_overall, claude_per = evaluate_spans(eval_records, claude_spans)
        _, _, claude_f1 = print_report("Claude Sonnet", claude_overall, claude_per, claude_time, len(texts))
        results['claude_sonnet'] = {'f1': claude_f1, 'time': claude_time, 'cost': claude_cost}
        print(f"  Cost: ~${claude_cost:.4f}")
    else:
        print("\n  Skipping Claude Sonnet (no ANTHROPIC_API_KEY)")

    # ════════════════════════════════════════════════════════════
    # PART 2: FINE-TUNED MODEL BENCHMARK
    # ════════════════════════════════════════════════════════════

    print(f"\n\n{'═' * 72}")
    print(f"PART 2: FINE-TUNED MODEL BENCHMARK ({len(ft_train)} train, {len(ft_test)} test)")
    print(f"{'═' * 72}")

    bert_models = [
        ("bert-base-uncased", "Generic BERT"),
        ("dmis-lab/biobert-base-cased-v1.2", "BioBERT (biomedical)"),
        ("medicalai/ClinicalBERT", "ClinicalBERT (clinical notes)"),
    ]

    for model_name, desc in bert_models:
        print(f"\n{'─' * 72}")
        print(f"Fine-tuning: {desc} ({model_name})")
        try:
            f1, report, train_time, avg_ms, params = finetune_and_eval(
                model_name, ft_train, ft_test, entity_types
            )
            print(f"\n{report}")
            print(f"  F1: {f1*100:.1f}%, Train: {train_time:.0f}s, Inference: {avg_ms:.1f}ms/record, Params: {params/1e6:.0f}M")
            results[model_name] = {'f1': f1, 'train_time': train_time, 'inference_ms': avg_ms, 'params': params}
        except Exception as ex:
            print(f"  FAILED: {ex}")
            results[model_name] = {'f1': 0, 'error': str(ex)}

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════

    print(f"\n\n{'═' * 72}")
    print("FULL BENCHMARK MATRIX")
    print(f"{'═' * 72}")
    print(f"\n  TAGGERS (label quality, evaluated on {len(eval_records)} records):")
    print(f"  {'Method':<40} {'F1':>8} {'Speed':>12} {'Cost':>10}")
    print(f"  {'─' * 72}")

    for name, key in [
        ("GLiNER-PII (local, 570MB)", "gliner"),
        ("qwen3.7-max (OpenRouter)", "qwen37max"),
        ("Claude Sonnet (Anthropic)", "claude_sonnet"),
    ]:
        if key in results:
            r = results[key]
            speed = f"{r['time']:.1f}s" if r['time'] < 60 else f"{r['time']/60:.1f}m"
            cost = f"${r['cost']:.4f}" if r.get('cost', 0) > 0 else "free"
            print(f"  {name:<40} {r['f1']*100:>7.1f}% {speed:>12} {cost:>10}")

    print(f"\n  FINE-TUNED MODELS (trained on {len(ft_train)} records, tested on {len(ft_test)}):")
    print(f"  {'Model':<40} {'F1':>8} {'Inference':>12} {'Params':>10}")
    print(f"  {'─' * 72}")

    for model_name, desc in bert_models:
        if model_name in results and results[model_name].get('f1', 0) > 0:
            r = results[model_name]
            print(f"  {desc:<40} {r['f1']*100:>7.1f}% {r['inference_ms']:>10.1f}ms {r['params']/1e6:>8.0f}M")

    print(f"\n  GLiNER-PII for reference:       90%+ F1, ~200ms/record, 570MB")
    print(f"  Fine-tuned target:              90%+ F1,  ~9ms/record,  50MB")

    # Save
    with open('experiments/output/full_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to experiments/output/full_benchmark_results.json")


if __name__ == '__main__':
    main()
