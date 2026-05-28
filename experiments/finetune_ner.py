#!/usr/bin/env python3
"""
Fine-tune a small NER model on Nemotron-PII healthcare data.

Downloads healthcare records from NVIDIA's Nemotron-PII dataset,
converts to BIO format, trains a token classifier, and evaluates
on a held-out test set.

Usage:
  python3 experiments/finetune_ner.py
"""

import ast
import json
import os
import random
from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report, f1_score

# ─── Config ──────────────────────────────────────────────────────

MODEL_NAME = "bert-base-uncased"  # start with base BERT, swap to ModernBERT later
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-5
SEED = 42
MAX_RECORDS = 20000  # scan this many records from the dataset
OUT_DIR = "experiments/output"

# Map Nemotron-PII labels to our coarser PII types
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

random.seed(SEED)
torch.manual_seed(SEED)


# ─── Data loading ────────────────────────────────────────────────

def load_healthcare_records(max_records=MAX_RECORDS):
    print(f"Loading Nemotron-PII dataset (scanning up to {max_records} records)...")
    ds = load_dataset("nvidia/Nemotron-PII", split="train", streaming=True)

    records = []
    total = 0
    for row in ds:
        total += 1
        domain = row.get("domain", "") or ""
        if "health" in domain.lower():
            spans = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
            mapped_spans = []
            for s in spans:
                coarse = LABEL_MAP.get(s["label"])
                if coarse:
                    mapped_spans.append({**s, "label": coarse})
            if mapped_spans:
                records.append({"text": row["text"], "spans": mapped_spans})
        if total >= max_records:
            break

    print(f"  Scanned {total} records, found {len(records)} healthcare records with mapped spans")
    return records


# ─── BIO conversion ─────────────────────────────────────────────

def spans_to_bio(text, spans, tokenizer, max_len=MAX_LEN):
    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = encoding["offset_mapping"][0].tolist()
    labels = ["O"] * len(offsets)

    sorted_spans = sorted(spans, key=lambda s: s["start"])

    for span in sorted_spans:
        s_start, s_end = span["start"], span["end"]
        label = span["label"]
        first_token = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue  # special token
            if tok_end <= s_start:
                continue
            if tok_start >= s_end:
                break
            if first_token:
                labels[i] = f"B-{label}"
                first_token = False
            else:
                labels[i] = f"I-{label}"

    return encoding, labels


# ─── Dataset class ───────────────────────────────────────────────

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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ─── Training ────────────────────────────────────────────────────

def train(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, batch {batch_idx+1}/{len(train_loader)}, loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


# ─── Evaluation ──────────────────────────────────────────────────

def evaluate(model, test_loader, id2label, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu()

            for i in range(preds.shape[0]):
                mask = attention_mask[i].cpu()
                pred_seq = []
                label_seq = []
                for j in range(len(mask)):
                    if mask[j] == 0:
                        continue
                    pred_label = id2label[preds[i][j].item()]
                    true_label = id2label[labels[i][j].item()]
                    # Skip special tokens (labeled O at position 0 and end)
                    if true_label == "O" and pred_label == "O":
                        pred_seq.append(pred_label)
                        label_seq.append(true_label)
                    else:
                        pred_seq.append(pred_label)
                        label_seq.append(true_label)
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    return all_preds, all_labels


# ─── Main ────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    records = load_healthcare_records()

    # Collect all labels
    label_counts = Counter()
    for r in records:
        for s in r["spans"]:
            label_counts[s["label"]] += 1

    print(f"\nEntity distribution ({sum(label_counts.values())} total spans):")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    # Build label set
    entity_types = sorted(set(s["label"] for r in records for s in r["spans"]))
    bio_labels = ["O"]
    for et in entity_types:
        bio_labels.append(f"B-{et}")
        bio_labels.append(f"I-{et}")

    label2id = {l: i for i, l in enumerate(bio_labels)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"\nBIO labels: {len(bio_labels)} ({len(entity_types)} entity types)")

    # Train/test split
    random.shuffle(records)
    split = int(len(records) * 0.8)
    train_records = records[:split]
    test_records = records[split:]
    print(f"\nTrain: {len(train_records)} records, Test: {len(test_records)} records")

    # Tokenizer + datasets
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Converting to BIO format...")
    train_dataset = NERDataset(train_records, tokenizer, label2id)
    test_dataset = NERDataset(test_records, tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Loading model: {MODEL_NAME}")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(bio_labels),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # Train
    print(f"\n{'='*60}")
    print(f"Training for {EPOCHS} epochs")
    print(f"{'='*60}")

    for epoch in range(EPOCHS):
        avg_loss = train(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {avg_loss:.4f}")

        # Eval after each epoch
        preds, labels = evaluate(model, test_loader, id2label, device)
        f1 = f1_score(labels, preds)
        print(f"  Test F1: {f1:.4f}")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation on held-out test set")
    print(f"{'='*60}\n")

    preds, labels = evaluate(model, test_loader, id2label, device)
    report = classification_report(labels, preds, digits=4)
    print(report)

    # Save results
    results = {
        "model": MODEL_NAME,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "epochs": EPOCHS,
        "entity_types": entity_types,
        "report": report,
    }

    with open(f"{OUT_DIR}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    model.save_pretrained(f"{OUT_DIR}/model")
    tokenizer.save_pretrained(f"{OUT_DIR}/model")
    print(f"\nModel saved to {OUT_DIR}/model")

    # Inference speed test
    print(f"\n{'='*60}")
    print("Inference speed test")
    print(f"{'='*60}")

    model.eval()
    test_text = "Patient Marcus Weber, DOB 15.03.1987, MRN MRN-2024-08391. Treated by Dr. Sarah Chen at Universitätsklinikum Heidelberg. Insurance: TK 109876543. Contact: marcus.weber@gmail.com"

    import time
    encoding = tokenizer(test_text, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(**encoding)

    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**encoding)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    avg_ms = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    print(f"\n  Text: \"{test_text[:80]}...\"")
    print(f"  Avg: {avg_ms:.1f}ms, P50: {p50:.1f}ms, P99: {p99:.1f}ms")
    print(f"  (vs ~30,000ms for qwen3:1.7b LLM detection)")

    # Show what the model detects on this text
    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu()
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].cpu())

    print(f"\n  Detected entities:")
    current_entity = None
    current_text = ""
    for tok, pred_id in zip(tokens, preds):
        label = id2label[pred_id.item()]
        if label.startswith("B-"):
            if current_entity:
                print(f"    {current_entity}: \"{current_text}\"")
            current_entity = label[2:]
            current_text = tok.replace("##", "")
        elif label.startswith("I-") and current_entity:
            current_text += tok.replace("##", "")
        else:
            if current_entity:
                print(f"    {current_entity}: \"{current_text}\"")
                current_entity = None
                current_text = ""

    if current_entity:
        print(f"    {current_entity}: \"{current_text}\"")


if __name__ == "__main__":
    main()
