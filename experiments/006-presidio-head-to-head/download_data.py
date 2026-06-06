#!/usr/bin/env python3
"""Download the eval sample for experiment 006.

ai4privacy/pii-masking-300k validation split, English only, seeded sample.
This is OUT-OF-DISTRIBUTION for our fine-tuned GLiNER (trained on Nemotron-PII)
— that's the point: every prior experiment was in-distribution Nemotron.

Output: data/sample.jsonl  {id, text, spans: [{start, end, label, value}]}
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    print("Loading ai4privacy/pii-masking-300k validation split...")
    ds = load_dataset("ai4privacy/pii-masking-300k", split="validation")
    print(f"  {len(ds)} records total")

    english = ds.filter(lambda r: r["language"].lower().startswith("en"))
    print(f"  {len(english)} English records (language field values: "
          f"{Counter(english['language']).most_common(5)})")

    rng = random.Random(SEED)
    indices = rng.sample(range(len(english)), args.n)
    sample = english.select(indices)

    label_counts: Counter = Counter()
    out_path = out_dir / "sample.jsonl"
    with out_path.open("w") as f:
        for rec in sample:
            spans = [
                {"start": m["start"], "end": m["end"],
                 "label": m["label"], "value": m["value"]}
                for m in rec["privacy_mask"]
            ]
            label_counts.update(s["label"] for s in spans)
            f.write(json.dumps({
                "id": rec["id"],
                "text": rec["source_text"],
                "spans": spans,
            }) + "\n")

    total_spans = sum(label_counts.values())
    print(f"\nWrote {args.n} records, {total_spans} ground-truth spans -> {out_path}")
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:24s} {count}")

    (out_dir / "label_distribution.json").write_text(
        json.dumps({"seed": SEED, "n": args.n, "total_spans": total_spans,
                    "labels": dict(label_counts.most_common())}, indent=2))


if __name__ == "__main__":
    main()
