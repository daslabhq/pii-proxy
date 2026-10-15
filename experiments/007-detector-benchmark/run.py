#!/usr/bin/env python3
"""Run one detector over one dataset; write its raw detections.

    .venv/bin/python run.py --model gliner2-pii --dataset spy
    .venv/bin/python run.py --model pii-proxy-ner --dataset spy --vocab full
    .venv/bin/python run.py --model gliner2-pii --dataset thai --segment-thai

Output: out/<dataset>/<model>[.full][.seg].jsonl, one {id, detections} per
record, plus a .meta.json with the labels asked for, the device and the time.
Scoring (score.py) applies thresholds and label mappings afterwards.
"""

import argparse
import json
import time
from pathlib import Path

from detectors import FLOOR, HF_IDS, detect, load_runner
from mapping import query_labels

HERE = Path(__file__).parent


def main() -> None:
    args = parse_args()
    records = load_records(args.dataset, args.limit)
    labels = labels_for(args.model, args.dataset, args.vocab)
    runner = load_runner(args.model, args.device)
    t0 = time.perf_counter()
    found = detect(runner, [r["text"] for r in records], labels,
                   segment_thai=args.segment_thai, batch_size=args.batch_size)
    seconds = time.perf_counter() - t0
    write_outputs(args, records, labels, found, seconds)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(HF_IDS))
    p.add_argument("--dataset", required=True, choices=["spy", "ai4privacy", "nemotron", "thai"])
    p.add_argument("--vocab", default="dataset", choices=["dataset", "full", "spy7"])
    p.add_argument("--segment-thai", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def load_records(dataset, limit):
    records = [json.loads(line) for line in (HERE / "data" / f"{dataset}.jsonl").open()]
    return records[:limit] if limit else records


def labels_for(model, dataset, vocab):
    if model in ("openai-privacy-filter", "presidio"):
        return []  # fixed label sets; every detection is mapped at scoring time
    return query_labels(model, dataset, vocab)


def output_path(args) -> Path:
    name = args.model + (f".{args.vocab}" if args.vocab != "dataset" else "") + (".seg" if args.segment_thai else "")
    out = HERE / "out" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{name}.jsonl"


def write_outputs(args, records, labels, found, seconds):
    path = output_path(args)
    with path.open("w") as f:
        for rec, dets in zip(records, found):
            f.write(json.dumps({"id": rec["id"], "detections": dets}, ensure_ascii=False) + "\n")
    meta = {"model": args.model, "hf_id": HF_IDS[args.model], "dataset": args.dataset,
            "vocab": args.vocab, "segment_thai": args.segment_thai, "labels": labels,
            "threshold_floor": FLOOR, "device": args.device, "records": len(records),
            "seconds": round(seconds, 1), "ms_per_record": round(1000 * seconds / len(records), 1)}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{args.model} on {args.dataset}: {len(records)} records, "
          f"{meta['ms_per_record']} ms/record on {args.device} -> {path}")


if __name__ == "__main__":
    main()
