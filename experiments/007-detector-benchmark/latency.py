#!/usr/bin/env python3
"""Per-text detection latency on CPU, batch size 1, SPY documents (~450 words).

Each GLiNER model is asked for its full native vocabulary — what a pii-proxy
detector would ask for — since GLiNER puts the labels in front of the text and
every label costs encoder time.

    .venv/bin/python latency.py --model pii-proxy-ner --threads 4

modal_run.py --latency calls measure() on a 2-core / 4 GB container — the size
of the instance a server would run it on. Run directly for a laptop.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from detectors import detect, load_runner
from mapping import query_labels

HERE = Path(__file__).parent


def measure(model: str, n: int = 50, threads: int = 2) -> dict:
    import torch
    torch.set_num_threads(threads)
    records = [json.loads(line) for line in (HERE / "data" / "spy.jsonl").open()][:n]
    labels = [] if model in ("openai-privacy-filter", "presidio") else query_labels(model, "spy", "full")
    runner = load_runner(model, "cpu")
    detect(runner, [records[0]["text"]], labels, batch_size=1)  # warm-up
    times = []
    for r in records:
        t0 = time.perf_counter()
        detect(runner, [r["text"]], labels, batch_size=1)
        times.append(1000 * (time.perf_counter() - t0))
    return {"model": model, "labels": len(labels), "n": n, "threads": threads, "machine": platform.machine(),
            "mean_words": round(statistics.mean(len(r["text"].split()) for r in records)),
            "ms_median": round(statistics.median(times)),
            "ms_p90": round(sorted(times)[int(0.9 * len(times))])}


def label_sweep(model: str, counts=(5, 20, 55), n: int = 20, threads: int = 2) -> list:
    """The same model, container and documents, asked for more and more of its
    labels: what each label costs, with the machine held fixed."""
    import torch
    torch.set_num_threads(threads)
    records = [json.loads(line) for line in (HERE / "data" / "spy.jsonl").open()][:n]
    vocab = query_labels(model, "spy", "full")
    runner = load_runner(model, "cpu")
    detect(runner, [records[0]["text"]], vocab[:5], batch_size=1)  # warm-up
    rows = []
    for k in counts:
        labels = vocab[:k]
        times = []
        for r in records:
            t0 = time.perf_counter()
            detect(runner, [r["text"]], labels, batch_size=1)
            times.append(1000 * (time.perf_counter() - t0))
        rows.append({"model": model, "labels": len(labels), "n": n, "threads": threads,
                     "ms_median": round(statistics.median(times))})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    print(json.dumps(measure(args.model, args.n, args.threads)))
