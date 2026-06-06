#!/usr/bin/env python3
"""Run our fine-tuned GLiNER (exp 004 checkpoint-1000) over the eval sample.

Two passes:
  native   — queried with its 24 training labels only (exp 003 lesson:
             bi-encoders must be queried with their native vocabulary)
  extended — native + 8 zero-shot labels the dataset annotates but the
             model never saw in training (username, passport, gender, ...).
             Tests off-vocabulary generalization.

Outputs: out/gliner-native.jsonl, out/gliner-extended.jsonl
         {id, detections: [{start, end, type, score}]}
"""

import json
import time
from pathlib import Path

from gliner import GLiNER

from mapping import GLINER_NATIVE, GLINER_EXTENDED

HERE = Path(__file__).parent
CHECKPOINT = HERE.parent / "004-finetune-fine-labels" / "model" / "checkpoint-1000"
THRESHOLD = 0.5  # same as every prior experiment


def run_pass(model, records, labels, out_path: Path) -> None:
    times = []
    with out_path.open("w") as out:
        for i, rec in enumerate(records):
            t0 = time.perf_counter()
            entities = model.predict_entities(rec["text"], labels, threshold=THRESHOLD)
            times.append((time.perf_counter() - t0) * 1000)
            out.write(json.dumps({
                "id": rec["id"],
                "detections": [
                    {"start": e["start"], "end": e["end"],
                     "type": e["label"], "score": round(e["score"], 3)}
                    for e in entities
                ],
            }) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1} records, avg {sum(times)/len(times):.0f}ms")
    print(f"Done: {len(times)} records, avg {sum(times)/len(times):.0f}ms -> {out_path}")


def main() -> None:
    print(f"Loading {CHECKPOINT} ...")
    model = GLiNER.from_pretrained(str(CHECKPOINT), local_files_only=True)

    records = [json.loads(l) for l in (HERE / "data" / "sample.jsonl").open()]
    (HERE / "out").mkdir(exist_ok=True)

    native_labels = list(GLINER_NATIVE.keys())
    print(f"\nPass 1: native vocabulary ({len(native_labels)} labels)")
    run_pass(model, records, native_labels, HERE / "out" / "gliner-native.jsonl")

    extended_labels = native_labels + list(GLINER_EXTENDED.keys())
    print(f"\nPass 2: extended vocabulary ({len(extended_labels)} labels)")
    run_pass(model, records, extended_labels, HERE / "out" / "gliner-extended.jsonl")


if __name__ == "__main__":
    main()
