#!/usr/bin/env python3
"""Run the exp-005 model (pii-proxy-ner: gliner_small fine-tuned on the FULL
Nemotron-PII, 55 entity types, 30 domains) over the eval sample.

This is the flagship model from the main README (90.1% F1, matches NVIDIA's
gliner-PII). Weights: Modal volume `pii-proxy-models`,
baseline-checkpoint/checkpoint-4641, downloaded to
../005-nvidia-baseline/model/checkpoint-4641.

Two passes, same protocol as run_gliner.py:
  native   — its 55 native Nemotron labels (data/nemotron_labels.json)
  extended — native + the same 8 zero-shot labels used for exp-004

Outputs: out/gliner005-native.jsonl, out/gliner005-extended.jsonl
"""

import json
from pathlib import Path

from gliner import GLiNER

from mapping import GLINER_EXTENDED
from run_gliner import run_pass

HERE = Path(__file__).parent
CHECKPOINT = HERE.parent / "005-nvidia-baseline" / "model" / "checkpoint-4641"


def main() -> None:
    native_labels = json.load((HERE / "data" / "nemotron_labels.json").open())
    print(f"Loading {CHECKPOINT} ({len(native_labels)} native labels)...")
    model = GLiNER.from_pretrained(str(CHECKPOINT), local_files_only=True)

    records = [json.loads(l) for l in (HERE / "data" / "sample.jsonl").open()]
    (HERE / "out").mkdir(exist_ok=True)

    print(f"\nPass 1: native vocabulary ({len(native_labels)} labels)")
    run_pass(model, records, native_labels, HERE / "out" / "gliner005-native.jsonl")

    extended_labels = native_labels + [
        l for l in GLINER_EXTENDED if l not in native_labels]
    print(f"\nPass 2: extended vocabulary ({len(extended_labels)} labels)")
    run_pass(model, records, extended_labels, HERE / "out" / "gliner005-extended.jsonl")


if __name__ == "__main__":
    main()
