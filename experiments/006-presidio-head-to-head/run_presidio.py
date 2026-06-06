#!/usr/bin/env python3
"""Run Microsoft Presidio over the eval sample.

Config: presidio-analyzer defaults with spaCy en_core_web_lg, all built-in
recognizers, no tuning, score threshold left at default. This is the same
"default Presidio" baseline most published comparisons use — Presidio can
be tuned per-deployment; we benchmark the out-of-the-box configuration and
say so.

Output: out/presidio.jsonl  {id, detections: [{start, end, type, score}]}
"""

import json
import time
from pathlib import Path

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

HERE = Path(__file__).parent


def main() -> None:
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())

    (HERE / "out").mkdir(exist_ok=True)
    out_path = HERE / "out" / "presidio.jsonl"
    times = []

    with (HERE / "data" / "sample.jsonl").open() as f, out_path.open("w") as out:
        for i, line in enumerate(f):
            rec = json.loads(line)
            t0 = time.perf_counter()
            results = analyzer.analyze(text=rec["text"], language="en", entities=None)
            times.append((time.perf_counter() - t0) * 1000)
            out.write(json.dumps({
                "id": rec["id"],
                "detections": [
                    {"start": r.start, "end": r.end,
                     "type": r.entity_type, "score": round(r.score, 3)}
                    for r in results
                ],
            }) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1} records, avg {sum(times)/len(times):.0f}ms")

    print(f"Done: {len(times)} records, avg {sum(times)/len(times):.0f}ms -> {out_path}")


if __name__ == "__main__":
    main()
