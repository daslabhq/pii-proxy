#!/usr/bin/env python3
"""Score all systems against ai4privacy ground truth.

Systems:
  presidio              — Microsoft Presidio, default config, en_core_web_lg
  pii-proxy-regex       — our shipped regex detector layer alone
  pii-proxy+gliner      — regex ∪ fine-tuned GLiNER (native vocabulary)
  pii-proxy+gliner-ext  — regex ∪ GLiNER (native + zero-shot extended labels)

Matching: greedy 1:1, bucket equality + char overlap (primary metric);
exact-boundary match reported secondarily. Detections whose native type
maps to None are dropped before scoring (unverifiable against this
dataset's annotations — not counted as FP, documented in README).
Recall denominator is ALWAYS all 6,357 annotated spans — a system that
doesn't attempt a bucket scores 0 recall on it, visible per-type.

Output: results.json + a markdown table on stdout.
"""

import json
from collections import defaultdict
from pathlib import Path

from mapping import (ALL_BUCKETS, GLINER_005, GLINER_EXTENDED, GLINER_NATIVE,
                     GROUND_TRUTH, PII_PROXY_REGEX, PRESIDIO)

HERE = Path(__file__).parent


def load_jsonl(path):
    return {r["id"]: r for r in (json.loads(l) for l in path.open())}


def map_detections(records, type_map):
    """Apply bucket mapping, drop unverifiable (None) detections."""
    out = {}
    for rid, rec in records.items():
        mapped = []
        for d in rec["detections"]:
            bucket = type_map.get(d["type"], None)
            if bucket is not None:
                mapped.append({"start": d["start"], "end": d["end"], "bucket": bucket})
        out[rid] = mapped
    return out


def raw_spans(records):
    """All detection spans regardless of type mapping (for coverage)."""
    return {rid: [{"start": d["start"], "end": d["end"]} for d in rec["detections"]]
            for rid, rec in records.items()}


def coverage(gold_by_id, raw_by_id):
    """Type-agnostic recall: fraction of GT spans overlapped by ANY
    detection. For leak prevention a type-confused detection still masks
    the PII — this is the safety number; type-correct F1 is the
    fake-quality number."""
    covered, total = 0, 0
    for rid, gold in gold_by_id.items():
        preds = raw_by_id.get(rid, [])
        for g in gold:
            total += 1
            if any(g["start"] < p["end"] and p["start"] < g["end"] for p in preds):
                covered += 1
    return covered / total if total else 0.0


def union(a, b):
    """Merge two detection sets with the library's exact detectAll
    semantics (src/detectors/index.ts): sort by start (earlier source
    wins ties, longer span breaks remaining ties), suppress ANY overlap
    regardless of type. This means a regex detection can shadow a
    better model detection — that's what the shipped pipeline does, so
    that's what we score. See verify.py check 7 for the cost."""
    out = {}
    for rid in a.keys() | b.keys():
        tagged = [(0, d) for d in a.get(rid, [])] + [(1, d) for d in b.get(rid, [])]
        tagged.sort(key=lambda t: (t[1]["start"], t[0], -(t[1]["end"] - t[1]["start"])))
        merged, last_end = [], -1
        for _, d in tagged:
            if d["start"] >= last_end:
                merged.append(d)
                last_end = d["end"]
        out[rid] = merged
    return out


def score(gold_by_id, detections_by_id, exact=False):
    """Greedy 1:1 matching. Returns micro totals + per-bucket counts."""
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for rid, gold in gold_by_id.items():
        preds = detections_by_id.get(rid, [])
        matched_gold = set()
        for p in preds:
            hit = None
            for gi, g in enumerate(gold):
                if gi in matched_gold or g["bucket"] != p["bucket"]:
                    continue
                if exact:
                    ok = g["start"] == p["start"] and g["end"] == p["end"]
                else:
                    ok = g["start"] < p["end"] and p["start"] < g["end"]
                if ok:
                    hit = gi
                    break
            if hit is not None:
                matched_gold.add(hit)
                per[p["bucket"]]["tp"] += 1
            else:
                per[p["bucket"]]["fp"] += 1
        for gi, g in enumerate(gold):
            if gi not in matched_gold:
                per[g["bucket"]]["fn"] += 1
    return per


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def micro(per):
    tot = {"tp": 0, "fp": 0, "fn": 0}
    for c in per.values():
        for k in tot:
            tot[k] += c[k]
    return tot


def main() -> None:
    gold_raw = [json.loads(l) for l in (HERE / "data" / "sample.jsonl").open()]
    gold_by_id = {
        r["id"]: [{"start": s["start"], "end": s["end"],
                   "bucket": GROUND_TRUTH[s["label"]]} for s in r["spans"]]
        for r in gold_raw
    }

    regex_raw = load_jsonl(HERE / "out" / "pii-proxy-regex.jsonl")
    presidio_raw = load_jsonl(HERE / "out" / "presidio.jsonl")
    gliner_nat_raw = load_jsonl(HERE / "out" / "gliner-native.jsonl")
    gliner_ext_raw = load_jsonl(HERE / "out" / "gliner-extended.jsonl")

    regex = map_detections(regex_raw, PII_PROXY_REGEX)
    presidio = map_detections(presidio_raw, PRESIDIO)
    gliner_map = {**GLINER_NATIVE, **GLINER_EXTENDED}
    gliner_nat = map_detections(gliner_nat_raw, gliner_map)
    gliner_ext = map_detections(gliner_ext_raw, gliner_map)

    # Detector ORDER is a PrivacyProxy constructor argument and first-wins
    # merge makes it consequential — so orderings are distinct systems.
    systems = {
        "presidio": presidio,
        "pii-proxy-regex": regex,
        "regex-first+gliner": union(regex, gliner_nat),
        "regex-first+gliner-ext": union(regex, gliner_ext),
        "gliner-ext-first+regex": union(gliner_ext, regex),
        "gliner-ext-alone": gliner_ext,
    }

    def merge_raw(a, b):
        return {rid: a.get(rid, []) + b.get(rid, [])
                for rid in a.keys() | b.keys()}

    combined_raw = merge_raw(raw_spans(regex_raw), raw_spans(gliner_ext_raw))
    raw_systems = {
        "presidio": raw_spans(presidio_raw),
        "pii-proxy-regex": raw_spans(regex_raw),
        "regex-first+gliner": merge_raw(raw_spans(regex_raw), raw_spans(gliner_nat_raw)),
        "regex-first+gliner-ext": combined_raw,
        "gliner-ext-first+regex": combined_raw,
        "gliner-ext-alone": raw_spans(gliner_ext_raw),
    }

    # pii-proxy-ner (exp 005 flagship, 55 native labels) — rows appear when
    # run_gliner_005.py outputs exist.
    ner005_path = HERE / "out" / "gliner005-extended.jsonl"
    if ner005_path.exists():
        ner005_raw = load_jsonl(ner005_path)
        ner005 = map_detections(ner005_raw, {**GLINER_005, **GLINER_EXTENDED})
        systems["pii-proxy-ner-first+regex"] = union(ner005, regex)
        systems["pii-proxy-ner-alone"] = ner005
        ner_combined_raw = merge_raw(raw_spans(regex_raw), raw_spans(ner005_raw))
        raw_systems["pii-proxy-ner-first+regex"] = ner_combined_raw
        raw_systems["pii-proxy-ner-alone"] = raw_spans(ner005_raw)

    results = {}
    for name, dets in systems.items():
        per = score(gold_by_id, dets, exact=False)
        per_exact = score(gold_by_id, dets, exact=True)
        m, me = micro(per), micro(per_exact)
        p, r, f1 = prf(m)
        pe, re_, f1e = prf(me)
        results[name] = {
            "overlap": {"p": p, "r": r, "f1": f1, **m},
            "exact": {"p": pe, "r": re_, "f1": f1e, **me},
            "coverage": coverage(gold_by_id, raw_systems[name]),
            "per_bucket": {
                b: dict(zip(("p", "r", "f1"), prf(per[b])), **per[b])
                for b in ALL_BUCKETS
            },
        }

    (HERE / "results.json").write_text(json.dumps(results, indent=2))

    # ── markdown summary ──
    print("\n## Overall (entity-level, overlap match)\n")
    print("| System | P | R | F1 | exact-F1 | coverage |")
    print("|---|---|---|---|---|---|")
    for name, res in results.items():
        o, e = res["overlap"], res["exact"]
        print(f"| {name} | {o['p']:.1%} | {o['r']:.1%} | {o['f1']:.1%} "
              f"| {e['f1']:.1%} | {res['coverage']:.1%} |")

    print("\n## Per-type F1 (overlap match)\n")
    header = "| Type | gold n | " + " | ".join(systems) + " |"
    print(header)
    print("|---" * (len(systems) + 2) + "|")
    gold_n = defaultdict(int)
    for spans in gold_by_id.values():
        for s in spans:
            gold_n[s["bucket"]] += 1
    for b in ALL_BUCKETS:
        row = [b, str(gold_n[b])]
        for name in systems:
            row.append(f"{results[name]['per_bucket'][b]['f1']:.1%}")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
