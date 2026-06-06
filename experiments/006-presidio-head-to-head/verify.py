#!/usr/bin/env python3
"""Adversarial audit of experiment 006 before publishing numbers.

Checks:
  1. Ground-truth integrity (offsets match values, no dup records)
  2. Granularity bias: greedy 1:1 matching punishes systems that detect
     one span where the dataset annotates two (Presidio PERSON "John Smith"
     vs gold GIVENNAME1+LASTNAME1). Re-score with relaxed many-to-many
     matching: recall = gold span overlapped by any same-bucket pred,
     precision = pred overlapping any same-bucket gold.
  3. Shared-vocabulary comparison: micro F1 restricted to buckets BOTH
     presidio and our pipeline map to (strips the vocabulary-coverage
     advantage out of the headline delta).
  4. Stretch-mapping impact: TPs riding on unique_id->ID_CARD and
     certificate_license_number->DRIVER_LICENSE.
  5. IP misses: IPv6 share of gold IP spans.
  6. Phone FPs: which gold types our phone regex actually overlaps.
  7. Union semantics: library detectAll dedups ANY overlap (earlier
     detector wins), not just same-bucket — re-score combined systems
     with faithful semantics and report the delta.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from mapping import (GLINER_EXTENDED, GLINER_NATIVE, GROUND_TRUTH,
                     PII_PROXY_REGEX, PRESIDIO)
from score import load_jsonl, map_detections, prf, score, micro

HERE = Path(__file__).parent


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def relaxed_score(gold_by_id, dets_by_id):
    """Many-to-many: each gold span TP-covered if ANY same-bucket pred
    overlaps it; each pred an FP only if it overlaps NO same-bucket gold."""
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for rid, gold in gold_by_id.items():
        preds = dets_by_id.get(rid, [])
        for g in gold:
            if any(p["bucket"] == g["bucket"] and overlaps(p, g) for p in preds):
                per[g["bucket"]]["tp"] += 1
            else:
                per[g["bucket"]]["fn"] += 1
        for p in preds:
            if not any(p["bucket"] == g["bucket"] and overlaps(p, g) for g in gold):
                per[p["bucket"]]["fp"] += 1
    return per


def library_union(a, b):
    """Faithful to src detectAll: sort by start (longer wins ties),
    suppress ANY overlap, earlier source (a) wins."""
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


def main() -> None:
    sample = [json.loads(l) for l in (HERE / "data" / "sample.jsonl").open()]

    # ── 1. ground-truth integrity ──
    bad_offsets, ids = 0, Counter()
    for rec in sample:
        ids[rec["id"]] += 1
        for s in rec["spans"]:
            if rec["text"][s["start"]:s["end"]] != s["value"]:
                bad_offsets += 1
    dups = [i for i, c in ids.items() if c > 1]
    print(f"1. GT integrity: {bad_offsets} bad offsets / 6357 spans, "
          f"{len(dups)} duplicate record ids")

    gold_by_id = {
        r["id"]: [{"start": s["start"], "end": s["end"],
                   "bucket": GROUND_TRUTH[s["label"]], "label": s["label"],
                   "value": s["value"]} for s in r["spans"]]
        for r in sample
    }

    regex_raw = load_jsonl(HERE / "out" / "pii-proxy-regex.jsonl")
    presidio_raw = load_jsonl(HERE / "out" / "presidio.jsonl")
    gnat_raw = load_jsonl(HERE / "out" / "gliner-native.jsonl")
    gext_raw = load_jsonl(HERE / "out" / "gliner-extended.jsonl")

    gliner_map = {**GLINER_NATIVE, **GLINER_EXTENDED}
    regex = map_detections(regex_raw, PII_PROXY_REGEX)
    presidio = map_detections(presidio_raw, PRESIDIO)
    gnat = map_detections(gnat_raw, gliner_map)
    gext = map_detections(gext_raw, gliner_map)

    ours_strictunion = library_union(regex, gext)

    # ── 2. relaxed vs 1:1 ──
    print("\n2. Greedy-1:1 vs relaxed many-to-many (micro P/R/F1):")
    for name, dets in [("presidio", presidio), ("ours(ext, lib-union)", ours_strictunion)]:
        m1 = micro(score(gold_by_id, dets))
        mr = micro(relaxed_score(gold_by_id, dets))
        p1, r1, f1 = prf(m1)
        pr_, rr, fr = prf(mr)
        print(f"   {name:22s} 1:1 {p1:.1%}/{r1:.1%}/{f1:.1%}   "
              f"relaxed {pr_:.1%}/{rr:.1%}/{fr:.1%}")

    # split-span evidence
    multi = 0
    for rid, rec in presidio_raw.items():
        gold = [g for g in gold_by_id[rid] if g["bucket"] == "NAME"]
        for d in rec["detections"]:
            if PRESIDIO.get(d["type"]) == "NAME":
                n = sum(1 for g in gold if overlaps(d, g))
                if n >= 2:
                    multi += 1
    print(f"   Presidio PERSON detections spanning >=2 gold name spans: {multi}")

    # ── 3. shared-vocabulary comparison ──
    shared = sorted(set(PRESIDIO.values()) & set(gliner_map.values())
                    & set(GROUND_TRUTH.values()) - {None})
    print(f"\n3. Shared-bucket micro (buckets: {', '.join(shared)}):")
    gold_shared = {rid: [g for g in gold if g["bucket"] in shared]
                   for rid, gold in gold_by_id.items()}
    for name, dets in [("presidio", presidio),
                       ("ours(regex-first)", ours_strictunion),
                       ("ours(gliner-first)", library_union(gext, regex))]:
        dets_shared = {rid: [d for d in ds if d["bucket"] in shared]
                       for rid, ds in dets.items()}
        m = micro(relaxed_score(gold_shared, dets_shared))
        p, r, f1 = prf(m)
        print(f"   {name:22s} relaxed {p:.1%}/{r:.1%}/{f1:.1%}")

    # ── 4. stretch mappings ──
    print("\n4. TPs riding on stretch mappings (relaxed, ours ext):")
    for native, bucket in [("unique_id", "ID_CARD"),
                           ("certificate_license_number", "DRIVER_LICENSE")]:
        tps = 0
        for rid, rec in gext_raw.items():
            gold = [g for g in gold_by_id[rid] if g["bucket"] == bucket]
            for d in rec["detections"]:
                if d["type"] == native and any(overlaps(d, g) for g in gold):
                    tps += 1
        gold_n = sum(1 for gs in gold_by_id.values() for g in gs if g["bucket"] == bucket)
        print(f"   {native}->{bucket}: {tps} overlapping detections / {gold_n} gold")

    # ── 5. IP misses ──
    ipv6 = sum(1 for gs in gold_by_id.values() for g in gs
               if g["bucket"] == "IP" and ":" in g["value"])
    total_ip = sum(1 for gs in gold_by_id.values() for g in gs if g["bucket"] == "IP")
    print(f"\n5. Gold IP spans: {total_ip}, containing ':' (IPv6): {ipv6}")

    # ── 6. phone FP anatomy ──
    print("\n6. Our phone-regex detections overlap gold types:")
    hits = Counter()
    for rid, rec in regex_raw.items():
        gold = gold_by_id[rid]
        for d in rec["detections"]:
            if d["type"] != "phone":
                continue
            labs = {g["label"] for g in gold if overlaps(d, g)}
            hits[tuple(sorted(labs)) or ("NONE",)] += 1
    for labs, c in hits.most_common(10):
        print(f"   {','.join(labs):30s} {c}")

    # ── 7. union semantics delta ──
    print("\n7. Combined-system union semantics (ours ext, relaxed):")

    def bucket_union(a, b):  # the BUG score.py originally had: same-bucket-only dedup
        out = {}
        for rid in a.keys() | b.keys():
            merged = list(a.get(rid, []))
            for d in b.get(rid, []):
                if not any(m["bucket"] == d["bucket"] and overlaps(m, d) for m in merged):
                    merged.append(d)
            out[rid] = merged
        return out

    for name, dets in [("same-bucket-only union (NOT what library does)", bucket_union(regex, gext)),
                       ("library any-overlap union", ours_strictunion)]:
        m = micro(relaxed_score(gold_by_id, dets))
        p, r, f1 = prf(m)
        print(f"   {name:42s} {p:.1%}/{r:.1%}/{f1:.1%}")


if __name__ == "__main__":
    main()
