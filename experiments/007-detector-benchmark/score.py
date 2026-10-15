#!/usr/bin/env python3
"""Score every detector output in out/ against its dataset's gold spans.

    .venv/bin/python score.py            # results.json + RESULTS.md
    .venv/bin/python score.py --out out-smoke

Per system:

  masked    share of gold spans whose every letter and digit lies inside some
            detection, of any type and before flattening — the PII a proxy
            would not let through. Independent of label names and of where a
            dataset draws a span's boundaries.
  exact     a detection counts when bucket and both boundaries match a gold
            span (SPY's convention, and GLiNER2-PII's published metric)
  overlap   bucket matches and the spans overlap (exp 006's primary metric)
  flagged_p share of detections, of any type, that overlap some gold span —
            masked's precision: what a proxy masks that was not PII. Low on
            SPY, whose text carries much unlabeled PII; meaningful elsewhere.
  touched   share of gold spans any detection overlaps at all (the lenient
            reading of masked; a first name alone "touches" a full name)

Matching is greedy one-to-one in document order (exp 006's score()). Recall's
denominator is every gold span, whether or not a system has a label for it.

Overlapping detections are flattened the way GLiNER's own flat decoder does it
— highest score keeps the region — for every system alike. It removes
GLiNER2-PII's nested name spans (person ⊃ full_name ⊃ first_name), which as
returned cost three false positives per name (kept under "nested"), and
Presidio's overlapping recognizers.

Where a dataset labels a whole name as one span (SPY, Thai), name parts that
only whitespace separates are merged into one detection, for every system:
models trained on Nemotron tag "Paul" and "Acosta" separately, which exact
match would otherwise count as one miss and two false positives. The
unmerged numbers are kept under "names_as_returned".

Thresholds: each model at its documented default, and per bucket tuned on
SPY's dev split (10%, seed 42) — reported on SPY's test split only.

Every default score carries a 95% interval from resampling documents (1,000
draws, seed 0) under "ci"; report.py does the paired version for the
comparisons it states.

SPY labels only the placeholder slots its generator filled; the text around
them carries more emails, URLs and phone numbers than the slots do (third
parties: "my lawyer, Ryan Thompson at ryan.thompson@example.com"), all
unlabeled. "exact_adj" therefore sets aside a detection that misses gold but
lands on an unlabeled email, URL or phone number a plain regex finds — neither
right nor wrong. Unlabeled names and addresses cannot be found this way, so
even exact_adj understates precision on SPY. Recall is unaffected.
"""

import argparse
import json
import re
from functools import lru_cache
from collections import defaultdict
from pathlib import Path

import numpy as np

from mapping import DATASETS, buckets

HERE = Path(__file__).parent
DEFAULT_THRESHOLD = {"knowledgator-pii": 0.3, "openai-privacy-filter": 0.0, "presidio": 0.0,
                     "regex": 0.0}
GRID = [round(0.2 + 0.05 * i, 2) for i in range(16)]  # 0.20 .. 0.95


def main() -> None:
    args = parse_args()
    out_dir = HERE / args.out
    results = {}
    for dataset_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        gold = load_gold(dataset)
        results[dataset] = score_dataset(dataset, gold, dataset_dir)
    (HERE / args.json).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.json}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out")
    p.add_argument("--json", default="results.json")
    return p.parse_args()


# ─── Per dataset ────────────────────────────────────────────────

def score_dataset(dataset, gold, dataset_dir):
    systems = {}
    for path in sorted(dataset_dir.glob("*.jsonl")):
        system = path.stem
        model = model_of(system)
        pipeline = "+" in model  # thresholds already applied when the pipeline was built
        raw = load_detections(path, gold)
        ids = [i for i in gold if i in raw]
        test = [i for i in ids if gold[i]["split"] == "test"]
        dev = [i for i in ids if gold[i]["split"] == "dev"]
        default = 0.0 if pipeline else DEFAULT_THRESHOLD.get(model, 0.5)
        docs = []
        entry = {"records": len(test), "default_threshold": default,
                 "default": evaluate(dataset, gold, raw, test, model, thresholds_all(default), docs=docs)}
        entry["ci"] = interval(docs)
        if model == "gliner2-pii":
            entry["nested"] = evaluate(dataset, gold, raw, test, model, thresholds_all(default),
                                       flat=False)
        if DATASETS[dataset].get("whole_names"):
            entry["names_as_returned"] = evaluate(dataset, gold, raw, test, model,
                                                  thresholds_all(default), merge_names=False)
        if dev and not pipeline and model != "regex":
            entry["dev"] = evaluate(dataset, gold, raw, dev, model, thresholds_all(default))
            tuned = tune_thresholds(dataset, gold, raw, dev, model)
            entry["tuned_thresholds"] = tuned
            entry["tuned"] = evaluate(dataset, gold, raw, test, model, tuned)
        if dataset == "spy":
            entry["by_domain"] = {d: evaluate(dataset, gold, raw,
                                              [i for i in test if gold[i]["domain"] == d],
                                              model, thresholds_all(default))
                                  for d in ("legal", "medical")}
        if dataset == "nemotron" and system.endswith(".full"):
            entry["fine"] = fine_labels(raw, test)
        systems[system] = entry
    return systems


def fine_labels(raw, ids):
    """Nemotron's own 55 labels, overlap + exact label, threshold 0.5 — the
    head-to-head's metric, for the two models trained on that vocabulary."""
    gold = {r["id"]: r["spans"] for r in map(json.loads, (HERE / "data" / "nemotron.jsonl").open())}
    counts = Counts()
    for rid in ids:
        g = [{**s, "bucket": s["label"]} for s in gold[rid]]
        dets = [{**d, "buckets": (d["type"],)} for d in raw[rid] if d["score"] >= 0.5]
        match(g, dets, False, counts, defaultdict(Counts))
    return counts.prf()


def load_detections(path, gold):
    """{id: detections}, each end clamped to its text: GLiNER2 reports an end
    one past the text for a span that runs to the text's end (verify.py
    checks that this is the only overrun)."""
    out = {}
    for r in map(json.loads, path.open()):
        n = gold[r["id"]]["length"]
        out[r["id"]] = [{**d, "end": min(d["end"], n)} for d in r["detections"]]
    return out


def model_of(system: str) -> str:
    """'gliner2-pii.full.seg' -> 'gliner2-pii'; 'pii-proxy-ner+regex' -> 'pii-proxy-ner+regex'."""
    return system.split(".")[0]


def thresholds_all(t):
    return defaultdict(lambda: t)


def tune_thresholds(dataset, gold, raw, dev, model):
    buckets = sorted(set(DATASETS[dataset]["gold"].values()) - {None})
    best = {}
    for b in buckets:
        scores = []
        for t in GRID:
            per = evaluate(dataset, gold, raw, dev, model, thresholds_all(t))["per_bucket"][b]
            scores.append((per["exact"]["f1"], -abs(t - 0.5), t))
        best[b] = max(scores)[2]
    return defaultdict(lambda: 0.5, best)


# ─── Scoring ────────────────────────────────────────────────────

def evaluate(dataset, gold, raw, ids, model, thresholds, flat=True, merge_names=True, docs=None):
    exact, overlap, adjusted = Counts(), Counts(), Counts()
    per = defaultdict(lambda: {"exact": Counts(), "overlap": Counts(), "masked": Counts()})
    masked = touched = total = flags = flags_on_pii = 0
    unlabeled = unlabeled_pii() if dataset == "spy" else {}
    whole_names = merge_names and flat and DATASETS[dataset].get("whole_names")
    for rid in ids:
        before = (masked, total, *exact.row(), *overlap.row())
        g, text = gold[rid]["spans"], gold[rid]["text"]
        kept = [d for d in raw[rid] if d.get("score", 1.0) >= threshold_of(thresholds, model, dataset, d)]
        all_dets = flatten(kept) if flat else kept
        dets = [{**d, "buckets": b} for d in all_dets if (b := bucket_of(model, dataset, d))]
        if whole_names:
            dets = merge_name_parts(dets, text)
        for counts, per_key, exact_match in ((exact, "exact", True), (overlap, "overlap", False)):
            match(g, dets, exact_match, counts, {b: per[b][per_key] for b in per_buckets(dataset)})
        if unlabeled:
            match(g, dets, True, adjusted, defaultdict(Counts), ignore=unlabeled.get(rid, []))
        for s in g:  # a masker masks every span it is given: the union, before flattening
            full = fully_masked(s, kept, text)
            masked += full
            per[s["bucket"]]["masked"].tp += full
            per[s["bucket"]]["masked"].fn += not full
            touched += any(overlaps(s, d) for d in kept)
        total += len(g)
        flags += len(kept)
        flags_on_pii += sum(1 for d in kept if any(overlaps(d, s) for s in g))
        if docs is not None:
            docs.append([a - b for a, b in zip((masked, total, *exact.row(), *overlap.row()), before)])
    return {"masked": masked / total if total else 0.0,
            "flagged_p": flags_on_pii / flags if flags else 0.0,
            "exact": exact.prf(), "overlap": overlap.prf(),
            **({"exact_adj": adjusted.prf()} if unlabeled else {}),
            "touched": touched / total if total else 0.0,
            "per_bucket": {b: {"exact": per[b]["exact"].prf(), "overlap": per[b]["overlap"].prf(),
                               "masked": per[b]["masked"].prf()["r"]}
                           for b in per_buckets(dataset)}}


def fully_masked(span, dets, text):
    """Every letter and digit of the gold span lies inside some detection."""
    return all(any(d["start"] <= i < d["end"] for d in dets)
               for i in range(span["start"], span["end"]) if text[i].isalnum())


def flatten(dets):
    kept = []
    for d in sorted(dets, key=lambda d: -d.get("score", 1.0)):
        if not any(overlaps(d, k) for k in kept):
            kept.append(d)
    return kept


def merge_name_parts(dets, text):
    """Merge NAME detections that only whitespace separates ("Paul" + "Acosta")."""
    out = []
    for d in sorted(dets, key=lambda d: d["start"]):
        prev = out[-1] if out else None
        if (prev and "NAME" in prev["buckets"] and "NAME" in d["buckets"]
                and prev["end"] <= d["start"] and not text[prev["end"]:d["start"]].strip()):
            out[-1] = {**prev, "end": d["end"], "buckets": ("NAME",),
                       "score": min(prev.get("score", 1.0), d.get("score", 1.0))}
        else:
            out.append(d)
    return out


def per_buckets(dataset):
    return sorted(set(DATASETS[dataset]["gold"].values()) - {None})


def bucket_of(model, dataset, det):
    """The dataset buckets a detection can match, () if the dataset does not
    annotate its type. Pipeline outputs carry the source of each detection
    (the model or the regex layer)."""
    source = det.get("source")
    name = "regex" if source == "regex" else model.split("+")[0]
    return buckets(name, dataset, det["type"])


def threshold_of(thresholds, model, dataset, det):
    b = bucket_of(model, dataset, det)
    return thresholds[b[0] if b else None]


def interval(docs, draws=1000):
    """95% bootstrap intervals over documents for masked, exact F1, overlap F1."""
    return {k: [round(float(v), 4) for v in np.percentile(samples, [2.5, 97.5])]
            for k, samples in bootstrap(np.array(docs), draws).items()}


def bootstrap(docs, draws=1000, seed=0):
    """{metric: array of `draws` resampled values}; docs rows are
    [masked, total, exact tp, fp, fn, overlap tp, fp, fn]."""
    idx = np.random.default_rng(seed).integers(0, len(docs), (draws, len(docs)))
    sums = docs[idx].sum(axis=1)
    return {"masked": sums[:, 0] / np.maximum(sums[:, 1], 1),
            "exact_f1": f1(sums[:, 2], sums[:, 3], sums[:, 4]),
            "overlap_f1": f1(sums[:, 5], sums[:, 6], sums[:, 7])}


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


class Counts:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def row(self):
        return self.tp, self.fp, self.fn

    def prf(self):
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"p": p, "r": r, "f1": f, "tp": self.tp, "fp": self.fp, "fn": self.fn}


def match(gold_spans, dets, exact, counts, per, ignore=()):
    """Greedy one-to-one, detections in document order (exp 006's score()).
    A miss that lands on an `ignore` region (unlabeled PII) is set aside."""
    used = set()
    for d in sorted(dets, key=lambda d: (d["start"], d["end"])):
        hit = next((i for i, g in enumerate(gold_spans) if i not in used and g["bucket"] in d["buckets"]
                    and (same(g, d) if exact else overlaps(g, d))), None)
        if hit is None and any(overlaps(d, r) for r in ignore):
            continue
        if hit is None:
            counts.fp += 1
            per[d["buckets"][0]].fp += 1
        else:
            used.add(hit)
            counts.tp += 1
            per[gold_spans[hit]["bucket"]].tp += 1
    for i, g in enumerate(gold_spans):
        if i not in used:
            counts.fn += 1
            per[g["bucket"]].fn += 1


def same(a, b):
    return a["start"] == b["start"] and a["end"] == b["end"]


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


UNLABELED_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                          # email
    re.compile(r"https?://\S+|www\.\S+"),                                # URL
    re.compile(r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}"),        # US-style phone
]


@lru_cache(maxsize=1)
def unlabeled_pii():
    """{id: [regions]} — emails, URLs and phone numbers in SPY's text that no
    gold span covers."""
    out = {}
    for r in map(json.loads, (HERE / "data" / "spy.jsonl").open()):
        regions = [{"start": m.start(), "end": m.end()} for p in UNLABELED_PATTERNS
                   for m in p.finditer(r["text"])]
        out[r["id"]] = [x for x in regions if not any(overlaps(x, g) for g in r["spans"])]
    return out


# ─── Gold ───────────────────────────────────────────────────────

def load_gold(dataset):
    """{id: {spans: [{start, end, bucket}], split, domain, text}}; gold labels outside
    the dataset's buckets (Nemotron's coarse view only) are left out."""
    to_bucket = DATASETS[dataset]["gold"]
    gold = {}
    for r in map(json.loads, (HERE / "data" / f"{dataset}.jsonl").open()):
        spans = [{"start": s["start"], "end": s["end"], "bucket": to_bucket[s["label"]],
                  "label": s["label"]} for s in r["spans"] if to_bucket.get(s["label"])]
        gold[r["id"]] = {"spans": spans, "split": r["split"], "domain": r.get("domain"),
                         "length": len(r["text"]), "text": r["text"]}
    return gold


if __name__ == "__main__":
    main()
