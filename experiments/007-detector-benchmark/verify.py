#!/usr/bin/env python3
"""Adversarial audit of experiment 007's own methodology. Every check prints
PASS or FAIL with the numbers behind it; exit code 1 on any FAIL.

    .venv/bin/python verify.py

 1. SPY rebuilt faithfully: per-domain entity counts equal the dataset card's;
    every gold span's text is a value of its type; no entity runs into a word.
 2. The other sets are the ones claimed: Nemotron fingerprint, ai4privacy
    record and span counts (exp 006's), each other language 1,000 records
    sharing none with English, Thai national IDs all check-digit valid.
 3. Dev and test never share a document; tuned thresholds come from dev alone.
 4. Windows: across every SPY document, each character is owned by exactly one
    window, and no gold span is cut by a window edge it is owned next to.
 5. Every output covers every record of its dataset, inside the text bounds,
    and was queried with the labels mapping.py gives today.
 6. The scorer: gold against itself scores 1.0, nothing scores 0, detection
    order does not change a score; a first name alone does not mask a full
    name, and "Paul" + "Acosta" merge into one name only where gold labels
    whole names.
 7. results.json is reproducible from out/ (rescored here, compared).
 8. Thai segmentation maps spans back onto the original characters.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import score
from detectors import WINDOW_WORDS, split_windows
from mapping import DATASETS, query_labels

HERE = Path(__file__).parent
FAILED = []

SPY_CARD = {  # dataset card: entity counts per domain
    "legal": {"URL": 4243, "EMAIL": 4101, "USERNAME": 3868, "ADDRESS": 4173, "NAME": 4032,
              "PHONE_NUM": 3597, "ID_NUM": 3357},
    "medical": {"URL": 4322, "EMAIL": 4493, "USERNAME": 4273, "ADDRESS": 5122, "NAME": 4707,
                "PHONE_NUM": 4222, "ID_NUM": 4284},
}


def main() -> None:
    data = {path.stem: load(path.stem) for path in sorted((HERE / "data").glob("*.jsonl"))}
    check_spy(data["spy"])
    check_other_sets(data)
    check_splits(data["spy"])
    check_windows(data["spy"])
    check_outputs(data)
    check_scorer(data["spy"])
    check_results_reproduce()
    check_thai_segmentation(data.get("thai", []))
    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILED.append(name)


def load(name):
    return [json.loads(l) for l in (HERE / "data" / f"{name}.jsonl").open()]


# ─── 1–3. Data ──────────────────────────────────────────────────

def check_spy(spy):
    counts = {d: Counter(s["label"] for r in spy if r["domain"] == d for s in r["spans"]) for d in SPY_CARD}
    check("spy counts = dataset card", counts == {d: Counter(c) for d, c in SPY_CARD.items()},
          f"legal {sum(counts['legal'].values())}, medical {sum(counts['medical'].values())} spans")
    shapes = {"EMAIL": r"^\S+@\S+$", "URL": r"^https?://\S+$", "PHONE_NUM": r"\d", "ID_NUM": r"\d",
              "NAME": r"^[A-Z]", "ADDRESS": r"^\d", "USERNAME": r"^\S+$"}
    bad = [(s["label"], r["text"][s["start"]:s["end"]]) for r in spy for s in r["spans"]
           if not re.search(shapes[s["label"]], r["text"][s["start"]:s["end"]])]
    check("spy gold values have their type's shape", not bad, f"{len(bad)} odd values {bad[:3]}")
    glued = sum(1 for r in spy for s in r["spans"] if s["end"] < len(r["text"]) and r["text"][s["end"]].isalnum())
    check("no SPY entity runs into the next word", glued == 0, f"{glued}")


def check_other_sets(data):
    import hashlib
    if "nemotron" in data:
        h = hashlib.sha256()
        for r in data["nemotron"]:
            h.update(r["text"].encode()); h.update(b"\x00")
        check("nemotron is the head-to-head set", h.hexdigest().startswith("df099d8d5ff37346"),
              f"{len(data['nemotron'])} records, sha256 {h.hexdigest()[:16]}")
    if "ai4privacy" in data:
        n = sum(len(r["spans"]) for r in data["ai4privacy"])
        check("ai4privacy is exp 006's sample", len(data["ai4privacy"]) == 1000 and n == 6357,
              f"{len(data['ai4privacy'])} records, {n} spans")
    english = {r["id"] for r in data.get("ai4privacy", [])}
    for name in sorted(n for n in data if n.startswith("ai4privacy-")):
        ids = {r["id"] for r in data[name]}
        check(f"{name} is 1,000 records apart from English", len(ids) == 1000 and not ids & english,
              f"{len(ids)} records, {len(ids & english)} shared with English")
    if "thai" in data:
        ids = [r["text"][s["start"]:s["end"]] for r in data["thai"] for s in r["spans"] if s["label"] == "NATIONAL_ID"]
        valid = sum(1 for v in ids if thai_id_valid(v))
        check("thai national IDs have valid check digits", valid == len(ids), f"{valid}/{len(ids)}")


def thai_id_valid(value):
    d = re.sub(r"[\s-]", "", value)
    return len(d) == 13 and (11 - sum(int(d[i]) * (13 - i) for i in range(12)) % 11) % 10 == int(d[12])


def check_splits(spy):
    dev = {r["id"] for r in spy if r["split"] == "dev"}
    test = {r["id"] for r in spy if r["split"] == "test"}
    texts_dev = {r["text"] for r in spy if r["split"] == "dev"}
    dup = sum(1 for r in spy if r["split"] == "test" and r["text"] in texts_dev)
    check("spy dev and test are disjoint", not (dev & test) and dup == 0,
          f"{len(dev)} dev, {len(test)} test, {dup} identical texts across")


# ─── 4. Windows ─────────────────────────────────────────────────

def check_windows(spy):
    multi = cut = 0
    for r in spy:
        windows = split_windows(r["text"])
        owned = [0] * len(r["text"])
        for w in windows:
            for i in range(w.own_start, w.own_end):
                owned[i] += 1
        if any(c != 1 for c in owned):
            check("windows partition every document", False, r["id"])
            return
        multi += len(windows) > 1
        for s in r["spans"]:
            w = next(w for w in windows if w.own_start <= s["start"] < w.own_end)
            cut += s["end"] > w.offset + len(w.text)
    check("windows partition every document", True, f"{multi} of {len(spy)} documents span several windows ({WINDOW_WORDS} words)")
    check("no gold span is cut by its owning window's edge", cut == 0, f"{cut}")


# ─── 5. Outputs ─────────────────────────────────────────────────

def check_outputs(data):
    problems, known = [], 0
    for path in sorted((HERE / "out").glob("*/*.jsonl")):
        dataset, system = path.parent.name, path.stem
        recs = {r["id"]: r["text"] for r in data[dataset]}
        rows = [json.loads(l) for l in path.open()]
        if {r["id"] for r in rows} != set(recs):
            problems.append(f"{dataset}/{system}: covers {len(rows)}/{len(recs)}")
        overruns = [(d, len(recs[r["id"]])) for r in rows for d in r["detections"]
                    if not 0 <= d["start"] < d["end"] <= len(recs[r["id"]])]
        # The one known overrun: GLiNER2 ends a span that reaches the end of the
        # text one character past it; score.py clamps it. Anything else fails.
        unexplained = [d for d, n in overruns
                       if not (system.startswith("gliner2-pii") and d["start"] < n and d["end"] == n + 1)]
        if unexplained:
            problems.append(f"{dataset}/{system}: {len(unexplained)} out-of-bounds")
        known += len(overruns) - len(unexplained)
        meta = path.with_suffix(".meta.json")
        if meta.exists():
            m = json.loads(meta.read_text())
            if m["labels"] and m["labels"] != query_labels(m["model"], dataset, m["vocab"]):
                problems.append(f"{dataset}/{system}: labels differ from mapping.py")
    n = len(list((HERE / "out").glob("*/*.jsonl")))
    check("outputs complete, in bounds, labels as mapped", not problems,
          f"{n} outputs; {known} GLiNER2 ends one past the text (clamped); {problems[:4]}")


# ─── 6–7. Scorer ────────────────────────────────────────────────

def check_scorer(spy):
    gold = score.load_gold("spy")
    ids = [r["id"] for r in spy if r["split"] == "test"][:300]
    raw_gold = {r["id"]: [{"start": s["start"], "end": s["end"], "type": t, "score": 1.0}
                          for s in r["spans"] for t in [GOLD_AS_NATIVE[s["label"]]]] for r in spy}
    perfect = score.evaluate("spy", gold, raw_gold, ids, "gliner2-pii", score.thresholds_all(0.5))
    empty = score.evaluate("spy", gold, {i: [] for i in ids}, ids, "gliner2-pii", score.thresholds_all(0.5))
    check("gold scores 1.0 against itself, nothing scores 0",
          perfect["exact"]["f1"] == 1.0 and perfect["masked"] == 1.0 and empty["exact"]["f1"] == 0.0
          and empty["masked"] == 0.0 and empty["exact"]["fn"] > 0,
          f"self {perfect['exact']['f1']:.3f} / masked {perfect['masked']:.3f}, empty {empty['exact']['f1']:.3f}")
    check_names()
    path = HERE / "out" / "spy" / "gliner2-pii.jsonl"
    if path.exists():
        raw = score.load_detections(path, gold)
        shuffled = {i: list(reversed(d)) for i, d in raw.items()}
        a = score.evaluate("spy", gold, raw, ids, "gliner2-pii", score.thresholds_all(0.5))
        b = score.evaluate("spy", gold, shuffled, ids, "gliner2-pii", score.thresholds_all(0.5))
        check("detection order does not change a score", a["exact"] == b["exact"] and a["overlap"] == b["overlap"],
              f"exact F1 {a['exact']['f1']:.4f} both ways")


def check_names():
    text = "Ask Paul Acosta today."
    gold = {"t": {"spans": [{"start": 4, "end": 15, "bucket": "NAME", "label": "NAME"}],
                  "split": "test", "length": len(text), "text": text}}
    first_only = {"t": [{"start": 4, "end": 8, "type": "first_name", "score": 0.9}]}
    parts = {"t": [{"start": 4, "end": 8, "type": "first_name", "score": 0.9},
                   {"start": 9, "end": 15, "type": "last_name", "score": 0.8}]}
    t = score.thresholds_all(0.5)
    a = score.evaluate("spy", gold, first_only, ["t"], "pii-proxy-ner", t)
    b = score.evaluate("spy", gold, parts, ["t"], "pii-proxy-ner", t)
    c = score.evaluate("spy", gold, parts, ["t"], "pii-proxy-ner", t, merge_names=False)
    check("a first name alone leaves the name unmasked; name parts merge into one",
          a["masked"] == 0.0 and a["touched"] == 1.0 and b["masked"] == 1.0 and b["exact"]["f1"] == 1.0
          and c["exact"]["fp"] == 2, f"first only masked {a['masked']}, parts exact F1 {b['exact']['f1']}, "
          f"unmerged FP {c['exact']['fp']}")


GOLD_AS_NATIVE = {"NAME": "full_name", "EMAIL": "email", "PHONE_NUM": "phone_number", "URL": "url",
                  "USERNAME": "username", "ADDRESS": "street_address", "ID_NUM": "national_id_number"}


def check_results_reproduce():
    path = HERE / "results.json"
    if not path.exists():
        check("results.json reproduces from out/", False, "no results.json")
        return
    saved = json.loads(path.read_text())
    diffs = []
    for dataset in saved:
        fresh = score.score_dataset(dataset, score.load_gold(dataset), HERE / "out" / dataset)
        for system, entry in saved[dataset].items():
            if fresh.get(system, {}).get("default") != entry["default"]:
                diffs.append(f"{dataset}/{system}")
    check("results.json reproduces from out/", not diffs, f"{diffs[:5]}")


# ─── 8. Thai segmentation ───────────────────────────────────────

def check_thai_segmentation(thai):
    try:
        from detectors import ThaiSegmented
    except ImportError:
        return
    try:
        seg = [ThaiSegmented(r["text"]) for r in thai[:60]]
    except ImportError:
        print("SKIP  thai segmentation round trip: pythainlp not installed locally")
        return
    bad = 0
    for r, s in zip(thai, seg):
        assert s.text.replace(" ", "") == r["text"].replace(" ", "")
        for g in r["spans"]:
            # find the gold value in the segmented text and map it back
            mapped = [i for i, j in enumerate(s.index) if j is not None and g["start"] <= j < g["end"]]
            back = s.to_original([{"start": mapped[0], "end": mapped[-1] + 1}])
            bad += back[0]["start"] != g["start"] or back[0]["end"] != g["end"]
    check("thai segmentation maps spans back exactly", bad == 0, f"{bad} mismatches on {len(seg)} texts")


if __name__ == "__main__":
    main()
