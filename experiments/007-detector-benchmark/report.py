#!/usr/bin/env python3
"""Render results.json as the README's tables (markdown on stdout).

    .venv/bin/python report.py > /tmp/tables.md

Also computes what results.json does not hold: paired bootstrap intervals for
the differences the README states, and the spread of SPY scores over random
100 + 100 document draws (the size of the evaluation GLiNER2-PII's report
used), so its published numbers can be read against sampling noise.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import score
from mapping import AI4PRIVACY_LANGUAGES

HERE = Path(__file__).parent

# GLiNER2-PII's published SPY numbers (exact match): model card and technical report.
PUBLISHED = {
    "card": {
        "gliner2-pii": ((.346, .750, .473), (.369, .686, .480)),
        "nvidia-gliner-pii": ((.343, .452, .390), (.368, .465, .411)),
        "gliner-multi-pii": ((.467, .317, .377), (.518, .351, .419)),
        "openai-privacy-filter": ((.242, .656, .354), (.287, .692, .406)),
    },
    "paper": {
        "gliner2-pii": ((.354, .722, .475), (.355, .681, .467)),
        "nvidia-gliner-pii": ((.374, .431, .401), (.341, .431, .381)),
        "gliner-multi-pii": ((.522, .308, .388), (.483, .314, .381)),
        "openai-privacy-filter": ((.250, .640, .360), (.271, .671, .386)),
        "knowledgator-pii": ((.398, .372, .385), (.389, .319, .350)),
    },
}

NAMES = {
    "gliner2-pii": "GLiNER2-PII", "openai-privacy-filter": "OpenAI Privacy Filter",
    "nvidia-gliner-pii": "NVIDIA gliner-PII", "pii-proxy-ner": "pii-proxy-ner",
    "knowledgator-pii": "knowledgator gliner-pii-base", "gliner-multi-pii": "gliner_multi_pii-v1",
    "pii-proxy-24": "daslabhq/pii-proxy (24-label)", "presidio": "Presidio (default)",
    "regex": "pii-proxy regex only",
}
VARIANT_NAMES = {"gliner2-pii-char": "GLiNER2-PII, its character splitter"}
LANGS = ["ai4privacy"] + [f"ai4privacy-{lang}" for lang in AI4PRIVACY_LANGUAGES]
LANG_NAMES = {"ai4privacy": "EN", "ai4privacy-de": "DE", "ai4privacy-fr": "FR",
              "ai4privacy-it": "IT", "ai4privacy-es": "ES", "ai4privacy-nl": "NL"}

# The differences the README states, each with a paired 95% interval.
COMPARISONS = [
    ("spy", "knowledgator-pii", "gliner2-pii", "masked"),
    ("spy", "openai-privacy-filter", "gliner2-pii", "masked"),
    ("spy", "pii-proxy-ner", "nvidia-gliner-pii", "masked"),
    ("spy", "gliner2-pii", "nvidia-gliner-pii", "exact_f1"),
    ("ai4privacy", "nvidia-gliner-pii", "pii-proxy-ner", "masked"),
    ("ai4privacy", "nvidia-gliner-pii", "pii-proxy-ner", "overlap_f1"),
    ("ai4privacy", "pii-proxy-ner", "gliner2-pii", "masked"),
    ("ai4privacy", "openai-privacy-filter", "nvidia-gliner-pii", "overlap_f1"),
    ("ai4privacy-de", "gliner2-pii", "nvidia-gliner-pii", "masked"),
    ("ai4privacy-de", "gliner2-pii", "pii-proxy-ner", "masked"),
    ("ai4privacy-fr", "gliner2-pii", "pii-proxy-ner", "masked"),
    ("thai", "gliner2-pii.seg", "openai-privacy-filter.seg", "masked"),
]


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    headline(results)
    as_shipped(results)
    identifiers_vs_context(results)
    comparisons(results)
    if "spy" in results:
        spy_typed(results["spy"])
        per_bucket_masked(results["spy"], "spy")
        long_documents()
        spy_published()
    english_typed(results)
    languages_typed(results)
    if "nemotron" in results:
        nemotron(results["nemotron"])
    if "thai" in results:
        thai(results["thai"])
    speed()
    label_cost()


# ─── The matrix ─────────────────────────────────────────────────

def datasets_in(results):
    return [d for d in ["spy", *LANGS, "nemotron", "thai"] if d in results]


def headline(results):
    """Masked, each model alone, asked with its own labels."""
    cols = datasets_in(results)
    print("\n### Share of PII fully masked, model alone\n")
    print("| Model | " + " | ".join(column(d) for d in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for model in NAMES:
        cells = []
        for d in cols:
            system = f"{model}.seg" if d == "thai" and f"{model}.seg" in results[d] else model
            e = results[d].get(system)
            cells.append(f"{e['default']['masked']:.1%}" if e else "—")
        if any(c != "—" for c in cells):
            print(f"| {NAMES[model]} | " + " | ".join(cells) + " |")


def as_shipped(results):
    cols = [d for d in datasets_in(results) if d != "nemotron"]
    print("\n### Share of PII fully masked, model + pii-proxy's regex layer (Thai pack on Thai)\n")
    print("| Model | " + " | ".join(column(d) for d in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for model in NAMES:
        if model in ("regex", "presidio"):
            continue
        cells = []
        for d in cols:
            system = f"{model}+regex.seg" if d == "thai" else f"{model}+regex"
            e = results[d].get(system)
            cells.append(f"{e['default']['masked']:.1%}" if e else "—")
        if any(c != "—" for c in cells):
            print(f"| {NAMES[model]} | " + " | ".join(cells) + " |")
    cells = [f"{results[d]['regex']['default']['masked']:.1%}" if "regex" in results[d] else "—" for d in cols]
    print("| regex layer alone | " + " | ".join(cells) + " |")


# ai4privacy's buckets split into values that identify someone on their own
# and the context around them.
IDENTIFIERS = {"NAME", "EMAIL", "PHONE", "NATIONAL_ID", "ID_CARD", "PASSPORT", "DRIVER_LICENSE",
               "IP", "USERNAME", "PASSWORD", "GEOCOORD"}
CONTEXT = {"DATETIME", "LOCATION", "TITLE", "SEX"}


def identifiers_vs_context(results):
    langs = [d for d in LANGS if d in results]
    print("\n### ai4privacy: identifiers masked / context masked, model alone\n")
    print("Identifiers: names, emails, phones, ID numbers, IPs, usernames, passwords, "
          "coordinates. Context: dates, times, places, titles, gender.\n")
    print("| Model | " + " | ".join(LANG_NAMES[d] for d in langs) + " | flags that are PII |")
    print("|---" * (len(langs) + 2) + "|")
    for model in NAMES:
        if model in ("regex", "presidio") or model not in results["ai4privacy"]:
            continue
        cells = []
        for d in langs:
            ident, ctx = masked_split(d, model)
            cells.append(f"{ident:.1%} / {ctx:.1%}")
        flagged = [results[d][model]["default"]["flagged_p"] for d in langs]
        print(f"| {NAMES[model]} | " + " | ".join(cells) + f" | {min(flagged):.0%}–{max(flagged):.0%} |")


def masked_split(dataset, system):
    gold = score.load_gold(dataset)
    raw = score.load_detections(HERE / "out" / dataset / f"{system}.jsonl", gold)
    t = threshold(score.model_of(system))
    hit, n = defaultdict(int), defaultdict(int)
    for rid, g in gold.items():
        dets = [d for d in raw[rid] if d.get("score", 1.0) >= t]
        for s in g["spans"]:
            k = "id" if s["bucket"] in IDENTIFIERS else "ctx"
            n[k] += 1
            hit[k] += score.fully_masked(s, dets, g["text"])
    return hit["id"] / n["id"], hit["ctx"] / n["ctx"]


def column(dataset):
    return {"spy": "SPY", "nemotron": "Nemotron", "thai": "Thai"}.get(dataset) or LANG_NAMES[dataset]


# ─── Paired intervals ───────────────────────────────────────────

def comparisons(results):
    print("\n### Stated differences, paired bootstrap over documents (1,000 draws)\n")
    print("| Dataset | A | B | metric | A − B | 95% interval |")
    print("|---|---|---|---|---|---|")
    for dataset, a, b, metric in COMPARISONS:
        if dataset not in results or a not in results[dataset] or b not in results[dataset]:
            continue
        diff, lo, hi = paired(dataset, a, b, metric)
        print(f"| {column(dataset)} | {label(a)} | {label(b)} | {metric} | {diff * 100:+.1f}pp "
              f"| [{lo * 100:+.1f}, {hi * 100:+.1f}] |")


def paired(dataset, a, b, metric):
    gold = score.load_gold(dataset)
    test = [i for i, g in gold.items() if g["split"] == "test"]
    rows = []
    for system in (a, b):
        raw = score.load_detections(HERE / "out" / dataset / f"{system}.jsonl", gold)
        model = score.model_of(system)
        docs = []
        score.evaluate(dataset, gold, raw, test, model, score.thresholds_all(threshold(model)), docs=docs)
        rows.append(np.array(docs))
    seed = np.random.default_rng(0).integers(0, len(test), (1000, len(test)))
    da, db = (bootstrap_with(r, seed)[metric] for r in rows)
    point_a, point_b = (bootstrap_with(r, np.arange(len(test))[None, :])[metric][0] for r in rows)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return point_a - point_b, lo, hi


def bootstrap_with(docs, idx):
    sums = docs[idx].sum(axis=1)
    return {"masked": sums[:, 0] / np.maximum(sums[:, 1], 1),
            "exact_f1": score.f1(sums[:, 2], sums[:, 3], sums[:, 4]),
            "overlap_f1": score.f1(sums[:, 5], sums[:, 6], sums[:, 7])}


def threshold(model):
    return 0.0 if "+" in model else score.DEFAULT_THRESHOLD.get(model, 0.5)


# ─── SPY ────────────────────────────────────────────────────────

def spy_typed(res):
    print("\n### SPY, typed scores (test split, 7,820 documents, 52,911 spans)\n")
    print("| System | masked | exact P | exact R | exact F1 | names as returned | overlap F1 |")
    print("|---|---|---|---|---|---|---|")
    for system in order(res):
        d = res[system]["default"]
        split = res[system].get("names_as_returned", d)["exact"]["f1"]
        print(f"| {label(system)} | **{d['masked']:.1%}** | {d['exact']['p']:.1%} | {d['exact']['r']:.1%} "
              f"| {d['exact']['f1']:.1%} | {split:.1%} | {d['overlap']['f1']:.1%} |")
    if "nested" in res.get("gliner2-pii", {}):
        n = res["gliner2-pii"]["nested"]["exact"]
        print(f"\nGLiNER2-PII with its nested spans as returned: exact F1 {n['f1']:.1%}.")


def per_bucket_masked(res, dataset):
    systems = [s for s in order(res) if "." not in s and "+" not in s]
    buckets = score.per_buckets(dataset)
    print(f"\n#### {column(dataset)}: share fully masked per type, model alone\n")
    print("| System | " + " | ".join(buckets) + " |")
    print("|---" * (len(buckets) + 1) + "|")
    for system in systems:
        pb = res[system]["default"]["per_bucket"]
        print(f"| {label(system)} | " + " | ".join(f"{pb[b]['masked']:.0%}" for b in buckets) + " |")


def long_documents():
    """The same models, whole documents handed to the GLiNER library (which cuts
    at max_len) against 200-word windows."""
    print("\n### Long documents: whole text to the library vs 200-word windows (SPY test)\n")
    print("| Model | max_len | masked, windows | masked, whole text | exact F1, windows | exact F1, whole text |")
    print("|---|---|---|---|---|---|")
    gold = score.load_gold("spy")
    test = [i for i, g in gold.items() if g["split"] == "test"]
    for model, max_len in [("nvidia-gliner-pii", 384), ("pii-proxy-ner", 384), ("gliner-multi-pii", 384),
                           ("knowledgator-pii", 2048)]:
        cells = []
        for path in (HERE / "out" / "spy" / f"{model}.jsonl", HERE / "out-repro" / "spy" / f"{model}.trunc.jsonl"):
            raw = score.load_detections(path, gold)
            cells.append(score.evaluate("spy", gold, raw, test, model, score.thresholds_all(threshold(model))))
        w, t = cells
        print(f"| {NAMES[model]} | {max_len} | {w['masked']:.1%} | {t['masked']:.1%} "
              f"| {w['exact']['f1']:.1%} | {t['exact']['f1']:.1%} |")


def spy_published():
    """Our runs against GLiNER2-PII's published SPY table."""
    path = HERE / "results-repro.json"
    if not path.exists():
        return
    ours = json.loads((HERE / "results.json").read_text())["spy"]
    repro = json.loads(path.read_text())["spy"]
    print("\n### Against the published SPY table (avg of legal and medical exact F1)\n")
    print("| Model | ours, windows, native | ours, windows, SPY's names | ours, whole text, SPY's names "
          "| card | report | 100+100 draws, 5–95% |")
    print("|---|---|---|---|---|---|---|")
    for model in ["gliner2-pii", "nvidia-gliner-pii", "gliner-multi-pii", "openai-privacy-filter",
                  "knowledgator-pii"]:
        native = avg_f1(ours.get(model))
        spy7 = avg_f1(ours.get(f"{model}.spy7"))
        whole = avg_f1(repro.get(f"{model}.spy7.trunc"))
        lo, hi = draws_spread(model)
        print(f"| {NAMES[model]} | {native} | {spy7} | {whole} | {avg_of(PUBLISHED['card'].get(model))} "
              f"| {avg_of(PUBLISHED['paper'].get(model))} | {lo:.3f}–{hi:.3f} |")


def avg_f1(entry):
    if not entry:
        return "—"
    d = entry["by_domain"]
    return f"{(d['legal']['exact']['f1'] + d['medical']['exact']['f1']) / 2:.3f}"


def avg_of(pub):
    return f"{(pub[0][2] + pub[1][2]) / 2:.3f}" if pub else "—"


def draws_spread(model, n_draws=1000, size=100):
    """Avg exact F1 over random draws of `size` legal + `size` medical test documents."""
    gold = score.load_gold("spy")
    raw = score.load_detections(HERE / "out" / "spy" / f"{model}.jsonl", gold)
    by_domain = defaultdict(list)
    for domain in ("legal", "medical"):
        ids = [i for i, g in gold.items() if g["split"] == "test" and g["domain"] == domain]
        docs = []
        score.evaluate("spy", gold, raw, ids, model, score.thresholds_all(threshold(model)), docs=docs)
        by_domain[domain] = docs
    rng = random.Random(0)
    avgs = []
    for _ in range(n_draws):
        f1s = []
        for domain in ("legal", "medical"):
            rows = rng.sample(by_domain[domain], size)
            tp, fp, fn = (sum(r[k] for r in rows) for k in (2, 3, 4))
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        avgs.append(sum(f1s) / 2)
    avgs.sort()
    return avgs[int(0.05 * n_draws)], avgs[int(0.95 * n_draws)]


# ─── ai4privacy, languages, Nemotron, Thai ──────────────────────

def english_typed(results):
    if "ai4privacy" not in results:
        return
    res = results["ai4privacy"]
    print("\n### ai4privacy English (exp 006's 1,000 records, 6,357 spans)\n")
    print("| System | masked | 95% | overlap P | overlap R | overlap F1 | 95% | exact F1 |")
    print("|---|---|---|---|---|---|---|---|")
    for system in order(res):
        d, ci = res[system]["default"], res[system]["ci"]
        print(f"| {label(system)} | **{d['masked']:.1%}** | {span(ci['masked'])} | {d['overlap']['p']:.1%} "
              f"| {d['overlap']['r']:.1%} | {d['overlap']['f1']:.1%} | {span(ci['overlap_f1'])} "
              f"| {d['exact']['f1']:.1%} |")
    per_bucket_masked(res, "ai4privacy")


def languages_typed(results):
    langs = [d for d in LANGS if d in results]
    if len(langs) < 2:
        return
    print("\n### ai4privacy by language: overlap F1 (masked share in the first table)\n")
    print("| Model | " + " | ".join(LANG_NAMES[d] for d in langs) + " |")
    print("|---" * (len(langs) + 1) + "|")
    for model in NAMES:
        cells = [f"{results[d][model]['default']['overlap']['f1']:.1%}" if model in results[d] else "—"
                 for d in langs]
        if any(c != "—" for c in cells):
            print(f"| {NAMES[model]} | " + " | ".join(cells) + " |")


def nemotron(res):
    print("\n### Nemotron-PII held-out (in-distribution for both Nemotron-trained models)\n")
    for system in ("pii-proxy-ner.full", "nvidia-gliner-pii.full"):
        if system in res and "fine" in res[system]:
            f = res[system]["fine"]
            print(f"- {label(system)}, 55 native labels, overlap + exact label: "
                  f"F1 {f['f1']:.1%} (P {f['p']:.1%}, R {f['r']:.1%})")
    if {"pii-proxy-ner.full", "nvidia-gliner-pii.full"} <= set(res):
        lo, hi = fine_difference()
        print(f"- difference, paired bootstrap over records: 95% interval [{lo * 100:+.1f}, {hi * 100:+.1f}]pp")
    print("\n| System | masked | coarse overlap F1 | coarse exact F1 |")
    print("|---|---|---|---|")
    for system in order(res):
        d = res[system]["default"]
        print(f"| {label(system)} | {d['masked']:.1%} | {d['overlap']['f1']:.1%} | {d['exact']['f1']:.1%} |")


def fine_difference(draws=2000):
    """pii-proxy-ner minus NVIDIA on Nemotron's 55 native labels (score.fine_labels' metric)."""
    gold = {r["id"]: r["spans"] for r in map(json.loads, (HERE / "data" / "nemotron.jsonl").open())}
    rows = []
    for system in ("pii-proxy-ner.full", "nvidia-gliner-pii.full"):
        raw = {r["id"]: r["detections"] for r in map(json.loads, (HERE / "out" / "nemotron" / f"{system}.jsonl").open())}
        per = []
        for rid, spans in gold.items():
            counts = score.Counts()
            score.match([{**g, "bucket": g["label"]} for g in spans],
                        [{**d, "buckets": (d["type"],)} for d in raw[rid] if d["score"] >= 0.5],
                        False, counts, defaultdict(score.Counts))
            per.append(counts.row())
        rows.append(np.array(per))
    idx = np.random.default_rng(0).integers(0, len(gold), (draws, len(gold)))
    a, b = (score.f1(*r[idx].sum(axis=1).T) for r in rows)
    return np.percentile(a - b, [2.5, 97.5])


def thai(res):
    print("\n### Thai (324 texts, 774 spans)\n")
    print("| System | masked | 95% | names masked | IDs masked | overlap F1 | exact F1 |")
    print("|---|---|---|---|---|---|---|")
    for system in order(res):
        d, ci = res[system]["default"], res[system]["ci"]
        pb = d["per_bucket"]
        print(f"| {label(system)} | **{d['masked']:.1%}** | {span(ci['masked'])} | {pb['NAME']['masked']:.1%} "
              f"| {pb['ID_NUM']['masked']:.1%} | {d['overlap']['f1']:.1%} | {d['exact']['f1']:.1%} |")


# ─── Speed and size ─────────────────────────────────────────────

def label_cost():
    path = HERE / "latency-labels.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    print("\n### What labels cost: same container, same 20 SPY documents, CPU (2 cores)\n")
    print("| Model | labels asked | median per document |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {NAMES[r['model']]} | {r['labels']} | {r['ms_median'] / 1000:.2f} s |")


def speed():
    lat = HERE / "latency-server-class.json"
    facts = HERE / "facts.json"
    if not lat.exists():
        return
    rows = {r["model"]: r for r in json.loads(lat.read_text()) if isinstance(r, dict) and "model" in r}
    params = json.loads(facts.read_text())["params"] if facts.exists() else {}
    print("\n### CPU latency, 2 CPU cores, every model in one container, one SPY document (~450 words), full label vocabulary\n")
    print("| Model | labels asked | median | p90 | parameters |")
    print("|---|---|---|---|---|")
    for model in NAMES:
        if model in rows:
            r = rows[model]
            p = f"{params[model] / 1e6:.0f}M" if model in params else "—"
            print(f"| {NAMES[model]} | {r.get('labels', '—')} | {r['ms_median'] / 1000:.2f} s "
                  f"| {r['ms_p90'] / 1000:.2f} s | {p} |")


# ─── Naming ─────────────────────────────────────────────────────

def order(res):
    """Models alone (dataset vocabulary) first, most masked first; then variants."""
    def key(system):
        return (("+" in system, "." in system and not system.endswith(".seg")),
                -res[system]["default"]["masked"])
    return sorted(res, key=key)


def span(ci):
    return f"{ci[0]:.1%}–{ci[1]:.1%}"


def label(system):
    base, *flags = system.split(".")
    model, plus, _ = base.partition("+")
    text = {**NAMES, **VARIANT_NAMES}.get(model, model) + (" + regex layer" if plus else "")
    if "full" in flags:
        text += " (full vocabulary)"
    if "spy7" in flags:
        text += " (SPY's 7 names)"
    if "trunc" in flags:
        text += ", whole text to the library"
    if "seg" in flags:
        text += ", segmented"
    return text


if __name__ == "__main__":
    main()
