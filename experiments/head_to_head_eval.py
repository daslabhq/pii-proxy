#!/usr/bin/env python3
"""
Head-to-head: full-55-label pii-proxy (checkpoint-31220) vs nvidia/gliner-PII,
on the EXACT same clean held-out `test`-split records.

Reproduces modal_train.py's clean-test methodology so pii-proxy's fine F1 should
land ~91.5% (a built-in sanity check that we loaded the same records + checkpoint),
then feeds the SAME in-memory records to nvidia/gliner-PII with two query variants
(native underscore labels + natural-language) and reports NVIDIA's best.

Deploy + fire-and-forget (survives connection drops):
    modal deploy experiments/head_to_head_eval.py
    modal run --detach experiments/head_to_head_eval.py::run   # or .spawn from python

Output lands at /models/head_to_head_result.json on the pii-proxy-models volume.
"""
import modal

app = modal.App("pii-proxy-head-to-head")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "gliner", "datasets", "accelerate",
                 "sentencepiece", "huggingface_hub")
)

volume = modal.Volume.from_name("pii-proxy-models", create_if_missing=True)
MODEL_DIR = "/models"
DATASET = "nvidia/Nemotron-PII"
CKPT = f"{MODEL_DIR}/checkpoint/checkpoint-31220"

FINE_TO_COARSE = {
    "first_name": "person_name", "last_name": "person_name", "middle_name": "person_name",
    "medical_record_number": "medical_record",
    "date_of_birth": "date_of_birth",
    "date": "date", "date_time": "date", "time": "date",
    "email": "email", "phone_number": "phone",
    "street_address": "location", "city": "location", "state": "location",
    "county": "location", "zip_code": "location", "country": "location",
    "ssn": "national_id", "health_plan_beneficiary_number": "insurance_id",
    "certificate_license_number": "national_id", "biometric_identifier": "national_id",
    "url": "url", "unique_id": "national_id", "blood_type": "medical_info",
    "pin": "national_id", "password": "credential", "swift_bic": "financial_id",
}


# ── Dataset loading (identical to modal_train._load_split, clean `test` split) ──

def _load_split(split, *, health_only=False, max_records=None, scan_limit=100_000):
    import ast
    from datasets import load_dataset
    ds = load_dataset(DATASET, split=split, streaming=True)
    records, total = [], 0
    for row in ds:
        total += 1
        domain = (row.get("domain", "") or "").lower()
        if health_only and "health" not in domain:
            if total >= scan_limit:
                break
            continue
        spans_raw = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
        gold = [{"start": s["start"], "end": s["end"], "text": s["text"], "label": s["label"]}
                for s in spans_raw]  # full_labels=True: keep every label
        if gold:
            records.append({"text": row["text"], "goldSpans": gold})
        if max_records and len(records) >= max_records:
            break
        if total >= scan_limit:
            break
    return records


def _fingerprint(records):
    """SHA256 over concatenated record texts — a receipt that both models saw the same set."""
    import hashlib
    h = hashlib.sha256()
    for r in records:
        h.update(r["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _assert_no_leakage(eval_sets, scan_limit=100_000):
    """Load train texts and assert zero overlap with each eval set."""
    train_texts = {r["text"] for r in _load_split("train", scan_limit=scan_limit)}
    report = {}
    total_overlap = 0
    for name, recs in eval_sets.items():
        overlap = sum(1 for r in recs if r["text"] in train_texts)
        total_overlap += overlap
        report[name] = {"records": len(recs), "overlap_with_train": overlap}
    report["train_texts_scanned"] = len(train_texts)
    report["total_overlap"] = total_overlap
    report["passed"] = (total_overlap == 0)
    assert total_overlap == 0, f"LEAKAGE: {total_overlap} eval records found in train"
    return report


# ── Evaluation (span-overlap + exact-label match, threshold 0.5) ──

def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "p": p, "r": r,
            "tp": tp, "fp": fp, "fn": fn}


def _eval_fine(model, recs, query_labels, label_map=None):
    """Query with query_labels; map predicted labels back to canonical via label_map."""
    tp = fp = fn = 0
    for rec in recs:
        ents = model.predict_entities(rec["text"], query_labels, threshold=0.5)
        pred = []
        for e in ents:
            lab = label_map[e["label"]] if label_map else e["label"]
            pred.append({"start": e["start"], "end": e["end"], "label": lab})
        gold = rec["goldSpans"]
        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm:
                    continue
                if p["label"] == g["label"] and _overlap(p, g):
                    pm.add(pi); gm.add(gi); tp += 1; break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)
    return _f1(tp, fp, fn)


def _eval_coarse(model, recs, query_labels, mapping, label_map=None):
    tp = fp = fn = 0
    for rec in recs:
        ents = model.predict_entities(rec["text"], query_labels, threshold=0.5)
        pred = []
        for e in ents:
            lab = label_map[e["label"]] if label_map else e["label"]
            c = mapping.get(lab)
            if c:
                pred.append({"start": e["start"], "end": e["end"], "label": c})
        gold = [{**s, "label": mapping[s["label"]]} for s in rec["goldSpans"] if s["label"] in mapping]
        gm, pm = set(), set()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gi in gm:
                    continue
                if p["label"] == g["label"] and _overlap(p, g):
                    pm.add(pi); gm.add(gi); tp += 1; break
        fp += sum(1 for pi in range(len(pred)) if pi not in pm)
        fn += sum(1 for gi in range(len(gold)) if gi not in gm)
    return _f1(tp, fp, fn)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(eval_size: int = 500, health_size: int = 200):
    import json, gc, time
    import torch
    from gliner import GLiNER

    t_start = time.time()

    # ── 1. Build ONE clean test set (identical records fed to both models) ──
    LABELS = json.load(open(f"{MODEL_DIR}/result.json"))["labels"]  # the exact 55 training labels
    print(f"Loaded {len(LABELS)} native labels from result.json")

    all_domain = _load_split("test", max_records=eval_size, scan_limit=300_000)
    healthcare = _load_split("test", health_only=True, max_records=health_size, scan_limit=300_000)
    print(f"Test set: {len(all_domain)} all-domain + {len(healthcare)} healthcare (clean `test` split)")

    fp_all = _fingerprint(all_domain)
    fp_health = _fingerprint(healthcare)

    # ── 2. Leakage / same-records assertion ──
    leakage = _assert_no_leakage({"all_domain": all_domain, "healthcare": healthcare})
    print(f"Leakage check: {leakage}")

    # Natural-language query variant for NVIDIA (deterministic underscore->space)
    nl_labels = [l.replace("_", " ") for l in LABELS]
    nl_back = {nl: canon for nl, canon in zip(nl_labels, LABELS)}

    # ── 3. pii-proxy (checkpoint-31220), native 55-label query ──
    print(f"\nLoading pii-proxy from {CKPT} ...")
    pp = GLiNER.from_pretrained(CKPT, load_tokenizer=True).to("cuda")
    pp.eval()
    pp_fine = _eval_fine(pp, all_domain, LABELS)
    pp_coarse = _eval_coarse(pp, all_domain, LABELS, FINE_TO_COARSE)
    pp_health = _eval_fine(pp, healthcare, LABELS)
    print(f"pii-proxy fine {pp_fine['f1']*100:.2f}% | coarse {pp_coarse['f1']*100:.2f}% | health {pp_health['f1']*100:.2f}%")
    del pp
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ── 4. nvidia/gliner-PII, native + natural-language query variants ──
    print("\nLoading nvidia/gliner-PII ...")
    nv = GLiNER.from_pretrained("nvidia/gliner-PII").to("cuda")
    nv.eval()
    nv_fine_native = _eval_fine(nv, all_domain, LABELS)
    nv_fine_nl = _eval_fine(nv, all_domain, nl_labels, label_map=nl_back)
    nv_coarse = _eval_coarse(nv, all_domain, LABELS, FINE_TO_COARSE)
    nv_health_native = _eval_fine(nv, healthcare, LABELS)
    nv_health_nl = _eval_fine(nv, healthcare, nl_labels, label_map=nl_back)
    print(f"NVIDIA fine native {nv_fine_native['f1']*100:.2f}% | nl {nv_fine_nl['f1']*100:.2f}%")

    nv_best_fine = max(nv_fine_native, nv_fine_nl, key=lambda x: x["f1"])
    nv_best_variant = "native" if nv_best_fine is nv_fine_native else "natural_language"
    nv_best_health = max(nv_health_native, nv_health_nl, key=lambda x: x["f1"])

    result = {
        "checkpoint": CKPT,
        "nvidia_model": "nvidia/gliner-PII",
        "eval_split": "test",
        "threshold": 0.5,
        "match_rule": "span-overlap + exact-label match",
        "records": {"all_domain": len(all_domain), "healthcare": len(healthcare)},
        "fingerprint_sha256": {"all_domain": fp_all, "healthcare": fp_health},
        "leakage_check": leakage,
        "labels_piiproxy_native": LABELS,
        "labels_nvidia_native": LABELS,
        "labels_nvidia_natural_language": nl_labels,
        "piiproxy": {"fine": pp_fine, "coarse": pp_coarse, "health_fine": pp_health},
        "nvidia": {
            "fine_native": nv_fine_native,
            "fine_natural_language": nv_fine_nl,
            "fine_best": nv_best_fine,
            "fine_best_variant": nv_best_variant,
            "coarse": nv_coarse,
            "health_fine_native": nv_health_native,
            "health_fine_nl": nv_health_nl,
            "health_fine_best": nv_best_health,
        },
        "delta_fine_pp": (pp_fine["f1"] - nv_best_fine["f1"]) * 100,
        "delta_health_pp": (pp_health["f1"] - nv_best_health["f1"]) * 100,
        "eval_seconds": time.time() - t_start,
    }

    with open(f"{MODEL_DIR}/head_to_head_result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()

    print("\n" + "=" * 64)
    print("HEAD-TO-HEAD (identical clean `test`-split records)")
    print(f"  pii-proxy full-55  fine {pp_fine['f1']*100:.2f}%  coarse {pp_coarse['f1']*100:.2f}%")
    print(f"  nvidia/gliner-PII  fine {nv_best_fine['f1']*100:.2f}%  ({nv_best_variant})")
    print(f"  delta (fine): {result['delta_fine_pp']:+.2f}pp")
    print("=" * 64)
    return result


@app.local_entrypoint()
def main(eval_size: int = 500, health_size: int = 200):
    import json
    print(json.dumps(run.remote(eval_size=eval_size, health_size=health_size), indent=2))
