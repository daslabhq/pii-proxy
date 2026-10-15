#!/usr/bin/env python3
"""Build the evaluation sets for experiment 007 as one record format.

    data/<name>.jsonl   {id, text, spans: [{start, end, label}], split, domain}

  spy         SPY (Savkin et al., NAACL SRW 2025), legal + medical, all 8,688
              documents. Entities filled with Faker, seeded (see build_spy).
  ai4privacy  exp 006's sample: pii-masking-300k validation, English, n=1000,
              seed 42 — the identical records, rebuilt the same way.
  ai4privacy-<de|fr|it|es|nl>
              the same split's other five languages, n=1000 each, drawn the
              same way (built on Modal: modal run modal_run.py --build-langs).
  nemotron    Nemotron-PII `test` split, first 500 records with spans — the
              set behind the main README's 91.5% / 90.9% head-to-head
              (fingerprint checked against that run's receipt).

The Thai set is built by build_thai.ts (it uses the Thai pack's generators).

    .venv/bin/python build_data.py            # all three
    .venv/bin/python build_data.py spy        # one
"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"

SPY_FILES = {
    "legal": "https://huggingface.co/datasets/mks-logic/SPY/resolve/main/data/legal_questions_placeholders.jsonl",
    "medical": "https://huggingface.co/datasets/mks-logic/SPY/resolve/main/data/medical_consultations_placeholders.jsonl",
}
SEED = 42
DEV_FRACTION = 0.1  # thresholds are tuned on dev, every reported number is on test

NEMOTRON_FINGERPRINT = "df099d8d5ff3734617e977ecaccd86cc7b7716841a73952158899b0115e746d5"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    wanted = sys.argv[1:] or ["spy", "ai4privacy", "nemotron"]
    builders = {"spy": build_spy, "ai4privacy": build_ai4privacy, "nemotron": build_nemotron}
    for name in wanted:
        records = builders[name]()
        write(name, records)


def write(name: str, records: list) -> None:
    path = DATA / f"{name}.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = Counter(s["label"] for r in records for s in r["spans"])
    splits = Counter(r["split"] for r in records)
    print(f"{name}: {len(records)} records {dict(splits)}, {sum(labels.values())} spans -> {path}")
    for label, n in labels.most_common():
        print(f"  {label:28s} {n}")


# ─── SPY ────────────────────────────────────────────────────────

def build_spy() -> list:
    """SPY ships documents with placeholders (<NAME>, <EMAIL>, ...) and a loader
    that fills them with Faker at load time. We fill them the way that loader
    does — one value per type per document, the same Faker functions, split
    evenly across functions and shuffled — but seed Python's `random` as well
    as Faker (the official loader seeds only Faker, so its shuffle differs per
    run) and build the text and char offsets directly from the placeholder
    tokens, whose own trailing whitespace is kept. The official loader's
    `trailing_whitespaces` flags are misaligned with its tokens, so a text
    rebuilt from them doubles or drops spaces; we never read them."""
    import urllib.request
    records = []
    for domain, url in SPY_FILES.items():
        cache = DATA / f"spy-{domain}-placeholders.jsonl"
        if not cache.exists():
            urllib.request.urlretrieve(url, cache)
        rows = [json.loads(line) for line in cache.open()]
        values = spy_entity_values(len(rows), seed=SEED)
        dev = set(random.Random(SEED).sample(range(len(rows)), int(len(rows) * DEV_FRACTION)))
        for i, row in enumerate(rows):
            text, spans = fill_placeholders(row["tokens"], row["ent_tags"], values[i])
            records.append({"id": f"spy-{domain}-{i}", "text": text, "spans": spans,
                            "split": "dev" if i in dev else "test", "domain": domain})
    return records


def spy_entity_values(n: int, seed: int) -> list:
    """One dict of {TYPE: value} per document, as SPY.py's generate_profiles."""
    from faker import Faker
    Faker.seed(seed)
    random.seed(seed)
    faker = Faker()
    funcs = {
        "EMAIL": [faker.ascii_email, faker.ascii_free_email],
        "NAME": [faker.name],
        "URL": [lambda: faker.uri(deep=1), lambda: faker.uri(deep=2)],
        "PHONE_NUM": [faker.phone_number],
        "ID_NUM": [faker.ripe_id, faker.msisdn, faker.ssn, faker.sbn9, faker.isbn10,
                   faker.isbn13, faker.credit_card_number, faker.aba, faker.bban, faker.iban],
        "ADDRESS": [faker.street_address],
        "USERNAME": [faker.user_name],
    }
    columns = {t: n_samples(fs, n) for t, fs in funcs.items()}
    # SPY.py splits each value on whitespace and rejoins the words with single spaces.
    return [{t: " ".join(columns[t][i].split()) for t in funcs} for i in range(n)]


def n_samples(funcs, n):
    sizes = [n // len(funcs) + (i < n % len(funcs)) for i in range(len(funcs))]
    out = [f() for f, size in zip(funcs, sizes) for _ in range(size)]
    random.shuffle(out)
    return out


def fill_placeholders(tokens, tags, values):
    """About 4% of placeholders run straight into the next word ("<PHONE_NUM>for"):
    the source lost a space there, so one is put back — only before a letter or
    digit, never before punctuation."""
    text, spans = "", []
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            label = tag[2:]
            placeholder = f"<{label}>"
            assert tok.startswith(placeholder), (tok, tag)
            value = values[label]
            spans.append({"start": len(text), "end": len(text) + len(value), "label": label})
            rest = tok[len(placeholder):]
            glued = not rest and i + 1 < len(tokens) and tokens[i + 1][:1].isalnum()
            text += value + (" " if glued else rest)
        else:
            assert tag == "O", tag  # the placeholder files carry only B- tags
            text += tok
    return text, spans


# ─── ai4privacy (exp 006's sample) ──────────────────────────────

LANGUAGES = {"de": "german", "fr": "french", "it": "italian", "es": "spanish", "nl": "dutch"}


def build_ai4privacy() -> list:
    return sample_ai4privacy(lambda language: language.startswith("en"), "ai4privacy")


def build_ai4privacy_language(code: str) -> list:
    names = {code, LANGUAGES[code]}
    return sample_ai4privacy(lambda language: language in names, f"ai4privacy-{code}")


def sample_ai4privacy(is_language, domain: str, n: int = 1000) -> list:
    """n validation records of one language, seed 42 (exp 006's draw for English)."""
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-300k", split="validation")
    subset = ds.filter(lambda r: is_language(r["language"].lower()))
    if len(subset) < n:
        raise ValueError(f"{domain}: {len(subset)} records; language values {Counter(ds['language'])}")
    records = []
    for rec in subset.select(random.Random(SEED).sample(range(len(subset)), n)):
        spans = [{"start": m["start"], "end": m["end"], "label": m["label"]} for m in rec["privacy_mask"]]
        records.append({"id": rec["id"], "text": rec["source_text"], "spans": spans,
                        "split": "test", "domain": domain})
    return records


# ─── Nemotron-PII held-out ──────────────────────────────────────

def build_nemotron() -> list:
    """Identical to head_to_head_eval._load_split('test', max_records=500)."""
    import ast
    from datasets import load_dataset
    ds = load_dataset("nvidia/Nemotron-PII", split="test", streaming=True)
    records = []
    for row in ds:
        raw = ast.literal_eval(row["spans"]) if isinstance(row["spans"], str) else row["spans"]
        spans = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in raw]
        if spans:
            records.append({"id": f"nemotron-{len(records)}", "text": row["text"], "spans": spans,
                            "split": "test", "domain": (row.get("domain") or "").lower()})
        if len(records) >= 500:
            break
    h = hashlib.sha256()
    for r in records:
        h.update(r["text"].encode("utf-8"))
        h.update(b"\x00")
    assert h.hexdigest() == NEMOTRON_FINGERPRINT, f"not the head-to-head set: {h.hexdigest()}"
    print(f"nemotron: fingerprint matches the head-to-head receipt ({NEMOTRON_FINGERPRINT[:16]})")
    return records


if __name__ == "__main__":
    main()
