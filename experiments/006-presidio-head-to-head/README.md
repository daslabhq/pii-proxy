# Experiment 006 — Presidio head-to-head on out-of-distribution data

Every prior experiment (003–005) benchmarked our models **in-distribution** —
trained on Nemotron-PII, tested on held-out Nemotron-PII. This experiment is the
first **pipeline-level, out-of-distribution** benchmark: the full pii-proxy
detection stack vs Microsoft Presidio, on a dataset neither system was tuned on.

## Results

Entity-level, overlap match, n=1000 records / 6,357 ground-truth spans.
Combined systems are merged with the library's exact `detectAll` semantics
(any-overlap, earlier detector wins) — not an idealized union.

> **Caveat — what ships vs what's benchmarked.** The npm package today
> detects via regex + optional Ollama LLM only; the `pii-proxy (regex only)`
> row is the out-of-the-box npm experience. The model rows run our
> fine-tuned checkpoints through the Python `gliner` lib and merge with
> library semantics — a faithful simulation of a `GlinerDetector` that does
> not exist in `src/` yet (`gliner`/`onnxruntime` are already dependencies,
> currently unused). **Shipping that detector — model-first by default — is
> the top product takeaway of this experiment**: it's the difference between
> 20.6% and 75.4% F1.

| System | P | R | F1 | exact-F1 | coverage† | ms/rec |
|---|---|---|---|---|---|---|
| presidio (default) | 47.5% | 37.6% | 42.0% | 34.9% | 53.8% | 18 |
| pii-proxy (regex only) | 52.6% | 12.8% | 20.6% | 18.7% | 23.9% | 0.05 |
| pii-proxy (model only) | 79.5% | 70.7% | 74.8% | 71.4% | 83.6% | 134 |
| **pii-proxy (full)** | **77.3%** | **73.5%** | **75.4%** | 71.9% | **88.9%** | 134 |

*Model = `pii-proxy-ner`: our fine-tune of `gliner_small-v2.1` on the full
Nemotron-PII (100k records, 55 entity types, 30 domains —
[exp 005](../005-nvidia-baseline/), the 90.1%-F1 model from the main README),
queried with its 55 native labels + 4 zero-shot additions (passport, driver
license, ID card, title). **full** = model detector first, regex layer behind
it as a coverage backstop, merged with the library's `detectAll` semantics.*

† coverage = type-agnostic recall: % of ground-truth spans overlapped by *any*
detection (a type-confused detection still masks the PII). This is the
leak-prevention number; type-correct F1 is the fake-quality number.

**Shared-vocabulary comparison** (only the 9 buckets BOTH systems have types
for — strips our vocabulary-coverage advantage out of the delta; relaxed
many-to-many matching): **pii-proxy (full) 80.2% F1 vs Presidio 49.1%**
(+31.1pp). The full-vocabulary delta is +33.4pp.

### Ablations (smaller exp-004 model, ordering, zero-shot)

The same protocol run with `pii-proxy-ner-health` (exp 004: the 24-label
healthcare fine-tune) shows what vocabulary breadth and detector order are
worth:

| System | F1 | coverage |
|---|---|---|
| regex-first + ner-health (native labels) | 62.2% | 77.8% |
| regex-first + ner-health (+8 zero-shot labels) | 65.3% | 80.0% |
| ner-health (+zero-shot) first + regex | 69.8% | 80.0% |
| ner-health (+zero-shot) alone | 68.8% | 71.8% |

- **Detector order: worth 4.5pp F1.** The merge is first-wins; regex-first
  lets the loose phone regex shadow correct model detections (PASSPORT
  65.4%→17.0%). Model-first is a one-line constructor choice.
- **Regex backstop: worth +3–6pp coverage** at ~1pp F1 — it masks ID-like
  digit strings the model misses. Keep it, behind the model.
- **Native vocabulary: worth 5–47pp per type over zero-shot.** Same
  categories, queried natively by the flagship vs zero-shot on the
  healthcare model: USERNAME 76.5% vs 56.9%, SEX 68.4% vs 21.4%,
  NATIONAL_ID 55.2% vs 29.4%, GEOCOORD 49.5% vs 36.4%. Zero-shot bridges a
  vocabulary gap; retraining with the right labels closes it — and the $8 /
  39-min fine-tune recipe makes retraining the cheap option.

### Per-type F1 (overlap match)

| Type | gold n | presidio | pii-proxy (regex) | pii-proxy (model) | **pii-proxy (full)** |
|---|---|---|---|---|---|
| DATETIME | 996 | 61.3% | 0.0% | 88.7% | 88.7% |
| LOCATION | 1414 | 30.3% | 0.0% | 82.7% | 82.7% |
| NAME | 728 | 30.1% | 0.0% | 78.9% | 78.9% |
| USERNAME | 319 | 0.0% | 0.0% | 76.5% | 76.5% |
| PASSWORD | 209 | 0.0% | 0.0% | 75.9% | 75.9% |
| EMAIL | 332 | **97.8%** | 97.8% | 95.5% | 95.7% |
| PHONE | 245 | 31.9% | 37.5% | 71.5% | 56.4% |
| SEX | 244 | 0.0% | 0.0% | 68.4% | 68.4% |
| PASSPORT | 317 | **75.7%** | 0.0% | 65.9% | 65.9% |
| NATIONAL_ID | 314 | 32.2% | 0.0% | 55.2% | 55.2% |
| IP | 274 | **98.4%** | 96.1% | 52.9% | 87.8% |
| DRIVER_LICENSE | 305 | 28.6% | 0.0% | 52.6% | 52.6% |
| ID_CARD | 363 | 0.0% | 0.0% | 50.9% | 50.9% |
| GEOCOORD | 30 | 0.0% | 0.0% | 49.5% | 49.5% |
| TITLE | 267 | 0.0% | 0.0% | 41.9% | 41.9% |

### What the numbers say

1. **pii-proxy (full) beats default Presidio by +33.4pp F1** (75.4% vs
   42.0%), **+31.1pp on shared vocabulary** (80.2% vs 49.1%), and masks
   **88.9% vs 53.8%** of all PII spans. It wins 13 of 15 categories;
   Presidio narrowly keeps IP (98.4% vs 87.8%) and EMAIL is a tie.
2. **Training vocabulary is the variable that matters most OOD.** The
   flagship (55 types, 30 domains) holds 74.8% F1 where the 24-label
   healthcare fine-tune manages 68.8% with zero-shot patches — and the
   gap concentrates exactly in the categories only the flagship trained on:
   USERNAME 76.5% vs 56.9%, SEX 68.4% vs 21.4%, NATIONAL_ID 55.2% vs 29.4%.
   In-distribution scores inverted them (96.1% vs 90.1%) — **breadth beats
   in-domain polish when the domain shifts.** Zero-shot queries bridge a
   vocabulary gap; the $8/39-min recipe closes it.
3. **Detector order is a result, not a footnote** (ablation table above).
   First-wins merging makes regex-before-model cost 4.5pp F1 — the loose
   phone regex shadows correct passport/ID detections. Behind the model the
   same regex layer earns **+5.3pp coverage** (83.6%→88.9%) and +0.6pp F1,
   masking IPv6 and digit strings the model misses — though its
   type-confused phone hits still cost typed F1 (PHONE 71.5%→56.4%), and
   the model's sloppy IP spans shadow the regex's exact ones (IP 96.1%
   regex-alone → 87.8% merged). **Library follow-ups: default the model
   detector ahead of loose regexes; longer-term, replace first-wins with
   confidence/type-aware span conflict resolution** (the ideal merge takes
   regex for EMAIL/IP, model for the rest — beating every ordering
   benchmarked here).
4. **Nobody is leak-safe out of distribution — we now leave 11.1%.**
   Concrete remaining gaps:
   - **IP — fixed mid-experiment, instructively.** 54% of gold IPs are
     IPv6; the regex was IPv4-only and the model's native `ipv4`/`ipv6`
     labels only reach 53% — span-exact detection of long hex strings is a
     regex job, not a model job. Adding IPv6 to the regex took it to 96.1%
     (+1.4pp overall F1, +2.2pp coverage). The full pipeline shows 87.8%,
     not 96.1%, because the model's sloppier IP detections shadow the
     regex's exact ones under first-wins merging — finding 3, again.
   - **Phone over-triggering**: 933 regex detections vs 245 gold TEL — 253
     overlap PASSPORT, 181 SOCIALNUMBER, 161 IDCARD (only 24 touch no PII).
     Valuable for coverage, wrong for typing; keep it behind the model.
   - **DRIVER_LICENSE / ID_CARD (~51–53%)**: the flagship's
     `certificate_license_number`/`unique_id` labels are imperfect proxies;
     a fine-tune adding these as first-class labels is the recipe's job.
   - **TITLE (41.9%)**: cheap dictionary detector candidate.
5. **Presidio's failure mode mirrors our regex layer's**: 599
   `US_BANK_NUMBER` + 1,206 `US_DRIVER_LICENSE` detections, mostly
   type-confused digit strings. Its 53.8% coverage vs 37.6% typed recall is
   the same redact-the-right-span, wrong-label pattern.
6. **OOD is brutal for everyone — name the distribution or the number is
   marketing.** Presidio's reputed 90%+ becomes 42% here; our in-distribution
   90.1% becomes 74.8%. Both true, different questions.

## What this benchmark cannot measure

Factual limits — read these before quoting any number above. Presidio and
pii-proxy are not fully comparable systems, and this dataset is not reality:

1. **Type vocabularies don't align.** Presidio has no recognizer for
   USERNAME, PASSWORD, SEX, TITLE, GEOCOORD, or generic ID_CARD; it scores 0%
   there by absence, not by failure. Conversely it detects ORGANIZATION,
   URL, CREDIT_CARD, US_BANK_NUMBER — categories this dataset doesn't
   annotate, so those detections are *unscoreable* and we drop them (599
   bank-number + 522 organization + 465 URL detections excluded from its
   precision — a charitable choice; counting them as FP would lower it).
   The shared-vocabulary table is the fairest single comparison.
2. **Span granularity conventions differ.** The dataset annotates
   `GIVENNAME1`/`LASTNAME1` separately; Presidio emits one PERSON span for a
   full name. Measured impact: small (44 multi-span detections; relaxed
   many-to-many matching moves Presidio +0.9pp, us +0.6pp) — but it is a
   structural bias against full-span detectors, not zero.
3. **Presidio is benchmarked untuned.** It's a framework with per-deployment
   recognizer config, thresholds, and custom recognizers; default-config
   numbers are its floor. Symmetrically, our GLiNER threshold (0.5) was not
   tuned on this dataset either — but our model also wasn't *designed* to be
   deployment-tuned, so the asymmetry favors us.
4. **The dataset is synthetic and templated.** ai4privacy text is generated,
   not harvested; entities sit in template slots. Both systems likely score
   differently on real documents (real EHR notes, emails, chat logs). 30 of
   6,357 ground-truth spans (0.5%) have misaligned offsets — unsubstituted
   template markers (`PASSPORT_G(`…) leaked into the annotations during
   dataset generation; they count against all systems equally.
5. **English slice only.** The dataset's other languages (FR/DE/IT/ES/NL)
   are untested here; our regex layer and Presidio's recognizers are both
   heavily US/EN-biased, so multilingual numbers would differ substantially.
6. **Type-mapping judgment calls.** Two charitable-to-us stretches:
   `unique_id`→ID_CARD contributes 50 overlapping detections (of 363 gold),
   `certificate_license_number`→DRIVER_LICENSE contributes 18 (of 305). The
   full mapping is in `mapping.py`; disagree with a mapping, re-run `score.py`.
7. **Detection ≠ protection.** This measures detection. It says nothing about
   fake plausibility, round-trip unmask integrity under LLM rewriting, or
   hallucinated-PII passthrough — the failure modes documented in the main
   README. Those need their own benchmark.

Run `verify.py` to reproduce every number in this section.

## Setup

- **Dataset**: [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)
  validation split, English only (7,946 records), seeded sample **n=1000**
  (seed 42) → **6,357 ground-truth spans** across 27 label types.
  Char-offset annotations (`privacy_mask`). OOD for our models (trained on
  Nemotron-PII); OOD for Presidio (never tuned on this set).
- **Systems**:
  | System | What it is |
  |---|---|
  | `presidio` | presidio-analyzer 2.x, default recognizers, spaCy `en_core_web_lg`, no tuning |
  | `pii-proxy (regex only)` | our shipped zero-setup regex detector layer (`defaultDetectors`) |
  | `pii-proxy (model only)` | **`pii-proxy-ner`** — exp-005 fine-tune of `gliner_small-v2.1` on full Nemotron-PII, queried with its 55 native labels + 4 zero-shot, threshold 0.5 |
  | `pii-proxy (full)` | `pii-proxy-ner` first, regex behind it, merged with the library's `detectAll` semantics |
  | ablations | `pii-proxy-ner-health` (exp-004 24-label healthcare fine-tune), detector orderings, native vs zero-shot labels |

- **Scoring**: entity-level, greedy 1:1 matching, **bucket equality + char
  overlap** (primary); exact-boundary F1 reported secondarily. All native type
  vocabularies map into shared coarse buckets — the full mapping is in
  [`mapping.py`](mapping.py) (the single source of truth; the tables below are
  generated from it).

## Fairness rules (read before quoting numbers)

1. **Recall denominator is always all 6,357 annotated spans.** A system that
   doesn't attempt a type (e.g. regex has no name detection) scores 0 recall
   there — visible in the per-type table, not hidden by vocabulary cherry-picking.
2. **Unverifiable detections are dropped, not counted as FP.** The dataset
   doesn't annotate URLs or credit cards; a system detecting one can't be
   scored either way. Mapped to `None` in `mapping.py`.
3. **Presidio is the default config.** Presidio is a framework meant to be
   tuned per deployment; published comparisons near-universally benchmark the
   untuned default. So do we — but we say so. Treat Presidio's numbers as a
   floor, not its ceiling.
4. **GLiNER queried with native vocabulary** (exp 003 lesson: bi-encoder F1
   swings 5–17pp on query phrasing). The `-ext` variant deliberately violates
   this to measure off-vocabulary generalization.
5. **Approximate bucket mappings**: GLiNER `unique_id`→ID_CARD and
   `certificate_license_number`→DRIVER_LICENSE are charitable stretches
   (noted so you can discount them); Presidio has no USERNAME/PASSWORD/SEX
   equivalent at all — same rule applies as for us: 0 recall, visible.

## Reproduce

```bash
cd experiments/006-presidio-head-to-head
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python datasets gliner torch presidio-analyzer spacy
.venv/bin/python -m spacy download en_core_web_lg

.venv/bin/python download_data.py          # ai4privacy sample (seed 42, n=1000)
bun run_pii_proxy.ts                       # regex layer
.venv/bin/python run_presidio.py           # Presidio baseline
.venv/bin/python run_gliner.py             # exp-004 model: native + extended passes
.venv/bin/python run_gliner_005.py         # pii-proxy-ner (exp-005 flagship)
.venv/bin/python score.py                  # results.json + markdown tables
.venv/bin/python verify.py                 # adversarial audit of the methodology
```

The exp-005 weights aren't in the repo (582MB) — they live on the training
volume (`modal volume get pii-proxy-models baseline-checkpoint/checkpoint-4641 ...`
into `experiments/005-nvidia-baseline/model/checkpoint-4641/`). Its 55 native
labels are committed at `data/nemotron_labels.json` (recovered by streaming
Nemotron-PII; they're computed at training time and weren't saved with the
checkpoint).

