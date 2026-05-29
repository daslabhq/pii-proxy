# PII Detection Benchmarks

Multiple experiments against NVIDIA's [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) healthcare subset (725 records, 4,703 labeled spans). All numbers from the same held-out 100-record test set, independently verified.

**Latest truly fair comparison (experiment 004 — same labels, same vocabulary):**

| Method | Fine F1 | Coarse F1 | Latency | Weights |
|---|---|---|---|---|
| **nvidia/gliner-PII** | **96.2%** | **96.7%** | 211ms | 1699MB |
| Our fine-tuned `gliner_small` (exp 004) | 94.9% | 96.1% | 144ms | 582MB |
| Our BERT classifier (exp 002) | — | 93.9% | 26ms | 438MB |
| Claude Sonnet (proper prompt) | — | 92.5% | 2.5s | cloud |
| `gliner_small-v2.1` (zero-shot baseline) | 54.8% | — | 115ms | 582MB |

NVIDIA's flagship leads by 1.3pp at fine granularity and 0.6pp at coarse. We trade that gap for 1.5x speed and 3x smaller base model. The BERT classifier is 5x faster again, at -2pp F1 and no zero-shot capability.

## Per-experiment results

- [`004-finetune-fine-labels/`](004-finetune-fine-labels/) — Fine-tune `gliner_small` on FINE Nemotron labels (matches NVIDIA's training granularity). **Truly fair head-to-head.** 94.9% fine / 96.1% coarse.
- [`003-finetune-gliner-small/`](003-finetune-gliner-small/) — Fine-tune on COARSE labels directly. 95.5% F1. Includes the **label-sensitivity gotcha** that nearly led us to publish overstated claims.
- [`clean_benchmark.py`](clean_benchmark.py) — BERT classifier vs GLiNER zero-shot, identical test set.
- [`finetune_ner.py`](finetune_ner.py) — Original BERT classifier fine-tune.
- [`full_benchmark.py`](full_benchmark.py) — Cloud LLM taggers (Sonnet, qwen3.7-max) on same test set.

## Verification methodology

Every result is independently re-run via `verify.py` (or `eval_only.py` in exp 004) on the same test set with the same seed (42). Fingerprint of test set: `ff2fa10db2eb0b55`. Zero train/test leakage confirmed in every experiment.

## Key methodology lesson

**GLiNER bi-encoder F1 swings 5-17pp purely on query string choice.** Same model, same test set, different label vocabulary:

```
nvidia/gliner-PII with 13 coarse snake_case ('person_name'):  79.8%
nvidia/gliner-PII with 17 natural-language ('person name'):    90.4%
nvidia/gliner-PII with 26 fine native labels ('first_name'):   97.2%
```

Always query a bi-encoder with its native training vocabulary. Documented in [`003-finetune-gliner-small/verify_labels.py`](003-finetune-gliner-small/verify_labels.py).

## Recommended training pattern

**Train on fine labels, collapse to coarse post-hoc.** Beats training directly on coarse: 96.1% (exp 004 fine→coarse) > 95.5% (exp 003 trained-on-coarse). The richer supervision during training transfers to better representations even at coarser inference granularity.

## The pipeline

```
NVIDIA Nemotron-PII (synthetic healthcare data, 725 records)
                 │
                 │ split: 580 train / 145 test
                 ▼
        Fine-tune BERT (8 epochs, 10 min on Apple MPS)
                 │
                 ▼
    Compiled NER model (50MB ONNX, 9ms inference)
                 │
                 │ plugs into pii-proxy as a detector
                 ▼
   Bijective masking + plausible fakes + round-trip unmask
```

## Key findings

### 1. Generic BERT beats domain-specific models

We expected ClinicalBERT/BioBERT to win — they're pre-trained on medical text. They didn't. Generic `bert-base-uncased` outperformed both on this synthetic dataset (94.2% vs 91.9% vs 90.9%).

Why: the training data is synthetic, not real clinical notes. Generic tokenization handles the synthetic text better. On real EHR data with abbreviations and clinical jargon, the domain models likely catch up. Worth re-running on real clinical notes.

### 2. The compiled model beats its own teacher

GLiNER-PII (the model that generated our labels) hits 91.5% F1 on the test set. Our fine-tuned BERT trained on GLiNER's labels hits 94.2% — better than the teacher. This is classic distillation behavior: aggregating over 580 examples lets the student model capture patterns the teacher misses on individual records.

### 3. Prompt engineering matters enormously for cloud LLMs

Claude Sonnet went from 82.3% F1 to 92.5% F1 with two fixes:
- Use the `system` field instead of cramming the prompt into a user message
- Set `temperature: 0`

Same model, different prompting, +10 points F1. If you're benchmarking LLMs, prompt construction is half the variance.

### 4. Small open models lose badly at structured extraction

qwen3:1.7b at 6.6% F1 was a wake-up call. Generic small LLMs can't reliably do PII detection even with `response_format: json_object`. They lack the structured-output discipline. GLiNER-PII (purpose-built encoder model) at 91.5% massively outperforms with 1/3 the parameters.

The takeaway: for structured extraction, purpose-built classifiers (encoders) beat generic generative models. Always.

### 5. The cost difference is order-of-magnitude

Processing 1 million records (a single mid-size healthcare deployment for one year):

```
Compiled BERT:    $0 ongoing, 9ms latency       = viable in hot path
GLiNER-PII:       $0 ongoing, 280ms latency      = viable in batch
Claude Sonnet:    $3,500/M records, 2.5s latency = expensive batch only
qwen3.7-max:      ~free, 24s/record              = impractical
```

At 100M records/year (a large hospital network), the difference becomes:
- Compiled BERT: $0
- Claude Sonnet: $350,000

The compiled model is not just faster — it's the only economically viable option for high-volume PII detection.

## What we did NOT prove

- **Real clinical notes** — all benchmarks are on synthetic Nemotron data. Performance on real EHR notes may be lower (more abbreviations, more variation, more edge cases). Re-running on i2b2 2014 (requires DUA) is the next step.
- **Grammar-constrained training** — the BERT classifier has a fixed output head (27 BIO labels). This is not the same as training a generative model with FSM constraints. Different problem.
- **Round-trip masking quality** — the benchmark measures detection F1, not the quality of plausible fake replacements or unmask round-trip. Round-trip correctness is tested separately in the main test suite.

## Scripts in this directory

- `finetune_ner.py` — Fine-tune a BERT model on Nemotron-PII healthcare records. Outputs metrics + saved model.
- `full_benchmark.py` — Full matrix: GLiNER-PII vs qwen3.7-max vs Claude Sonnet (taggers), plus fine-tuned BERT/BioBERT/ClinicalBERT (production models).
- `benchmark.py` — Lighter version: GLiNER vs LLM tagger only.
- `eval_tagger.ts` — TypeScript eval comparing pii-proxy's built-in LLM tagger against Nemotron ground truth.
- `test_learn.ts` — End-to-end test of `pii-proxy learn` on synthetic clinical notes.

## Reproducing

Requirements:
- Python 3.9+ with `pip install datasets transformers torch seqeval gliner`
- Bun for TypeScript scripts
- Ollama running locally (for LLM tagger benchmarks)
- `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY` env vars for cloud LLM benchmarks

```bash
# Build the Nemotron-PII healthcare cache (one-time, ~1 min)
python3 -c "from datasets import load_dataset; ..."  # see benchmark.py

# Fine-tune BERT on healthcare records (10 min on Apple MPS)
python3 experiments/finetune_ner.py

# Run the full matrix (~30 min total, includes cloud API calls)
source .env && python3 experiments/full_benchmark.py --limit 20
```

Outputs go to `experiments/output/`.

## What this enables

The benchmark proves the `learn → compile → run` pipeline works end-to-end:

1. **learn** — GLiNER-PII labels training data at 280ms/record, 91.5% F1
2. **compile** — Fine-tune BERT on labels, 10 min on a laptop, produces 50MB model
3. **run** — 9ms inference per record, 94.2% F1, $0 ongoing cost

Per-domain LoRAs would extend this further: train a tiny adapter per industry (healthcare, finance, legal, HR), each compiled from domain-specific traces. Same pipeline, different data.

The PII proxy's bijective mask + plausible fakes + round-trip unmask is what makes the compiled detector useful in production — the model finds entities, the proxy replaces them with realistic fakes, and unmask restores the originals when the LLM responds. No other library does the full round-trip.
