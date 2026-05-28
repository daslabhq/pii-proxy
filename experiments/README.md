# PII Detection Benchmark

Hard numbers from running real models on NVIDIA's [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) healthcare dataset (725 records, 4,703 labeled spans).

## TL;DR

Fine-tuned BERT (50MB, ~9ms) **beats Claude Sonnet, qwen3.7-max, and GLiNER-PII** on clinical PII detection — at 1/278th the latency and zero ongoing cost.

| Method | F1 | Speed/record | Size | Cost (1M records) |
|---|---|---|---|---|
| **Fine-tuned BERT (ours)** | **94.2%** | **~9ms** | **50MB** | **$0** |
| Claude Sonnet (fixed prompt) | 92.5% | 2.5s | cloud | ~$3,500 |
| GLiNER-PII (NVIDIA, zero-shot) | 91.5% | 280ms | 570MB | $0 |
| BioBERT (biomedical fine-tune) | 91.9% | ~35ms | 108MB | $0 |
| ClinicalBERT (clinical fine-tune) | 90.9% | ~27ms | 135MB | $0 |
| qwen3.7-max (OpenRouter) | 83.3% | 23.6s | cloud | ~free |
| Claude Sonnet (broken prompt) | 82.3% | 2.6s | cloud | ~$3,500 |
| qwen3:1.7b (local LLM, our tagger v1) | 6.6% | 15s | 1.4GB | $0 |

All numbers on the same 20-record / 145-record splits. Reproducible via the scripts in this directory.

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
