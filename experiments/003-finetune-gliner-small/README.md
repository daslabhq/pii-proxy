# Experiment 003 — Fine-tune `gliner_small-v2.1` on Nemotron-PII

## Honest summary

Fine-tuned `urchade/gliner_small-v2.1` (582MB) on Nemotron-PII healthcare records reaches **95.5% F1** on a held-out test set.

NVIDIA's `gliner-PII` (1699MB, ~3x larger) scores **97.2% F1** on the same test set when queried with its native training labels. We're **1.7pp behind** their flagship, but with a smaller model and faster inference.

This is a real, reproducible result. Earlier framing ("beats NVIDIA") was based on a methodology error caught during verification — see [Methodology lessons](#methodology-lessons) below.

## Setup

- Base model: `urchade/gliner_small-v2.1` (582MB on disk, "small" variant of GLiNER large family)
- Dataset: Nemotron-PII healthcare subset (725 records, 0.7% of full Nemotron-PII)
- Split: 625 train (575 + 50 in-loop eval) / 100 held-out test (same as experiment 002, fingerprint `ff2fa10db2eb0b55`)
- Epochs: 5
- Batch size: 4 (MPS memory constraint)
- LR: 5e-6 encoder, 5e-6 head
- Focal loss: disabled (unstable on MPS with default params)
- Hardware: Apple MPS, 36GB RAM

## Results (verified independently via `verify.py`)

**Held-out test set (100 records, 589 gold entities):**

| Configuration | F1 | Precision | Recall | Latency | Weights |
|---|---|---|---|---|---|
| `gliner_small-v2.1` zero-shot | 54.8% | 59.0% | 51.1% | 115ms | 582MB |
| **`gliner_small-v2.1` fine-tuned (ours)** | **95.5%** | **95.4%** | **95.6%** | **126ms** | 582MB |
| Improvement from fine-tuning | **+40.7pp** | | | | |

Training: 891s (~15 min) on Apple MPS.

## Fair comparison with NVIDIA

| Method | F1 | Latency | Weights | Zero-shot capable |
|---|---|---|---|---|
| **nvidia/gliner-PII (native labels)** | **97.2%** | 210ms | 1699MB | Yes |
| Our fine-tuned gliner_small | 95.5% | 126ms | 582MB | Yes |
| nvidia/gliner-PII (natural-language labels) | 90.4% | 211ms | 1699MB | Yes |
| BERT classifier fine-tuned (exp 002) | 93.9% | 26ms | 438MB | No |
| `gliner_small-v2.1` zero-shot | 54.8% | 115ms | 582MB | Yes |

**Delta vs NVIDIA flagship:**
- F1: **-1.7pp** (we lose on accuracy)
- Speed: **1.7x faster** (we win on latency)
- Size: **3x smaller** (we win on disk/memory)

The honest tradeoff: smaller model with nearly-flagship accuracy, faster inference, comparable zero-shot flexibility.

## Methodology lessons

**Bi-encoder NER models are sensitive to query string choice.** GLiNER lets you pass entity types as natural language at inference time, but the model scores differently depending on whether you use the exact strings it was trained on vs paraphrases.

Tested on the same model, same test set:

```
nvidia/gliner-PII queried with "person name", "medical record number":  90.4% F1
nvidia/gliner-PII queried with "first_name", "medical_record_number":   97.2% F1
                                                                        +6.8pp
```

The 6.8pp gap comes purely from query string choice. Our original benchmark used natural-language labels (the obvious-looking choice) and unfairly handicapped NVIDIA's model by ~7pp.

**Lesson:** any benchmark comparing GLiNER-family models must test both natural-language AND model-native query strings. Report the higher of the two for each model, or pick a fixed convention and disclose it.

See [`verify_labels.py`](verify_labels.py) for the test.

## Key findings (honest)

1. **Fine-tuning a small GLiNER on a domain subset works.** +40.7pp F1 over zero-shot, reaching 95.5%. The learn → compile pipeline is valid.

2. **NVIDIA's flagship still wins on accuracy** by 1.7pp when queried fairly with native labels. Their large model + full 100k training data still has a real edge.

3. **The small model is competitive** at 1.7x faster inference and 3x smaller disk footprint. For latency-sensitive or memory-constrained deployment, our model is the better choice.

4. **The compile pipeline produces a usable model in 15 minutes on a laptop.** Reproducibility is the differentiator — NVIDIA shipped weights but no recipe.

5. **We've only trained on 0.7% of the available data.** With full 100k Nemotron records on a real GPU, we'd likely match or beat NVIDIA. ~6 hours on a single A100, ~$15 cloud cost.

## Caveats

- **Synthetic data, not real clinical notes.** Same caveat as experiment 002 — real EHR notes likely have lower scores due to abbreviations, jargon, edge cases.
- **MPS memory constraints** forced batch size = 4 and disabled focal loss. CUDA training with the original recipe would likely score higher.
- **5 epochs only.** Loss plateaued by epoch 3-4. More epochs may not help with this dataset size; more data definitely would.
- **Healthcare subset only.** Our 95.5% is on healthcare PII. Untested on finance, legal, HR PII — likely much worse out-of-domain.

## Reproducing

```bash
# Free disk space if low (Xcode DerivedData can be huge)
rm -rf ~/Library/Developer/Xcode/DerivedData/*

# Ensure deps
pip3 install -U accelerate gliner

# Train (~15 min on Apple MPS)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 experiments/003-finetune-gliner-small/run.py

# Verify all claims independently
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 experiments/003-finetune-gliner-small/verify.py

# Test label-sensitivity of NVIDIA's model
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 experiments/003-finetune-gliner-small/verify_labels.py
```

## What we'd do next

1. **Train on full 100k Nemotron** — ~6 hours on rented A100 (~$15). Tests if scale alone closes the 1.7pp gap.
2. **Train on real clinical notes** (i2b2 2014 if DUA approved) — tests if synthetic-data performance transfers.
3. **Test on finance/legal/HR subsets** — confirms or refutes the "specialist beats generalist" hypothesis on other domains.
4. **Try `gliner_medium-v2.1`** — middle ground between our small fine-tune and NVIDIA's large.

## Files

- `run.py` — training + eval script (`model.train_model()` via official API)
- `verify.py` — independent verification of all claims
- `verify_labels.py` — label-sensitivity test (NVIDIA scores 6.8pp higher with native labels)
- `results.json` — training-time metrics
- `verification.json` — verification metrics
- `label_sensitivity.json` — NVIDIA-with-different-labels metrics
- `output.log` — full training log (gitignored)
- `model/` — trained model checkpoints (gitignored)
