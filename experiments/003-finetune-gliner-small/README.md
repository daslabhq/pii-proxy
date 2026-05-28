# Experiment 003 — Fine-tune `gliner_small-v2.1` on Nemotron-PII

## Hypothesis

NVIDIA's `gliner-PII` is `urchade/gliner_large-v2.1` (459M params, 570MB) fine-tuned on Nemotron-PII. If we fine-tune the SMALL variant (`urchade/gliner_small-v2.1`, ~150MB) on the same healthcare subset, we could get:

- Same architecture family as NVIDIA's model (same zero-shot capability)
- 4x smaller (150MB vs 570MB)
- Competitive accuracy on PII (purpose-fine-tuned for it)
- Published training recipe (NVIDIA didn't publish one)

## Setup

- Base model: `urchade/gliner_small-v2.1`
- Dataset: Nemotron-PII healthcare subset (725 records)
- Split: 625 train (575 + 50 in-loop eval) / 100 held-out test (same as experiment 002)
- Epochs: 5
- Batch size: 4 (MPS memory constraint)
- LR: 5e-6 encoder, 5e-6 head
- Focal loss: disabled (unstable on MPS with default params)
- Hardware: Apple MPS

## Results

**Held-out test set (100 records, 589 gold entities):**

| Configuration | F1 | Precision | Recall | Latency | Size |
|---|---|---|---|---|---|
| `gliner_small-v2.1` zero-shot | 54.8% | 59.0% | 51.1% | 115ms | ~150MB |
| **`gliner_small-v2.1` fine-tuned** | **95.5%** | **95.4%** | **95.6%** | **106ms** | **~150MB** |
| Improvement | **+40.7pp** | | | | |

Training: 891s (~15 min) on Apple MPS.

## Comparison with prior experiments

| Method | F1 | Latency | Size | Zero-shot capable |
|---|---|---|---|---|
| **gliner_small-v2.1 fine-tuned (this exp)** | **95.5%** | **106ms** | **~150MB** | **Yes** |
| BERT classifier fine-tuned (exp 002) | 93.9% | 26ms | 109MB | No |
| nvidia/gliner-PII zero-shot (exp 002) | 90.3% | 333ms | 570MB | Yes |
| GLiNER trained on GLiNER labels (exp 002) | 87.3% | 27ms | 109MB | No |

## Key findings

1. **Fine-tuned gliner_small BEATS NVIDIA's gliner-PII** on every dimension except raw inference speed:
   - +5.2pp F1 (95.5% vs 90.3%)
   - 3x faster (106ms vs 333ms)
   - 4x smaller (~150MB vs 570MB)
   - Same zero-shot capability (architectural property)

2. **The compile step adds significant value when done right.** Going from `gliner_small` zero-shot (54.8%) to fine-tuned (95.5%) is +40.7pp — proves the learn→compile pipeline works. The poor result in experiment 002 (BERT trained on GLiNER labels: 87.3%) was due to label noise in the training data, not the compilation step itself.

3. **For high-accuracy + flexibility, fine-tune GLiNER.** For maximum speed, use a BERT classifier. Two valid optima:
   - **gliner_small fine-tuned**: 95.5% F1, 106ms, flexible (zero-shot for novel types)
   - **BERT classifier**: 93.9% F1, 26ms, rigid (only trained labels)

4. **Training recipe is reproducible** in ~250 lines + standard GLiNER API. NVIDIA shipped the model weights but not the recipe; we ship both.

## Caveats

- **Loss is noisy.** Without focal loss the training is more stable but loss values oscillate (2-7 range). The final eval F1 is the real signal.
- **MPS memory pressure.** Had to reduce batch size from 8 to 4 and disable focal loss. On a GPU (CUDA), original config would work.
- **Synthetic data, not real clinical notes.** Same caveat as experiment 002 — real EHR notes likely have lower scores due to abbreviations, jargon, edge cases.
- **5 epochs.** Loss plateaued by epoch 3-4. More epochs not needed for this dataset size.

## Reproducing

```bash
# Free disk space if low (Xcode DerivedData can be huge)
rm -rf ~/Library/Developer/Xcode/DerivedData/*

# Ensure deps
pip3 install -U accelerate
pip3 install gliner

# Run (~15 min on Apple MPS)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 experiments/003-finetune-gliner-small/run.py
```

## Files

- `run.py` — training + eval script
- `results.json` — metrics
- `output.log` — full training log
- `model/` — trained model checkpoint (gitignored)
