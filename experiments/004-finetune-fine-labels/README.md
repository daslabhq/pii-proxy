# Experiment 004 — Fine-tune `gliner_small-v2.1` on FINE Nemotron labels

## Why

Experiment 003 trained our model on COARSE labels (13 types: `Person Name`, `Location`, etc.) by collapsing Nemotron's 55+ fine labels during preprocessing. NVIDIA's `gliner-PII` trained on the raw FINE labels (`first_name`, `last_name`, `street_address`, `city`, ...). That meant exp 003 was apples-to-oranges with NVIDIA.

This experiment retrains on the SAME fine labels NVIDIA used. Now both models can be compared head-to-head at fine granularity, AND we can derive coarse types post-hoc via label mapping (the same simple wrapper API for pii-proxy).

## Setup

- Base model: `urchade/gliner_small-v2.1`
- Dataset: Nemotron-PII healthcare subset (725 records)
- Labels: **26 fine labels** (raw Nemotron, healthcare subset)
- Split: 625 train (575 + 50 in-loop eval) / 100 held-out test (same seed as exp 002-003)
- Epochs: 5 planned, **3.85 actually completed** (MPS crashed near end with leaked semaphore)
- Batch size: 2 (reduced for memory)
- LR: 5e-6 encoder, 5e-6 head
- Focal loss: disabled

Used `checkpoint-1000` (epoch ~3.5, after loss converged) for evaluation since the final checkpoint never saved.

## Results — verified head-to-head with `eval_only.py`

**Fine granularity** (26 native Nemotron labels, both models trained on this vocabulary):

| Method | F1 | P | R | Latency | TP/FP/FN |
|---|---|---|---|---|---|
| **nvidia/gliner-PII** | **96.2%** | 95.3% | 97.1% | 211ms | 572/28/17 |
| Ours (ckpt-1000, ~3.5 epochs) | 94.9% | 94.0% | 95.9% | 144ms | 565/36/24 |
| Delta | **-1.3pp** | | | **1.5x faster** | |

**Coarse granularity** (13 types, predictions collapsed post-hoc):

| Method | F1 |
|---|---|
| nvidia/gliner-PII (fine model, collapsed) | 96.7% |
| Ours exp 004 (fine model, collapsed) | 96.1% |
| Ours exp 003 (trained directly on coarse) | 95.5% |

## Key findings

1. **Truly fair comparison: we trail NVIDIA by 1.3pp at fine granularity.** Same labels, same test set, same query strategy. NVIDIA still wins on accuracy but we're close.

2. **We win on speed (1.5x faster) and size** (582MB base vs 1699MB NVIDIA base).

3. **Training on fine labels + collapsing beats training directly on coarse.** 96.1% (exp 004) > 95.5% (exp 003) on the same coarse evaluation. The model learns richer representations when given more specific labels during training. **Recommended pattern: always train fine, collapse for downstream use.**

4. **We only completed ~3.5 epochs of planned 5.** MPS crashes near training end (leaked semaphore — observed across all our GLiNER runs). A full 5-epoch run on a stable GPU (or with CPU fallback) would likely close more of the gap.

5. **Methodology pitfall confirmed across configurations.** GLiNER label sensitivity (documented in exp 003) holds at fine granularity too.

## Caveats

- **Crashed training.** Used checkpoint-1000 (epoch ~3.5) not the final model. With full 5-epoch convergence on stable hardware we might match or slightly beat NVIDIA.
- **Synthetic data.** Same caveat as prior experiments.
- **Healthcare subset only.** 625 train records vs NVIDIA's 100k across 50+ industries.
- **MPS memory issues at scale.** Cannot reliably train past ~4 epochs on 36GB MPS with batch_size=2. Need CUDA GPU or CPU fallback for production-grade runs.

## What this tells the pii-proxy use case

For coarse PII detection (the pii-proxy API need):
- Best accuracy: nvidia/gliner-PII (96.7% F1, but 1.7GB, 211ms)
- Best balance: our exp 004 model (96.1% F1, 582MB base, 144ms)
- Best speed: BERT classifier from exp 002 (93.9% F1, 26ms, no zero-shot)

All three are valid. Choose by your latency/size/accuracy constraints.

## Files

- `run.py` — training script (crashed at step 1118/1440)
- `eval_only.py` — eval-only script that uses the saved checkpoint-1000
- `output.log` — training log (gitignored)
- `eval_only.log` — eval log (gitignored)
- `eval_only_results.json` — final verified metrics
- `model/` — saved checkpoints (gitignored)
