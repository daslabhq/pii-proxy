# Experiment 005 — NVIDIA baseline replication (full 100k Nemotron-PII)

## Result

Fine-tuning `gliner_small-v2.1` on the **full Nemotron-PII dataset** (100k records,
all 55 entity types) matches NVIDIA's flagship `gliner-PII` on a fair, held-out
comparison — using a base model with ~3x fewer parameters.

| Model | Base | F1 | Precision | Recall |
|---|---|---|---|---|
| **Our fine-tuned `gliner_small-v2.1`** | small (~50M) | **90.1%** | 90.5% | 89.7% |
| `nvidia/gliner-PII` | large (~300M) | 89.3% | 90.0% | 88.6% |
| Delta | | +0.8pp | | |

**Read this as a statistical tie, not "we beat NVIDIA."** +0.8pp on a 500-record /
4,190-entity test set is within noise (one misclassified record ≈ 0.1–0.2pp). The
honest claim: **our pipeline reproduces NVIDIA flagship quality with a 3x smaller
base model.** That's the result that matters — it validates the compile-to-small-model
thesis.

## Setup (verified from the run)

- Base model: `urchade/gliner_small-v2.1` (NVIDIA used `gliner_large-v2.1`)
- Data: full Nemotron-PII — **99,994 records loaded, 55 entity types, 30 domains**
- Split: 98,994 train / 500 in-loop eval (from train) / **500 held-out test**
- Training: 3 epochs, batch 16/device × 4 A100 (effective batch 64), bf16,
  cosine LR, focal loss (NVIDIA recipe)
- Hardware: 4× A100 on Modal
- Wall-clock: **39 min** (~$8)
- Both models evaluated on the SAME 500 held-out records with the SAME 55 native
  labels at threshold 0.5

## Methodology audit (why this result is trustworthy)

This experiment was explicitly audited after a prior session produced overstated
claims. The checks:

1. **No train/test leakage.** `test = records[:500]` is sliced off before
   `train = records[500:]`; the in-loop `eval_split` is carved from `train`, never
   from `test`. The 500 test records are never seen in training. Verified by reading
   the data-split code (lines 77–112 of `modal_baseline.py`).
2. **Same test set for both models.** `evaluate(model, test_records, all_labels)` is
   called identically for ours and NVIDIA's — same 500 records, same eval function.
3. **Fair label vocabulary.** Both models queried with `all_labels` (the 55 native
   Nemotron labels). This avoids the label-sensitivity trap from exp 003 where
   querying NVIDIA with coarse/natural-language labels under-rated it by 7–10pp.
   NVIDIA's 89.3% here is its real native-label performance.
4. **Counts reconcile.** 98,994 train + 500 in-loop eval + 500 test = 99,994 loaded.

## Honest caveats

- **Synthetic data.** Nemotron-PII is synthetic. Real clinical/financial notes will
  score lower (abbreviations, jargon, edge cases). The general "matches NVIDIA" claim
  is on synthetic data only.
- **500-record test set is small.** ±0.2pp noise. The +0.8pp delta is not
  statistically meaningful — treat as a tie. A larger test set would tighten this.
- **Config drift during the session.** Earlier edits attempting `gliner_large` /
  batch 32 / 1 epoch did not take (overwritten during duplicate launches); the run
  used the file defaults: `gliner_small`, batch 16, 3 epochs. The result is clean and
  reproducible from the committed script — it's just `small`, not `large`, which is
  the *better* story.
- **Lineage caveat (important):** NVIDIA's `gliner-PII` was itself trained on
  Nemotron-PII. So both models trained on the same distribution as the test set. This
  measures "can we reproduce NVIDIA's training" — NOT generalization to out-of-domain
  data. The real-world test (i2b2 clinical, or a customer's domain) is still owed.

## What this validates

The compile-to-small thesis: **a 3x smaller base + the same data + a reproducible
recipe = flagship-class results.** NVIDIA shipped weights with no recipe; we have
both. For on-device / edge PII detection, the small model that matches the large one
is the whole game.

## Reproduce

```bash
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
modal run experiments/005-nvidia-baseline/modal_baseline.py
# defaults: gliner_small, 100k records, 3 epochs, 4x A100, ~39 min, ~$8
```

## Files

- `modal_baseline.py` — the run (Modal, 4× A100)
- result JSON written to the `pii-proxy-models` Modal volume as `baseline_result.json`
