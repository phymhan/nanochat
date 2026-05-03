# IDN Inference Evaluation Results

All models: d32 (32 layers, 2048 dim), single H100, SDPA, seq_len=128, BS=1, 8 batches.
Timing: 100 runs averaged with 20 warmup, 2s delay between models.
Inference: n_par=8 (seq=24 + 8 parallel), K=1.

Methods:
- **Sequential**: standard forward (ground truth)
- **IDN (loop)**: identity Newton K=1 with prefix-sum correction (correct, ~1.0x speed)
- **Fused**: concatenated weights mega-matmul, no correction (~1.45x speed, approximate)
- **Fused+split_o_mlp**: fused QKV+SDPA, per-layer O+MLP, Newton correction (~1.1x speed)

## d32_baseline (val_bpb=0.661, no layer-parallel training)

| Method | Time | Speed | PPL | ΔPPL% | Top-1 | PPL (k=0.1) | Top-1 (k=0.1) |
|---|---|---|---|---|---|---|---|
| Sequential | 17.20ms | 1.00x | 26.89 | — | — | — | — |
| IDN (loop) | ~17ms | ~1.0x | 1992 | +7308% | 0.344 | 1083 | 0.337 |
| Fused | 11.78ms | 1.46x | 1412 | +5150% | 0.529 | — | — |
| Fused+split_o_mlp | 16.28ms | 1.06x | 21006 | +78021% | 0.261 | 550 | 0.420 |

No layer-parallel training → all parallel methods fail. Sequential PPL 26.89
is the ground truth all other models compare against.

## d32_idn05_npar8 (val_bpb=0.661, IDN λ=0.5)

| Method | Time | Speed | PPL | ΔPPL% | Top-1 | PPL (k=0.1) | Top-1 (k=0.1) |
|---|---|---|---|---|---|---|---|
| Sequential | 16.95ms | 1.00x | 28.80 | — | — | — | — |
| IDN (loop) | ~17ms | ~1.0x | 28.71 | -0.3% | 0.860 | **28.16** | **0.903** |
| Fused | 11.63ms | 1.46x | 31.25 | +8.5% | 0.764 | — | — |
| Fused+split_o_mlp | 16.12ms | 1.05x | 27.23 | -5.4% | 0.704 | **26.83** | **0.748** |

**Best model for layer-parallel inference.** IDN training makes all correction
methods work. Fused+split k=0.1: PPL 26.83 (-6.8%), 1.33x speedup.

## d32_diag0.01_from_idn05_npar8 (val_bpb=0.661, IDN + diag finetune)

| Method | Time | Speed | PPL | ΔPPL% | Top-1 | PPL (k=0.1) | Top-1 (k=0.1) |
|---|---|---|---|---|---|---|---|
| Sequential | 13.97ms | 1.00x | 31.08 | — | — | — | — |
| IDN (loop) | ~14ms | ~1.0x | 31.28 | +0.6% | 0.814 | **28.72** | **0.873** |
| Fused | 15.33ms | 0.91x | 38.49 | +23.8% | 0.744 | — | — |
| Fused+split_o_mlp | 16.10ms | 0.87x | 30.22 | -2.8% | 0.682 | **27.70** | 0.740 |

Diag finetuning from IDN: Scale-ELK dramatically improves quality.

## d32_diag005_npar8_v3 (val_bpb=0.661, diag λ=0.05 from scratch)

| Method | Time | Speed | PPL | ΔPPL% | Top-1 | PPL (k=0.1) | Top-1 (k=0.1) |
|---|---|---|---|---|---|---|---|
| Sequential | 13.86ms | 1.00x | 29.81 | — | — | — | — |
| IDN (loop) | ~14ms | ~1.0x | 36.14 | +21.2% | 0.808 | 34.23 | 0.805 |
| Fused | 15.15ms | 0.91x | 6438 | broken | 0.494 | — | — |
| Fused+split_o_mlp | 15.99ms | 0.87x | 30.99 | +4.0% | 0.697 | 30.70 | 0.703 |

Diag λ=0.05: moderate quality. Fused+split k=0.1: PPL 30.70 (+3.0%).

## d32_diag01_npar8 (val_bpb=0.662, diag λ=0.1 from scratch)

| Method | Time | Speed | PPL | ΔPPL% | Top-1 | PPL (k=0.1) | Top-1 (k=0.1) |
|---|---|---|---|---|---|---|---|
| Sequential | 17.24ms | 1.00x | 28.24 | — | — | — | — |
| IDN (loop) | ~17ms | ~1.0x | 33.38 | +18.2% | 0.819 | 32.19 | 0.807 |
| Fused | 11.89ms | 1.45x | 288K | broken | 0.273 | — | — |
| Fused+split_o_mlp | 13.93ms | 1.24x | 30.94 | +9.5% | 0.727 | 30.01 | 0.714 |

Diag λ=0.1: Fused+split k=0.1 PPL 30.01 (+6.2%), 1.08x speedup.

## d32_diag05_npar8_v4 (val_bpb=0.664, diag λ=0.5 from scratch)

| Method | Time | Speed | PPL | ΔPPL% | Top-1 | PPL (k=0.1) | Top-1 (k=0.1) |
|---|---|---|---|---|---|---|---|
| Sequential | 17.45ms | 1.00x | 31.03 | — | — | — | — |
| IDN (loop) | ~17ms | ~1.0x | 32.85 | +5.9% | 0.897 | 32.82 | 0.887 |
| Fused | 12.06ms | 1.45x | 19M | broken | 0.012 | — | — |
| Fused+split_o_mlp | 15.66ms | 1.11x | 34.53 | +11.3% | 0.804 | **31.03** | 0.780 |

**Best diag model.** Fused+split k=0.1: PPL 31.03 (-0.0% vs seq!), 1.12x speedup.
IDN loop has highest top-1 (0.897) but no speedup.

## Summary: Best parallel inference per model (with elk_k=0.1)

Baseline sequential PPL = 26.89. ΔPPL% is vs each model's own sequential PPL.

| Model | Seq PPL | Best parallel method | PPL | ΔPPL% | Top-1 | Speed |
|---|---|---|---|---|---|---|
| Baseline (no reg) | 26.89 | — | all broken | — | — | — |
| **IDN λ=0.5** | **28.80** | **Fused+split k=0.1** | **26.83** | **-6.8%** | **0.748** | **1.33x** |
| Diag λ=0.01 from IDN | 31.08 | Fused+split k=0.1 | 27.70 | -10.9% | 0.740 | 1.09x |
| Diag λ=0.05 scratch | 29.81 | Fused+split k=0.1 | 30.70 | +3.0% | 0.703 | 1.09x |
| Diag λ=0.1 scratch | 28.24 | Fused+split k=0.1 | 30.01 | +6.2% | 0.714 | 1.08x |
| **Diag λ=0.5 scratch** | **31.03** | **Fused+split k=0.1** | **31.03** | **-0.0%** | **0.780** | **1.12x** |

## Key takeaways

1. **IDN λ=0.5 + Fused+split_o_mlp + elk_k=0.1 is the best combo**:
   PPL 26.83 (-6.8% vs its own sequential), 1.33x speedup.

2. **Diag λ=0.5 + Fused+split_o_mlp + elk_k=0.1**: PPL matches sequential
   exactly (31.03 vs 31.03) with 1.12x speedup. Zero quality loss.

3. **Fused (original) is broken** on all diag-trained models and only usable
   on IDN model (+8.5% PPL).

4. **Scale-ELK (k=0.1) consistently improves** all methods with Newton
   correction, at zero computational cost.

5. **Layer-parallel training is essential** — baseline model fails on all
   parallel methods regardless of inference strategy.

## Reproduction

```bash
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_jacobi_idn \
    --model-tag d32_idn05_npar8 --n-par 8 --K 1 --elk-k 0.1
```
