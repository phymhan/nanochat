# Layer-Parallel Inference: d32 Models (n_embd=2048, 32 layers, ~2.8B params)

depth=32, n_embd=2048, 16 heads, ~2.8B params. Single H100, SDPA, seq_len=128, BS=8.
PPL: **42M tokens** from validation set (same as training eval).
Timing: **250 runs + 50 warmup** on single batch. n_par=8 (seq=24 + 8 parallel).

These models were trained with n_par=8 only. All trained for 9600 steps.

## Summary

Baseline seq PPL = 37.94 (42M tokens).
Old eval (1024 tokens) reported baseline PPL = 26.89 — unstable due to tiny eval set.

| Model | val_bpb | Seq PPL | Warm IDN K=1 PPL | dPPL% | Speed |
|---|---|---|---|---|---|
| Baseline (no reg) | 0.661 | 37.94 | 37.95 | +0.01% | 0.85x |
| IDN λ=0.5, npar=8 | 0.661 | 35.98 | 35.98 | -0.00% | 0.85x |
| Diag λ=0.01 from IDN | 0.661 | 39.03 | 39.03 | -0.00% | 0.85x |
| Diag λ=0.05 scratch | 0.661 | 43.65 | 43.65 | -0.00% | 0.80x |
| Diag λ=0.1 scratch | 0.662 | 44.15 | 44.17 | +0.04% | 0.79x |
| Diag λ=0.5 scratch | 0.664 | 47.90 | 47.92 | +0.04% | 0.81x |

**Key observations:**

1. **IDN λ=0.5 has best seq PPL** (35.98 vs baseline 37.94, -5.2%). IDN reg
   improves generalization, consistent with d32s findings.

2. **Diag reg degrades seq PPL** at these lambda values. Diag λ=0.05 (43.65)
   and λ=0.5 (47.90) are significantly worse than baseline.

3. **No speedup at D=2048**: all methods are 0.79-0.85x (slower than sequential).
   The GPU is saturated by individual (128, 2048)×(2048, 2048) matmuls.
   Batching 8 of them via einsum doesn't parallelize further.

4. **Warm-start gives +0.0% PPL** regardless of model, confirming the d32s finding.

## Baseline (no reg) (seq PPL=37.94, seq=16.9ms)

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 16.9ms | 1.00x | 37.94 | --- |
| IDN K=1 h0 | h0 | 20.4ms | 0.83x | 1566.60 | +4028.9% |
| Fused+corr K=1 h0 | h0 | 18.8ms | 0.90x | 1566.83 | +4029.5% |
| IDN K=2 h0 | h0 | 26.5ms | 0.64x | 68101.23 | +179387.0% |
| Fused+corr K=2 h0 | h0 | 22.1ms | 0.76x | 68100.88 | +179386.1% |
| IDN K=4 h0 | h0 | 39.1ms | 0.43x | 47.59 | +25.4% |
| Fused+corr K=4 h0 | h0 | 27.2ms | 0.62x | 47.59 | +25.4% |
| Fused h0 | h0 | 15.9ms | 1.06x | 3016.35 | +7849.9% |
| IDN K=1 fwd | fwd | 24.8ms | 0.68x | 254.37 | +570.4% |
| Fused+corr K=1 fwd | fwd | 23.3ms | 0.73x | 254.38 | +570.4% |
| IDN K=2 fwd | fwd | 30.8ms | 0.55x | 137.25 | +261.7% |
| Fused+corr K=2 fwd | fwd | 26.1ms | 0.65x | 137.25 | +261.7% |
| IDN K=4 fwd | fwd | 42.4ms | 0.40x | 40.97 | +8.0% |
| Fused+corr K=4 fwd | fwd | 30.9ms | 0.55x | 40.97 | +8.0% |
| IDN K=1 warm | warm | 19.8ms | 0.85x | 37.95 | +0.0% |
| Fused+corr K=1 warm | warm | 18.4ms | 0.92x | 37.94 | -0.0% |
| IDN K=2 warm | warm | 25.9ms | 0.65x | 37.95 | +0.0% |
| Fused+corr K=2 warm | warm | 21.1ms | 0.80x | 37.94 | -0.0% |
| IDN K=4 warm | warm | 38.0ms | 0.45x | 37.95 | +0.0% |
| Fused+corr K=4 warm | warm | 26.9ms | 0.63x | 37.94 | -0.0% |
| Diag jvp p=4 K=1 warm | warm | 85.4ms | 0.20x | 37.95 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 38.3ms | 0.44x | 37.95 | +0.0% |

## IDN λ=0.5, npar=8 (seq PPL=35.98, seq=16.9ms)

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 16.9ms | 1.00x | 35.98 | --- |
| IDN K=1 h0 | h0 | 20.0ms | 0.85x | 36.75 | +2.2% |
| Fused+corr K=1 h0 | h0 | 18.6ms | 0.91x | 36.75 | +2.1% |
| IDN K=2 h0 | h0 | 26.4ms | 0.64x | 38.20 | +6.2% |
| Fused+corr K=2 h0 | h0 | 21.8ms | 0.78x | 38.20 | +6.2% |
| IDN K=4 h0 | h0 | 38.7ms | 0.44x | 36.03 | +0.1% |
| Fused+corr K=4 h0 | h0 | 26.7ms | 0.63x | 36.03 | +0.1% |
| Fused h0 | h0 | 15.9ms | 1.07x | 38.77 | +7.8% |
| IDN K=1 fwd | fwd | 24.3ms | 0.70x | 37.27 | +3.6% |
| Fused+corr K=1 fwd | fwd | 22.8ms | 0.74x | 37.27 | +3.6% |
| IDN K=2 fwd | fwd | 30.4ms | 0.56x | 37.20 | +3.4% |
| Fused+corr K=2 fwd | fwd | 25.6ms | 0.66x | 37.20 | +3.4% |
| IDN K=4 fwd | fwd | 42.5ms | 0.40x | 36.00 | +0.1% |
| Fused+corr K=4 fwd | fwd | 31.0ms | 0.55x | 36.00 | +0.1% |
| IDN K=1 warm | warm | 19.8ms | 0.85x | 35.98 | -0.0% |
| Fused+corr K=1 warm | warm | 18.4ms | 0.92x | 35.98 | +0.0% |
| IDN K=2 warm | warm | 25.9ms | 0.65x | 35.98 | -0.0% |
| Fused+corr K=2 warm | warm | 21.1ms | 0.80x | 35.98 | +0.0% |
| IDN K=4 warm | warm | 38.0ms | 0.44x | 35.98 | -0.0% |
| Fused+corr K=4 warm | warm | 26.5ms | 0.64x | 35.98 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 85.4ms | 0.20x | 35.98 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 38.3ms | 0.44x | 35.98 | +0.0% |

## Diag λ=0.01 from IDN (seq PPL=39.03, seq=16.9ms)

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 16.9ms | 1.00x | 39.03 | --- |
| IDN K=1 h0 | h0 | 20.0ms | 0.85x | 40.33 | +3.3% |
| Fused+corr K=1 h0 | h0 | 18.6ms | 0.91x | 40.33 | +3.3% |
| IDN K=2 h0 | h0 | 26.4ms | 0.64x | 45.64 | +16.9% |
| Fused+corr K=2 h0 | h0 | 21.6ms | 0.78x | 45.64 | +16.9% |
| IDN K=4 h0 | h0 | 38.2ms | 0.44x | 39.25 | +0.6% |
| Fused+corr K=4 h0 | h0 | 26.7ms | 0.63x | 39.25 | +0.6% |
| Fused h0 | h0 | 15.8ms | 1.07x | 53.95 | +38.2% |
| IDN K=1 fwd | fwd | 24.3ms | 0.70x | 41.35 | +6.0% |
| Fused+corr K=1 fwd | fwd | 22.8ms | 0.74x | 41.36 | +6.0% |
| IDN K=2 fwd | fwd | 30.4ms | 0.56x | 42.41 | +8.7% |
| Fused+corr K=2 fwd | fwd | 25.6ms | 0.66x | 42.41 | +8.7% |
| IDN K=4 fwd | fwd | 42.5ms | 0.40x | 39.11 | +0.2% |
| Fused+corr K=4 fwd | fwd | 31.0ms | 0.54x | 39.12 | +0.2% |
| IDN K=1 warm | warm | 19.9ms | 0.85x | 39.03 | -0.0% |
| Fused+corr K=1 warm | warm | 18.4ms | 0.92x | 39.03 | +0.0% |
| IDN K=2 warm | warm | 25.9ms | 0.65x | 39.03 | -0.0% |
| Fused+corr K=2 warm | warm | 21.1ms | 0.80x | 39.03 | +0.0% |
| IDN K=4 warm | warm | 38.0ms | 0.44x | 39.03 | -0.0% |
| Fused+corr K=4 warm | warm | 26.5ms | 0.64x | 39.03 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 85.4ms | 0.20x | 39.04 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 38.8ms | 0.44x | 39.04 | +0.0% |

## Diag λ=0.05 scratch (seq PPL=43.65, seq=17.0ms)

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 17.0ms | 1.00x | 43.65 | --- |
| IDN K=1 h0 | h0 | 22.3ms | 0.76x | 48.73 | +11.6% |
| Fused+corr K=1 h0 | h0 | 20.9ms | 0.81x | 48.72 | +11.6% |
| IDN K=2 h0 | h0 | 28.4ms | 0.60x | 51.06 | +17.0% |
| Fused+corr K=2 h0 | h0 | 23.9ms | 0.71x | 51.06 | +17.0% |
| IDN K=4 h0 | h0 | 40.5ms | 0.42x | 44.36 | +1.6% |
| Fused+corr K=4 h0 | h0 | 28.8ms | 0.59x | 44.36 | +1.6% |
| Fused h0 | h0 | 15.9ms | 1.07x | 3901.36 | +8837.6% |
| IDN K=1 fwd | fwd | 25.6ms | 0.66x | 48.11 | +10.2% |
| Fused+corr K=1 fwd | fwd | 24.1ms | 0.71x | 48.11 | +10.2% |
| IDN K=2 fwd | fwd | 31.5ms | 0.54x | 47.65 | +9.2% |
| Fused+corr K=2 fwd | fwd | 26.7ms | 0.64x | 47.64 | +9.1% |
| IDN K=4 fwd | fwd | 43.6ms | 0.39x | 43.86 | +0.5% |
| Fused+corr K=4 fwd | fwd | 32.2ms | 0.53x | 43.86 | +0.5% |
| IDN K=1 warm | warm | 21.2ms | 0.80x | 43.65 | -0.0% |
| Fused+corr K=1 warm | warm | 19.7ms | 0.86x | 43.65 | +0.0% |
| IDN K=2 warm | warm | 27.1ms | 0.63x | 43.65 | -0.0% |
| Fused+corr K=2 warm | warm | 22.2ms | 0.76x | 43.65 | +0.0% |
| IDN K=4 warm | warm | 39.1ms | 0.43x | 43.65 | +0.0% |
| Fused+corr K=4 warm | warm | 28.1ms | 0.61x | 43.65 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 85.9ms | 0.20x | 43.75 | +0.2% |
| Diag vjp p=4 K=1 warm | warm | 38.9ms | 0.44x | 43.66 | +0.0% |

## Diag λ=0.1 scratch (seq PPL=44.15, seq=16.7ms)

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 16.7ms | 1.00x | 44.15 | --- |
| IDN K=1 h0 | h0 | 20.8ms | 0.81x | 48.44 | +9.7% |
| Fused+corr K=1 h0 | h0 | 19.5ms | 0.86x | 48.54 | +10.0% |
| IDN K=2 h0 | h0 | 27.3ms | 0.61x | 48.02 | +8.8% |
| Fused+corr K=2 h0 | h0 | 22.6ms | 0.74x | 48.00 | +8.7% |
| IDN K=4 h0 | h0 | 39.4ms | 0.42x | 43.95 | -0.5% |
| Fused+corr K=4 h0 | h0 | 27.5ms | 0.61x | 43.92 | -0.5% |
| Fused h0 | h0 | 15.7ms | 1.07x | diverged | --- |
| IDN K=1 fwd | fwd | 25.2ms | 0.66x | 48.88 | +10.7% |
| Fused+corr K=1 fwd | fwd | 23.8ms | 0.70x | 48.92 | +10.8% |
| IDN K=2 fwd | fwd | 31.7ms | 0.53x | 48.50 | +9.9% |
| Fused+corr K=2 fwd | fwd | 27.0ms | 0.62x | 48.47 | +9.8% |
| IDN K=4 fwd | fwd | 43.8ms | 0.38x | 44.29 | +0.3% |
| Fused+corr K=4 fwd | fwd | 32.5ms | 0.51x | 44.27 | +0.3% |
| IDN K=1 warm | warm | 21.2ms | 0.79x | 44.17 | +0.0% |
| Fused+corr K=1 warm | warm | 19.6ms | 0.85x | 44.15 | +0.0% |
| IDN K=2 warm | warm | 27.1ms | 0.62x | 44.17 | +0.0% |
| Fused+corr K=2 warm | warm | 22.3ms | 0.75x | 44.15 | +0.0% |
| IDN K=4 warm | warm | 38.6ms | 0.43x | 44.17 | +0.0% |
| Fused+corr K=4 warm | warm | 27.3ms | 0.61x | 44.15 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 85.6ms | 0.20x | 44.15 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 38.7ms | 0.43x | 44.15 | -0.0% |

## Diag λ=0.5 scratch (seq PPL=47.90, seq=16.7ms)

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 16.7ms | 1.00x | 47.90 | --- |
| IDN K=1 h0 | h0 | 20.9ms | 0.80x | 48.83 | +1.9% |
| Fused+corr K=1 h0 | h0 | 19.5ms | 0.86x | 48.88 | +2.1% |
| IDN K=2 h0 | h0 | 27.4ms | 0.61x | 49.49 | +3.3% |
| Fused+corr K=2 h0 | h0 | 22.6ms | 0.74x | 49.46 | +3.3% |
| IDN K=4 h0 | h0 | 39.0ms | 0.43x | 47.91 | +0.0% |
| Fused+corr K=4 h0 | h0 | 27.6ms | 0.61x | 47.89 | -0.0% |
| Fused h0 | h0 | 15.7ms | 1.06x | diverged | --- |
| IDN K=1 fwd | fwd | 25.1ms | 0.67x | 49.28 | +2.9% |
| Fused+corr K=1 fwd | fwd | 23.8ms | 0.70x | 49.28 | +2.9% |
| IDN K=2 fwd | fwd | 31.2ms | 0.54x | 48.81 | +1.9% |
| Fused+corr K=2 fwd | fwd | 26.4ms | 0.63x | 48.81 | +1.9% |
| IDN K=4 fwd | fwd | 43.3ms | 0.39x | 47.92 | +0.0% |
| Fused+corr K=4 fwd | fwd | 31.9ms | 0.52x | 47.90 | -0.0% |
| IDN K=1 warm | warm | 20.6ms | 0.81x | 47.92 | +0.0% |
| Fused+corr K=1 warm | warm | 19.0ms | 0.88x | 47.90 | +0.0% |
| IDN K=2 warm | warm | 26.5ms | 0.63x | 47.92 | +0.0% |
| Fused+corr K=2 warm | warm | 21.7ms | 0.77x | 47.90 | +0.0% |
| IDN K=4 warm | warm | 38.5ms | 0.43x | 47.92 | +0.0% |
| Fused+corr K=4 warm | warm | 27.2ms | 0.61x | 47.90 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 85.5ms | 0.20x | 47.90 | -0.0% |
| Diag vjp p=4 K=1 warm | warm | 39.1ms | 0.43x | 47.89 | -0.0% |

## Comparison with Old Eval (eval_jacobi_idn.md)

The old eval used only 1024 tokens (8 batches × 1 × 128), causing high variance:

| Model | Old Seq PPL (1K tok) | New Seq PPL (42M tok) | Old IDN loop PPL | New Warm IDN PPL |
|---|---|---|---|---|
| Baseline (no reg) | 26.89 | 37.94 | 1992.00 | 37.95 |
| IDN λ=0.5, npar=8 | 28.80 | 35.98 | 28.71 | 35.98 |
| Diag λ=0.01 from IDN | 31.08 | 39.03 | 31.28 | 39.03 |
| Diag λ=0.05 scratch | 29.81 | 43.65 | 36.14 | 43.65 |
| Diag λ=0.1 scratch | 28.24 | 44.15 | 33.38 | 44.17 |
| Diag λ=0.5 scratch | 31.03 | 47.90 | 32.85 | 47.92 |

## Reproduction

```bash
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_jacobi_full \
    --model-tag d32_baseline --model-tag d32_idn05_npar8 \
    --n-par 8 --report eval_jacobi_d32p8.md
```
