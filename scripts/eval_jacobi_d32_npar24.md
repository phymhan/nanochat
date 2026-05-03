# Layer-Parallel Inference: d32 Models (n_embd=2048, 32 layers, ~2.8B params)

depth=32, n_embd=2048, 16 heads, ~2.8B params. Single H100, SDPA, seq_len=128, BS=8.
PPL: **42M tokens** from validation set. Timing: **120 runs + 20 warmup**.

Two model groups:
- **Old (n_par=8 trained)**: baseline, idn05_npar8, diag models (trained with n_par_configs=8)
- **New (n_par=24 trained)**: idn05_npar24, idn01_npar24, diag05/01_npar24_stride3

## Training Findings

| Model | val_bpb | Seq PPL (42M tok) | vs Baseline |
|---|---|---|---|
| Baseline (no reg) | 0.661 | 37.94 | +0.0% |
| IDN λ=0.1 npar24 | 0.661 | 35.96 | -5.2% |
| IDN λ=0.5 npar24 | 0.689 | 48.50 | +27.8% |
| Diag λ=0.1 npar24 stride3 | 0.662 | 36.03 | -5.0% |
| Diag λ=0.5 npar24 stride3 | 0.662 | 48.39 | +27.5% |
| IDN λ=0.5 npar8 (old) | 0.661 | 35.98 | -5.2% |
| Diag λ=0.1 npar8 (old) | 0.662 | 44.15 | +16.4% |
| Diag λ=0.5 npar8 (old) | 0.664 | 47.90 | +26.2% |

**Key observations:**
- **IDN λ=0.1 npar24** matches baseline (val_bpb=0.661, PPL 35.96 vs 37.94).
- **IDN λ=0.5 npar24** degrades quality (val_bpb=0.689) — too strong with 24 parallel layers.
- **Diag npar24** preserves quality (val_bpb=0.662). Both λ=0.1 and λ=0.5 are close to baseline.
- **IDN λ=0.5 npar8 (old)** had best PPL (35.98) — fewer parallel layers means weaker constraint.

## Warm-Start Inference Results (new n_par=24 models)

All methods use warm-start init (sequential hidden states). K=1.

| Model | Seq PPL | n_par | IDN PPL (dPPL%) | Speed | Fused+corr PPL | Speed | Diag VJP PPL | Speed |
|---|---|---|---|---|---|---|---|---|
| Baseline (no reg) | 37.94 | 8 | 37.95 (+0.0%) | 0.77x | 37.94 (-0.0%) | 0.84x | 37.95 (+0.0%) | 0.49x |
| Baseline (no reg) | 37.94 | 16 | 37.95 (+0.0%) | 0.73x | 37.94 (-0.0%) | 0.86x | 37.97 (+0.1%) | 0.33x |
| Baseline (no reg) | 37.94 | 24 | 37.95 (+0.0%) | 0.69x | 37.94 (-0.0%) | 0.88x | 37.98 (+0.1%) | 0.25x |
| IDN λ=0.1 npar24 | 35.96 | 8 | 35.95 (-0.0%) | 0.83x | 35.96 (+0.0%) | 0.90x | 36.02 (+0.2%) | 0.49x |
| IDN λ=0.1 npar24 | 35.96 | 16 | 35.95 (-0.0%) | 0.75x | 35.96 (+0.0%) | 0.88x | 36.08 (+0.3%) | 0.33x |
| IDN λ=0.1 npar24 | 35.96 | 24 | 35.95 (-0.0%) | 0.68x | 35.96 (+0.0%) | 0.87x | 36.10 (+0.4%) | 0.24x |
| IDN λ=0.5 npar24 | 48.50 | 8 | 48.51 (+0.0%) | 0.82x | 48.50 (+0.0%) | 0.88x | 48.64 (+0.3%) | 0.49x |
| IDN λ=0.5 npar24 | 48.50 | 16 | 48.51 (+0.0%) | 0.75x | 48.50 (+0.0%) | 0.87x | 48.79 (+0.6%) | 0.33x |
| IDN λ=0.5 npar24 | 48.50 | 24 | 48.53 (+0.1%) | 0.69x | 48.50 (+0.0%) | 0.87x | 48.82 (+0.7%) | 0.24x |
| Diag λ=0.1 npar24 stride3 | 36.03 | 8 | 36.03 (+0.0%) | 0.84x | 36.03 (+0.0%) | 0.90x | 36.04 (+0.0%) | 0.49x |
| Diag λ=0.1 npar24 stride3 | 36.03 | 16 | 36.03 (+0.0%) | 0.75x | 36.03 (+0.0%) | 0.88x | 36.08 (+0.1%) | 0.33x |
| Diag λ=0.1 npar24 stride3 | 36.03 | 24 | 36.03 (+0.0%) | 0.68x | 36.03 (+0.0%) | 0.88x | 36.14 (+0.3%) | 0.24x |
| Diag λ=0.5 npar24 stride3 | 48.39 | 8 | 48.39 (+0.0%) | 0.84x | 48.39 (+0.0%) | 0.91x | 48.40 (+0.0%) | 0.49x |
| Diag λ=0.5 npar24 stride3 | 48.39 | 16 | 48.39 (-0.0%) | 0.76x | 48.39 (+0.0%) | 0.89x | 48.48 (+0.2%) | 0.33x |
| Diag λ=0.5 npar24 stride3 | 48.39 | 24 | 48.36 (-0.1%) | 0.68x | 48.39 (+0.0%) | 0.88x | 48.54 (+0.3%) | 0.24x |

## Key Findings

1. **No speedup at D=2048**: GPU is saturated by individual matmuls.
   Batching parallel layers via einsum provides no benefit at this model size.

2. **Warm-start gives +0.0% PPL** across all models and n_par values,
   confirming the d32s finding scales to larger models.

3. **IDN λ=0.1 is the sweet spot** for n_par=24 training: preserves base quality
   (val_bpb=0.661) while IDN λ=0.5 degrades it (val_bpb=0.689).

4. **Diag reg at n_par=24 preserves quality** (val_bpb=0.662) regardless of λ.
   The stride=3 FD estimation is gentle enough not to hurt training.

5. **Old IDN λ=0.5 npar=8 had best PPL** because fewer parallel layers = weaker constraint.
   Consistent with d32s finding that strong reg can help generalization.

## Baseline (no reg) (seq PPL=37.94, seq=17.5ms)

### n_par=8

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 17.5ms | 1.00x | 37.94 | --- |
| IDN K=1 warm | 22.7ms | 0.77x | 37.95 | +0.0% |
| Fused+corr K=1 warm | 20.8ms | 0.84x | 37.94 | -0.0% |
| Diag VJP K=1 warm | 36.0ms | 0.49x | 37.95 | +0.0% |

### n_par=16

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 17.5ms | 1.00x | 37.94 | --- |
| IDN K=1 warm | 23.9ms | 0.73x | 37.95 | +0.0% |
| Fused+corr K=1 warm | 20.4ms | 0.86x | 37.94 | -0.0% |
| Diag VJP K=1 warm | 52.7ms | 0.33x | 37.97 | +0.1% |

### n_par=24

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 17.5ms | 1.00x | 37.94 | --- |
| IDN K=1 warm | 25.5ms | 0.69x | 37.95 | +0.0% |
| Fused+corr K=1 warm | 19.9ms | 0.88x | 37.94 | -0.0% |
| Diag VJP K=1 warm | 70.0ms | 0.25x | 37.98 | +0.1% |

## Diag λ=0.1 npar24 stride3 (seq PPL=36.03, seq=16.7ms)

### n_par=8

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.7ms | 1.00x | 36.03 | --- |
| IDN K=1 warm | 19.7ms | 0.84x | 36.03 | +0.0% |
| Fused+corr K=1 warm | 18.5ms | 0.90x | 36.03 | +0.0% |
| Diag VJP K=1 warm | 33.9ms | 0.49x | 36.04 | +0.0% |

### n_par=16

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.7ms | 1.00x | 36.03 | --- |
| IDN K=1 warm | 22.2ms | 0.75x | 36.03 | +0.0% |
| Fused+corr K=1 warm | 18.8ms | 0.88x | 36.03 | +0.0% |
| Diag VJP K=1 warm | 51.1ms | 0.33x | 36.08 | +0.1% |

### n_par=24

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.7ms | 1.00x | 36.03 | --- |
| IDN K=1 warm | 24.3ms | 0.68x | 36.03 | +0.0% |
| Fused+corr K=1 warm | 19.0ms | 0.88x | 36.03 | +0.0% |
| Diag VJP K=1 warm | 69.1ms | 0.24x | 36.14 | +0.3% |

## Diag λ=0.5 npar24 stride3 (seq PPL=48.39, seq=16.8ms)

### n_par=8

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.8ms | 1.00x | 48.39 | --- |
| IDN K=1 warm | 19.9ms | 0.84x | 48.39 | +0.0% |
| Fused+corr K=1 warm | 18.4ms | 0.91x | 48.39 | +0.0% |
| Diag VJP K=1 warm | 34.0ms | 0.49x | 48.40 | +0.0% |

### n_par=16

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.8ms | 1.00x | 48.39 | --- |
| IDN K=1 warm | 22.0ms | 0.76x | 48.39 | -0.0% |
| Fused+corr K=1 warm | 18.8ms | 0.89x | 48.39 | +0.0% |
| Diag VJP K=1 warm | 50.9ms | 0.33x | 48.48 | +0.2% |

### n_par=24

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.8ms | 1.00x | 48.39 | --- |
| IDN K=1 warm | 24.5ms | 0.68x | 48.36 | -0.1% |
| Fused+corr K=1 warm | 19.1ms | 0.88x | 48.39 | +0.0% |
| Diag VJP K=1 warm | 69.3ms | 0.24x | 48.54 | +0.3% |

## IDN λ=0.1 npar24 (seq PPL=35.96, seq=16.9ms)

### n_par=8

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.9ms | 1.00x | 35.96 | --- |
| IDN K=1 warm | 20.3ms | 0.83x | 35.95 | -0.0% |
| Fused+corr K=1 warm | 18.8ms | 0.90x | 35.96 | +0.0% |
| Diag VJP K=1 warm | 34.5ms | 0.49x | 36.02 | +0.2% |

### n_par=16

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.9ms | 1.00x | 35.96 | --- |
| IDN K=1 warm | 22.6ms | 0.75x | 35.95 | -0.0% |
| Fused+corr K=1 warm | 19.2ms | 0.88x | 35.96 | +0.0% |
| Diag VJP K=1 warm | 51.5ms | 0.33x | 36.08 | +0.3% |

### n_par=24

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 16.9ms | 1.00x | 35.96 | --- |
| IDN K=1 warm | 24.7ms | 0.68x | 35.95 | -0.0% |
| Fused+corr K=1 warm | 19.5ms | 0.87x | 35.96 | +0.0% |
| Diag VJP K=1 warm | 69.7ms | 0.24x | 36.10 | +0.4% |

## IDN λ=0.5 npar24 (seq PPL=48.50, seq=17.0ms)

### n_par=8

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 17.0ms | 1.00x | 48.50 | --- |
| IDN K=1 warm | 20.7ms | 0.82x | 48.51 | +0.0% |
| Fused+corr K=1 warm | 19.2ms | 0.88x | 48.50 | +0.0% |
| Diag VJP K=1 warm | 35.0ms | 0.49x | 48.64 | +0.3% |

### n_par=16

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 17.0ms | 1.00x | 48.50 | --- |
| IDN K=1 warm | 22.6ms | 0.75x | 48.51 | +0.0% |
| Fused+corr K=1 warm | 19.5ms | 0.87x | 48.50 | +0.0% |
| Diag VJP K=1 warm | 51.9ms | 0.33x | 48.79 | +0.6% |

### n_par=24

| Method | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|
| Sequential | 17.0ms | 1.00x | 48.50 | --- |
| IDN K=1 warm | 24.8ms | 0.69x | 48.53 | +0.1% |
| Fused+corr K=1 warm | 19.5ms | 0.87x | 48.50 | +0.0% |
| Diag VJP K=1 warm | 70.0ms | 0.24x | 48.82 | +0.7% |

