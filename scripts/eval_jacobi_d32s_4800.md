# Layer-Parallel Inference: d32s 4800-step Models

depth=32, n_embd=640, 5 heads, ~535M params. Single H100, SDPA, seq_len=128, BS=8.
PPL: **42,000,000 tokens** from validation set (same as training eval).
Timing: **250 runs + 50 warmup** on single batch (decoupled from PPL).

Init strategies:
- **h0**: all parallel layers = prefix output
- **fwd**: batched forward init — each layer starts at f_j(h_init)
- **warm**: sequential hidden states as init (simulates AR decoding)

## Training Findings

| Model | Seq PPL | vs Baseline |
|---|---|---|
| baseline | 69.94 | +0.0% |
| diag01 | 65.42 | -6.5% |
| diag05 | 280.79 | +301.5% |
| idn01 | 76.05 | +8.7% |
| idn05 | 59.50 | -14.9% |

IDN λ=0.5 gives strongest improvement (-14.9%). diag01 also helps (-6.5%).
IDN λ=0.1 hurts at 4800 steps (+8.7%) — needs more training. diag05 severely undertrained (+301%).

## Summary: Warm-Start IDN K=1 (AR decoding scenario)

| Model | Seq PPL | n_par=8 |  | n_par=16 |  | n_par=24 |  |
|---|---|---|---|---|---|---|---|
|  |  | PPL (dPPL%) | Speed | PPL (dPPL%) | Speed | PPL (dPPL%) | Speed |
| **baseline** | 69.94 | 69.95 (+0.0%) | 0.83x | 69.94 (+0.0%) | 1.06x | 69.95 (+0.0%) | 1.27x |
| **diag01** | 65.42 | 65.43 (+0.0%) | 0.82x | 65.43 (+0.0%) | 1.07x | 65.43 (+0.0%) | 1.31x |
| **diag05** | 280.79 | 280.80 (+0.0%) | 0.92x | 280.79 (+0.0%) | 1.14x | 280.79 (+0.0%) | 1.39x |
| **idn01** | 76.05 | 76.08 (+0.0%) | 0.84x | 76.13 (+0.1%) | 1.08x | 76.10 (+0.1%) | 1.20x |
| **idn05** | 59.50 | 59.57 (+0.1%) | 0.90x | 59.63 (+0.2%) | 1.10x | 59.67 (+0.3%) | 1.34x |

## Summary: Best Speedup with Warm-Start (speed > 1.0x)

| Model | Seq PPL | n_par | Method | PPL | dPPL% | Speed |
|---|---|---|---|---|---|---|
| baseline | 69.94 | 24 | Fused+corr K=1 warm | 69.94 | +0.00% | 1.42x |
| diag01 | 65.42 | 24 | Fused+corr K=1 warm | 65.42 | +0.00% | 1.46x |
| diag05 | 280.79 | 24 | Fused+corr K=1 warm | 280.79 | +0.00% | 1.54x |
| idn01 | 76.05 | 24 | Fused+corr K=1 warm | 76.05 | -0.00% | 1.32x |
| idn05 | 59.50 | 24 | Fused+corr K=1 warm | 59.50 | +0.00% | 1.50x |

## baseline (seq PPL=69.94, seq=13.1ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 69.94 | --- |
| IDN K=1 h0 | h0 | 16.1ms | 0.81x | 1733.55 | +2378.5% |
| Fused+corr K=1 h0 | h0 | 15.9ms | 0.83x | 1733.80 | +2378.9% |
| IDN K=2 h0 | h0 | 17.9ms | 0.73x | 264.73 | +278.5% |
| Fused+corr K=2 h0 | h0 | 17.9ms | 0.73x | 264.74 | +278.5% |
| IDN K=4 h0 | h0 | 21.9ms | 0.60x | 71.61 | +2.4% |
| Fused+corr K=4 h0 | h0 | 20.4ms | 0.64x | 71.61 | +2.4% |
| Fused h0 | h0 | 11.1ms | 1.18x | 96.87 | +38.5% |
| IDN K=1 fwd | fwd | 17.1ms | 0.77x | 156.61 | +123.9% |
| Fused+corr K=1 fwd | fwd | 17.0ms | 0.77x | 156.63 | +123.9% |
| IDN K=2 fwd | fwd | 19.0ms | 0.69x | 128.44 | +83.6% |
| Fused+corr K=2 fwd | fwd | 18.0ms | 0.73x | 128.43 | +83.6% |
| IDN K=4 fwd | fwd | 22.7ms | 0.58x | 70.41 | +0.7% |
| Fused+corr K=4 fwd | fwd | 21.5ms | 0.61x | 70.41 | +0.7% |
| IDN K=1 warm | warm | 15.9ms | 0.83x | 69.95 | +0.0% |
| Fused+corr K=1 warm | warm | 16.0ms | 0.82x | 69.94 | +0.0% |
| IDN K=2 warm | warm | 17.9ms | 0.73x | 69.95 | +0.0% |
| Fused+corr K=2 warm | warm | 16.9ms | 0.78x | 69.94 | +0.0% |
| IDN K=4 warm | warm | 21.7ms | 0.60x | 69.95 | +0.0% |
| Fused+corr K=4 warm | warm | 21.0ms | 0.62x | 69.94 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 40.7ms | 0.32x | 69.95 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 24.0ms | 0.55x | 69.95 | +0.0% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 69.94 | --- |
| IDN K=1 h0 | h0 | 13.4ms | 0.98x | 416.57 | +495.6% |
| Fused+corr K=1 h0 | h0 | 12.4ms | 1.06x | 416.60 | +495.6% |
| IDN K=2 h0 | h0 | 15.5ms | 0.84x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 13.8ms | 0.95x | diverged | --- |
| IDN K=4 h0 | h0 | 21.6ms | 0.61x | 3202.12 | +4478.2% |
| Fused+corr K=4 h0 | h0 | 18.1ms | 0.73x | 3200.60 | +4476.0% |
| Fused h0 | h0 | 7.8ms | 1.68x | 1032.84 | +1376.7% |
| IDN K=1 fwd | fwd | 13.8ms | 0.95x | 137.98 | +97.3% |
| Fused+corr K=1 fwd | fwd | 13.3ms | 0.98x | 137.98 | +97.3% |
| IDN K=2 fwd | fwd | 17.3ms | 0.76x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 15.7ms | 0.84x | diverged | --- |
| IDN K=4 fwd | fwd | 23.6ms | 0.56x | 487.82 | +597.5% |
| Fused+corr K=4 fwd | fwd | 20.0ms | 0.65x | 487.64 | +597.2% |
| IDN K=1 warm | warm | 12.4ms | 1.06x | 69.94 | +0.0% |
| Fused+corr K=1 warm | warm | 11.5ms | 1.14x | 69.94 | +0.0% |
| IDN K=2 warm | warm | 15.7ms | 0.84x | 69.98 | +0.1% |
| Fused+corr K=2 warm | warm | 14.0ms | 0.93x | 69.94 | +0.0% |
| IDN K=4 warm | warm | 21.9ms | 0.60x | 70.08 | +0.2% |
| Fused+corr K=4 warm | warm | 18.6ms | 0.71x | 69.94 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 49.8ms | 0.26x | 69.95 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 22.8ms | 0.57x | 69.95 | +0.0% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 69.94 | --- |
| IDN K=1 h0 | h0 | 10.3ms | 1.27x | 8139.48 | +11537.4% |
| Fused+corr K=1 h0 | h0 | 9.2ms | 1.43x | 8140.23 | +11538.5% |
| IDN K=2 h0 | h0 | 14.9ms | 0.88x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 12.2ms | 1.07x | diverged | --- |
| IDN K=4 h0 | h0 | 24.5ms | 0.53x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 17.6ms | 0.74x | diverged | --- |
| Fused h0 | h0 | 5.6ms | 2.34x | 34118.85 | +48681.4% |
| IDN K=1 fwd | fwd | 12.8ms | 1.02x | 1791.86 | +2461.9% |
| Fused+corr K=1 fwd | fwd | 11.3ms | 1.16x | 1791.88 | +2461.9% |
| IDN K=2 fwd | fwd | 17.3ms | 0.76x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 14.0ms | 0.93x | diverged | --- |
| IDN K=4 fwd | fwd | 26.9ms | 0.49x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 20.3ms | 0.65x | diverged | --- |
| IDN K=1 warm | warm | 10.3ms | 1.27x | 69.95 | +0.0% |
| Fused+corr K=1 warm | warm | 9.2ms | 1.42x | 69.94 | +0.0% |
| IDN K=2 warm | warm | 15.0ms | 0.87x | 70.04 | +0.1% |
| Fused+corr K=2 warm | warm | 12.1ms | 1.08x | 69.94 | +0.0% |
| IDN K=4 warm | warm | 24.3ms | 0.54x | 70.92 | +1.4% |
| Fused+corr K=4 warm | warm | 17.9ms | 0.73x | 69.94 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.4ms | 0.19x | 69.96 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 26.7ms | 0.49x | 69.97 | +0.0% |

## diag01 (seq PPL=65.42, seq=13.6ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.6ms | 1.00x | 65.42 | --- |
| IDN K=1 h0 | h0 | 16.4ms | 0.83x | 64.12 | -2.0% |
| Fused+corr K=1 h0 | h0 | 16.3ms | 0.83x | 64.12 | -2.0% |
| IDN K=2 h0 | h0 | 18.4ms | 0.74x | 177.79 | +171.8% |
| Fused+corr K=2 h0 | h0 | 18.3ms | 0.74x | 177.79 | +171.7% |
| IDN K=4 h0 | h0 | 22.7ms | 0.60x | 63.55 | -2.9% |
| Fused+corr K=4 h0 | h0 | 21.1ms | 0.64x | 63.55 | -2.9% |
| Fused h0 | h0 | 11.4ms | 1.19x | 141.24 | +115.9% |
| IDN K=1 fwd | fwd | 17.6ms | 0.77x | 78.47 | +19.9% |
| Fused+corr K=1 fwd | fwd | 17.5ms | 0.78x | 78.47 | +19.9% |
| IDN K=2 fwd | fwd | 19.6ms | 0.69x | 71.51 | +9.3% |
| Fused+corr K=2 fwd | fwd | 19.0ms | 0.71x | 71.51 | +9.3% |
| IDN K=4 fwd | fwd | 23.2ms | 0.59x | 65.23 | -0.3% |
| Fused+corr K=4 fwd | fwd | 22.1ms | 0.62x | 65.23 | -0.3% |
| IDN K=1 warm | warm | 16.6ms | 0.82x | 65.43 | +0.0% |
| Fused+corr K=1 warm | warm | 16.5ms | 0.82x | 65.42 | +0.0% |
| IDN K=2 warm | warm | 18.5ms | 0.73x | 65.43 | +0.0% |
| Fused+corr K=2 warm | warm | 17.1ms | 0.80x | 65.42 | +0.0% |
| IDN K=4 warm | warm | 21.0ms | 0.65x | 65.43 | +0.0% |
| Fused+corr K=4 warm | warm | 20.2ms | 0.67x | 65.42 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 39.9ms | 0.34x | 65.43 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 23.3ms | 0.58x | 65.43 | +0.0% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.6ms | 1.00x | 65.42 | --- |
| IDN K=1 h0 | h0 | 12.7ms | 1.07x | 126.19 | +92.9% |
| Fused+corr K=1 h0 | h0 | 12.4ms | 1.10x | 126.19 | +92.9% |
| IDN K=2 h0 | h0 | 16.0ms | 0.85x | 395.84 | +505.0% |
| Fused+corr K=2 h0 | h0 | 14.8ms | 0.92x | 395.73 | +504.9% |
| IDN K=4 h0 | h0 | 22.3ms | 0.61x | 80.44 | +23.0% |
| Fused+corr K=4 h0 | h0 | 19.3ms | 0.70x | 80.44 | +23.0% |
| Fused h0 | h0 | 8.3ms | 1.64x | 4108.73 | +6180.2% |
| IDN K=1 fwd | fwd | 14.5ms | 0.94x | 187.46 | +186.5% |
| Fused+corr K=1 fwd | fwd | 13.9ms | 0.98x | 187.45 | +186.5% |
| IDN K=2 fwd | fwd | 17.6ms | 0.77x | 288.26 | +340.6% |
| Fused+corr K=2 fwd | fwd | 16.1ms | 0.84x | 288.19 | +340.5% |
| IDN K=4 fwd | fwd | 24.0ms | 0.57x | 73.46 | +12.3% |
| Fused+corr K=4 fwd | fwd | 20.7ms | 0.66x | 73.45 | +12.3% |
| IDN K=1 warm | warm | 12.7ms | 1.07x | 65.43 | +0.0% |
| Fused+corr K=1 warm | warm | 12.2ms | 1.11x | 65.42 | +0.0% |
| IDN K=2 warm | warm | 15.9ms | 0.86x | 65.44 | +0.0% |
| Fused+corr K=2 warm | warm | 14.1ms | 0.96x | 65.42 | +0.0% |
| IDN K=4 warm | warm | 21.8ms | 0.62x | 65.46 | +0.1% |
| Fused+corr K=4 warm | warm | 19.2ms | 0.71x | 65.42 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 50.1ms | 0.27x | 65.44 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 23.2ms | 0.59x | 65.43 | +0.0% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.6ms | 1.00x | 65.42 | --- |
| IDN K=1 h0 | h0 | 10.3ms | 1.32x | 426.35 | +551.7% |
| Fused+corr K=1 h0 | h0 | 9.4ms | 1.44x | 426.35 | +551.7% |
| IDN K=2 h0 | h0 | 15.1ms | 0.90x | 5348.56 | +8075.2% |
| Fused+corr K=2 h0 | h0 | 12.4ms | 1.10x | 5346.89 | +8072.7% |
| IDN K=4 h0 | h0 | 24.5ms | 0.55x | 671.22 | +926.0% |
| Fused+corr K=4 h0 | h0 | 18.4ms | 0.74x | 671.00 | +925.6% |
| Fused h0 | h0 | 5.7ms | 2.39x | 85282.11 | +130253.2% |
| IDN K=1 fwd | fwd | 12.6ms | 1.08x | diverged | --- |
| Fused+corr K=1 fwd | fwd | 11.3ms | 1.20x | diverged | --- |
| IDN K=2 fwd | fwd | 17.5ms | 0.78x | 677.50 | +935.6% |
| Fused+corr K=2 fwd | fwd | 14.2ms | 0.96x | 677.15 | +935.0% |
| IDN K=4 fwd | fwd | 26.9ms | 0.51x | 7527.62 | +11405.9% |
| Fused+corr K=4 fwd | fwd | 20.6ms | 0.66x | 7511.86 | +11381.8% |
| IDN K=1 warm | warm | 10.4ms | 1.31x | 65.43 | +0.0% |
| Fused+corr K=1 warm | warm | 9.3ms | 1.46x | 65.42 | +0.0% |
| IDN K=2 warm | warm | 15.1ms | 0.90x | 65.45 | +0.0% |
| Fused+corr K=2 warm | warm | 12.2ms | 1.11x | 65.42 | +0.0% |
| IDN K=4 warm | warm | 24.4ms | 0.56x | 65.55 | +0.2% |
| Fused+corr K=4 warm | warm | 18.4ms | 0.74x | 65.42 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.4ms | 0.20x | 65.44 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 26.7ms | 0.51x | 65.45 | +0.0% |

## diag05 (seq PPL=280.79, seq=14.3ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 14.3ms | 1.00x | 280.79 | --- |
| IDN K=1 h0 | h0 | 16.7ms | 0.85x | 339.07 | +20.8% |
| Fused+corr K=1 h0 | h0 | 17.0ms | 0.84x | 339.07 | +20.8% |
| IDN K=2 h0 | h0 | 19.1ms | 0.75x | 281.39 | +0.2% |
| Fused+corr K=2 h0 | h0 | 19.0ms | 0.75x | 281.39 | +0.2% |
| IDN K=4 h0 | h0 | 23.5ms | 0.61x | 276.87 | -1.4% |
| Fused+corr K=4 h0 | h0 | 21.8ms | 0.65x | 276.86 | -1.4% |
| Fused h0 | h0 | 11.8ms | 1.22x | 16536.73 | +5789.4% |
| IDN K=1 fwd | fwd | 18.4ms | 0.78x | 362.97 | +29.3% |
| Fused+corr K=1 fwd | fwd | 18.3ms | 0.78x | 362.95 | +29.3% |
| IDN K=2 fwd | fwd | 20.2ms | 0.71x | 273.85 | -2.5% |
| Fused+corr K=2 fwd | fwd | 19.8ms | 0.72x | 273.85 | -2.5% |
| IDN K=4 fwd | fwd | 24.2ms | 0.59x | 280.81 | +0.0% |
| Fused+corr K=4 fwd | fwd | 21.8ms | 0.65x | 280.79 | +0.0% |
| IDN K=1 warm | warm | 15.5ms | 0.92x | 280.80 | +0.0% |
| Fused+corr K=1 warm | warm | 15.3ms | 0.93x | 280.79 | +0.0% |
| IDN K=2 warm | warm | 17.2ms | 0.83x | 280.81 | +0.0% |
| Fused+corr K=2 warm | warm | 16.7ms | 0.85x | 280.79 | +0.0% |
| IDN K=4 warm | warm | 20.9ms | 0.69x | 280.82 | +0.0% |
| Fused+corr K=4 warm | warm | 19.8ms | 0.72x | 280.79 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 39.2ms | 0.36x | 280.86 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 22.9ms | 0.63x | 280.83 | +0.0% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 14.3ms | 1.00x | 280.79 | --- |
| IDN K=1 h0 | h0 | 12.5ms | 1.14x | 142.48 | -49.3% |
| Fused+corr K=1 h0 | h0 | 12.2ms | 1.18x | 142.48 | -49.3% |
| IDN K=2 h0 | h0 | 15.9ms | 0.90x | 441.87 | +57.4% |
| Fused+corr K=2 h0 | h0 | 14.4ms | 0.99x | 441.78 | +57.3% |
| IDN K=4 h0 | h0 | 22.0ms | 0.65x | 365.23 | +30.1% |
| Fused+corr K=4 h0 | h0 | 18.9ms | 0.75x | 365.25 | +30.1% |
| Fused h0 | h0 | 8.1ms | 1.77x | diverged | --- |
| IDN K=1 fwd | fwd | 14.3ms | 1.00x | 268.15 | -4.5% |
| Fused+corr K=1 fwd | fwd | 13.6ms | 1.05x | 268.14 | -4.5% |
| IDN K=2 fwd | fwd | 17.4ms | 0.82x | 771.05 | +174.6% |
| Fused+corr K=2 fwd | fwd | 15.9ms | 0.90x | 770.91 | +174.6% |
| IDN K=4 fwd | fwd | 23.6ms | 0.60x | 429.20 | +52.9% |
| Fused+corr K=4 fwd | fwd | 20.3ms | 0.70x | 429.29 | +52.9% |
| IDN K=1 warm | warm | 12.6ms | 1.14x | 280.79 | +0.0% |
| Fused+corr K=1 warm | warm | 12.0ms | 1.19x | 280.79 | +0.0% |
| IDN K=2 warm | warm | 15.7ms | 0.91x | 280.80 | +0.0% |
| Fused+corr K=2 warm | warm | 14.2ms | 1.01x | 280.79 | +0.0% |
| IDN K=4 warm | warm | 21.9ms | 0.65x | 280.83 | +0.0% |
| Fused+corr K=4 warm | warm | 18.8ms | 0.76x | 280.79 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 49.7ms | 0.29x | 280.87 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 22.9ms | 0.63x | 280.86 | +0.0% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 14.3ms | 1.00x | 280.79 | --- |
| IDN K=1 h0 | h0 | 10.3ms | 1.39x | 650.99 | +131.8% |
| Fused+corr K=1 h0 | h0 | 9.4ms | 1.52x | 650.98 | +131.8% |
| IDN K=2 h0 | h0 | 15.0ms | 0.95x | 1337.23 | +376.2% |
| Fused+corr K=2 h0 | h0 | 12.3ms | 1.17x | 1337.14 | +376.2% |
| IDN K=4 h0 | h0 | 24.4ms | 0.59x | 488.41 | +73.9% |
| Fused+corr K=4 h0 | h0 | 18.3ms | 0.78x | 488.48 | +74.0% |
| Fused h0 | h0 | 5.6ms | 2.56x | diverged | --- |
| IDN K=1 fwd | fwd | 12.8ms | 1.12x | 2135.25 | +660.4% |
| Fused+corr K=1 fwd | fwd | 11.3ms | 1.27x | 2134.90 | +660.3% |
| IDN K=2 fwd | fwd | 17.4ms | 0.82x | 1332.03 | +374.4% |
| Fused+corr K=2 fwd | fwd | 14.0ms | 1.02x | 1331.97 | +374.4% |
| IDN K=4 fwd | fwd | 26.7ms | 0.53x | 320.20 | +14.0% |
| Fused+corr K=4 fwd | fwd | 20.4ms | 0.70x | 320.29 | +14.1% |
| IDN K=1 warm | warm | 10.3ms | 1.39x | 280.79 | +0.0% |
| Fused+corr K=1 warm | warm | 9.3ms | 1.54x | 280.79 | +0.0% |
| IDN K=2 warm | warm | 14.9ms | 0.96x | 280.80 | +0.0% |
| Fused+corr K=2 warm | warm | 12.2ms | 1.17x | 280.79 | +0.0% |
| IDN K=4 warm | warm | 24.3ms | 0.59x | 280.84 | +0.0% |
| Fused+corr K=4 warm | warm | 18.1ms | 0.79x | 280.79 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.1ms | 0.21x | 280.87 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 26.5ms | 0.54x | 280.83 | +0.0% |

## idn01 (seq PPL=76.05, seq=13.3ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.3ms | 1.00x | 76.05 | --- |
| IDN K=1 h0 | h0 | 16.1ms | 0.83x | 76.12 | +0.1% |
| Fused+corr K=1 h0 | h0 | 15.4ms | 0.86x | 76.12 | +0.1% |
| IDN K=2 h0 | h0 | 17.8ms | 0.75x | 90.22 | +18.6% |
| Fused+corr K=2 h0 | h0 | 17.6ms | 0.76x | 90.20 | +18.6% |
| IDN K=4 h0 | h0 | 21.9ms | 0.61x | 76.00 | -0.1% |
| Fused+corr K=4 h0 | h0 | 20.1ms | 0.66x | 75.97 | -0.1% |
| Fused h0 | h0 | 11.2ms | 1.19x | 279.34 | +267.3% |
| IDN K=1 fwd | fwd | 17.0ms | 0.78x | 72.96 | -4.1% |
| Fused+corr K=1 fwd | fwd | 17.1ms | 0.78x | 72.95 | -4.1% |
| IDN K=2 fwd | fwd | 18.8ms | 0.71x | 88.56 | +16.4% |
| Fused+corr K=2 fwd | fwd | 18.4ms | 0.72x | 88.55 | +16.4% |
| IDN K=4 fwd | fwd | 22.6ms | 0.59x | 76.02 | -0.0% |
| Fused+corr K=4 fwd | fwd | 21.3ms | 0.63x | 75.99 | -0.1% |
| IDN K=1 warm | warm | 15.8ms | 0.84x | 76.08 | +0.0% |
| Fused+corr K=1 warm | warm | 15.6ms | 0.85x | 76.05 | -0.0% |
| IDN K=2 warm | warm | 17.8ms | 0.75x | 76.09 | +0.0% |
| Fused+corr K=2 warm | warm | 17.1ms | 0.78x | 76.05 | -0.0% |
| IDN K=4 warm | warm | 21.4ms | 0.62x | 76.09 | +0.1% |
| Fused+corr K=4 warm | warm | 20.4ms | 0.65x | 76.05 | -0.0% |
| Diag jvp p=4 K=1 warm | warm | 40.1ms | 0.33x | 76.11 | +0.1% |
| Diag vjp p=4 K=1 warm | warm | 23.4ms | 0.57x | 75.91 | -0.2% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.3ms | 1.00x | 76.05 | --- |
| IDN K=1 h0 | h0 | 13.6ms | 0.98x | 308.19 | +305.3% |
| Fused+corr K=1 h0 | h0 | 12.6ms | 1.05x | 308.25 | +305.3% |
| IDN K=2 h0 | h0 | 16.4ms | 0.81x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 15.0ms | 0.89x | diverged | --- |
| IDN K=4 h0 | h0 | 22.6ms | 0.59x | 1154.92 | +1418.6% |
| Fused+corr K=4 h0 | h0 | 18.6ms | 0.72x | 1158.56 | +1423.4% |
| Fused h0 | h0 | 8.1ms | 1.65x | 204.68 | +169.1% |
| IDN K=1 fwd | fwd | 14.1ms | 0.94x | 441.98 | +481.2% |
| Fused+corr K=1 fwd | fwd | 13.3ms | 1.00x | 441.92 | +481.1% |
| IDN K=2 fwd | fwd | 17.2ms | 0.77x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 15.5ms | 0.86x | diverged | --- |
| IDN K=4 fwd | fwd | 23.5ms | 0.56x | 98.44 | +29.4% |
| Fused+corr K=4 fwd | fwd | 19.8ms | 0.67x | 98.32 | +29.3% |
| IDN K=1 warm | warm | 12.3ms | 1.08x | 76.13 | +0.1% |
| Fused+corr K=1 warm | warm | 11.6ms | 1.15x | 76.05 | -0.0% |
| IDN K=2 warm | warm | 15.4ms | 0.86x | 76.16 | +0.1% |
| Fused+corr K=2 warm | warm | 15.8ms | 0.84x | 76.05 | -0.0% |
| IDN K=4 warm | warm | 23.1ms | 0.58x | 76.24 | +0.3% |
| Fused+corr K=4 warm | warm | 20.8ms | 0.64x | 76.05 | -0.0% |
| Diag jvp p=4 K=1 warm | warm | 51.0ms | 0.26x | 76.32 | +0.4% |
| Diag vjp p=4 K=1 warm | warm | 24.6ms | 0.54x | 75.85 | -0.3% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.3ms | 1.00x | 76.05 | --- |
| IDN K=1 h0 | h0 | 11.1ms | 1.20x | 663.00 | +771.8% |
| Fused+corr K=1 h0 | h0 | 10.3ms | 1.30x | 663.00 | +771.8% |
| IDN K=2 h0 | h0 | 15.9ms | 0.84x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 13.6ms | 0.98x | diverged | --- |
| IDN K=4 h0 | h0 | 25.3ms | 0.53x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 20.6ms | 0.65x | diverged | --- |
| Fused h0 | h0 | 6.2ms | 2.14x | 16851.71 | +22058.8% |
| IDN K=1 fwd | fwd | 13.5ms | 0.98x | 1803.91 | +2272.0% |
| Fused+corr K=1 fwd | fwd | 12.1ms | 1.10x | 1804.14 | +2272.3% |
| IDN K=2 fwd | fwd | 18.2ms | 0.73x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 15.4ms | 0.86x | diverged | --- |
| IDN K=4 fwd | fwd | 27.6ms | 0.48x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 22.7ms | 0.59x | diverged | --- |
| IDN K=1 warm | warm | 11.1ms | 1.20x | 76.10 | +0.1% |
| Fused+corr K=1 warm | warm | 10.1ms | 1.32x | 76.05 | -0.0% |
| IDN K=2 warm | warm | 15.7ms | 0.85x | 76.21 | +0.2% |
| Fused+corr K=2 warm | warm | 13.3ms | 1.00x | 76.05 | -0.0% |
| IDN K=4 warm | warm | 25.1ms | 0.53x | 76.56 | +0.7% |
| Fused+corr K=4 warm | warm | 20.0ms | 0.66x | 76.05 | -0.0% |
| Diag jvp p=4 K=1 warm | warm | 68.4ms | 0.19x | 76.33 | +0.4% |
| Diag vjp p=4 K=1 warm | warm | 27.5ms | 0.48x | 75.75 | -0.4% |

## idn05 (seq PPL=59.50, seq=14.0ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 14.0ms | 1.00x | 59.50 | --- |
| IDN K=1 h0 | h0 | 16.5ms | 0.85x | 76.24 | +28.1% |
| Fused+corr K=1 h0 | h0 | 16.3ms | 0.86x | 76.18 | +28.0% |
| IDN K=2 h0 | h0 | 18.2ms | 0.77x | 61.31 | +3.0% |
| Fused+corr K=2 h0 | h0 | 17.5ms | 0.80x | 61.24 | +2.9% |
| IDN K=4 h0 | h0 | 21.4ms | 0.66x | 59.62 | +0.2% |
| Fused+corr K=4 h0 | h0 | 19.8ms | 0.71x | 59.55 | +0.1% |
| Fused h0 | h0 | 10.8ms | 1.30x | 1385.63 | +2228.6% |
| IDN K=1 fwd | fwd | 16.4ms | 0.85x | 67.49 | +13.4% |
| Fused+corr K=1 fwd | fwd | 16.5ms | 0.85x | 67.32 | +13.1% |
| IDN K=2 fwd | fwd | 18.3ms | 0.77x | 66.15 | +11.2% |
| Fused+corr K=2 fwd | fwd | 17.9ms | 0.78x | 66.04 | +11.0% |
| IDN K=4 fwd | fwd | 22.0ms | 0.64x | 59.59 | +0.2% |
| Fused+corr K=4 fwd | fwd | 20.7ms | 0.68x | 59.53 | +0.0% |
| IDN K=1 warm | warm | 15.6ms | 0.90x | 59.57 | +0.1% |
| Fused+corr K=1 warm | warm | 15.3ms | 0.91x | 59.50 | +0.0% |
| IDN K=2 warm | warm | 17.3ms | 0.81x | 59.58 | +0.1% |
| Fused+corr K=2 warm | warm | 16.8ms | 0.83x | 59.50 | +0.0% |
| IDN K=4 warm | warm | 20.9ms | 0.67x | 59.58 | +0.1% |
| Fused+corr K=4 warm | warm | 20.1ms | 0.70x | 59.50 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 39.9ms | 0.35x | 59.61 | +0.2% |
| Diag vjp p=4 K=1 warm | warm | 22.6ms | 0.62x | 59.53 | +0.0% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 14.0ms | 1.00x | 59.50 | --- |
| IDN K=1 h0 | h0 | 12.8ms | 1.09x | 4872.82 | +8089.1% |
| Fused+corr K=1 h0 | h0 | 12.3ms | 1.14x | 4854.14 | +8057.7% |
| IDN K=2 h0 | h0 | 16.0ms | 0.88x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 14.7ms | 0.95x | diverged | --- |
| IDN K=4 h0 | h0 | 22.3ms | 0.63x | 43916.70 | +73704.6% |
| Fused+corr K=4 h0 | h0 | 19.3ms | 0.73x | 44165.09 | +74122.0% |
| Fused h0 | h0 | 8.3ms | 1.69x | diverged | --- |
| IDN K=1 fwd | fwd | 14.5ms | 0.97x | 9487.45 | +15844.2% |
| Fused+corr K=1 fwd | fwd | 13.8ms | 1.02x | 9493.10 | +15853.7% |
| IDN K=2 fwd | fwd | 17.6ms | 0.80x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 16.1ms | 0.87x | diverged | --- |
| IDN K=4 fwd | fwd | 23.9ms | 0.59x | 11392.76 | +19046.2% |
| Fused+corr K=4 fwd | fwd | 20.6ms | 0.68x | 11195.62 | +18714.9% |
| IDN K=1 warm | warm | 12.8ms | 1.10x | 59.63 | +0.2% |
| Fused+corr K=1 warm | warm | 12.2ms | 1.15x | 59.50 | +0.0% |
| IDN K=2 warm | warm | 15.9ms | 0.88x | 59.80 | +0.5% |
| Fused+corr K=2 warm | warm | 14.5ms | 0.97x | 59.50 | +0.0% |
| IDN K=4 warm | warm | 22.1ms | 0.63x | 59.79 | +0.5% |
| Fused+corr K=4 warm | warm | 19.2ms | 0.73x | 59.50 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 49.9ms | 0.28x | 60.18 | +1.1% |
| Diag vjp p=4 K=1 warm | warm | 22.9ms | 0.61x | 59.58 | +0.1% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 14.0ms | 1.00x | 59.50 | --- |
| IDN K=1 h0 | h0 | 10.5ms | 1.33x | 24615.64 | +41268.0% |
| Fused+corr K=1 h0 | h0 | 9.4ms | 1.49x | 24609.33 | +41257.4% |
| IDN K=2 h0 | h0 | 15.2ms | 0.92x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 12.4ms | 1.13x | diverged | --- |
| IDN K=4 h0 | h0 | 24.6ms | 0.57x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 18.5ms | 0.76x | diverged | --- |
| Fused h0 | h0 | 5.7ms | 2.48x | diverged | --- |
| IDN K=1 fwd | fwd | 12.8ms | 1.09x | 8477.28 | +14146.6% |
| Fused+corr K=1 fwd | fwd | 11.4ms | 1.23x | 8488.38 | +14165.2% |
| IDN K=2 fwd | fwd | 17.6ms | 0.80x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 14.2ms | 0.99x | diverged | --- |
| IDN K=4 fwd | fwd | 27.0ms | 0.52x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 20.7ms | 0.68x | diverged | --- |
| IDN K=1 warm | warm | 10.4ms | 1.34x | 59.67 | +0.3% |
| Fused+corr K=1 warm | warm | 9.4ms | 1.50x | 59.50 | +0.0% |
| IDN K=2 warm | warm | 15.1ms | 0.93x | 60.07 | +1.0% |
| Fused+corr K=2 warm | warm | 12.3ms | 1.14x | 59.50 | +0.0% |
| IDN K=4 warm | warm | 24.5ms | 0.57x | 60.53 | +1.7% |
| Fused+corr K=4 warm | warm | 18.4ms | 0.76x | 59.50 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.4ms | 0.21x | 60.26 | +1.3% |
| Diag vjp p=4 K=1 warm | warm | 26.8ms | 0.52x | 59.73 | +0.4% |

## Key Findings

1. **Warm-start IDN K=1 gives <0.3% PPL loss** at 1.2-1.5x speedup (n_par=24).

2. **Fused+corr K=1 warm gives best speedup** (1.3-1.5x at n_par=24, +0.0% PPL).

3. **IDN/diag reg improves sequential PPL** — acts as beneficial regularization.
   At 9600 steps, diag01 has best seq PPL. idn05 overfits (λ=0.5 too strong).

4. **No regularization needed for warm-start quality**: baseline achieves same
   parallel inference quality as regularized models with warm-start init.

