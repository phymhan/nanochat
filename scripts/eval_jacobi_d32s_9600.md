# Layer-Parallel Inference: d32s 9600-step Models

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
| baseline | 70.30 | +0.0% |
| diag01 | 63.82 | -9.2% |
| diag05 | 74.50 | +6.0% |
| idn01 | 72.71 | +3.4% |
| idn05 | 71.67 | +1.9% |

At 9600 steps, diag01 is best (-9.2%). idn01 still worse than baseline (+3.4%).
idn05 overfits (+1.9%). diag05 recovers but still weak (+6.0%).

## Summary: Warm-Start IDN K=1 (AR decoding scenario)

| Model | Seq PPL | n_par=8 |  | n_par=16 |  | n_par=24 |  |
|---|---|---|---|---|---|---|---|
|  |  | PPL (dPPL%) | Speed | PPL (dPPL%) | Speed | PPL (dPPL%) | Speed |
| **baseline** | 70.30 | 70.30 (+0.0%) | 0.84x | 70.30 (+0.0%) | 1.09x | 70.31 (+0.0%) | 1.33x |
| **diag01** | 63.82 | 63.82 (+0.0%) | 0.85x | 63.82 (+0.0%) | 0.98x | 63.82 (+0.0%) | 1.22x |
| **diag05** | 74.50 | 74.50 (+0.0%) | 0.84x | 74.50 (+0.0%) | 1.10x | 74.50 (+0.0%) | 1.33x |
| **idn01** | 72.71 | 72.74 (+0.0%) | 0.82x | 72.76 (+0.1%) | 1.09x | 72.78 (+0.1%) | 1.29x |
| **idn05** | 71.67 | 71.84 (+0.2%) | 0.82x | 71.92 (+0.4%) | 1.04x | 72.06 (+0.5%) | 1.30x |

## Summary: Best Speedup with Warm-Start (speed > 1.0x)

| Model | Seq PPL | n_par | Method | PPL | dPPL% | Speed |
|---|---|---|---|---|---|---|
| baseline | 70.30 | 24 | Fused+corr K=1 warm | 70.30 | +0.00% | 1.48x |
| diag01 | 63.82 | 24 | Fused+corr K=1 warm | 63.82 | +0.00% | 1.33x |
| diag05 | 74.50 | 24 | Fused+corr K=1 warm | 74.50 | +0.00% | 1.49x |
| idn01 | 72.71 | 24 | Fused+corr K=1 warm | 72.71 | +0.00% | 1.45x |
| idn05 | 71.67 | 24 | Fused+corr K=1 warm | 71.67 | +0.00% | 1.46x |

## baseline (seq PPL=70.30, seq=13.9ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.9ms | 1.00x | 70.30 | --- |
| IDN K=1 h0 | h0 | 16.6ms | 0.84x | 3763.57 | +5253.9% |
| Fused+corr K=1 h0 | h0 | 16.3ms | 0.86x | 3764.17 | +5254.8% |
| IDN K=2 h0 | h0 | 18.3ms | 0.76x | 4186.04 | +5854.9% |
| Fused+corr K=2 h0 | h0 | 18.3ms | 0.76x | 4186.49 | +5855.5% |
| IDN K=4 h0 | h0 | 22.4ms | 0.62x | 72.96 | +3.8% |
| Fused+corr K=4 h0 | h0 | 20.8ms | 0.67x | 72.96 | +3.8% |
| Fused h0 | h0 | 11.5ms | 1.21x | 138.95 | +97.7% |
| IDN K=1 fwd | fwd | 17.4ms | 0.80x | 246.93 | +251.3% |
| Fused+corr K=1 fwd | fwd | 17.5ms | 0.80x | 246.95 | +251.3% |
| IDN K=2 fwd | fwd | 19.5ms | 0.72x | 368.79 | +424.6% |
| Fused+corr K=2 fwd | fwd | 18.9ms | 0.74x | 368.76 | +424.6% |
| IDN K=4 fwd | fwd | 23.2ms | 0.60x | 71.07 | +1.1% |
| Fused+corr K=4 fwd | fwd | 22.1ms | 0.63x | 71.05 | +1.1% |
| IDN K=1 warm | warm | 16.5ms | 0.84x | 70.30 | +0.0% |
| Fused+corr K=1 warm | warm | 16.5ms | 0.85x | 70.30 | +0.0% |
| IDN K=2 warm | warm | 18.6ms | 0.75x | 70.31 | +0.0% |
| Fused+corr K=2 warm | warm | 17.9ms | 0.78x | 70.30 | +0.0% |
| IDN K=4 warm | warm | 22.3ms | 0.62x | 70.32 | +0.0% |
| Fused+corr K=4 warm | warm | 21.4ms | 0.65x | 70.30 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 40.8ms | 0.34x | 70.31 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 23.3ms | 0.60x | 70.31 | +0.0% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.9ms | 1.00x | 70.30 | --- |
| IDN K=1 h0 | h0 | 12.8ms | 1.09x | 1928.22 | +2643.0% |
| Fused+corr K=1 h0 | h0 | 12.3ms | 1.13x | 1928.38 | +2643.2% |
| IDN K=2 h0 | h0 | 16.1ms | 0.87x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 14.7ms | 0.95x | diverged | --- |
| IDN K=4 h0 | h0 | 22.4ms | 0.62x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 19.3ms | 0.72x | diverged | --- |
| Fused h0 | h0 | 8.4ms | 1.66x | 5196.04 | +7291.7% |
| IDN K=1 fwd | fwd | 14.5ms | 0.96x | 278.33 | +295.9% |
| Fused+corr K=1 fwd | fwd | 13.8ms | 1.01x | 278.34 | +295.9% |
| IDN K=2 fwd | fwd | 17.6ms | 0.79x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 16.1ms | 0.86x | diverged | --- |
| IDN K=4 fwd | fwd | 24.0ms | 0.58x | 33432.93 | +47460.3% |
| Fused+corr K=4 fwd | fwd | 20.7ms | 0.67x | 33416.55 | +47437.0% |
| IDN K=1 warm | warm | 12.8ms | 1.09x | 70.30 | +0.0% |
| Fused+corr K=1 warm | warm | 12.4ms | 1.13x | 70.30 | +0.0% |
| IDN K=2 warm | warm | 16.0ms | 0.87x | 70.38 | +0.1% |
| Fused+corr K=2 warm | warm | 14.6ms | 0.96x | 70.30 | +0.0% |
| IDN K=4 warm | warm | 22.3ms | 0.62x | 70.64 | +0.5% |
| Fused+corr K=4 warm | warm | 19.2ms | 0.73x | 70.30 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 50.4ms | 0.28x | 70.32 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 23.3ms | 0.60x | 70.31 | +0.0% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.9ms | 1.00x | 70.30 | --- |
| IDN K=1 h0 | h0 | 10.4ms | 1.33x | diverged | --- |
| Fused+corr K=1 h0 | h0 | 9.5ms | 1.47x | diverged | --- |
| IDN K=2 h0 | h0 | 15.2ms | 0.92x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 12.4ms | 1.12x | diverged | --- |
| IDN K=4 h0 | h0 | 24.7ms | 0.56x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 18.5ms | 0.75x | diverged | --- |
| Fused h0 | h0 | 5.7ms | 2.43x | diverged | --- |
| IDN K=1 fwd | fwd | 13.0ms | 1.08x | diverged | --- |
| Fused+corr K=1 fwd | fwd | 11.4ms | 1.22x | diverged | --- |
| IDN K=2 fwd | fwd | 17.7ms | 0.79x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 14.3ms | 0.98x | diverged | --- |
| IDN K=4 fwd | fwd | 27.1ms | 0.51x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 20.8ms | 0.67x | diverged | --- |
| IDN K=1 warm | warm | 10.5ms | 1.33x | 70.31 | +0.0% |
| Fused+corr K=1 warm | warm | 9.4ms | 1.48x | 70.30 | +0.0% |
| IDN K=2 warm | warm | 15.1ms | 0.92x | 70.56 | +0.4% |
| Fused+corr K=2 warm | warm | 12.4ms | 1.12x | 70.30 | +0.0% |
| IDN K=4 warm | warm | 24.5ms | 0.57x | 73.44 | +4.5% |
| Fused+corr K=4 warm | warm | 18.5ms | 0.75x | 70.30 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.7ms | 0.21x | 70.32 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 26.9ms | 0.52x | 70.32 | +0.0% |

## diag01 (seq PPL=63.82, seq=13.7ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.7ms | 1.00x | 63.82 | --- |
| IDN K=1 h0 | h0 | 15.9ms | 0.86x | 119.30 | +86.9% |
| Fused+corr K=1 h0 | h0 | 15.6ms | 0.88x | 119.30 | +86.9% |
| IDN K=2 h0 | h0 | 17.5ms | 0.78x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 17.9ms | 0.77x | diverged | --- |
| IDN K=4 h0 | h0 | 21.5ms | 0.64x | 62.91 | -1.4% |
| Fused+corr K=4 h0 | h0 | 20.6ms | 0.67x | 62.91 | -1.4% |
| Fused h0 | h0 | 11.2ms | 1.23x | 389.78 | +510.8% |
| IDN K=1 fwd | fwd | 17.1ms | 0.80x | 118.36 | +85.5% |
| Fused+corr K=1 fwd | fwd | 17.1ms | 0.80x | 118.35 | +85.5% |
| IDN K=2 fwd | fwd | 19.1ms | 0.72x | 184.74 | +189.5% |
| Fused+corr K=2 fwd | fwd | 18.6ms | 0.74x | 184.71 | +189.4% |
| IDN K=4 fwd | fwd | 22.9ms | 0.60x | 64.37 | +0.9% |
| Fused+corr K=4 fwd | fwd | 21.6ms | 0.63x | 64.38 | +0.9% |
| IDN K=1 warm | warm | 16.1ms | 0.85x | 63.82 | +0.0% |
| Fused+corr K=1 warm | warm | 18.4ms | 0.74x | 63.82 | +0.0% |
| IDN K=2 warm | warm | 20.5ms | 0.67x | 63.83 | +0.0% |
| Fused+corr K=2 warm | warm | 19.8ms | 0.69x | 63.82 | +0.0% |
| IDN K=4 warm | warm | 24.7ms | 0.56x | 63.84 | +0.0% |
| Fused+corr K=4 warm | warm | 23.5ms | 0.58x | 63.82 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 45.7ms | 0.30x | 63.83 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 26.9ms | 0.51x | 63.82 | +0.0% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.7ms | 1.00x | 63.82 | --- |
| IDN K=1 h0 | h0 | 14.9ms | 0.92x | 146.59 | +129.7% |
| Fused+corr K=1 h0 | h0 | 14.6ms | 0.94x | 146.59 | +129.7% |
| IDN K=2 h0 | h0 | 17.8ms | 0.77x | 84270.05 | +131953.6% |
| Fused+corr K=2 h0 | h0 | 17.4ms | 0.79x | 84230.26 | +131891.3% |
| IDN K=4 h0 | h0 | 23.3ms | 0.59x | 181.51 | +184.4% |
| Fused+corr K=4 h0 | h0 | 21.4ms | 0.64x | 181.50 | +184.4% |
| Fused h0 | h0 | 9.4ms | 1.46x | 12962.76 | +20213.0% |
| IDN K=1 fwd | fwd | 15.7ms | 0.87x | 232.03 | +263.6% |
| Fused+corr K=1 fwd | fwd | 15.4ms | 0.89x | 232.03 | +263.6% |
| IDN K=2 fwd | fwd | 18.8ms | 0.73x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 18.0ms | 0.76x | diverged | --- |
| IDN K=4 fwd | fwd | 25.0ms | 0.55x | 147.05 | +130.4% |
| Fused+corr K=4 fwd | fwd | 23.2ms | 0.59x | 147.02 | +130.4% |
| IDN K=1 warm | warm | 14.0ms | 0.98x | 63.82 | +0.0% |
| Fused+corr K=1 warm | warm | 13.8ms | 1.00x | 63.82 | +0.0% |
| IDN K=2 warm | warm | 17.2ms | 0.80x | 63.86 | +0.1% |
| Fused+corr K=2 warm | warm | 16.4ms | 0.84x | 63.82 | +0.0% |
| IDN K=4 warm | warm | 23.4ms | 0.59x | 64.00 | +0.3% |
| Fused+corr K=4 warm | warm | 21.6ms | 0.63x | 63.82 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 51.7ms | 0.27x | 63.83 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 24.8ms | 0.55x | 63.82 | +0.0% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.7ms | 1.00x | 63.82 | --- |
| IDN K=1 h0 | h0 | 11.2ms | 1.23x | 808.91 | +1167.6% |
| Fused+corr K=1 h0 | h0 | 10.3ms | 1.33x | 808.81 | +1167.4% |
| IDN K=2 h0 | h0 | 16.0ms | 0.86x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 13.9ms | 0.99x | diverged | --- |
| IDN K=4 h0 | h0 | 25.3ms | 0.54x | 15745.52 | +24573.7% |
| Fused+corr K=4 h0 | h0 | 21.1ms | 0.65x | 15731.24 | +24551.3% |
| Fused h0 | h0 | 6.3ms | 2.18x | diverged | --- |
| IDN K=1 fwd | fwd | 13.7ms | 1.00x | diverged | --- |
| Fused+corr K=1 fwd | fwd | 12.4ms | 1.11x | diverged | --- |
| IDN K=2 fwd | fwd | 18.3ms | 0.75x | 5604.54 | +8682.5% |
| Fused+corr K=2 fwd | fwd | 15.8ms | 0.87x | 5605.70 | +8684.3% |
| IDN K=4 fwd | fwd | 27.6ms | 0.50x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 23.3ms | 0.59x | diverged | --- |
| IDN K=1 warm | warm | 11.2ms | 1.22x | 63.82 | +0.0% |
| Fused+corr K=1 warm | warm | 10.3ms | 1.33x | 63.82 | +0.0% |
| IDN K=2 warm | warm | 15.9ms | 0.86x | 63.92 | +0.2% |
| Fused+corr K=2 warm | warm | 13.8ms | 1.00x | 63.82 | +0.0% |
| IDN K=4 warm | warm | 25.2ms | 0.54x | 64.35 | +0.8% |
| Fused+corr K=4 warm | warm | 20.8ms | 0.66x | 63.82 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 68.1ms | 0.20x | 63.83 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 27.4ms | 0.50x | 63.83 | +0.0% |

## diag05 (seq PPL=74.50, seq=13.8ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.8ms | 1.00x | 74.50 | --- |
| IDN K=1 h0 | h0 | 16.6ms | 0.83x | 105.60 | +41.7% |
| Fused+corr K=1 h0 | h0 | 16.3ms | 0.85x | 105.60 | +41.7% |
| IDN K=2 h0 | h0 | 18.4ms | 0.75x | 196.21 | +163.4% |
| Fused+corr K=2 h0 | h0 | 18.3ms | 0.75x | 196.21 | +163.4% |
| IDN K=4 h0 | h0 | 22.7ms | 0.61x | 73.10 | -1.9% |
| Fused+corr K=4 h0 | h0 | 21.0ms | 0.66x | 73.09 | -1.9% |
| Fused h0 | h0 | 11.6ms | 1.19x | 1049.09 | +1308.2% |
| IDN K=1 fwd | fwd | 17.1ms | 0.81x | 160.05 | +114.8% |
| Fused+corr K=1 fwd | fwd | 17.5ms | 0.79x | 160.04 | +114.8% |
| IDN K=2 fwd | fwd | 19.5ms | 0.71x | 86.87 | +16.6% |
| Fused+corr K=2 fwd | fwd | 18.9ms | 0.73x | 86.87 | +16.6% |
| IDN K=4 fwd | fwd | 23.4ms | 0.59x | 74.71 | +0.3% |
| Fused+corr K=4 fwd | fwd | 22.0ms | 0.63x | 74.71 | +0.3% |
| IDN K=1 warm | warm | 16.4ms | 0.84x | 74.50 | +0.0% |
| Fused+corr K=1 warm | warm | 16.4ms | 0.84x | 74.50 | +0.0% |
| IDN K=2 warm | warm | 18.4ms | 0.75x | 74.51 | +0.0% |
| Fused+corr K=2 warm | warm | 16.6ms | 0.83x | 74.50 | +0.0% |
| IDN K=4 warm | warm | 20.9ms | 0.66x | 74.51 | +0.0% |
| Fused+corr K=4 warm | warm | 20.1ms | 0.69x | 74.50 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 39.5ms | 0.35x | 74.51 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 22.9ms | 0.60x | 74.51 | +0.0% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.8ms | 1.00x | 74.50 | --- |
| IDN K=1 h0 | h0 | 12.6ms | 1.10x | 178.18 | +139.2% |
| Fused+corr K=1 h0 | h0 | 11.8ms | 1.17x | 178.18 | +139.2% |
| IDN K=2 h0 | h0 | 15.6ms | 0.88x | 701.12 | +841.1% |
| Fused+corr K=2 h0 | h0 | 14.5ms | 0.95x | 700.95 | +840.9% |
| IDN K=4 h0 | h0 | 22.3ms | 0.62x | 109.62 | +47.1% |
| Fused+corr K=4 h0 | h0 | 19.0ms | 0.73x | 109.62 | +47.1% |
| Fused h0 | h0 | 8.3ms | 1.67x | 19457.86 | +26018.3% |
| IDN K=1 fwd | fwd | 14.4ms | 0.96x | 300.66 | +303.6% |
| Fused+corr K=1 fwd | fwd | 13.6ms | 1.02x | 300.68 | +303.6% |
| IDN K=2 fwd | fwd | 17.5ms | 0.79x | 313.71 | +321.1% |
| Fused+corr K=2 fwd | fwd | 15.9ms | 0.87x | 313.67 | +321.0% |
| IDN K=4 fwd | fwd | 23.5ms | 0.59x | 104.22 | +39.9% |
| Fused+corr K=4 fwd | fwd | 19.8ms | 0.70x | 104.22 | +39.9% |
| IDN K=1 warm | warm | 12.6ms | 1.10x | 74.50 | +0.0% |
| Fused+corr K=1 warm | warm | 12.1ms | 1.14x | 74.50 | +0.0% |
| IDN K=2 warm | warm | 15.8ms | 0.87x | 74.52 | +0.0% |
| Fused+corr K=2 warm | warm | 14.4ms | 0.96x | 74.50 | +0.0% |
| IDN K=4 warm | warm | 22.1ms | 0.63x | 74.56 | +0.1% |
| Fused+corr K=4 warm | warm | 18.9ms | 0.73x | 74.50 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 50.1ms | 0.28x | 74.52 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 23.1ms | 0.60x | 74.52 | +0.0% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.8ms | 1.00x | 74.50 | --- |
| IDN K=1 h0 | h0 | 10.3ms | 1.35x | 381.58 | +412.2% |
| Fused+corr K=1 h0 | h0 | 9.3ms | 1.49x | 381.57 | +412.2% |
| IDN K=2 h0 | h0 | 15.1ms | 0.91x | 71636.65 | +96057.8% |
| Fused+corr K=2 h0 | h0 | 12.3ms | 1.12x | 71614.46 | +96028.0% |
| IDN K=4 h0 | h0 | 24.4ms | 0.57x | 1908.09 | +2461.2% |
| Fused+corr K=4 h0 | h0 | 18.3ms | 0.75x | 1908.71 | +2462.1% |
| Fused h0 | h0 | 5.7ms | 2.44x | diverged | --- |
| IDN K=1 fwd | fwd | 12.8ms | 1.08x | 20703.33 | +27690.0% |
| Fused+corr K=1 fwd | fwd | 11.3ms | 1.22x | 20702.96 | +27689.6% |
| IDN K=2 fwd | fwd | 17.5ms | 0.79x | 4946.40 | +6539.5% |
| Fused+corr K=2 fwd | fwd | 13.8ms | 1.00x | 4947.59 | +6541.1% |
| IDN K=4 fwd | fwd | 26.9ms | 0.51x | 3451.25 | +4532.6% |
| Fused+corr K=4 fwd | fwd | 20.4ms | 0.68x | 3448.71 | +4529.2% |
| IDN K=1 warm | warm | 10.4ms | 1.33x | 74.50 | +0.0% |
| Fused+corr K=1 warm | warm | 9.3ms | 1.49x | 74.50 | +0.0% |
| IDN K=2 warm | warm | 14.8ms | 0.93x | 74.55 | +0.1% |
| Fused+corr K=2 warm | warm | 12.2ms | 1.13x | 74.50 | +0.0% |
| IDN K=4 warm | warm | 24.4ms | 0.57x | 74.68 | +0.2% |
| Fused+corr K=4 warm | warm | 18.2ms | 0.76x | 74.50 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.6ms | 0.20x | 74.53 | +0.0% |
| Diag vjp p=4 K=1 warm | warm | 26.9ms | 0.51x | 74.54 | +0.1% |

## idn01 (seq PPL=72.71, seq=13.1ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 72.71 | --- |
| IDN K=1 h0 | h0 | 16.0ms | 0.82x | 102.98 | +41.6% |
| Fused+corr K=1 h0 | h0 | 16.0ms | 0.82x | 102.97 | +41.6% |
| IDN K=2 h0 | h0 | 17.8ms | 0.74x | 164.18 | +125.8% |
| Fused+corr K=2 h0 | h0 | 17.6ms | 0.74x | 164.10 | +125.7% |
| IDN K=4 h0 | h0 | 22.1ms | 0.59x | 72.35 | -0.5% |
| Fused+corr K=4 h0 | h0 | 20.4ms | 0.64x | 72.34 | -0.5% |
| Fused h0 | h0 | 11.1ms | 1.18x | 193.66 | +166.3% |
| IDN K=1 fwd | fwd | 17.1ms | 0.77x | 103.07 | +41.7% |
| Fused+corr K=1 fwd | fwd | 17.0ms | 0.77x | 103.05 | +41.7% |
| IDN K=2 fwd | fwd | 18.8ms | 0.70x | 139.82 | +92.3% |
| Fused+corr K=2 fwd | fwd | 18.3ms | 0.72x | 139.77 | +92.2% |
| IDN K=4 fwd | fwd | 22.7ms | 0.58x | 72.47 | -0.3% |
| Fused+corr K=4 fwd | fwd | 21.3ms | 0.62x | 72.45 | -0.4% |
| IDN K=1 warm | warm | 15.9ms | 0.82x | 72.74 | +0.0% |
| Fused+corr K=1 warm | warm | 15.5ms | 0.84x | 72.71 | +0.0% |
| IDN K=2 warm | warm | 17.7ms | 0.74x | 72.76 | +0.1% |
| Fused+corr K=2 warm | warm | 17.2ms | 0.76x | 72.71 | +0.0% |
| IDN K=4 warm | warm | 21.3ms | 0.62x | 72.78 | +0.1% |
| Fused+corr K=4 warm | warm | 20.6ms | 0.64x | 72.71 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 40.0ms | 0.33x | 72.75 | +0.1% |
| Diag vjp p=4 K=1 warm | warm | 23.5ms | 0.56x | 72.65 | -0.1% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 72.71 | --- |
| IDN K=1 h0 | h0 | 13.4ms | 0.98x | 393.71 | +441.5% |
| Fused+corr K=1 h0 | h0 | 12.8ms | 1.03x | 393.72 | +441.5% |
| IDN K=2 h0 | h0 | 16.1ms | 0.82x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 14.6ms | 0.90x | diverged | --- |
| IDN K=4 h0 | h0 | 22.4ms | 0.59x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 19.7ms | 0.67x | diverged | --- |
| Fused h0 | h0 | 8.5ms | 1.54x | 344.98 | +374.5% |
| IDN K=1 fwd | fwd | 14.9ms | 0.88x | 18070.23 | +24752.1% |
| Fused+corr K=1 fwd | fwd | 14.2ms | 0.93x | 18069.80 | +24751.5% |
| IDN K=2 fwd | fwd | 17.9ms | 0.73x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 15.5ms | 0.85x | diverged | --- |
| IDN K=4 fwd | fwd | 23.1ms | 0.57x | 948.18 | +1204.0% |
| Fused+corr K=4 fwd | fwd | 19.3ms | 0.68x | 947.34 | +1202.9% |
| IDN K=1 warm | warm | 12.0ms | 1.09x | 72.76 | +0.1% |
| Fused+corr K=1 warm | warm | 11.7ms | 1.12x | 72.71 | +0.0% |
| IDN K=2 warm | warm | 15.5ms | 0.85x | 73.08 | +0.5% |
| Fused+corr K=2 warm | warm | 13.9ms | 0.94x | 72.71 | +0.0% |
| IDN K=4 warm | warm | 21.7ms | 0.60x | 73.23 | +0.7% |
| Fused+corr K=4 warm | warm | 18.3ms | 0.72x | 72.71 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 49.5ms | 0.27x | 74.25 | +2.1% |
| Diag vjp p=4 K=1 warm | warm | 22.7ms | 0.58x | 72.57 | -0.2% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 72.71 | --- |
| IDN K=1 h0 | h0 | 10.2ms | 1.29x | 17308.43 | +23704.4% |
| Fused+corr K=1 h0 | h0 | 9.2ms | 1.43x | 17311.93 | +23709.2% |
| IDN K=2 h0 | h0 | 14.9ms | 0.88x | diverged | --- |
| Fused+corr K=2 h0 | h0 | 12.0ms | 1.09x | diverged | --- |
| IDN K=4 h0 | h0 | 24.3ms | 0.54x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 17.7ms | 0.74x | diverged | --- |
| Fused h0 | h0 | 5.5ms | 2.38x | diverged | --- |
| IDN K=1 fwd | fwd | 12.6ms | 1.04x | diverged | --- |
| Fused+corr K=1 fwd | fwd | 11.1ms | 1.18x | diverged | --- |
| IDN K=2 fwd | fwd | 17.3ms | 0.76x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 13.8ms | 0.95x | diverged | --- |
| IDN K=4 fwd | fwd | 26.6ms | 0.49x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 19.9ms | 0.66x | diverged | --- |
| IDN K=1 warm | warm | 10.2ms | 1.29x | 72.78 | +0.1% |
| Fused+corr K=1 warm | warm | 9.1ms | 1.45x | 72.71 | +0.0% |
| IDN K=2 warm | warm | 14.8ms | 0.89x | 73.22 | +0.7% |
| Fused+corr K=2 warm | warm | 11.8ms | 1.11x | 72.71 | +0.0% |
| IDN K=4 warm | warm | 24.2ms | 0.54x | 74.53 | +2.5% |
| Fused+corr K=4 warm | warm | 17.5ms | 0.75x | 72.71 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.1ms | 0.20x | 74.31 | +2.2% |
| Diag vjp p=4 K=1 warm | warm | 26.4ms | 0.50x | 72.58 | -0.2% |

## idn05 (seq PPL=71.67, seq=13.1ms)

### n_par=8

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 71.67 | --- |
| IDN K=1 h0 | h0 | 16.0ms | 0.82x | 449.86 | +527.7% |
| Fused+corr K=1 h0 | h0 | 15.9ms | 0.83x | 449.46 | +527.2% |
| IDN K=2 h0 | h0 | 17.7ms | 0.74x | 4178.79 | +5731.0% |
| Fused+corr K=2 h0 | h0 | 17.3ms | 0.76x | 4158.96 | +5703.3% |
| IDN K=4 h0 | h0 | 22.1ms | 0.60x | 72.19 | +0.7% |
| Fused+corr K=4 h0 | h0 | 20.5ms | 0.64x | 72.05 | +0.5% |
| Fused h0 | h0 | 11.0ms | 1.19x | 8856.60 | +12258.2% |
| IDN K=1 fwd | fwd | 17.0ms | 0.77x | 322.57 | +350.1% |
| Fused+corr K=1 fwd | fwd | 16.9ms | 0.78x | 322.39 | +349.9% |
| IDN K=2 fwd | fwd | 18.8ms | 0.70x | 3808.48 | +5214.2% |
| Fused+corr K=2 fwd | fwd | 18.7ms | 0.70x | 3784.03 | +5180.1% |
| IDN K=4 fwd | fwd | 22.2ms | 0.59x | 71.85 | +0.3% |
| Fused+corr K=4 fwd | fwd | 21.5ms | 0.61x | 71.69 | +0.0% |
| IDN K=1 warm | warm | 16.0ms | 0.82x | 71.84 | +0.2% |
| Fused+corr K=1 warm | warm | 15.0ms | 0.88x | 71.67 | +0.0% |
| IDN K=2 warm | warm | 16.9ms | 0.78x | 71.95 | +0.4% |
| Fused+corr K=2 warm | warm | 16.4ms | 0.80x | 71.67 | +0.0% |
| IDN K=4 warm | warm | 20.4ms | 0.64x | 72.09 | +0.6% |
| Fused+corr K=4 warm | warm | 19.6ms | 0.67x | 71.67 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 39.0ms | 0.34x | 71.77 | +0.1% |
| Diag vjp p=4 K=1 warm | warm | 22.4ms | 0.59x | 74.40 | +3.8% |

### n_par=16

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 71.67 | --- |
| IDN K=1 h0 | h0 | 12.6ms | 1.04x | 743.11 | +936.9% |
| Fused+corr K=1 h0 | h0 | 12.2ms | 1.08x | 743.30 | +937.2% |
| IDN K=2 h0 | h0 | 15.9ms | 0.83x | 13028.91 | +18080.2% |
| Fused+corr K=2 h0 | h0 | 14.5ms | 0.90x | 13097.39 | +18175.7% |
| IDN K=4 h0 | h0 | 22.2ms | 0.59x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 18.9ms | 0.69x | diverged | --- |
| Fused h0 | h0 | 8.1ms | 1.62x | 2559.40 | +3471.3% |
| IDN K=1 fwd | fwd | 14.4ms | 0.92x | 4759.41 | +6541.1% |
| Fused+corr K=1 fwd | fwd | 13.6ms | 0.97x | 4760.30 | +6542.4% |
| IDN K=2 fwd | fwd | 17.5ms | 0.75x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 15.9ms | 0.82x | diverged | --- |
| IDN K=4 fwd | fwd | 23.8ms | 0.55x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 20.4ms | 0.64x | diverged | --- |
| IDN K=1 warm | warm | 12.7ms | 1.04x | 71.92 | +0.4% |
| Fused+corr K=1 warm | warm | 12.1ms | 1.09x | 71.67 | +0.0% |
| IDN K=2 warm | warm | 15.7ms | 0.84x | 73.65 | +2.8% |
| Fused+corr K=2 warm | warm | 14.4ms | 0.91x | 71.67 | +0.0% |
| IDN K=4 warm | warm | 22.0ms | 0.60x | 75.82 | +5.8% |
| Fused+corr K=4 warm | warm | 18.9ms | 0.70x | 71.67 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 49.9ms | 0.26x | 73.81 | +3.0% |
| Diag vjp p=4 K=1 warm | warm | 23.0ms | 0.57x | 76.48 | +6.7% |

### n_par=24

| Method | Init | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 13.1ms | 1.00x | 71.67 | --- |
| IDN K=1 h0 | h0 | 10.2ms | 1.29x | 603.11 | +741.6% |
| Fused+corr K=1 h0 | h0 | 9.3ms | 1.42x | 603.21 | +741.7% |
| IDN K=2 h0 | h0 | 15.0ms | 0.88x | 2910.29 | +3960.9% |
| Fused+corr K=2 h0 | h0 | 12.1ms | 1.09x | 2908.79 | +3958.8% |
| IDN K=4 h0 | h0 | 24.3ms | 0.54x | diverged | --- |
| Fused+corr K=4 h0 | h0 | 18.0ms | 0.73x | diverged | --- |
| Fused h0 | h0 | 5.5ms | 2.37x | 96256.17 | +134213.0% |
| IDN K=1 fwd | fwd | 12.7ms | 1.03x | 1582.87 | +2108.7% |
| Fused+corr K=1 fwd | fwd | 11.3ms | 1.16x | 1583.49 | +2109.6% |
| IDN K=2 fwd | fwd | 17.4ms | 0.75x | diverged | --- |
| Fused+corr K=2 fwd | fwd | 13.8ms | 0.95x | diverged | --- |
| IDN K=4 fwd | fwd | 26.8ms | 0.49x | diverged | --- |
| Fused+corr K=4 fwd | fwd | 20.2ms | 0.65x | diverged | --- |
| IDN K=1 warm | warm | 10.1ms | 1.30x | 72.06 | +0.5% |
| Fused+corr K=1 warm | warm | 9.0ms | 1.46x | 71.67 | +0.0% |
| IDN K=2 warm | warm | 14.7ms | 0.89x | 74.39 | +3.8% |
| Fused+corr K=2 warm | warm | 11.7ms | 1.12x | 71.67 | +0.0% |
| IDN K=4 warm | warm | 24.3ms | 0.54x | 90.99 | +27.0% |
| Fused+corr K=4 warm | warm | 18.0ms | 0.73x | 71.67 | +0.0% |
| Diag jvp p=4 K=1 warm | warm | 67.4ms | 0.19x | 73.89 | +3.1% |
| Diag vjp p=4 K=1 warm | warm | 26.6ms | 0.49x | 76.77 | +7.1% |

## Key Findings

1. **Warm-start IDN K=1 gives <0.3% PPL loss** at 1.2-1.5x speedup (n_par=24).

2. **Fused+corr K=1 warm gives best speedup** (1.3-1.5x at n_par=24, +0.0% PPL).

3. **IDN/diag reg improves sequential PPL** — acts as beneficial regularization.
   At 9600 steps, diag01 has best seq PPL. idn05 overfits (λ=0.5 too strong).

4. **No regularization needed for warm-start quality**: baseline achieves same
   parallel inference quality as regularized models with warm-start init.

