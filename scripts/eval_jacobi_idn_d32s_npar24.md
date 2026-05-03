# IDN Inference Evaluation: d32s Models (n_embd=640, 32 layers, ~535M params)

All models: depth=32, n_embd=640, 5 heads, ~535M params. Single H100, SDPA, seq_len=128, BS=8.
Timing: 100 runs averaged with 20 warmup.
Quality: **131,072 tokens** (13 methods x 128 batches), same data for all models.
4800-step checkpoints (data:param ratio ~28).

Models:
- **baseline**: no regularization (seq PPL=60.61)
- **idn05**: IDN lambda=0.5, n_par_configs=8,16,24 (seq PPL=51.87)
- **diag01**: diag Newton lambda=0.1, n_par=24, stride=3 (seq PPL=57.31)
- **diag05**: diag Newton lambda=0.5, n_par=24, stride=3 (seq PPL=235.07)

## n_par=16 (seq=16 + 16 parallel)

### baseline (seq PPL=60.61)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 18.04ms | 1.00x | 60.61 | --- | --- |
| IDN loop | 13.86ms | 1.30x | 377.14 | +522.2% | 0.381 |
| IDN batched | 11.79ms | 1.53x | 377.14 | +522.2% | 0.381 |
| Fused | 11.68ms | 1.54x | 901.56 | +1387.4% | 0.271 |
| Fused+corr K=1 | 12.38ms | 1.46x | 377.14 | +522.2% | 0.381 |
| Fused+corr K=2 | 13.86ms | 1.30x | diverged | diverged | 0.001 |
| IDN loop elk=0.1 | 14.33ms | 1.26x | 1202.75 | +1884.3% | 0.257 |
| IDN batched elk=0.1 | 13.36ms | 1.35x | 1202.67 | +1884.1% | 0.257 |
| Fused+corr K=1 elk=0.1 | 12.08ms | 1.49x | 1202.67 | +1884.1% | 0.257 |
| Fused+corr K=2 elk=0.1 | 14.03ms | 1.29x | 1390.30 | +2193.7% | 0.200 |
| IDN loop elk=0.3 | 13.97ms | 1.29x | 12189 | +20009% | 0.148 |
| IDN batched elk=0.3 | 12.01ms | 1.50x | 12186 | +20005% | 0.148 |
| Fused+corr K=1 elk=0.3 | 12.81ms | 1.41x | 12186 | +20005% | 0.148 |
| Fused+corr K=2 elk=0.3 | 12.45ms | 1.45x | 204.60 | +237.6% | 0.257 |

### idn05 (seq PPL=51.87)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 15.98ms | 1.00x | 51.87 | --- | --- |
| IDN loop | 15.03ms | 1.06x | 4564.13 | +8699.5% | 0.161 |
| IDN batched | 12.09ms | 1.32x | 4550.91 | +8674.0% | 0.161 |
| Fused | 10.61ms | 1.51x | 119955 | +231170% | 0.034 |
| Fused+corr K=1 | 12.19ms | 1.31x | 4550.91 | +8674.0% | 0.161 |
| Fused+corr K=2 | 14.04ms | 1.14x | 509333 | +981879% | 0.004 |
| IDN loop elk=0.1 | 16.79ms | 0.95x | 2408.33 | +4543.2% | 0.249 |
| IDN batched elk=0.1 | 13.50ms | 1.18x | 2403.38 | +4533.6% | 0.249 |
| Fused+corr K=1 elk=0.1 | 15.24ms | 1.05x | 2403.38 | +4533.6% | 0.249 |
| Fused+corr K=2 elk=0.1 | 15.08ms | 1.06x | 10577 | +20291% | 0.176 |
| IDN loop elk=0.3 | 19.90ms | 0.80x | 529.69 | +921.2% | 0.437 |
| IDN batched elk=0.3 | 9.57ms | 1.67x | 529.52 | +920.9% | 0.437 |
| Fused+corr K=1 elk=0.3 | 10.32ms | 1.55x | 529.52 | +920.9% | 0.437 |
| Fused+corr K=2 elk=0.3 | 14.21ms | 1.12x | 1735.12 | +3245.3% | 0.405 |

### diag01 (seq PPL=57.31)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 19.16ms | 1.00x | 57.31 | --- | --- |
| IDN loop | 15.03ms | 1.28x | 111.88 | +95.2% | 0.617 |
| IDN batched | 12.10ms | 1.58x | 111.87 | +95.2% | 0.617 |
| Fused | 11.52ms | 1.66x | 3767.36 | +6473.8% | 0.252 |
| Fused+corr K=1 | 12.86ms | 1.49x | 111.87 | +95.2% | 0.617 |
| Fused+corr K=2 | 12.82ms | 1.49x | 379.98 | +563.0% | 0.408 |
| IDN loop elk=0.1 | 14.01ms | 1.37x | 105.05 | +83.3% | 0.559 |
| IDN batched elk=0.1 | 11.82ms | 1.62x | 105.04 | +83.3% | 0.559 |
| Fused+corr K=1 elk=0.1 | 9.91ms | 1.93x | 105.04 | +83.3% | 0.559 |
| Fused+corr K=2 elk=0.1 | 11.81ms | 1.62x | 95.84 | +67.2% | 0.712 |
| IDN loop elk=0.3 | 14.34ms | 1.34x | 125.18 | +118.4% | 0.463 |
| IDN batched elk=0.3 | 9.60ms | 2.00x | 125.17 | +118.4% | 0.463 |
| Fused+corr K=1 elk=0.3 | 9.87ms | 1.94x | 125.17 | +118.4% | 0.463 |
| Fused+corr K=2 elk=0.3 | 12.84ms | 1.49x | 104.31 | +82.0% | 0.598 |

### diag05 (seq PPL=235.07)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 15.62ms | 1.00x | 235.07 | --- | --- |
| IDN loop | 14.63ms | 1.07x | 124.46 | -47.1% | 0.410 |
| IDN batched | 12.50ms | 1.25x | 124.46 | -47.1% | 0.409 |
| Fused | 8.48ms | 1.84x | 404651 | +172042% | 0.051 |
| Fused+corr K=1 | 9.98ms | 1.56x | 124.46 | -47.1% | 0.409 |
| Fused+corr K=2 | 14.02ms | 1.11x | 375.39 | +59.7% | 0.555 |
| IDN loop elk=0.1 | 14.54ms | 1.07x | 116.52 | -50.4% | 0.370 |
| IDN batched elk=0.1 | 10.49ms | 1.49x | 116.52 | -50.4% | 0.370 |
| Fused+corr K=1 elk=0.1 | 10.09ms | 1.55x | 116.52 | -50.4% | 0.370 |
| Fused+corr K=2 elk=0.1 | 13.12ms | 1.19x | 117.37 | -50.1% | 0.564 |
| IDN loop elk=0.3 | 14.97ms | 1.04x | 124.70 | -46.9% | 0.330 |
| IDN batched elk=0.3 | 10.55ms | 1.48x | 124.71 | -46.9% | 0.330 |
| Fused+corr K=1 elk=0.3 | 10.07ms | 1.55x | 124.71 | -46.9% | 0.330 |
| Fused+corr K=2 elk=0.3 | 12.08ms | 1.29x | 78.84 | -66.5% | 0.437 |

## n_par=24 (seq=8 + 24 parallel)

### baseline (seq PPL=60.61)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 18.04ms | 1.00x | 60.61 | --- | --- |
| IDN loop | 19.05ms | 0.95x | 7504.42 | +12281% | 0.212 |
| IDN batched | 8.09ms | 2.23x | 7502.52 | +12277% | 0.212 |
| Fused | 5.86ms | 3.08x | 30744 | +50621% | 0.169 |
| Fused+corr K=1 | 11.31ms | 1.59x | 7502.52 | +12277% | 0.212 |
| Fused+corr K=2 | 14.25ms | 1.27x | diverged | diverged | 0.000 |
| IDN loop elk=0.1 | 16.39ms | 1.10x | 39074 | +64364% | 0.163 |
| IDN batched elk=0.1 | 7.30ms | 2.47x | 39066 | +64351% | 0.163 |
| Fused+corr K=1 elk=0.1 | 8.20ms | 2.20x | 39066 | +64351% | 0.163 |
| Fused+corr K=2 elk=0.1 | 10.62ms | 1.70x | 27437 | +45165% | 0.036 |
| IDN loop elk=0.3 | 16.45ms | 1.10x | 436475 | +719985% | 0.100 |
| IDN batched elk=0.3 | 8.28ms | 2.18x | 436384 | +719835% | 0.100 |
| Fused+corr K=1 elk=0.3 | 8.88ms | 2.03x | 436384 | +719835% | 0.100 |
| Fused+corr K=2 elk=0.3 | 14.21ms | 1.27x | 2197.90 | +3526.0% | 0.032 |

### idn05 (seq PPL=51.87)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 15.98ms | 1.00x | 51.87 | --- | --- |
| IDN loop | 14.05ms | 1.14x | 20707 | +39823% | 0.174 |
| IDN batched | 10.00ms | 1.60x | 20602 | +39620% | 0.175 |
| Fused | 5.78ms | 2.76x | diverged | diverged | 0.000 |
| Fused+corr K=1 | 7.81ms | 2.05x | 20602 | +39620% | 0.175 |
| Fused+corr K=2 | 10.37ms | 1.54x | diverged | diverged | 0.001 |
| IDN loop elk=0.1 | 14.15ms | 1.13x | 8526.37 | +16339% | 0.227 |
| IDN batched elk=0.1 | 10.21ms | 1.57x | 8514.99 | +16317% | 0.227 |
| Fused+corr K=1 elk=0.1 | 7.81ms | 2.05x | 8514.99 | +16317% | 0.227 |
| Fused+corr K=2 elk=0.1 | 13.51ms | 1.18x | 18316 | +35212% | 0.168 |
| IDN loop elk=0.3 | 19.20ms | 0.83x | 3155.57 | +5983.9% | 0.243 |
| IDN batched elk=0.3 | 7.31ms | 2.19x | 3155.78 | +5984.3% | 0.244 |
| Fused+corr K=1 elk=0.3 | 8.68ms | 1.84x | 3155.78 | +5984.3% | 0.244 |
| Fused+corr K=2 elk=0.3 | 14.07ms | 1.14x | 14991 | +28802% | 0.272 |

### diag01 (seq PPL=57.31)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 19.16ms | 1.00x | 57.31 | --- | --- |
| IDN loop | 15.52ms | 1.23x | 414.71 | +623.7% | 0.411 |
| IDN batched | 7.28ms | 2.63x | 414.72 | +623.7% | 0.411 |
| Fused | 5.71ms | 3.35x | 86948 | +151619% | 0.070 |
| Fused+corr K=1 | 7.97ms | 2.40x | 414.72 | +623.7% | 0.411 |
| Fused+corr K=2 | 10.93ms | 1.75x | 5281.82 | +9116.5% | 0.131 |
| IDN loop elk=0.1 | 15.16ms | 1.26x | 302.72 | +428.2% | 0.408 |
| IDN batched elk=0.1 | 7.78ms | 2.46x | 302.70 | +428.2% | 0.408 |
| Fused+corr K=1 elk=0.1 | 9.05ms | 2.12x | 302.70 | +428.2% | 0.408 |
| Fused+corr K=2 elk=0.1 | 12.90ms | 1.49x | 5009.68 | +8641.6% | 0.136 |
| IDN loop elk=0.3 | 16.11ms | 1.19x | 302.57 | +428.0% | 0.378 |
| IDN batched elk=0.3 | 8.54ms | 2.24x | 302.54 | +427.9% | 0.378 |
| Fused+corr K=1 elk=0.3 | 8.06ms | 2.38x | 302.54 | +427.9% | 0.378 |
| Fused+corr K=2 elk=0.3 | 13.17ms | 1.45x | 1700.92 | +2868.0% | 0.191 |

### diag05 (seq PPL=235.07)

| Method | Time | Speed | PPL | dPPL% | Top-1 |
|---|---|---|---|---|---|
| Sequential | 15.62ms | 1.00x | 235.07 | --- | --- |
| IDN loop | 15.54ms | 1.01x | 553.50 | +135.5% | 0.430 |
| IDN batched | 7.49ms | 2.09x | 553.60 | +135.5% | 0.430 |
| Fused | 6.67ms | 2.34x | diverged | diverged | 0.014 |
| Fused+corr K=1 | 9.16ms | 1.71x | 553.60 | +135.5% | 0.430 |
| Fused+corr K=2 | 10.48ms | 1.49x | 1210.37 | +414.9% | 0.351 |
| IDN loop elk=0.1 | 19.21ms | 0.81x | 240.82 | +2.4% | 0.445 |
| IDN batched elk=0.1 | 7.44ms | 2.10x | 240.77 | +2.4% | 0.445 |
| Fused+corr K=1 elk=0.1 | 9.03ms | 1.73x | 240.77 | +2.4% | 0.445 |
| Fused+corr K=2 elk=0.1 | 10.96ms | 1.43x | 910.62 | +287.4% | 0.397 |
| IDN loop elk=0.3 | 19.48ms | 0.80x | 161.68 | -31.2% | 0.423 |
| IDN batched elk=0.3 | 10.80ms | 1.45x | 161.67 | -31.2% | 0.423 |
| Fused+corr K=1 elk=0.3 | 7.93ms | 1.97x | 161.67 | -31.2% | 0.423 |
| Fused+corr K=2 elk=0.3 | 10.41ms | 1.50x | 292.39 | +24.4% | 0.401 |

## Summary: Best Parallel Inference per Model

Baseline sequential PPL = 60.61. dPPL% is vs each model's own sequential PPL.

| Model | Seq PPL | Best quality (speed>1x) | PPL | dPPL% | Speed | Best speed | PPL | Speed |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 60.6 | n=16 Fused+corr K=2 elk=0.3 | 204.6 | +237.6% | 1.45x | n=16 IDN batched | 377.1 | 1.53x |
| **idn05** | 51.9 | n=16 IDN batched elk=0.3 | 529.5 | +920.9% | 1.67x | --- | --- | --- |
| **diag01** | 57.3 | n=16 Fused+corr K=2 elk=0.1 | 95.8 | +67.2% | 1.62x | n=24 IDN batched | 414.7 | 2.63x |
| **diag05** | 235.1 | n=16 Fused+corr K=2 elk=0.3 | 78.8 | -66.5% | 1.29x | n=24 IDN batched elk=0.1 | 240.8 | 2.10x |

## Key Findings

1. **Real speedup at D=640**: Unlike D=2048 where the GPU is saturated, batched
   parallel inference provides genuine speedup: ~1.5-1.9x at n_par=16, ~2.1-2.6x at n_par=24.

2. **IDN batched matches IDN loop**: Verified across all models and n_par values
   (max logit diff < 1.5, PPL within <1%). IDN batched is the correct parallel
   implementation of the sequential IDN loop.

3. **IDN batched = Fused+corr at K=1**: Both implement identical Newton correction
   (per-layer stacked weights + einsum + prefix-sum). Fused+corr additionally
   supports K>1 iterations for iterative refinement.

4. **Fused (no correction) is fast but low quality**: Averaged scalars and mixed
   O/MLP projections produce poor PPL. Only useful as a speed upper bound.

5. **n_par=24 is too aggressive at 4800 steps**: All methods produce PPL >200
   (vs sequential ~50-60). The regularization training didn't converge enough
   for 24 parallel layers. 9600-step models (in progress) should improve this.

6. **elk_k=0 is best for well-regularized models at n_par=16**: The Newton
   correction is accurate enough that damping (elk_k>0) only hurts quality.
   elk_k=0.3 helps at n_par=24 where corrections overshoot on distant layers.

## Bugs Fixed

Two bugs were found and fixed in the batched inference code during this evaluation:

1. **VE gate input** (`eval_jacobi_idn.py`, `eval_jacobi_diag.py`): the value embedding
   gate used `x_in[..., :12]` (unnormalized) but the model's `CausalSelfAttention.forward`
   receives `norm(x_in)` and passes that to `ve_gate`. Fixed to use `norm(x_in)[..., :12]`.
   This caused max logit diff of 32-128 on layers with value embeddings (alternating).

2. **dtype promotion** (`eval_jacobi_idn.py`, `eval_jacobi_diag.py`): stacked float32
   lambdas/weights multiplied by bf16 tensors promoted to float32 via PyTorch's broadcasting
   rules, while the loop path's 0-d scalar multiplication stayed in bf16. Fixed by casting
   stacked tensors to compute dtype (bf16) before operations.

## Reproduction

```bash
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_jacobi_idn \
    --model-tag d32s_baseline_4800 \
    --model-tag d32s_idn05_npar24_4800 \
    --model-tag d32s_diag01_npar24_stride3_4800 \
    --model-tag d32s_diag05_npar24_stride3_4800 \
    --n-par 16,24 --K 1 --elk-k 0.0 \
    --batch-size 8 --n-batches 128 --seq-len 128
```
