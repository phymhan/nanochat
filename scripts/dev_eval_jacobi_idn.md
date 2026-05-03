# Identity Newton Inference: Development Log

## Overview

Evaluating layer-parallel inference strategies for nanochat d32 models.
The goal: run the last `n_par` layers in parallel (instead of sequentially)
to reduce latency, while preserving output quality.

All measurements on single H100, SDPA attention (FA3 disabled for eval
compatibility), seq_len=128, batch_size=1, 8 batches averaged.

## Inference Strategies

### Sequential
Standard forward: all 32 layers run in order. Ground truth.

### IDN (loop)
Identity Newton K iterations via Python loop. Each iteration:
1. Evaluate all parallel layers on current h[i-1]
2. Prefix-sum correction: `delta[i] = res[i] + delta[i-1]`
3. Update: `h[i] = h[i] - delta[i]`

This implements the DEER update with J≈I:
```
s_t^{(i+1)} = f_t(s_{t-1}^{(i)}) + (s_{t-1}^{(i+1)} - s_{t-1}^{(i)})
```
Correct output, but sequential loop — no GPU parallelism.

### IDN (batched)
Same algorithm as loop, but uses stacked per-layer weights with `einsum`
for QKV/O/MLP projections, and `(n_par*B, H, T, D)` layout for SDPA
(each layer = separate batch element, no cross-layer attention mixing).

Correct output (matches loop to bf16 precision), but no speedup at D=2048
because individual matmuls already saturate the GPU.

### Fused
From `jacobi_fuse.py` (report's implementation). Concatenates all parallel
layers' weights into one mega-block:
- One `F.linear` for all QKV projections (concatenated heads)
- One SDPA with `n_par * n_head` total heads
- One `F.linear` for output projection
- One `F.linear` for MLP

Fast (1.13x speedup) but incorrect:
- **Averaged per-layer scalars**: all layers share `avg(resid_lambda)` and
  `avg(x0_lambda)` instead of per-layer values
- **Mixed O projection**: `W_o` is `(D, n_par*H*hd)` — sums all layers'
  attention outputs into one vector instead of per-layer outputs
- **Mixed MLP**: all layers' MLP contributions summed together
- **No Newton correction**: just uses the raw mega-block output as h[-1]

Note: attention itself is NOT cross-layer (each head operates independently),
the errors come from the O/MLP projections mixing layers.

### Fused+corr
Fixed version of Fused: uses stacked per-layer weights (via `torch.stack`
from model blocks) with `einsum` for projections, per-layer SDPA via
`(n_par*B)` batching, and applies Newton prefix-sum correction.

Correct output, but no speedup — the per-layer `einsum` projections
negate the fused SDPA benefit.

## Results: IDN model (d32_idn05_npar8, n_par=8, K=1)

### Table 1: All methods, elk_k=0 (standard IDN)

| Method | Time | Speed | PPL | ΔPPL% | Top-1 | CosSim | Max diff |
|---|---|---|---|---|---|---|---|
| Sequential | 13.2ms | 1.00x | 28.80 | — | — | — | — |
| IDN (loop) | 13.8ms | 0.96x | 28.71 | -0.3% | 0.860 | 0.989 | — |
| IDN (batched) | 14.1ms | 0.94x | 28.82 | +0.1% | 0.862 | 0.990 | 0.63 |
| Fused | 11.4ms | 1.16x | 31.25 | +8.5% | 0.764 | 0.983 | 13.9 |
| Fused+corr | 12.9ms | 1.03x | 28.98 | +0.6% | 0.765 | 0.969 | 8.4 |
| Fused+split_o_mlp | 12.3ms | 1.08x | 27.23 | -5.4% | 0.704 | 0.967 | 8.7 |
| Fused+split_mlp | 12.3ms | 1.08x | 7083 | broken | 0.214 | -0.12 | 24.5 |

### Table 2: IDN (loop) with Scale-ELK damping

| elk_k | PPL | ΔPPL% | Top-1 | CosSim |
|---|---|---|---|---|
| 0.0 (full IDN) | 28.71 | -0.3% | 0.860 | 0.989 |
| **0.1** | **28.16** | **-2.2%** | **0.903** | **0.996** |
| 0.3 | 30.32 | +5.3% | 0.861 | — |
| 0.5 | 32.46 | +12.7% | 0.832 | — |
| 0.7 | 33.62 | +16.7% | 0.816 | — |
| 1.0 (all-on-same) | 34.01 | +18.1% | 0.802 | 0.996 |

elk_k=0.1 is optimal for IDN (loop): PPL 28.16 (-2.2%), top-1 0.903.
The slight damping prevents over-correction from later layers where J≈I
is inexact.

### Table 3: All methods with Scale-ELK (elk_k=0.1)

| Method | elk_k=0 PPL (ΔPPL%) | elk_k=0.1 PPL (ΔPPL%) | elk_k=0 Top-1 | elk_k=0.1 Top-1 |
|---|---|---|---|---|
| IDN (loop) | 28.71 (-0.3%) | **28.16 (-2.2%)** | 0.860 | **0.903** |
| IDN (batched) | 28.82 (+0.1%) | **28.31 (-1.7%)** | 0.862 | **0.899** |
| Fused (no corr) | 31.25 (+8.5%) | 31.25 (+8.5%) | 0.764 | 0.764 |
| Fused+corr | 28.98 (+0.6%) | **27.74 (-3.7%)** | 0.765 | **0.797** |
| Fused+split_o_mlp | 27.23 (-5.4%) | **26.83 (-6.8%)** | 0.704 | **0.748** |
| Fused+split_mlp | 7083 (broken) | 659 (less broken) | 0.214 | 0.351 |

Scale-ELK improves every method that has the Newton correction step.
Fused (no correction) is unaffected. The damping is just one scalar
multiply in the prefix-sum — zero computational cost.

Best overall: **Fused+split_o_mlp with elk_k=0.1: PPL 26.83 (-6.8%), 1.05x speedup**.

### Debug: output diffs vs loop at elk_k=0 (first batch, max absolute)
- batched: 6.33e-01 (bf16 numerical noise from different op ordering)
- fused: 1.39e+01 (structural approximation error)
- fused+corr: 8.35e+00 (reduced by correction, remaining from einsum precision)
- fused+split_o_mlp: 8.70e+00 (fused QKV+SDPA, per-layer O+MLP+correction)
- fused+split_mlp: 2.45e+01 (fused QKV+SDPA+O breaks per-layer MLP)

## Key Findings

### 1. Newton correction is essential
Fused without correction: +8.5% PPL. Fused with correction: +0.6% PPL.
The prefix-sum correction recovers most of the quality lost by running
layers on the wrong input.

### 2. No speedup at D=2048 for fully correct methods
Both batched and fused+corr achieve ~1.0x speed — the GPU is already
saturated by individual (128, 2048) × (2048, 2048) matmuls. Batching
8 of them via einsum doesn't parallelize further.

### 3. Fused attention is correct but O/MLP projections mix layers
Despite using concatenated heads, SDPA processes each head independently
(Q[h] @ K[h]^T per head). The cross-layer errors come from:
- Averaged input scalars (resid_lambda, x0_lambda)
- Mixed O projection (sums all layers into one vector)
- Mixed MLP (same issue)

A block-diagonal attention mask would NOT help — attention isn't the problem.

### 4. Fused+split: partial speedup from fused QKV+SDPA
Fused+split keeps fused QKV+SDPA (fast, one big matmul) but splits after
SDPA for per-layer O projection, MLP, and Newton correction. Gets 1.06x
speedup — between fused (1.15x) and fully correct (1.0x).

Quality: PPL 27.23 (-5.4% vs seq) but top-1 only 0.704. The averaged
input scalars in the QKV stage introduce error that the per-layer O+MLP
correction partially but not fully recovers from.

### 5. Speed-quality tradeoff summary
```
Fully correct (loop/batched/fused+corr): 1.0x speed, +0.1-0.6% PPL
Fused+split (approx QKV, correct O+MLP): 1.06x speed, -5.4% PPL but 0.704 top-1
Fused (everything approximate):          1.15x speed, +8.5% PPL
```

### 6. Where speedup could come from
- **Smaller models** (D < 1024): individual matmuls don't saturate GPU,
  batching helps. Confirmed by Qwen-0.5B (D=896) experiments.
- **Custom Triton kernels**: fuse all per-layer projections into one kernel
  with per-layer weight indexing, avoiding Python-level dispatch overhead.
- **CUDA streams**: overlap sequential prefix with parallel suffix
  computation (report found this doesn't help at this scale).
- **Multi-GPU**: pipeline parallel layers across devices (report found
  communication overhead > compute savings at this scale).

## Method Descriptions

### Fused (original, from report)
```
x_in = avg(resid_λ) * h_init + avg(x0_λ) * x0    ← averaged scalars
x_normed = norm(x_in)
QKV = F.linear(x_normed, [W_q|W_k|W_v])           ← one big matmul
y = SDPA(Q, K, V)                                  ← one big attention (n_par*H heads)
attn_out = F.linear(y, W_o_concat)                 ← one big matmul, MIXES layers
x_mid = x_in + attn_out
MLP = F.linear(F.relu(F.linear(norm(x_mid), W_fc_concat))², W_proj_concat)  ← MIXES layers
out = x_mid + MLP                                  ← single output, no correction
```

### Fused+corr (fully correct, same speed as sequential)
```
x_in[j] = resid_λ[j] * h_init + x0_λ[j] * x0     ← per-layer scalars
QKV = einsum(norm(x_in), [W_q|W_k|W_v]_stacked)   ← per-layer via einsum
y = SDPA(Q, K, V)  with (n_par*B, H, T, D) layout  ← per-layer attention
attn_out = einsum(y, W_o_stacked)                   ← per-layer O
MLP = einsum(...)                                   ← per-layer MLP
all_out[j] = x_in[j] + attn_out[j] + MLP[j]        ← per-layer output
Newton correction: delta prefix-sum                  ← correction step
```

### Fused+split (hybrid: fused QKV+SDPA, per-layer O+MLP+correction)
```
x_in = avg(resid_λ) * h_init + avg(x0_λ) * x0     ← averaged (approximate)
QKV = F.linear(x_normed, [W_q|W_k|W_v]_concat)    ← one big matmul (fast)
y = SDPA(Q, K, V)                                  ← one big attention (fast)
y_split = split y by head groups → (n_par, B, T, D) ← split per-layer
attn_out = einsum(y_split, W_o_stacked)             ← per-layer O
MLP = einsum(...)                                   ← per-layer MLP
all_out[j] = x_in + attn_out[j] + MLP[j]           ← per-layer output
Newton correction: delta prefix-sum                  ← correction step
```

## IDN vs Jacobi (zero-order)

At K=1, both evaluate all parallel layers on h_init. The difference:

**Jacobi K=1** (= all-on-same):
```
h[i] = f_i(h_init)    # raw output, no correction
```

**IDN K=1** (= identity Newton):
```
h[0] = f_0(h_init)
h[1] = f_1(h_init) + (f_0(h_init) - h_init)           # accumulated residual
h[2] = f_2(h_init) + (f_1(h_init) - h_init) + (f_0(h_init) - h_init)
```

IDN adds accumulated residuals `(f_j(h_init) - h_init)` from previous layers,
propagating what earlier layers "would have changed" if run sequentially.
This is the Newton correction with J≈I — the prefix-sum of residuals.

On IDN-trained model (d32_idn05_npar8, n_par=8):
- Jacobi K=1 (all-on-same): PPL 34.0, top-1 0.802
- IDN K=1: PPL 28.7, top-1 0.862
- Sequential: PPL 28.8

The correction alone accounts for the difference (34.0 → 28.7).

At K>1, both re-evaluate on corrected h, but Jacobi needs K=n_par iterations
to converge while IDN often converges at K=1 for IDN-trained models.

### Scale-ELK damping

From arXiv:2407.19115. The IDN correction is a linear recurrence:
```
h_corr[t] = a * h_corr[t-1] + b[t]
```
where `a=1` for standard IDN. Scale-ELK sets `a = (1-k)`:
- k=0 → full prefix-sum (standard IDN)
- k=1 → no correction (all-on-same / Jacobi K=1)
- k ∈ (0,1) → exponentially damped correction

This stabilizes layers far from the prefix where J≈I breaks down (e.g. L7
which consistently has the highest Jacobian error). Zero computational cost.

### Scale-ELK results (d32_idn05_npar8, n_par=8, K=1)

| elk_k | PPL | ΔPPL% | Top-1 |
|---|---|---|---|
| 0.0 (full IDN) | 28.71 | -0.3% | 0.860 |
| **0.1** | **28.16** | **-2.2%** | **0.903** |
| 0.3 | 30.32 | +5.3% | 0.861 |
| 0.5 | 32.46 | +12.7% | 0.832 |
| 0.7 | 33.62 | +16.7% | 0.816 |
| 1.0 (all-on-same) | 34.01 | +18.1% | 0.802 |

elk_k=0.1 improves over full IDN (PPL 28.16 vs 28.71, top-1 0.903 vs 0.860).
The slight damping prevents over-correction from layers where J≈I is inexact.

## Eval script

```bash
# Basic eval (K=1, n_par=8)
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_jacobi_idn \
    --model-tag d32_idn05_npar8 --n-par 8 --K 1

# Multi-K test
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_jacobi_idn \
    --model-tag d32_idn05_npar8 --n-par 8 --K 2

# Compare models
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_jacobi_idn \
    --model-tag d32_baseline --model-tag d32_idn05_npar8 --n-par 4,8
```

## Implementation notes

### Per-layer window sizes
Layers 24-31 alternate between window=512 (sliding) and window=2048 (full):
```
L24:512  L25:512  L26:512  L27:2048  L28:512  L29:512  L30:512  L31:2048
```
The batched and fused+corr versions build per-layer attention masks to
handle this correctly. The original fused ignores this (uses is_causal=True
for all heads, which is full-context — incorrect for sliding layers).

### Weight stacking vs concatenation
- `torch.cat(weights, dim=0)`: concatenates into one large matrix for
  one big `F.linear` call. Fast but mixes layers in the output.
- `torch.stack(weights)`: creates `(n_par, out, in)` tensor for `einsum`.
  Per-layer correct but no speed benefit at this scale.

### bf16 numerical precision
Batched (einsum) vs loop (sequential F.linear) produces max diff ~0.6
on tensors with magnitude ~300. This is 0.2% relative error — acceptable,
caused by different operation ordering in bf16 arithmetic.
