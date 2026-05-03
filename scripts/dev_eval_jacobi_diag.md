# Diagonal Newton (quasi-DEER) Inference: Development Log

## Overview

Evaluating diagonal Newton inference for nanochat models. Unlike IDN
(which approximates J=I), diagonal Newton estimates diag(J) via Hutchinson
and solves the diagonal linear recurrence via associative parallel prefix scan.

All measurements on single H100, SDPA attention, seq_len=128.
d32s models: n_embd=640, 32 layers, ~535M params. diag01 checkpoint (4800 steps).

## Key Result: K=1 Newton Diverges, K≥4 Converges (All Methods)

The earlier conclusion that "Jacobian methods are broken" was based on K=1
only. With more Newton iterations (K≥4), **all methods converge** — including
full JVP, diag scan, and even VJP (wrong direction).

### Convergence test (single token prediction, "Hello...")

**n_par=8 (layers 24-31):**

| Method | K=1 | K=4 | K=8 |
|---|---|---|---|
| IDN loop | ✓ | ✓ | ✓ |
| Full JVP (J*v) | ✗ | ✓ (exact) | ✓ (exact) |
| Diag scan JVP p=1 | ✗ | ✓ | ✓ |
| Diag scan VJP p=4 | ✗ | ✓ | ✓ |

**n_par=16 (layers 16-31):**

| Method | K=1 | K=4 | K=8 | K=16 |
|---|---|---|---|---|
| IDN loop | ✓ | ✓ | ✓ | ✓ |
| Full JVP (J*v) | ✗ | ✓ | ✓ (exact) | ✓ (exact) |
| Diag scan JVP p=1 | ✗ | ✓ | ✓ | ✓ |
| Diag scan VJP p=4 | ✗ | ✓ | ✓ | ✓ |

The Newton iteration is self-correcting: each iteration re-evaluates layers
on corrected inputs, washing out Hutchinson noise and J/J^T direction errors.

**Qwen 0.5B behaves identically** — full JVP on all 24 layers gives garbage
at K=1, but matches sequential exactly at K=16.

### Speed vs quality with multiple iterations (diag01, 65K tokens)

**n_par=8:**

| Method | K | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 15.9ms | 1.00x | 64.0 | --- |
| IDN loop | 1 | 14.1ms | 1.13x | 63.3 | -1.2% |
| Fused+corr (batched IDN) | 1 | 12.2ms | **1.30x** | 66.2 | +3.4% |
| Fused+corr | 4 | 15.6ms | 1.01x | 60.0 | **-6.3%** |
| Fused+corr | 8 | 19.2ms | 0.82x | 61.8 | -3.6% |
| Diag scan JVP p=1 | 4 | 57.3ms | 0.28x | 280.4 | +338% |
| Diag scan JVP p=1 | 8 | 102ms | 0.16x | 64.4 | +0.5% |

**n_par=16:**

| Method | K | Time | Speed | PPL | dPPL% |
|---|---|---|---|---|---|
| Sequential | - | 15.9ms | 1.00x | 64.0 | --- |
| Fused+corr (batched IDN) | 1 | 10.4ms | **1.53x** | 125.9 | +96.5% |
| Fused+corr | 4 | 15.1ms | **1.05x** | 79.1 | +23.6% |
| Fused+corr | 8 | 22.1ms | 0.72x | **64.1** | **+0.0%** |
| Diag scan JVP p=1 | 4 | 63.7ms | 0.25x | 1684 | +2530% |
| Diag scan JVP p=1 | 8 | 109ms | 0.14x | 74.2 | +15.8% |

**Key observations:**
- **Fused+corr K=4 at n_par=16: 1.05x speed, PPL 79 (+24%)** — speed parity
  with reasonable quality.
- **Fused+corr K=8 at n_par=16: exact PPL match** but 0.72x (slower than seq).
- **Diag scan JVP is ~7x slower** per iteration than Fused+corr (math SDPA +
  double-backward overhead). Not practical despite convergence at K=8.
- **IDN K=1 remains the best speed/quality tradeoff** for n_par=8.

## K=1 Analysis: Why Newton Diverges at K=1

### Full Jacobian Newton at K=1 (exact J*v, 131K tokens)

**diag01 model** (seq PPL=57.31):

| n_par | IDN (J=I) | Full JVP (J*v) | Full VJP (J^T*v) |
|---|---|---|---|
| 4 | **54.37** (-5.1%) | 78.85 (+37.6%) | 24,986 (broken) |
| 8 | **57.08** (-0.4%) | 1,185 (+1967%) | 47,558 (broken) |
| 16 | **111.88** (+95.2%) | 17,265 (broken) | 710,623 (broken) |

**diag05 model** (seq PPL=235.07):

| n_par | IDN (J=I) | Full JVP (J*v) | Full VJP (J^T*v) |
|---|---|---|---|
| 4 | **186.76** (-20.6%) | 290.18 (+23.4%) | 17,137 (broken) |
| 8 | **282.38** (+20.1%) | 8,524 (broken) | 258,180 (broken) |
| 16 | **124.46** (-47.1%) | 3,556,580 (broken) | 37,832,843 (broken) |

At K=1, IDN wins at every depth. Full JVP and VJP diverge.

### Diagonal Hutchinson at K=1 (diag(J) via scan)

| Method | diag01 n=16 PPL | diag05 n=16 PPL |
|---|---|---|
| IDN batched (J=I) | **112** | **124** |
| Diag scan VJP p=4 | 345 | 744 |
| Diag scan JVP p=4 | 9,108 | 29,084 |
| Diag scan VJP p=1 | 5,435 | 108,796 |

### Root cause: tangent magnitude

The tangent `h_new[j-1] - h_init` grows rapidly due to nanochat's `x0_lambda`
blending:

```
f(h) = block(resid_lambda * h + x0_lambda * x0)
f(h) - h = (resid_lambda - 1)*h + x0_lambda*x0 + attn + mlp
```

With `x0_lambda` values of ±13, the tangent reaches 10,000-46,000 while
`h_init` has magnitude ~130. The first-order linearization `f(x+δ) ≈ f(x) + Jδ`
breaks down for such large δ.

```
n_par  | tangent mag | h_init mag
  2    |    10,012   |    129
  4    |    14,293   |    129
  8    |    18,873   |    129
 16    |    46,079   |    129
```

IDN (J=I) survives at K=1 because it caps amplification at 1.0:
`h[j] = f_j(h_init) + 1 * tangent` (no amplification regardless of ||J||).

At K≥4, all methods survive because each iteration re-evaluates on corrected
inputs, shrinking the tangent progressively until the linearization is accurate.

## Technical Details

### J vs J^T (JVP vs VJP)

VJP computes `J^T*v` instead of `J*v`. Verified on single layer:
`|J*z - J^T*z|` relative diff = 23.7%. J is non-symmetric due to attention
and MLP with non-symmetric weights.

At K=1, VJP (wrong direction) diverges faster than JVP (correct).
At K≥4, both converge — the iteration corrects the direction error.

### Why JVP requires math SDPA

`torch.autograd.functional.jvp` uses double-backward. The flash SDPA kernel
(`aten::_scaled_dot_product_flash_attention_backward`) lacks second-order
derivatives. The math SDPA backend (explicit matmul+softmax) supports all
AD modes but is ~3-4x slower.

```python
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Sanity checks

1. **Diag scan vs diag loop**: identical PPL (344 vs 344), confirming
   associative scan is correct.

2. **Qwen 0.5B comparison**: same behavior — K=1 diverges on 24 layers,
   K=16 converges to exact sequential output. Not a nanochat-specific bug.

3. **Full JVP convergence**: K=8 gives h[-1] diff=0.0 for n_par=8,
   K=16 gives diff=0.0 for n_par=16. Exact convergence verified.

## Initialization Strategy: batched_fwd vs h0

The hidden state initialization for parallel layers is critical for convergence.

**h0 init** (what we used initially):
```
hs = [h_init, h_init, h_init, ..., h_init]
```
All layers start at the prefix output. At K=1, every layer evaluates on h_init.
The tangent `f_j(h_init) - h_init` is large (10K-46K) → Newton correction diverges.

**batched_fwd init** (from the Qwen demo):
```
hs = [h_init, f_0(h_init), f_1(h_init), ..., f_{n-1}(h_init)]
```
Each layer starts at its own output given h_init. The linearization point for
layer j is `hs[j] = f_{j-1}(h_init)` which is close to the true sequential input.
The correction tangent `h_new[j] - hs[j]` is much smaller → better linearization.

### Diag scan with batched_fwd init + JVP (single token, diag01 model)

**n_par=8:**

| Config | h0 init | fwd init |
|---|---|---|
| p=1, K=1 | ✗ | ✗ |
| **p=4, K=1** | **✗** | **✓ (diff=15K)** |
| p=8, K=4 | ✓ | ✓ (diff=320) |
| IDN K=1 | ✓ (diff=13K) | ✓ (diff=7.7K) |

**n_par=16:**

| Config | h0 init | fwd init |
|---|---|---|
| p=1, K=1 | ✗ | ✗ |
| **p=4, K=1** | **✗** | **✓ (diff=8.7K)** |
| p=8, K=4 | ✗ | ✓ (diff=1.3K) |
| IDN K=1 | ✓ (diff=11K) | ✓ (diff=3.8K) |

**Two fixes make diag scan work at K=1:**
1. **batched_fwd init**: gives each layer a starting point close to its true
   sequential input, minimizing the correction tangent.
2. **≥4 probes**: reduces Hutchinson noise enough for stable correction.

With p=8 K=4 fwd_init, diag scan achieves diff=320 (n_par=8) and 1344 (n_par=16)
— better than IDN K=1 (7680/3840). The diagonal Jacobian information DOES help
when properly initialized.

The cost: batched_fwd init requires one extra batched forward pass (computing
f_j(h_init) for all j). This is the same cost as one Newton iteration with IDN.

## AR Generation with KV Cache

### Previous warm-start PPL eval was invalid

The `eval_jacobi_full.py` warm-start used sequential hidden states from the
SAME position as init — equivalent to knowing the answer. In real AR decoding,
you only have hidden states from the PREVIOUS token. When tested with proper
AR generation (token-by-token with KV cache), warm-start from previous token
fails completely (0-2/20 tokens match for IDN K=1).

### Proper KV cache implementation

For layer-parallel AR decode, the KV cache must be handled carefully:

1. **Extract** cached K,V for parallel layers as plain tensors
2. **Concatenate** with current token's K,V in the batched forward
3. Run Newton correction to get corrected hidden states
4. **Recompute** K,V from corrected inputs and **insert** back into cache

This follows the Qwen demo's approach (`_batched_cache_update`). The nanochat
implementation uses `_batched_block_forward_kv` which accepts `cached_k, cached_v`
tensors and does explicit matmul attention (not SDPA) to handle the S+T context.

### AR generation results (IDN, baseline_9600, proper KV cache)

| n_par | K=1 | K=4 | K=8 |
|---|---|---|---|
| 8 | 2/20, 0.89x | 4/20, 0.76x | **18/20, 0.56x** |
| 16 | 1/20, 1.00x | 3/20, 0.81x | 5/20, 0.55x |
| 24 | 1/20, **1.27x** | 1/20, 0.94x | 1/20, 0.68x |

Key: `match/total, speedup`. Sequential is 1.00x.

- **n_par=8 K=8 converges** (18-20/20 match) but 0.56x (slower than sequential)
- **n_par=24 K=1 gives 1.27x speedup** but garbage quality (baseline, no reg)
- **n_par=16,24 don't converge even at K=8** with IDN — too many layers

### IDN training helps convergence

| Model | n_par=8 K=8 match |
|---|---|
| baseline_9600 | 16/20 |
| diag01_9600 | 15/20 |
| **idn01_9600** | **20/20** |

IDN-trained model converges perfectly at K=8 n_par=8.

### Qwen demo comparison

Qwen 0.5B (D=896, 24 layers) with diag scan VJP K=1 n_par=24:
- Sequential: 67 tok/s
- Diag scan: **91 tok/s (1.36x)** — garbage quality (untrained)

This confirms real speedup is achievable. The open question: does diag-trained
nanochat model converge at K=1 n_par=24 with this speedup?

## Conclusions

1. **K=1 Newton with h0 init diverges** because tangent magnitudes are too large
   for the first-order linearization. IDN (J=I) is the only method that works
   at K=1 with h0 init because it avoids amplification.

2. **Proper AR generation with KV cache works** — IDN K=8 n_par=8 converges
   to 18-20/20 token match with sequential. IDN training (idn01) gives perfect
   20/20 convergence.

3. **n_par=24 gives 1.27x speedup** but quality doesn't converge with IDN at
   any K. Need diag scan (better Jacobian approximation) for convergence at
   aggressive parallelism levels.

4. **Previous warm-start PPL evals were invalid** — used same-position hidden
   states (cheating). Real AR warm-start from previous token is much harder.

5. **Next step**: implement diag scan in AR generation with KV cache. Test if
   diag-trained model + diag scan K=1 n_par=24 converges AND gives speedup.
