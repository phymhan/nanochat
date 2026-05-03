# Qwen quasi-DEER Jacobian mode comparison

All runs use `CUDA_VISIBLE_DEVICES=0`, model `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, bf16).

Fixed settings: `--hidden-init warm_start --cache-update kv_only`

Script: `scripts/demo_depth_quasi_deer_qwen.py`

## CLI interface

Two independent axes control the algorithm:

```
--jacobian {full, diag}   Whether to use the full Jacobian or diag(J)
--jvp {jvp, fd, vjp}      How to compute the Jacobian-vector product
--scan                     Use associative scan (parallel) vs forward substitution (sequential)
```

| `--jacobian` | `--scan` | `--jvp` | Description |
|---|---|---|---|
| full | off | jvp | Exact JVP Newton, sequential forward substitution |
| full | off | fd | FD-Newton, sequential forward substitution |
| full | off | vjp | VJP (transpose) Newton, sequential forward substitution |
| full | on | jvp | Materialize full J, matrix scan (O(D³), debug only) |
| diag | on | jvp | Batched manual fwd + Hutchinson diag + scan |
| diag | on | fd | FD Hutchinson diag + scan |
| diag | on | vjp | VJP Hutchinson diag + scan |

## Recurrent reference

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/demo_depth_quasi_deer_qwen.py \
  --prompt "Hello" --max-new-tokens 10 --no-chat-template --layer-forward recurrent
```

Output: `I am trying to create a simple program that`
Decode: 0.126s

## 1. Full Jacobian modes (iters=20)

`--jacobian full`, varying `--jvp` and `--layer-start`.

| jvp | seq | par | Match | Decode |
|-----|----:|----:|:-----:|-------:|
| jvp |   0 |  24 | YES   | 15.70s |
| fd  |   0 |  24 | NO    |  4.10s |
| vjp |   0 |  24 | NO    |  6.79s |
| jvp |  12 |  12 | YES   |  3.14s |
| fd  |  12 |  12 | YES   |  1.74s |
| vjp |  12 |  12 | YES   |  2.34s |
| jvp |  20 |   4 | YES   |  0.76s |
| fd  |  20 |   4 | YES   |  0.59s |
| vjp |  20 |   4 | YES   |  0.69s |

- **`jvp` (exact)** matches at all configs including seq=0. Most robust.
- **`fd`** matches for seq>=4. FD bf16 noise amplifies over 24 layers.
- **`vjp`** matches for seq>=4. Similar amplification issue.

## 2. Diagonal Jacobian + scan modes (iters=20)

`--jacobian diag --scan`, varying `--jvp` and `--layer-start`.

| jvp | seq | par | Match | Decode |
|-----|----:|----:|:-----:|-------:|
| jvp |  20 |   4 | 3/5\* |  1.25s |
| fd  |  20 |   4 | 0/5\* |  0.51s |
| vjp |  20 |   4 | 4/5\* |  0.71s |
| jvp |  12 |  12 | NO    |  1.33s |
| vjp |  12 |  12 | YES†  |  0.75s |

\* Tested across 5 random seeds — diagonal modes are stochastic due to the Hutchinson estimator.

† Single seed; may vary.

### Observations

The diagonal approximation is inherently noisier than the full Jacobian:

- **`diag jvp`**: works ~60% of the time at par=4 due to Hutchinson variance.
- **`diag fd`**: never converges. FD noise in bf16 compounds with Hutchinson noise — two layers of approximation.
- **`diag vjp`**: works ~80% of the time at par=4. Slightly more robust than jvp, possibly because Jᵀz and J are closer when the Jacobian is near-symmetric.

## 3. Convergence vs iteration count (seq=20, par=4)

`--jacobian full`, varying `--parallel-iters`.

| jvp | iters | Match | Decode |
|-----|------:|:-----:|-------:|
| jvp |     2 | NO    | 0.395s |
| fd  |     2 | NO    | 0.200s |
| vjp |     2 | NO    | 0.275s |
| jvp |     4 | YES   | 0.645s |
| fd  |     4 | YES   | 0.262s |
| vjp |     4 | YES   | 0.343s |
| jvp |     6 | YES   | 0.909s |
| fd  |     6 | YES   | 0.299s |
| vjp |     6 | YES   | 0.408s |

All full-J modes converge at **4 iterations** for 4 parallel layers with warm start.

## Summary

| Mode | Jacobian | Scan | JVP | Max par (iters=20) | Cost / layer / iter | Notes |
|------|----------|:----:|-----|:------------------:|:-------------------:|-------|
| full jvp | full | no | jvp | 24 (all) | ~6.1x fwd | Most robust, needs eager attn |
| full fd | full | no | fd | 20 (seq>=4) | ~2.0x fwd | Fastest per-iter |
| full vjp | full | no | vjp | 20 (seq>=4) | ~3.6x fwd | Works with SDPA/Flash |
| diag jvp | diag | yes | jvp | ~4 (stochastic) | ~6.1x fwd (batched) | Parallel, Hutchinson noise |
| diag vjp | diag | yes | vjp | ~4 (stochastic) | ~3.6x fwd (batched) | Parallel, less noisy |
| diag fd | diag | yes | fd | fails | ~2.0x fwd (batched) | FD + Hutchinson = too noisy |

Cost ratios from measured ops benchmark (`scripts/test_ops_bench.md`):
- `forward`: 0.554ms (1.00x)
- `FD total` (f(x) + f(x+εv)): 1.111ms (2.01x)
- `VJP` (Jᵀv via backward AD): 1.999ms (3.61x)
- `JVP` (Jv via forward AD, includes primal): 3.369ms (6.08x)

**Recommendations:**
- Correctness at any split: `--jacobian full --jvp jvp`
- Speed with seq>=4: `--jacobian full --jvp fd` (fastest per-iter)
- Without eager attn: `--jacobian full --jvp vjp` (works with SDPA/Flash)
- Parallel with scan: `--jacobian diag --jvp vjp --scan` (best diagonal robustness)

## 4. Wall-clock upper bound: can DEER beat recurrent?

Measured decode time for 20 tokens, varying `--layer-start` at `--parallel-iters 1`.

```
                     decode/20tok  per_tok
recurrent  (24L)       0.254s      12.7ms
```

**Sequential layers scale flat** — running 4 vs 24 layers recurrently costs the
same (~0.255s), because at batch=1 single-token decode the per-layer compute
(~7µs) is negligible relative to Python loop + kernel launch overhead (~12ms).

### DEER at 1 iteration

| Method | seq | par | Decode | per_tok |
|--------|----:|----:|-------:|--------:|
| **diag/vjp/scan** | 0 | 24 | **0.169s** | **8.5ms** |
| diag/vjp/scan | 4 | 20 | 0.204s | 10.2ms |
| diag/vjp/scan | 8 | 16 | 0.245s | 12.3ms |
| diag/vjp/scan | 12 | 12 | 0.273s | 13.7ms |
| diag/vjp/scan | 20 | 4 | 0.332s | 16.6ms |
| full/fd | 0 | 24 | 0.611s | 30.6ms |
| full/fd | 20 | 4 | 0.327s | 16.4ms |
| full/fd | 22 | 2 | 0.297s | 14.9ms |

**`diag/vjp/scan` at seq=0 is 1.5x faster than recurrent** (8.5ms vs 12.7ms per
token).  The batched manual-layer forward amortizes kernel launch overhead into
a few large einsum calls instead of 24 sequential Python-loop iterations.

Adding sequential prefix layers erodes the advantage (~2ms per 4 sequential
layers).  At seq=8 it is roughly breakeven.

`full/*` (sequential forward substitution) is always slower than recurrent — the
Python for-loop kills any parallelism benefit.

**Speed ceiling:** DEER is faster than recurrent when `K × 8.5ms < 12.7ms`,
i.e. **K = 1**.  If a model converges at 1 iteration (via training), this is a
genuine 1.5x wall-clock speedup.

## 5. Convergence quality at K=1: diag vs identity Newton

Cosine similarity of the last-token logits with recurrent ground truth,
1 iteration, untrained model (prompt = "The quick brown fox jumps over the lazy
dog").

| seq | par | diag/jvp/scan | diag/vjp/scan | identity (J=I) |
|----:|----:|--------------:|--------------:|---------------:|
|   0 |  24 |        0.2030 |        0.1276 |         0.0098 |
|   4 |  20 |        0.0456 |       -0.0400 |         0.0366 |
|   8 |  16 |        0.0203 |       -0.0203 |         0.0485 |
|  12 |  12 |        0.1289 |        0.1289 |         0.1158 |
|  16 |   8 |        0.2030 |        0.1608 |         0.3072 |
|  20 |   4 |        0.8557 |        0.5110 |         0.6486 |

At K=1 on an **untrained** model, all methods are poor for large par.  Neither
diag nor identity consistently dominates.

### Convergence trajectory (seq=20, par=4)

| iters | diag/vjp/scan | diag/jvp/scan | full/jvp | full/fd |
|------:|--------------:|--------------:|---------:|--------:|
|     1 |         0.511 |         0.856 |    0.858 |   0.376 |
|     2 |         0.996 |         0.999 |    1.000 |   0.215 |
|     3 |         0.999 |         0.977 |    1.000 |   1.000 |
|     4 |         0.999 |         0.999 |    1.000 |   1.000 |

### Convergence trajectory (seq=16, par=8)

| iters | diag/vjp/scan | diag/jvp/scan | full/jvp | full/fd |
|------:|--------------:|--------------:|---------:|--------:|
|     1 |         0.128 |         0.203 |    0.174 |  -0.061 |
|     2 |         0.765 |         0.978 |    0.874 |  -0.101 |
|     3 |         0.999 |         0.965 |    0.994 |   0.085 |
|     4 |         0.986 |         0.270 |    1.000 |   0.054 |

Note: `diag/jvp` at seq=16 regresses from 0.965→0.270 at iter 4 — this is the
single-sample Hutchinson variance problem (a bad random z draw).

## 6. Implications for training

**Diagonal-Newton-aware training** could be a weaker and potentially better
alternative to identity-Newton training:

| Training regularization | Constraint | What it forces |
|---|---|---|
| Identity Newton (J ≈ I) | Strong | All off-diagonal entries ≈ 0, diagonal ≈ 1 |
| Diagonal Newton (diag(J)) | Weak | Only needs `f(x) ≈ diag(J)·x + bias` to hold |

The diagonal variant:
- Is a strictly better approximation than identity (uses actual diag(J) instead of 1)
- Imposes a weaker constraint → potentially less base PPL degradation
- Has the same training cost (1 JVP per sampled N to get the Hutchinson estimate)
- At inference uses the same fast `diag + scan` path (8.5ms vs 12.7ms recurrent)

The training loss would be:
```
L_diag = || h_L^{diag-Newton}(N, K=1) - h_L^{seq} || / || h_L^{seq} || + eps)
```
where `h_L^{diag-Newton}` runs the parallel layers through 1 iteration of
diagonal quasi-Newton with the Hutchinson diagonal estimate, using the correct
sequential prefix as input.

## 7. Scale-ELK: stabilizing diagonal DEER with eigenvalue damping

Scale-ELK (from [arXiv:2407.19115](https://arxiv.org/abs/2407.19115)) attenuates
the diagonal Jacobian by `(1-k)` before the scan:
```
diag_j = (1 - k) * diag(J)
```
- `k=0` → standard DEER (no damping)
- `k=1` → identity Newton (drop all Jacobian info)
- `k ∈ (0,1)` → eigenvalues scaled down, improving scan stability

Added as `--elk-k` in `demo_depth_quasi_deer_qwen.py`.

### Stability across seeds (iters=20)

Tested across 5 random seeds (Hutchinson z vectors vary per seed).

**diag / jvp / scan:**

| seq | par | k=0.0 | k=0.1 | k=0.3 | k=0.5 |
|----:|----:|:-----:|:-----:|:-----:|:-----:|
|  12 |  12 | 0/5   | 0/5   | 0/5   | 2/5   |
|  16 |   8 | 1/5   | 1/5   | 3/5   | 4/5   |
|  20 |   4 | 3/5   | **5/5** | 5/5 | 5/5   |

**diag / vjp / scan:**

| seq | par | k=0.0 | k=0.1 | k=0.3 | k=0.5 |
|----:|----:|:-----:|:-----:|:-----:|:-----:|
|  16 |   8 | 5/5   | 3/5   | 5/5   | 5/5   |
|  20 |   4 | 4/5   | **5/5** | 5/5 | 5/5   |

### Observations

- **k=0.1 fully stabilizes par=4** for both jvp and vjp (3/5 → 5/5, 4/5 → 5/5).
- For par=8, k=0.5 gets jvp to 4/5 (from 1/5). VJP is already stable at k=0.
- For par=12, even k=0.5 only reaches 2/5 — the diagonal approximation itself
  is too coarse for 12 layers, not just a stability issue.
- Note: k=0.1 *hurts* vjp at par=8 (5/5 → 3/5) because damping slows
  convergence — 20 iterations may not be enough for the damped version.
  Higher k values (0.3+) recover because the stabilization benefit outweighs
  the slower convergence.
- **Cost**: zero — just one scalar multiply on `diag_j` before the scan.

## 8. Quasi-ELK: trust region via scalar Kalman filter

Full quasi-ELK ([arXiv:2407.19115](https://arxiv.org/abs/2407.19115)) replaces
the 2-tuple diagonal scan with a 5-tuple scalar Kalman filter scan.  The
previous iteration's states serve as noisy observations with variance σ²,
anchoring the Newton step to a trust region.

Added as `--elk-sigmasq` in `demo_depth_quasi_deer_qwen.py`.

- `sigmasq=0` → disabled (plain DEER or Scale-ELK)
- `sigmasq` large (1e4+) → weak trust region ≈ DEER
- `sigmasq` small (1e1-1e2) → strong damping
- Cost: ~2.5x the diagonal scan (5-tuple vs 2-tuple), still O(LD)

### Stability across seeds (iters=20, diag/jvp/scan)

| seq | par | DEER (k=0) | σ²=1e8 | σ²=1e4 | σ²=1e2 |
|----:|----:|:----------:|:------:|:------:|:------:|
|  20 |   4 | 3/5        | **5/5** | 5/5   | 5/5    |
|  16 |   8 | 1/5        | 0/5    | 0/5    | **4/5** |

### Head-to-head: Scale-ELK vs quasi-ELK at par=8

| Method | Seeds matching |
|--------|:--------------:|
| Scale-ELK k=0.1 | 1/5 |
| Scale-ELK k=0.3 | 3/5 |
| Scale-ELK k=0.5 | 4/5 |
| Scale-ELK k=0.7 | **5/5** |
| Quasi-ELK σ²=1e4 | 0/5 |
| Quasi-ELK σ²=1e3 | 0/5 |
| Quasi-ELK σ²=1e2 | 4/5 |
| Quasi-ELK σ²=1e1 | **5/5** |

Both methods achieve comparable stabilization.  Scale-ELK is simpler (one scalar
multiply), quasi-ELK is more principled (Kalman trust region; scan itself is ~3.5x costlier
but total iteration overhead is only ~7-8% since the scan is a small fraction).  For par=4, even very light damping (k=0.1 or σ²=1e8) is sufficient.
For par=8, stronger damping is needed (k=0.5-0.7 or σ²=1e1-1e2).
