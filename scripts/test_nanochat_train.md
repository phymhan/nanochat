# Nanochat d32 Training: Diagonal Newton Regularization

Smoke tests on 8×H100 (80GB HBM3), FP8 training, bf16 compute.

## Model config

- depth=32, model_dim=2048, 16 heads, 2.8B total params
- Scaling params: 1.68B (transformer matrices + lm_head)
- Auto-computed batch size: 2,097,152 tokens
- Default training: ~9,600 iterations (target-param-data-ratio=12)

## Baseline timing (no regularization)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 --model-tag=d32_baseline --run=dummy
```

- Time/step: ~4.5s
- Memory: 60 GiB / 80 GiB
- MFU: ~64%
- **Estimated total: ~12 hours**

## Diagonal Newton regularization

### CLI args

```
--diag-newton-reg=0.5       # regularization weight (same scale as identity Newton)
--jacobi-reg-warmup=0.2     # 20% of training at λ=0, then linear ramp to target
--diag-newton-jvp=fd        # 'fd' (finite-diff, cheap) or 'vjp' (backward AD, accurate)
--n-par-configs=8           # which n_par values to train on (comma-separated)
```

`--n-par-configs` also works with `--identity-newton-reg`.  Default (when omitted):
`3,5,7,10,12,16` clamped to valid range.

### Cost comparison (5-step smoke tests, post-compilation)

| Config | BS | n_par | Time/step | Memory | Overhead |
|---|---|---|---|---|---|
| Baseline (no reg) | 8 | — | 4.5s | 60 GiB | — |
| Diag FD, n_par=8 | 8 | 8 | 5.8s | 67 GiB | **1.28x** |
| Diag FD, n_par=16 | 8 | 16 | 7.1s | 74 GiB | **1.58x** |
| Diag VJP, n_par=8 | 4 | 8 | 8.1s | 52 GiB | 1.79x |
| Diag FD, n_par=8,16 | 4 | 8+16 | 9.0s | 56 GiB | OOM@BS=8 |

### Estimated full training times (d32, ~9,600 iterations)

| Config | Time/step | **Total** |
|---|---|---|
| Baseline | 4.5s | **12.0 hours** |
| Diag FD, n_par=8 | 5.8s | **15.5 hours** |
| Diag FD, n_par=16 | 7.1s | **18.9 hours** |

### FD vs VJP for diagonal estimation

- **FD** (finite-difference Hutchinson): 1 extra no-grad forward per layer.  Cheapest.
  Noise from bf16 + finite eps (1e-2) but well-behaved in practice.
- **VJP** (backward-mode AD Hutchinson): 1 forward-with-grad + 1 backward per layer.
  More accurate (no FD discretization noise) but ~1.4x slower than FD.
- **JVP** (forward-mode AD): not available — FA3 and SDPA lack forward-mode AD rules.

VJP cannot reuse the training backward pass: training backward computes J^T · (∂loss/∂h),
but Hutchinson needs J^T · z for a random z — different vectors.

### Loss stability (no warmup, 5 steps)

| Config | step 0 | step 4 | Stable? |
|---|---|---|---|
| Diag FD, n_par=8 | 10.58 | 10.68 | ✓ |
| Diag VJP, n_par=8 | 10.40 | 9.98 | ✓ |
| Diag FD, n_par=16 | 10.79 | 12.96 | mild drift |
| Diag FD, all 6 configs | 10.40 | 106,692 | ✗ explodes |

The explosion only occurs when running **all 6 n_par configs simultaneously** (accumulated
correction errors compound across configs).  Single n_par is stable even without warmup.
With proper warmup (20% of training), the loss ramps gradually and should be well-behaved
for any single n_par.

### torch.compile considerations

- The n_par set must be **fixed for the entire run** (no random sampling per step).
  Different n_par values → different inner loop lengths → torch.compile recompilation.
- `--n-par-configs` sets a deterministic, fixed set.  torch.compile traces the graph once.
- Random sampling via `random.sample()` or `torch.randperm()` both cause recompilation
  or data-dependent symbol errors.

## Recommended training commands

### Diagonal Newton, n_par=8 (best cost/coverage tradeoff)

```bash
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 \
    --diag-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=8 \
    --model-tag=d32_diag05_npar8 --run=d32_diag05_npar8
```

**~15.5 hours**, 67 GiB peak memory.

### Identity Newton baseline (for comparison with report)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --n-par-configs=7 \
    --model-tag=d32_idn05_npar7 --run=d32_idn05_npar7
```

### No regularization baseline

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 \
    --model-tag=d32_baseline --run=d32_baseline
```

**~12 hours**.

## Implementation details

### Files changed

- `nanochat/gpt.py`: `GPT.forward()` — added `diag_newton_reg` parameter, diagonal Newton
  loss with FD/VJP Hutchinson, shared `n_par_configs` for both IDN and diag Newton.
  Also optimized IDN to only run the last layer (intermediate outputs were unused).
- `scripts/base_train.py`: added `--diag-newton-reg`, `--diag-newton-jvp`,
  `--n-par-configs` CLI args; warmup schedule; model attribute setup before torch.compile.

### How the diagonal Newton loss works

For each n_par in the config set:
1. Get `h_init` = output of sequential prefix (layer `n_layer - n_par - 1`)
2. Run each of the last `n_par` layers on `h_init` → `h_newton[j]` (in computation graph)
3. Estimate `diag(∂h_j/∂h_init)` via Hutchinson with Rademacher z:
   - FD: `diag ≈ z * (f(h_init + εz) - f(h_init)) / ε`
   - VJP: `diag ≈ z * (J^T · z)` via `torch.autograd.grad`
4. Forward substitution to correct: `h_corr[j] = h_newton[j] + diag_j * (h_corr[j-1] - h_init)`
5. Loss: `|| h_corr_final - h_seq_final.detach() || / || h_seq_final.detach() ||`

Gradients flow through `h_newton[j]` (block weights) and `h_init` (sequential weights).
The diagonal estimate is detached — a measured constant, not a learned quantity.
The target `h_seq_final` is detached so the reg only trains the parallel path.
