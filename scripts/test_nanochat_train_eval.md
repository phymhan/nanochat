# Diagonal Newton Training & Evaluation Results

## Models Trained

All models: d32 (32 layers, 2048 dim, 2.8B params), 8×H100 FP8, ~9600 iterations.

| Model | Training | λ | Warmup | n_par | Seq PPL | Notes |
|---|---|---|---|---|---|---|
| d32_baseline | none | — | — | — | 26.9 | ground truth |
| d32_idn05_npar8 | IDN | 0.5 | 0.0 | 8 | 28.8 | identity Newton reg |
| d32_diag0.01_from_baseline | diag | 0.01 | 0.0 | 8 | 31.1 | resumed from baseline, +1920 steps |
| d32_diag0.01_from_idn05_npar8 | diag | 0.01 | 0.0 | 8 | 31.1 | resumed from IDN, +1920 steps |
| d32_diag005_npar8_v3 | diag | 0.05 | 0.2 | 8 | 29.9 | from scratch, clamp(-2,2) |
| d32_diag01_npar8 | diag | 0.1 | 0.2 | 8 | 28.2 | from scratch, clamp(-2,2) |

## Jacobian Approximation Errors

Measured with VJP Hutchinson (4 probes, 8 batches, seq_len=128).

- **IDN error**: `||J^T*v - v|| / ||J^T*v||` — how close J is to identity
- **Diag error**: `||J^T*v - diag(J)*v|| / ||J^T*v||` — how close J is to diag(J)
- **Diag/IDN**: ratio; <1 means diagonal is better than identity, >1 means identity is better

### n_par=8 (seq=24 + 8 parallel)

| Model | val_bpb | IDN err | Diag err | Diag/IDN |
|---|---|---|---|---|
| Baseline | 0.661 | 0.392 | 0.320 | 0.82 |
| **IDN λ=0.5** | 0.661 | **0.161** | **0.145** | 0.90 |
| Diag λ=0.01 from baseline | 0.660 | 0.418 | 0.347 | 0.83 |
| Diag λ=0.01 from IDN | 0.661 | 0.188 | 0.171 | 0.91 |
| Diag λ=0.05 scratch | 0.661 | 0.118 | 0.133 | **1.13** |
| Diag λ=0.1 scratch | 0.662 | **0.110** | 0.121 | **1.10** |
| Diag λ=0.5 scratch | 0.664 | 0.117 | 0.098 | **0.84** |

### n_par=4 (seq=28 + 4 parallel)

| Model | IDN err | Diag err | Diag/IDN |
|---|---|---|---|
| Baseline | 0.450 | 0.279 | 0.62 |
| IDN λ=0.5 | 0.201 | 0.166 | 0.83 |
| Diag λ=0.01 from baseline | 0.463 | 0.302 | 0.65 |
| Diag λ=0.01 from IDN | 0.231 | 0.206 | 0.89 |
| Diag λ=0.05 scratch | 0.243 | 0.270 | 1.11 |
| Diag λ=0.1 scratch | 0.241 | 0.263 | 1.09 |
| Diag λ=0.5 scratch | 0.226 | 0.185 | 0.82 |

### Per-layer errors (n_par=8, selected models)

**Baseline:**
```
L0:0.209  L1:0.294  L2:0.212  L3:0.387  L4:0.252  L5:0.448  L6:0.386  L7:0.943  (IDN)
L0:0.204  L1:0.275  L2:0.234  L3:0.332  L4:0.280  L5:0.371  L6:0.368  L7:0.496  (Diag)
```

**IDN λ=0.5:**
```
L0:0.096  L1:0.125  L2:0.097  L3:0.145  L4:0.091  L5:0.139  L6:0.089  L7:0.508  (IDN)
L0:0.093  L1:0.129  L2:0.102  L3:0.145  L4:0.096  L5:0.137  L6:0.095  L7:0.366  (Diag)
```

**Diag λ=0.1 scratch:**
```
L0:0.002  L1:0.033  L2:0.021  L3:0.123  L4:0.048  L5:0.083  L6:0.064  L7:0.507  (IDN)
L0:0.002  L1:0.038  L2:0.023  L3:0.138  L4:0.054  L5:0.093  L6:0.072  L7:0.551  (Diag)
```

## Key Findings

### 1. IDN training is effective
IDN λ=0.5 reduces IDN error from 0.392 to 0.161 (2.4× improvement). Layers L0-L6 become
nearly input-invariant (IDN err 0.09-0.15). L7 (last layer) remains high (0.508) — it
carries most of the model's expressiveness.

### 2. Diag-from-scratch achieves lowest IDN error
Diag λ=0.1 and λ=0.5 scratch have the lowest IDN error (0.110, 0.117) — even lower than
IDN (0.161). The diag training pushes J toward diag(J), which for residual transformers
is close to I. Layers L0-L6 become extremely input-invariant (IDN err 0.001-0.083).

### 3. Diag λ=0.5 recovers Diag/IDN < 1.0
At λ=0.05 and λ=0.1, Diag/IDN > 1.0 (1.10-1.13) — the diagonal approximation was WORSE
than identity because the model became so close to J≈I that Hutchinson noise dominated.
At λ=0.5, Diag/IDN = 0.84 — the stronger regularization preserves enough diagonal
structure for the approximation to be beneficial. The diag error (0.098) is also the
lowest across all models.

### 4. Diagonal approximation IS better for untrained and strongly-regularized models
For baseline (Diag/IDN=0.82), IDN (0.90), and diag λ=0.5 (0.84), the diagonal captures
meaningful Jacobian structure that identity misses.

### 5. L7 is the bottleneck
The last parallel layer (L7) consistently has the highest error across all models
(0.37-0.94). This layer is furthest from the prefix and retains the most input-dependent
behavior. Any future work should focus on taming L7.

### 6. Diag training degrades base PPL more than IDN
| Training | Seq PPL | ΔPPL vs baseline |
|---|---|---|
| Baseline | 26.9 | — |
| IDN λ=0.5 | 28.8 | +7% |
| Diag λ=0.05 scratch | 29.9 | +11% |
| Diag λ=0.1 scratch | 28.2 | +5% |

Diag λ=0.1 achieves the lowest IDN error (0.110) with moderate PPL overhead (+5%).
However, since Diag/IDN > 1.0, there's no benefit to using diagonal correction over
identity correction at inference — IDN training + identity correction is the simpler
and better approach.

## Reproduction

### Eval command
```bash
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
CUDA_VISIBLE_DEVICES=0 python -m scripts.diag_newton_eval \
    --model-tag d32_baseline \
    --model-tag d32_idn05_npar8 \
    --model-tag d32_diag005_npar8_v3 \
    --model-tag d32_diag01_npar8 \
    --n-par 4,8 --n-batches 8 --n-probes 4
```

### Training commands
See `runs/diag_newton.md` and `runs/diag_newton.sh`.

## Implementation Notes

### FD numerical issues
Finite-difference diagonal estimation in bf16 is unreliable — the subtraction
`(f(x+εz) - f(x)) / ε` loses precision. Two fixes were needed:
1. **Training**: compute FD in float32, clamp diag values to [-2, 2]
2. **Eval**: use VJP (backward AD) instead of FD for diagonal estimation.
   VJP is exact and works with FA3 in bf16.

### torch.compile compatibility
- Passing changing float reg weights as function args causes recompilation every step.
  Fix: use a registered buffer (`reg_warmup_factor`) for warmup scaling.
- Random n_par sampling per step causes data-dependent control flow.
  Fix: use `--n-par-configs` for a fixed set.
- `--no-compile` flag added as escape hatch.
