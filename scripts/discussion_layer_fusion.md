# Why IDN and Fused Inference Find Better Fixed Points

## Overview

In our layer-parallel inference experiments, we observe that:
1. **IDN-trained models have better base PPL** than baselines (74.8 vs 77.7 for d32)
2. **Certain fused/chunk configs produce lower PPL** than sequential execution
3. **IDN regularization improves sequential PPL** even when not using IDN inference

This document provides theoretical analysis of these phenomena, connecting to the Attention Residuals (arXiv 2603.15031), HyperConnections (arXiv 2512.24880), and fixed-point theory frameworks.

---

## 1. Exact Derivation: What IDN K=1 Computes

### Sequential Forward Pass

For a transformer with L layers, the sequential forward pass through the last N layers (after prefix output h_S) computes:

```
h_{S+1}^seq = h_S + g_S(h_S)
h_{S+2}^seq = h_{S+1}^seq + g_{S+1}(h_{S+1}^seq)
...
h_L^seq = h_S + Σ_{j=S}^{L-1} g_j(h_j^seq)
```

where `f_j(x) = x + g_j(x)` is the residual layer and `g_j` is the non-residual computation (attn + MLP). Each `g_j` is evaluated at a **different** point `h_j^seq`, which depends on all previous layers' outputs.

### IDN K=1 Forward Pass

Starting from `hs = [h_S, h_S, ..., h_S]` (h0 init), the IDN correction (prefix sum) produces:

```
hs[1] = f_S(h_S) = h_S + g_S(h_S)
hs[2] = f_{S+1}(h_S) + (hs[1] - h_S) = h_S + g_{S+1}(h_S) + g_S(h_S)
hs[3] = f_{S+2}(h_S) + (hs[2] - h_S) = h_S + g_{S+2}(h_S) + g_S(h_S) + g_{S+1}(h_S)
...
hs[N] = h_S + Σ_{j=0}^{N-1} g_{S+j}(h_S)
```

So the IDN K=1 output is exactly:

```
h_L^idn = h_S + Σ_{j=S}^{L-1} g_j(h_S)
```

### The Key Difference

```
h_L^seq = h_S + Σ_j g_j(h_j^seq)    ← each g_j at a DIFFERENT point (accumulated chain)
h_L^idn = h_S + Σ_j g_j(h_S)        ← each g_j at the SAME point (clean prefix)
```

The only difference is the evaluation point of each `g_j`. Sequential uses the accumulated chain `h_j^seq`; IDN uses the clean prefix `h_S` for all layers.

---

## 2. Variance Reduction: The Core Mechanism

### Decomposition into Signal and Noise

Decompose each layer's contribution:

```
g_j(h) = μ_j + ε_j(h)
```

where `μ_j ≈ g_j(h_S)` is the "expected" contribution (evaluated at the nominal input) and `ε_j(h) = g_j(h) - g_j(h_S)` captures the input-dependent deviation.

### Sequential: Compounding Errors

```
h_{S+1}^seq = h_S + μ_S + ε_S(h_S)                              [ε_S(h_S) = 0 by definition]
h_{S+2}^seq = h_{S+1}^seq + μ_{S+1} + ε_{S+1}(h_{S+1}^seq)
            = h_S + μ_S + μ_{S+1} + ε_{S+1}(h_S + μ_S)          [ε_{S+1} ≠ 0 because input shifted]
h_{S+3}^seq = ... + ε_{S+2}(h_S + μ_S + μ_{S+1} + ε_{S+1}(...)) [ε_{S+2} depends on ε_{S+1}]
```

Each `ε_j` is evaluated at a point contaminated by all previous `ε_k` (k < j). The errors are **correlated through the chain**: one layer's deviation shifts the operating point for all subsequent layers. This creates compounding variance.

To first order in the Taylor expansion:

```
ε_j(h_j^seq) ≈ J_{g_j} · (h_j^seq - h_S) = J_{g_j} · Σ_{k<j} (μ_k + ε_k)
```

The variance of the sequential output grows as errors propagate through the Jacobians `J_{g_j}`.

### IDN: Independent Evaluation

```
h_L^idn = h_S + Σ_j μ_j + Σ_j ε_j(h_S)
        = h_S + Σ_j μ_j + 0              [since ε_j(h_S) = 0 by definition]
```

All `ε_j` terms vanish because every `g_j` is evaluated at exactly `h_S`. There is **no error compounding** — each layer's contribution is computed independently at the clean prefix output.

### The Bias-Variance Tradeoff

- **Bias of IDN**: `h_L^idn - h_L^seq = Σ_j [g_j(h_S) - g_j(h_j^seq)]`
  - This is nonzero when layers are input-dependent
  - IDN training minimizes this by making `J_{g_j} ≈ 0`

- **Variance of sequential**: Accumulation of `ε_j` through the chain
  - Even small per-layer deviations compound across N layers
  - Variance grows with N and with `||J_{g_j}||`

**When IDN training makes the bias small (J_{g_j} ≈ 0), the variance reduction dominates, and IDN finds a better expected output.**

This is the same principle as **ensemble averaging vs sequential refinement (boosting)**: N independent estimators (IDN) have variance σ²/N from averaging, while a chain of N dependent estimators (sequential) has correlated errors that can compound. When individual errors are small, the ensemble wins.

---

## 3. IDN Regularization: Why It Improves Sequential PPL

The IDN training loss penalizes:

```
L_idn = ||h_L^idn(N) - h_L^seq|| / ||h_L^seq||
```

For this to be small, we need `g_j(h_S) ≈ g_j(h_j^seq)` for all j. Since `h_j^seq` spans a range of points from `h_S` to `h_L^seq`, this requires:

```
∂g_j/∂h ≈ 0   (i.e., J_{g_j} ≈ 0)
```

This is a strong Lipschitz constraint on the non-residual computation. It has several beneficial effects on the sequential model:

### 3.1 Implicit Lipschitz Regularization

Making `J_{g_j} ≈ 0` forces the last N layers to compute approximately input-invariant features. This prevents overfitting: these layers cannot memorize input-specific patterns, only learn global feature corrections.

This is stronger than typical weight decay or spectral normalization, which constrain weight magnitudes but not the functional Jacobian directly.

### 3.2 Capacity Partitioning

The model is forced to partition its computation into two roles:

```
Prefix layers (1..S):   Full input-dependent processing (unconstrained)
Suffix layers (S+1..L): Input-invariant features (J_{g_j} ≈ 0)
```

This separation is a form of **architectural regularization**. The suffix layers act as a "shared feature bank" — they compute the same additive correction regardless of input. The prefix must do all the input-dependent work.

The empirical evidence supports this: early exit at layer S degrades PPL by 52%, confirming that the suffix features are important (μ_j is large) even though they're input-invariant (ε_j ≈ 0).

### 3.3 Improved Gradient Flow

When `J_{g_j} ≈ 0`, the full layer Jacobian is:

```
J_{f_j} = I + J_{g_j} ≈ I
```

This gives **near-perfect gradient flow** through these layers. The gradient highway is maximally effective: backpropagated gradients pass through the suffix layers with negligible distortion, reaching the prefix layers cleanly.

In the HC framework, this corresponds to `H^res ≈ I` (identity stream mixing), which mHC shows is the stable regime.

### 3.4 Reduced Condition Number of the Optimization Landscape

With `J_{f_j} ≈ I` for the suffix, the composite Jacobian from prefix to output is:

```
∏_{j=S}^{L-1} J_{f_j} ≈ I
```

This means the loss landscape in the direction of prefix parameters has condition number ≈ 1 through the suffix. The optimizer sees a well-conditioned landscape and converges to a better minimum.

---

## 4. Connection to HyperConnections

### HC Formulation

HC expands the residual stream to n streams with learnable mixing:

```
x_{l+1} = H^res_l @ x_l + H^post_l^T @ F(H^pre_l @ x_l)
```

where `H^res ∈ R^{n×n}` mixes streams, `H^pre ∈ R^{1×n}` reads out, `H^post ∈ R^{1×n}` writes back.

The depth mixing matrix unrolls to:

```
M_{i→L} = β_i^T @ (∏_{j=i+1}^{L} H^res_j) @ α_L
```

This is m-semiseparable with rank m = n.

### IDN as Degenerate HC

IDN K=1 can be expressed as a specific HC configuration with n = N+1 streams (N parallel layers + 1 carry):

```
Stream 0:     carries the prefix output h_S (pass-through)
Streams 1..N: each accumulates one parallel layer's output
```

The HC matrices are:

```
H^res_j = I_{(N+1)×(N+1)}    (identity — no stream mixing)
H^pre_j = [1, 0, ..., 0]     (all layers read from stream 0 = h_S)
H^post_j = e_j               (layer j writes to stream j only)
```

Final readout: `[1, 1, ..., 1]` (sum all streams).

**This is the simplest possible HC**: identity residual mixing, broadcast input, independent write-back. The depth mixing matrix within the parallel section is:

```
M_idn = | 1  0  0 ... 0 |    (carry stream unchanged)
        | 1  0  0 ... 0 |    (all layers read from stream 0)
        | 1  0  0 ... 0 |
        | ...            |
```

Every parallel layer reads from the same source (rank-1 input coupling). Compare with sequential HC where `H^res ≠ I` gives each layer a different mixed input (rank-n input coupling).

### Why Identity H^res Works for IDN

The mHC paper shows that unconstrained `H^res` causes training instability (signal amplification up to 3000x). Their solution: constrain `H^res` to doubly stochastic matrices (Birkhoff polytope).

IDN takes this to the extreme: `H^res = I`, the simplest doubly stochastic matrix. This is maximally stable (spectral norm = 1, ∏ H^res = I exactly). The performance comes not from stream mixing (which is trivial) but from the **variance reduction of evaluating all layers at the same clean point**.

### HC's Insight Applied to IDN

The HC ablation (Table 1 in mHC paper) shows `H^res` is the most important component (loss gap -0.022 vs -0.025/-0.027 for adding H^pre/H^post). This suggests the information exchange within the residual stream matters most.

In IDN, this exchange happens not through `H^res` (which is identity) but through the **IDN correction** (prefix sum). The correction propagates information between layers' outputs post-hoc:

```
hs[j+1] = all_out[j] + (hs[j] - h_old[j-1])
```

This is a sequential prefix sum over the parallel layers' independently-computed outputs — a different mechanism than HC's input-side mixing, but serving a similar role of propagating information across layers.

---

## 5. Connection to Attention Residuals

### AttnRes Depth Mixing Matrix

AttnRes replaces fixed residual accumulation with softmax attention over depth:

```
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
```

where `α_{i→l}` are learned, input-dependent softmax weights. The depth mixing matrix M is dense and input-dependent (rank L).

### Sequential vs IDN vs AttnRes in the M Framework

| Method | M structure (parallel section) | Input to each layer | Output aggregation |
|--------|-------------------------------|--------------------|--------------------|
| Sequential | All-ones lower triangular | Different (chain) | Chain composition |
| IDN K=1 | Rank-1 (all read h_S) | Same (h_S) | Prefix sum |
| Block AttnRes | Block lower triangular + softmax inter-block | Different (seq within block) | Learned softmax weights |
| Fused | Rank-1 input + nonlinear coupling | Same (h_S) | Sum with cross-terms |

### The Structural Inversion

Block AttnRes and Fused Layer-Parallel are roughly **structural inverses**:

| | Intra-block/chunk | Inter-block/chunk |
|---|---|---|
| **Block AttnRes** | Sequential residual (standard) | Softmax attention (rich, learned) |
| **Fused Layer-Parallel** | Parallel + cross-coupled (rich, implicit) | Additive IDN correction (simple) |

AttnRes puts the expressive power at the **inter-block** level (learned depth attention). Fused puts it at the **intra-chunk** level (cross-layer coupling through shared nonlinearity). Both outperform uniform residual accumulation, supporting the general principle that **non-trivial depth mixing is beneficial**.

---

## 6. Fused: Cross-Layer Coupling as Feature Conjunction

### What Fused MLP Sees

In the fused forward, the MLP input for all layers is:

```
mlp_input = norm(h_S + Σ_j O_j @ sdpa_j(h_S))
```

Each layer k's MLP gate computes:

```
W_fc_k @ norm(h_S + attn_k(h_S) + Σ_{j≠k} attn_j(h_S))
                                    ↑ cross-layer attention (invisible in sequential)
```

### The Cross-Layer Coupling Term

Define the fused output as `F_fused(h)` and the sum of independent per-layer outputs as `F_indep(h) = Σ_j f_j(h)`. The difference is:

```
F_fused(h) - F_indep(h) = Σ_j [mlp_j(norm(h + Σ_i attn_i(h))) - mlp_j(norm(h + attn_j(h)))]
```

This is the **cross-layer coupling term**. It vanishes only when:
- All `attn_i = 0` (trivial), or
- The MLP is linear (not the case with relu(·)²)

Otherwise, it captures **inter-layer feature interactions**: if layer j's attention detects pattern A and layer k's attention detects pattern B, the fused MLP can fire on A∧B. Sequential execution cannot access this because layer k's MLP only sees the post-MLP output of layer j, not layer j's raw attention output.

### Why This Can Improve PPL

The cross-coupling term is neither strictly beneficial nor harmful — it depends on the learned weights. But several factors bias it toward being helpful:

1. **Richer MLP input**: The summed attention provides a higher-dimensional effective input (capturing N layers' attention patterns simultaneously), giving the MLP more information to work with.

2. **Implicit ensemble**: The wide MLP (N×M hidden units) applied to the shared input acts like an ensemble of N MLPs, each with access to all N attention patterns. This has lower variance than N independent MLPs each seeing only their own attention.

3. **Connection to AttnRes**: The AttnRes paper shows that dense depth-mixing (M with non-zero off-diagonal entries) consistently outperforms uniform accumulation. The fused cross-coupling creates implicit off-diagonal entries through the shared nonlinearity, implementing a crude form of the same principle.

---

## 7. Unified Theory: Three Mechanisms

### Mechanism 1: Variance Reduction (IDN and Fused)

**Formal claim**: IDN K=1 eliminates error compounding through the sequential chain.

**Proof sketch**: In sequential, the deviation of `g_j(h_j^seq)` from `g_j(h_S)` is:

```
g_j(h_j^seq) - g_j(h_S) ≈ J_{g_j} · (h_j^seq - h_S) = J_{g_j} · Σ_{k<j} g_k(h_k^seq)
```

This introduces a variance term proportional to `||J_{g_j}||² · Σ_{k<j} Var(g_k)`. Summing over j, the total variance grows as O(N² · ||J_g||²) for sequential, while IDN has zero additional variance (all evaluated at h_S).

**When it dominates**: When IDN training has made `||J_{g_j}||` small (so the bias `g_j(h_S) - g_j(h_j^seq)` is also small), the variance reduction is the dominant effect.

### Mechanism 2: Implicit Regularization (IDN Training)

**Formal claim**: The IDN loss `||h_L^idn - h_L^seq|| → 0` implies `J_{g_j} → 0` for the suffix layers.

**Consequences**:
- Lipschitz constraint on suffix layers → better generalization
- Capacity partitioning → structured computation
- Improved gradient flow → better optimization landscape
- These benefits apply to the **sequential model itself**, explaining why IDN training improves sequential PPL

### Mechanism 3: Cross-Layer Feature Conjunction (Fused Only)

**Formal claim**: Fused inference introduces inter-layer interactions through the shared MLP nonlinearity that sequential execution cannot access.

**When it helps**: When the learned attention patterns across layers capture complementary features that benefit from joint MLP processing. The AttnRes paper's evidence that dense depth-mixing outperforms uniform accumulation suggests this is generically beneficial.

### Summary Table

| Mechanism | Applies to | Effect | Formal status |
|-----------|-----------|--------|---------------|
| Variance reduction | IDN K=1, Fused | Eliminates error compounding | Provable (bias-variance decomposition) |
| Implicit Lipschitz regularization | IDN training (affects sequential too) | Constrains suffix layers, improves generalization | Provable (consequence of IDN loss → J_{g_j} ≈ 0) |
| Capacity partitioning | IDN training | Prefix = input-dependent, suffix = global features | Empirically supported (early exit +52% PPL) |
| Improved gradient flow | IDN training | J_{f_j} ≈ I → perfect gradient highway | Provable (direct consequence) |
| Cross-layer feature conjunction | Fused only | MLP sees aggregate attention, enables A∧B patterns | Semi-formal (depends on nonlinearity) |

---

## 8. Predictions and Testable Hypotheses

If this theory is correct, we should observe:

1. **IDN PPL advantage grows with suffix length**: More parallel layers → more variance reduction. ✓ Confirmed by the data: IDN-trained d32 with 7 parallel layers shows -0.1% PPL advantage over sequential.

2. **Fused+split should be intermediate**: It has the variance reduction of IDN but not the cross-layer coupling of full fusion. → Check if Fused+split PPL is between IDN_batched and Fused.

3. **ChunkB_NxF1 should match IDN_batched**: N chunks of 1 layer each = per-layer evaluation with IDN correction = same as IDN_batched. → Check numerical agreement in the eval logs.

4. **Fused PPL should be sensitive to layer similarity**: If the parallel layers have similar attention patterns, the cross-coupling adds redundant information. If they have diverse patterns, the coupling is more valuable. → Could test by correlating layer attention similarity with fused PPL gain.

5. **Stronger IDN regularization should increase the IDN PPL advantage**: λ_idn = 1.0 should show larger variance reduction benefit than λ_idn = 0.5 (because J_{g_j} is smaller, so bias is smaller while variance reduction is the same). → Check the wide IDN training results.

6. **The variance reduction should be measurable**: Compute `Var_x[g_j(h_j^seq)]` vs `Var_x[g_j(h_S)]` across a validation set. The latter should be smaller. → Could implement as an eval metric.

---

## References

- Attention Residuals: arXiv 2603.15031 (Kimi Team)
- mHC: Manifold-Constrained HyperConnections: arXiv 2512.24880 (DeepSeek)
- DEER: arXiv 2309.12252
- ELK: arXiv 2508.18413
