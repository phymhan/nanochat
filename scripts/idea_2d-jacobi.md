# Research Plan: Combining Layer-DEER and SJD/Jacobi Decoding on a 2D Layer-Sequence Lattice

## Context

This repo explores accelerating LLM inference by applying the DEER framework along the **layer axis** of a Transformer. Standard Transformer inference evaluates layers sequentially:

\[
h_i = f_i(h_{i-1}), \quad i=1,\dots,L.
\]

Layer-DEER / Identity Newton tries to make some suffix layers executable in parallel by treating layer forwarding as a nonlinear recurrence and solving it with an iterative Newton-style update. In the current implementation, the most practical method is **Identity Newton (IDN)**, where the Jacobian is approximated as identity, reducing the Newton correction to cheap additions.

Separate from this, SJD / Jacobi decoding parallelizes along the **sequence axis**. Instead of decoding one token at a time strictly left-to-right, it proposes or refines multiple future tokens in parallel using Jacobi-style iteration.

The hypothesis to explore:

> Can we combine layer-DEER and SJD/Jacobi decoding into one iterative algorithm over a 2D lattice, with axes corresponding to layer and sequence position?

The lattice state is:

\[
h_{\ell,t}
\]

where:
- \(\ell\) is layer index,
- \(t\) is token position.

This creates a joint iterative refinement problem over the layer-token plane.

---

## Main Question

Can a joint layer-sequence parallel decoding algorithm give better latency or quality than simply composing the two methods naively?

The naive composition is:

```python
for jacobi_iter in range(J):
    # current guessed tokens x_1:T
    logits = model_forward_with_layer_deer(x_1:T, K_layer_iters)
    x_1:T = jacobi_update(logits)
```

That is:

\[
\text{cost} \approx J_{\text{token}} \times K_{\text{layer}}
\]

where each Jacobi decoding iteration contains a full layer-DEER solve.

A more coupled version is:

```python
for global_iter in range(R):
    update the whole layer-token plane h_{ell,t}
    read logits from h_{L,t}
    update token guesses x_t
```

This treats hidden states and token guesses as a single coupled fixed-point system.

---

## Important Clarification: What “Schedule” Means Here

Assume GPU is not saturated and batching/fusion always helps. Then we should **never** run parallelizable layer forwards sequentially.

So “schedule” should **not** mean choosing between sequential and parallel execution for operations that can already be batched.

Instead, in this project:

> A schedule means a rule for deciding which cells \((\ell,t)\) in the layer-token plane are recomputed at each global iteration, while the others reuse stale values or cheap approximations.

Full update:

\[
h_{\ell,t}^{r+1} = F_{\ell,t}(h^r, x^r)
\quad \forall \ell,t
\]

Sparse / scheduled update:

\[
h_{\ell,t}^{r+1}
=
\begin{cases}
F_{\ell,t}(h^r, x^r), & M_{\ell,t}^r = 1 \\
h_{\ell,t}^{r}, & M_{\ell,t}^r = 0
\end{cases}
\]

where \(M_{\ell,t}^r\) is an update mask.

So scheduling is essentially **stale-state reuse / early stopping / active-set refinement**, not intentionally serializing batched work.

---

## Baselines to Implement First

### Baseline 1: SJD/JD with standard model forward

This is the existing SJD baseline.

```python
for j in range(J):
    logits = model(x_guess)
    x_guess = jacobi_update(logits)
```

Measure:
- accepted tokens per iteration
- wall-clock latency
- PPL / downstream quality
- number of model forwards

---

### Baseline 2: SJD/JD with layer-DEER model forward

Replace model forward inside SJD with layer-DEER / IDN.

```python
for j in range(J):
    logits = layer_deer_model(x_guess, K=1)
    x_guess = jacobi_update(logits)
```

This is the clean naive integration.

If IDN gives \(K=1\), this should be the first strong baseline.

Measure:
- whether SJD acceptance changes
- whether layer-parallel speedup survives in SJD setting
- whether hidden/logit mismatch hurts token acceptance

---

### Baseline 3: Coupled full-plane update

Instead of nested loops, do one global iteration over the 2D state:

```python
for r in range(R):
    update all h_{ell,t}
    logits_t = lm_head(h_{L,t})
    x_t = jacobi_update(logits_t)
```

This updates the entire \(L \times T\) layer-token plane every iteration.

This is not necessarily faster than Baseline 2, but it is conceptually important because it tests whether layer and token refinement can be coupled rather than nested.

---

## Research Direction 1: Full-Plane Coupled Iteration

### Goal

Test whether updating the layer-token plane jointly has any advantage over nested SJD + layer-DEER.

### Hypothesis

The coupled update may allow layer errors and token errors to co-evolve and correct each other. This may reduce the number of outer Jacobi iterations or improve acceptance length.

### Key Question

Does coupled iteration reduce:

\[
J_{\text{token}} \times K_{\text{layer}}
\]

to a smaller number of global iterations \(R\)?

### Experiments

Use short token windows first:
- \(T = 4, 8, 16\)
- small nanochat model
- IDN-trained checkpoints
- baseline checkpoints

Compare:
- standard SJD
- SJD + IDN model forward
- coupled full-plane iteration

Metrics:
- accepted tokens per global iteration
- final PPL
- token top-1 agreement with sequential
- hidden-state error at final layer
- wall-clock latency
- memory

---

## Research Direction 2: Sparse / Active-Set Scheduling

### Goal

After full-plane update is implemented, test whether we can reduce computation by updating only selected cells.

A sparse schedule updates a subset of the layer-token plane:

\[
M_{\ell,t}^r \in \{0,1\}
\]

Skipped cells reuse stale states:

\[
h_{\ell,t}^{r+1} = h_{\ell,t}^{r}
\]

### Candidate Schedules

#### 2.1 Token-sparse schedule

Update all layers, but only for uncertain token positions.

```python
S_r = select_uncertain_tokens(logits, entropy, changed_tokens)
for t in S_r:
    update all layers for token t
for t not in S_r:
    reuse stale hidden states
```

Criteria:
- token changed in previous Jacobi iteration
- high entropy
- low confidence
- disagreement between draft and verifier
- high final-layer residual

This is closest to early stopping over token positions.

---

#### 2.2 Layer-sparse schedule

Update all tokens, but only selected layer chunks.

```python
C_r = select_layer_chunks(residuals, idn_safety)
for c in C_r:
    update layer chunk c for all tokens
for c not in C_r:
    reuse stale states
```

Possible rules:
- update IDN-unsafe layers more often
- update IDN-safe layers less often
- shallow-to-deep refinement
- late-layer-only refinement after tokens stabilize

---

#### 2.3 Diagonal / wavefront schedule

Update diagonal bands in the layer-token plane.

For example:

\[
\ell + t = r
\]

or chunked version:

\[
\text{layer_chunk} + \text{token_chunk} = r.
\]

This is useful only if the computation has wavefront-like dependency propagation. It is not automatically better than full-plane update.

The main thing to test:

> Does wavefront update reduce wasted computation on deep layers for token guesses that are still unstable?

---

#### 2.4 Residual active-set schedule

Update cells whose residual is large:

\[
M_{\ell,t}^r = 1
\quad \text{if} \quad
\|h_{\ell,t}^r - F_{\ell,t}(h^r, x^r)\| > \tau.
\]

This is the most principled sparse schedule, but likely hardest to implement efficiently.

---

## Research Direction 3: What Is the Right 2D Fixed-Point Formulation?

### Option A: Nested formulation

Token update is outside layer update.

\[
x^{j+1} = \mathrm{JacobiUpdate}(\mathrm{LayerDEER}(x^j))
\]

This is simple and modular.

### Option B: Coupled hidden-state formulation

Hidden states and tokens are both part of the state:

\[
\mathcal{S}^r = \{h_{\ell,t}^r, x_t^r\}_{\ell,t}
\]

A global update is:

\[
\mathcal{S}^{r+1} = \Phi(\mathcal{S}^r).
\]

This is more like iterative latent inference.

### Option C: Logit-only coupled formulation

Do not maintain all hidden states across iterations. Only keep token guesses/logits, and call layer-DEER as a black-box approximate model forward.

This is closest to practical SJD.

### Recommendation

Implement in this order:

1. Option C: easiest, practical baseline.
2. Option A: nested DEER + Jacobi with explicit control of \(K\) and \(J\).
3. Option B: full 2D latent-state formulation only if early results motivate it.

---

## Research Direction 4: Sequence-Axis IDN Regularization

### Motivation

Layer-axis IDN regularization makes layers robust to stale / shared layer inputs.

Question:

> Can a similar regularization make token positions robust to stale / guessed sequence context?

If yes, this may improve SJD/Jacobi decoding.

### Candidate Losses

#### 4.1 Logit consistency loss

Run normal AR teacher-forced forward and parallel Jacobi-style forward. Match logits:

\[
\mathcal{L}_{seq} =
\sum_t \mathrm{KL}(p_t^{seq} \| p_t^{parallel})
\]

#### 4.2 Hidden-state consistency loss

Match hidden states:

\[
\mathcal{L}_{seq-hid} =
\sum_t
\frac{\|\hat h_t - h_t^{seq}\|}{\|h_t^{seq}\| + \epsilon}
\]

#### 4.3 First-error / accepted-prefix loss

Optimize directly for Jacobi/SJD behavior:

- improve accepted prefix length
- reduce first rejection probability
- improve verifier agreement

### Experiments

Train variants:
- baseline CE only
- layer-IDN only
- sequence-IDN only
- layer-IDN + sequence-IDN

Evaluate:
- AR PPL
- SJD accepted length
- Jacobi iterations to convergence
- generation speed
- downstream quality

Important risk:
- sequence-axis regularization may hurt causal modeling more than layer-axis regularization. Start with short spans \(T=2,4,8\).

---

## Research Direction 5: Separable Local Linearization on the 2D Plane

### Motivation

The full Jacobian over the layer-token plane is enormous:

\[
J \in \mathbb{R}^{(L T d) \times (L T d)}.
\]

A possible approximation is that layer coupling and sequence coupling are partially separable:

\[
J \approx J_{layer} \otimes I_{seq} + I_{layer} \otimes J_{seq}.
\]

Or in local form:

\[
\Delta h_{\ell,t}
\approx
A_\ell \Delta h_{\ell-1,t}
+
B_t \Delta h_{\ell,t-1}.
\]

For IDN-trained models, maybe:

\[
A_\ell \approx I
\]

so the main remaining coupling is sequence-axis coupling.

### Questions

1. Is the layer-token Jacobian approximately separable?
2. Does IDN reduce the layer-axis coupling enough that only sequence coupling matters?
3. Can we design cheap 2D correction rules using this structure?

### Diagnostics

Estimate via random probes:
- layer-axis sensitivity
- sequence-axis sensitivity
- cross layer-sequence interaction
- residual norm per cell
- diagonal dominance / block diagonal dominance

Possible test:

\[
\Delta h_{\ell,t}^{actual}
\approx
\Delta h_{\ell-1,t} + B_t \Delta h_{\ell,t-1}
\]

If this approximation is good, then IDN + sequence correction may be enough.

---

## Research Direction 6: Early Stopping and Compute-Quality Tradeoff

### Motivation

Pure layer-DEER converged to completion only recovers the sequential model. That is an acceleration trick, not inference-time scaling.

But early-stopped iterative states may still be useful. This is analogous to parallel MCMC, where intermediate Newton iterates may already produce useful approximate samples.

### Questions

1. Are intermediate 2D iterates useful before convergence?
2. Can early stopping produce acceptable tokens faster?
3. Can extra iterations improve quality if the model is trained for it?
4. Can we expose a compute-quality knob?

### Experiments

For nested and coupled algorithms, sweep:
- number of global iterations \(R\)
- number of layer-DEER iterations \(K\)
- number of Jacobi iterations \(J\)
- sparse schedule thresholds

Measure:
- PPL
- accepted tokens
- reasoning accuracy
- latency
- logit convergence
- hidden residual

Important distinction:
- If training only matches sequential forward, extra iterations cannot exceed sequential model quality in principle.
- To get true inference-time scaling, train the iterative process itself, e.g. variable-iteration training or recurrent-depth style training.

---

## Research Direction 7: Speed Model

### Idealized Layer Speedup

If batched/fused operation for \(N\) parallel layers has the same latency as one layer, then the parallelized suffix gets \(N\times\) speedup.

But full-model speedup is limited by the sequential prefix:

\[
\text{speedup} = \frac{L}{S + K}
\]

where:
- \(L\) = total layers,
- \(S\) = sequential prefix layers,
- \(K\) = number of layer-DEER iterations,
- assume fused suffix latency equals one layer latency.

For \(L=32\), \(N=24\), \(S=8\), \(K=1\):

\[
\text{speedup} = \frac{32}{9} \approx 3.56\times.
\]

The suffix itself has \(24\times\) speedup, but full-model speedup is only \(3.56\times\).

### For 2D SJD + Layer-DEER

If SJD needs \(J\) iterations and each forward is layer-parallel:

\[
T_{nested} \approx J(S + K)\tau.
\]

If coupled 2D iteration needs \(R\) global iterations:

\[
T_{coupled} \approx R \cdot T_{2D-update}.
\]

The coupled method only wins if:

\[
R \cdot T_{2D-update} < J(S+K)\tau.
\]

Sparse scheduling only matters if it reduces \(T_{2D-update}\) or reduces \(R\).

---

## Implementation Notes

### Existing files to inspect

- `scripts/demo_sjd_nanochat.py`
- `scripts/dev_sjd.md`
- `scripts/eval_jacobi_d32s_4800.md`
- `scripts/eval_jacobi_d32s_9600.md`
- `scripts/dev_eval_jacobi_idn.md`
- `scripts/eval_jacobi_idn.md`
- `report.typ`
- `report.pdf`
- `reference/elk`
- `reference/Accelerating-T2I-AR-with-SJD`

### First implementation target

Modify `scripts/demo_sjd_nanochat.py` to support a model-forward backend flag:

```bash
--forward_backend sequential
--forward_backend idn_layer_deer
--forward_backend coupled_2d
```

Start with:

```bash
--forward_backend idn_layer_deer
```

Then compare against existing SJD.

### Useful knobs

- `n_parallel_layers`
- `idn_K`
- `jacobi_iters`
- `token_window`
- `schedule_type`
- `active_threshold`
- `use_stale_hidden`
- `use_layer_cache`
- `use_token_cache`

---

## Suggested Experiment Order

### Experiment 1: Naive composition

Run SJD with sequential forward vs SJD with IDN/layer-DEER forward.

Goal:
- verify the simplest combination works
- measure speed and acceptance

---

### Experiment 2: Layer-DEER mismatch effect

For the same token guesses, compare logits from:
- sequential model
- IDN model
- diag model, if available

Measure:
- KL divergence
- top-1 agreement
- effect on SJD accepted length

---

### Experiment 3: Coupled full-plane prototype

Implement a small-scale coupled update for short token windows.

Goal:
- determine whether hidden-state coupling gives any benefit over nested loops

---

### Experiment 4: Token-sparse update

Use final-layer confidence to skip updates for stable token positions.

Goal:
- test whether sparse scheduling saves compute without hurting accepted length

---

### Experiment 5: Sequence-axis IDN training

Add sequence consistency regularization and evaluate Jacobi/SJD convergence.

Goal:
- test whether the IDN principle generalizes from layer axis to sequence axis

---

## Go / No-Go Criteria

### Continue 2D coupled iteration if:

- coupled full-plane update reduces global iterations vs nested method, or
- sparse schedules preserve quality with fewer cell updates, or
- sequence-axis IDN improves SJD acceptance, or
- hidden-state residuals show exploitable structure across the 2D plane.

### Stop or deprioritize if:

- SJD + IDN model forward already dominates coupled methods,
- coupled state maintenance adds too much memory overhead,
- sparse schedules hurt token acceptance too much,
- sequence-axis regularization hurts AR PPL significantly.

---

## Most Likely Paper Framing

The strongest framing may be:

> Layer-parallel IDN and token-parallel Jacobi decoding are two orthogonal relaxations of autoregressive Transformer inference. Their combination naturally defines a 2D layer-token iterative inference problem. The practical question is whether one should solve this problem by naive nesting, full-plane coupled iteration, or adaptive sparse refinement.

Potential contributions:

1. Formalize the 2D layer-token lattice view.
2. Show naive SJD + layer-IDN gives multiplicative or partially multiplicative speed benefits.
3. Analyze why full-plane coupling helps or does not help.
4. Introduce active-set / stale-state scheduling if useful.
5. Explore sequence-axis IDN regularization as a training-time co-design principle.

---

## Main Warnings

1. Do not call scheduling “choosing sequential vs parallel execution.” In this setting, parallel execution is always preferred when possible.
2. Schedule means sparse/stale update over the layer-token plane.
3. Full-plane update is the clean baseline. Sparse schedules only matter if full-plane update wastes compute.
4. Diagonal + scan is likely not useful if IDN already gives lossless PPL and lower cost.
5. Do not claim inference-time scaling unless extra iterations improve quality beyond simply recovering the sequential forward.
6. For true inference-time scaling, the iterative process must be trained as part of the model, e.g. variable-iteration or recurrent-depth style training.
