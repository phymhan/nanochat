# SJD (Speculative Jacobi Decoding) — Development Log

Exploring SJD (arXiv:2512.07503) along the token/sequence axis, with the goal
of combining it with DEER layer-parallel inference.

Scripts: `scripts/demo_sjd_nanochat.py`, `scripts/demo_sjd_qwen.py`

## 1. Implementations

Four decoding modes in `demo_sjd_nanochat.py`:

### AR — baseline autoregressive
One forward pass per token. `model.forward([prefix]) → sample last logit`.

### JD — vanilla Jacobi Decoding (deterministic)
Inner loop iterates until fixed point. Always uses argmax (no randomness).
```
While not converged (up to max_jacobi_iters):
  Forward [prefix | drafts] → argmax predictions
  If draft[0..k] == prediction[0..k] (consecutive match):
    Accept k+1 tokens (matched + bonus), break
  Else: drafts ← predictions (Jacobi update), continue
```

### SJD — Speculative Jacobi Decoding (probabilistic)
No inner loop. One forward pass per step. Requires stochastic sampling.
Uses speculative acceptance criterion from the paper.

### SJD++ — SJD with confidence-based token reuse
Same as SJD but unaccepted tokens with high `p_new/p_old` ratio are kept
unchanged instead of being resampled.

## 2. SJD Code Walkthrough → Paper Mapping

The SJD `generate_sjd()` function (line ~225 in demo_sjd_nanochat.py):

### State between iterations
```python
drafts      # (1, n_lookahead) — current draft tokens
draft_dist  # (1, n_lookahead, V) — full p(x|ctx_{j-1}) from PREVIOUS iteration
```

### Per iteration (exactly one forward pass):

**Step 1 — Draft preparation** (paper Sec 3.1, Step 1):
```python
current_drafts = drafts[:, :n_draft]        # drafts from prev iter (warm-started)
current_dist = draft_dist[:, :n_draft, :]   # their probability distributions
```
Drafts come from: (a) previous iteration's model predictions (refinement/warm-start),
(b) tokens reused via SJD++ confidence check, or (c) random init for new positions.

**Step 2 — Parallel decoding** (paper Sec 3.1, Step 2):
```python
full_seq = torch.cat([ids, current_drafts], dim=1)   # [prefix | drafts]
logits = model.forward(full_seq)                       # one forward pass
new_probs = _get_probs(logits[:, T-1:T-1+n_draft, :]) # p(x|ctx_j) per position
next_tokens = sample(new_probs)                        # sampled predictions
```
`new_probs[j]` = model output at position T-1+j, predicting token at position T+j.
This is the "advanced" distribution. `next_tokens[j]` is a sample from it.

**Step 3 — Speculative verification** (paper Eq 2):
```python
for j in range(n_draft):
    tok = current_drafts[0, j]                      # draft token
    p_new = new_probs[0, j, tok]                    # p(tok | ctx_j)  [current iter]
    p_old = current_dist[0, j, tok]                 # p(tok | ctx_{j-1}) [prev iter]
    accept_prob = min(1.0, p_new / p_old)
    if rand() < accept_prob: n_accept += 1
    else: break   # first rejection → stop
```
The ratio `p_new/p_old` compares the same token under TWO consecutive Jacobi
iterations. Both come from the same model, just with different context (because
earlier tokens in the window may have changed). This is NOT standard SD with a
separate draft model — the "draft" is the previous iteration's prediction.

**Step 4a — First rejected position: calibrated resampling** (paper Eq 3):
```python
residual = (p_new_dist - p_old_dist).clamp(min=0)   # max(0, p_new - p_old)
bonus = sample(residual / residual.sum())             # resample from residual
```
This focuses sampling on tokens whose probability INCREASED in the new iteration.

**Step 4b — Remaining positions: refinement or reuse**:
- **SJD (Eq 5)**: `next_tokens[j]` already sampled from `p_new` → use directly
  as next iteration's draft. Store `new_probs[j]` as their distribution.
- **SJD++ (Eq 6)**: If `C_j = p_new(draft[j]) / p_old(draft[j]) > τ`, keep
  `draft[j]` unchanged (reuse). Otherwise use `next_tokens[j]`.

```python
for j in range(n_consumed, n_draft):
    if use_reuse and confidence > reuse_threshold:
        new_draft = current_drafts[:, j]   # reuse old draft
    else:
        new_draft = next_tokens[:, j]      # refined (sampled from p_new)
    new_draft_dist = new_probs[:, j, :]    # always update distribution
```

**Step 5 — Window shift**:
Accepted tokens → append to prefix.
Remaining drafts (refined/reused) + new random tokens → next iteration's drafts.

## 3. Results (nanochat d32 baseline, 128 tokens, temp=1.0, top_k=50)

Prompt: "Once upon a time in a land far away" (with CUDA warmup)

| Method | fwd | tok/pass | reused | tok/s |
|--------|-----|----------|--------|-------|
| AR | 128 | 1.00 | — | 56.5 |
| JD(la=16) | 124 | 1.03 | — | 80.5 |
| SJD(la=16) | 120 | 1.07 | 0 | 80.0 |
| SJD++(τ=0.0) | 119 | 1.08 | 907 | 78.4 |
| **SJD++(τ=0.1)** | **111** | **1.15** | **473** | **83.6** |
| SJD++(τ=0.3) | 125 | 1.02 | 317 | 74.5 |
| SJD++(τ=0.5) | 116 | 1.10 | 201 | 78.9 |
| SJD++(τ=1.0) | 118 | 1.08 | 94 | 77.6 |
| SJD++(τ=5.0) | 121 | 1.06 | 14 | 75.7 |

### Observations

1. **SJD++ τ=0.1 is the sweet spot**: 1.15 tok/pass, best throughput.

2. **τ=0.0 reuses too aggressively** (907 tokens) — includes bad tokens that
   fail the speculative check → slightly worse than τ=0.1.

3. **Higher τ → approaches SJD** behavior (fewer reuses, less benefit).

4. **JD and SJD are roughly equal** (~1.03-1.07 tok/pass on text) — the
   speculative sampling mechanism gives only marginal benefit because text
   tokens have high entropy.

5. **All methods faster than AR** in tok/s (56 vs 75-84) partly because the
   16 extra draft tokens add negligible compute (GPU memory-bound at batch=1),
   and partly because fewer forward passes reduces Python loop overhead.
   But this is not a fair comparison — AR should use KV cache for real speedup.

## 4. Text vs Image: Why Acceptance is Low

SJD gets 2-3x step compression on images but only ~1.07-1.15x on text:

- **Image codebook**: ~8K tokens. Many adjacent patches share similar tokens →
  drafts are often close to correct after refinement.
- **Text vocabulary**: 32K+ tokens. High per-token entropy. Even after
  refinement, the conditional distribution changes significantly when the
  context changes (strong long-range dependencies).
- **Stochastic sampling**: With top-k=50 on text, each position has ~50 viable
  tokens. The chance that a draft matches drops with vocabulary spread.
- **Spatial locality**: Images have 2D grid structure where neighbors correlate.
  Text is purely sequential with weaker local correlations.

## 5. Next Steps

1. **Test on Qwen** — update `demo_sjd_qwen.py` with proper SJD/SJD++
2. **KV-cache-aware SJD** — current impl recomputes full sequence each pass;
   with KV cache, only process draft tokens → faster per-pass
3. **Combine with DEER** — SJD reduces pass count, DEER speeds each pass
4. **Larger lookahead windows** — test la=32, 64, 96 (paper uses up to 96)
5. **Different prompts/tasks** — test on more diverse text to understand
   where SJD helps (repetitive text? code? structured output?)

## 6. Commands

```bash
# AR + JD + SJD + SJD++ (all modes)
CUDA_VISIBLE_DEVICES=0 NANOCHAT_BASE_DIR=cache uv run python -m scripts.demo_sjd_nanochat \
  --mode ar jd sjd sjd++ --max-new-tokens 128 --n-lookahead 16 \
  --temperature 1.0 --top-k 50 --reuse-threshold 0.1

# SJD++ threshold sweep
for tau in 0.0 0.1 0.3 0.5 1.0 2.0; do
  CUDA_VISIBLE_DEVICES=0 NANOCHAT_BASE_DIR=cache uv run python -m scripts.demo_sjd_nanochat \
    --mode sjd++ --max-new-tokens 128 --n-lookahead 16 \
    --temperature 1.0 --top-k 50 --reuse-threshold $tau
done
```
