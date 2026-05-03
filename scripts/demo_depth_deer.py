"""
For diagonosis purpose, for demo see demo_depth_quasi_deer_qwen.py

demo_depth_deer.py — Layer-parallel inference via DEER for Qwen2.5 models.

Demonstrates Newton's method variants for solving the layer fixed-point problem
along the depth (layer) axis. Key concepts:

  Fixed-point problem: G(h) = h - F(h) = 0, where F_i(h) = f_i(h_{i-1})
  Newton update:       h^{k+1} = h^k - J_G^{-1} G(h^k)
  Forward substitution (bidiagonal J_G):
      δ_0 = r_0
      δ_i = r_i + J_i · δ_{i-1}     (J_i = ∂f_i/∂h_{i-1})
      h^{new} = h - δ

Methods:
  - Jacobi:          h_i^{k+1} = f_i(h_{i-1}^k)  [no Jacobian, diverges for σ > 1]
  - FD-Newton:       J·v ≈ (f(x + εv) - f(x)) / ε  [2 fwd per layer per iter]
  - VJP-Newton:      Jᵀ·v via backward AD  [~3.6x fwd cost, uses transpose J]
  - JVP-Newton:      J·v via forward AD (autograd_jvp)  [needs eager attention]
  - Identity Newton: J ≈ I, δ_i = r_i + δ_{i-1}  [prefix sum, zero extra cost]

Usage:
  # Compare all methods (seq_layers=20, 4 parallel layers)
  python scripts/demo_depth_deer.py --seq-layers 20 --iters 6

  # Just FD-Newton
  python scripts/demo_depth_deer.py --method fd-newton --seq-layers 20 --iters 6

  # Verify JVP correctness with FD
  python scripts/demo_depth_deer.py --verify-jvp

  # Generation mode
  python scripts/demo_depth_deer.py --mode generate --method fd-newton --seq-layers 20
"""

import argparse
import os
import time
from typing import Optional

import torch
from torch.autograd.functional import jvp as autograd_jvp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Model helper
# ---------------------------------------------------------------------------

class LayerRunner:
    """Wraps a HF causal LM to expose individual layer calls."""

    def __init__(self, model):
        self.hf_model = model
        self.inner = model.model  # e.g. Qwen2Model
        self.lm_head = model.lm_head
        self.config = model.config

    @property
    def device(self):
        return next(self.hf_model.parameters()).device

    @property
    def num_layers(self):
        return self.config.num_hidden_layers

    def embed(self, input_ids):
        """Compute embeddings, position info, and causal mask (no KV cache)."""
        inputs_embeds = self.inner.embed_tokens(input_ids)
        cache_position = torch.arange(
            input_ids.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.inner.rotary_emb(inputs_embeds, position_ids)

        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        return inputs_embeds, causal_mask, position_ids, cache_position, position_embeddings

    def embed_with_cache(self, input_ids, past_key_values, cache_position=None):
        """Compute embeddings with KV cache context."""
        inputs_embeds = self.inner.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values else 0
            cache_position = torch.arange(
                past_seen, past_seen + input_ids.shape[1], device=inputs_embeds.device
            )
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.inner.rotary_emb(inputs_embeds, position_ids)

        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        return inputs_embeds, causal_mask, position_ids, cache_position, position_embeddings

    def run_layer(self, i, hidden_states, causal_mask, position_ids,
                  cache_position, position_embeddings,
                  past_key_values=None, use_cache=False):
        """Run decoder layer i."""
        layer = self.inner.layers[i]
        return layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    def to_logits(self, hidden_states):
        """Final norm + lm_head."""
        hidden_states = self.inner.norm(hidden_states)
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Forward methods (all operate on full sequences, no KV cache)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def sequential_forward(runner: LayerRunner, input_ids):
    """Ground truth: run all layers sequentially."""
    x, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)
    intermediates = [x]
    for i in range(runner.num_layers):
        x = runner.run_layer(i, x, mask, pos_ids, cache_pos, pos_emb)
        intermediates.append(x)
    logits = runner.to_logits(x)
    return logits, intermediates


def _run_sequential_prefix(runner, x0, mask, pos_ids, cache_pos, pos_emb,
                           seq_layers, n):
    """Run sequential prefix, init parallel suffix. Returns h[0..n]."""
    h = [None] * (n + 1)
    h[0] = x0
    x = x0
    for i in range(seq_layers):
        x = runner.run_layer(i, x, mask, pos_ids, cache_pos, pos_emb)
        h[i + 1] = x
    for i in range(seq_layers, n):
        h[i + 1] = x.clone()
    return h


def _compute_diagnostics(h_new, h_old, seq_layers, n):
    """Compute max relative delta over parallel layers."""
    deltas = []
    for i in range(seq_layers, n):
        delta = (h_new[i+1] - h_old[i+1]).float().norm() / (h_old[i+1].float().norm() + 1e-8)
        deltas.append(delta.item())
    return max(deltas) if deltas else 0.0


@torch.inference_mode()
def jacobi_forward(runner, input_ids, seq_layers, max_iters):
    """
    Hybrid sequential/Jacobi forward.

    Jacobi: h_i^{k+1} = f_i(h_{i-1}^k)
    Zeroth-order, no Jacobian. Diverges when σ_max > 1.
    """
    x0, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)
    n = runner.num_layers
    h = _run_sequential_prefix(runner, x0, mask, pos_ids, cache_pos, pos_emb, seq_layers, n)

    diagnostics = []
    for k in range(max_iters):
        h_new = list(h)
        for i in range(seq_layers, n):
            h_new[i + 1] = runner.run_layer(i, h[i], mask, pos_ids, cache_pos, pos_emb)
        diagnostics.append(_compute_diagnostics(h_new, h, seq_layers, n))
        h = h_new

    logits = runner.to_logits(h[n])
    return logits, h, diagnostics


@torch.inference_mode()
def newton_fd_forward(runner, input_ids, seq_layers, max_iters, eps=1e-3):
    """
    Hybrid sequential + FD-Newton.

    Newton forward substitution:
        δ_i = r_i + J_i · δ_{i-1}
    where J_i · v ≈ (f_i(x + εv) - f_i(x)) / ε.

    Cost: 2 layer evals per parallel layer per iteration (one for F, one for FD).
    """
    x0, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)
    n = runner.num_layers
    h = _run_sequential_prefix(runner, x0, mask, pos_ids, cache_pos, pos_emb, seq_layers, n)

    def fwd(i, h_in):
        return runner.run_layer(i, h_in, mask, pos_ids, cache_pos, pos_emb)

    diagnostics = []
    for k in range(max_iters):
        # Step 1: Evaluate all parallel layers
        F_h = {i: fwd(i, h[i]) for i in range(seq_layers, n)}

        # Step 2: Residuals
        residuals = {i: h[i+1] - F_h[i] for i in range(seq_layers, n)}

        # Step 3: Forward substitution with FD-JVP
        corrections = {}
        corrections[seq_layers] = residuals[seq_layers]

        for i in range(seq_layers + 1, n):
            c_prev = corrections[i - 1]
            if c_prev.float().norm().item() < 1e-8:
                corrections[i] = residuals[i]
            else:
                F_pert = fwd(i, h[i] + eps * c_prev)
                jvp_approx = (F_pert - F_h[i]) / eps
                corrections[i] = residuals[i] + jvp_approx

        # Step 4: Apply corrections
        h_new = list(h)
        for i in range(seq_layers, n):
            h_new[i + 1] = h[i + 1] - corrections[i]

        diagnostics.append(_compute_diagnostics(h_new, h, seq_layers, n))
        h = h_new

    logits = runner.to_logits(h[n])
    return logits, h, diagnostics


def newton_vjp_forward(runner, input_ids, seq_layers, max_iters):
    """
    Hybrid sequential + VJP-Newton (transpose Newton via backward AD).

    Uses Jᵀ·v instead of J·v in the forward substitution.
    Works with any attention implementation (SDPA, Flash, eager).
    Empirically converges ~25% faster than FD for non-symmetric Jacobians.

    Cost: ~3.6x forward per layer per iter (backward pass, measured on Qwen2.5-0.5B).
    """
    x0, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)
    n = runner.num_layers

    def fwd(i, h_in):
        return runner.run_layer(i, h_in, mask, pos_ids, cache_pos, pos_emb)

    def vjp(i, h_in, v):
        """Compute Jᵀ·v via backward AD."""
        with torch.enable_grad():
            h_g = h_in.detach().requires_grad_(True)
            out = runner.inner.layers[i](
                h_g,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_pos,
                position_embeddings=pos_emb,
            )
            grad = torch.autograd.grad(out, h_g, grad_outputs=v.detach())[0]
        return grad.detach()

    # Sequential prefix
    h = [None] * (n + 1)
    h[0] = x0
    x = x0
    with torch.inference_mode():
        for i in range(seq_layers):
            x = fwd(i, x)
            h[i + 1] = x
    for i in range(seq_layers, n):
        h[i + 1] = x.clone()

    diagnostics = []
    for k in range(max_iters):
        with torch.no_grad():
            F_h = {i: fwd(i, h[i]) for i in range(seq_layers, n)}
            residuals = {i: h[i+1] - F_h[i] for i in range(seq_layers, n)}

        corrections = {}
        corrections[seq_layers] = residuals[seq_layers]

        for i in range(seq_layers + 1, n):
            c_prev = corrections[i - 1]
            if c_prev.float().norm().item() < 1e-8:
                corrections[i] = residuals[i]
            else:
                Jtv = vjp(i, h[i], c_prev)
                corrections[i] = residuals[i] + Jtv

        with torch.no_grad():
            h_new = list(h)
            for i in range(seq_layers, n):
                h_new[i + 1] = h[i + 1] - corrections[i]
            diagnostics.append(_compute_diagnostics(h_new, h, seq_layers, n))
            h = h_new

    with torch.inference_mode():
        logits = runner.to_logits(h[n])
    return logits, h, diagnostics


def newton_jvp_forward(runner, input_ids, seq_layers, max_iters):
    """
    Hybrid sequential + exact JVP-Newton (forward-mode AD via autograd_jvp).

    Uses exact J·v computation. Requires eager attention for forward AD.

    Cost: ~6.1x forward per layer per iter (dual numbers include primal; measured on Qwen2.5-0.5B).
    """
    x0, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)
    n = runner.num_layers

    def fwd(i, h_in):
        return runner.run_layer(i, h_in, mask, pos_ids, cache_pos, pos_emb)

    def make_layer_fn(i):
        """Pure function for layer i (no side effects)."""
        layer = runner.inner.layers[i]
        def fn(x):
            return layer(
                x,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_pos,
                position_embeddings=pos_emb,
            )
        return fn

    def exact_jvp(i, h_in, v):
        """Compute J·v via forward-mode AD."""
        fn = make_layer_fn(i)
        with torch.enable_grad():
            x = h_in.detach().requires_grad_(True)
            _, Jv = autograd_jvp(fn, (x,), (v.detach(),),
                                 create_graph=False, strict=False)
        return Jv.detach()

    # Sequential prefix
    h = [None] * (n + 1)
    h[0] = x0
    x = x0
    with torch.inference_mode():
        for i in range(seq_layers):
            x = fwd(i, x)
            h[i + 1] = x
    for i in range(seq_layers, n):
        h[i + 1] = x.clone()

    diagnostics = []
    for k in range(max_iters):
        with torch.no_grad():
            F_h = {i: fwd(i, h[i]) for i in range(seq_layers, n)}
            residuals = {i: h[i+1] - F_h[i] for i in range(seq_layers, n)}

        corrections = {}
        corrections[seq_layers] = residuals[seq_layers]

        for i in range(seq_layers + 1, n):
            c_prev = corrections[i - 1]
            if c_prev.float().norm().item() < 1e-8:
                corrections[i] = residuals[i]
            else:
                Jv = exact_jvp(i, h[i], c_prev)
                corrections[i] = residuals[i] + Jv

        with torch.no_grad():
            h_new = list(h)
            for i in range(seq_layers, n):
                h_new[i + 1] = h[i + 1] - corrections[i]
            diagnostics.append(_compute_diagnostics(h_new, h, seq_layers, n))
            h = h_new

    with torch.inference_mode():
        logits = runner.to_logits(h[n])
    return logits, h, diagnostics


@torch.inference_mode()
def newton_identity_forward(runner, input_ids, seq_layers, max_iters=1):
    """
    Identity Newton: J ≈ I.

    Forward substitution becomes a prefix sum:
        δ_i = r_i + δ_{i-1}
    Zero JVP cost. Equivalent to running all parallel layers on the
    sequential prefix output, then applying additive correction.

    On a standard model (no IDN training), this is essentially
    "all-on-same-input" with a prefix-sum correction.
    """
    x0, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)
    n = runner.num_layers
    h = _run_sequential_prefix(runner, x0, mask, pos_ids, cache_pos, pos_emb, seq_layers, n)

    def fwd(i, h_in):
        return runner.run_layer(i, h_in, mask, pos_ids, cache_pos, pos_emb)

    diagnostics = []
    for k in range(max_iters):
        F_h = {i: fwd(i, h[i]) for i in range(seq_layers, n)}
        residuals = {i: h[i+1] - F_h[i] for i in range(seq_layers, n)}

        # Prefix sum: δ_i = r_i + δ_{i-1}
        corrections = {}
        corrections[seq_layers] = residuals[seq_layers]
        for i in range(seq_layers + 1, n):
            corrections[i] = residuals[i] + corrections[i - 1]

        h_new = list(h)
        for i in range(seq_layers, n):
            h_new[i + 1] = h[i + 1] - corrections[i]

        diagnostics.append(_compute_diagnostics(h_new, h, seq_layers, n))
        h = h_new

    logits = runner.to_logits(h[n])
    return logits, h, diagnostics


# ---------------------------------------------------------------------------
# Generation with KV cache
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_sequential(runner, input_ids, max_new_tokens, eos_token_id=None):
    """Standard autoregressive generation (baseline)."""
    cache = DynamicCache(config=runner.config)
    n = runner.num_layers

    # Prefill
    x, mask, pos_ids, cache_pos, pos_emb = runner.embed_with_cache(input_ids, cache)
    for i in range(n):
        x = runner.run_layer(i, x, mask, pos_ids, cache_pos, pos_emb,
                             past_key_values=cache, use_cache=True)
    logits = runner.to_logits(x)
    next_token = logits[:, -1:].argmax(dim=-1)
    out = torch.cat([input_ids, next_token], dim=1)

    # Decode
    for _ in range(max_new_tokens - 1):
        x, mask, pos_ids, cache_pos, pos_emb = runner.embed_with_cache(next_token, cache)
        for i in range(n):
            x = runner.run_layer(i, x, mask, pos_ids, cache_pos, pos_emb,
                                 past_key_values=cache, use_cache=True)
        logits = runner.to_logits(x)
        next_token = logits[:, -1:].argmax(dim=-1)
        out = torch.cat([out, next_token], dim=1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return out


@torch.inference_mode()
def generate_newton(runner, input_ids, max_new_tokens, seq_layers,
                    newton_iters=4, method="fd", eps=1e-3, eos_token_id=None):
    """
    Autoregressive generation with Newton decode.

    Prefill: sequential (populate KV cache correctly).
    Decode: sequential prefix layers use cache, then Newton on suffix.

    For the suffix during decode, we DON'T use KV cache (layers are called
    without cache, since their hidden states haven't converged yet). After
    Newton converges, we do a final sequential pass on the suffix to populate
    the cache.
    """
    cache = DynamicCache(config=runner.config)
    n = runner.num_layers

    # Prefill: full sequential with cache
    x, mask, pos_ids, cache_pos, pos_emb = runner.embed_with_cache(input_ids, cache)
    for i in range(n):
        x = runner.run_layer(i, x, mask, pos_ids, cache_pos, pos_emb,
                             past_key_values=cache, use_cache=True)
    logits = runner.to_logits(x)
    next_token = logits[:, -1:].argmax(dim=-1)
    out = torch.cat([input_ids, next_token], dim=1)

    # Decode with Newton
    for _ in range(max_new_tokens - 1):
        x, mask, pos_ids, cache_pos, pos_emb = runner.embed_with_cache(next_token, cache)

        # Phase 1: Sequential prefix with cache
        h = x
        for i in range(seq_layers):
            h = runner.run_layer(i, h, mask, pos_ids, cache_pos, pos_emb,
                                 past_key_values=cache, use_cache=True)
        seq_output = h

        # Save suffix cache state before Newton iterations
        n_suffix = n - seq_layers
        saved_suffix = [(cache.layers[i].keys.clone(), cache.layers[i].values.clone())
                        for i in range(seq_layers, n)]

        def _restore_suffix_cache():
            for j, i in enumerate(range(seq_layers, n)):
                cache.layers[i].keys = saved_suffix[j][0].clone()
                cache.layers[i].values = saved_suffix[j][1].clone()

        # Phase 2: Newton on suffix (crop-and-retry for cache)
        hs = [seq_output] + [seq_output.clone() for _ in range(n_suffix)]

        for k in range(newton_iters):
            _restore_suffix_cache()

            F_h = {}
            for j, i in enumerate(range(seq_layers, n)):
                F_h[j] = runner.run_layer(i, hs[j], mask, pos_ids, cache_pos, pos_emb,
                                          past_key_values=cache, use_cache=True)

            residuals = {j: hs[j+1] - F_h[j] for j in range(n_suffix)}
            corrections = {0: residuals[0]}

            for j in range(1, n_suffix):
                c_prev = corrections[j - 1]
                i = seq_layers + j
                if c_prev.float().norm().item() < 1e-8:
                    corrections[j] = residuals[j]
                elif method == "fd":
                    _restore_suffix_cache()
                    F_pert = runner.run_layer(i, hs[j] + eps * c_prev,
                                             mask, pos_ids, cache_pos, pos_emb,
                                             past_key_values=cache, use_cache=True)
                    corrections[j] = residuals[j] + (F_pert - F_h[j]) / eps
                else:  # identity
                    corrections[j] = residuals[j] + c_prev

            for j in range(n_suffix):
                hs[j + 1] = hs[j + 1] - corrections[j]

        # Phase 3: Final sequential pass on suffix to populate cache correctly
        _restore_suffix_cache()
        h = seq_output
        for i in range(seq_layers, n):
            h = runner.run_layer(i, h, mask, pos_ids, cache_pos, pos_emb,
                                 past_key_values=cache, use_cache=True)

        logits = runner.to_logits(h)
        next_token = logits[:, -1:].argmax(dim=-1)
        out = torch.cat([out, next_token], dim=1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def verify_jvp(runner, input_ids, layer_idx=None):
    """
    Verify that autograd_jvp gives correct JVP via Taylor test.

    If JVP is correct:  ||f(x+εv) - f(x) - ε·Jv|| = O(ε²)
    If JVP is broken:   ||f(x+εv) - f(x) - ε·Jv|| = O(ε)  (same as zeroth order)
    """
    # Prepare inputs outside inference mode (needed for autograd)
    with torch.no_grad():
        x0, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)

    if layer_idx is None:
        layer_idx = runner.num_layers // 2

    # Get input to the target layer via sequential forward
    x = x0
    with torch.no_grad():
        for i in range(layer_idx):
            x = runner.run_layer(i, x, mask, pos_ids, cache_pos, pos_emb)
    h_in = x.detach().clone()

    # Random tangent direction
    v = torch.randn_like(h_in)
    v = v / v.norm()

    layer = runner.inner.layers[layer_idx]
    def fn(x):
        return layer(x, attention_mask=mask, position_ids=pos_ids,
                     past_key_values=None, use_cache=False,
                     cache_position=cache_pos, position_embeddings=pos_emb)

    # Baseline: f(x)
    with torch.no_grad():
        f_x = fn(h_in).float()

    # Exact JVP via autograd_jvp
    with torch.enable_grad():
        x_g = h_in.clone().requires_grad_(True)
        _, exact_jvp = autograd_jvp(fn, (x_g,), (v.clone(),),
                                    create_graph=False, strict=False)
    exact_jvp = exact_jvp.detach().float()

    # VJP (Jᵀ·v) via backward AD
    with torch.enable_grad():
        x_g = h_in.clone().requires_grad_(True)
        out = fn(x_g)
        vjp_result = torch.autograd.grad(out, x_g, grad_outputs=v.clone())[0]
    vjp_result = vjp_result.detach().float()

    # Taylor test: compare zeroth-order and first-order error at decreasing ε
    # zeroth = ||f(x+εv) - f(x)||  should be O(ε)
    # first  = ||f(x+εv) - f(x) - ε·Jv|| should be O(ε²) if JVP correct
    print(f"\n  JVP Verification (layer {layer_idx}):")
    print(f"  ||JVP||  = {exact_jvp.norm().item():.4f},  ||VJP|| = {vjp_result.norm().item():.4f}")

    zeroth_vals, first_vals = [], []
    for eps in [0.1, 0.01, 0.001]:
        with torch.no_grad():
            f_x_eps = fn(h_in + eps * v).float()
        zeroth = (f_x_eps - f_x).norm().item()
        first = (f_x_eps - f_x - eps * exact_jvp).norm().item()
        zeroth_vals.append(zeroth)
        first_vals.append(first)
        improvement = zeroth / first if first > 0 else float("inf")
        print(f"  eps={eps:.0e}: zeroth={zeroth:.4e}  first={first:.4e}  "
              f"improvement={improvement:.1f}x")

    if exact_jvp.norm().item() < 1e-10:
        print("  FAIL: JVP is zero — forward AD not working!")
        print("  Ensure model loaded with attn_implementation='eager'")
    else:
        # JVP is correct if first-order error << zeroth-order error
        best_improvement = max(z / f if f > 0 else float("inf")
                               for z, f in zip(zeroth_vals, first_vals))
        if best_improvement > 5:
            print(f"  OK: JVP provides up to {best_improvement:.0f}x improvement — correct.")
        else:
            print("  WARNING: JVP not providing significant improvement.")


def compare_methods(runner, input_ids, seq_layers, max_iters, methods=None):
    """Run all methods and compare against sequential ground truth."""
    if methods is None:
        methods = ["jacobi", "fd-newton", "vjp-newton", "jvp-newton", "identity-newton"]

    print(f"\nModel: {runner.num_layers} layers, "
          f"seq_layers={seq_layers}, parallel={runner.num_layers - seq_layers}, "
          f"iters={max_iters}")
    print(f"Input: {input_ids.shape[1]} tokens")

    # Ground truth
    t0 = time.perf_counter()
    seq_logits, seq_h = sequential_forward(runner, input_ids)
    t_seq = time.perf_counter() - t0
    seq_top1 = seq_logits[:, -1].argmax(dim=-1)
    print(f"\n{'Method':<20} {'Top-1':>6} {'Cos-sim':>8} {'Max-δ':>10} {'Time':>8}")
    print(f"{'Sequential':<20} {'—':>6} {'1.000':>8} {'—':>10} {t_seq*1000:>7.1f}ms")

    results = {}
    dispatch = {
        "jacobi": lambda: jacobi_forward(runner, input_ids, seq_layers, max_iters),
        "fd-newton": lambda: newton_fd_forward(runner, input_ids, seq_layers, max_iters),
        "vjp-newton": lambda: newton_vjp_forward(runner, input_ids, seq_layers, max_iters),
        "jvp-newton": lambda: newton_jvp_forward(runner, input_ids, seq_layers, max_iters),
        "identity-newton": lambda: newton_identity_forward(runner, input_ids, seq_layers, max_iters),
    }

    for name in methods:
        if name not in dispatch:
            print(f"Unknown method: {name}")
            continue

        t0 = time.perf_counter()
        logits, h, diag = dispatch[name]()
        elapsed = time.perf_counter() - t0

        # Metrics
        top1 = logits[:, -1].argmax(dim=-1)
        top1_match = (top1 == seq_top1).all().item()

        last_h = h[runner.num_layers].float()
        ref_h = seq_h[runner.num_layers].float()
        cos_sim = torch.nn.functional.cosine_similarity(
            last_h.reshape(1, -1), ref_h.reshape(1, -1)
        ).item()

        last_delta = diag[-1] if diag else 0.0

        mark = "✓" if top1_match else "✗"
        print(f"{name:<20} {mark:>6} {cos_sim:>8.4f} {last_delta:>10.6f} {elapsed*1000:>7.1f}ms")

        if diag:
            delta_str = " → ".join(f"{d:.4f}" for d in diag[:8])
            print(f"  convergence: {delta_str}")

        results[name] = {
            "top1_match": top1_match, "cos_sim": cos_sim,
            "diagnostics": diag, "time": elapsed,
        }

    return results


# ---------------------------------------------------------------------------
# Primitive cost benchmark
# ---------------------------------------------------------------------------

def benchmark_local_ops(runner, input_ids, layer_idx=None, eps=1e-3, n_warmup=10, n_runs=50):
    """
    Benchmark the local cost of one layer forward, FD-JVP, exact JVP, and VJP.

    This isolates a single layer on a fixed hidden state so the ratios reflect
    the raw local Jacobian primitive cost, separate from Newton iteration logic.
    """
    device = runner.device

    with torch.no_grad():
        x0, mask, pos_ids, cache_pos, pos_emb = runner.embed(input_ids)

    n = runner.num_layers
    if layer_idx is None:
        layer_idx = n // 2
    if not (0 <= layer_idx < n):
        raise ValueError(f"layer_idx must be in [0, {n}), got {layer_idx}")

    with torch.no_grad():
        h_in = x0
        for i in range(layer_idx):
            h_in = runner.run_layer(i, h_in, mask, pos_ids, cache_pos, pos_emb)
        h_in = h_in.detach()

    v = torch.randn_like(h_in)
    v = v / (v.float().norm() + 1e-8)
    layer = runner.inner.layers[layer_idx]

    def fn(x):
        out = layer(
            x,
            attention_mask=mask,
            position_ids=pos_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_pos,
            position_embeddings=pos_emb,
        )
        return out[0] if isinstance(out, tuple) else out

    def op_forward():
        with torch.no_grad():
            return fn(h_in)

    def op_fd_total():
        with torch.no_grad():
            f_x = fn(h_in)
            f_x_eps = fn(h_in + eps * v)
            return (f_x_eps - f_x) / eps

    def op_fd_extra():
        with torch.no_grad():
            f_x_eps = fn(h_in + eps * v)
            return f_x_eps

    def op_jvp():
        with torch.enable_grad():
            x = h_in.detach().requires_grad_(True)
            _, jv = autograd_jvp(fn, (x,), (v.detach(),), create_graph=False, strict=False)
        return jv.detach()

    def op_vjp():
        with torch.enable_grad():
            x = h_in.detach().requires_grad_(True)
            out = fn(x)
            jt_v = torch.autograd.grad(out, x, grad_outputs=v.detach(), retain_graph=False)[0]
        return jt_v.detach()

    def bench(name, op):
        for _ in range(n_warmup):
            op()
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        sample = None
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            sample = op()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        avg_ms = sum(times) / len(times) * 1000
        std_ms = (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5 * 1000
        return {
            "name": name,
            "avg_ms": avg_ms,
            "std_ms": std_ms,
            "norm": sample.float().norm().item() if sample is not None else float("nan"),
        }

    results = [
        bench("forward", op_forward),
        bench("fd_total", op_fd_total),
        bench("fd_extra", op_fd_extra),
    ]

    try:
        results.append(bench("jvp", op_jvp))
        jvp_error = None
    except Exception as exc:
        jvp_error = repr(exc)

    try:
        results.append(bench("vjp", op_vjp))
        vjp_error = None
    except Exception as exc:
        vjp_error = repr(exc)

    forward_ms = next(item["avg_ms"] for item in results if item["name"] == "forward")

    print(f"\nPrimitive cost benchmark: layer={layer_idx}, seq_len={input_ids.shape[1]}, eps={eps}")
    print(f"{'Op':<10} {'Time':>10} {'Ratio':>8} {'Norm':>12}")
    print(f"{'-'*10} {'-'*10} {'-'*8} {'-'*12}")
    for item in results:
        ratio = item["avg_ms"] / forward_ms if forward_ms > 0 else float('inf')
        print(f"{item['name']:<10} {item['avg_ms']:>8.3f}ms {ratio:>7.2f}x {item['norm']:>12.4f}")

    if jvp_error is not None:
        print(f"jvp failed: {jvp_error}")
        print("Hint: load with --attn-impl eager if forward-mode AD is unsupported.")
    if vjp_error is not None:
        print(f"vjp failed: {vjp_error}")

    print("\nNotes:")
    print("  forward  = one plain layer evaluation")
    print("  fd_total = f(x) and f(x+eps*v), i.e. total finite-difference JVP cost")
    print("  fd_extra = only the perturbed forward, assuming f(x) was already computed")
    print("  jvp      = exact J*v via forward-mode AD")
    print("  vjp      = exact J^T*v via reverse-mode AD")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Layer-parallel inference via DEER for Qwen2.5 models"
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa"],
                   help="Attention implementation. 'eager' needed for JVP-Newton.")
    p.add_argument("--seed", type=int, default=42)

    # Method selection
    p.add_argument("--mode", default="compare",
                   choices=["compare", "generate", "verify-jvp", "bench-costs"])
    p.add_argument("--method", default="all",
                   help="Method(s) to run. 'all' or comma-separated: "
                        "jacobi,fd-newton,vjp-newton,jvp-newton,identity-newton")
    p.add_argument("--seq-layers", type=int, default=None,
                   help="Number of sequential prefix layers (default: n_layers - 4)")
    p.add_argument("--iters", type=int, default=6,
                   help="Newton/Jacobi iterations")
    p.add_argument("--layer-idx", type=int, default=None,
                   help="Target layer for --mode bench-costs (default: middle layer)")
    p.add_argument("--eps", type=float, default=1e-3,
                   help="Finite-difference epsilon for FD/JVP benchmark")
    p.add_argument("--bench-warmup", type=int, default=10,
                   help="Warmup runs for --mode bench-costs")
    p.add_argument("--bench-runs", type=int, default=50,
                   help="Timed runs for --mode bench-costs")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--no-chat-template", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    print(f"Loading {args.model} (dtype={args.dtype}, attn={args.attn_impl}, local_only=True)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_impl,
        local_files_only=True,
    ).eval()

    runner = LayerRunner(model)
    n = runner.num_layers

    seq_layers = args.seq_layers
    if seq_layers is None:
        seq_layers = max(0, n - 4)
    seq_layers = min(seq_layers, n)

    print(f"Model: {n} layers, hidden_size={runner.config.hidden_size}")
    print(f"Config: seq_layers={seq_layers}, parallel={n - seq_layers}, iters={args.iters}")

    # Tokenize
    if not args.no_chat_template and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = args.prompt

    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(runner.device)
    print(f"Prompt: {input_ids.shape[1]} tokens")

    if args.mode == "verify-jvp":
        print("\nVerifying JVP correctness...")
        for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
            verify_jvp(runner, input_ids, layer_idx=i)

    elif args.mode == "bench-costs":
        benchmark_local_ops(
            runner,
            input_ids,
            layer_idx=args.layer_idx,
            eps=args.eps,
            n_warmup=args.bench_warmup,
            n_runs=args.bench_runs,
        )

    elif args.mode == "compare":
        if args.method == "all":
            methods = ["jacobi", "fd-newton", "vjp-newton", "jvp-newton", "identity-newton"]
        else:
            methods = [m.strip() for m in args.method.split(",")]

        compare_methods(runner, input_ids, seq_layers, args.iters, methods)

    elif args.mode == "generate":
        method = args.method if args.method != "all" else "fd-newton"

        # Sequential baseline
        print("\n--- Sequential generation ---")
        t0 = time.perf_counter()
        out_seq = generate_sequential(runner, input_ids, args.max_new_tokens,
                                      eos_token_id=tokenizer.eos_token_id)
        t_seq = time.perf_counter() - t0
        text_seq = tokenizer.decode(out_seq[0], skip_special_tokens=True)
        print(f"[{t_seq:.2f}s] {text_seq}")

        # Newton generation
        newton_method = "fd" if "fd" in method else "identity"
        print(f"\n--- {method} generation (seq={seq_layers}, iters={args.iters}) ---")
        t0 = time.perf_counter()
        out_newton = generate_newton(
            runner, input_ids, args.max_new_tokens,
            seq_layers=seq_layers, newton_iters=args.iters,
            method=newton_method, eos_token_id=tokenizer.eos_token_id,
        )
        t_newton = time.perf_counter() - t0
        text_newton = tokenizer.decode(out_newton[0], skip_special_tokens=True)
        print(f"[{t_newton:.2f}s] {text_newton}")

        # Compare
        match = (out_seq[:, :out_newton.shape[1]] == out_newton).all().item()
        print(f"\nTokens match: {'Yes' if match else 'No'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
