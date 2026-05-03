"""
SJD + IDN Layer-Parallel Codesign for nanochat.

Combines Speculative Jacobi Decoding (sequence-parallel) with Identity Newton
layer-parallel inference (layer-parallel) on a 2D layer×sequence lattice.

Coupling modes:
  naive:  Each JD/SJD iteration runs IDN from scratch (h0 init). K1-JD / K2-JD.
  h0:     Coupled. Fresh h_init (recompute seq prefix), stale layers 1+ from
          previous JD iteration. IDN correction propagates delta. h0-JD.
  stale:  Fully stale. No prefix recomputation between JD iterations. Stale-JD.

Forward backends:
  sequential:        Standard model.forward()
  idn_batched:       All parallel layers via stacked einsum + IDN correction
  chunkwise_batched: Divide into equal-size chunks, fuse within, IDN between

Usage:
    # Baseline: sequential forward
    python -m scripts.demo_sjd_idn_nanochat --forward-backend sequential --coupling naive

    # Naive K=2 with ChunkB_12xF1
    python -m scripts.demo_sjd_idn_nanochat --forward-backend chunkwise_batched \
        --n-par 12 --K 2 --chunk-size 1 --coupling naive

    # Coupled h0-JD K=1 with ChunkB_4xF3
    python -m scripts.demo_sjd_idn_nanochat --forward-backend chunkwise_batched \
        --n-par 12 --K 1 --chunk-size 3 --coupling h0
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.jacobi_forward import _embed, _run_block, _logits

import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'
fa.USE_FA3 = False

from scripts.eval_jacobi_idn_new import (
    PrecomputedWeights, ChunkWeights,
    _batched_forward, _batched_chunks_forward,
    _idn_correction,
)
from scripts.eval_jacobi_idn import load_model
from scripts.demo_sjd_nanochat import get_tokenizer, print_stats

device = torch.device(autodetect_device_type())
dtype = COMPUTE_DTYPE


# ---------------------------------------------------------------------------
# Forward function factories
# ---------------------------------------------------------------------------

def make_forward_fn(model, method, n_par, K=1, elk_k=0.0, avg=False, chunk_size=1):
    """Create a stateful forward function for IDN layer-parallel inference.

    Returns forward_fn(idx, h_warm=None) -> (logits, h_state)
      idx:    (B, T) token ids
      h_warm: None (h0 init) or list of tensors (warm-start parallel layers)
      logits: (B, T, V)
      h_state: list of tensors for warm-starting next call

    For coupling='naive', caller always passes h_warm=None.
    For coupling='h0', caller passes h_state from previous JD iteration.
    """
    L = model.config.n_layer
    seq_layers = L - n_par
    par = list(range(seq_layers, L))

    if method == 'sequential':
        @torch.inference_mode()
        def forward_fn(idx, h_warm=None):
            logits = model.forward(idx)
            return logits, None
        return forward_fn, {'method': 'sequential', 'n_par': 0, 'K': 0}

    pw = PrecomputedWeights(model, par)

    if method == 'idn_batched':
        @torch.inference_mode()
        def forward_fn(idx, h_warm=None):
            x0, cos_sin, ve = _embed(model, idx)
            h = [None] * L; x = x0
            for i in range(seq_layers):
                x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
            h_init = x

            if h_warm is not None:
                hs = [h_init] + list(h_warm)
            else:
                hs = [h_init] + [h_init.clone() for _ in range(n_par)]

            for _k in range(K):
                all_hp = torch.stack([hs[j] for j in range(n_par)])
                all_out = _batched_forward(all_hp, x0, cos_sin, ve, pw, model)
                _idn_correction(list(all_out), hs, h_init, n_par, elk_k)

            for j, li in enumerate(par):
                h[li] = hs[j + 1]
            logits = _logits(model, h)
            h_state = [hs[j + 1] for j in range(n_par)]
            return logits, h_state

        info = {'method': 'idn_batched', 'n_par': n_par, 'K': K,
                'elk_k': elk_k, 'seq_layers': seq_layers}
        return forward_fn, info

    elif method in ('chunkwise', 'chunkwise_batched'):
        chunks = [par[i:i + chunk_size] for i in range(0, n_par, chunk_size)]
        chunk_weights_list = [ChunkWeights(model, c) for c in chunks]
        n_chunks = len(chunk_weights_list)
        use_batched = (method == 'chunkwise_batched'
                       and len(set(cw.n for cw in chunk_weights_list)) == 1
                       and n_chunks > 1)

        @torch.inference_mode()
        def forward_fn(idx, h_warm=None):
            x0, cos_sin, ve = _embed(model, idx)
            h = [None] * L; x = x0
            for i in range(seq_layers):
                x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
            h_init = x

            if h_warm is not None:
                hs = [h_init] + list(h_warm)
            else:
                hs = [h_init] + [h_init.clone() for _ in range(n_chunks)]

            for _k in range(K):
                if use_batched:
                    all_h_in = torch.stack([hs[c] for c in range(n_chunks)])
                    all_out = _batched_chunks_forward(
                        all_h_in, x0, cos_sin, chunk_weights_list, model, avg=avg)
                    chunk_outs = [all_out[c] for c in range(n_chunks)]
                else:
                    chunk_outs = []
                    for c, cw in enumerate(chunk_weights_list):
                        from scripts.eval_jacobi_idn_new import _fused_forward_chunk
                        out = _fused_forward_chunk(hs[c], x0, cos_sin, cw, model, avg=avg)
                        chunk_outs.append(out)
                _idn_correction(chunk_outs, hs, h_init, n_chunks, elk_k)

            for i in range(seq_layers, L):
                h[i] = h_init
            h[L - 1] = hs[n_chunks]
            backout = L // 2
            if backout >= seq_layers:
                for c, cw in enumerate(chunk_weights_list):
                    if backout <= cw.layers[-1]:
                        h[backout] = hs[c + 1]
                        break
            logits = _logits(model, h)
            h_state = [hs[c + 1] for c in range(n_chunks)]
            return logits, h_state

        label = f'{n_chunks}xF{chunk_size}'
        info = {'method': method, 'n_par': n_par, 'K': K, 'elk_k': elk_k,
                'chunk_size': chunk_size, 'n_chunks': n_chunks, 'label': label,
                'seq_layers': seq_layers, 'avg': avg, 'batched': use_batched}
        return forward_fn, info

    else:
        raise ValueError(f"Unknown method: {method}")


def make_stale_forward_fn(model, method, n_par, elk_k=0.0, avg=False, chunk_size=1):
    """Create a stale forward function that skips sequential prefix recomputation.

    For Stale-JD: after the first call, does NOT re-run sequential prefix layers.
    Only re-embeds (needed for x0 in residual blending) and runs IDN on stale h.

    Returns forward_fn(idx, h_warm) -> (logits, h_state)
      h_warm must include h_init as first element: [h_init, h1, h2, ...]
    """
    L = model.config.n_layer
    seq_layers = L - n_par
    par = list(range(seq_layers, L))
    pw = PrecomputedWeights(model, par)

    if method == 'idn_batched':
        @torch.inference_mode()
        def forward_fn(idx, h_warm):
            x0, cos_sin, ve = _embed(model, idx)

            if h_warm is None:
                h = [None] * L; x = x0
                for i in range(seq_layers):
                    x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
                h_init = x
                hs = [h_init] + [h_init.clone() for _ in range(n_par)]
            else:
                h_init = h_warm[0]
                hs = [h_init] + list(h_warm[1:])
                h = [None] * L
                for i in range(seq_layers):
                    h[i] = h_init

            all_hp = torch.stack([hs[j] for j in range(n_par)])
            all_out = _batched_forward(all_hp, x0, cos_sin, ve, pw, model)
            _idn_correction(list(all_out), hs, h_init, n_par, elk_k)

            for j, li in enumerate(par):
                h[li] = hs[j + 1]
            logits = _logits(model, h)
            h_state = [h_init] + [hs[j + 1] for j in range(n_par)]
            return logits, h_state

    elif method in ('chunkwise', 'chunkwise_batched'):
        chunks = [par[i:i + chunk_size] for i in range(0, n_par, chunk_size)]
        chunk_weights_list = [ChunkWeights(model, c) for c in chunks]
        n_chunks = len(chunk_weights_list)
        use_batched = (method == 'chunkwise_batched'
                       and len(set(cw.n for cw in chunk_weights_list)) == 1
                       and n_chunks > 1)

        @torch.inference_mode()
        def forward_fn(idx, h_warm):
            x0, cos_sin, ve = _embed(model, idx)

            if h_warm is None:
                h = [None] * L; x = x0
                for i in range(seq_layers):
                    x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
                h_init = x
                hs = [h_init] + [h_init.clone() for _ in range(n_chunks)]
            else:
                h_init = h_warm[0]
                hs = [h_init] + list(h_warm[1:])
                h = [None] * L
                for i in range(seq_layers):
                    h[i] = h_init

            if use_batched:
                all_h_in = torch.stack([hs[c] for c in range(n_chunks)])
                all_out = _batched_chunks_forward(
                    all_h_in, x0, cos_sin, chunk_weights_list, model, avg=avg)
                chunk_outs = [all_out[c] for c in range(n_chunks)]
            else:
                chunk_outs = []
                for c, cw in enumerate(chunk_weights_list):
                    from scripts.eval_jacobi_idn_new import _fused_forward_chunk
                    out = _fused_forward_chunk(hs[c], x0, cos_sin, cw, model, avg=avg)
                    chunk_outs.append(out)
            _idn_correction(chunk_outs, hs, h_init, n_chunks, elk_k)

            for i in range(seq_layers, L):
                h[i] = h_init
            h[L - 1] = hs[n_chunks]
            backout = L // 2
            if backout >= seq_layers:
                for c, cw in enumerate(chunk_weights_list):
                    if backout <= cw.layers[-1]:
                        h[backout] = hs[c + 1]
                        break
            logits = _logits(model, h)
            h_state = [h_init] + [hs[c + 1] for c in range(n_chunks)]
            return logits, h_state

    else:
        raise ValueError(f"Stale mode not supported for method: {method}")

    info = {'method': f'stale_{method}', 'n_par': n_par, 'K': 1,
            'elk_k': elk_k, 'chunk_size': chunk_size}
    return forward_fn, info


# ---------------------------------------------------------------------------
# AR generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_ar(input_ids, max_new_tokens, forward_fn,
                temperature=0.0, top_k=None):
    ids = input_ids.clone()
    n_prompt = ids.shape[1]

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        logits, _ = forward_fn(ids, h_warm=None)
        logits = logits[:, -1, :]
        if temperature > 0:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    n_gen = ids.shape[1] - n_prompt
    return ids, {
        'method': 'AR', 'n_prompt': n_prompt, 'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': n_gen, 'acceptance_rate': 1.0,
    }


# ---------------------------------------------------------------------------
# JD — Naive (K-JD)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_jd(input_ids, max_new_tokens, forward_fn, n_lookahead=5,
                temperature=0.0, top_k=None, max_jacobi_iters=32):
    _test_logits, _ = forward_fn(input_ids, h_warm=None)
    vocab_size = _test_logits.shape[-1]

    ids = input_ids.clone()
    n_prompt = ids.shape[1]
    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
    total_accepted = 0
    total_forward_passes = 0
    total_steps = 0
    iters_per_step = []

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    while total_accepted < max_new_tokens:
        remaining = max_new_tokens - total_accepted
        n_draft = min(n_lookahead, remaining - 1) if remaining > 1 else 0

        if n_draft == 0:
            logits, _ = forward_fn(ids, h_warm=None)
            total_forward_passes += 1
            ids = torch.cat([ids, logits[:, -1:, :].argmax(dim=-1)], dim=1)
            total_accepted += 1
            total_steps += 1
            continue

        current_drafts = drafts[:, :n_draft]
        accepted_this_step = False

        for ji in range(max_jacobi_iters):
            full_seq = torch.cat([ids, current_drafts], dim=1)
            logits, _ = forward_fn(full_seq, h_warm=None)
            total_forward_passes += 1
            predicted = logits.argmax(dim=-1)
            T = ids.shape[1]
            model_preds = predicted[:, T - 1:T - 1 + n_draft]

            match = (model_preds == current_drafts).squeeze(0)
            n_match = 0
            for j in range(n_draft):
                if match[j]:
                    n_match += 1
                else:
                    break

            if n_match > 0:
                accepted = current_drafts[:, :n_match]
                bonus = predicted[:, T - 1 + n_match:T + n_match]
                ids = torch.cat([ids, accepted, bonus], dim=1)
                total_accepted += n_match + 1
                total_steps += 1
                iters_per_step.append(ji + 1)
                accepted_this_step = True
                n_consumed = n_match + 1
                if n_consumed < n_draft:
                    updated = model_preds[:, n_consumed:]
                    n_new = n_lookahead - updated.shape[1]
                    drafts = torch.cat([updated, torch.randint(0, vocab_size, (1, n_new), device=device)], dim=1)
                else:
                    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
                break

            current_drafts = model_preds.clone()

        if not accepted_this_step:
            ids = torch.cat([ids, predicted[:, T - 1:T]], dim=1)
            total_accepted += 1
            total_steps += 1
            iters_per_step.append(max_jacobi_iters)
            updated = model_preds[:, 1:]
            n_new = n_lookahead - updated.shape[1]
            drafts = torch.cat([updated, torch.randint(0, vocab_size, (1, n_new), device=device)], dim=1)

    ids = ids[:, :n_prompt + max_new_tokens]
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    n_gen = ids.shape[1] - n_prompt
    avg_iters = sum(iters_per_step) / len(iters_per_step) if iters_per_step else 0
    return ids, {
        'method': f'JD(la={n_lookahead})',
        'n_prompt': n_prompt, 'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': total_forward_passes,
        'n_steps': total_steps,
        'acceptance_rate': n_gen / total_forward_passes if total_forward_passes > 0 else 0,
        'avg_jacobi_iters': avg_iters,
        'iters_per_step': iters_per_step,
    }


# ---------------------------------------------------------------------------
# JD — Coupled (h0-JD and Stale-JD)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_jd_coupled(input_ids, max_new_tokens, forward_fn, n_lookahead=5,
                        temperature=0.0, top_k=None, max_jacobi_iters=32,
                        coupling='h0'):
    _test_logits, _ = forward_fn(input_ids, h_warm=None)
    vocab_size = _test_logits.shape[-1]

    ids = input_ids.clone()
    n_prompt = ids.shape[1]
    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
    total_accepted = 0
    total_forward_passes = 0
    total_steps = 0
    iters_per_step = []

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    while total_accepted < max_new_tokens:
        remaining = max_new_tokens - total_accepted
        n_draft = min(n_lookahead, remaining - 1) if remaining > 1 else 0

        if n_draft == 0:
            logits, _ = forward_fn(ids, h_warm=None)
            total_forward_passes += 1
            ids = torch.cat([ids, logits[:, -1:, :].argmax(dim=-1)], dim=1)
            total_accepted += 1
            total_steps += 1
            continue

        current_drafts = drafts[:, :n_draft]
        accepted_this_step = False
        h_warm = None

        for ji in range(max_jacobi_iters):
            full_seq = torch.cat([ids, current_drafts], dim=1)

            if coupling == 'stale' and h_warm is not None:
                logits, h_warm = forward_fn(full_seq, h_warm=h_warm)
            else:
                logits, h_warm = forward_fn(full_seq, h_warm=h_warm)
            total_forward_passes += 1

            predicted = logits.argmax(dim=-1)
            T = ids.shape[1]
            model_preds = predicted[:, T - 1:T - 1 + n_draft]

            match = (model_preds == current_drafts).squeeze(0)
            n_match = 0
            for j in range(n_draft):
                if match[j]:
                    n_match += 1
                else:
                    break

            if n_match > 0:
                accepted = current_drafts[:, :n_match]
                bonus = predicted[:, T - 1 + n_match:T + n_match]
                ids = torch.cat([ids, accepted, bonus], dim=1)
                total_accepted += n_match + 1
                total_steps += 1
                iters_per_step.append(ji + 1)
                accepted_this_step = True
                n_consumed = n_match + 1
                if n_consumed < n_draft:
                    updated = model_preds[:, n_consumed:]
                    n_new = n_lookahead - updated.shape[1]
                    drafts = torch.cat([updated, torch.randint(0, vocab_size, (1, n_new), device=device)], dim=1)
                else:
                    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
                break

            current_drafts = model_preds.clone()

        if not accepted_this_step:
            ids = torch.cat([ids, predicted[:, T - 1:T]], dim=1)
            total_accepted += 1
            total_steps += 1
            iters_per_step.append(max_jacobi_iters)
            updated = model_preds[:, 1:]
            n_new = n_lookahead - updated.shape[1]
            drafts = torch.cat([updated, torch.randint(0, vocab_size, (1, n_new), device=device)], dim=1)

    ids = ids[:, :n_prompt + max_new_tokens]
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    n_gen = ids.shape[1] - n_prompt
    avg_iters = sum(iters_per_step) / len(iters_per_step) if iters_per_step else 0
    return ids, {
        'method': f'JD_coupled_{coupling}(la={n_lookahead})',
        'n_prompt': n_prompt, 'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': total_forward_passes,
        'n_steps': total_steps,
        'acceptance_rate': n_gen / total_forward_passes if total_forward_passes > 0 else 0,
        'avg_jacobi_iters': avg_iters,
        'iters_per_step': iters_per_step,
    }


# ---------------------------------------------------------------------------
# SJD — Naive
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_sjd(input_ids, max_new_tokens, forward_fn, n_lookahead=16,
                 temperature=1.0, top_k=50, reuse_threshold=-1.0):
    assert temperature > 0, "SJD requires stochastic sampling (temperature > 0)"
    use_reuse = reuse_threshold >= 0
    _test_logits, _ = forward_fn(input_ids, h_warm=None)
    vocab_size = _test_logits.shape[-1]

    ids = input_ids.clone()
    n_prompt = ids.shape[1]

    def _one_hot(tokens):
        dist = torch.zeros(tokens.shape[0], tokens.shape[1], vocab_size, device=device)
        return dist.scatter_(2, tokens.unsqueeze(-1), 1.0)

    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
    draft_dist = _one_hot(drafts)

    total_accepted = 0
    total_forward_passes = 0
    total_steps = 0
    total_reused = 0
    bootstrap_complete = False

    def _get_probs(logits):
        logits_s = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits_s, min(top_k, logits_s.size(-1)), dim=-1)
            logits_s[logits_s < v[..., [-1]]] = -float('inf')
        return F.softmax(logits_s, dim=-1)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    while total_accepted < max_new_tokens:
        if not bootstrap_complete:
            logits, _ = forward_fn(ids, h_warm=None)
            total_forward_passes += 1
            probs = _get_probs(logits[:, -1, :])
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
            total_accepted += 1
            total_steps += 1
            bootstrap_complete = True
            continue

        remaining = max_new_tokens - total_accepted
        n_draft = min(n_lookahead, remaining - 1) if remaining > 1 else 0

        if n_draft == 0:
            logits, _ = forward_fn(ids, h_warm=None)
            total_forward_passes += 1
            probs = _get_probs(logits[:, -1, :])
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
            total_accepted += 1
            total_steps += 1
            continue

        current_drafts = drafts[:, :n_draft]
        current_dist = draft_dist[:, :n_draft, :]

        full_seq = torch.cat([ids, current_drafts], dim=1)
        logits, _ = forward_fn(full_seq, h_warm=None)
        total_forward_passes += 1

        T = ids.shape[1]
        new_probs = _get_probs(logits[:, T - 1:T - 1 + n_draft, :])
        next_tokens = torch.multinomial(
            new_probs.view(-1, vocab_size), 1).view(1, n_draft)

        n_accept = 0
        for j in range(n_draft):
            tok = current_drafts[0, j].item()
            p_new = new_probs[0, j, tok].item()
            p_old = current_dist[0, j, tok].item()
            accept_prob = min(1.0, p_new / p_old) if p_old > 0 else (1.0 if p_new > 0 else 0.0)
            if torch.rand(1, device=device).item() < accept_prob:
                n_accept += 1
            else:
                break

        if n_accept > 0:
            ids = torch.cat([ids, current_drafts[:, :n_accept]], dim=1)

        reject_pos = n_accept
        if reject_pos < n_draft:
            p_new_dist = new_probs[0, reject_pos, :]
            p_old_dist = current_dist[0, reject_pos, :]
            residual = (p_new_dist - p_old_dist).clamp(min=0)
            residual_sum = residual.sum()
            if residual_sum > 1e-8:
                bonus = torch.multinomial(residual.unsqueeze(0) / residual_sum, 1)
            else:
                bonus = torch.multinomial(p_new_dist.unsqueeze(0), 1)
            ids = torch.cat([ids, bonus], dim=1)
        else:
            bonus_probs = _get_probs(logits[:, T - 1 + n_draft:T + n_draft, :])
            bonus = torch.multinomial(bonus_probs.squeeze(1), 1)
            ids = torch.cat([ids, bonus], dim=1)

        total_accepted += n_accept + 1
        total_steps += 1

        n_consumed = n_accept + 1
        new_draft_tokens = []
        new_draft_dists = []

        for j in range(n_consumed, n_draft):
            if use_reuse:
                tok = current_drafts[0, j].item()
                p_new_j = new_probs[0, j, tok].item()
                p_old_j = current_dist[0, j, tok].item()
                confidence = p_new_j / p_old_j if p_old_j > 0 else float('inf')
                if confidence > reuse_threshold:
                    new_draft_tokens.append(current_drafts[:, j:j + 1])
                    new_draft_dists.append(new_probs[:, j:j + 1, :])
                    total_reused += 1
                else:
                    new_draft_tokens.append(next_tokens[:, j:j + 1])
                    new_draft_dists.append(new_probs[:, j:j + 1, :])
            else:
                new_draft_tokens.append(next_tokens[:, j:j + 1])
                new_draft_dists.append(new_probs[:, j:j + 1, :])

        n_remaining = len(new_draft_tokens)
        n_new = n_lookahead - n_remaining
        if n_new > 0:
            rand_d = torch.randint(0, vocab_size, (1, n_new), device=device)
            new_draft_tokens.append(rand_d)
            new_draft_dists.append(_one_hot(rand_d))

        if new_draft_tokens:
            drafts = torch.cat(new_draft_tokens, dim=1)
            draft_dist = torch.cat(new_draft_dists, dim=1)
        else:
            drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
            draft_dist = _one_hot(drafts)

    ids = ids[:, :n_prompt + max_new_tokens]
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    n_gen = ids.shape[1] - n_prompt
    method = f'SJD(la={n_lookahead},t={temperature},k={top_k})'
    if use_reuse:
        method = f'SJD++(la={n_lookahead},t={temperature},k={top_k},τ={reuse_threshold})'
    return ids, {
        'method': method,
        'n_prompt': n_prompt, 'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': total_forward_passes,
        'n_steps': total_steps,
        'acceptance_rate': n_gen / total_forward_passes if total_forward_passes > 0 else 0,
        'n_reused': total_reused,
    }


# ---------------------------------------------------------------------------
# SJD — Coupled (h0-JD and Stale-JD)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_sjd_coupled(input_ids, max_new_tokens, forward_fn, n_lookahead=16,
                         temperature=1.0, top_k=50, reuse_threshold=-1.0,
                         coupling='h0'):
    assert temperature > 0, "SJD requires stochastic sampling (temperature > 0)"
    use_reuse = reuse_threshold >= 0
    _test_logits, _ = forward_fn(input_ids, h_warm=None)
    vocab_size = _test_logits.shape[-1]

    ids = input_ids.clone()
    n_prompt = ids.shape[1]

    def _one_hot(tokens):
        dist = torch.zeros(tokens.shape[0], tokens.shape[1], vocab_size, device=device)
        return dist.scatter_(2, tokens.unsqueeze(-1), 1.0)

    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
    draft_dist = _one_hot(drafts)

    total_accepted = 0
    total_forward_passes = 0
    total_steps = 0
    total_reused = 0
    bootstrap_complete = False
    h_warm = None

    def _get_probs(logits):
        logits_s = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits_s, min(top_k, logits_s.size(-1)), dim=-1)
            logits_s[logits_s < v[..., [-1]]] = -float('inf')
        return F.softmax(logits_s, dim=-1)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    while total_accepted < max_new_tokens:
        if not bootstrap_complete:
            logits, h_warm = forward_fn(ids, h_warm=None)
            total_forward_passes += 1
            probs = _get_probs(logits[:, -1, :])
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
            total_accepted += 1
            total_steps += 1
            bootstrap_complete = True
            h_warm = None
            continue

        remaining = max_new_tokens - total_accepted
        n_draft = min(n_lookahead, remaining - 1) if remaining > 1 else 0

        if n_draft == 0:
            logits, _ = forward_fn(ids, h_warm=None)
            total_forward_passes += 1
            probs = _get_probs(logits[:, -1, :])
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
            total_accepted += 1
            total_steps += 1
            h_warm = None
            continue

        current_drafts = drafts[:, :n_draft]
        current_dist = draft_dist[:, :n_draft, :]

        full_seq = torch.cat([ids, current_drafts], dim=1)

        # SJD: sequence length changes every step, so h_warm from previous step
        # has wrong shape. Always use h_warm=None (coupling is a no-op for SJD).
        # Coupling only helps in JD where inner iterations keep sequence length fixed.
        logits, _ = forward_fn(full_seq, h_warm=None)
        total_forward_passes += 1

        T = ids.shape[1]
        new_probs = _get_probs(logits[:, T - 1:T - 1 + n_draft, :])
        next_tokens = torch.multinomial(
            new_probs.view(-1, vocab_size), 1).view(1, n_draft)

        n_accept = 0
        for j in range(n_draft):
            tok = current_drafts[0, j].item()
            p_new = new_probs[0, j, tok].item()
            p_old = current_dist[0, j, tok].item()
            accept_prob = min(1.0, p_new / p_old) if p_old > 0 else (1.0 if p_new > 0 else 0.0)
            if torch.rand(1, device=device).item() < accept_prob:
                n_accept += 1
            else:
                break

        if n_accept > 0:
            ids = torch.cat([ids, current_drafts[:, :n_accept]], dim=1)

        reject_pos = n_accept
        if reject_pos < n_draft:
            p_new_dist = new_probs[0, reject_pos, :]
            p_old_dist = current_dist[0, reject_pos, :]
            residual = (p_new_dist - p_old_dist).clamp(min=0)
            residual_sum = residual.sum()
            if residual_sum > 1e-8:
                bonus = torch.multinomial(residual.unsqueeze(0) / residual_sum, 1)
            else:
                bonus = torch.multinomial(p_new_dist.unsqueeze(0), 1)
            ids = torch.cat([ids, bonus], dim=1)
        else:
            bonus_probs = _get_probs(logits[:, T - 1 + n_draft:T + n_draft, :])
            bonus = torch.multinomial(bonus_probs.squeeze(1), 1)
            ids = torch.cat([ids, bonus], dim=1)

        total_accepted += n_accept + 1
        total_steps += 1

        n_consumed = n_accept + 1
        new_draft_tokens = []
        new_draft_dists = []

        for j in range(n_consumed, n_draft):
            if use_reuse:
                tok = current_drafts[0, j].item()
                p_new_j = new_probs[0, j, tok].item()
                p_old_j = current_dist[0, j, tok].item()
                confidence = p_new_j / p_old_j if p_old_j > 0 else float('inf')
                if confidence > reuse_threshold:
                    new_draft_tokens.append(current_drafts[:, j:j + 1])
                    new_draft_dists.append(new_probs[:, j:j + 1, :])
                    total_reused += 1
                else:
                    new_draft_tokens.append(next_tokens[:, j:j + 1])
                    new_draft_dists.append(new_probs[:, j:j + 1, :])
            else:
                new_draft_tokens.append(next_tokens[:, j:j + 1])
                new_draft_dists.append(new_probs[:, j:j + 1, :])

        n_remaining = len(new_draft_tokens)
        n_new = n_lookahead - n_remaining
        if n_new > 0:
            rand_d = torch.randint(0, vocab_size, (1, n_new), device=device)
            new_draft_tokens.append(rand_d)
            new_draft_dists.append(_one_hot(rand_d))

        if new_draft_tokens:
            drafts = torch.cat(new_draft_tokens, dim=1)
            draft_dist = torch.cat(new_draft_dists, dim=1)
        else:
            drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
            draft_dist = _one_hot(drafts)

    ids = ids[:, :n_prompt + max_new_tokens]
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    n_gen = ids.shape[1] - n_prompt
    method = f'SJD_coupled_{coupling}(la={n_lookahead},t={temperature},k={top_k})'
    if use_reuse:
        method = f'SJD++_coupled_{coupling}(la={n_lookahead},t={temperature},k={top_k},τ={reuse_threshold})'
    return ids, {
        'method': method,
        'n_prompt': n_prompt, 'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': total_forward_passes,
        'n_steps': total_steps,
        'acceptance_rate': n_gen / total_forward_passes if total_forward_passes > 0 else 0,
        'n_reused': total_reused,
    }


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------

def load_eval_batches(device, seq_len=128, batch_size=1, n_batches=8):
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    data_dir = os.path.join(base_dir, 'base_data_climbmix')
    import pyarrow.parquet as pq
    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
    table = pq.read_table(os.path.join(data_dir, parquet_files[-1]), columns=['text'])
    all_tokens = []
    for text in table.column('text').to_pylist():
        all_tokens.extend(tokenizer.encode(text, prepend=bos))
        if len(all_tokens) >= (seq_len + 1) * batch_size * n_batches * 2:
            break
    batches = []
    offset = 0
    for _ in range(n_batches):
        seqs = []
        for _ in range(batch_size):
            seq = all_tokens[offset:offset + seq_len + 1]
            if len(seq) < seq_len + 1:
                break
            seqs.append(seq)
            offset += seq_len
        if len(seqs) < batch_size:
            break
        batches.append(torch.tensor(seqs, dtype=torch.long, device=device))
    return batches


@torch.inference_mode()
def evaluate_ppl(forward_fn, batches):
    ppls = []
    for batch in batches:
        idx, targets = batch[:, :-1], batch[:, 1:]
        logits, _ = forward_fn(idx, h_warm=None)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='mean').item()
        ppls.append(math.exp(min(loss, 20.0)))
    return sum(ppls) / len(ppls)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SJD + IDN layer-parallel codesign")
    parser.add_argument("--model-tag", type=str, default="d32s_idn05_npar24_4800")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--mode", type=str, nargs='+', default=["ar", "jd", "sjd"],
                        choices=["ar", "jd", "sjd", "sjd++", "ppl"])
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-lookahead", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--reuse-threshold", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=8)

    parser.add_argument("--coupling", type=str, default="naive",
                        choices=["naive", "h0", "stale"])
    parser.add_argument("--forward-backend", type=str, default="sequential",
                        choices=["sequential", "idn_batched", "chunkwise", "chunkwise_batched"])
    parser.add_argument("--n-par", type=int, default=12)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--elk-k", type=float, default=0.0)
    parser.add_argument("--avg", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1)
    args = parser.parse_args()

    model = load_model(args.model_tag, step=args.step, device=device)
    tokenizer = get_tokenizer()

    if args.coupling == 'stale' and args.forward_backend == 'sequential':
        print("Error: stale coupling requires an IDN backend, not sequential")
        return

    if args.coupling == 'stale':
        forward_fn, info = make_stale_forward_fn(
            model, args.forward_backend, args.n_par,
            elk_k=args.elk_k, avg=args.avg, chunk_size=args.chunk_size)
    else:
        forward_fn, info = make_forward_fn(
            model, args.forward_backend, args.n_par,
            K=args.K, elk_k=args.elk_k, avg=args.avg, chunk_size=args.chunk_size)

    print(f"\nModel: {args.model_tag} ({model.config.n_layer}L, {model.config.n_embd}d)")
    print(f"Backend: {info}")
    print(f"Coupling: {args.coupling}")
    print(f"Device: {device}")

    bos = tokenizer.get_bos_token_id()
    prompt_tokens = tokenizer.encode(args.prompt, prepend=bos)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    print(f"Prompt ({len(prompt_tokens)} tokens): {args.prompt!r}")

    results = {}

    if "ar" in args.mode:
        print(f"\n{'=' * 60}\nAR Generation\n{'=' * 60}")
        ids_ar, stats_ar = generate_ar(
            input_ids, args.max_new_tokens, forward_fn,
            temperature=args.temperature, top_k=args.top_k)
        text_ar = tokenizer.decode(ids_ar[0].tolist())
        print(f"Output: {text_ar!r}")
        print_stats(stats_ar)
        results['ar'] = (ids_ar, stats_ar)

    if "jd" in args.mode:
        print(f"\n{'=' * 60}\nJD ({args.coupling}, K={args.K}, la={args.n_lookahead})\n{'=' * 60}")
        if args.coupling == 'naive':
            ids_jd, stats_jd = generate_jd(
                input_ids, args.max_new_tokens, forward_fn,
                n_lookahead=args.n_lookahead,
                temperature=args.temperature, top_k=args.top_k)
        else:
            ids_jd, stats_jd = generate_jd_coupled(
                input_ids, args.max_new_tokens, forward_fn,
                n_lookahead=args.n_lookahead,
                temperature=args.temperature, top_k=args.top_k,
                coupling=args.coupling)
        text_jd = tokenizer.decode(ids_jd[0].tolist())
        print(f"Output: {text_jd!r}")
        print_stats(stats_jd)
        if 'avg_jacobi_iters' in stats_jd:
            print(f"  Avg Jacobi iters: {stats_jd['avg_jacobi_iters']:.2f}")
        results['jd'] = (ids_jd, stats_jd)

    if "sjd" in args.mode:
        temp = args.temperature if args.temperature > 0 else 1.0
        topk = args.top_k if args.top_k is not None else 50
        print(f"\n{'=' * 60}\nSJD ({args.coupling}, K={args.K}, la={args.n_lookahead}, t={temp})\n{'=' * 60}")
        if args.temperature == 0:
            print("  (SJD requires stochastic sampling; using temp=1.0, top_k=50)")
        if args.coupling == 'naive':
            ids_sjd, stats_sjd = generate_sjd(
                input_ids, args.max_new_tokens, forward_fn,
                n_lookahead=args.n_lookahead, temperature=temp, top_k=topk)
        else:
            ids_sjd, stats_sjd = generate_sjd_coupled(
                input_ids, args.max_new_tokens, forward_fn,
                n_lookahead=args.n_lookahead, temperature=temp, top_k=topk,
                coupling=args.coupling)
        text_sjd = tokenizer.decode(ids_sjd[0].tolist())
        print(f"Output: {text_sjd!r}")
        print_stats(stats_sjd)
        results['sjd'] = (ids_sjd, stats_sjd)

    if "sjd++" in args.mode:
        temp = args.temperature if args.temperature > 0 else 1.0
        topk = args.top_k if args.top_k is not None else 50
        tau = args.reuse_threshold
        print(f"\n{'=' * 60}\nSJD++ ({args.coupling}, K={args.K}, la={args.n_lookahead}, τ={tau})\n{'=' * 60}")
        if args.temperature == 0:
            print("  (SJD++ requires stochastic sampling; using temp=1.0, top_k=50)")
        if args.coupling == 'naive':
            ids_sjdpp, stats_sjdpp = generate_sjd(
                input_ids, args.max_new_tokens, forward_fn,
                n_lookahead=args.n_lookahead, temperature=temp, top_k=topk,
                reuse_threshold=tau)
        else:
            ids_sjdpp, stats_sjdpp = generate_sjd_coupled(
                input_ids, args.max_new_tokens, forward_fn,
                n_lookahead=args.n_lookahead, temperature=temp, top_k=topk,
                reuse_threshold=tau, coupling=args.coupling)
        text_sjdpp = tokenizer.decode(ids_sjdpp[0].tolist())
        print(f"Output: {text_sjdpp!r}")
        print_stats(stats_sjdpp)
        results['sjd++'] = (ids_sjdpp, stats_sjdpp)

    # Comparisons
    if 'ar' in results:
        ar_ids = results['ar'][0]
        for name in ['jd', 'sjd', 'sjd++']:
            if name in results:
                other_ids = results[name][0]
                min_len = min(ar_ids.shape[1], other_ids.shape[1])
                overlap = (ar_ids[:, :min_len] == other_ids[:, :min_len]).float().mean().item()
                speedup = results['ar'][1]['time'] / results[name][1]['time'] if results[name][1]['time'] > 0 else 0
                print(f"\n  AR vs {name}: token overlap={overlap:.1%}, speedup={speedup:.2f}x")

    if "ppl" in args.mode:
        print(f"\n{'=' * 60}\nPPL Evaluation\n{'=' * 60}")
        batches = load_eval_batches(device, seq_len=args.seq_len, n_batches=args.n_batches)
        ppl = evaluate_ppl(forward_fn, batches)
        print(f"  PPL: {ppl:.2f} ({len(batches)} batches, seq_len={args.seq_len})")
        results['ppl'] = ppl

    # Save results
    outdir = 'cache/sjd_logs'
    os.makedirs(outdir, exist_ok=True)
    backend = args.forward_backend
    if backend != 'sequential':
        backend += f'_c{args.chunk_size}'
    tag = f"sjd_{args.model_tag}_{backend}_{args.coupling}_K{args.K}_npar{args.n_par}_la{args.n_lookahead}"
    outpath = os.path.join(outdir, f"{tag}.json")
    save_data = {
        'model': args.model_tag, 'backend': info, 'coupling': args.coupling,
        'args': {k: v for k, v in vars(args).items() if k != 'mode'},
        'results': {},
    }
    for k, v in results.items():
        if k == 'ppl':
            save_data['results']['ppl'] = v
        else:
            save_data['results'][k] = v[1]
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
