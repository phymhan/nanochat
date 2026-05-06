"""
IDN layer-parallel inference evaluation for Gemma 3 1B.

Gemma 3 architecture specifics vs LLaMA/Qwen:
  - 4 layernorms per layer (input, post-attn, pre-ffn, post-ffn)
  - QK RMSNorm with learned weights (per head_dim)
  - GELU activation (not SiLU)
  - Dual RoPE (local for sliding layers, global for full-attention layers)
  - head_dim=256, H=4, KV=1, D=1152 (H*head_dim=1024 != D)
  - Sliding window=512 on most layers (every 6th is full attention)
  - Since seq_len=128 < sliding_window=512, sliding is effectively full causal

Usage:
    HF_HOME=/mnt/nvme0n1/ligong/huggingface CUDA_VISIBLE_DEVICES=5 \
        uv run --extra gpu --group dev python -m scripts.eval_idn_gemma3 --n-par 8
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(x, weight, eps):
    """Gemma3 RMSNorm: uses (1 + weight) instead of weight."""
    x_f = x.float()
    x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_norm * (1.0 + weight.float())).to(x.dtype)


def _rms_norm_headwise(x, weight, eps):
    """Gemma3 RMSNorm per-head. weight shape: (head_dim,) or (L, head_dim)."""
    x_f = x.float()
    x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_norm * (1.0 + weight.float())).to(x.dtype)


def _apply_rope_batched(x, cos, sin):
    """x: (L, B, H, T, hd), cos/sin: (1, B, 1, T, hd)."""
    hd_half = x.shape[-1] // 2
    x1, x2 = x[..., :hd_half], x[..., hd_half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _apply_rope(x, cos, sin):
    """x: (B, T, H, hd), cos/sin: (B, T, hd)."""
    hd_half = x.shape[-1] // 2
    x1, x2 = x[..., :hd_half], x[..., hd_half:]
    c = cos.unsqueeze(2)
    s = sin.unsqueeze(2)
    return x * c + torch.cat((-x2, x1), dim=-1) * s


# ---------------------------------------------------------------------------
# Weight precomputation
# ---------------------------------------------------------------------------

class PrecomputedWeights:
    """Stacked per-layer weights for batched forward."""

    def __init__(self, model, par):
        layers = [model.model.layers[i] for i in par]
        dt = next(model.parameters()).dtype

        self.W_q = torch.stack([l.self_attn.q_proj.weight for l in layers]).to(dt)
        self.W_k = torch.stack([l.self_attn.k_proj.weight for l in layers]).to(dt)
        self.W_v = torch.stack([l.self_attn.v_proj.weight for l in layers]).to(dt)
        self.W_o = torch.stack([l.self_attn.o_proj.weight for l in layers]).to(dt)

        self.W_gate = torch.stack([l.mlp.gate_proj.weight for l in layers]).to(dt)
        self.W_up = torch.stack([l.mlp.up_proj.weight for l in layers]).to(dt)
        self.W_down = torch.stack([l.mlp.down_proj.weight for l in layers]).to(dt)

        self.ln_input = torch.stack([l.input_layernorm.weight for l in layers]).to(dt)
        self.ln_post_attn = torch.stack([l.post_attention_layernorm.weight for l in layers]).to(dt)
        self.ln_pre_ffn = torch.stack([l.pre_feedforward_layernorm.weight for l in layers]).to(dt)
        self.ln_post_ffn = torch.stack([l.post_feedforward_layernorm.weight for l in layers]).to(dt)

        self.q_norm = torch.stack([l.self_attn.q_norm.weight for l in layers]).to(dt)
        self.k_norm = torch.stack([l.self_attn.k_norm.weight for l in layers]).to(dt)

        self.is_sliding = [l.self_attn.is_sliding for l in layers]

        self.par = par
        self.n_par = len(par)


class ChunkWeights:
    """Fused (concatenated) weights for one chunk of layers."""

    def __init__(self, model, chunk_layers):
        layers = [model.model.layers[i] for i in chunk_layers]
        dt = next(model.parameters()).dtype
        n = len(chunk_layers)

        self.W_q = torch.cat([l.self_attn.q_proj.weight for l in layers], dim=0).to(dt)
        self.W_k = torch.cat([l.self_attn.k_proj.weight for l in layers], dim=0).to(dt)
        self.W_v = torch.cat([l.self_attn.v_proj.weight for l in layers], dim=0).to(dt)
        self.W_o = torch.cat([l.self_attn.o_proj.weight for l in layers], dim=1).to(dt)

        self.W_gate = torch.cat([l.mlp.gate_proj.weight for l in layers], dim=0).to(dt)
        self.W_up = torch.cat([l.mlp.up_proj.weight for l in layers], dim=0).to(dt)
        self.W_down = torch.cat([l.mlp.down_proj.weight for l in layers], dim=1).to(dt)

        self.ln_input = torch.stack([l.input_layernorm.weight for l in layers]).mean(0).to(dt)
        self.ln_post_attn = torch.stack([l.post_attention_layernorm.weight for l in layers]).mean(0).to(dt)
        self.ln_pre_ffn = torch.stack([l.pre_feedforward_layernorm.weight for l in layers]).mean(0).to(dt)
        self.ln_post_ffn = torch.stack([l.post_feedforward_layernorm.weight for l in layers]).mean(0).to(dt)

        self.q_norm = torch.stack([l.self_attn.q_norm.weight for l in layers]).mean(0).to(dt)
        self.k_norm = torch.stack([l.self_attn.k_norm.weight for l in layers]).mean(0).to(dt)

        self.layers = chunk_layers
        self.n = n


# ---------------------------------------------------------------------------
# Core forward functions
# ---------------------------------------------------------------------------

def _batched_forward(all_h_prev, pos_emb_global, pos_emb_local, pw, config):
    """Batched forward for n_par Gemma3 layers. Returns (n_par, B, T, D)."""
    n_par, B, T, D = all_h_prev.shape
    eps = config.rms_norm_eps
    n_head = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    n_groups = n_head // n_kv
    attn_dim = n_head * hd

    # Input LayerNorm
    x_norm = _rms_norm(all_h_prev, pw.ln_input[:, None, None, :], eps)

    # QKV
    all_q = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_q)
    all_k = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_k)
    all_v = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_v)

    all_q = all_q.view(n_par, B, T, n_head, hd).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)

    # QK RMSNorm (per-layer, per-head)
    all_q = _rms_norm_headwise(all_q, pw.q_norm[:, None, None, None, :], eps)
    all_k = _rms_norm_headwise(all_k, pw.k_norm[:, None, None, None, :], eps)

    # RoPE: select local vs global per layer, apply per-layer
    cos_g, sin_g = pos_emb_global
    cos_l, sin_l = pos_emb_local
    cos_g_r = cos_g.unsqueeze(0).unsqueeze(2)
    sin_g_r = sin_g.unsqueeze(0).unsqueeze(2)
    cos_l_r = cos_l.unsqueeze(0).unsqueeze(2)
    sin_l_r = sin_l.unsqueeze(0).unsqueeze(2)

    # Build per-layer cos/sin: (n_par, B, 1, T, hd)
    cos_stack = torch.where(
        torch.tensor(pw.is_sliding, device=all_q.device).view(n_par, 1, 1, 1, 1),
        cos_l_r.expand(n_par, -1, -1, -1, -1),
        cos_g_r.expand(n_par, -1, -1, -1, -1),
    )
    sin_stack = torch.where(
        torch.tensor(pw.is_sliding, device=all_q.device).view(n_par, 1, 1, 1, 1),
        sin_l_r.expand(n_par, -1, -1, -1, -1),
        sin_g_r.expand(n_par, -1, -1, -1, -1),
    )

    all_q = _apply_rope_batched(all_q, cos_stack, sin_stack)
    all_k = _apply_rope_batched(all_k, cos_stack, sin_stack)

    # GQA expand
    if n_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(
            n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)
        all_v = all_v[:, :, :, None, :, :].expand(
            n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)

    # SDPA
    q_s = all_q.reshape(n_par * B, n_head, T, hd)
    k_s = all_k.reshape(n_par * B, n_head, T, hd)
    v_s = all_v.reshape(n_par * B, n_head, T, hd)
    y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True,
                                          scale=config.head_dim ** -0.5)

    attn_out = y_s.reshape(n_par, B, n_head, T, hd).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, attn_dim)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o)

    # Post-attention LN (applied to attn output, before residual)
    attn_out = _rms_norm(attn_out, pw.ln_post_attn[:, None, None, :], eps)
    hidden = all_h_prev + attn_out

    # Pre-FFN LN + GELU MLP + Post-FFN LN
    h_norm = _rms_norm(hidden, pw.ln_pre_ffn[:, None, None, :], eps)
    gate = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_gate)
    up = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_up)
    mlp = torch.einsum('lbti,ldi->lbtd', F.gelu(gate, approximate='tanh') * up, pw.W_down)
    mlp = _rms_norm(mlp, pw.ln_post_ffn[:, None, None, :], eps)

    return hidden + mlp


def _fused_forward_chunk(h_in, pos_emb_global, pos_emb_local, cw, config, is_sliding_majority=True, avg=False):
    """Fused chunk forward. Returns (B, T, D)."""
    B, T, D = h_in.shape
    eps = config.rms_norm_eps
    n_head = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    total_q = cw.n * n_head
    total_kv = cw.n * n_kv
    n_groups = n_head // n_kv
    attn_dim = total_q * hd

    x_norm = _rms_norm(h_in, cw.ln_input, eps)

    q = F.linear(x_norm, cw.W_q).view(B, T, total_q, hd)
    k = F.linear(x_norm, cw.W_k).view(B, T, total_kv, hd)
    v = F.linear(x_norm, cw.W_v).view(B, T, total_kv, hd)

    # QK norm with averaged weights
    q = _rms_norm_headwise(q, cw.q_norm, eps)
    k = _rms_norm_headwise(k, cw.k_norm, eps)

    # Use local RoPE for sliding-majority chunks, global for full-attention
    pos_emb = pos_emb_local if is_sliding_majority else pos_emb_global
    cos, sin = pos_emb
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if n_groups > 1:
        k = k[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)
        v = v[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=hd ** -0.5)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)

    attn_out = F.linear(y, cw.W_o)
    if avg and cw.n > 1:
        attn_out = attn_out / cw.n
    attn_out = _rms_norm(attn_out, cw.ln_post_attn, eps)
    x_mid = h_in + attn_out

    h_norm = _rms_norm(x_mid, cw.ln_pre_ffn, eps)
    gate = F.linear(h_norm, cw.W_gate)
    up = F.linear(h_norm, cw.W_up)
    mlp = F.linear(F.gelu(gate, approximate='tanh') * up, cw.W_down)
    if avg and cw.n > 1:
        mlp = mlp / cw.n
    mlp = _rms_norm(mlp, cw.ln_post_ffn, eps)
    return x_mid + mlp


def _batched_chunks_forward(all_h_in, pos_emb_global, pos_emb_local, chunk_weights_list,
                             chunk_sliding_flags, config, avg=False):
    """Batched forward for C equal-size chunks. Returns (C, B, T, D)."""
    C = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n
    B, T, D = all_h_in.shape[1:]
    eps = config.rms_norm_eps
    n_head = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    q_per_chunk = chunk_size * n_head
    kv_per_chunk = chunk_size * n_kv
    n_groups = n_head // n_kv

    W_q = torch.stack([cw.W_q for cw in chunk_weights_list])
    W_k = torch.stack([cw.W_k for cw in chunk_weights_list])
    W_v = torch.stack([cw.W_v for cw in chunk_weights_list])
    W_o = torch.stack([cw.W_o for cw in chunk_weights_list])
    W_gate = torch.stack([cw.W_gate for cw in chunk_weights_list])
    W_up = torch.stack([cw.W_up for cw in chunk_weights_list])
    W_down = torch.stack([cw.W_down for cw in chunk_weights_list])

    ln_input = torch.stack([cw.ln_input for cw in chunk_weights_list])
    ln_post_attn = torch.stack([cw.ln_post_attn for cw in chunk_weights_list])
    ln_pre_ffn = torch.stack([cw.ln_pre_ffn for cw in chunk_weights_list])
    ln_post_ffn = torch.stack([cw.ln_post_ffn for cw in chunk_weights_list])
    q_norm_w = torch.stack([cw.q_norm for cw in chunk_weights_list])
    k_norm_w = torch.stack([cw.k_norm for cw in chunk_weights_list])

    x_norm = _rms_norm(all_h_in, ln_input[:, None, None, :], eps)

    all_q = torch.einsum('cbtd,chd->cbth', x_norm, W_q)
    all_k = torch.einsum('cbtd,chd->cbth', x_norm, W_k)
    all_v = torch.einsum('cbtd,chd->cbth', x_norm, W_v)

    all_q = all_q.view(C, B, T, q_per_chunk, hd).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)

    # QK norm
    all_q = _rms_norm_headwise(all_q, q_norm_w[:, None, None, None, :], eps)
    all_k = _rms_norm_headwise(all_k, k_norm_w[:, None, None, None, :], eps)

    # Per-chunk RoPE selection
    cos_g, sin_g = pos_emb_global
    cos_l, sin_l = pos_emb_local
    cos_g_r = cos_g.unsqueeze(0).unsqueeze(2)
    sin_g_r = sin_g.unsqueeze(0).unsqueeze(2)
    cos_l_r = cos_l.unsqueeze(0).unsqueeze(2)
    sin_l_r = sin_l.unsqueeze(0).unsqueeze(2)

    sliding_t = torch.tensor(chunk_sliding_flags, device=all_q.device).view(C, 1, 1, 1, 1)
    cos_stack = torch.where(sliding_t, cos_l_r.expand(C, -1, -1, -1, -1), cos_g_r.expand(C, -1, -1, -1, -1))
    sin_stack = torch.where(sliding_t, sin_l_r.expand(C, -1, -1, -1, -1), sin_g_r.expand(C, -1, -1, -1, -1))

    all_q = _apply_rope_batched(all_q, cos_stack, sin_stack)
    all_k = _apply_rope_batched(all_k, cos_stack, sin_stack)

    if n_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(
            C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)
        all_v = all_v[:, :, :, None, :, :].expand(
            C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)

    q_s = all_q.reshape(C * B, q_per_chunk, T, hd)
    k_s = all_k.reshape(C * B, q_per_chunk, T, hd)
    v_s = all_v.reshape(C * B, q_per_chunk, T, hd)
    y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True, scale=hd ** -0.5)

    y = y_s.reshape(C, B, q_per_chunk, T, hd).permute(0, 1, 3, 2, 4)
    y = y.reshape(C, B, T, q_per_chunk * hd)

    attn_out = torch.einsum('cbth,cdh->cbtd', y, W_o)
    if avg and chunk_size > 1:
        attn_out = attn_out / chunk_size
    attn_out = _rms_norm(attn_out, ln_post_attn[:, None, None, :], eps)
    hidden = all_h_in + attn_out

    h_norm = _rms_norm(hidden, ln_pre_ffn[:, None, None, :], eps)
    gate = torch.einsum('cbtd,cid->cbti', h_norm, W_gate)
    up = torch.einsum('cbtd,cid->cbti', h_norm, W_up)
    mlp = torch.einsum('cbti,cdi->cbtd', F.gelu(gate, approximate='tanh') * up, W_down)
    if avg and chunk_size > 1:
        mlp = mlp / chunk_size
    mlp = _rms_norm(mlp, ln_post_ffn[:, None, None, :], eps)

    return hidden + mlp


def _idn_correction(all_out, hs, h_init, n_units, elk_k=0.0):
    a = 1.0 - elk_k
    h_old = [hs[j + 1].clone() for j in range(n_units)]
    h_corr = all_out[0]
    hs[1] = h_corr
    for j in range(1, n_units):
        h_corr = all_out[j] + a * (h_corr - h_old[j - 1])
        hs[j + 1] = h_corr


# ---------------------------------------------------------------------------
# Sequential prefix helper
# ---------------------------------------------------------------------------

def _run_prefix(model, input_ids, seq_layers):
    """Run embedding + layers 0..seq_layers-1. Returns (h_init, pos_emb_global, pos_emb_local)."""
    hidden = model.model.embed_tokens(input_ids)

    B, T, D = hidden.shape
    position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
    pos_emb_global = model.model.rotary_emb(hidden, position_ids)
    pos_emb_local = model.model.rotary_emb_local(hidden, position_ids)

    if seq_layers > 0:
        from transformers.masking_utils import create_causal_mask
        cache_position = torch.arange(T, device=input_ids.device)
        causal_mask = create_causal_mask(
            config=model.config, input_embeds=hidden,
            attention_mask=None, cache_position=cache_position,
            past_key_values=None, position_ids=position_ids,
        )
        for i in range(seq_layers):
            layer = model.model.layers[i]
            mask = causal_mask
            if isinstance(causal_mask, dict):
                mask = causal_mask.get(layer.attention_type, None)
            out = layer(
                hidden,
                position_embeddings_global=pos_emb_global,
                position_embeddings_local=pos_emb_local,
                attention_mask=mask,
                position_ids=position_ids,
            )
            hidden = out[0] if isinstance(out, tuple) else out

    return hidden, pos_emb_global, pos_emb_local


# ---------------------------------------------------------------------------
# High-level forward functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_sequential(model, input_ids):
    return model(input_ids).logits.float()


@torch.inference_mode()
def forward_idn_batched(model, input_ids, seq_layers, pw, K=1, elk_k=0.0, init='h0'):
    h_init, peg, pel = _run_prefix(model, input_ids, seq_layers)
    n_par = pw.n_par

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward(all_h0, peg, pel, pw, model.config)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    else:
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _ in range(K):
        all_hp = torch.stack([hs[j] for j in range(n_par)])
        all_out = _batched_forward(all_hp, peg, pel, pw, model.config)
        _idn_correction(all_out, hs, h_init, n_par, elk_k)

    hidden = model.model.norm(hs[n_par])
    return model.lm_head(hidden).float()


@torch.inference_mode()
def forward_fused(model, input_ids, seq_layers, chunk_weights, K=1, elk_k=0.0,
                   avg=False, is_sliding_majority=True):
    h_init, peg, pel = _run_prefix(model, input_ids, seq_layers)

    out = _fused_forward_chunk(h_init, peg, pel, chunk_weights[0], model.config,
                                is_sliding_majority=is_sliding_majority, avg=avg)
    for _ in range(1, K):
        out = _fused_forward_chunk(out, peg, pel, chunk_weights[0], model.config,
                                    is_sliding_majority=is_sliding_majority, avg=avg)

    hidden = model.model.norm(out)
    return model.lm_head(hidden).float()


@torch.inference_mode()
def forward_chunkwise_batched(model, input_ids, seq_layers, chunk_weights_list,
                                chunk_sliding_flags, K=1, elk_k=0.0, avg=False, init='h0'):
    h_init, peg, pel = _run_prefix(model, input_ids, seq_layers)
    n_chunks = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_chunks, -1, -1, -1)
        if all(cw.n == chunk_size for cw in chunk_weights_list) and n_chunks > 1:
            all_init = _batched_chunks_forward(all_h0, peg, pel, chunk_weights_list,
                                                chunk_sliding_flags, model.config, avg=avg)
            chunk_estimates = [all_init[c] for c in range(n_chunks)]
        else:
            chunk_estimates = []
            for c, cw in enumerate(chunk_weights_list):
                chunk_estimates.append(_fused_forward_chunk(h_init, peg, pel, cw, model.config,
                                                             is_sliding_majority=chunk_sliding_flags[c], avg=avg))
    else:
        chunk_estimates = [h_init.clone() for _ in range(n_chunks)]

    hs = [h_init] + list(chunk_estimates)

    for _ in range(K):
        if all(cw.n == chunk_size for cw in chunk_weights_list) and n_chunks > 1:
            all_h_in = torch.stack([hs[c] for c in range(n_chunks)])
            all_out = _batched_chunks_forward(all_h_in, peg, pel, chunk_weights_list,
                                               chunk_sliding_flags, model.config, avg=avg)
            chunk_outs = [all_out[c] for c in range(n_chunks)]
        else:
            chunk_outs = []
            for c, cw in enumerate(chunk_weights_list):
                chunk_outs.append(_fused_forward_chunk(hs[c], peg, pel, cw, model.config,
                                                        is_sliding_majority=chunk_sliding_flags[c], avg=avg))
        _idn_correction(chunk_outs, hs, h_init, n_chunks, elk_k)

    hidden = model.model.norm(hs[n_chunks])
    return model.lm_head(hidden).float()


# ---------------------------------------------------------------------------
# Data loading & benchmarking
# ---------------------------------------------------------------------------

def load_val_data(tokenizer, device, seq_len=128, max_tokens=1_000_000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = enc["input_ids"][0]
    batches = []
    offset = 0
    while offset + seq_len + 1 <= len(all_ids) and len(batches) * seq_len < max_tokens:
        batches.append(all_ids[offset:offset + seq_len + 1].unsqueeze(0).to(device))
        offset += seq_len
    return batches


def bench(fn, device, n_warm=50, n_run=150):
    for _ in range(n_warm):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--n-par", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--avg", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    use_avg = args.avg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        dtype=dtype,
    ).eval().to(device)

    L = model.config.num_hidden_layers
    n_par = args.n_par
    seq_layers = L - n_par
    par = list(range(seq_layers, L))

    print(f"Model: {args.model}, L={L}, D={model.config.hidden_size}, "
          f"H={model.config.num_attention_heads}, KV={model.config.num_key_value_heads}, "
          f"head_dim={model.config.head_dim}")
    print(f"n_par={n_par}, seq_layers={seq_layers}")
    for i in par:
        print(f"  Layer {i}: {'sliding' if model.model.layers[i].self_attn.is_sliding else 'FULL'}")

    print("Loading wikitext-2 validation data...")
    batches = load_val_data(tokenizer, device, args.seq_len, args.max_tokens)
    total_tokens = len(batches) * args.seq_len
    print(f"Val: {total_tokens:,} tokens, {len(batches)} batches, BS=1, seq_len={args.seq_len}")

    print("Precomputing weights...")
    pw = PrecomputedWeights(model, par)

    # Chunk configs
    chunk_configs = {}
    chunk_sliding = {}

    chunk_configs[f'1xF{n_par}'] = [ChunkWeights(model, par)]
    chunk_sliding[f'1xF{n_par}'] = [sum(1 for i in par if model.model.layers[i].self_attn.is_sliding) > len(par) // 2]

    chunk_configs[f'{n_par}xF1'] = [ChunkWeights(model, [li]) for li in par]
    chunk_sliding[f'{n_par}xF1'] = [model.model.layers[li].self_attn.is_sliding for li in par]

    candidate_N = set()
    candidate_N.add(n_par)
    if n_par >= 2:
        candidate_N.add(2)
    if n_par >= 4:
        candidate_N.add(4)
    if n_par % 2 == 0:
        candidate_N.add(n_par // 2)
    if n_par % 4 == 0:
        candidate_N.add(n_par // 4)

    for N in sorted(candidate_N):
        if N <= 0 or N > n_par or n_par % N != 0:
            continue
        M = n_par // N
        key = f'{N}xF{M}'
        if key not in chunk_configs:
            chunks = [ChunkWeights(model, par[i * M:(i + 1) * M]) for i in range(N)]
            chunk_configs[key] = chunks
            flags = []
            for i in range(N):
                chunk_layers = par[i * M:(i + 1) * M]
                n_sliding = sum(1 for li in chunk_layers if model.model.layers[li].self_attn.is_sliding)
                flags.append(n_sliding > M // 2)
            chunk_sliding[key] = flags

    print(f"Chunk configs: {list(chunk_configs.keys())}")

    # Sequential baseline
    idx_bench = batches[0][:, :-1]
    seq_ms = bench(lambda: forward_sequential(model, idx_bench), device)
    seq_losses = []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch in batches:
        idx, tgt = batch[:, :-1], batch[:, 1:]
        logits = forward_sequential(model, idx)
        seq_losses.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seq_ppl_s = time.perf_counter() - t0
    seq_ppl = math.exp(sum(seq_losses) / len(seq_losses))
    print(f"Sequential: {seq_ms:.2f}ms, PPL={seq_ppl:.2f}, ppl_eval={seq_ppl_s:.1f}s")

    # Determine fused sliding majority
    fused_sliding = sum(1 for i in par if model.model.layers[i].self_attn.is_sliding) > n_par // 2

    methods = {
        'IDN_batched': lambda idx, K, ek, init_: forward_idn_batched(
            model, idx, seq_layers, pw, K=K, elk_k=ek, init=init_),
        'Fused': lambda idx, K, ek, init_: forward_fused(
            model, idx, seq_layers, chunk_configs[f'1xF{n_par}'], K=K, elk_k=ek,
            avg=use_avg, is_sliding_majority=fused_sliding),
    }
    for cname, cws in chunk_configs.items():
        if len(set(cw.n for cw in cws)) == 1 and len(cws) > 1:
            flags = chunk_sliding[cname]
            methods[f'ChunkB_{cname}'] = lambda idx, K, ek, init_, _cws=cws, _fl=flags: forward_chunkwise_batched(
                model, idx, seq_layers, _cws, _fl, K=K, elk_k=ek, avg=use_avg, init=init_)

    K_values = [1, 2, 4, 8]
    init_strategies = ['h0', 'batch_fwd']

    results = {
        'seq_ppl': seq_ppl, 'seq_ms': seq_ms, 'seq_ppl_s': seq_ppl_s,
        'model': args.model, 'n_par': n_par, 'L': L, 'avg': use_avg,
    }

    for K in K_values:
        k_key = f'K={K}'
        results[k_key] = {}

        for method_name, method_fn in methods.items():
            results[k_key][method_name] = {}

            for init_name in init_strategies:
                print(f"  {k_key} {method_name} {init_name}...", end='', flush=True)

                ms = bench(lambda _mfn=method_fn, _K=K, _init=init_name:
                           _mfn(idx_bench, _K, 0.0, _init), device)

                losses = []
                for batch in batches:
                    idx, tgt = batch[:, :-1], batch[:, 1:]
                    logits = method_fn(idx, K, 0.0, init_name)
                    losses.append(F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
                ppl = math.exp(min(sum(losses) / len(losses), 20))

                speed = seq_ms / ms
                dppl = 100 * (ppl - seq_ppl) / seq_ppl

                results[k_key][method_name][init_name] = {
                    'ms': ms, 'speed': speed, 'ppl': ppl, 'dppl': dppl,
                }
                print(f" {ms:.1f}ms {speed:.2f}x PPL={ppl:.2f} ({dppl:+.1f}%)")

    outpath = args.output or f'cache/eval_logs/idn_gemma3_npar{n_par}.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    print(f"\n{'=' * 70}")
    print(f"Gemma3 IDN Eval: n_par={n_par} (seq PPL={seq_ppl:.2f}, seq_ms={seq_ms:.1f})")
    print(f"{'=' * 70}")
    for K in K_values:
        k_key = f'K={K}'
        if k_key not in results:
            continue
        print(f"\n**K={K}**\n")
        header = "| Method |"
        sep = "|--------|"
        for init_name in init_strategies:
            header += f" {init_name} |"
            sep += ":---:|"
        print(header)
        print(sep)
        for method_name in results[k_key]:
            row = f"| {method_name} |"
            for init_name in init_strategies:
                r = results[k_key][method_name].get(init_name, {})
                if r:
                    row += f" {r['ppl']:.1f} ({r['speed']:.2f}x) |"
                else:
                    row += " --- |"
            print(row)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
