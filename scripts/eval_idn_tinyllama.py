"""
IDN layer-parallel inference evaluation for TinyLlama 1.1B.

TinyLlama uses standard LLaMA architecture: standard residuals, no QKV biases,
SwiGLU MLP, GQA (H=32, KV=4), L=22, D=2048.

Methods: Sequential, IDN batched, Fused, Chunkwise batched
Init strategies: h0, batch_fwd
Evaluations: PPL (wikitext-2), timing

Usage:
    HF_HOME=/mnt/nvme0n1/ligong/huggingface CUDA_VISIBLE_DEVICES=4 \
        uv run --extra gpu --group dev python -m scripts.eval_idn_tinyllama --n-par 8
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
    x_f = x.float()
    x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_norm * weight).to(x.dtype)


def _apply_rope_batched(x, cos, sin):
    """RoPE for batched tensors. x: (L, B, H, T, hd), cos/sin: (1, B, 1, T, hd)."""
    hd_half = x.shape[-1] // 2
    x1, x2 = x[..., :hd_half], x[..., hd_half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _apply_rope(x, cos, sin):
    """RoPE for single tensors. x: (B, T, H, hd), cos/sin: (B, T, hd)."""
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

        self.ln1_w = torch.stack([l.input_layernorm.weight for l in layers]).to(dt)
        self.ln2_w = torch.stack([l.post_attention_layernorm.weight for l in layers]).to(dt)

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

        self.ln1_w = torch.stack([l.input_layernorm.weight for l in layers]).mean(0).to(dt)
        self.ln2_w = torch.stack([l.post_attention_layernorm.weight for l in layers]).mean(0).to(dt)

        self.layers = chunk_layers
        self.n = n


# ---------------------------------------------------------------------------
# Core forward functions
# ---------------------------------------------------------------------------

def _batched_forward(all_h_prev, position_embeddings, pw, config):
    """Batched forward for n_par LLaMA layers. Returns (n_par, B, T, D)."""
    n_par, B, T, D = all_h_prev.shape
    eps = config.rms_norm_eps
    n_head = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = D // n_head
    n_groups = n_head // n_kv

    x_norm = _rms_norm(all_h_prev, pw.ln1_w[:, None, None, :], eps)

    all_q = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_q)
    all_k = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_k)
    all_v = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_v)

    all_q = all_q.view(n_par, B, T, n_head, hd).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)

    cos, sin = position_embeddings
    cos_r = cos.unsqueeze(0).unsqueeze(2)
    sin_r = sin.unsqueeze(0).unsqueeze(2)
    all_q = _apply_rope_batched(all_q, cos_r, sin_r)
    all_k = _apply_rope_batched(all_k, cos_r, sin_r)

    if n_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(
            n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)
        all_v = all_v[:, :, :, None, :, :].expand(
            n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)

    q_s = all_q.reshape(n_par * B, n_head, T, hd)
    k_s = all_k.reshape(n_par * B, n_head, T, hd)
    v_s = all_v.reshape(n_par * B, n_head, T, hd)
    y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

    attn_out = y_s.reshape(n_par, B, n_head, T, hd).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o)

    hidden = all_h_prev + attn_out

    h_norm = _rms_norm(hidden, pw.ln2_w[:, None, None, :], eps)
    gate = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_gate)
    up = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_up)
    mlp = torch.einsum('lbti,ldi->lbtd', F.silu(gate) * up, pw.W_down)

    return hidden + mlp


def _fused_forward_chunk(h_in, position_embeddings, cw, config, avg=False):
    """Fused chunk forward. Returns (B, T, D)."""
    B, T, D = h_in.shape
    eps = config.rms_norm_eps
    n_head = config.num_attention_heads
    n_kv = config.num_key_value_heads
    total_q = cw.n * n_head
    total_kv = cw.n * n_kv
    hd = D // n_head
    n_groups = n_head // n_kv

    x_norm = _rms_norm(h_in, cw.ln1_w, eps)

    q = F.linear(x_norm, cw.W_q).view(B, T, total_q, hd)
    k = F.linear(x_norm, cw.W_k).view(B, T, total_kv, hd)
    v = F.linear(x_norm, cw.W_v).view(B, T, total_kv, hd)

    cos, sin = position_embeddings
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if n_groups > 1:
        k = k[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)
        v = v[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)

    attn_out = F.linear(y, cw.W_o)
    if avg and cw.n > 1:
        attn_out = attn_out / cw.n
    x_mid = h_in + attn_out

    h_norm = _rms_norm(x_mid, cw.ln2_w, eps)
    gate = F.linear(h_norm, cw.W_gate)
    up = F.linear(h_norm, cw.W_up)
    mlp = F.linear(F.silu(gate) * up, cw.W_down)
    if avg and cw.n > 1:
        mlp = mlp / cw.n
    return x_mid + mlp


def _batched_chunks_forward(all_h_in, position_embeddings, chunk_weights_list, config, avg=False):
    """Batched forward for C equal-size chunks. Returns (C, B, T, D)."""
    C = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n
    B, T, D = all_h_in.shape[1:]
    eps = config.rms_norm_eps
    n_head = config.num_attention_heads
    n_kv = config.num_key_value_heads
    q_per_chunk = chunk_size * n_head
    kv_per_chunk = chunk_size * n_kv
    hd = D // n_head
    n_groups = n_head // n_kv

    W_q = torch.stack([cw.W_q for cw in chunk_weights_list])
    W_k = torch.stack([cw.W_k for cw in chunk_weights_list])
    W_v = torch.stack([cw.W_v for cw in chunk_weights_list])
    W_o = torch.stack([cw.W_o for cw in chunk_weights_list])
    W_gate = torch.stack([cw.W_gate for cw in chunk_weights_list])
    W_up = torch.stack([cw.W_up for cw in chunk_weights_list])
    W_down = torch.stack([cw.W_down for cw in chunk_weights_list])

    ln1_w = torch.stack([cw.ln1_w for cw in chunk_weights_list])
    ln2_w = torch.stack([cw.ln2_w for cw in chunk_weights_list])

    x_norm = _rms_norm(all_h_in, ln1_w[:, None, None, :], eps)

    all_q = torch.einsum('cbtd,chd->cbth', x_norm, W_q)
    all_k = torch.einsum('cbtd,chd->cbth', x_norm, W_k)
    all_v = torch.einsum('cbtd,chd->cbth', x_norm, W_v)

    all_q = all_q.view(C, B, T, q_per_chunk, hd).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)

    cos, sin = position_embeddings
    cos_r = cos.unsqueeze(0).unsqueeze(2)
    sin_r = sin.unsqueeze(0).unsqueeze(2)
    all_q = _apply_rope_batched(all_q, cos_r, sin_r)
    all_k = _apply_rope_batched(all_k, cos_r, sin_r)

    if n_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(
            C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)
        all_v = all_v[:, :, :, None, :, :].expand(
            C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)

    q_s = all_q.reshape(C * B, q_per_chunk, T, hd)
    k_s = all_k.reshape(C * B, q_per_chunk, T, hd)
    v_s = all_v.reshape(C * B, q_per_chunk, T, hd)
    y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

    y = y_s.reshape(C, B, q_per_chunk, T, hd).permute(0, 1, 3, 2, 4)
    y = y.reshape(C, B, T, q_per_chunk * hd)

    attn_out = torch.einsum('cbth,cdh->cbtd', y, W_o)
    if avg and chunk_size > 1:
        attn_out = attn_out / chunk_size

    hidden = all_h_in + attn_out

    h_norm = _rms_norm(hidden, ln2_w[:, None, None, :], eps)
    gate = torch.einsum('cbtd,cid->cbti', h_norm, W_gate)
    up = torch.einsum('cbtd,cid->cbti', h_norm, W_up)
    mlp = torch.einsum('cbti,cdi->cbtd', F.silu(gate) * up, W_down)
    if avg and chunk_size > 1:
        mlp = mlp / chunk_size

    return hidden + mlp


def _idn_correction(all_out, hs, h_init, n_units, elk_k=0.0):
    """IDN prefix-sum correction. Updates hs in-place."""
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
    """Run embedding + layers 0..seq_layers-1. Returns (h_init, position_embeddings)."""
    hidden = model.model.embed_tokens(input_ids)
    B, T, D = hidden.shape
    position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)

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
                mask = causal_mask.get(getattr(layer, 'attention_type', 'full_attention'), None)
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

    return hidden, position_embeddings


# ---------------------------------------------------------------------------
# High-level forward functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_sequential(model, input_ids):
    return model(input_ids).logits.float()


@torch.inference_mode()
def forward_idn_batched(model, input_ids, seq_layers, pw, K=1, elk_k=0.0,
                         init='h0'):
    h_init, pos_emb = _run_prefix(model, input_ids, seq_layers)
    n_par = pw.n_par

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward(all_h0, pos_emb, pw, model.config)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    else:
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _ in range(K):
        all_hp = torch.stack([hs[j] for j in range(n_par)])
        all_out = _batched_forward(all_hp, pos_emb, pw, model.config)
        _idn_correction(all_out, hs, h_init, n_par, elk_k)

    hidden = model.model.norm(hs[n_par])
    return model.lm_head(hidden).float()


@torch.inference_mode()
def forward_fused(model, input_ids, seq_layers, chunk_weights, K=1, elk_k=0.0,
                   avg=False):
    h_init, pos_emb = _run_prefix(model, input_ids, seq_layers)

    out = _fused_forward_chunk(h_init, pos_emb, chunk_weights[0], model.config, avg=avg)
    for _ in range(1, K):
        out = _fused_forward_chunk(out, pos_emb, chunk_weights[0], model.config, avg=avg)

    hidden = model.model.norm(out)
    return model.lm_head(hidden).float()


@torch.inference_mode()
def forward_chunkwise_batched(model, input_ids, seq_layers, chunk_weights_list,
                                K=1, elk_k=0.0, avg=False, init='h0'):
    h_init, pos_emb = _run_prefix(model, input_ids, seq_layers)
    n_chunks = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_chunks, -1, -1, -1)
        if all(cw.n == chunk_size for cw in chunk_weights_list) and n_chunks > 1:
            all_init = _batched_chunks_forward(all_h0, pos_emb, chunk_weights_list, model.config, avg=avg)
            chunk_estimates = [all_init[c] for c in range(n_chunks)]
        else:
            chunk_estimates = [_fused_forward_chunk(h_init, pos_emb, cw, model.config, avg=avg)
                               for cw in chunk_weights_list]
    else:
        chunk_estimates = [h_init.clone() for _ in range(n_chunks)]

    hs = [h_init] + list(chunk_estimates)

    for _ in range(K):
        if all(cw.n == chunk_size for cw in chunk_weights_list) and n_chunks > 1:
            all_h_in = torch.stack([hs[c] for c in range(n_chunks)])
            all_out = _batched_chunks_forward(all_h_in, pos_emb, chunk_weights_list, model.config, avg=avg)
            chunk_outs = [all_out[c] for c in range(n_chunks)]
        else:
            chunk_outs = [_fused_forward_chunk(hs[c], pos_emb, cw, model.config, avg=avg)
                          for c, cw in enumerate(chunk_weights_list)]
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
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
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
          f"H={model.config.num_attention_heads}, KV={model.config.num_key_value_heads}")
    print(f"n_par={n_par}, seq_layers={seq_layers}")

    print("Loading wikitext-2 validation data...")
    batches = load_val_data(tokenizer, device, args.seq_len, args.max_tokens)
    total_tokens = len(batches) * args.seq_len
    print(f"Val: {total_tokens:,} tokens, {len(batches)} batches, BS=1, seq_len={args.seq_len}")

    print("Precomputing weights...")
    pw = PrecomputedWeights(model, par)

    chunk_configs = {}
    chunk_configs[f'1xF{n_par}'] = [ChunkWeights(model, par)]
    chunk_configs[f'{n_par}xF1'] = [ChunkWeights(model, [li]) for li in par]

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
            chunk_configs[key] = [
                ChunkWeights(model, par[i * M:(i + 1) * M]) for i in range(N)
            ]

    print(f"Chunk configs: {list(chunk_configs.keys())}")

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

    methods = {
        'IDN_batched': lambda idx, K, ek, init_: forward_idn_batched(
            model, idx, seq_layers, pw, K=K, elk_k=ek, init=init_),
        'Fused': lambda idx, K, ek, init_: forward_fused(
            model, idx, seq_layers, chunk_configs[f'1xF{n_par}'], K=K, elk_k=ek, avg=use_avg),
    }
    for cname, cws in chunk_configs.items():
        if len(set(cw.n for cw in cws)) == 1 and len(cws) > 1:
            methods[f'ChunkB_{cname}'] = lambda idx, K, ek, init_, _cws=cws: forward_chunkwise_batched(
                model, idx, seq_layers, _cws, K=K, elk_k=ek, avg=use_avg, init=init_)

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

    outpath = args.output or f'cache/eval_logs/idn_tinyllama_npar{n_par}.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    print(f"\n{'=' * 70}")
    print(f"TinyLlama IDN Eval: n_par={n_par} (seq PPL={seq_ppl:.2f}, seq_ms={seq_ms:.1f})")
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
