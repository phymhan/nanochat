"""
Evaluate identity Newton (IDN) layer-parallel inference.

Compares three inference strategies:
  1. Sequential:   standard forward (ground truth)
  2. IDN (loop):   identity Newton K=1 via sequential Python loop
  3. IDN (batched): identity Newton K=1 with batched parallel layer evaluation
  4. Fused:        layer fusion (concatenated weights, single mega-matmul)

Reports PPL, top-1 agreement, cosine similarity, and wall-clock time.

Usage:
    python -m scripts.eval_jacobi_idn --model-tag d32_idn05_npar8
    python -m scripts.eval_jacobi_idn --model-tag d32_baseline --model-tag d32_idn05_npar8
    python -m scripts.eval_jacobi_idn --model-tag d32_idn05_npar8 --n-par 4,8,12
"""

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, norm, apply_rotary_emb
from nanochat.jacobi_forward import _embed, _run_block, _logits


# ---------------------------------------------------------------------------
# Model loading and data
# ---------------------------------------------------------------------------

def load_model(model_tag, step=None, device=None):
    from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, find_last_step
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    path = os.path.join(base_dir, 'base_checkpoints', model_tag)
    if step is None:
        step = find_last_step(path)
    print(f"  Loading {path} step {step}")
    md, _, meta = load_checkpoint(path, step, device)
    md = {k.removeprefix("_orig_mod."): v for k, v in md.items()}
    cfg = meta["model_config"]
    _patch_missing_config_keys(cfg)
    config = GPTConfig(**cfg)
    _patch_missing_keys(md, config)
    md = {k: v.to(device) for k, v in md.items()}
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(md, strict=True, assign=True)
    model.eval()
    return model


def load_batches(model, device, seq_len=128, batch_size=1, n_batches=8):
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    data_dir = os.path.join(base_dir, 'base_data_climbmix')
    try:
        import pyarrow.parquet as pq
        from nanochat.tokenizer import get_tokenizer
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
    except Exception as e:
        print(f"  Dataset load failed ({e}), using random tokens")
        vocab = model.config.vocab_size
        return [torch.randint(0, vocab, (batch_size, seq_len + 1), device=device) for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Inference strategies
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_sequential(model, idx):
    """Standard sequential forward (ground truth)."""
    return model.forward(idx)


@torch.inference_mode()
def forward_idn_loop(model, idx, seq_layers, K=1, elk_k=0.0):
    """
    Identity Newton via sequential loop, with optional Scale-ELK damping.

    Each iteration applies the DEER update with J≈I:
      s_t^{(i+1)} = f_t(s_{t-1}^{(i)}) + (s_{t-1}^{(i+1)} - s_{t-1}^{(i)})

    Equivalently via linear recurrence with a=(1-elk_k), b=f(h)-a*h:
      h_corr[t] = a * h_corr[t-1] + b[t]

    Scale-ELK (arXiv:2407.19115) damps the accumulation:
      elk_k=0 → full prefix-sum (standard identity Newton)
      elk_k=1 → no correction (all-on-same / Jacobi K=1)
      elk_k ∈ (0,1) → exponentially damped correction, stabilizes layers
                       far from prefix where J≈I breaks down
    """
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)

    # Sequential prefix
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
        h[i] = x

    # Initialize parallel layers with prefix output
    par = list(range(seq_layers, L))
    for i in par:
        h[i] = x.clone()

    a = 1.0 - elk_k  # damping factor: 1.0 = full IDN, 0.0 = all-on-same

    # K Newton iterations
    for _ in range(K):
        # Step 1: evaluate all parallel layers on current h[i-1]
        F_h = {}
        for i in par:
            F_h[i] = _run_block(model, i, h[i - 1], x0, cos_sin, ve)

        # Step 2: linear recurrence with damping
        # h_corr[t] = F_h[t] + a * (h_corr[t-1] - h_guess[t-1])
        # where h_guess[t] is the BEFORE-correction value for this iteration.
        # For a=1 (elk_k=0): full prefix-sum correction (standard IDN)
        # For a=0 (elk_k=1): h_corr[t] = F_h[t] (Jacobi / all-on-same)
        h_guess = {i: h[i] for i in par}  # pre-correction values
        # First parallel layer: h_corr[-1] = prefix output (exact), h_guess[-1] = prefix output
        # So correction = a * (prefix - prefix) = 0 → h_corr[0] = F_h[0]
        h_corr = F_h[par[0]]
        h[par[0]] = h_corr
        for i in par[1:]:
            h_corr = F_h[i] + a * (h_corr - h_guess[i - 1])
            h[i] = h_corr

    return _logits(model, h)


@torch.inference_mode()
def forward_idn_batched(model, idx, seq_layers, elk_k=0.0):
    """
    Identity Newton K=1 with truly batched parallel layer evaluation.

    All n_par parallel layers read the same input h_init.  We stack per-layer
    weights into (n_par, ...) tensors and use einsum for projections.  Attention
    is per-layer: reshape (n_par, B, H, T, D) → (n_par*B, H, T, D) for SDPA,
    so each layer's heads attend only within that layer (no cross-layer mixing).

    Same output as loop version (verified via max-diff check), with GPU-parallel
    execution of the n_par layers in a single set of matmuls.
    """
    L = model.config.n_layer
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // n_head
    n_kv_groups = n_head // n_kv_head

    x0, cos_sin, ve = _embed(model, idx)
    B, T, D = x0.shape
    cos, sin = cos_sin  # each (1, T, 1, head_dim//2)

    # Sequential prefix
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
        h[i] = x

    h_init = x  # (B, T, D)
    par = list(range(seq_layers, L))
    n_par = len(par)

    # --- Stack per-layer weights ---
    blocks = [model.transformer.h[i] for i in par]
    # Stack and cast weights to compute dtype (model stores float32, einsum needs matching dtypes)
    dtype = h_init.dtype
    W_q = torch.stack([b.attn.c_q.weight for b in blocks]).to(dtype)      # (n_par, n_head*hd, D)
    W_k = torch.stack([b.attn.c_k.weight for b in blocks]).to(dtype)      # (n_par, n_kv*hd, D)
    W_v = torch.stack([b.attn.c_v.weight for b in blocks]).to(dtype)      # (n_par, n_kv*hd, D)
    W_o = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dtype)    # (n_par, D, n_head*hd)
    W_fc = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dtype)      # (n_par, 4*D, D)
    W_proj = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dtype)  # (n_par, D, 4*D)

    # Per-layer resid/x0 lambdas — cast to compute dtype to match the loop path.
    # Without this, stacked float32 lambdas × bf16 hidden states promotes to float32
    # (PyTorch broadcasting rule), while the loop's scalar × tensor stays in bf16
    # (PyTorch scalar rule). The dtype mismatch causes large numerical differences.
    resid_lambdas = torch.stack([model.resid_lambdas[i] for i in par]).to(dtype)  # (n_par,)
    x0_lambdas = torch.stack([model.x0_lambdas[i] for i in par]).to(dtype)        # (n_par,)

    # --- Per-layer inputs: x_in[j] = resid_lambda[j] * h_init + x0_lambda[j] * x0 ---
    # (n_par, 1, 1, 1) * (B, T, D) → (n_par, B, T, D)
    all_x_in = (resid_lambdas.view(n_par, 1, 1, 1) * h_init.unsqueeze(0)
                + x0_lambdas.view(n_par, 1, 1, 1) * x0.unsqueeze(0))

    # --- RMS norm (no learnable weight in nanochat) ---
    all_x_normed = F.rms_norm(all_x_in, (D,))  # (n_par, B, T, D)

    # --- QKV projections via einsum ---
    all_q = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_q)  # (n_par, B, T, n_head*hd)
    all_k = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_k)  # (n_par, B, T, n_kv*hd)
    all_v = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_v)  # (n_par, B, T, n_kv*hd)

    # Reshape to (n_par, B, T, H, hd) → (n_par, B, H, T, hd)
    all_q = all_q.view(n_par, B, T, n_head, head_dim).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)

    # --- Value embeddings (ResFormer) ---
    # The model's attention receives norm(x_in) and uses x[..., :12] for the gate.
    # Must use all_x_normed (not all_x_in) to match the model's behavior.
    for j, li in enumerate(par):
        ve_j = ve.get(str(li))
        if ve_j is not None and blocks[j].attn.ve_gate is not None:
            ve_j = ve_j.view(B, T, n_kv_head, head_dim)  # (B, T, n_kv, hd)
            gate = 3 * torch.sigmoid(
                F.linear(all_x_normed[j, :, :, :12], blocks[j].attn.ve_gate.weight.to(dtype))
            )  # (B, T, n_kv)
            all_v[j] = all_v[j] + gate.unsqueeze(-1).transpose(1, 2) * ve_j.permute(0, 2, 1, 3)

    # --- RoPE ---
    # cos/sin are (1, T, 1, hd//2), need to broadcast over (n_par, B, H, T, hd//2)
    cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)  # (1, 1, 1, T, hd//2)
    sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

    def _rope_batched(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos_r + x2 * sin_r,
                          x1 * (-sin_r) + x2 * cos_r], dim=-1)

    all_q = _rope_batched(all_q)
    all_k = _rope_batched(all_k)

    # --- QK norm ---
    all_q = F.rms_norm(all_q, (head_dim,)) * 1.2
    all_k = F.rms_norm(all_k, (head_dim,)) * 1.2

    # --- GQA expand ---
    if n_kv_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(
            n_par, B, n_kv_head, n_kv_groups, T, head_dim
        ).reshape(n_par, B, n_head, T, head_dim)
        all_v = all_v[:, :, :, None, :, :].expand(
            n_par, B, n_kv_head, n_kv_groups, T, head_dim
        ).reshape(n_par, B, n_head, T, head_dim)

    # --- Attention: reshape (n_par, B, H, T, D) → (n_par*B, H, T, D) for SDPA ---
    # Each layer is a separate batch element — no cross-layer head mixing
    q_sdpa = all_q.reshape(n_par * B, n_head, T, head_dim)
    k_sdpa = all_k.reshape(n_par * B, n_head, T, head_dim)
    v_sdpa = all_v.reshape(n_par * B, n_head, T, head_dim)

    # Per-layer sliding window attention masks
    # Build a (n_par*B, 1, T, T) mask that handles different window sizes per layer
    windows = [model.window_sizes[i] for i in par]
    all_full = all(w[0] < 0 or w[0] >= T for w in windows)
    if all_full:
        y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
    else:
        row_idx = torch.arange(T, device=x0.device).unsqueeze(1)
        col_idx = torch.arange(T, device=x0.device).unsqueeze(0)
        causal = col_idx <= row_idx  # (T, T)
        masks = []
        for j in range(n_par):
            w = windows[j][0]
            if w < 0 or w >= T:
                m = causal  # full context
            else:
                m = causal & ((row_idx - col_idx) <= w)
            # Expand for B batch elements
            masks.append(m.unsqueeze(0).expand(B, -1, -1))
        # (n_par*B, T, T) → (n_par*B, 1, T, T) for SDPA
        attn_mask = torch.cat(masks, dim=0).unsqueeze(1)
        y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)

    # Reshape back: (n_par*B, H, T, hd) → (n_par, B, T, D)
    attn_out = y_sdpa.reshape(n_par, B, n_head, T, head_dim)
    attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)

    # --- Output projection ---
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, W_o)

    # --- First residual ---
    all_hidden = all_x_in + attn_out

    # --- MLP: norm → fc → relu² → proj ---
    all_h_normed = F.rms_norm(all_hidden, (D,))
    h_fc = torch.einsum('lbtd,lid->lbti', all_h_normed, W_fc)
    h_sq = F.relu(h_fc).square()
    mlp_out = torch.einsum('lbti,ldi->lbtd', h_sq, W_proj)

    # --- Second residual ---
    all_out = all_hidden + mlp_out  # (n_par, B, T, D)

    # --- Prefix-sum correction with Scale-ELK damping ---
    a = 1.0 - elk_k
    F_h = [all_out[j] for j in range(n_par)]
    res = [h_init - F_h[j] for j in range(n_par)]
    delta = [res[0]]
    for j in range(1, n_par):
        delta.append(res[j] + a * delta[j - 1])

    for j, li in enumerate(par):
        h[li] = h_init - delta[j]

    return _logits(model, h)


@torch.inference_mode()
def forward_fused(model, idx, seq_layers, fused_weights, n_par, n_head, head_dim):
    """
    Layer fusion: concatenated weights into single mega-matmul.

    WARNING: this averages per-layer resid_lambda/x0_lambda and runs a single
    shared attention over all concatenated heads. This is an approximation that
    differs from the exact identity Newton correction.
    """
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)

    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
    h_prefix = x

    W_q, W_k, W_v, W_o, W_fc, W_proj, avg_resid, avg_x0 = fused_weights
    B, T, D = h_prefix.shape
    total_heads = n_par * n_head

    x_in = avg_resid * h_prefix + avg_x0 * x0
    x_normed = norm(x_in)

    q = F.linear(x_normed, W_q.to(x_normed.dtype)).view(B, T, total_heads, head_dim)
    k = F.linear(x_normed, W_k.to(x_normed.dtype)).view(B, T, total_heads, head_dim)
    v = F.linear(x_normed, W_v.to(x_normed.dtype)).view(B, T, total_heads, head_dim)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    q = q * 1.2; k = k * 1.2

    q_t = q.transpose(1, 2); k_t = k.transpose(1, 2); v_t = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)

    attn_out = F.linear(y, W_o.to(y.dtype))
    x_mid = x_in + attn_out

    x_mid_normed = norm(x_mid)
    h_fc = F.linear(x_mid_normed, W_fc.to(x_mid_normed.dtype))
    h_sq = F.relu(h_fc).square()
    mlp_out = F.linear(h_sq, W_proj.to(h_sq.dtype))

    fused_out = x_mid + mlp_out

    h = [h_prefix] * L
    h[-1] = fused_out
    return _logits(model, h)


@torch.inference_mode()
def forward_fused_corrected(model, idx, seq_layers, n_par, n_head, head_dim, K=1, elk_k=0.0):
    """
    Fused QKV + per-layer O/MLP + Newton correction.

    Uses concatenated Q/K/V weights for one big SDPA (no cross-layer mixing
    since each head is independent), but splits the attention output per-layer
    for correct O projection, MLP, and residuals.  Then applies identity
    Newton prefix-sum correction.

    This combines fused's speed advantage (one big SDPA) with loop's correctness
    (per-layer projections and Newton correction).
    """
    L = model.config.n_layer
    n_kv_head = model.config.n_kv_head
    x0, cos_sin, ve = _embed(model, idx)

    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
    h_init = x

    par = list(range(seq_layers, L))
    B, T, D = h_init.shape

    # Initialize h
    h = [None] * L
    for i in range(seq_layers):
        h[i] = h_init
    for i in par:
        h[i] = h_init.clone()

    blocks = [model.transformer.h[i] for i in par]

    # Stack per-layer weights directly from model blocks
    dtype = h_init.dtype
    W_q_stacked = torch.stack([b.attn.c_q.weight for b in blocks]).to(dtype)
    W_k_stacked = torch.stack([b.attn.c_k.weight for b in blocks]).to(dtype)
    W_v_stacked = torch.stack([b.attn.c_v.weight for b in blocks]).to(dtype)
    W_o_stacked = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dtype)
    W_fc_stacked = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dtype)
    W_proj_stacked = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dtype)

    for _ in range(K):
        # --- Per-layer inputs (using current h[i-1]) ---
        all_x_in = []
        for j, li in enumerate(par):
            x_in_j = model.resid_lambdas[li] * h[li - 1] + model.x0_lambdas[li] * x0
            all_x_in.append(x_in_j)
        all_x_in = torch.stack(all_x_in)  # (n_par, B, T, D)

        # --- RMS norm ---
        all_x_normed = F.rms_norm(all_x_in, (D,))

        all_q = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_q_stacked)
        all_k = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_k_stacked)
        all_v = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_v_stacked)

        # Reshape to (n_par, B, H, T, hd)
        all_q = all_q.view(n_par, B, T, n_head, head_dim).permute(0, 1, 3, 2, 4)
        all_k = all_k.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)
        all_v = all_v.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)

        # --- Value embeddings (use normed input for gate, matching the model) ---
        for j, li in enumerate(par):
            ve_j = ve.get(str(li))
            if ve_j is not None and blocks[j].attn.ve_gate is not None:
                ve_j = ve_j.view(B, T, n_kv_head, head_dim)
                gate = 3 * torch.sigmoid(
                    F.linear(all_x_normed[j, :, :, :12], blocks[j].attn.ve_gate.weight.to(dtype))
                )
                all_v[j] = all_v[j] + gate.unsqueeze(-1).transpose(1, 2) * ve_j.permute(0, 2, 1, 3)

        # --- RoPE ---
        cos, sin = cos_sin
        cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)
        sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

        def _rope(x):
            d = x.shape[-1] // 2
            x1, x2 = x[..., :d], x[..., d:]
            return torch.cat([x1 * cos_r + x2 * sin_r,
                              x1 * (-sin_r) + x2 * cos_r], dim=-1)

        all_q = _rope(all_q)
        all_k = _rope(all_k)

        # --- QK norm ---
        all_q = F.rms_norm(all_q, (head_dim,)) * 1.2
        all_k = F.rms_norm(all_k, (head_dim,)) * 1.2

        # --- GQA expand ---
        n_kv_groups = n_head // n_kv_head
        if n_kv_groups > 1:
            all_k = all_k[:, :, :, None, :, :].expand(
                n_par, B, n_kv_head, n_kv_groups, T, head_dim
            ).reshape(n_par, B, n_head, T, head_dim)
            all_v = all_v[:, :, :, None, :, :].expand(
                n_par, B, n_kv_head, n_kv_groups, T, head_dim
            ).reshape(n_par, B, n_head, T, head_dim)

        # --- SDPA: (n_par*B, H, T, hd) — each layer is a separate batch element ---
        q_sdpa = all_q.reshape(n_par * B, n_head, T, head_dim)
        k_sdpa = all_k.reshape(n_par * B, n_head, T, head_dim)
        v_sdpa = all_v.reshape(n_par * B, n_head, T, head_dim)

        windows = [model.window_sizes[i] for i in par]
        all_full = all(w[0] < 0 or w[0] >= T for w in windows)
        if all_full:
            y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
        else:
            row_idx = torch.arange(T, device=x0.device).unsqueeze(1)
            col_idx = torch.arange(T, device=x0.device).unsqueeze(0)
            causal = col_idx <= row_idx
            masks = []
            for j in range(n_par):
                w = windows[j][0]
                m = causal if (w < 0 or w >= T) else (causal & ((row_idx - col_idx) <= w))
                masks.append(m.unsqueeze(0).expand(B, -1, -1))
            attn_mask = torch.cat(masks, dim=0).unsqueeze(1)
            y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)

        # Reshape back to (n_par, B, T, D)
        attn_out = y_sdpa.reshape(n_par, B, n_head, T, head_dim)
        attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)

        # --- Per-layer O projection via einsum ---
        attn_out = torch.einsum('lbtd,lod->lbto', attn_out, W_o_stacked)

        # --- First residual ---
        all_hidden = all_x_in + attn_out

        # --- Per-layer MLP ---
        all_h_normed = F.rms_norm(all_hidden, (D,))
        h_fc = torch.einsum('lbtd,lid->lbti', all_h_normed, W_fc_stacked)
        h_sq = F.relu(h_fc).square()
        mlp_out = torch.einsum('lbti,ldi->lbtd', h_sq, W_proj_stacked)

        # --- Second residual → per-layer block outputs ---
        all_out = all_hidden + mlp_out  # (n_par, B, T, D)

        # --- Newton correction with Scale-ELK damping ---
        a = 1.0 - elk_k
        F_h = [all_out[j] for j in range(n_par)]
        res = [h[par[j]] - F_h[j] for j in range(n_par)]
        delta = [res[0]]
        for j in range(1, n_par):
            delta.append(res[j] + a * delta[j - 1])
        for j, li in enumerate(par):
            h[li] = h[li] - delta[j]

    return _logits(model, h)


@torch.inference_mode()
def forward_fused_split_o_mlp(model, idx, seq_layers, fused_weights, n_par, n_head, head_dim, K=1, elk_k=0.0):
    """
    Fused QKV+SDPA (fast), then per-layer O projection + MLP + Newton correction.

    Keeps fused approach for QKV projections and SDPA (one big matmul, one big
    attention). After SDPA, splits the attention output per-layer (by head groups)
    for per-layer O projection, per-layer MLP, and Newton correction.

    Approximation: averaged input scalars, no per-layer value embeddings.
    But attention, O projection, and MLP are per-layer correct.
    """
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)

    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
    h_init = x

    par = list(range(seq_layers, L))
    B, T, D = h_init.shape
    W_q, W_k, W_v, W_o, W_fc, W_proj, avg_resid, avg_x0 = fused_weights
    total_heads = n_par * n_head
    dtype = h_init.dtype

    # Stack per-layer MLP weights
    blocks = [model.transformer.h[i] for i in par]
    W_fc_stacked = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dtype)    # (n_par, 4D, D)
    W_proj_stacked = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dtype) # (n_par, D, 4D)

    # Per-layer resid/x0 lambdas for the correction
    resid_lambdas = [model.resid_lambdas[i] for i in par]
    x0_lambdas = [model.x0_lambdas[i] for i in par]

    h = [None] * L
    for i in range(seq_layers):
        h[i] = h_init
    for i in par:
        h[i] = h_init.clone()

    for _ in range(K):
        # === Fused QKV + SDPA (same as original fused) ===
        x_in = avg_resid * h_init + avg_x0 * x0
        x_normed = norm(x_in)

        q = F.linear(x_normed, W_q.to(dtype)).view(B, T, total_heads, head_dim)
        k = F.linear(x_normed, W_k.to(dtype)).view(B, T, total_heads, head_dim)
        v = F.linear(x_normed, W_v.to(dtype)).view(B, T, total_heads, head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2; k = k * 1.2

        q_t = q.transpose(1, 2); k_t = k.transpose(1, 2); v_t = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        # y: (B, total_heads, T, head_dim)

        # === Split per-layer after SDPA ===
        # (B, n_par*H, T, hd) → (n_par, B, H, T, hd) → (n_par, B, T, D)
        y_split = y.view(B, n_par, n_head, T, head_dim).permute(1, 0, 3, 2, 4)
        y_split = y_split.reshape(n_par, B, T, n_head * head_dim)

        # Per-layer O projection via einsum
        W_o_stacked = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dtype)
        attn_out = torch.einsum('lbtd,lod->lbto', y_split, W_o_stacked)

        # Per-layer first residual
        x_in_expanded = x_in.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_hidden = x_in_expanded + attn_out

        # === Per-layer MLP via einsum ===
        all_h_normed = F.rms_norm(all_hidden, (D,))
        h_fc = torch.einsum('lbtd,lid->lbti', all_h_normed, W_fc_stacked)
        h_sq = F.relu(h_fc).square()
        mlp_out = torch.einsum('lbti,ldi->lbtd', h_sq, W_proj_stacked)

        # Per-layer block outputs
        all_out = all_hidden + mlp_out  # (n_par, B, T, D)

        # === Newton correction with Scale-ELK damping ===
        a = 1.0 - elk_k
        F_h = [all_out[j] for j in range(n_par)]
        res = [h[par[j]] - F_h[j] for j in range(n_par)]
        delta = [res[0]]
        for j in range(1, n_par):
            delta.append(res[j] + a * delta[j - 1])
        for j, li in enumerate(par):
            h[li] = h[li] - delta[j]

    return _logits(model, h)


@torch.inference_mode()
def forward_fused_split_mlp(model, idx, seq_layers, fused_weights, n_par, n_head, head_dim, K=1, elk_k=0.0):
    """
    Fused QKV+SDPA+O (all fast, one big matmul each), per-layer MLP + Newton correction.

    Keeps the full original fused pipeline through the attention residual (x_mid).
    Only the MLP is per-layer: each layer's MLP runs on the shared x_mid,
    producing n_par independent outputs for Newton prefix-sum correction.

    Approximation: averaged input scalars, fused O projection (mixes layers
    in the attention output). Per-layer MLP + Newton correction partially
    recovers from this.
    """
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)

    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
    h_init = x

    par = list(range(seq_layers, L))
    B, T, D = h_init.shape
    W_q, W_k, W_v, W_o, W_fc, W_proj, avg_resid, avg_x0 = fused_weights
    total_heads = n_par * n_head
    dtype = h_init.dtype

    blocks = [model.transformer.h[i] for i in par]
    W_fc_stacked = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dtype)
    W_proj_stacked = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dtype)

    h = [None] * L
    for i in range(seq_layers):
        h[i] = h_init
    for i in par:
        h[i] = h_init.clone()

    for _ in range(K):
        # === Fused QKV + SDPA + O (same as original fused) ===
        x_in = avg_resid * h_init + avg_x0 * x0
        x_normed = norm(x_in)

        q = F.linear(x_normed, W_q.to(dtype)).view(B, T, total_heads, head_dim)
        k = F.linear(x_normed, W_k.to(dtype)).view(B, T, total_heads, head_dim)
        v = F.linear(x_normed, W_v.to(dtype)).view(B, T, total_heads, head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2; k = k * 1.2

        q_t = q.transpose(1, 2); k_t = k.transpose(1, 2); v_t = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        attn_out = F.linear(y, W_o.to(dtype))  # fused O: (B, T, D)
        x_mid = x_in + attn_out                # shared across all layers

        # === Per-layer MLP from shared x_mid ===
        x_mid_expanded = x_mid.unsqueeze(0).expand(n_par, -1, -1, -1)  # (n_par, B, T, D)
        x_mid_normed = F.rms_norm(x_mid_expanded, (D,))

        h_fc = torch.einsum('lbtd,lid->lbti', x_mid_normed, W_fc_stacked)
        h_sq = F.relu(h_fc).square()
        mlp_out = torch.einsum('lbti,ldi->lbtd', h_sq, W_proj_stacked)

        all_out = x_mid_expanded + mlp_out  # (n_par, B, T, D)

        # === Newton correction with Scale-ELK damping ===
        a = 1.0 - elk_k
        F_h = [all_out[j] for j in range(n_par)]
        res = [h[par[j]] - F_h[j] for j in range(n_par)]
        delta = [res[0]]
        for j in range(1, n_par):
            delta.append(res[j] + a * delta[j - 1])
        for j, li in enumerate(par):
            h[li] = h[li] - delta[j]

    return _logits(model, h)


def build_fused_weights(model, seq_layers):
    """Concatenate weights from parallel layers into one mega-block."""
    L = model.config.n_layer
    n_par = L - seq_layers

    W_q = torch.cat([model.transformer.h[i].attn.c_q.weight for i in range(seq_layers, L)], dim=0)
    W_k = torch.cat([model.transformer.h[i].attn.c_k.weight for i in range(seq_layers, L)], dim=0)
    W_v = torch.cat([model.transformer.h[i].attn.c_v.weight for i in range(seq_layers, L)], dim=0)
    W_o = torch.cat([model.transformer.h[i].attn.c_proj.weight for i in range(seq_layers, L)], dim=1)
    W_fc = torch.cat([model.transformer.h[i].mlp.c_fc.weight for i in range(seq_layers, L)], dim=0)
    W_proj = torch.cat([model.transformer.h[i].mlp.c_proj.weight for i in range(seq_layers, L)], dim=1)

    avg_resid = sum(model.resid_lambdas[i].item() for i in range(seq_layers, L)) / n_par
    avg_x0 = sum(model.x0_lambdas[i].item() for i in range(seq_layers, L)) / n_par

    return W_q, W_k, W_v, W_o, W_fc, W_proj, avg_resid, avg_x0


# ---------------------------------------------------------------------------
# Metrics and timing
# ---------------------------------------------------------------------------

def compute_metrics(logits_ref, logits_test, targets):
    loss = F.cross_entropy(logits_test.reshape(-1, logits_test.size(-1)),
                           targets.reshape(-1), reduction='mean').item()
    ppl = math.exp(min(loss, 20.0))
    r = logits_ref.reshape(-1, logits_ref.size(-1))
    t = logits_test.reshape(-1, logits_test.size(-1))
    top1 = (r.argmax(-1) == t.argmax(-1)).float().mean().item()
    cs = F.cosine_similarity(r, t, dim=-1)
    cos_sim = cs[~cs.isnan()].mean().item() if not cs.isnan().all() else 0.0
    return ppl, top1, cos_sim


def bench(fn, device, n_warm=10, n_run=30):
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

def _eval_method(fn, batches, device):
    """Evaluate a method across batches, return (avg_ppl, avg_top1, avg_cos_sim, timing_ms)."""
    idx_bench = batches[0][:, :-1]
    ms = bench(fn, device) if callable(fn) else 0.0
    ppls, top1s, css = [], [], []
    for batch in batches:
        idx, targets = batch[:, :-1], batch[:, 1:]
        ref = forward_sequential(None, idx) if False else None  # placeholder
        test = fn()  # placeholder
    # This is filled in by evaluate_model below
    return ppls, top1s, css, ms


def evaluate_model(model, batches, n_par_list, device, K=1, elk_k=0.0):
    L = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head

    idx_bench = batches[0][:, :-1]

    # Sequential
    seq_ms = bench(lambda: forward_sequential(model, idx_bench), device)
    seq_ppls = []
    for batch in batches:
        idx, targets = batch[:, :-1], batch[:, 1:]
        logits = forward_sequential(model, idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='mean').item()
        seq_ppls.append(math.exp(min(loss, 20.0)))
    seq_ppl = sum(seq_ppls) / len(seq_ppls)

    print(f"  Sequential: {seq_ms:.2f}ms, PPL={seq_ppl:.2f}, K={K}")
    print()

    for n_par in n_par_list:
        if n_par >= L:
            continue
        seq_layers = L - n_par
        fw = build_fused_weights(model, seq_layers)

        # Define all methods with their eval functions
        method_defs = [
            ('IDN (loop)',      lambda: forward_idn_loop(model, idx_bench, seq_layers, K=K, elk_k=elk_k)),
            ('IDN (batched)',   lambda: forward_idn_batched(model, idx_bench, seq_layers, elk_k=elk_k)),
            ('Fused',           lambda: forward_fused(model, idx_bench, seq_layers, fw, n_par, n_head, head_dim)),
            ('Fused+corr',      lambda: forward_fused_corrected(model, idx_bench, seq_layers, n_par, n_head, head_dim, K=K, elk_k=elk_k)),
            ('Fused+split_o_mlp',     lambda: forward_fused_split_o_mlp(model, idx_bench, seq_layers, fw, n_par, n_head, head_dim, K=K, elk_k=elk_k)),
            ('Fused+split_mlp',       lambda: forward_fused_split_mlp(model, idx_bench, seq_layers, fw, n_par, n_head, head_dim, K=K, elk_k=elk_k)),
        ]

        # Timing
        timings = {}
        for name, fn in method_defs:
            timings[name] = bench(fn, device)

        # Quality
        results = {}
        eval_fns = {
            'IDN (loop)':    lambda idx: forward_idn_loop(model, idx, seq_layers, K=K, elk_k=elk_k),
            'IDN (batched)': lambda idx: forward_idn_batched(model, idx, seq_layers, elk_k=elk_k),
            'Fused':         lambda idx: forward_fused(model, idx, seq_layers, fw, n_par, n_head, head_dim),
            'Fused+corr':    lambda idx: forward_fused_corrected(model, idx, seq_layers, n_par, n_head, head_dim, K=K, elk_k=elk_k),
            'Fused+split_o_mlp':   lambda idx: forward_fused_split_o_mlp(model, idx, seq_layers, fw, n_par, n_head, head_dim, K=K, elk_k=elk_k),
            'Fused+split_mlp':     lambda idx: forward_fused_split_mlp(model, idx, seq_layers, fw, n_par, n_head, head_dim, K=K, elk_k=elk_k),
        }
        for name, fn in eval_fns.items():
            ppls, top1s, css = [], [], []
            for batch in batches:
                idx, targets = batch[:, :-1], batch[:, 1:]
                ref = forward_sequential(model, idx)
                test = fn(idx)
                ppl, top1, cs = compute_metrics(ref, test, targets)
                ppls.append(ppl); top1s.append(top1); css.append(cs)
            results[name] = (sum(ppls)/len(ppls), sum(top1s)/len(top1s), sum(css)/len(css))

        # Print
        print(f"  n_par={n_par} (seq={seq_layers} + {n_par} parallel, K={K})")
        print(f"  {'Method':<16s} | {'Time':>7s} | {'Speed':>6s} | {'PPL':>8s} | {'ΔPPL%':>7s} | {'Top-1':>6s} | {'CosSim':>7s}")
        print(f"  {'-'*16}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
        print(f"  {'Sequential':<16s} | {seq_ms:6.2f}ms | {'1.00x':>6s} | {seq_ppl:8.2f} | {'—':>7s} | {'—':>6s} | {'—':>7s}")
        for name in ['IDN (loop)', 'IDN (batched)', 'Fused', 'Fused+corr', 'Fused+split_o_mlp', 'Fused+split_mlp']:
            ms = timings[name]
            ppl, top1, cs = results[name]
            dppl = 100 * (ppl - seq_ppl) / seq_ppl
            speedup = seq_ms / ms
            ppl_s = f"{ppl:8.2f}" if ppl < 1e5 else f"{ppl:8.0f}"
            print(f"  {name:<16s} | {ms:6.2f}ms | {speedup:5.2f}x | {ppl_s} | {dppl:+6.1f}% | {top1:6.3f} | {cs:7.4f}")

        # Debug diffs
        idx = batches[0][:, :-1]
        out_loop = forward_idn_loop(model, idx, seq_layers, K=K, elk_k=elk_k)
        out_batched = forward_idn_batched(model, idx, seq_layers, elk_k=elk_k)
        out_fused = forward_fused(model, idx, seq_layers, fw, n_par, n_head, head_dim)
        out_fc = forward_fused_corrected(model, idx, seq_layers, n_par, n_head, head_dim, K=K, elk_k=elk_k)
        out_fso = forward_fused_split_o_mlp(model, idx, seq_layers, fw, n_par, n_head, head_dim, K=K, elk_k=elk_k)
        out_fsm = forward_fused_split_mlp(model, idx, seq_layers, fw, n_par, n_head, head_dim, K=K, elk_k=elk_k)
        print(f"\n  Debug max diff vs loop:")
        print(f"    batched:          {(out_loop.float() - out_batched.float()).abs().max().item():.2e}")
        print(f"    fused:            {(out_loop.float() - out_fused.float()).abs().max().item():.2e}")
        print(f"    fused+corr:       {(out_loop.float() - out_fc.float()).abs().max().item():.2e}")
        print(f"    fused+split_o_mlp:{(out_loop.float() - out_fso.float()).abs().max().item():.2e}")
        print(f"    fused+split_mlp:  {(out_loop.float() - out_fsm.float()).abs().max().item():.2e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate IDN layer-parallel inference")
    parser.add_argument("--model-tag", type=str, action='append', required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--n-par", type=str, default="4,8",
                        help="comma-separated n_par values")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--K", type=int, default=1, help="number of Newton iterations")
    parser.add_argument("--elk-k", type=float, default=0.0, help="Scale-ELK damping: 0=full IDN, 1=all-on-same, (0,1)=damped")
    args = parser.parse_args()

    # Force SDPA (FA3 segfaults in some eval paths)
    import nanochat.flash_attention as fa
    fa._override_impl = 'sdpa'; fa.USE_FA3 = False

    device = torch.device(autodetect_device_type())
    n_par_list = [int(x) for x in args.n_par.split(',')]

    for tag in args.model_tag:
        print(f"\n{'='*80}")
        print(f"  Model: {tag}")
        print(f"{'='*80}")
        model = load_model(tag, step=args.step, device=device)
        batches = load_batches(model, device, args.seq_len, args.batch_size, args.n_batches)
        evaluate_model(model, batches, n_par_list, device, K=args.K, elk_k=args.elk_k)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
