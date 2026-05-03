"""AR generation v3: use _batched_block_forward_kv with extracted KV cache."""
import os, sys, time, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/workspace/home/ligong/code/nanochat")
import torch.nn.functional as F
import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'; fa.USE_FA3 = False
from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import norm, apply_rotary_emb
from nanochat.engine import KVCache
from nanochat.jacobi_forward import _embed
from scripts.eval_jacobi_idn import load_model
from scripts.eval_jacobi_diag import associative_scan_diagonal

device = torch.device(autodetect_device_type())
dtype = COMPUTE_DTYPE

def _batched_block_forward_kv(all_h_prev, x0, cos_sin, ve, par, model,
                               cached_k=None, cached_v=None):
    """Batched forward with KV cache. Returns (output, new_k, new_v)."""
    n_par = len(par)
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // n_head
    n_kv_groups = n_head // n_kv_head
    B, T, D = x0.shape
    cos, sin = cos_sin
    blocks = [model.transformer.h[i] for i in par]
    dt = all_h_prev.dtype

    W_q = torch.stack([b.attn.c_q.weight for b in blocks]).to(dt)
    W_k = torch.stack([b.attn.c_k.weight for b in blocks]).to(dt)
    W_v = torch.stack([b.attn.c_v.weight for b in blocks]).to(dt)
    W_o = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dt)
    W_fc = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dt)
    W_proj = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dt)

    rl = torch.stack([model.resid_lambdas[i] for i in par]).to(dt).view(n_par,1,1,1)
    xl = torch.stack([model.x0_lambdas[i] for i in par]).to(dt).view(n_par,1,1,1)
    all_x_in = rl * all_h_prev + xl * x0.unsqueeze(0)
    all_x_normed = F.rms_norm(all_x_in, (D,))

    all_q = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_q)
    all_k = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_k)
    all_v = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_v)

    all_q = all_q.view(n_par,B,T,n_head,head_dim).permute(0,1,3,2,4)    # (L,B,H,T,hd)
    all_k = all_k.view(n_par,B,T,n_kv_head,head_dim).permute(0,1,3,2,4) # (L,B,Hkv,T,hd)
    all_v = all_v.view(n_par,B,T,n_kv_head,head_dim).permute(0,1,3,2,4)

    # Value embeddings
    v_list = list(all_v.unbind(0))
    for j, li in enumerate(par):
        ve_j = ve.get(str(li))
        if ve_j is not None and blocks[j].attn.ve_gate is not None:
            ve_r = ve_j.view(B,T,n_kv_head,head_dim)
            gate = 3*torch.sigmoid(F.linear(all_x_normed[j,:,:,:12], blocks[j].attn.ve_gate.weight.to(dt)))
            v_list[j] = v_list[j] + gate.unsqueeze(-1).transpose(1,2) * ve_r.permute(0,2,1,3)
    all_v = torch.stack(v_list)

    # RoPE
    cos_r = cos.unsqueeze(0).permute(0,1,3,2,4)
    sin_r = sin.unsqueeze(0).permute(0,1,3,2,4)
    def _rope(x):
        d=x.shape[-1]//2; x1,x2=x[...,:d],x[...,d:]
        return torch.cat([x1*cos_r+x2*sin_r, x1*(-sin_r)+x2*cos_r], dim=-1)
    all_q = _rope(all_q)
    all_k = _rope(all_k)
    all_q = F.rms_norm(all_q,(head_dim,))*1.2
    all_k = F.rms_norm(all_k,(head_dim,))*1.2

    # Save new K,V for cache (post-RoPE, post-norm, pre-GQA)
    # Shape: (n_par, B, n_kv_head, T, head_dim)
    new_k = all_k.clone()
    new_v = all_v.clone()

    # Concat cached KV
    # cached_k/v from KV cache: (n_par, B, S, n_kv_head, head_dim) — FA3 layout (B,T,H,D)
    # Need to transpose to (n_par, B, n_kv_head, S, head_dim) to match our layout
    if cached_k is not None and cached_k.shape[2] > 0:
        ck = cached_k.transpose(2, 3)  # (n_par, B, n_kv_head, S, hd)
        cv = cached_v.transpose(2, 3)
        all_k = torch.cat([ck, all_k], dim=-2)
        all_v = torch.cat([cv, all_v], dim=-2)

    S_plus_T = all_k.shape[-2]
    if n_kv_groups > 1:
        all_k = all_k[:,:,:,None,:,:].expand(n_par,B,n_kv_head,n_kv_groups,S_plus_T,head_dim).reshape(n_par,B,n_head,S_plus_T,head_dim)
        all_v = all_v[:,:,:,None,:,:].expand(n_par,B,n_kv_head,n_kv_groups,S_plus_T,head_dim).reshape(n_par,B,n_head,S_plus_T,head_dim)

    # Attention with explicit matmul
    scaling = head_dim ** -0.5
    attn_w = torch.matmul(all_q, all_k.transpose(-2,-1)) * scaling  # (L,B,H,T,S+T)
    if T > 1:
        ri = torch.arange(T,device=device).unsqueeze(1)
        ci = torch.arange(S_plus_T,device=device).unsqueeze(0)
        S = S_plus_T - T
        causal = ci <= (ri + S)
        attn_w = attn_w.masked_fill(~causal[None,None,None], float('-inf'))
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(dt)
    attn_out = torch.matmul(attn_w, all_v)

    attn_out = attn_out.permute(0,1,3,2,4).reshape(n_par,B,T,D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, W_o)
    all_hidden = all_x_in + attn_out

    h_n = F.rms_norm(all_hidden, (D,))
    hf = torch.einsum('lbtd,lid->lbti', h_n, W_fc)
    mlp = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), W_proj)

    return all_hidden + mlp, new_k, new_v


def generate_sequential(model, prompt_ids, max_new=20):
    L = model.config.n_layer
    kv = KVCache(1, model.config.n_kv_head, len(prompt_ids)+max_new+1,
                 model.config.n_embd//model.config.n_head, L, device, dtype)
    ids = torch.tensor([prompt_ids], device=device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    logits = model.forward(ids, kv_cache=kv)
    torch.cuda.synchronize(); pf_t = time.perf_counter()-t0
    gen = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(max_new):
        tok = logits[0,-1].argmax().item(); gen.append(tok)
        logits = model.forward(torch.tensor([[tok]],device=device), kv_cache=kv)
    torch.cuda.synchronize(); dec_t = time.perf_counter()-t0
    return gen, pf_t, dec_t


def generate_parallel(model, prompt_ids, max_new=20, n_par=8, K=1, method='idn', n_probes=1):
    L = model.config.n_layer
    seq_layers = L - n_par
    par = list(range(seq_layers, L))
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // model.config.n_head

    kv = KVCache(1, n_kv_head, len(prompt_ids)+max_new+1, head_dim, L, device, dtype)
    ids = torch.tensor([prompt_ids], device=device)

    # Prefill sequential
    torch.cuda.synchronize(); t0 = time.perf_counter()
    logits = model.forward(ids, kv_cache=kv)
    torch.cuda.synchronize(); pf_t = time.perf_counter()-t0

    # Get warm-start hidden states from prefill
    x0_pf, cs_pf, ve_pf = _embed(model, ids)
    prev_h = [None]*L; x = x0_pf
    for i in range(L):
        x_in = model.resid_lambdas[i]*x + model.x0_lambdas[i]*x0_pf
        x = model.transformer.h[i](x_in, ve_pf.get(str(i)), cs_pf, model.window_sizes[i], None)
        prev_h[i] = x[:,-1:,:].clone()

    gen = []
    torch.cuda.synchronize(); t0 = time.perf_counter()

    for step in range(max_new):
        tok = logits[0,-1].argmax().item(); gen.append(tok)
        ids_step = torch.tensor([[tok]], device=device)

        x0 = model.transformer.wte(ids_step).to(dtype); x0 = norm(x0)
        if kv.prev_embedding is not None:
            gate = model.smear_lambda.to(dtype)*torch.sigmoid(model.smear_gate(x0[:,:,:24]))
            x0 = x0 + gate * kv.prev_embedding

        pos = kv.get_pos()
        cos_sin = model.cos[:,pos:pos+1], model.sin[:,pos:pos+1]
        ve = {k: model.value_embeds[k](ids_step).to(dtype) for k in model.value_embeds}

        # Sequential prefix with KV cache
        x = x0; h_step = [None]*L
        for i in range(seq_layers):
            x_in = model.resid_lambdas[i]*x + model.x0_lambdas[i]*x0
            x = model.transformer.h[i](x_in, ve.get(str(i)), cos_sin, model.window_sizes[i], kv)
            h_step[i] = x
        h_init = x

        # Extract cached K,V for parallel layers
        # KV cache layout: (n_layers, B, T_max, n_kv_head, head_dim)
        cached_k = kv.k_cache[par, :, :pos, :, :]  # (n_par, B, S, Hkv, hd)
        cached_v = kv.v_cache[par, :, :pos, :, :]

        # Warm-start init
        hs = [h_init] + [prev_h[par[j]].clone() for j in range(n_par)]

        for _k in range(K):
            all_hp = torch.stack([hs[j] for j in range(n_par)])

            if method == 'diag_vjp':
                # Single forward with grad, then VJP for diagonal estimation
                with torch.enable_grad():
                    hg = all_hp.detach().requires_grad_(True)
                    all_out_g, _, _ = _batched_block_forward_kv(
                        hg, x0, cos_sin, ve, par, model,
                        cached_k=cached_k, cached_v=cached_v)
                    diag_acc = torch.zeros_like(all_hp)
                    for _p in range(n_probes):
                        z = torch.randint(0, 2, all_hp.shape, device=device, dtype=dtype) * 2 - 1
                        jtz = torch.autograd.grad(all_out_g, hg, grad_outputs=z,
                                                   retain_graph=(_p < n_probes - 1))[0]
                        diag_acc += z * jtz.detach()
                all_out = all_out_g.detach()
                diag_j = diag_acc / n_probes
            else:
                # IDN: diag_j = 1
                with torch.no_grad():
                    all_out, new_k, new_v = _batched_block_forward_kv(
                        all_hp, x0, cos_sin, ve, par, model,
                        cached_k=cached_k, cached_v=cached_v)
                diag_j = torch.ones_like(all_out)

            # Diagonal scan correction
            bias = all_out - diag_j * all_hp
            a = torch.cat([torch.ones_like(diag_j[:1]), diag_j])
            b = torch.cat([torch.zeros_like(bias[:1]), bias])
            a_s, b_s = associative_scan_diagonal(a, b)
            with torch.no_grad():
                for l in range(n_par):
                    hs[l+1] = a_s[l+1]*h_init + b_s[l+1]

        for j, li in enumerate(par):
            h_step[li] = hs[j+1]

        # Update KV cache: recompute K,V from corrected inputs for ALL parallel layers
        # First layer's input = h_init (prefix output), rest = corrected h_step[li-1]
        for j, li in enumerate(par):
            corr_in = h_init if j == 0 else h_step[li - 1]
            x_in = model.resid_lambdas[li].to(dtype) * corr_in + model.x0_lambdas[li].to(dtype) * x0
            x_n = norm(x_in)
            blk = model.transformer.h[li]
            k_new = blk.attn.c_k(x_n).view(1, 1, n_kv_head, head_dim)
            v_new = blk.attn.c_v(x_n).view(1, 1, n_kv_head, head_dim)
            ve_i = ve.get(str(li))
            if ve_i is not None and blk.attn.ve_gate is not None:
                ve_r = ve_i.view(1, 1, n_kv_head, head_dim)
                gate = 3 * torch.sigmoid(blk.attn.ve_gate(x_n[..., :blk.attn.ve_gate_channels]))
                v_new = v_new + gate.unsqueeze(-1) * ve_r
            cos_k, sin_k = cos_sin
            k_new = apply_rotary_emb(k_new, cos_k, sin_k)
            k_new = norm(k_new) * 1.2
            kv.k_cache[li, :, pos:pos+1, :, :] = k_new
            kv.v_cache[li, :, pos:pos+1, :, :] = v_new

        # Advance cache
        kv.cache_seqlens += 1
        kv.prev_embedding = norm(model.transformer.wte(ids_step).to(dtype))

        # Save warm-start
        for i in range(L):
            if h_step[i] is not None:
                prev_h[i] = h_step[i].clone()

        # Logits
        x_final = h_step[L-1]
        bo = L//2
        if bo < L and h_step[bo] is not None:
            x_final = x_final - model.backout_lambda.to(dtype)*h_step[bo]
        x_final = norm(x_final)
        logits = model.lm_head(x_final)[...,:model.config.vocab_size].float()
        logits = 15*torch.tanh(logits/15)

    torch.cuda.synchronize(); dec_t = time.perf_counter()-t0
    return gen, pf_t, dec_t


# === Main ===
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--prompt", type=str, default="The quick brown fox")
parser.add_argument("--max-new-tokens", type=int, default=30)
parser.add_argument("--n-par", type=int, default=8)
parser.add_argument("--method", type=str, default="idn", choices=["idn", "diag_vjp"])
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--n-probes", type=int, default=1)
args = parser.parse_args()

from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer()
model = load_model(args.model_tag, device=device)
tokens = tokenizer.encode(args.prompt, prepend=tokenizer.get_bos_token_id())

seq_gen, _, seq_dec = generate_sequential(model, tokens, args.max_new_tokens)
seq_tps = args.max_new_tokens / seq_dec
print(f"Sequential: '{tokenizer.decode(seq_gen)}'")
print(f"  Decode: {seq_dec:.3f}s ({seq_tps:.1f} tok/s)")

par_gen, _, par_dec = generate_parallel(model, tokens, args.max_new_tokens,
                                         n_par=args.n_par, K=args.K,
                                         method=args.method, n_probes=args.n_probes)
par_tps = args.max_new_tokens / par_dec
fd = next((i for i,(a,b) in enumerate(zip(seq_gen,par_gen)) if a!=b), len(seq_gen))
print(f"\n{args.method} K={args.K} n_par={args.n_par}: '{tokenizer.decode(par_gen)}'")
print(f"  Decode: {par_dec:.3f}s ({par_tps:.1f} tok/s, {par_tps/seq_tps:.2f}x)")
print(f"  Match: first {fd}/{args.max_new_tokens}")
