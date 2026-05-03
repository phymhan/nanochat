"""
Newton correction and ELK damping evaluation for Huginn depth-recurrent model.

Three hypotheses:
  H1: Within-iteration Newton correction (across 4 core blocks)
  H2: DEQ solvers (Broyden / Anderson) replacing Picard iteration
  H3: ELK-style damping (damped Picard, quasi-ELK Kalman smoothing)

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run python -m scripts.eval_huginn_newton --hypothesis h3a --max-tokens 50000
    CUDA_VISIBLE_DEVICES=5 uv run python -m scripts.eval_huginn_newton --hypothesis h1 --max-tokens 50000
    CUDA_VISIBLE_DEVICES=6 uv run python -m scripts.eval_huginn_newton --hypothesis h2 --max-tokens 50000
"""

import argparse
import json
import math
import os
import time
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name="tomg-group-umd/huginn-0125", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading & PPL helpers
# ---------------------------------------------------------------------------

def load_val_data(tokenizer, max_tokens=50_000, seq_len=128):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    batches = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        batches.append(tokens[i : i + seq_len + 1].unsqueeze(0))
    return batches


def compute_ppl(batches, forward_fn, device):
    losses = []
    for batch in batches:
        batch = batch.to(device)
        idx, tgt = batch[:, :-1], batch[:, 1:]
        logits = forward_fn(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


def bench(fn, device, n_warm=10, n_run=30):
    for _ in range(n_warm):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000


# ---------------------------------------------------------------------------
# Baseline: standard sequential forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_sequential(model, input_ids, num_steps=32):
    out = model(input_ids, num_steps=num_steps)
    return out.logits


# ---------------------------------------------------------------------------
# H1: Within-iteration Newton correction (across 4 core blocks)
# ---------------------------------------------------------------------------

def core_block_forward_newton_idn(model, x, input_embeds, freqs_cis, mask,
                                   past_key_values, block_idx, step, elk_k=0.0):
    block_idx = block_idx.detach().clone()
    x = model._maybe_inject_noise(x, step)
    h0 = model.transformer.adapter(torch.cat([x, input_embeds.to(x.device)], dim=-1))
    h = h0
    a = 1.0 - elk_k
    for block in model.transformer.core_block:
        block_idx += 1
        h_new = block(h, freqs_cis, block_idx, mask, past_key_values)
        h_new = h_new + a * (h - h0)
        h = h_new
    return h, block_idx


def core_block_forward_newton_jvp(model, x, input_embeds, freqs_cis, mask,
                                   past_key_values, block_idx, step, elk_k=0.0):
    block_idx = block_idx.detach().clone()
    x = model._maybe_inject_noise(x, step)
    h0 = model.transformer.adapter(torch.cat([x, input_embeds.to(x.device)], dim=-1))
    h = h0
    a = 1.0 - elk_k
    for block in model.transformer.core_block:
        block_idx += 1
        tangent = h - h0
        with torch.enable_grad():
            h_g = h.detach().requires_grad_(True)
            h_out = block(h_g, freqs_cis, block_idx, mask, past_key_values)
            jvp_result = torch.autograd.grad(
                h_out, h_g, grad_outputs=tangent.detach(), retain_graph=False)[0]
        h_new = h_out.detach() + a * jvp_result.detach()
        h = h_new
    return h, block_idx


@torch.no_grad()
def forward_h1(model, input_ids, num_steps=32, method="idn", elk_k=0.0):
    input_embeds, block_idx = model.embed_inputs(input_ids)
    freqs_cis = model.freqs_cis[:, :input_ids.shape[1]]
    x = model.initialize_state(input_embeds)

    core_fn = core_block_forward_newton_idn if method == "idn" else core_block_forward_newton_jvp

    for k in range(num_steps):
        x, block_idx = core_fn(
            model, x, input_embeds, freqs_cis, mask=None,
            past_key_values=None, block_idx=block_idx, step=k, elk_k=elk_k)

    x = model.transformer.ln_f(x)
    out = model.predict_from_latents(x)
    return out.logits


# ---------------------------------------------------------------------------
# H2: DEQ solvers (Broyden / Anderson)
# ---------------------------------------------------------------------------

def _safe_norm(v):
    if not torch.isfinite(v).all():
        return float("inf")
    return torch.norm(v).item()


def matvec(part_Us, part_VTs, x):
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum("bdij, bij -> bd", part_VTs, x)
    return -x + torch.einsum("bijd, bd -> bij", part_Us, VTx)


def rmatvec(part_Us, part_VTs, x):
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum("bij, bijd -> bd", x, part_Us)
    return -x + torch.einsum("bd, bdij -> bij", xTU, part_VTs)


def broyden(f, x0, threshold, eps=1e-3, ls=False):
    bsz, total_hsize, seq_len = x0.size()
    g = lambda y: f(y) - y
    dev = x0.device

    x_est = x0
    gx = g(x_est)
    nstep = 0

    Us = torch.zeros(bsz, total_hsize, seq_len, threshold, device=dev, dtype=x0.dtype)
    VTs = torch.zeros(bsz, threshold, total_hsize, seq_len, device=dev, dtype=x0.dtype)
    update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)

    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_xest, lowest_gx = x_est, gx
    trace = []

    while nstep < threshold:
        x_est = x_est + update
        gx_new = g(x_est)
        delta_x = update
        delta_gx = gx_new - gx
        gx = gx_new
        nstep += 1

        abs_diff = _safe_norm(gx)
        rel_diff = abs_diff / (_safe_norm(gx + x_est) + 1e-9)
        trace.append({"abs": abs_diff, "rel": rel_diff})

        if abs_diff < lowest_dict["abs"]:
            lowest_dict["abs"] = abs_diff
            lowest_dict["rel"] = rel_diff
            lowest_xest = x_est.clone().detach()

        if rel_diff < eps:
            break
        if abs_diff > 1e6 * x0.shape[-1]:
            break

        part_Us, part_VTs = Us[:, :, :, :nstep - 1], VTs[:, :nstep - 1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / (
            torch.einsum("bij, bij -> b", vT, delta_gx)[:, None, None] + 1e-10)
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, nstep - 1] = vT
        Us[:, :, :, nstep - 1] = u
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)

    return {"result": lowest_xest, "nstep": nstep, "trace": trace,
            "lowest_abs": lowest_dict["abs"], "lowest_rel": lowest_dict["rel"]}


def anderson(f, x0, m=6, lam=1e-4, threshold=50, eps=1e-3, beta=1.0):
    bsz, d, L = x0.shape
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F_buf = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0] = x0.reshape(bsz, -1)
    F_buf[:, 0] = f(x0).reshape(bsz, -1)
    X[:, 1] = F_buf[:, 0]
    F_buf[:, 1] = f(F_buf[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_xest = x0
    trace = []

    for k in range(2, threshold):
        n = min(k, m)
        G = F_buf[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(
            n, dtype=x0.dtype, device=x0.device)[None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]

        X[:, k % m] = beta * (alpha[:, None] @ F_buf[:, :n])[:, 0] + (1 - beta) * (
            alpha[:, None] @ X[:, :n])[:, 0]
        F_buf[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F_buf[:, k % m] - X[:, k % m]).view_as(x0)

        abs_diff = _safe_norm(gx)
        rel_diff = abs_diff / (1e-5 + _safe_norm(F_buf[:, k % m]))
        trace.append({"abs": abs_diff, "rel": rel_diff})

        if abs_diff < lowest_dict["abs"]:
            lowest_dict["abs"] = abs_diff
            lowest_dict["rel"] = rel_diff
            lowest_xest = X[:, k % m].view_as(x0).clone().detach()

        if rel_diff < eps:
            break

    return {"result": lowest_xest, "nstep": k if trace else 0, "trace": trace,
            "lowest_abs": lowest_dict["abs"], "lowest_rel": lowest_dict["rel"]}


@torch.no_grad()
def forward_h2(model, input_ids, solver="broyden", threshold=32, eps=1e-5, num_steps=None):
    input_embeds, block_idx_base = model.embed_inputs(input_ids)
    x0 = model.initialize_state(input_embeds)
    freqs_cis = model.freqs_cis[:, :input_ids.shape[1]]

    def f_huginn(x_flat):
        x = x_flat.permute(0, 2, 1)
        block_idx = block_idx_base.detach().clone()
        x_out, _ = model.core_block_forward(
            x, input_embeds, freqs_cis, None, None, block_idx, 0)
        return x_out.permute(0, 2, 1)

    B, T, D = x0.shape
    x0_flat = x0.permute(0, 2, 1)

    if solver == "broyden":
        result = broyden(f_huginn, x0_flat, threshold=threshold, eps=eps)
    elif solver == "anderson":
        result = anderson(f_huginn, x0_flat, threshold=threshold, eps=eps)
    elif solver == "picard":
        x = x0
        trace = []
        for k in range(threshold):
            block_idx = block_idx_base.detach().clone()
            x_new, _ = model.core_block_forward(
                x, input_embeds, freqs_cis, None, None, block_idx, k)
            residual = (x_new - x).float().norm().item()
            trace.append({"abs": residual, "rel": residual / (x.float().norm().item() + 1e-9)})
            x = x_new
        result = {"result": x.permute(0, 2, 1), "nstep": threshold, "trace": trace}
    else:
        raise ValueError(f"Unknown solver: {solver}")

    x_solved = result["result"].permute(0, 2, 1)
    x_solved = model.transformer.ln_f(x_solved)
    out = model.predict_from_latents(x_solved)
    return out.logits


# ---------------------------------------------------------------------------
# H3a: Damped Picard iteration
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_h3a(model, input_ids, num_steps=32, elk_k=0.0):
    input_embeds, block_idx = model.embed_inputs(input_ids)
    freqs_cis = model.freqs_cis[:, :input_ids.shape[1]]
    x = model.initialize_state(input_embeds)

    for k in range(num_steps):
        block_idx_k = block_idx.detach().clone()
        x_new, block_idx = model.core_block_forward(
            x, input_embeds, freqs_cis, None, None, block_idx_k, k)
        x = elk_k * x + (1.0 - elk_k) * x_new  # damped Picard

    x = model.transformer.ln_f(x)
    out = model.predict_from_latents(x)
    return out.logits


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def build_methods(model, hypothesis, num_steps=32):
    methods = {}

    if hypothesis in ("all", "baseline"):
        for N in [8, 16, 24, 32]:
            methods[f"seq_N{N}"] = lambda ids, _N=N: forward_sequential(
                model, ids, num_steps=_N)

    if hypothesis in ("all", "h1"):
        for elk_k in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            methods[f"h1_idn_N{num_steps}_elk{elk_k}"] = lambda ids, _ek=elk_k: forward_h1(
                model, ids, num_steps=num_steps, method="idn", elk_k=_ek)
        for elk_k in [0.0, 0.1, 0.3, 0.5]:
            methods[f"h1_jvp_N{num_steps}_elk{elk_k}"] = lambda ids, _ek=elk_k: forward_h1(
                model, ids, num_steps=num_steps, method="jvp", elk_k=_ek)

    if hypothesis in ("all", "h2"):
        for T in [8, 16, 32, 64]:
            methods[f"h2_picard_T{T}"] = lambda ids, _T=T: forward_h2(
                model, ids, solver="picard", threshold=_T)
            methods[f"h2_broyden_T{T}"] = lambda ids, _T=T: forward_h2(
                model, ids, solver="broyden", threshold=_T)
            methods[f"h2_anderson_T{T}"] = lambda ids, _T=T: forward_h2(
                model, ids, solver="anderson", threshold=_T)

    if hypothesis in ("all", "h3a"):
        for elk_k in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            methods[f"h3a_N{num_steps}_elk{elk_k}"] = lambda ids, _ek=elk_k: forward_h3a(
                model, ids, num_steps=num_steps, elk_k=_ek)
        for N in [16, 24]:
            for elk_k in [0.3, 0.5]:
                methods[f"h3a_N{N}_elk{elk_k}"] = lambda ids, _N=N, _ek=elk_k: forward_h3a(
                    model, ids, num_steps=_N, elk_k=_ek)

    return methods


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hypothesis", default="h3a", choices=["h1", "h2", "h3a", "baseline", "all"])
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=50_000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, args.device)
    device = next(model.parameters()).device
    print(f"Loaded. Device: {device}")

    print(f"Loading validation data ({args.max_tokens:,} tokens)...")
    batches = load_val_data(tokenizer, max_tokens=args.max_tokens)
    print(f"  {len(batches)} batches")

    # Baseline PPL
    print("\nBaseline (seq N=32)...")
    baseline_ppl = compute_ppl(batches, lambda ids: forward_sequential(model, ids, 32), device)
    print(f"  PPL = {baseline_ppl:.2f}")

    if not args.skip_timing:
        idx_bench = batches[0].to(device)[:, :-1]
        baseline_ms = bench(lambda: forward_sequential(model, idx_bench, 32), device)
        print(f"  Time = {baseline_ms:.1f}ms")

    # Build and run methods
    methods = build_methods(model, args.hypothesis, args.num_steps)
    results = {"baseline_ppl": baseline_ppl, "hypothesis": args.hypothesis}

    print(f"\nEvaluating {len(methods)} configs for {args.hypothesis}...")
    for name, fn in methods.items():
        print(f"  {name}...", end="", flush=True)
        try:
            ppl = compute_ppl(batches, fn, device)
            dppl = 100 * (ppl - baseline_ppl) / baseline_ppl

            ms = None
            if not args.skip_timing:
                ms = bench(lambda: fn(idx_bench), device)

            results[name] = {"ppl": ppl, "dppl": dppl, "ms": ms}
            ms_str = f" {ms:.1f}ms" if ms else ""
            print(f" PPL={ppl:.2f} ({dppl:+.1f}%){ms_str}")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f" ERROR: {e}")

    # Save results
    outpath = args.output or f"cache/eval_logs/huginn_{args.hypothesis}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
