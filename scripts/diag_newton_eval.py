"""
Measure Jacobian approximation quality for layer-parallel inference.

Pure measurement — no inference strategies. Measures per-layer:
  IDN error:  ||J*v - v||   / ||J*v||   (how close J is to identity)
  Diag error: ||J*v - diag(J)*v|| / ||J*v||   (how close J is to diag(J))

Also reports sequential PPL as a model quality reference.

Usage:
    python -m scripts.diag_newton_eval --model-tag d32_baseline
    python -m scripts.diag_newton_eval --model-tag d32_baseline --model-tag d32_idn05_npar8
    python -m scripts.diag_newton_eval --random --depth 32
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, norm


def load_model(model_tag=None, checkpoint_dir=None, step=None, depth=32, random=False, device=None):
    """Load a trained model or create one with random weights."""
    if random:
        print(f"  Creating random model with depth={depth}")
        aspect_ratio, head_dim = 64, 128
        n_head = max(1, round(depth * aspect_ratio / head_dim))
        config = GPTConfig(n_layer=depth, n_embd=n_head * head_dim, n_head=n_head, n_kv_head=n_head)
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        s = 1.0 / (config.n_embd ** 0.5)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(123)
        for block in model.transformer.h:
            block.attn.c_proj.weight.data = torch.randn(config.n_embd, config.n_embd, generator=rng) * s
            block.mlp.c_proj.weight.data = torch.randn(config.n_embd, 4 * config.n_embd, generator=rng) * s
        model.eval()
        return model

    from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, find_last_step
    if checkpoint_dir is None:
        base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
        checkpoint_dir = os.path.join(base_dir, 'base_checkpoints', model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    print(f"  Loading {checkpoint_dir} step {step}")
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    config_kw = meta_data["model_config"]
    _patch_missing_config_keys(config_kw)
    config = GPTConfig(**config_kw)
    _patch_missing_keys(model_data, config)
    model_data = {k: v.to(device) for k, v in model_data.items()}
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    return model


def create_test_batches(model, device, seq_len=128, batch_size=1, n_batches=8):
    """Load real text from ClimbMix validation shard, fall back to random."""
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    data_dir = os.path.join(base_dir, 'base_data_climbmix')
    if os.path.isdir(data_dir):
        try:
            import pyarrow.parquet as pq
            from nanochat.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()
            bos = tokenizer.get_bos_token_id()
            parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
            table = pq.read_table(os.path.join(data_dir, parquet_files[-1]), columns=['text'])
            texts = table.column('text').to_pylist()
            all_tokens = []
            for text in texts:
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
            print(f"  Loaded {len(batches)} batches from ClimbMix val split")
            return batches
        except Exception as e:
            print(f"  Dataset load failed ({e}), using random tokens")
    vocab_size = model.config.vocab_size
    return [torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device) for _ in range(n_batches)]


@torch.no_grad()
def sequential_forward_with_intermediates(model, idx):
    """Run full sequential forward, return per-layer hidden states and x0."""
    B, T = idx.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0 = x
    all_h = []
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.window_sizes[i], None)
        all_h.append(x)
    return all_h, x0, cos_sin


def _layer_fn(model, li, h_input, x0, idx, cos_sin):
    """Run a single layer (supports grad when called without no_grad)."""
    x_in = model.resid_lambdas[li] * h_input + model.x0_lambdas[li] * x0
    ve = model.value_embeds[str(li)](idx).to(x_in.dtype) if str(li) in model.value_embeds else None
    return model.transformer.h[li](x_in, ve, cos_sin, model.window_sizes[li], None)


def _compute_jtv(model, li, h_init, x0, idx, cos_sin, v):
    """Compute J^T * v via backward AD (VJP). Works with FA3."""
    with torch.enable_grad():
        h_g = h_init.detach().requires_grad_(True)
        out = _layer_fn(model, li, h_g, x0.detach(), idx, cos_sin)
        jtv = torch.autograd.grad(out, h_g, grad_outputs=v, retain_graph=False)[0]
    return jtv.float()


@torch.no_grad()
def measure_jacobian_errors(model, batches, n_par_list, n_probes=4):
    """
    Measure per-layer Jacobian approximation errors.

    For each parallel layer, estimates via VJP Hutchinson:
      IDN error:  mean ||J^T*v - v||   / ||J^T*v||   (J vs I)
      Diag error: mean ||J^T*v - diag(J)*v|| / ||J^T*v||   (J vs diag(J))
    """
    n_layer = model.config.n_layer
    device = next(model.parameters()).device

    results = {}
    for n_par in n_par_list:
        if n_par >= n_layer:
            continue
        seq_layers = n_layer - n_par

        idn_errors = [[] for _ in range(n_par)]
        diag_errors = [[] for _ in range(n_par)]

        for batch in batches:
            idx = batch[:, :-1]
            all_h, x0, cos_sin = sequential_forward_with_intermediates(model, idx)
            h_init = all_h[seq_layers - 1] if seq_layers > 0 else x0

            for j in range(n_par):
                li = seq_layers + j

                # Estimate diag(J) via Hutchinson (average over n_probes)
                diag_accum = torch.zeros_like(h_init, dtype=torch.float32)
                for _ in range(n_probes):
                    z = torch.randint(0, 2, h_init.shape, device=device, dtype=h_init.dtype) * 2 - 1
                    jtv_z = _compute_jtv(model, li, h_init, x0, idx, cos_sin, z)
                    diag_accum += z.float() * jtv_z
                diag_j = diag_accum / n_probes

                # Measure errors with a fresh random probe
                v = torch.randn_like(h_init)
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                jtv = _compute_jtv(model, li, h_init, x0, idx, cos_sin, v)
                jtv_norm = jtv.norm(dim=-1).clamp(min=1e-8)

                # IDN error: ||J^T*v - v|| / ||J^T*v||
                idn_err = (jtv - v.float()).norm(dim=-1) / jtv_norm
                idn_errors[j].append(idn_err.mean().item())

                # Diag error: ||J^T*v - diag(J)*v|| / ||J^T*v||
                diag_v = diag_j * v.float()
                diag_err = (jtv - diag_v).norm(dim=-1) / jtv_norm
                diag_errors[j].append(diag_err.mean().item())

        # Average across batches
        avg_idn = [sum(e) / len(e) for e in idn_errors]
        avg_diag = [sum(e) / len(e) for e in diag_errors]
        results[n_par] = {'idn_errors': avg_idn, 'diag_errors': avg_diag}

    return results


def print_results(model_tag, ref_ppl, results):
    """Print results table."""
    print(f"\n{'='*90}")
    print(f"  Model: {model_tag}  |  Sequential PPL: {ref_ppl:.2f}")
    print(f"{'='*90}")

    for n_par in sorted(results.keys()):
        r = results[n_par]
        n_layer = 32  # will be correct for d32 models
        idn = r['idn_errors']
        diag = r['diag_errors']

        print(f"\n  n_par={n_par} (seq={n_layer - n_par} + {n_par} parallel)")
        print(f"  {'':>4s} | {'IDN err':>8s} | {'Diag err':>8s} | {'Diag/IDN':>8s}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for j in range(n_par):
            ratio = diag[j] / idn[j] if idn[j] > 1e-8 else float('inf')
            print(f"  L{j:<2d} | {idn[j]:8.4f} | {diag[j]:8.4f} | {ratio:8.4f}")

        mean_idn = sum(idn) / len(idn)
        mean_diag = sum(diag) / len(diag)
        mean_ratio = mean_diag / mean_idn if mean_idn > 1e-8 else float('inf')
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        print(f"  avg | {mean_idn:8.4f} | {mean_diag:8.4f} | {mean_ratio:8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Measure Jacobian approximation quality")
    parser.add_argument("--model-tag", type=str, action='append', default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--n-par", type=str, default="4,8",
                        help="comma-separated n_par values to test")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--n-probes", type=int, default=4,
                        help="Hutchinson probes for diagonal estimation")
    args = parser.parse_args()

    device = torch.device(autodetect_device_type())
    n_par_list = [int(x) for x in args.n_par.split(',')]

    if args.random:
        tags = ['random']
    elif args.model_tag:
        tags = args.model_tag
    elif args.checkpoint_dir:
        tags = [args.checkpoint_dir]
    else:
        parser.error("Specify --model-tag, --checkpoint-dir, or --random")

    for tag in tags:
        print(f"\nLoading model: {tag}")
        if tag == 'random':
            model = load_model(random=True, depth=args.depth, device=device)
        elif os.path.isdir(tag):
            model = load_model(checkpoint_dir=tag, step=args.step, device=device)
        else:
            model = load_model(model_tag=tag, step=args.step, device=device)

        batches = create_test_batches(model, device, args.seq_len, args.batch_size, args.n_batches)

        # Sequential PPL
        ref_ppls = []
        for batch in batches:
            idx, targets = batch[:, :-1], batch[:, 1:]
            with torch.no_grad():
                logits = model.forward(idx)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), reduction='mean').item()
            ref_ppls.append(math.exp(min(loss, 20.0)))
        ref_ppl = sum(ref_ppls) / len(ref_ppls)

        # Jacobian errors
        results = measure_jacobian_errors(model, batches, n_par_list, n_probes=args.n_probes)
        print_results(tag, ref_ppl, results)

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
