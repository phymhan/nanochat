"""
Comprehensive layer-parallel inference evaluation.

Tests IDN batched, Fused+corr (multi-K), and Diag scan JVP with three
initialization strategies: h0, batched_fwd, warm_start.

Warm-start simulates AR decoding: runs sequential forward first to get
hidden states, then uses them as initialization for parallel inference.

Usage:
    # Single model
    python -m scripts.eval_jacobi_full --model-tag d32s_baseline_4800 --n-par 8,16

    # All models, save report
    python -m scripts.eval_jacobi_full \
        --model-tag d32s_baseline_4800 \
        --model-tag d32s_idn05_npar24_4800 \
        --model-tag d32s_diag01_npar24_stride3_4800 \
        --model-tag d32s_diag05_npar24_stride3_4800 \
        --n-par 8,16,24 --report eval_jacobi_d32p24_4800.md
"""

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type
from nanochat.jacobi_forward import _embed, _run_block, _logits


# ---------------------------------------------------------------------------
# Imports from existing eval scripts
# ---------------------------------------------------------------------------

from scripts.eval_jacobi_idn import (
    load_model, forward_sequential, forward_idn_batched,
    forward_fused_corrected, build_fused_weights, bench,
)
from scripts.eval_jacobi_diag import (
    _batched_block_forward, associative_scan_diagonal, forward_diag_scan,
)


# ---------------------------------------------------------------------------
# Data loading (deterministic, shared across all models)
# ---------------------------------------------------------------------------

def load_shared_data(device, seq_len=128, batch_size=8, n_batches=128):
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    data_dir = os.path.join(base_dir, 'base_data_climbmix')
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


# ---------------------------------------------------------------------------
# Unified forward with init strategy
# ---------------------------------------------------------------------------

def forward_with_init(model, idx, seq_layers, method='idn', K=1, elk_k=0.0,
                      init='h0', n_probes=4, warm_h=None):
    """
    Unified forward for layer-parallel inference with configurable init.

    Args:
        method: 'idn' (J=I), 'fused_corr' (batched IDN with K>1), 'diag_jvp'
        init: 'h0' (all h_init), 'fwd' (batched_fwd), 'warm' (from warm_h)
        warm_h: list of hidden states from previous sequential run (for warm init)
    """
    L = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    par = list(range(seq_layers, L))
    n_par = len(par)

    x0, cos_sin, ve = _embed(model, idx)

    # Sequential prefix
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
        h[i] = x
    h_init = x

    # --- Initialization ---
    if init == 'warm' and warm_h is not None:
        # Warm start: use previous sequential hidden states
        hs = [h_init] + [warm_h[par[j]].clone() for j in range(n_par)]
    elif init == 'fwd':
        # Batched forward init: evaluate all layers on h_init
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        with torch.no_grad():
            init_out = _batched_block_forward(all_h0, x0, cos_sin, ve, par, model)
        hs = [h_init] + [init_out[j] for j in range(n_par)]
    else:
        # h0 init: all layers start at prefix output
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    # --- Newton iterations ---
    if method == 'diag_jvp':
        from torch.autograd.functional import jvp as autograd_jvp
        for _k in range(K):
            all_h_prev = torch.stack([hs[j] for j in range(n_par)])
            diag_acc = torch.zeros_like(all_h_prev)
            dtype = all_h_prev.dtype
            device = all_h_prev.device

            def fwd_math(h_in):
                return _batched_block_forward(h_in, x0, cos_sin, ve, par, model,
                                              use_math_sdpa=True)

            with torch.no_grad():
                all_fx = _batched_block_forward(all_h_prev, x0, cos_sin, ve, par, model)

            for _p in range(n_probes):
                z = torch.randint(0, 2, all_h_prev.shape, device=device, dtype=dtype) * 2 - 1
                with torch.enable_grad():
                    _, jz = autograd_jvp(fwd_math, (all_h_prev.detach(),), (z,))
                diag_acc = diag_acc + z * jz.detach()
            diag_j = diag_acc / n_probes

            if elk_k > 0:
                diag_j = (1.0 - elk_k) * diag_j

            bias = all_fx - diag_j * all_h_prev
            a_full = torch.cat([torch.ones_like(diag_j[:1]), diag_j], dim=0)
            b_full = torch.cat([torch.zeros_like(bias[:1]), bias], dim=0)
            a_scan, b_scan = associative_scan_diagonal(a_full, b_full)
            with torch.no_grad():
                for l in range(n_par):
                    hs[l + 1] = a_scan[l + 1] * h_init + b_scan[l + 1]
    else:
        # IDN / Fused+corr path
        for _k in range(K):
            all_h_prev = torch.stack([hs[j] for j in range(n_par)])
            with torch.no_grad():
                all_fx = _batched_block_forward(all_h_prev, x0, cos_sin, ve, par, model)

            # IDN: diag_j = 1
            bias = all_fx - all_h_prev
            a_full = torch.cat([torch.ones_like(bias[:1]), torch.ones_like(bias)], dim=0)
            b_full = torch.cat([torch.zeros_like(bias[:1]), bias], dim=0)
            a_scan, b_scan = associative_scan_diagonal(a_full, b_full)
            with torch.no_grad():
                for l in range(n_par):
                    hs[l + 1] = a_scan[l + 1] * h_init + b_scan[l + 1]

    # Build final h list
    with torch.no_grad():
        for j, li in enumerate(par):
            h[li] = hs[j + 1]
        return _logits(model, h)


# ---------------------------------------------------------------------------
# Get sequential hidden states for warm-start
# ---------------------------------------------------------------------------

def get_sequential_hidden(model, idx):
    """Run sequential forward and return all hidden states."""
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)
    h = [None] * L
    x = x0
    for i in range(L):
        x = _run_block(model, i, x, x0, cos_sin, ve)
        h[i] = x
    return h


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, batches, n_par_list, device, n_warm=20, n_run=100):
    """Evaluate all methods and return results dict."""
    L = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    idx_bench = batches[0][:, :-1]

    # Sequential baseline
    seq_ms = bench(lambda: forward_sequential(model, idx_bench), device, n_warm, n_run)
    seq_losses = []
    for batch in batches:
        idx, targets = batch[:, :-1], batch[:, 1:]
        logits = forward_sequential(model, idx)
        seq_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           targets.reshape(-1)).item())
    seq_ppl = math.exp(sum(seq_losses) / len(seq_losses))

    results = {}
    for n_par in n_par_list:
        seq_layers = L - n_par
        rows = [('Sequential', '-', '-', seq_ms, seq_ppl, None)]

        # Methods to test
        configs = []
        for init in ['h0', 'fwd', 'warm']:
            for K in [1, 2, 4, 8]:
                configs.append((f'IDN K={K} {init}', 'idn', K, 0.0, init, 0))
            # Diag JVP: only K=1,2,4 (slow)
            for K in [1, 2]:
                configs.append((f'Diag JVP p=4 K={K} {init}', 'diag_jvp', K, 0.0, init, 4))

        for name, method, K, elk_k, init, n_probes in configs:
            # Timing (use first batch)
            def make_bench_fn(m=method, k=K, ek=elk_k, ini=init, np=n_probes):
                if ini == 'warm':
                    wh = get_sequential_hidden(model, idx_bench)
                    return lambda: forward_with_init(model, idx_bench, seq_layers,
                                                     method=m, K=k, elk_k=ek,
                                                     init=ini, n_probes=np, warm_h=wh)
                else:
                    return lambda: forward_with_init(model, idx_bench, seq_layers,
                                                     method=m, K=k, elk_k=ek,
                                                     init=ini, n_probes=np)

            ms = bench(make_bench_fn(), device, 5, 20)

            # PPL
            losses = []
            for batch in batches:
                idx, targets = batch[:, :-1], batch[:, 1:]
                warm_h = get_sequential_hidden(model, idx) if init == 'warm' else None
                logits = forward_with_init(model, idx, seq_layers,
                                           method=method, K=K, elk_k=elk_k,
                                           init=init, n_probes=n_probes, warm_h=warm_h)
                losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                               targets.reshape(-1)).item())
            ppl = math.exp(min(sum(losses) / len(losses), 20))
            top1 = None  # skip for speed
            rows.append((name, method, init, ms, ppl, top1))

        results[n_par] = rows
    return seq_ppl, seq_ms, results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results, total_tokens, report_path=None):
    lines = []
    def p(s=""): lines.append(s)

    p("# Layer-Parallel Inference Evaluation: d32s Models")
    p()
    p(f"depth=32, n_embd=640, 5 heads, ~535M params. Single H100, SDPA, seq_len=128, BS=8.")
    p(f"Quality: **{total_tokens:,} tokens**, same data for all models.")
    p()
    p("Init strategies:")
    p("- **h0**: all parallel layers initialized to prefix output (h_init)")
    p("- **fwd**: batched forward init — each layer starts at f_j(h_init)")
    p("- **warm**: warm start — sequential hidden states as init (simulates AR decoding)")
    p()

    for tag, (seq_ppl, seq_ms, results) in all_results.items():
        tag_short = tag.replace('d32s_', '').replace('_stride3', '').replace('_npar24', '')
        p(f"## {tag_short} (seq PPL={seq_ppl:.2f}, seq={seq_ms:.1f}ms)")
        p()

        for n_par, rows in sorted(results.items()):
            p(f"### n_par={n_par}")
            p()
            p("| Method | Init | Time | Speed | PPL | dPPL% |")
            p("|---|---|---|---|---|---|")

            for name, method, init, ms, ppl, top1 in rows:
                speed = seq_ms / ms
                if name == 'Sequential':
                    p(f"| Sequential | - | {ms:.1f}ms | 1.00x | {seq_ppl:.2f} | --- |")
                    continue
                dppl = 100 * (ppl - seq_ppl) / seq_ppl
                ppl_s = f"{ppl:.2f}" if ppl < 1e5 else "diverged"
                dppl_s = f"{dppl:+.1f}%" if ppl < 1e5 else "---"
                p(f"| {name} | {init} | {ms:.1f}ms | {speed:.2f}x | {ppl_s} | {dppl_s} |")
            p()

    if report_path:
        full_path = os.path.join(os.path.dirname(__file__), report_path)
        with open(full_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Report saved to {full_path}")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive layer-parallel inference eval")
    parser.add_argument("--model-tag", type=str, action='append', required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--n-par", type=str, default="8,16,24")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-batches", type=int, default=128)
    parser.add_argument("--report", type=str, default=None, help="output markdown filename")
    args = parser.parse_args()

    import nanochat.flash_attention as fa
    fa._override_impl = 'sdpa'
    fa.USE_FA3 = False

    device = torch.device(autodetect_device_type())
    n_par_list = [int(x) for x in args.n_par.split(',')]

    # Load data once
    batches = load_shared_data(device, args.seq_len, args.batch_size, args.n_batches)
    total_tokens = len(batches) * args.batch_size * args.seq_len
    print(f"Data: {total_tokens:,} tokens")

    all_results = {}
    for tag in args.model_tag:
        print(f"\n{'='*70}")
        print(f"  {tag}")
        print(f"{'='*70}")
        model = load_model(tag, step=args.step, device=device)
        seq_ppl, seq_ms, results = evaluate_model(model, batches, n_par_list, device)

        # Print summary
        for n_par, rows in sorted(results.items()):
            print(f"\n  n_par={n_par} (seq PPL={seq_ppl:.2f}, seq={seq_ms:.1f}ms):")
            for name, method, init, ms, ppl, _ in rows:
                if name == 'Sequential':
                    continue
                speed = seq_ms / ms
                dppl = 100 * (ppl - seq_ppl) / seq_ppl if ppl < 1e5 else float('inf')
                ppl_s = f"{ppl:.2f}" if ppl < 1e5 else "diverged"
                print(f"    {name:<30s} | {ms:6.1f}ms {speed:5.2f}x | PPL {ppl_s:>10s}")

        all_results[tag] = (seq_ppl, seq_ms, results)
        del model
        torch.cuda.empty_cache()
        time.sleep(2)

    # Generate report
    if args.report:
        generate_report(all_results, total_tokens, args.report)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
