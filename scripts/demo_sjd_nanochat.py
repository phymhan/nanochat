"""
Speculative Jacobi Decoding (SJD) demo for nanochat.

Three decoding modes along the token (sequence) axis:
  1. AR:  standard autoregressive decoding (baseline)
  2. JD:  vanilla Jacobi Decoding — deterministic convergence (draft == argmax)
  3. SJD: Speculative Jacobi Decoding (arXiv:2512.07503) — probabilistic
         acceptance via speculative sampling, works with stochastic decoding

SJD combines Jacobi iteration (parallel multi-token prediction) with speculative
decoding (probabilistic draft-and-verify). Each iteration:
  1. Forward pass on [prefix | drafts] → per-position distributions p(x|ctx_j)
  2. Accept draft[i] with prob min(1, p(x|ctx_j) / p(x|ctx_{j-1}))  [Eq 2]
  3. First rejected token: resample from max(0, p_new - p_old)         [Eq 3]
  4. Remaining unaccepted: resample from p(x|ctx_j)  (refinement)      [Eq 5]

Usage:
    # AR generation (greedy)
    uv run python -m scripts.demo_sjd_nanochat --mode ar --max-new-tokens 64

    # Jacobi Decoding (deterministic, greedy)
    uv run python -m scripts.demo_sjd_nanochat --mode jd --n-lookahead 5

    # SJD (requires stochastic sampling)
    uv run python -m scripts.demo_sjd_nanochat --mode sjd --n-lookahead 16 \
        --temperature 1.0 --top-k 50

    # All modes + PPL
    uv run python -m scripts.demo_sjd_nanochat --mode ar jd sjd ppl \
        --temperature 1.0 --top-k 50
"""

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_tag, step=None, device=None):
    from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, find_last_step
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    path = os.path.join(base_dir, 'base_checkpoints', model_tag)
    if step is None:
        step = find_last_step(path)
    print(f"Loading {path} step {step}")
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


def get_tokenizer():
    from nanochat.tokenizer import get_tokenizer as _get_tokenizer
    return _get_tokenizer()


# ---------------------------------------------------------------------------
# AR generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_ar(model, input_ids, max_new_tokens, temperature=0.0, top_k=None):
    """
    Standard autoregressive generation (no KV cache, recomputes full prefix).
    Returns (output_ids, stats_dict).
    """
    device = input_ids.device
    ids = input_ids.clone()  # (1, T)
    n_prompt = ids.shape[1]

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        logits = model.forward(ids)  # (1, T, V)
        logits = logits[:, -1, :]    # (1, V)
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
        'method': 'AR',
        'n_prompt': n_prompt,
        'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': n_gen,
        'acceptance_rate': 1.0,
    }


# ---------------------------------------------------------------------------
# JD (vanilla Jacobi Decoding) — deterministic convergence
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_jd(model, input_ids, max_new_tokens, n_lookahead=5,
                temperature=0.0, top_k=None, max_jacobi_iters=32):
    """
    Vanilla Jacobi Decoding along the token axis (deterministic).

    Always uses argmax for convergence check and token selection.
    Inner loop iterates until fixed point or max_jacobi_iters.
    """
    device = input_ids.device
    vocab_size = model.config.vocab_size
    ids = input_ids.clone()
    n_prompt = ids.shape[1]
    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
    total_accepted = 0
    total_forward_passes = 0
    total_steps = 0

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    while total_accepted < max_new_tokens:
        remaining = max_new_tokens - total_accepted
        n_draft = min(n_lookahead, remaining - 1) if remaining > 1 else 0
        current_drafts = drafts[:, :n_draft]

        if n_draft == 0:
            logits = model.forward(ids)
            total_forward_passes += 1
            ids = torch.cat([ids, logits[:, -1:, :].argmax(dim=-1)], dim=1)
            total_accepted += 1
            total_steps += 1
            continue

        accepted_this_step = False
        for ji in range(max_jacobi_iters):
            full_seq = torch.cat([ids, current_drafts], dim=1)
            logits = model.forward(full_seq)
            total_forward_passes += 1
            # JD always uses argmax (deterministic)
            predicted = logits.argmax(dim=-1)  # (1, T+n_draft)

            T = ids.shape[1]
            model_preds = predicted[:, T-1:T-1+n_draft]

            match = (model_preds == current_drafts).squeeze(0)
            n_match = 0
            for j in range(n_draft):
                if match[j]:
                    n_match += 1
                else:
                    break

            if n_match > 0:
                accepted = current_drafts[:, :n_match]
                bonus = predicted[:, T-1+n_match:T+n_match]
                ids = torch.cat([ids, accepted, bonus], dim=1)
                total_accepted += n_match + 1
                total_steps += 1
                accepted_this_step = True
                n_consumed = n_match + 1
                if n_consumed < n_draft:
                    updated = model_preds[:, n_consumed:]
                    n_new = n_lookahead - updated.shape[1]
                    drafts = torch.cat([updated, torch.randint(0, vocab_size, (1, n_new), device=device)], dim=1)
                else:
                    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
                break

            # Jacobi update: drafts ← argmax predictions
            current_drafts = model_preds.clone()

        if not accepted_this_step:
            ids = torch.cat([ids, predicted[:, T-1:T]], dim=1)
            total_accepted += 1
            total_steps += 1
            updated = model_preds[:, 1:]
            n_new = n_lookahead - updated.shape[1]
            drafts = torch.cat([updated, torch.randint(0, vocab_size, (1, n_new), device=device)], dim=1)

    ids = ids[:, :n_prompt + max_new_tokens]
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    n_gen = ids.shape[1] - n_prompt
    return ids, {
        'method': f'JD(lookahead={n_lookahead})',
        'n_prompt': n_prompt, 'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': total_forward_passes,
        'n_steps': total_steps,
        'acceptance_rate': n_gen / total_forward_passes if total_forward_passes > 0 else 0,
    }


# ---------------------------------------------------------------------------
# SJD / SJD++ (Speculative Jacobi Decoding) — probabilistic acceptance
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_sjd(model, input_ids, max_new_tokens, n_lookahead=16,
                 temperature=1.0, top_k=50, reuse_threshold=-1.0):
    """
    Speculative Jacobi Decoding (arXiv:2512.07503).

    reuse_threshold controls SJD vs SJD++:
      < 0: SJD  — unaccepted tokens after rejection are resampled from p_new (Eq 5)
      >= 0: SJD++ — unaccepted tokens with confidence ratio > threshold are kept (Eq 6)
                    set to 0 to reuse everything, higher = more selective reuse

    Per iteration (one forward pass, no inner loop):

    Step 1 — Draft preparation:
        Build input [prefix | drafts]. Drafts are warm-started from previous
        iteration's model outputs (unaccepted tokens), with their probability
        distributions stored as draft_dist. Freshly introduced draft slots use
        one-hot proposal distributions on their initialized token, matching the
        reference code's bootstrap behavior.

    Step 2 — Parallel decoding:
        Forward pass → logits → sample tokens and compute probs for all positions.
        next_tokens[j] ~ p(x | x_{1:T+j-1}^{(j)})  via temp+top_k sampling
        new_probs[j]   = p(x | x_{1:T+j-1}^{(j)})   full distribution

    Step 3 — Speculative verification (Eq 2):
        For each draft position j (sequentially):
          accept draft[j] if r < min(1, p_new(draft[j]) / p_old(draft[j]))
          where p_new = new_probs[j, draft[j]]   = p(draft[j] | ctx from THIS iter)
                p_old = draft_dist[j, draft[j]]   = p(draft[j] | ctx from PREV iter)
        Stop at first rejection.

    Step 4 — Token resampling and reusing:
        First rejected position: resample from max(0, p_new - p_old) (Eq 3)
        Remaining unaccepted positions:
          SJD:  resample from p_new (Eq 5) — next_tokens[j] already sampled
          SJD++: if confidence C_j = p_new(draft[j])/p_old(draft[j]) > τ,
                 reuse draft[j] unchanged (Eq 6), else resample from p_new
        These become drafts for next iteration (warm-start via refinement).
    """
    assert temperature > 0, "SJD requires stochastic sampling (temperature > 0)"
    use_reuse = reuse_threshold >= 0  # SJD++ mode
    device = input_ids.device
    vocab_size = model.config.vocab_size
    ids = input_ids.clone()
    n_prompt = ids.shape[1]

    def _one_hot_draft_dist(tokens):
        """Reference bootstrap: fresh draft slots get one-hot proposal scores."""
        dist = torch.zeros(tokens.shape[0], tokens.shape[1], vocab_size, device=device)
        return dist.scatter_(2, tokens.unsqueeze(-1), 1.0)

    # Initialize drafts and their bootstrap proposal distributions.
    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
    draft_dist = _one_hot_draft_dist(drafts)

    total_accepted = 0
    total_forward_passes = 0
    total_steps = 0
    total_reused = 0  # SJD++: count tokens reused via confidence threshold
    bootstrap_complete = False

    def _get_probs(logits):
        """logits (..., V) → probs with temperature + top-k."""
        logits_s = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits_s, min(top_k, logits_s.size(-1)), dim=-1)
            logits_s[logits_s < v[..., [-1]]] = -float('inf')
        return F.softmax(logits_s, dim=-1)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    while total_accepted < max_new_tokens:
        # Reference scheduler starts with a single-token decoding step before
        # opening the Jacobi window, so every reused draft thereafter comes
        # from an actual model forward.
        if not bootstrap_complete:
            logits = model.forward(ids)
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
            logits = model.forward(ids)
            total_forward_passes += 1
            probs = _get_probs(logits[:, -1, :])
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
            total_accepted += 1
            total_steps += 1
            continue

        current_drafts = drafts[:, :n_draft]          # (1, n_draft)
        current_dist = draft_dist[:, :n_draft, :]     # (1, n_draft, V)

        # ── Step 2: Parallel decoding (one forward pass) ──
        full_seq = torch.cat([ids, current_drafts], dim=1)  # (1, T + n_draft)
        logits = model.forward(full_seq)                     # (1, T + n_draft, V)
        total_forward_passes += 1

        T = ids.shape[1]
        # new_probs[j] = p(x | x_{1:T-1+j}^{(j)}) at position T-1+j in the output
        # This predicts what should be at position T+j in the sequence.
        new_probs = _get_probs(logits[:, T-1:T-1+n_draft, :])  # (1, n_draft, V)
        # Sample from each position's distribution (for refinement / Eq 5)
        next_tokens = torch.multinomial(
            new_probs.view(-1, vocab_size), 1
        ).view(1, n_draft)  # (1, n_draft) — model's sampled predictions

        # ── Step 3: Speculative verification (Eq 2) ──
        n_accept = 0
        for j in range(n_draft):
            tok = current_drafts[0, j].item()
            p_new = new_probs[0, j, tok].item()
            p_old = current_dist[0, j, tok].item()

            if p_old > 0:
                accept_prob = min(1.0, p_new / p_old)
            else:
                accept_prob = 1.0 if p_new > 0 else 0.0

            r = torch.rand(1, device=device).item()
            if r < accept_prob:
                n_accept += 1
            else:
                break

        # Append accepted draft tokens to prefix
        if n_accept > 0:
            ids = torch.cat([ids, current_drafts[:, :n_accept]], dim=1)

        # ── Step 4: Token resampling / reusing ──
        reject_pos = n_accept

        # 4a. First rejected position: resample from calibrated residual (Eq 3)
        if reject_pos < n_draft:
            p_new_dist = new_probs[0, reject_pos, :]    # (V,)
            p_old_dist = current_dist[0, reject_pos, :] # (V,)
            residual = (p_new_dist - p_old_dist).clamp(min=0)
            residual_sum = residual.sum()
            if residual_sum > 1e-8:
                bonus = torch.multinomial(residual.unsqueeze(0) / residual_sum, 1)
            else:
                bonus = torch.multinomial(p_new_dist.unsqueeze(0), 1)
            ids = torch.cat([ids, bonus], dim=1)
        else:
            # All accepted — bonus from position beyond the window
            bonus_probs = _get_probs(logits[:, T-1+n_draft:T+n_draft, :])
            bonus = torch.multinomial(bonus_probs.squeeze(1), 1)
            ids = torch.cat([ids, bonus], dim=1)

        total_accepted += n_accept + 1
        total_steps += 1

        # 4b. Remaining unaccepted positions → build drafts for next iteration
        n_consumed = n_accept + 1  # accepted prefix + resampled/bonus
        new_draft_tokens = []
        new_draft_dists = []

        for j in range(n_consumed, n_draft):
            if use_reuse:
                # SJD++ (Eq 6): check confidence ratio C_j = p_new/p_old
                tok = current_drafts[0, j].item()
                p_new_j = new_probs[0, j, tok].item()
                p_old_j = current_dist[0, j, tok].item()
                confidence = p_new_j / p_old_j if p_old_j > 0 else float('inf')

                if confidence > reuse_threshold:
                    # Reuse: keep draft token unchanged, update dist to new
                    new_draft_tokens.append(current_drafts[:, j:j+1])
                    new_draft_dists.append(new_probs[:, j:j+1, :])
                    total_reused += 1
                else:
                    # Resample from p_new
                    new_draft_tokens.append(next_tokens[:, j:j+1])
                    new_draft_dists.append(new_probs[:, j:j+1, :])
            else:
                # SJD (Eq 5): use model's sampled prediction as refined draft
                new_draft_tokens.append(next_tokens[:, j:j+1])
                new_draft_dists.append(new_probs[:, j:j+1, :])

        # Fill remaining window with random tokens and one-hot proposal scores,
        # matching the reference implementation's fresh-slot bootstrap.
        n_remaining = len(new_draft_tokens)
        n_new = n_lookahead - n_remaining
        if n_new > 0:
            rand_d = torch.randint(0, vocab_size, (1, n_new), device=device)
            rand_dist = _one_hot_draft_dist(rand_d)
            new_draft_tokens.append(rand_d)
            new_draft_dists.append(rand_dist)

        if new_draft_tokens:
            drafts = torch.cat(new_draft_tokens, dim=1)
            draft_dist = torch.cat(new_draft_dists, dim=1)
        else:
            drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)
            draft_dist = _one_hot_draft_dist(drafts)

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
# PPL evaluation
# ---------------------------------------------------------------------------

def load_eval_batches(device, seq_len=128, batch_size=1, n_batches=8):
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    data_dir = os.path.join(base_dir, 'base_data_climbmix')
    try:
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
    except Exception as e:
        print(f"  Dataset load failed ({e}), using random tokens")
        return [torch.randint(0, 32768, (batch_size, seq_len + 1), device=device) for _ in range(n_batches)]


@torch.inference_mode()
def evaluate_ppl(model, batches):
    """Compute perplexity on batches. Returns average PPL."""
    ppls = []
    for batch in batches:
        idx, targets = batch[:, :-1], batch[:, 1:]
        logits = model.forward(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='mean').item()
        ppls.append(math.exp(min(loss, 20.0)))
    return sum(ppls) / len(ppls)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_stats(stats):
    print(f"  Method:          {stats['method']}")
    print(f"  Prompt tokens:   {stats['n_prompt']}")
    print(f"  Generated:       {stats['n_generated']}")
    print(f"  Forward passes:  {stats['n_forward_passes']}")
    if 'n_steps' in stats:
        print(f"  Accept steps:    {stats['n_steps']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2f} tok/pass")
    if stats.get('n_reused', 0) > 0:
        print(f"  Tokens reused:   {stats['n_reused']}")
    print(f"  Time:            {stats['time']:.3f}s")
    print(f"  Throughput:      {stats['tok_per_sec']:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="SJD demo for nanochat")
    parser.add_argument("--model-tag", type=str, default="d32_baseline")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--mode", type=str, nargs='+', default=["ar", "jd", "sjd", "ppl"],
                        choices=["ar", "jd", "sjd", "sjd++", "ppl"],
                        help="Which modes to run: ar=autoregressive, jd=Jacobi, sjd=Speculative Jacobi, sjd++=with reuse")
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-lookahead", type=int, default=5,
                        help="Number of draft tokens in the lookahead window (SJD)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--reuse-threshold", type=float, default=0.5,
                        help="SJD++ confidence threshold for token reuse (Eq 6). 0=reuse all, higher=more selective")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for PPL eval")
    parser.add_argument("--n-batches", type=int, default=8, help="Number of batches for PPL eval")
    args = parser.parse_args()

    # Force SDPA
    import nanochat.flash_attention as fa
    fa._override_impl = 'sdpa'
    fa.USE_FA3 = False

    device = torch.device(autodetect_device_type())
    model = load_model(args.model_tag, step=args.step, device=device)
    tokenizer = get_tokenizer()

    print(f"\nModel: {args.model_tag} ({model.config.n_layer} layers, {model.config.n_embd}d)")
    print(f"Device: {device}")

    # Encode prompt
    bos = tokenizer.get_bos_token_id()
    prompt_tokens = tokenizer.encode(args.prompt, prepend=bos)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    print(f"\nPrompt ({len(prompt_tokens)} tokens): {args.prompt!r}")

    results = {}

    # --- AR generation ---
    if "ar" in args.mode:
        print(f"\n{'='*60}")
        print("AR Generation")
        print(f"{'='*60}")
        ids_ar, stats_ar = generate_ar(model, input_ids, args.max_new_tokens,
                                        temperature=args.temperature, top_k=args.top_k)
        text_ar = tokenizer.decode(ids_ar[0].tolist())
        print(f"\nOutput: {text_ar!r}")
        print_stats(stats_ar)
        results['ar'] = (ids_ar, stats_ar)

    # --- JD (vanilla Jacobi) generation ---
    if "jd" in args.mode:
        print(f"\n{'='*60}")
        print(f"JD Generation (n_lookahead={args.n_lookahead})")
        print(f"{'='*60}")
        ids_jd, stats_jd = generate_jd(model, input_ids, args.max_new_tokens,
                                        n_lookahead=args.n_lookahead,
                                        temperature=args.temperature, top_k=args.top_k)
        text_jd = tokenizer.decode(ids_jd[0].tolist())
        print(f"\nOutput: {text_jd!r}")
        print_stats(stats_jd)
        results['jd'] = (ids_jd, stats_jd)

    # --- SJD (Speculative Jacobi) generation ---
    if "sjd" in args.mode:
        temp = args.temperature if args.temperature > 0 else 1.0
        topk = args.top_k if args.top_k is not None else 50
        print(f"\n{'='*60}")
        print(f"SJD Generation (n_lookahead={args.n_lookahead}, temp={temp}, top_k={topk})")
        print(f"{'='*60}")
        if args.temperature == 0:
            print("  (Note: SJD requires stochastic sampling; using temp=1.0, top_k=50)")
        ids_sjd, stats_sjd = generate_sjd(model, input_ids, args.max_new_tokens,
                                           n_lookahead=args.n_lookahead,
                                           temperature=temp, top_k=topk)
        text_sjd = tokenizer.decode(ids_sjd[0].tolist())
        print(f"\nOutput: {text_sjd!r}")
        print_stats(stats_sjd)
        results['sjd'] = (ids_sjd, stats_sjd)

    # --- SJD++ generation ---
    if "sjd++" in args.mode:
        temp = args.temperature if args.temperature > 0 else 1.0
        topk = args.top_k if args.top_k is not None else 50
        tau = args.reuse_threshold
        print(f"\n{'='*60}")
        print(f"SJD++ Generation (la={args.n_lookahead}, temp={temp}, top_k={topk}, τ={tau})")
        print(f"{'='*60}")
        if args.temperature == 0:
            print("  (Note: SJD++ requires stochastic sampling; using temp=1.0, top_k=50)")
        ids_sjdpp, stats_sjdpp = generate_sjd(model, input_ids, args.max_new_tokens,
                                               n_lookahead=args.n_lookahead,
                                               temperature=temp, top_k=topk,
                                               reuse_threshold=tau)
        text_sjdpp = tokenizer.decode(ids_sjdpp[0].tolist())
        print(f"\nOutput: {text_sjdpp!r}")
        print_stats(stats_sjdpp)
        results['sjd++'] = (ids_sjdpp, stats_sjdpp)

    # --- Comparisons ---
    if 'ar' in results:
        ar_ids, ar_stats = results['ar']
        for name in ['jd', 'sjd', 'sjd++']:
            if name in results:
                other_ids, other_stats = results[name]
                print(f"\n  --- AR vs {name.upper()} ---")
                min_len = min(ar_ids.shape[1], other_ids.shape[1])
                token_match = (ar_ids[:, :min_len] == other_ids[:, :min_len]).float().mean().item()
                print(f"  Token overlap:  {token_match:.1%}")
                if ar_stats['time'] > 0:
                    speedup = ar_stats['time'] / other_stats['time']
                    print(f"  Speedup:        {speedup:.2f}x")

    # --- PPL evaluation ---
    if "ppl" in args.mode:
        print(f"\n{'='*60}")
        print("PPL Evaluation")
        print(f"{'='*60}")
        batches = load_eval_batches(device, seq_len=args.seq_len, n_batches=args.n_batches)
        ppl = evaluate_ppl(model, batches)
        print(f"  PPL: {ppl:.2f} ({len(batches)} batches, seq_len={args.seq_len})")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
