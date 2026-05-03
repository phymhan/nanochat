"""
Speculative Jacobi Decoding (SJD) demo for Qwen2.5-0.5B-Instruct.

Implements SJD (arXiv:2512.07503) along the token (sequence) axis:
  Standard AR generates one token at a time.
  SJD maintains a lookahead window of n_lookahead draft tokens, runs the model
  on the full sequence (prefix + drafts), checks which draft positions converge
  (model prediction == current draft), and accepts the converged prefix.

Two modes:
  1. AR:  standard autoregressive decoding (baseline)
  2. SJD: Speculative Jacobi Decoding with configurable lookahead

Also evaluates PPL on wikitext-2 (or random data as fallback).

Usage:
    # AR generation
    uv run python -m scripts.demo_sjd_qwen --mode ar --max-new-tokens 64

    # SJD generation
    uv run python -m scripts.demo_sjd_qwen --mode sjd --n-lookahead 5 --max-new-tokens 64

    # Both + PPL
    uv run python -m scripts.demo_sjd_qwen --mode ar sjd ppl

    # Multiple prompts
    uv run python -m scripts.demo_sjd_qwen --prompt "Hello" "The meaning of life is"
"""

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, dtype, device):
    print(f"Loading {model_name} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True,
        dtype=dtype, device_map=device, low_cpu_mem_usage=True,
    ).eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# AR generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_ar(model, input_ids, max_new_tokens, temperature=0.0, top_k=None,
                eos_token_id=None):
    """
    Standard autoregressive generation using HF's model.generate() for
    correctness, but implemented manually for fair timing comparison.
    """
    device = input_ids.device
    ids = input_ids.clone()  # (1, T)
    n_prompt = ids.shape[1]

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        outputs = model(ids)
        logits = outputs.logits[:, -1, :]  # (1, V)

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
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

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
# SJD generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_sjd(model, input_ids, max_new_tokens, n_lookahead=5,
                 temperature=0.0, top_k=None, eos_token_id=None,
                 max_jacobi_iters=32):
    """
    Speculative Jacobi Decoding along the token axis.

    Uses Jacobi iteration to refine draft tokens:
      1. Forward pass on [prefix | drafts] → logits for all positions
      2. Compute model predictions for each draft position
      3. Check convergence: draft[j] is a fixed point if prediction[j] == draft[j]
      4. If converged prefix found → accept + bonus, shift window, update drafts
      5. If no convergence → update ALL drafts to model predictions (Jacobi update),
         repeat from step 1 (up to max_jacobi_iters)
      6. After max iters → accept first predicted token (AR fallback)

    Returns (output_ids, stats_dict).
    """
    device = input_ids.device
    vocab_size = model.config.vocab_size
    ids = input_ids.clone()  # (1, T)
    n_prompt = ids.shape[1]

    # Initialize draft tokens (random — only for the very first window)
    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)

    total_accepted = 0
    total_forward_passes = 0
    total_steps = 0

    def _sample(logits):
        if temperature > 0:
            logits_s = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits_s, min(top_k, logits_s.size(-1)), dim=-1)
                logits_s[logits_s < v[..., [-1]]] = -float('inf')
            probs = F.softmax(logits_s, dim=-1)
            return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.shape[:-1])
        return logits.argmax(dim=-1)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    while total_accepted < max_new_tokens:
        remaining = max_new_tokens - total_accepted
        n_draft = min(n_lookahead, remaining - 1) if remaining > 1 else 0
        current_drafts = drafts[:, :n_draft]

        if n_draft == 0:
            outputs = model(ids)
            total_forward_passes += 1
            next_token = _sample(outputs.logits)[:, -1:]
            ids = torch.cat([ids, next_token], dim=1)
            total_accepted += 1
            total_steps += 1
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            continue

        # --- Jacobi iteration loop ---
        accepted_this_step = False
        for ji in range(max_jacobi_iters):
            full_seq = torch.cat([ids, current_drafts], dim=1)
            outputs = model(full_seq)
            logits = outputs.logits
            total_forward_passes += 1
            predicted = _sample(logits)

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
                    updated_remaining = model_preds[:, n_consumed:]
                    n_new = n_lookahead - updated_remaining.shape[1]
                    new_drafts = torch.randint(0, vocab_size, (1, n_new), device=device)
                    drafts = torch.cat([updated_remaining, new_drafts], dim=1)
                else:
                    drafts = torch.randint(0, vocab_size, (1, n_lookahead), device=device)

                if eos_token_id is not None and ids[0, -1].item() == eos_token_id:
                    break
                break

            # No convergence — Jacobi update: drafts ← model predictions
            current_drafts = model_preds.clone()

        if not accepted_this_step:
            bonus = predicted[:, T-1:T]
            ids = torch.cat([ids, bonus], dim=1)
            total_accepted += 1
            total_steps += 1

            updated = model_preds[:, 1:]
            n_new = n_lookahead - updated.shape[1]
            new_drafts = torch.randint(0, vocab_size, (1, n_new), device=device)
            drafts = torch.cat([updated, new_drafts], dim=1)

            if eos_token_id is not None and ids[0, -1].item() == eos_token_id:
                break

    # Trim to exact max_new_tokens (excluding EOS)
    max_len = n_prompt + max_new_tokens
    if ids.shape[1] > max_len:
        if eos_token_id is not None and ids[0, -1].item() == eos_token_id:
            pass  # keep the EOS
        else:
            ids = ids[:, :max_len]

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()

    n_gen = ids.shape[1] - n_prompt
    acceptance_rate = n_gen / total_forward_passes if total_forward_passes > 0 else 0
    return ids, {
        'method': f'SJD(lookahead={n_lookahead})',
        'n_prompt': n_prompt,
        'n_generated': n_gen,
        'time': t1 - t0,
        'tok_per_sec': n_gen / (t1 - t0) if t1 > t0 else float('inf'),
        'n_forward_passes': total_forward_passes,
        'n_steps': total_steps,
        'acceptance_rate': acceptance_rate,
    }


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_ppl(model, tokenizer, device, seq_len=256, n_batches=8):
    """Compute PPL on wikitext-2-raw-v1 or random data as fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in ds["text"] if t.strip()])
        enc = tokenizer(text, return_tensors="pt", truncation=False)
        all_ids = enc["input_ids"][0]
    except Exception as e:
        print(f"  Wikitext load failed ({e}), using random tokens")
        all_ids = torch.randint(0, model.config.vocab_size, (seq_len * n_batches * 2,))

    ppls = []
    offset = 0
    for _ in range(n_batches):
        if offset + seq_len + 1 > len(all_ids):
            break
        chunk = all_ids[offset:offset + seq_len + 1].unsqueeze(0).to(device)
        idx, targets = chunk[:, :-1], chunk[:, 1:]
        outputs = model(idx)
        logits = outputs.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='mean').item()
        ppls.append(math.exp(min(loss, 20.0)))
        offset += seq_len

    return sum(ppls) / len(ppls) if ppls else float('inf')


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
    print(f"  Time:            {stats['time']:.3f}s")
    print(f"  Throughput:      {stats['tok_per_sec']:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="SJD demo for Qwen2.5-0.5B")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mode", type=str, nargs='+', default=["ar", "sjd", "ppl"],
                        choices=["ar", "sjd", "ppl"])
    parser.add_argument("--prompt", type=str, nargs='+',
                        default=["The quick brown fox jumps over the lazy dog"],
                        help="One or more prompts to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-lookahead", type=int, default=5,
                        help="Number of draft tokens in the lookahead window (SJD)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Skip chat template and use raw prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seq-len", type=int, default=256, help="Seq length for PPL eval")
    parser.add_argument("--n-batches", type=int, default=8)
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.model, dtype, device)

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"\nModel: {args.model} ({n_layers} layers, {hidden}d)")
    print(f"Device: {device}, dtype: {args.dtype}")

    for prompt_text in args.prompt:
        # Encode prompt
        if not args.no_chat_template and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt_text}]
            encoded_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            encoded_text = prompt_text

        enc = tokenizer(encoded_text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        print(f"\nPrompt ({input_ids.shape[1]} tokens): {prompt_text!r}")

        # --- AR generation ---
        if "ar" in args.mode:
            print(f"\n  {'─'*50}")
            print(f"  AR Generation")
            print(f"  {'─'*50}")
            ids_ar, stats_ar = generate_ar(
                model, input_ids, args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id)
            text_ar = tokenizer.decode(ids_ar[0], skip_special_tokens=False)
            print(f"  Output: {text_ar!r}")
            print_stats(stats_ar)

        # --- SJD generation ---
        if "sjd" in args.mode:
            print(f"\n  {'─'*50}")
            print(f"  SJD Generation (n_lookahead={args.n_lookahead})")
            print(f"  {'─'*50}")
            ids_sjd, stats_sjd = generate_sjd(
                model, input_ids, args.max_new_tokens,
                n_lookahead=args.n_lookahead,
                temperature=args.temperature, top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id)
            text_sjd = tokenizer.decode(ids_sjd[0], skip_special_tokens=False)
            print(f"  Output: {text_sjd!r}")
            print_stats(stats_sjd)

        # --- Compare ---
        if "ar" in args.mode and "sjd" in args.mode:
            print(f"\n  {'─'*50}")
            print(f"  Comparison")
            print(f"  {'─'*50}")
            min_len = min(ids_ar.shape[1], ids_sjd.shape[1])
            match = torch.equal(ids_ar[:, :min_len], ids_sjd[:, :min_len])
            token_match = (ids_ar[:, :min_len] == ids_sjd[:, :min_len]).float().mean().item()
            print(f"  Exact match:    {'YES' if match else 'NO'}")
            print(f"  Token overlap:  {token_match:.1%}")
            if stats_ar['time'] > 0:
                speedup = stats_ar['time'] / stats_sjd['time']
                print(f"  SJD speedup:    {speedup:.2f}x")

    # --- PPL evaluation ---
    if "ppl" in args.mode:
        print(f"\n{'='*60}")
        print("PPL Evaluation")
        print(f"{'='*60}")
        ppl = evaluate_ppl(model, tokenizer, device,
                           seq_len=args.seq_len, n_batches=args.n_batches)
        print(f"  PPL: {ppl:.2f} ({args.n_batches} batches, seq_len={args.seq_len})")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
