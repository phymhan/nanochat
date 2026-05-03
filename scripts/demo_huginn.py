"""
Basic Huginn depth-recurrent model demo.

Loads tomg-group-umd/huginn-0125 (3.5B), runs:
1. Text generation
2. PPL evaluation on wikitext-2 (~10K tokens)
3. Convergence diagnostics (||x_{k+1} - x_k|| / ||x_k|| per iteration)

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run python -m scripts.demo_huginn
"""

import argparse
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="tomg-group-umd/huginn-0125", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, prompt="The meaning of life is", max_new_tokens=64):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def eval_ppl(model, tokenizer, max_tokens=10_000, seq_len=128):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    losses = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i : i + seq_len + 1].unsqueeze(0).to(model.device)
        idx, tgt = chunk[:, :-1], chunk[:, 1:]
        out = model(idx, labels=tgt)
        losses.append(out.loss.item())

    ppl = math.exp(sum(losses) / len(losses))
    return ppl, len(losses) * seq_len


@torch.no_grad()
def convergence_diagnostics(model, tokenizer, prompt="The quick brown fox jumps over the lazy dog", num_steps=32):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    input_embeds, block_idx = model.embed_inputs(input_ids)
    x = model.initialize_state(input_embeds)

    print(f"\n{'Step':>4}  {'||x_new - x||':>14}  {'||x_new - x|| / ||x||':>22}  {'||F(x) - x||':>14}")
    print("-" * 62)

    for k in range(num_steps):
        x_prev = x.clone()
        x, block_idx, _ = model.iterate_one_step(
            input_embeds, x, block_idx=block_idx, current_step=k)

        diff = (x - x_prev).float().norm().item()
        rel_diff = diff / (x_prev.float().norm().item() + 1e-8)
        residual = diff  # ||F(x) - x|| = ||x_new - x_prev|| since x_new = F(x_prev)
        print(f"{k:4d}  {diff:14.6f}  {rel_diff:22.8f}  {residual:14.6f}")

    logits = model.predict_from_latents(x).logits
    last_token_logits = logits[0, -1]
    top5 = torch.topk(last_token_logits, 5)
    print(f"\nTop-5 predictions after {num_steps} iterations:")
    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i+1}. {tokenizer.decode([idx.item()])!r:>12}  (logit={val.item():.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=10_000)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, args.device)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"  n_embd={model.config.n_embd}, n_heads={model.config.num_attention_heads}")
    print(f"  mean_recurrence={model.config.mean_recurrence}")
    print(f"  n_layers_in_recurrent_block={model.config.n_layers_in_recurrent_block}")
    print(f"  prelude={model.config.n_layers_in_prelude}, coda={model.config.n_layers_in_coda}")

    if not args.skip_gen:
        print("\n--- Text Generation ---")
        for prompt in ["The meaning of life is", "Machine learning is"]:
            text = generate_text(model, tokenizer, prompt)
            print(f"Prompt: {prompt!r}")
            print(f"Output: {text}\n")

    if not args.skip_ppl:
        print("--- PPL Evaluation ---")
        t0 = time.time()
        ppl, n_tokens = eval_ppl(model, tokenizer, max_tokens=args.max_tokens)
        print(f"PPL = {ppl:.2f} ({n_tokens:,} tokens, {time.time() - t0:.1f}s)")

    print("\n--- Convergence Diagnostics ---")
    convergence_diagnostics(model, tokenizer, num_steps=args.num_steps)


if __name__ == "__main__":
    main()
