"""
Fine-tune TinyLlama 1.1B with IDN / diag-Newton regularization on ClimbMix data.

Trains the model to have input-invariant later layers (J ≈ I), enabling
layer-parallel inference via identity Newton correction.

Usage (single GPU, quick test):
    HF_HOME=/mnt/nvme0n1/ligong/huggingface \
    NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache \
    CUDA_VISIBLE_DEVICES=0 \
    uv run --extra gpu --group dev python -m scripts.train_idn_tinyllama \
        --num-steps=100 --eval-every=50 --save-every=-1

Usage (4 GPU):
    HF_HOME=/mnt/nvme0n1/ligong/huggingface \
    NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc_per_node=4 -m scripts.train_idn_tinyllama \
        --identity-newton-reg=0.5 --num-steps=2000
"""

import argparse
import json
import math
import os
import time

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_ddp():
    return dist.is_initialized()

def rank():
    return dist.get_rank() if is_ddp() else 0

def world_size():
    return dist.get_world_size() if is_ddp() else 1

def print0(*args, **kwargs):
    if rank() == 0:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CLIMBMIX_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542

def get_data_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR", "cache")
    return os.path.join(base, "base_data_climbmix")


def download_shard(index, data_dir=None):
    import requests
    if data_dir is None:
        data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath
    url = f"{CLIMBMIX_URL}/{filename}"
    print(f"Downloading {filename}...")
    for attempt in range(5):
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            tmp = filepath + ".tmp"
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(tmp, filepath)
            print(f"Downloaded {filename}")
            return filepath
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            for p in [filepath + ".tmp", filepath]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            if attempt < 4:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Failed to download {filename}")


def data_iterator(tokenizer, seq_len, device, n_shards=10, batch_size=1,
                  ddp_rank=0, ddp_world=1):
    """Yield (input_ids, targets) batches from ClimbMix parquet shards."""
    data_dir = get_data_dir()

    # Download shards on rank 0, barrier, then all ranks read
    if ddp_rank == 0:
        for i in range(n_shards):
            download_shard(i, data_dir)
    if is_ddp():
        dist.barrier()

    token_buffer = []

    while True:
        for shard_idx in range(n_shards):
            filepath = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(ddp_rank, pf.num_row_groups, ddp_world):
                texts = pf.read_row_group(rg_idx).column('text').to_pylist()
                for text in texts:
                    enc = tokenizer(text, add_special_tokens=True, truncation=False)
                    token_buffer.extend(enc['input_ids'])

                    while len(token_buffer) >= batch_size * (seq_len + 1):
                        batch_inputs = []
                        batch_targets = []
                        for _ in range(batch_size):
                            chunk = token_buffer[:seq_len + 1]
                            token_buffer = token_buffer[seq_len:]
                            t = torch.tensor(chunk, dtype=torch.long, device=device)
                            batch_inputs.append(t[:-1])
                            batch_targets.append(t[1:])
                        yield torch.stack(batch_inputs), torch.stack(batch_targets)


def load_val_data(tokenizer, device, seq_len=128, max_tokens=100_000):
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


# ---------------------------------------------------------------------------
# Forward with hidden states
# ---------------------------------------------------------------------------

def forward_with_hidden_states(model, input_ids):
    """Run full sequential forward, saving all intermediate hidden states.

    Returns (logits, all_h, position_embeddings) where all_h[i] is the output
    of layer i (0-indexed). all_h has n_layer elements.
    """
    m = model.module if isinstance(model, DDP) else model
    hidden = m.model.embed_tokens(input_ids)
    B, T, D = hidden.shape
    position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
    position_embeddings = m.model.rotary_emb(hidden, position_ids)

    from transformers.masking_utils import create_causal_mask
    cache_position = torch.arange(T, device=input_ids.device)
    causal_mask = create_causal_mask(
        config=m.config, input_embeds=hidden,
        attention_mask=None, cache_position=cache_position,
        past_key_values=None, position_ids=position_ids,
    )

    all_h = []
    for i in range(m.config.num_hidden_layers):
        layer = m.model.layers[i]
        mask = causal_mask
        if isinstance(causal_mask, dict):
            mask = causal_mask.get(getattr(layer, 'attention_type', 'full_attention'), None)
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        all_h.append(hidden)

    hidden = m.model.norm(hidden)
    logits = m.lm_head(hidden).float()
    return logits, all_h, position_embeddings


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------

def compute_idn_loss(model, all_h, position_embeddings, n_par_configs):
    """Identity Newton reg: penalize gap between last-layer(h_init) and sequential output.

    For each n_par config, runs the last layer on the prefix output h_init
    instead of the true sequential input. Penalizes the relative L2 distance.
    """
    m = model.module if isinstance(model, DDP) else model
    n_layer = m.config.num_hidden_layers
    loss = torch.tensor(0.0, device=all_h[0].device)

    from transformers.masking_utils import create_causal_mask
    B, T, D = all_h[0].shape
    position_ids = torch.arange(T, device=all_h[0].device).unsqueeze(0)
    cache_position = torch.arange(T, device=all_h[0].device)
    causal_mask = create_causal_mask(
        config=m.config, input_embeds=all_h[0],
        attention_mask=None, cache_position=cache_position,
        past_key_values=None, position_ids=position_ids,
    )
    last_layer = m.model.layers[n_layer - 1]
    mask = causal_mask
    if isinstance(causal_mask, dict):
        mask = causal_mask.get(getattr(last_layer, 'attention_type', 'full_attention'), None)

    h_seq_final = all_h[n_layer - 1].detach()

    for n_par in n_par_configs:
        if n_par >= n_layer:
            continue
        seq_layers = n_layer - n_par
        h_init = all_h[seq_layers - 1].detach() if seq_layers > 0 else all_h[0].detach()

        h_newton = last_layer(
            h_init,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        diff = (h_newton.float() - h_seq_final.float()).norm(dim=-1)
        ref = h_seq_final.float().norm(dim=-1).clamp(min=1e-6)
        loss = loss + (diff / ref).mean()

    return loss


def compute_diag_newton_loss(model, all_h, position_embeddings, n_par_configs,
                              diag_stride=3, fd_eps=1e-2):
    """Diagonal Newton reg: estimate diag(J) via FD, do forward-substitution correction.

    For each n_par config, runs all parallel layers on h_init, estimates
    diagonal Jacobian via finite differences (Hutchinson), applies the
    diagonal Newton correction, and penalizes the gap to sequential output.
    """
    m = model.module if isinstance(model, DDP) else model
    n_layer = m.config.num_hidden_layers
    loss = torch.tensor(0.0, device=all_h[0].device)

    from transformers.masking_utils import create_causal_mask
    B, T, D = all_h[0].shape
    position_ids = torch.arange(T, device=all_h[0].device).unsqueeze(0)
    cache_position = torch.arange(T, device=all_h[0].device)
    causal_mask = create_causal_mask(
        config=m.config, input_embeds=all_h[0],
        attention_mask=None, cache_position=cache_position,
        past_key_values=None, position_ids=position_ids,
    )

    h_seq_final = all_h[n_layer - 1].detach()

    for n_par in n_par_configs:
        if n_par >= n_layer:
            continue
        seq_layers = n_layer - n_par
        h_init = all_h[seq_layers - 1].detach() if seq_layers > 0 else all_h[0].detach()

        z = torch.randint(0, 2, h_init.shape, device=h_init.device, dtype=h_init.dtype) * 2 - 1

        h_newton = []
        diags = []
        for j in range(n_par):
            li = seq_layers + j
            layer = m.model.layers[li]
            mask = causal_mask
            if isinstance(causal_mask, dict):
                mask = causal_mask.get(getattr(layer, 'attention_type', 'full_attention'), None)

            layer_kwargs = dict(
                attention_mask=mask, position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            estimate_diag = (diag_stride <= 1 or j % diag_stride == 0)
            h_j = layer(h_init, **layer_kwargs)

            if estimate_diag:
                with torch.no_grad():
                    h_j_pert = layer(h_init + fd_eps * z, **layer_kwargs)
                    diag_j = z * (h_j_pert.float() - h_j.float().detach()) / fd_eps
                    diag_j = diag_j.clamp(-2, 2)
                diags.append(diag_j.to(h_j.dtype).detach())
            else:
                diags.append(None)
            h_newton.append(h_j)

        # Forward substitution correction
        h_corr = h_newton[0]
        for j in range(1, n_par):
            if diags[j] is not None:
                h_corr = h_newton[j] + diags[j] * (h_corr - h_init)
            else:
                h_corr = h_newton[j] + (h_corr - h_init)

        diff = (h_corr.float() - h_seq_final.float()).norm(dim=-1)
        ref = h_seq_final.float().norm(dim=-1).clamp(min=1e-6)
        loss = loss + (diff / ref).mean()

    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_ppl(model, val_batches):
    m = model.module if isinstance(model, DDP) else model
    m.eval()
    losses = []
    for batch in val_batches:
        idx, tgt = batch[:, :-1], batch[:, 1:]
        logits = m(idx).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        losses.append(loss.item())
    m.train()
    if not losses:
        return float('inf')
    return math.exp(sum(losses) / len(losses))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--n-shards", type=int, default=10, help="Number of ClimbMix shards to use")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="cache/tinyllama_checkpoints")

    parser.add_argument("--identity-newton-reg", type=float, default=0.0)
    parser.add_argument("--diag-newton-reg", type=float, default=0.0)
    parser.add_argument("--diag-stride", type=int, default=3)
    parser.add_argument("--n-par-configs", type=str, default=None,
                        help="Comma-separated n_par values, e.g. 4,8,12")
    parser.add_argument("--reg-warmup", type=float, default=0.2,
                        help="Fraction of training for reg warmup")
    parser.add_argument("--model-tag", type=str, default=None)

    args = parser.parse_args()

    # DDP init
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print0(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.train()

    L = model.config.num_hidden_layers
    print0(f"Model: L={L}, D={model.config.hidden_size}, "
           f"H={model.config.num_attention_heads}, KV={model.config.num_key_value_heads}")

    # Parse n_par configs
    if args.n_par_configs:
        n_par_configs = [int(x) for x in args.n_par_configs.split(',')]
    else:
        n_par_configs = [4, 8, min(12, L - 1), min(16, L - 1)]
    print0(f"n_par_configs: {n_par_configs}")
    print0(f"IDN reg: {args.identity_newton_reg}, Diag reg: {args.diag_newton_reg}")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR schedule: linear warmup + cosine decay
    def get_lr(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.num_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    # Data
    print0(f"Setting up data loader ({args.n_shards} shards)...")
    train_iter = data_iterator(
        tokenizer, args.seq_len, device, n_shards=args.n_shards,
        batch_size=args.batch_size, ddp_rank=rank(), ddp_world=world_size(),
    )

    print0("Loading validation data...")
    val_batches = load_val_data(tokenizer, device, seq_len=128, max_tokens=50_000)
    print0(f"Val: {len(val_batches)} batches")

    # Output dir
    tag = args.model_tag or f"tinyllama_idn{args.identity_newton_reg}_diag{args.diag_newton_reg}"
    output_dir = os.path.join(args.output_dir, tag)
    if rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Training loop
    print0(f"\nStarting training: {args.num_steps} steps, bs={args.batch_size}, seq_len={args.seq_len}")
    smooth_loss = 0.0
    t_start = time.time()

    for step in range(args.num_steps):
        # Eval
        if args.eval_every > 0 and step % args.eval_every == 0:
            ppl = evaluate_ppl(model, val_batches)
            print0(f"step {step}: val PPL = {ppl:.2f}")

        # Save
        if args.save_every > 0 and step > 0 and step % args.save_every == 0 and rank() == 0:
            m = model.module if isinstance(model, DDP) else model
            ckpt_path = os.path.join(output_dir, f"model_{step:06d}.pt")
            torch.save(m.state_dict(), ckpt_path)
            print0(f"Saved checkpoint: {ckpt_path}")

        # Forward
        model.train()
        x, y = next(train_iter)
        t0 = time.time()

        has_reg = (args.identity_newton_reg > 0 or args.diag_newton_reg > 0)
        if has_reg:
            logits, all_h, pos_emb = forward_with_hidden_states(model, x)
        else:
            m = model.module if isinstance(model, DDP) else model
            logits = m(x).logits.float()

        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Reg losses
        reg_warmup = min(1.0, step / max(1, int(args.reg_warmup * args.num_steps))) if args.reg_warmup > 0 else 1.0
        idn_loss = torch.tensor(0.0, device=device)
        diag_loss = torch.tensor(0.0, device=device)

        if args.identity_newton_reg > 0 and has_reg:
            idn_loss = compute_idn_loss(model, all_h, pos_emb, n_par_configs)

        if args.diag_newton_reg > 0 and has_reg:
            diag_loss = compute_diag_newton_loss(
                model, all_h, pos_emb, n_par_configs,
                diag_stride=args.diag_stride,
            )

        total_loss = ce_loss + reg_warmup * (
            args.identity_newton_reg * idn_loss +
            args.diag_newton_reg * diag_loss
        )

        # Backward
        total_loss.backward()

        # LR schedule
        lr_mult = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_mult

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0

        # Logging
        ce_f = ce_loss.item()
        smooth_loss = 0.9 * smooth_loss + 0.1 * ce_f if step > 0 else ce_f

        if step % args.log_every == 0:
            idn_f = idn_loss.item() if args.identity_newton_reg > 0 else 0
            diag_f = diag_loss.item() if args.diag_newton_reg > 0 else 0
            elapsed = time.time() - t_start
            print0(f"step {step:5d} | ce={ce_f:.4f} smooth={smooth_loss:.4f} | "
                   f"idn={idn_f:.4f} diag={diag_f:.4f} | "
                   f"lr={args.lr * lr_mult:.2e} warmup={reg_warmup:.2f} | "
                   f"dt={dt*1000:.0f}ms elapsed={elapsed:.0f}s")

    # Final eval
    ppl = evaluate_ppl(model, val_batches)
    print0(f"\nFinal val PPL = {ppl:.2f}")

    # Final save
    if rank() == 0:
        m = model.module if isinstance(model, DDP) else model
        ckpt_path = os.path.join(output_dir, f"model_{args.num_steps:06d}.pt")
        torch.save(m.state_dict(), ckpt_path)
        meta = {
            'model': args.model, 'step': args.num_steps, 'val_ppl': ppl,
            'identity_newton_reg': args.identity_newton_reg,
            'diag_newton_reg': args.diag_newton_reg,
            'n_par_configs': n_par_configs,
            'lr': args.lr, 'seq_len': args.seq_len,
            'batch_size': args.batch_size, 'n_shards': args.n_shards,
        }
        with open(os.path.join(output_dir, f"meta_{args.num_steps:06d}.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print0(f"Saved final checkpoint: {ckpt_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
