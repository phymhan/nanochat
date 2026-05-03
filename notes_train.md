# Training Notes

## Training script

`scripts/base_train.py` — main training entry point.

### Single GPU

```bash
NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache \
  uv run python -m scripts.base_train -- \
  --depth=32 --aspect-ratio=20 --device-batch-size=8 \
  --model-tag=my_model --run=dummy
```

### Multi-GPU (torchrun)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache \
  torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
  --depth=32 --aspect-ratio=20 --device-batch-size=8 \
  --model-tag=my_model --run=dummy
```

**Important:** vocab_size=32768 requires power-of-2 GPU count for Muon reduce_scatter.

### Key args

| Arg | Default | Description |
|-----|---------|-------------|
| `--depth` | 20 | Number of layers |
| `--aspect-ratio` | 64 | n_embd = depth × aspect_ratio (rounded to head_dim multiple) |
| `--head-dim` | 128 | Head dimension |
| `--device-batch-size` | 32 | Per-GPU batch size |
| `--num-iterations` | -1 | Explicit step count (-1 = use data:param ratio) |
| `--target-param-data-ratio` | 12 | Auto-calculate steps from ratio |
| `--save-every` | -1 | Checkpoint interval (-1 = only at end) |
| `--eval-every` | 250 | Val BPB evaluation interval |
| `--model-tag` | auto | Checkpoint directory name |

### Layer-parallel regularization

| Arg | Description |
|-----|-------------|
| `--identity-newton-reg=0.5` | IDN reg weight (trains layers for J≈I) |
| `--diag-newton-reg=0.1` | Diagonal Newton reg weight |
| `--jacobi-reg-warmup=0.2` | Fraction of training for reg warmup |
| `--n-par-configs=8,16,24` | Comma-separated n_par values for reg |
| `--diag-newton-jvp=fd` | Diag estimation method: fd or vjp |
| `--diag-stride=3` | Estimate diag every N layers |

### Architecture ablations

| Arg | Description |
|-----|-------------|
| `--no-x0-resid` | Disable x0_lambdas and resid scaling (standard transformer residual) |
| `--no-ve` | Disable value embeddings in forward (params still exist, just unused) |

### d32s model config

- depth=32, aspect_ratio=20 → n_embd=640, n_head=5, head_dim=128
- ~535M params
- Step time: ~1.0s (baseline), ~1.1s (IDN), ~1.7s (diag stride=3) on 4×H100
- Peak memory: ~20GB per GPU

### Existing training runs

```bash
# Original runs (with x0_resid and VE)
bash runs/d32s_layer_parallel.sh    # baseline, idn05, diag01, diag05

# Ablation runs (no x0_resid, no VE)
bash runs/d32s_nox0ve.sh            # nox0ve_baseline, nox0ve_idn05
```

### Example: train baseline + IDN (4800 steps, 8 GPUs)

```bash
nohup bash -c '
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

COMMON="--depth=32 --aspect-ratio=20 --device-batch-size=8 --num-iterations=4800 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1"

# Baseline on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m scripts.base_train -- $COMMON --model-tag=d32s_baseline --run=dummy \
  2>&1 | tee cache/baseline_train.log &

# IDN on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m scripts.base_train -- $COMMON \
  --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 --n-par-configs=8,16,24 \
  --model-tag=d32s_idn05_npar24 --run=dummy \
  2>&1 | tee cache/idn05_train.log &

wait
' > cache/train.log 2>&1 &
```

### Checkpoints

Saved to `$NANOCHAT_BASE_DIR/base_checkpoints/{model_tag}/`:
- `model_{step:06d}.pt` — model weights
- `meta_{step:06d}.json` — config, val_bpb, user_config
- `optim_{step:06d}_rank{N}.pt` — optimizer state per rank

The `load_model` function in `scripts/eval_jacobi_idn.py` loads checkpoints by tag name,
finding the latest step automatically.

### Monitoring

```bash
# Watch training loss
tail -f cache/my_train.log

# Check val BPB
grep "Validation bpb" cache/my_train.log

# Check step progress
grep "^step " cache/my_train.log | tail -5
```
