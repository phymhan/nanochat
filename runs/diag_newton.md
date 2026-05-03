# Diagonal Newton Training Experiments

Three d32 models trained sequentially on 8×H100 with FP8. All use default
Chinchilla-optimal training horizon (target-param-data-ratio=12, ~9,600 iters).

## Commands

### Run all three sequentially

```bash
bash runs/diag_newton.sh
# or in screen:
screen -L -Logfile runs/diag_newton.log -S diag_newton bash runs/diag_newton.sh
```

### Individual runs

**Run 1: Baseline (no regularization)**
```bash
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 \
    --save-every=2000 --eval-every=500 --sample-every=2000 --core-metric-every=-1 \
    --model-tag=d32_baseline --run=d32_baseline
```
~12 hours, 60 GiB peak memory.

**Run 2: Identity Newton, n_par=8**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 --n-par-configs=8 \
    --save-every=2000 --eval-every=500 --sample-every=2000 --core-metric-every=-1 \
    --model-tag=d32_idn05_npar8 --run=d32_idn05_npar8
```
~13 hours (IDN only evaluates the last layer per config).

**Run 3: Diagonal Newton FD, n_par=8**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 \
    --diag-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=8 \
    --save-every=2000 --eval-every=500 --sample-every=2000 --core-metric-every=-1 \
    --model-tag=d32_diag05_npar8 --run=d32_diag05_npar8
```
~15.5 hours, 67 GiB peak memory.

## Checkpoint locations

All under `$NANOCHAT_BASE_DIR/base_checkpoints/`:
- `d32_baseline/` — no reg
- `d32_idn05_npar8/` — identity Newton λ=0.5
- `d32_diag05_npar8/` — diagonal Newton λ=0.5, FD Hutchinson

Each checkpoint directory contains:
- `model_NNNNNN.pt` — model weights
- `optim_NNNNNN_rankN.pt` — per-rank optimizer state
- `meta_NNNNNN.json` — metadata (config, step, val_bpb, dataloader state)

Checkpoints saved every 2000 steps + at the end of training.

## Evaluation

After training, evaluate all three:
```bash
for tag in d32_baseline d32_idn05_npar8 d32_diag05_npar8; do
    python -m scripts.base_eval --model-tag=$tag --device-batch-size=8 --eval bpb,sample
done
```

For layer-parallel inference evaluation, see `scripts/jacobi_eval.py`.

## What to compare

1. **Base PPL** — does diag Newton degrade less than IDN?
2. **Layer-parallel fidelity** — at n_par=8, does diag Newton achieve higher top-1
   accuracy and lower PPL delta than baseline (untrained) with K=1 diagonal scan?
3. **Comparison with IDN** — does diag Newton's weaker constraint yield a better
   base-quality vs parallelism tradeoff?
