#!/bin/bash
# Layer-parallel training on 0.5B model (depth=32, n_embd=640, ~535M params).
# 4 runs: baseline, IDN λ=0.5, diag λ=0.1, diag λ=0.5.
# All with n_par=8,16,24 (IDN) or n_par=24 stride=3 (diag).
#
# Runs on 4×H100 (GPUs 2,3,4,5), bf16, no FP8.
# NOTE: vocab_size=32768 requires power-of-2 GPU count for Muon reduce_scatter.
#
# Model: depth=32, n_embd=640, 5 heads, ~535M params
# Step time: ~1.1s (IDN), ~1.7s (diag stride=3) on 4×H100
# Peak memory: ~20GB per GPU
#
# Usage:
#   nohup bash runs/d32s_layer_parallel.sh > cache/d32s_layer_parallel.log 2>&1 &
#
# Monitor:
#   tail -f cache/d32s_layer_parallel.log
#   grep "step " cache/d32s_*_train.log | tail -5
#   grep "Validation bpb" cache/d32s_*_train.log

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

NPROC=4
GPUS="2,3,4,5"
# bf16, device-batch-size=8
COMMON_ARGS="--depth=32 --aspect-ratio=20 --device-batch-size=8 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --------------------------------------------------------------------------
# Run 1/4: Baseline (no regularization)
# --------------------------------------------------------------------------
log "Starting Run 1/4: d32s baseline"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --model-tag=d32s_baseline \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_baseline_train.log"
log "Finished Run 1/4: d32s baseline"

# --------------------------------------------------------------------------
# Run 2/4: Identity Newton λ=0.5, n_par=8,16,24
# IDN reg is cheap (1 extra block forward per config = 3 total per step).
# --------------------------------------------------------------------------
log "Starting Run 2/4: d32s IDN λ=0.5 n_par=8,16,24"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32s_idn05_npar24 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_idn05_npar24_train.log"
log "Finished Run 2/4: d32s IDN λ=0.5 n_par=8,16,24"

# --------------------------------------------------------------------------
# Run 3/4: Diagonal Newton FD λ=0.1, n_par=24, stride=3
# Estimates diag(J) every 3rd layer (8 FD evals out of 24 layers).
# Skipped layers use J≈I (identity Newton).
# --------------------------------------------------------------------------
log "Starting Run 3/4: d32s diag FD λ=0.1 n_par=24 stride=3"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.1 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=24 --diag-stride=3 \
    --model-tag=d32s_diag01_npar24_stride3 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_diag01_npar24_stride3_train.log"
log "Finished Run 3/4: d32s diag FD λ=0.1 n_par=24 stride=3"

# --------------------------------------------------------------------------
# Run 4/4: Diagonal Newton FD λ=0.5, n_par=24, stride=3
# --------------------------------------------------------------------------
log "Starting Run 4/4: d32s diag FD λ=0.5 n_par=24 stride=3"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=24 --diag-stride=3 \
    --model-tag=d32s_diag05_npar24_stride3 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_diag05_npar24_stride3_train.log"
log "Finished Run 4/4: d32s diag FD λ=0.5 n_par=24 stride=3"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
log "=== RESULTS SUMMARY ==="
for tag in d32s_baseline d32s_idn05_npar24 d32s_diag01_npar24_stride3 d32s_diag05_npar24_stride3; do
    logf="$NANOCHAT_BASE_DIR/${tag}_train.log"
    if [ -f "$logf" ]; then
        final_bpb=$(grep "Validation bpb" "$logf" | tail -1 | grep -oP '[\d.]+$')
        train_time=$(grep "Total training time" "$logf" | tail -1 | grep -oP '[\d.]+')
        log "$tag: val_bpb=$final_bpb  time=${train_time}m"
    fi
done
log "All runs complete!"
log "Checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/d32s_*/"
