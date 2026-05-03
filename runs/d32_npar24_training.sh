#!/bin/bash
# Train d32 models (2048 dim, 32 layers) with layer-parallel reg on n_par=8,16,24.
# Same architecture as original d32 runs but with wider parallelism (n_par_configs=8,16,24).
# Baseline already exists at d32_baseline (9600 steps, val_bpb=0.661).
#
# 8×H100, FP8, ~9600 iterations each.
# Estimated time: ~12-15 hours per run, ~50-60 hours total.
#
# Usage:
#   nohup bash runs/d32_npar24_training.sh > cache/d32_npar24_training.log 2>&1 &

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

NPROC=8
COMMON_ARGS="--depth=32 --device-batch-size=8 --fp8 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --------------------------------------------------------------------------
# Run 1/4: IDN λ=0.5, n_par=8,16,24
# --------------------------------------------------------------------------
log "Starting Run 1/4: d32 IDN λ=0.5 n_par=8,16,24"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32_idn05_npar24 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_idn05_npar24_train.log"
log "Finished Run 1/4: d32 IDN λ=0.5"

# --------------------------------------------------------------------------
# Run 2/4: IDN λ=0.1, n_par=8,16,24
# --------------------------------------------------------------------------
log "Starting Run 2/4: d32 IDN λ=0.1 n_par=8,16,24"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --identity-newton-reg=0.1 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32_idn01_npar24 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_idn01_npar24_train.log"
log "Finished Run 2/4: d32 IDN λ=0.1"

# --------------------------------------------------------------------------
# Run 3/4: Diag Newton λ=0.5, n_par=24, stride=3
# --------------------------------------------------------------------------
log "Starting Run 3/4: d32 diag λ=0.5 n_par=24 stride=3"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=24 --diag-stride=3 \
    --model-tag=d32_diag05_npar24_stride3 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag05_npar24_stride3_train.log"
log "Finished Run 3/4: d32 diag λ=0.5"

# --------------------------------------------------------------------------
# Run 4/4: Diag Newton λ=0.1, n_par=24, stride=3
# --------------------------------------------------------------------------
log "Starting Run 4/4: d32 diag λ=0.1 n_par=24 stride=3"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.1 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=24 --diag-stride=3 \
    --model-tag=d32_diag01_npar24_stride3 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag01_npar24_stride3_train.log"
log "Finished Run 4/4: d32 diag λ=0.1"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
log "=== RESULTS SUMMARY ==="
for tag in d32_idn05_npar24 d32_idn01_npar24 d32_diag05_npar24_stride3 d32_diag01_npar24_stride3; do
    logf="$NANOCHAT_BASE_DIR/${tag}_train.log"
    if [ -f "$logf" ]; then
        final_bpb=$(grep "Validation bpb" "$logf" | tail -1 | grep -oP '[\d.]+$')
        train_time=$(grep "Total training time" "$logf" | tail -1 | grep -oP '[\d.]+')
        log "$tag: val_bpb=$final_bpb  time=${train_time}m"
    fi
done
log "All runs complete!"
