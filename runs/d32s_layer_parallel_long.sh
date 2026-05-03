#!/bin/bash
# Extended training (9600 steps) for d32s models. Same configs as d32s_layer_parallel.sh
# but with explicit --num-iterations=9600. Separate model tags (*_9600) to preserve
# the 2040-step checkpoints.
#
# 9600 steps = data:param ratio ~56 (vs 12 at 2040 steps).
# Estimated time per run: ~1.4h (baseline/IDN), ~2.3h (diag). Total: ~16h.
#
# Usage:
#   nohup bash runs/d32s_layer_parallel_long.sh > cache/d32s_layer_parallel_long.log 2>&1 &

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

NPROC=4
GPUS="2,3,4,5"
COMMON_ARGS="--depth=32 --aspect-ratio=20 --device-batch-size=8 --num-iterations=9600 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --------------------------------------------------------------------------
# Run 1/4: Baseline
# --------------------------------------------------------------------------
log "Starting Run 1/4: d32s baseline (9600 steps)"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --model-tag=d32s_baseline_9600 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_baseline_9600_train.log"
log "Finished Run 1/4: d32s baseline (9600 steps)"

# --------------------------------------------------------------------------
# Run 2/4: IDN λ=0.5, n_par=8,16,24
# --------------------------------------------------------------------------
log "Starting Run 2/4: d32s IDN λ=0.5 (9600 steps)"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32s_idn05_npar24_9600 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_idn05_npar24_9600_train.log"
log "Finished Run 2/4: d32s IDN λ=0.5 (9600 steps)"

# --------------------------------------------------------------------------
# Run 3/4: Diag FD λ=0.1, n_par=24, stride=3
# --------------------------------------------------------------------------
log "Starting Run 3/4: d32s diag FD λ=0.1 (9600 steps)"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.1 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=24 --diag-stride=3 \
    --model-tag=d32s_diag01_npar24_stride3_9600 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_diag01_npar24_stride3_9600_train.log"
log "Finished Run 3/4: d32s diag FD λ=0.1 (9600 steps)"

# --------------------------------------------------------------------------
# Run 4/4: Diag FD λ=0.5, n_par=24, stride=3
# --------------------------------------------------------------------------
log "Starting Run 4/4: d32s diag FD λ=0.5 (9600 steps)"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=24 --diag-stride=3 \
    --model-tag=d32s_diag05_npar24_stride3_9600 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_diag05_npar24_stride3_9600_train.log"
log "Finished Run 4/4: d32s diag FD λ=0.5 (9600 steps)"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
log "=== RESULTS SUMMARY ==="
for tag in d32s_baseline_9600 d32s_idn05_npar24_9600 d32s_diag01_npar24_stride3_9600 d32s_diag05_npar24_stride3_9600; do
    logf="$NANOCHAT_BASE_DIR/${tag}_train.log"
    if [ -f "$logf" ]; then
        final_bpb=$(grep "Validation bpb" "$logf" | tail -1 | grep -oP '[\d.]+$')
        train_time=$(grep "Total training time" "$logf" | tail -1 | grep -oP '[\d.]+')
        log "$tag: val_bpb=$final_bpb  time=${train_time}m"
    fi
done
log "All runs complete!"
