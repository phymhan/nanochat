#!/bin/bash
# Remaining runs: IDN and diag Newton.
# Baseline already done: val_bpb=0.661, checkpoint at d32_baseline/
#
# Usage:
#   nohup bash runs/diag_newton_remaining.sh > cache/diag_newton_remaining.log 2>&1 &

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

NPROC=8
COMMON_ARGS="--depth=32 --device-batch-size=8 --fp8 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --------------------------------------------------------------------------
# Run 2: Identity Newton, n_par=8
# --------------------------------------------------------------------------
log "Starting Run 2/3: d32 IDN n_par=8"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.0 \
    --n-par-configs=8 \
    --model-tag=d32_idn05_npar8 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_idn05_npar8_train.log"
log "Finished Run 2/3: d32 IDN n_par=8"

# --------------------------------------------------------------------------
# Run 3: Diagonal Newton FD, n_par=8
# --------------------------------------------------------------------------
log "Starting Run 3/3: d32 diag FD n_par=8"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.5 --jacobi-reg-warmup=0.0 \
    --diag-newton-jvp=fd --n-par-configs=8 \
    --model-tag=d32_diag05_npar8 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag05_npar8_train.log"
log "Finished Run 3/3: d32 diag FD n_par=8"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
log "=== RESULTS SUMMARY ==="
for tag in d32_baseline d32_idn05_npar8 d32_diag05_npar8; do
    logf="$NANOCHAT_BASE_DIR/${tag}_train.log"
    final_bpb=$(grep "Validation bpb" "$logf" | tail -1 | grep -oP '[\d.]+$')
    train_time=$(grep "Total training time" "$logf" | tail -1 | grep -oP '[\d.]+')
    log "$tag: val_bpb=$final_bpb  time=${train_time}m"
done
log "All runs complete!"
