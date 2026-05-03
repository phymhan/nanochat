#!/bin/bash
# Diagonal Newton experiment: baseline vs IDN vs diag Newton on d32.
# Runs three experiments sequentially on 8×H100 with FP8.
# Total estimated time: ~12 + ~13 + ~15.5 ≈ 40 hours.
#
# Usage:
#   bash runs/diag_newton.sh
#   # or in a screen session:
#   screen -L -Logfile runs/diag_newton.log -S diag_newton bash runs/diag_newton.sh
#
# Monitor:
#   tail -f cache/diag_newton_all.log
#   # or just losses:
#   grep "step " cache/d32_baseline_train.log | tail -5
#   # or val bpb:
#   grep "Validation bpb" cache/d32_baseline_train.log

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

NPROC=8
# save-every=-1: only save final checkpoint (~11GB per save, disk is tight)
# eval-every=500: val bpb every 500 steps for loss curves
COMMON_ARGS="--depth=32 --device-batch-size=8 --fp8 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --------------------------------------------------------------------------
# Run 1: Baseline (no regularization)
# Expected: ~12 hours, ~9600 iterations
# --------------------------------------------------------------------------
log "Starting Run 1/3: d32 baseline"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --model-tag=d32_baseline \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_baseline_train.log"
log "Finished Run 1/3: d32 baseline"

# --------------------------------------------------------------------------
# Run 2: Identity Newton, n_par=8
# Expected: ~13 hours
# --------------------------------------------------------------------------
log "Starting Run 2/3: d32 IDN n_par=8"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8 \
    --model-tag=d32_idn05_npar8 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_idn05_npar8_train.log"
log "Finished Run 2/3: d32 IDN n_par=8"

# --------------------------------------------------------------------------
# Run 3: Diagonal Newton FD, n_par=8
# Expected: ~15.5 hours
# --------------------------------------------------------------------------
log "Starting Run 3/3: d32 diag FD n_par=8"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --diag-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --diag-newton-jvp=fd --n-par-configs=8 \
    --model-tag=d32_diag05_npar8 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag05_npar8_train.log"
log "Finished Run 3/3: d32 diag FD n_par=8"

# --------------------------------------------------------------------------
# Summary: extract final val bpb and training time from each run
# --------------------------------------------------------------------------
log "=== RESULTS SUMMARY ==="
for tag in d32_baseline d32_idn05_npar8 d32_diag05_npar8; do
    logf="$NANOCHAT_BASE_DIR/${tag}_train.log"
    final_bpb=$(grep "Validation bpb" "$logf" | tail -1 | grep -oP '[\d.]+$')
    final_loss=$(grep "^step " "$logf" | tail -1 | grep -oP 'loss: [\d.]+' | grep -oP '[\d.]+')
    train_time=$(grep "Total training time" "$logf" | tail -1 | grep -oP '[\d.]+')
    log "$tag: val_bpb=$final_bpb  loss=$final_loss  time=${train_time}m"
done
log "All runs complete!"
log "Checkpoints at: $NANOCHAT_BASE_DIR/base_checkpoints/d32_{baseline,idn05_npar8,diag05_npar8}/"
