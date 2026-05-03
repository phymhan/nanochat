#!/bin/bash
# Resume experiments: test diag Newton reg from trained checkpoints.
# 4 runs × 1920 steps each, ~30 min per run = ~2 hours total.
#
# Usage:
#   nohup bash runs/diag_resume_test.sh > cache/diag_resume_test.log 2>&1 &

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

NPROC=8
# Resume from step 9600, run 1920 more steps (= 9600 + 1920 = 11520 total)
COMMON_ARGS="--depth=32 --device-batch-size=8 --fp8 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 --num-iterations=11520 --resume-from-step=9600 --diag-newton-jvp=fd --n-par-configs=8 --jacobi-reg-warmup=0.0 --run=dummy"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --- λ=0.01 from baseline ---
log "Run 1/4: diag λ=0.01 from baseline"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS --diag-newton-reg=0.01 \
    --model-tag=d32_diag0.01_from_baseline \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag0.01_from_baseline_train.log"
log "Done 1/4"

# --- λ=0.01 from IDN ---
log "Run 2/4: diag λ=0.01 from IDN"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS --diag-newton-reg=0.01 \
    --model-tag=d32_diag0.01_from_idn05_npar8 \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag0.01_from_idn05_npar8_train.log"
log "Done 2/4"

# --- λ=0.001 from baseline ---
log "Run 3/4: diag λ=0.001 from baseline"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS --diag-newton-reg=0.001 \
    --model-tag=d32_diag0.001_from_baseline \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag0.001_from_baseline_train.log"
log "Done 3/4"

# --- λ=0.001 from IDN ---
log "Run 4/4: diag λ=0.001 from IDN"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS --diag-newton-reg=0.001 \
    --model-tag=d32_diag0.001_from_idn05_npar8 \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_diag0.001_from_idn05_npar8_train.log"
log "Done 4/4"

# --- Summary ---
log "=== RESULTS ==="
for tag in d32_diag0.01_from_baseline d32_diag0.01_from_idn05_npar8 d32_diag0.001_from_baseline d32_diag0.001_from_idn05_npar8; do
    logf="$NANOCHAT_BASE_DIR/${tag}_train.log"
    final_bpb=$(grep "Validation bpb" "$logf" | tail -1 | grep -oP '[\d.]+$')
    final_loss=$(grep "^step " "$logf" | tail -1 | grep -oP 'loss: [\d.]+' | grep -oP '[\d.]+')
    log "$tag: val_bpb=$final_bpb  loss=$final_loss"
done
log "All done!"
