#!/bin/bash
# Train d32s models WITHOUT x0_lambdas and WITHOUT VE (ablation study).
# Purpose: isolate whether improved chunkwise PPL comes from layer fusion or from dropping VE.
#
# GPU 0-3: baseline (no x0, no VE), 4800 steps
# GPU 4-7: IDN 0.5 (no x0, no VE), 4800 steps
# Both run in parallel.
#
# Usage:
#   nohup bash runs/d32s_nox0ve.sh > cache/d32s_nox0ve.log 2>&1 &

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
source .venv/bin/activate

NPROC=4
COMMON_ARGS="--depth=32 --aspect-ratio=20 --device-batch-size=8 --num-iterations=4800 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 --no-x0-resid --no-ve"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Run 1: Baseline (no x0, no VE) on GPUs 0-3
log "Starting: d32s_nox0ve_baseline on GPUs 0-3"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --model-tag=d32s_nox0ve_baseline \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_nox0ve_baseline_train.log" &
PID1=$!

# Run 2: IDN 0.5 (no x0, no VE) on GPUs 4-7
log "Starting: d32s_nox0ve_idn05_npar24 on GPUs 4-7"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    $COMMON_ARGS \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32s_nox0ve_idn05_npar24 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_nox0ve_idn05_npar24_train.log" &
PID2=$!

log "Launched both jobs: baseline(PID=$PID1) idn05(PID=$PID2)"
log "Monitor: tail -f $NANOCHAT_BASE_DIR/d32s_nox0ve_baseline_train.log"

wait $PID1
log "Finished: d32s_nox0ve_baseline"
wait $PID2
log "Finished: d32s_nox0ve_idn05_npar24"
log "All runs complete!"
