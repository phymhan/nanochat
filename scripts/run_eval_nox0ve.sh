#!/bin/bash
# Run IDN eval on nox0ve models (no x0_resid, no VE)
# 2 models × 4 n_par on 8 GPUs
#
# Usage:
#   bash scripts/run_eval_nox0ve.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
mkdir -p cache/eval_logs_nox0ve

run_model() {
    local MODEL=$1 GPU=$2 NPAR=$3
    echo "[$(date)] GPU $GPU: $MODEL n_par=$NPAR"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m scripts.eval_jacobi_idn_new \
        --model-tag $MODEL --n-par $NPAR \
        --output cache/eval_logs_nox0ve/nox0ve_${MODEL}_npar${NPAR}.json \
        2>&1 | tee cache/eval_logs_nox0ve/nox0ve_${MODEL}_npar${NPAR}.log
}

M1=d32s_nox0ve_baseline
M2=d32s_nox0ve_idn05_npar24

NPARS=(8 12 16 24)

# GPUs 0-3: baseline
for i in 0 1 2 3; do
    run_model $M1 $i ${NPARS[$i]} &
done

# GPUs 4-7: idn05
for i in 0 1 2 3; do
    gpu=$((i + 4))
    run_model $M2 $gpu ${NPARS[$i]} &
done

echo "Launched 8 GPU jobs"
echo "  GPUs 0-3: $M1 (n_par=8,12,16,24)"
echo "  GPUs 4-7: $M2 (n_par=8,12,16,24)"
