#!/bin/bash
# Run PPL-only IDN eval with avg reduction on 8 GPUs
# 3 models × 4 n_par, interleaved across GPUs
#
# Usage:
#   bash scripts/run_eval_idn_avg.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
mkdir -p cache/eval_logs_avg

run_model() {
    local MODEL=$1 GPU=$2 NPAR=$3
    echo "[$(date)] GPU $GPU: $MODEL n_par=$NPAR (avg)"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m scripts.eval_jacobi_idn_new \
        --model-tag $MODEL --n-par $NPAR --avg \
        --output cache/eval_logs_avg/idn_avg_${MODEL}_npar${NPAR}.json \
        2>&1 | tee cache/eval_logs_avg/idn_avg_${MODEL}_npar${NPAR}.log
}

M1=d32s_idn05_npar24_4800
M2=d32s_baseline_4800
M3=d32s_diag01_npar24_stride3_9600

NPARS=(8 12 16 24)

# GPUs 0-3: idn05 then baseline (sequential per GPU)
for i in 0 1 2 3; do
    (run_model $M1 $i ${NPARS[$i]} && run_model $M2 $i ${NPARS[$i]}) &
done

# GPUs 4-7: diag01
for i in 0 1 2 3; do
    gpu=$((i + 4))
    run_model $M3 $gpu ${NPARS[$i]} &
done

echo "Launched 8 GPU jobs (avg reduction)"
echo "  GPUs 0-3: $M1 then $M2 (n_par=8,12,16,24)"
echo "  GPUs 4-7: $M3 (n_par=8,12,16,24)"
echo "Monitor: tail -f cache/eval_logs_avg/idn_avg_${M1}_npar8.log"
