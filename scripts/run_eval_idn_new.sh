#!/bin/bash
# Run comprehensive IDN eval on 8 GPUs
# GPUs 0-3: d32s_idn05_npar24_4800 then d32s_baseline_4800 (sequential)
# GPUs 4-7: d32s_diag01_npar24_stride3_9600
#
# Usage:
#   bash scripts/run_eval_idn_new.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
mkdir -p cache/eval_logs

run_model() {
    local MODEL=$1 GPU=$2 NPAR=$3
    echo "[$(date)] GPU $GPU: $MODEL n_par=$NPAR"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m scripts.eval_jacobi_idn_new \
        --model-tag $MODEL --n-par $NPAR \
        --output cache/eval_logs/idn_new_${MODEL}_npar${NPAR}.json \
        2>&1 | tee cache/eval_logs/idn_new_${MODEL}_npar${NPAR}.log
}

M1=d32s_idn05_npar24_4800
M2=d32s_baseline_4800
M3=d32s_diag01_npar24_stride3_9600

NPARS=(8 12 16 24)

# GPUs 0-3: idn05_4800 (all 4 n_par), then baseline_4800 (all 4 n_par)
for i in 0 1 2 3; do
    (run_model $M1 $i ${NPARS[$i]} && run_model $M2 $i ${NPARS[$i]}) &
done

# GPUs 4-7: diag01_9600 (all 4 n_par)
for i in 0 1 2 3; do
    gpu=$((i + 4))
    run_model $M3 $gpu ${NPARS[$i]} &
done

echo "Launched 8 GPU jobs"
echo "  GPUs 0-3: $M1 then $M2 (n_par=8,12,16,24)"
echo "  GPUs 4-7: $M3 (n_par=8,12,16,24)"
echo "Monitor: tail -f cache/eval_logs/idn_new_${M1}_npar8.log"
