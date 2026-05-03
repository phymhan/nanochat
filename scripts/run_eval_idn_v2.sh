#!/bin/bash
# Run IDN eval v2 with fixed timing (batch_fwd init cost included)
# GPUs 0-3, all 3 models sequentially per GPU
#
# Usage:
#   bash scripts/run_eval_idn_v2.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
mkdir -p cache/eval_logs_2

run_model() {
    local MODEL=$1 GPU=$2 NPAR=$3
    echo "[$(date)] GPU $GPU: $MODEL n_par=$NPAR"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m scripts.eval_jacobi_idn_new \
        --model-tag $MODEL --n-par $NPAR \
        --output cache/eval_logs_2/idn_v2_${MODEL}_npar${NPAR}.json \
        2>&1 | tee cache/eval_logs_2/idn_v2_${MODEL}_npar${NPAR}.log
}

M1=d32s_idn05_npar24_4800
M2=d32s_diag01_npar24_stride3_9600
M3=d32s_baseline_4800

NPARS=(8 12 16 24)

# GPUs 0-3: idn05 → diag01 → baseline, one n_par per GPU
for i in 0 1 2 3; do
    (run_model $M1 $i ${NPARS[$i]} && run_model $M2 $i ${NPARS[$i]} && run_model $M3 $i ${NPARS[$i]}) &
done

echo "Launched 4 GPU jobs (3 models sequential per GPU)"
echo "  GPU 0: n_par=8   (idn05 → diag01 → baseline)"
echo "  GPU 1: n_par=12  (idn05 → diag01 → baseline)"
echo "  GPU 2: n_par=16  (idn05 → diag01 → baseline)"
echo "  GPU 3: n_par=24  (idn05 → diag01 → baseline)"
echo "Monitor: tail -f cache/eval_logs_2/idn_v2_${M1}_npar8.log"
