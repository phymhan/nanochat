#!/bin/bash
# Run IDN evaluation for Qwen 2.5 0.5B on GPUs 4-7
set -e

export HF_HOME=/mnt/nvme0n1/ligong/huggingface
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p cache/eval_logs

NPARS=(8 12 16 20 24)
GPUS=(4 5 6 7)

echo "Starting IDN Qwen evaluation: n_par=${NPARS[*]}"
echo "Using GPUs: ${GPUS[*]}"
echo ""

# Launch first 4 in parallel on GPUs 4-7
for i in 0 1 2 3; do
    npar=${NPARS[$i]}
    gpu=${GPUS[$i]}
    echo "[$(date)] Starting n_par=$npar on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run --extra gpu --group dev \
        python -m scripts.eval_idn_qwen \
        --n-par $npar \
        --output cache/eval_logs/idn_qwen_npar${npar}.json \
        2>&1 | tee cache/eval_logs/idn_qwen_npar${npar}.log &
done

echo "[$(date)] Waiting for first batch (n_par=8,12,16,20)..."
wait

# Run n_par=24 on GPU 4
npar=24
gpu=${GPUS[0]}
echo "[$(date)] Starting n_par=$npar on GPU $gpu"
CUDA_VISIBLE_DEVICES=$gpu uv run --extra gpu --group dev \
    python -m scripts.eval_idn_qwen \
    --n-par $npar \
    --output cache/eval_logs/idn_qwen_npar${npar}.json \
    2>&1 | tee cache/eval_logs/idn_qwen_npar${npar}.log

echo ""
echo "[$(date)] All done!"
