#!/bin/bash
# Run IDN evaluation for TinyLlama 1.1B on GPUs 0-3
set -e

export HF_HOME=/mnt/nvme0n1/ligong/huggingface
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p cache/eval_logs

NPARS=(8 12 16 20 22)
GPUS=(0 1 2 3)

echo "Starting IDN TinyLlama evaluation: n_par=${NPARS[*]}"
echo "Using GPUs: ${GPUS[*]}"
echo ""

for i in 0 1 2 3; do
    npar=${NPARS[$i]}
    gpu=${GPUS[$i]}
    echo "[$(date)] Starting n_par=$npar on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run --extra gpu --group dev \
        python -m scripts.eval_idn_tinyllama \
        --n-par $npar \
        --output cache/eval_logs/idn_tinyllama_npar${npar}.json \
        2>&1 | tee cache/eval_logs/idn_tinyllama_npar${npar}.log &
done

echo "[$(date)] Waiting for first batch (n_par=8,12,16,20)..."
wait

npar=22
gpu=${GPUS[0]}
echo "[$(date)] Starting n_par=$npar on GPU $gpu"
CUDA_VISIBLE_DEVICES=$gpu uv run --extra gpu --group dev \
    python -m scripts.eval_idn_tinyllama \
    --n-par $npar \
    --output cache/eval_logs/idn_tinyllama_npar${npar}.json \
    2>&1 | tee cache/eval_logs/idn_tinyllama_npar${npar}.log

echo ""
echo "[$(date)] All done!"
