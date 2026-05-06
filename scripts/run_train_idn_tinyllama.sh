#!/bin/bash
# Train TinyLlama 1.1B: baseline + IDN reg on ClimbMix data
# Baseline on GPUs 0-3, IDN on GPUs 4-7
set -e

export HF_HOME=/mnt/nvme0n1/ligong/huggingface
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

COMMON="--num-steps=2000 --batch-size=4 --seq-len=1024 --n-shards=10 \
        --eval-every=200 --save-every=500 --log-every=10 \
        --lr=2e-5 --warmup-steps=100"

echo "[$(date)] Starting TinyLlama training runs"

# Baseline (no reg) on GPUs 0-3
echo "[$(date)] Starting baseline on GPUs 0-3"
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra gpu --group dev \
    torchrun --standalone --nproc_per_node=4 -m scripts.train_idn_tinyllama \
    $COMMON --model-tag=tinyllama_baseline \
    2>&1 | tee cache/tinyllama_baseline_train.log &

# IDN reg on GPUs 4-7
echo "[$(date)] Starting IDN reg=0.5 on GPUs 4-7"
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --extra gpu --group dev \
    torchrun --standalone --nproc_per_node=4 -m scripts.train_idn_tinyllama \
    $COMMON \
    --identity-newton-reg=0.5 --reg-warmup=0.2 --n-par-configs=4,8,12 \
    --model-tag=tinyllama_idn05 \
    2>&1 | tee cache/tinyllama_idn05_train.log &

wait
echo "[$(date)] All training done!"
