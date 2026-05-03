#!/bin/bash
# SJD + IDN layer-parallel codesign evaluation sweep
# GPUs 4-7
#
# Usage:
#   bash scripts/run_sjd_idn_eval.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache
mkdir -p cache/sjd_logs

MODEL=d32s_idn05_npar24_4800
NPAR=12
LA=5
TOKENS=64
PROMPT="The quick brown fox"

run() {
    local GPU=$1 BACKEND=$2 COUPLING=$3 K=$4 CHUNK=$5 LABEL=$6
    echo "[$(date)] GPU $GPU: $LABEL"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m scripts.demo_sjd_idn_nanochat \
        --model-tag $MODEL \
        --mode ar jd sjd sjd++ ppl \
        --forward-backend $BACKEND \
        --coupling $COUPLING \
        --n-par $NPAR --K $K --chunk-size $CHUNK \
        --n-lookahead $LA \
        --max-new-tokens $TOKENS \
        --temperature 0.0 \
        --prompt "$PROMPT" \
        2>&1 | tee cache/sjd_logs/${LABEL}.log
}

# GPU 4: Sequential baseline
(
    run 4 sequential naive 1 1 "seq_baseline"
) &

# GPU 5: ChunkB_12xF1 (chunk_size=1, 12 chunks)
(
    run 5 chunkwise_batched naive 1 1 "chunkB_12xF1_naive_K1" && \
    run 5 chunkwise_batched naive 2 1 "chunkB_12xF1_naive_K2" && \
    run 5 chunkwise_batched h0    1 1 "chunkB_12xF1_h0_K1" && \
    run 5 chunkwise_batched h0    2 1 "chunkB_12xF1_h0_K2" && \
    run 5 chunkwise_batched stale 1 1 "chunkB_12xF1_stale_K1"
) &

# GPU 6: ChunkB_4xF3 (chunk_size=3, 4 chunks)
(
    run 6 chunkwise_batched naive 1 3 "chunkB_4xF3_naive_K1" && \
    run 6 chunkwise_batched naive 2 3 "chunkB_4xF3_naive_K2" && \
    run 6 chunkwise_batched h0    1 3 "chunkB_4xF3_h0_K1" && \
    run 6 chunkwise_batched h0    2 3 "chunkB_4xF3_h0_K2" && \
    run 6 chunkwise_batched stale 1 3 "chunkB_4xF3_stale_K1"
) &

# GPU 7: IDN batched (per-layer, no fusion)
(
    run 7 idn_batched naive 1 1 "idn_batched_naive_K1" && \
    run 7 idn_batched naive 2 1 "idn_batched_naive_K2" && \
    run 7 idn_batched h0    1 1 "idn_batched_h0_K1" && \
    run 7 idn_batched h0    2 1 "idn_batched_h0_K2" && \
    run 7 idn_batched stale 1 1 "idn_batched_stale_K1"
) &

echo "Launched 4 GPU jobs"
echo "  GPU 4: sequential baseline"
echo "  GPU 5: ChunkB_12xF1 (5 configs)"
echo "  GPU 6: ChunkB_4xF3 (5 configs)"
echo "  GPU 7: IDN batched (5 configs)"
echo "Monitor: tail -f cache/sjd_logs/seq_baseline.log"
wait
echo "All done."
