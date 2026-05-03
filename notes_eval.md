# Eval Notes

## Main eval script

`scripts/eval_jacobi_idn_new.py` — comprehensive IDN layer-parallel inference evaluation.

### Usage

```bash
# Single model + n_par
CUDA_VISIBLE_DEVICES=0 NANOCHAT_BASE_DIR=/workspace/home/ligong/code/nanochat/cache \
  PYTHONUNBUFFERED=1 uv run python -m scripts.eval_jacobi_idn_new \
  --model-tag d32s_idn05_npar24_4800 --n-par 8 \
  --output cache/eval_logs/result.json

# Quick test with fewer tokens (default 1M)
  --max-tokens 10000

# With avg reduction for fused O/MLP (experimental, makes PPL worse)
  --avg
```

### Key args

- `--model-tag`: checkpoint name under `cache/base_checkpoints/`
- `--n-par`: number of parallel layers (8, 12, 16, 24)
- `--max-tokens`: validation tokens (default 1M = 7813 batches). Use 10K for quick tests.
- `--output`: JSON output path

### Chunk configs

The eval script auto-generates chunk configs `{1xFn, nxF1, 2xF(n/2), 4xF(n/4)}` when divisible.
This misses intermediate configs. For a complete sweep, the desired set is:

`set({npar, npar/2, npar/4, 4, 2, 1})` — i.e. chunk counts, each with chunk_size = npar/count.

Example for npar=12: `{12, 6, 3, 4, 2, 1}` → configs `12xF1, 6xF2, 4xF3, 3xF4, 2xF6, 1xF12`.
The auto-generated set only has `{12, 4, 2, 1}`, missing `6xF2` and `3xF4`.

**TODO**: Add `--chunk-counts` CLI arg or fix `chunk_configs` generation to include `npar/2` and `npar/4` chunk counts.

### What it measures

**PPL eval (1M tokens, BS=1):**
- Methods: Sequential, IDN_batched, Fused, Fused+split, Chunk_NxFM, ChunkB_NxFM
- Init strategies: h0, warm, batch_fwd, preheat
- K values: 1, 2, 4, 8 + K=1 with elk_k=0.1

**Timing (single-batch latency):**
- batch_fwd/preheat: init cost included (computed inside forward)
- warm: init cost excluded (simulates AR where prev hidden states are free)
- h0: no init cost

**AR generation:**
- 8 prompts, prefill 8 tokens, generate 16
- IDN_batched: per-layer KV cache (all init strategies)
- Chunkwise: per-chunk KV cache (h0, warm)

### Batch launch scripts

```bash
# Original models (baseline, idn05, diag01)
bash scripts/run_eval_idn_new.sh      # 8 GPUs, results in cache/eval_logs/
bash scripts/run_eval_idn_v2.sh        # 4 GPUs, results in cache/eval_logs_2/

# nox0ve models (no x0_resid, no VE)
bash scripts/run_eval_nox0ve.sh        # 8 GPUs, results in cache/eval_logs_nox0ve/
```

### Compiling results

Results are compiled by inline python scripts that read the JSON files and produce markdown tables.
See `scripts/eval_idn_new_results_all.md` and `scripts/eval_idn_new_results_all_2.md` for examples.

Table format: each cell is `PPL (speedup) / AR match% (AR speedup)`.
Rows = methods, columns = init strategies (h0, warm, preheat, batch_fwd).

### Important notes

- Always use `PYTHONUNBUFFERED=1` when piping through `| tee` to avoid lost output on crash
- Use `@torch.inference_mode()` on all eval/generation functions to prevent memory leaks
- The `load_model` function in `scripts/eval_jacobi_idn.py` handles checkpoint loading with config patching
- For nox0ve models: `model.config.no_x0_resid=True, no_ve=True` — eval code automatically sets lambdas to 1/0 and skips VE

### Checkpoint naming

| Tag | Description |
|-----|-------------|
| `d32s_baseline_4800` | Baseline, 4800 steps |
| `d32s_idn05_npar24_4800` | IDN λ=0.5, n_par=8,16,24 |
| `d32s_diag01_npar24_stride3_9600` | Diag λ=0.1, stride=3 |
| `d32s_nox0ve_baseline` | No x0_resid, no VE baseline |
| `d32s_nox0ve_idn05_npar24` | No x0_resid, no VE, IDN λ=0.5 |
