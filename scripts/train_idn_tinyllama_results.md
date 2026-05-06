# TinyLlama 1.1B IDN Fine-Tuning Experiments

Fine-tuning TinyLlama/TinyLlama-1.1B-Chat-v1.0 on ClimbMix-400B data with IDN regularization to test whether IDN training generalizes to standard (non-nanochat) architectures.

Model: L=22, D=2048, H=32, KV=4 (standard LLaMA, no HC/x0-resid/VE)
Data: ClimbMix-400B (10 parquet shards, ~600M tokens)
Optimizer: AdamW (betas=0.9,0.95), cosine LR schedule
Eval: PPL on wikitext-2 (~100K tokens)

## Experiment Grid

| Run | IDN reg | n_par configs | LR | Steps | GPUs | Final PPL | IDN loss (start→end) |
|-----|---------|---------------|-----|-------|------|-----------|---------------------|
| baseline | 0 | — | 2e-5 | 2000 | 4×H100 | 17.82 | — |
| idn05 | 0.5 | 4,8,12 | 2e-5 | 2000 | 4×H100 | 17.91 | 3.0→2.2 |
| idn2_npar48 | 2.0 | 4,8 | 5e-5 | 5000 | 1×H100 | 19.29 | 1.9→1.2 |
| idn5_npar4 | 5.0 | 4 | 1e-4 | 5000 | 1×H100 | 26.03 | 0.8→0.4 |
| idn10_npar4 | 10.0 | 4 | 1e-4 | 5000 | 1×H100 | 26.04 | 0.8→0.4 |
| idn5_npar4812 | 5.0 | 4,8,12 | 5e-5 | 5000 | 1×H100 | 19.53 | 3.0→1.9 |
| idn1_long | 1.0 | 4,8 | 1e-5 | 4000* | 4×H100 | ~17.8 | 1.9→1.5 |
| idn5_long_npar4 | 5.0 | 4 | 1e-5 | 4000* | 4×H100 | ~17.8 | 0.8→0.6 |

\* Crashed (SIGABRT/OOM) at step ~4000 of planned 10000. Checkpoint at step 2000 usable.

## Key Results: IDN Eval on Checkpoints

### Baseline (seq PPL=16.76) vs first IDN run (seq PPL=16.84)

| Config | Baseline K=1 | IDN 0.5 K=1 | Baseline K=8 | IDN 0.5 K=8 |
|--------|:---:|:---:|:---:|:---:|
| n_par=4 h0 | 33.93 (+102%) | 38.73 (+130%) | 16.76 (0%) | 16.84 (0%) |
| n_par=8 h0 | 181.57 (+983%) | 167.76 (+896%) | 16.76 (0%) | 16.84 (0%) |
| n_par=12 h0 | 1232.86 (+7255%) | 394.17 (+2240%) | 17.77 (+6%) | 17.85 (+6%) |
| n_par=16 h0 | 19433 (+115835%) | 1560 (+9166%) | 58.37 (+248%) | 57.65 (+242%) |

**Positive finding:** IDN reg significantly reduces K=1 divergence at larger n_par:
- n_par=12: 1232→394 (3.1× better)
- n_par=16: 19433→1560 (12.5× better)

### Aggressive IDN runs

Best results from each aggressive run at n_par=4 (the IDN target):

| Run | seq PPL | n_par=4 K=1 h0 | n_par=4 K=1 batch_fwd | n_par=4 K=2 batch_fwd |
|-----|---------|:---:|:---:|:---:|
| baseline | 16.76 | 33.93 (+102%) | 26.14 (+56%) | 17.77 (+6%) |
| idn2_npar48 | 18.23 | 40.55 (+123%) | 28.35 (+56%) | 19.07 (+5%) |
| idn5_npar4 | 24.41 | 46.58 (+91%) | 34.20 (+40%) | 25.24 (+3%) |
| idn10_npar4 | 24.52 | 45.06 (+84%) | 33.32 (+36%) | 25.30 (+3%) |
| idn5_npar4812 | 18.46 | 40.25 (+118%) | 28.21 (+53%) | 19.20 (+4%) |
| idn1_long_2k | 16.74 | 37.50 (+124%) | 26.07 (+56%) | 17.74 (+6%) |
| idn5_long_2k | 16.78 | 38.48 (+129%) | 26.38 (+57%) | 17.80 (+6%) |

## Analysis

### What worked
- IDN reg does reduce the relative K=1 gap for the trained n_par configs
- The idn5/idn10 runs with n_par=4 got relative degradation at K=1 down to +84-91% (vs baseline's +102%)
- No run beat sequential PPL

### Why no config beats sequential

1. **Base PPL cost is too high.** Aggressive reg (λ=5-10) pushes IDN loss to ~0.4 but base PPL degrades from 17.5 to 24-26. The absolute K=1 PPL (34-46) is worse than baseline's sequential (16.8).

2. **Fine-tuning can't reshape entrenched Jacobians.** Nanochat trains from scratch with IDN reg, so the model learns input-invariant layers organically. A pretrained model's layer Jacobians (σ_max ≈ 44) are deeply non-contractive. Fine-tuning at lr=1e-5 barely moves them; lr=1e-4 moves them but destroys base quality.

3. **IDN loss plateau.** Even after 5000 steps at λ=10, IDN loss only reaches ~0.4. In nanochat's IDN training (from scratch), this goes well below 0.1 for the target layers. The fine-tuning gradient signal is too weak relative to the pretrained weights' inertia.

4. **Diminishing returns from stronger reg.** idn=5.0 and idn=10.0 give nearly identical results (PPL 24.41 vs 24.52, IDN loss 0.44 vs 0.42). The bottleneck isn't reg strength — it's the learning rate / training duration.

### Comparison to nanochat IDN training

| Aspect | Nanochat (from scratch) | TinyLlama (fine-tune) |
|--------|------------------------|----------------------|
| Training | 4800 steps, Muon lr=0.02 | 2000-5000 steps, AdamW lr=1e-5 to 1e-4 |
| IDN loss (final) | < 0.1 for target layers | 0.4-2.2 |
| Base PPL impact | Better than baseline (74.8 vs 77.7) | Worse than baseline (17.9-26 vs 16.8-17.5) |
| K=1 convergence | 100% top-1 fidelity at n_par=7 | +84-130% PPL degradation at n_par=4 |

### Conclusion

IDN regularization via fine-tuning is insufficient for achieving layer-parallel inference on pretrained models. The method requires training from scratch (or very early in pretraining) to allow the model to learn input-invariant later layers as part of its core architecture, rather than trying to retrofit this property onto an already-trained model.
