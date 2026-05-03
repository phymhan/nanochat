# Qwen Single-Token Op Benchmark

This note records a focused microbenchmark of one-layer local operator costs for
`Qwen/Qwen2.5-0.5B-Instruct` in the same setting relevant to layer-parallel
decode:

- one-token decode
- real KV cache
- `cuda:0`
- eager attention
- middle layer (`layer_idx = 12`)
- prompt prefilling done once, then benchmark on the decode token
- cache state reset between trials so every timing sees the same cache contents

## Goal

Measure the local cost of:

- one plain layer `forward`
- finite-difference JVP (`FD`)
- exact forward-mode `JVP`
- reverse-mode `VJP`

The key question was whether exact `JVP` is close to backward cost, as some
papers report, or whether it is significantly slower in this PyTorch + HF Qwen
stack.

## Important Clarification

`autograd_jvp(fn, (x,), (v,))` returns both:

- the primal output `f(x)`
- the tangent output `J v`

So one exact JVP call does **not** need an extra forward pass.

By contrast, finite-difference JVP computes

`(f(x + eps * v) - f(x)) / eps`

and therefore costs two forwards if `f(x)` is not already available. In the
Newton code path, `f(x)` is often already computed for the residual, so the
*extra* FD cost can be just one more forward.

## Benchmark Setup

Model:

- `Qwen/Qwen2.5-0.5B-Instruct`

Runtime:

- `local_files_only=True`
- `attn_implementation="eager"`
- `dtype=torch.bfloat16`
- `CUDA_VISIBLE_DEVICES=0`

Decode regime:

- prompt length after tokenization: `21`
- decode exactly one next token using the real KV cache
- benchmark the target layer only, with `past_key_values=cache` and
  `use_cache=True`

Benchmark details:

- warmup runs: `20`
- timed runs: `120`
- epsilon for FD: `1e-3`

## Results

Measured on one-token decode with KV cache:

| Op | Mean Time | Ratio vs Forward |
|---|---:|---:|
| `forward` | `0.554 ms` | `1.00x` |
| `fd_total` | `1.111 ms` | `2.01x` |
| `jvp` | `3.369 ms` | `6.08x` |
| `vjp` | `1.999 ms` | `3.61x` |

Additional sanity checks from the same run:

- `||f(x)|| = 13.85`
- `||J v|| = 0.978`
- `||J^T v|| = 0.966`

These numbers confirm that:

- FD total cost is essentially exactly `2x` a forward in this setup.
- Exact JVP is **much slower** than both FD and VJP here.
- VJP is also substantially more expensive than one forward, but still much
  cheaper than exact JVP.

## Interpretation

For this HF Qwen eager-attention stack:

- `FD total ~= 2x forward`
- `VJP ~= 3.6x forward`
- `exact JVP ~= 6.1x forward`

So the common heuristic

- "one backward is about `2x` forward"
- "one exact JVP is around backward cost"

does **not** hold here for exact `autograd_jvp`.

This is likely an implementation/runtime effect rather than a mathematical one:

- reverse-mode AD is heavily optimized in PyTorch because training depends on it
- forward-mode AD is much less optimized on this graph
- eager attention expands into many primitive ops, and forward-mode overhead
  appears to accumulate significantly

## SDPA Note

With `attn_implementation="sdpa"`:

- `VJP` still works
- exact `JVP` fails because forward-mode AD support is missing for the fused
  SDPA path

In an earlier smoke test, the error was:

```text
RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
```

This is consistent with the practical rule:

- VJP works on SDPA/Flash because reverse-mode backward kernels exist
- JVP often needs eager attention because forward-mode AD rules are missing for
  fused attention kernels

## Bottom Line

For the current Qwen stack and decode regime we care about:

1. Exact JVP is **not** cheap.
2. It already includes the primal forward, so the `6.08x` number is for a
   single exact JVP call, not "JVP plus extra forward".
3. FD behaves as expected at about `2x` total cost.
4. VJP is an in-between option: more expensive than FD, much cheaper than exact
   JVP, and supported on fused attention backends.
