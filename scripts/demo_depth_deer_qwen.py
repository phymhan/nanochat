"""
For optimized version and more jvp methods, see demo_depth_quasi_deer_qwen.py

Standalone Qwen DEER demo with cache-aware diagonal decode.

This is a lightly simplified copy of `reference/demo_depth_deer.py`:
  1. `--hidden-init {warm_start,batched_fwd}` replaces `--no-warm-start/--cold-init`
  2. `--init-noise-std` perturbs whichever hidden-state initializer is selected
  3. diagonal mode uses the real KV cache during the solve when `use_cache=True`
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch.autograd.functional import jvp as autograd_jvp
from torch.func import functional_call, jvp as func_jvp, stack_module_state, vmap
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def associative_scan_diagonal(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel prefix scan for the diagonal linear recursion:
        h_l = a_l * h_{l-1} + b_l,   l = 0, ..., L-1

    Returns tensors `a_scan`, `b_scan` such that:
        h_l = a_scan[l] * h_0 + b_scan[l]
    """
    num_layers = a.shape[0]
    if num_layers <= 1:
        return a, b

    a_scan = a.clone()
    b_scan = b.clone()

    num_steps = int(math.ceil(math.log2(num_layers)))
    for depth in range(num_steps):
        stride = 1 << depth
        idx = torch.arange(stride, num_layers, device=a.device)
        src = idx - stride
        new_a = a_scan[idx] * a_scan[src]
        new_b = a_scan[idx] * b_scan[src] + b_scan[idx]
        a_scan = a_scan.clone()
        b_scan = b_scan.clone()
        a_scan[idx] = new_a
        b_scan[idx] = new_b

    return a_scan, b_scan


def associative_scan_matrix(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel prefix scan for the full linear recursion:
        h_l = A_l @ h_{l-1} + b_l,   l = 0, ..., L-1

    Returns tensors `a_scan`, `b_scan` such that:
        h_l = A_scan[l] @ h_0 + b_scan[l]
    """
    num_layers = a.shape[0]
    if num_layers <= 1:
        return a, b

    a_scan = a.clone()
    b_scan = b.clone()

    num_steps = int(math.ceil(math.log2(num_layers)))
    for depth in range(num_steps):
        stride = 1 << depth
        idx = torch.arange(stride, num_layers, device=a.device)
        src = idx - stride
        new_a = torch.matmul(a_scan[idx], a_scan[src])
        new_b = torch.matmul(a_scan[idx], b_scan[src].unsqueeze(-1)).squeeze(-1) + b_scan[idx]
        a_scan = a_scan.clone()
        b_scan = b_scan.clone()
        a_scan[idx] = new_a
        b_scan[idx] = new_b

    return a_scan, b_scan


def crop_cache_layers(cache: DynamicCache, layer_ids: List[int], max_length: int) -> None:
    """Crop only the specified cache layers back to `max_length`."""
    for layer_id in layer_ids:
        if layer_id < len(cache.layers):
            cache.layers[layer_id].crop(max_length)


def _batched_layer_fwd(
    all_x: torch.Tensor,
    stacked_params: dict,
    cached_k: torch.Tensor,
    cached_v: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attn_mask: Optional[torch.Tensor],
    config,
) -> torch.Tensor:
    """
    Full batched forward for L decoder layers simultaneously.

    Reimplements the Qwen2 decoder layer using batched tensor ops so that all
    L parallel layers execute in a single set of matmuls.  Cached K,V are passed
    as plain tensors (not a DynamicCache) so forward-mode AD works.

    Args:
        all_x:      (L, B, T, D)   hidden-state inputs for each layer
        stacked_params: dict of (L, ...) tensors from stack_module_state
        cached_k:   (L, B, H_kv, S, d_h)  prefill keys   (empty → zeros with S=0)
        cached_v:   (L, B, H_kv, S, d_h)  prefill values
        position_embeddings: (cos, sin) each (B, T, d_h)
        attn_mask:  causal mask or None
        config:     model config

    Returns:
        all_out: (L, B, T, D)
    """
    L, B, T, D = all_x.shape
    eps = config.rms_norm_eps
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = D // num_heads
    num_kv_groups = num_heads // num_kv_heads
    scaling = head_dim ** -0.5

    # ---------- input layernorm (RMSNorm) ----------
    ln_w = stacked_params['input_layernorm.weight']           # (L, D)
    x_f = all_x.float()
    x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    x_norm = (x_norm.to(all_x.dtype) * ln_w[:, None, None, :])  # (L, B, T, D)

    # ---------- QKV projections ----------
    q_w = stacked_params['self_attn.q_proj.weight']           # (L, num_heads*d_h, D)
    k_w = stacked_params['self_attn.k_proj.weight']           # (L, num_kv_heads*d_h, D)
    v_w = stacked_params['self_attn.v_proj.weight']           # (L, num_kv_heads*d_h, D)
    all_q = torch.einsum('lbtd,lhd->lbth', x_norm, q_w)      # (L, B, T, num_heads*d_h)
    all_k = torch.einsum('lbtd,lhd->lbth', x_norm, k_w)      # (L, B, T, num_kv_heads*d_h)
    all_v = torch.einsum('lbtd,lhd->lbth', x_norm, v_w)      # (L, B, T, num_kv_heads*d_h)
    if 'self_attn.q_proj.bias' in stacked_params:
        all_q = all_q + stacked_params['self_attn.q_proj.bias'][:, None, None, :]
    if 'self_attn.k_proj.bias' in stacked_params:
        all_k = all_k + stacked_params['self_attn.k_proj.bias'][:, None, None, :]
    if 'self_attn.v_proj.bias' in stacked_params:
        all_v = all_v + stacked_params['self_attn.v_proj.bias'][:, None, None, :]

    # reshape to (L, B, num_heads, T, d_h)
    all_q = all_q.view(L, B, T, num_heads, head_dim).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(L, B, T, num_kv_heads, head_dim).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(L, B, T, num_kv_heads, head_dim).permute(0, 1, 3, 2, 4)

    # ---------- RoPE ----------
    cos, sin = position_embeddings                             # each (B, T, d_h)
    cos = cos.unsqueeze(0).unsqueeze(2)                        # (1, B, 1, T, d_h)
    sin = sin.unsqueeze(0).unsqueeze(2)

    def _rope(x):
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    all_q = _rope(all_q)
    all_k = _rope(all_k)

    # ---------- concat cached KV ----------
    # cached_k: (L, B, H_kv, S, d_h),  all_k: (L, B, H_kv, T, d_h)
    if cached_k is not None and cached_k.shape[-2] > 0:
        all_k = torch.cat([cached_k, all_k], dim=-2)          # (L, B, H_kv, S+T, d_h)
        all_v = torch.cat([cached_v, all_v], dim=-2)

    # ---------- GQA expand ----------
    if num_kv_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(
            L, B, num_kv_heads, num_kv_groups, -1, head_dim
        ).reshape(L, B, num_heads, -1, head_dim)
        all_v = all_v[:, :, :, None, :, :].expand(
            L, B, num_kv_heads, num_kv_groups, -1, head_dim
        ).reshape(L, B, num_heads, -1, head_dim)

    # ---------- attention ----------
    attn_w = torch.matmul(all_q, all_k.transpose(-2, -1)) * scaling  # (L,B,H,T,S+T)
    if attn_mask is not None:
        attn_w = attn_w + attn_mask[:, :, :, :all_k.shape[-2]]
    attn_w = torch.nn.functional.softmax(attn_w, dim=-1, dtype=torch.float32).to(all_x.dtype)
    attn_out = torch.matmul(attn_w, all_v)                    # (L, B, H, T, d_h)

    # ---------- output projection ----------
    attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(L, B, T, D)  # (L, B, T, D)
    o_w = stacked_params['self_attn.o_proj.weight']            # (L, D, num_heads*d_h)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, o_w)  # (L, B, T, D)

    # ---------- first residual ----------
    hidden = all_x + attn_out

    # ---------- post-attention layernorm ----------
    ln2_w = stacked_params['post_attention_layernorm.weight']  # (L, D)
    h_f = hidden.float()
    h_norm = h_f * torch.rsqrt(h_f.pow(2).mean(-1, keepdim=True) + eps)
    h_norm = h_norm.to(hidden.dtype) * ln2_w[:, None, None, :]

    # ---------- MLP: gate_proj, up_proj, silu, down_proj ----------
    gate_w = stacked_params['mlp.gate_proj.weight']            # (L, I, D)
    up_w   = stacked_params['mlp.up_proj.weight']              # (L, I, D)
    down_w = stacked_params['mlp.down_proj.weight']            # (L, D, I)
    gate = torch.einsum('lbtd,lid->lbti', h_norm, gate_w)
    up   = torch.einsum('lbtd,lid->lbti', h_norm, up_w)
    mlp_out = torch.einsum('lbti,ldi->lbtd', torch.nn.functional.silu(gate) * up, down_w)

    # ---------- second residual ----------
    return hidden + mlp_out


@dataclass
class ForwardConfig:
    layer_start: int = 0
    layer_end: Optional[int] = None
    run_layers: Optional[List[int]] = None
    skip_layers: Optional[List[int]] = None
    layer_forward: str = "recurrent"
    parallel_iters: int = 4
    parallel_damping: float = 1.0
    hidden_init: str = "warm_start"
    init_noise_std: float = 0.0
    cache_update: str = "kv_only"
    jacobian: str = "diagonal"


class Qwen2ExplicitLoop(torch.nn.Module):
    """
    Wrapper around a HF Qwen2(.5) causal LM that exposes a custom forward with
    an explicit layer loop, supporting both sequential and DEER-based forward.
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base = base_model

        if not hasattr(base_model, "model") or not hasattr(base_model, "lm_head"):
            raise TypeError("Expected a Qwen2ForCausalLM-like model with `.model` and `.lm_head`")

        self.model = base_model.model
        self.lm_head = base_model.lm_head
        self.config = base_model.config

        if hasattr(self.config, "_attn_implementation"):
            self.config._attn_implementation = "eager"

        self.embed_tokens = self.model.embed_tokens
        self.layers = self.model.layers
        self.norm = self.model.norm
        self.rotary_emb = self.model.rotary_emb
        self.has_sliding_layers = getattr(self.model, "has_sliding_layers", False)

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    @staticmethod
    def _match_prev_state(prev_state: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if prev_state.shape == target.shape:
            return prev_state.detach()

        if (
            prev_state.ndim == target.ndim == 3
            and prev_state.shape[0] == target.shape[0]
            and prev_state.shape[-1] == target.shape[-1]
            and prev_state.shape[1] >= target.shape[1]
        ):
            return prev_state[:, -target.shape[1] :, :].detach()

        return None

    def _warm_start_states(
        self,
        prev_hs: Optional[List[torch.Tensor]],
        h0: torch.Tensor,
        num_layers: int,
    ) -> Optional[List[torch.Tensor]]:
        if prev_hs is None or len(prev_hs) != num_layers + 1:
            return None

        matched = []
        for idx in range(1, num_layers + 1):
            state = self._match_prev_state(prev_hs[idx], h0)
            if state is None:
                return None
            matched.append(state)
        return matched

    @staticmethod
    def _add_init_noise(hidden_states: torch.Tensor, init_noise_std: float) -> torch.Tensor:
        if init_noise_std <= 0.0:
            return hidden_states
        scale = hidden_states.float().std(dim=-1, keepdim=True).to(hidden_states.dtype).clamp_min(1e-6)
        return hidden_states + init_noise_std * scale * torch.randn_like(hidden_states)

    @staticmethod
    def _materialize_layer_jacobian(
        fn,
        x_base: torch.Tensor,
        *,
        layer_ids: List[int],
        saved_cache_len: Optional[int],
        past_key_values,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Materialize the full Jacobian by applying JVP to basis vectors.

        This is a debugging path intended for small hidden-state sizes. It matches
        the same linearization used by `jacobian="full"` but makes the matrix
        explicit so we can test the scan implementation separately.
        """
        x_base = x_base.detach().requires_grad_(True)
        flat_dim = x_base.numel()
        basis = torch.zeros(flat_dim, device=x_base.device, dtype=x_base.dtype)
        jac_columns = []
        fx0 = None

        for basis_idx in range(flat_dim):
            basis.zero_()
            basis[basis_idx] = 1
            tangent = basis.view_as(x_base)
            fx_col, j_col = autograd_jvp(
                fn,
                (x_base,),
                (tangent,),
                create_graph=False,
                strict=False,
            )
            if fx0 is None:
                fx0 = fx_col.detach()
            jac_columns.append(j_col.reshape(-1).detach())
            if saved_cache_len is not None and past_key_values is not None:
                crop_cache_layers(past_key_values, layer_ids, saved_cache_len)

        if fx0 is None:
            raise RuntimeError("Failed to materialize Jacobian")

        jacobian = torch.stack(jac_columns, dim=1)
        return fx0, jacobian

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        *,
        use_cache: bool = False,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        fwd_cfg: Optional[ForwardConfig] = None,
    ) -> torch.Tensor:
        if fwd_cfg is None:
            fwd_cfg = ForwardConfig()

        if attention_mask is not None and attention_mask.dtype not in (torch.long, torch.bool):
            attention_mask = attention_mask.to(torch.long)

        inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        num_layers = int(self.config.num_hidden_layers)
        skip = set(fwd_cfg.skip_layers or [])
        save_for_deer = getattr(self, "_deer_save_prefill_hs", False)
        cache_base_len = past_key_values.get_seq_length() if use_cache and past_key_values is not None else None

        if fwd_cfg.run_layers is not None:
            layer_indices: Iterable[int] = fwd_cfg.run_layers
            if fwd_cfg.layer_forward == "deer":
                hidden_states = self.deer_layer_forward(
                    hidden_states=hidden_states,
                    layer_indices=layer_indices,
                    skip=skip,
                    causal_mask_mapping=causal_mask_mapping,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    cache_base_len=cache_base_len,
                    num_iters=int(max(1, fwd_cfg.parallel_iters)),
                    damping=float(fwd_cfg.parallel_damping),
                    hidden_init=fwd_cfg.hidden_init,
                    init_noise_std=float(fwd_cfg.init_noise_std),
                    cache_update=fwd_cfg.cache_update,
                    jacobian=fwd_cfg.jacobian,
                )
            else:
                hidden_states = self.recurrent_layer_forward(
                    hidden_states=hidden_states,
                    layer_indices=layer_indices,
                    skip=skip,
                    causal_mask_mapping=causal_mask_mapping,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    save_intermediates=save_for_deer,
                )
        else:
            layer_start = int(max(0, fwd_cfg.layer_start))
            layer_end = int(num_layers if fwd_cfg.layer_end is None else fwd_cfg.layer_end)
            layer_end = min(layer_end, num_layers)

            if layer_start > 0:
                hidden_states = self.recurrent_layer_forward(
                    hidden_states=hidden_states,
                    layer_indices=range(0, layer_start),
                    skip=skip,
                    causal_mask_mapping=causal_mask_mapping,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    save_intermediates=False,
                )

            if layer_start < layer_end:
                block_indices = range(layer_start, layer_end)
                if fwd_cfg.layer_forward == "deer":
                    hidden_states = self.deer_layer_forward(
                        hidden_states=hidden_states,
                        layer_indices=block_indices,
                        skip=skip,
                        causal_mask_mapping=causal_mask_mapping,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        cache_base_len=cache_base_len,
                        num_iters=int(max(1, fwd_cfg.parallel_iters)),
                        damping=float(fwd_cfg.parallel_damping),
                        hidden_init=fwd_cfg.hidden_init,
                        init_noise_std=float(fwd_cfg.init_noise_std),
                        cache_update=fwd_cfg.cache_update,
                        jacobian=fwd_cfg.jacobian,
                    )
                else:
                    hidden_states = self.recurrent_layer_forward(
                        hidden_states=hidden_states,
                        layer_indices=block_indices,
                        skip=skip,
                        causal_mask_mapping=causal_mask_mapping,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        save_intermediates=save_for_deer,
                    )

            if layer_end < num_layers:
                hidden_states = self.recurrent_layer_forward(
                    hidden_states=hidden_states,
                    layer_indices=range(layer_end, num_layers),
                    skip=skip,
                    causal_mask_mapping=causal_mask_mapping,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    save_intermediates=False,
                )

        hidden_states = self.norm(hidden_states)

        if logits_to_keep and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def recurrent_layer_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        layer_indices: Iterable[int],
        skip: set[int],
        causal_mask_mapping: dict,
        position_ids: torch.LongTensor,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        save_intermediates: bool = False,
    ) -> torch.Tensor:
        intermediates = [hidden_states] if save_intermediates else None
        for idx in layer_indices:
            if int(idx) in skip:
                continue
            decoder_layer = self.layers[int(idx)]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if save_intermediates:
                intermediates.append(hidden_states)
        if save_intermediates:
            self._deer_prev_hs = intermediates
        return hidden_states

    def deer_layer_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        layer_indices: Iterable[int],
        skip: set[int],
        causal_mask_mapping: dict,
        position_ids: torch.LongTensor,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_base_len: Optional[int] = None,
        num_iters: int = 4,
        damping: float = 1.0,
        hidden_init: str = "warm_start",
        init_noise_std: float = 0.0,
        cache_update: str = "kv_only",
        jacobian: str = "diagonal",
    ) -> torch.Tensor:
        layer_ids = [int(idx) for idx in layer_indices if int(idx) not in skip]
        if not layer_ids:
            return hidden_states

        damping = float(damping)
        if not (0.0 < damping <= 1.0):
            raise ValueError(f"damping must be in (0, 1], got {damping}")

        num_layers = len(layer_ids)
        h0 = hidden_states

        selected_layers = [self.layers[layer_id] for layer_id in layer_ids]
        base_layer = selected_layers[0]
        stacked_params, stacked_buffers = stack_module_state(selected_layers)
        attn_mask = causal_mask_mapping[base_layer.attention_type]

        def single_layer_fwd(params, buffers, x):
            out = functional_call(
                base_layer,
                (params, buffers),
                args=(x,),
                kwargs=dict(
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                ),
            )
            return out[0] if isinstance(out, tuple) else out

        def jvp_single(params, buffers, x, z):
            primals_out, tangents_out = func_jvp(
                lambda x_: single_layer_fwd(params, buffers, x_),
                (x,),
                (z,),
            )
            return primals_out, tangents_out

        def layer_fn(layer_id: int):
            decoder_layer = self.layers[layer_id]

            def fn(x: torch.Tensor) -> torch.Tensor:
                out = decoder_layer(
                    x,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                return out[0] if isinstance(out, tuple) else out

            return fn

        batched_jvp = vmap(jvp_single)
        batched_fwd = vmap(single_layer_fwd)

        prev_hs = getattr(self, "_deer_prev_hs", None)
        if hidden_init not in {"warm_start", "batched_fwd"}:
            raise ValueError(f"hidden_init must be 'warm_start' or 'batched_fwd', got {hidden_init!r}")

        warm_states = self._warm_start_states(prev_hs, h0, num_layers) if hidden_init == "warm_start" else None
        if warm_states is not None:
            hs = [h0] + [self._add_init_noise(state, init_noise_std) for state in warm_states]
        else:
            init_source = self._add_init_noise(h0, init_noise_std)
            all_h0 = init_source.unsqueeze(0).expand(num_layers, -1, -1, -1)
            init_outputs = batched_fwd(stacked_params, stacked_buffers, all_h0)
            hs = [h0] + [init_outputs[layer_idx] for layer_idx in range(num_layers)]

        saved_cache_len = None
        if use_cache and past_key_values is not None:
            saved_cache_len = cache_base_len if cache_base_len is not None else past_key_values.get_seq_length()

        for _ in range(num_iters):
            if jacobian == "full":
                with torch.enable_grad():
                    hs_new = [h0]
                    for layer_offset, layer_id in enumerate(layer_ids):
                        x_base = hs[layer_offset].detach().requires_grad_(True)
                        x_target = hs_new[layer_offset].detach()
                        fx0, j_delta = autograd_jvp(
                            layer_fn(layer_id),
                            (x_base,),
                            (x_target - x_base,),
                            create_graph=False,
                            strict=False,
                        )
                        hs_new.append(fx0.detach() + j_delta.detach())
                        if saved_cache_len is not None and past_key_values is not None:
                            crop_cache_layers(past_key_values, layer_ids, saved_cache_len)
            elif jacobian == "full_scan":
                flat_dim = h0.numel()
                h0_vec = h0.reshape(-1)
                affine_mats = []
                affine_biases = []

                with torch.enable_grad():
                    for layer_offset, layer_id in enumerate(layer_ids):
                        x_base = hs[layer_offset].detach()
                        fx0, jac_mat = self._materialize_layer_jacobian(
                            layer_fn(layer_id),
                            x_base,
                            layer_ids=layer_ids,
                            saved_cache_len=saved_cache_len,
                            past_key_values=past_key_values,
                        )
                        x_base_vec = x_base.reshape(-1)
                        bias_vec = fx0.reshape(-1) - torch.matmul(jac_mat, x_base_vec)
                        affine_mats.append(jac_mat)
                        affine_biases.append(bias_vec)

                a_full = torch.cat(
                    [
                        torch.eye(flat_dim, device=h0.device, dtype=h0.dtype).unsqueeze(0),
                        torch.stack(affine_mats, dim=0),
                    ],
                    dim=0,
                )
                b_full = torch.cat(
                    [
                        torch.zeros(1, flat_dim, device=h0.device, dtype=h0.dtype),
                        torch.stack(affine_biases, dim=0),
                    ],
                    dim=0,
                )
                a_scan, b_scan = associative_scan_matrix(a_full, b_full)

                hs_new = [h0]
                for layer_idx in range(1, num_layers + 1):
                    h_vec = torch.matmul(a_scan[layer_idx], h0_vec) + b_scan[layer_idx]
                    hs_new.append(h_vec.view_as(h0))
            elif jacobian == "diagonal_batched_cached":
                # Cache-aware batched diagonal: re-implements the full layer
                # forward with plain tensor ops so JVP works via func_jvp
                # without needing the mutable DynamicCache.
                all_x = torch.stack([hs[layer_idx].detach() for layer_idx in range(num_layers)])

                # Pre-extract cached K,V as stacked tensors
                if use_cache and past_key_values is not None:
                    cached_k = torch.stack([past_key_values[lid][0] for lid in layer_ids])
                    cached_v = torch.stack([past_key_values[lid][1] for lid in layer_ids])
                else:
                    num_kv_heads = self.config.num_key_value_heads
                    kv_head_dim = self.config.hidden_size // self.config.num_attention_heads
                    cached_k = all_x.new_zeros(num_layers, all_x.shape[1], num_kv_heads, 0, kv_head_dim)
                    cached_v = cached_k

                all_z = torch.randint(0, 2, all_x.shape, device=all_x.device, dtype=all_x.dtype) * 2 - 1

                def _batched_fwd_for_jvp(x):
                    return _batched_layer_fwd(
                        x, stacked_params, cached_k, cached_v,
                        position_embeddings, attn_mask, self.config,
                    )

                all_fx0, all_jz = func_jvp(_batched_fwd_for_jvp, (all_x,), (all_z,))

                diag_j = all_z * all_jz
                bias = all_fx0 - diag_j * all_x

                a_full = torch.cat([torch.ones_like(diag_j[:1]), diag_j], dim=0)
                b_full = torch.cat([torch.zeros_like(bias[:1]), bias], dim=0)
                a_scan, b_scan = associative_scan_diagonal(a_full, b_full)

                hs_new = [h0]
                for layer_idx in range(1, num_layers + 1):
                    hs_new.append(a_scan[layer_idx] * h0 + b_scan[layer_idx])
            elif jacobian in {"diagonal", "diagonal_batched"}:
                all_x = torch.stack([hs[layer_idx].detach() for layer_idx in range(num_layers)])

                if jacobian == "diagonal_batched":
                    # Batched path: compute Hutchinson JVPs via vmap across layers.
                    # WARNING: only correct with --no-cache.  With KV cache enabled the
                    # vmap'd functional_call passes past_key_values=None (vmap cannot
                    # handle the mutable DynamicCache object), so each layer's attention
                    # loses the prefill context and computes a different function than
                    # the sequential path.  Use `diagonal` for cache-aware DEER.
                    all_z = torch.randint(0, 2, all_x.shape, device=all_x.device, dtype=all_x.dtype) * 2 - 1
                    all_fx0, all_jz = batched_jvp(stacked_params, stacked_buffers, all_x, all_z)
                elif use_cache and past_key_values is not None:
                    # Cache-aware diagonal path: each JVP must see the real prefix cache,
                    # so we run per-layer JVPs and crop the mutated cache after each call.
                    all_z = []
                    all_fx0 = []
                    all_jz = []
                    with torch.enable_grad():
                        for layer_offset, layer_id in enumerate(layer_ids):
                            x_base = all_x[layer_offset].detach().requires_grad_(True)
                            z = torch.randint(0, 2, x_base.shape, device=x_base.device, dtype=x_base.dtype) * 2 - 1
                            fx0, jz = autograd_jvp(
                                layer_fn(layer_id),
                                (x_base,),
                                (z,),
                                create_graph=False,
                                strict=False,
                            )
                            all_z.append(z.detach())
                            all_fx0.append(fx0.detach())
                            all_jz.append(jz.detach())
                            crop_cache_layers(past_key_values, layer_ids, saved_cache_len)
                    all_z = torch.stack(all_z)
                    all_fx0 = torch.stack(all_fx0)
                    all_jz = torch.stack(all_jz)
                else:
                    all_z = torch.randint(0, 2, all_x.shape, device=all_x.device, dtype=all_x.dtype) * 2 - 1
                    all_fx0, all_jz = batched_jvp(stacked_params, stacked_buffers, all_x, all_z)

                diag_j = all_z * all_jz
                bias = all_fx0 - diag_j * all_x

                a_full = torch.cat([torch.ones_like(diag_j[:1]), diag_j], dim=0)
                b_full = torch.cat([torch.zeros_like(bias[:1]), bias], dim=0)
                a_scan, b_scan = associative_scan_diagonal(a_full, b_full)

                hs_new = [h0]
                for layer_idx in range(1, num_layers + 1):
                    hs_new.append(a_scan[layer_idx] * h0 + b_scan[layer_idx])
            else:
                raise ValueError(f"Unsupported jacobian mode: {jacobian}")

            for layer_idx in range(1, num_layers + 1):
                hs[layer_idx] = (1.0 - damping) * hs[layer_idx] + damping * hs_new[layer_idx]

        if use_cache and past_key_values is not None:
            crop_cache_layers(past_key_values, layer_ids, saved_cache_len)

            if cache_update == "full_fwd":
                for layer_offset, layer_id in enumerate(layer_ids):
                    decoder_layer = self.layers[layer_id]
                    out = decoder_layer(
                        hs[layer_offset].detach(),
                        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    hs[layer_offset + 1] = out[0] if isinstance(out, tuple) else out
            else:
                all_layer_inputs = torch.stack([hs[layer_idx].detach() for layer_idx in range(num_layers)])
                _batched_cache_update(
                    past_key_values,
                    stacked_params,
                    all_layer_inputs,
                    layer_ids,
                    position_embeddings,
                    self.config,
                    cache_position,
                )

        self._deer_prev_hs = [h.detach() for h in hs]
        return hs[-1]


def _batched_cache_update(
    cache: DynamicCache,
    stacked_params: dict,
    all_layer_inputs: torch.Tensor,
    layer_ids: List[int],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    config,
    cache_position: Optional[torch.LongTensor] = None,
) -> None:
    """
    Compute K, V for all layers in batch and insert them into the cache.

    Only does RMS norm -> K/V projections -> reshape -> rotary -> cache.update.
    """
    num_layers, batch_size, seq_len, _ = all_layer_inputs.shape
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    eps = config.rms_norm_eps

    ln_weight = stacked_params["input_layernorm.weight"]
    x = all_layer_inputs.float()
    x_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x_norm.to(all_layer_inputs.dtype) * ln_weight[:, None, None, :]

    k_weight = stacked_params["self_attn.k_proj.weight"]
    v_weight = stacked_params["self_attn.v_proj.weight"]
    all_k = torch.einsum("lbtd,lkd->lbtk", x_norm, k_weight)
    all_v = torch.einsum("lbtd,lkd->lbtk", x_norm, v_weight)
    if "self_attn.k_proj.bias" in stacked_params:
        all_k = all_k + stacked_params["self_attn.k_proj.bias"][:, None, None, :]
    if "self_attn.v_proj.bias" in stacked_params:
        all_v = all_v + stacked_params["self_attn.v_proj.bias"][:, None, None, :]

    all_k = all_k.view(num_layers, batch_size, seq_len, num_kv_heads, head_dim).transpose(2, 3)
    all_v = all_v.view(num_layers, batch_size, seq_len, num_kv_heads, head_dim).transpose(2, 3)

    cos, sin = position_embeddings
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    k1 = all_k[..., : head_dim // 2]
    k2 = all_k[..., head_dim // 2 :]
    all_k = all_k * cos + torch.cat((-k2, k1), dim=-1) * sin

    cache_kwargs = {"cache_position": cache_position} if cache_position is not None else None
    for layer_idx in range(num_layers):
        cache.update(all_k[layer_idx], all_v[layer_idx], layer_ids[layer_idx], cache_kwargs)


@dataclass
class GenerationStats:
    output_ids: torch.LongTensor
    num_prompt_tokens: int
    num_generated_tokens: int
    prefill_time: float
    decode_time: float

    @property
    def total_time(self) -> float:
        return self.prefill_time + self.decode_time

    @property
    def prefill_tok_per_sec(self) -> float:
        return self.num_prompt_tokens / self.prefill_time if self.prefill_time > 0 else float("inf")

    @property
    def decode_tok_per_sec(self) -> float:
        return self.num_generated_tokens / self.decode_time if self.decode_time > 0 else float("inf")

    def print_stats(self) -> None:
        print(f"  Prompt tokens:    {self.num_prompt_tokens}")
        print(f"  Generated tokens: {self.num_generated_tokens}")
        print(f"  Prefill:          {self.prefill_time:.3f}s  ({self.prefill_tok_per_sec:.1f} tok/s)")
        print(f"  Decode:           {self.decode_time:.3f}s  ({self.decode_tok_per_sec:.1f} tok/s)")
        print(f"  Total:            {self.total_time:.3f}s")


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate_greedy(
    model: Qwen2ExplicitLoop,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    fwd_cfg: Optional[ForwardConfig] = None,
    use_cache: bool = True,
) -> GenerationStats:
    if fwd_cfg is None:
        fwd_cfg = ForwardConfig()

    num_prompt_tokens = input_ids.shape[1]

    if use_cache:
        cache = DynamicCache(config=model.config)
        prefill_cfg = ForwardConfig(
            layer_start=fwd_cfg.layer_start,
            layer_end=fwd_cfg.layer_end,
            run_layers=fwd_cfg.run_layers,
            skip_layers=fwd_cfg.skip_layers,
            layer_forward="recurrent",
        )

        if fwd_cfg.layer_forward == "deer" and fwd_cfg.hidden_init == "warm_start":
            model._deer_save_prefill_hs = True

        _sync_cuda()
        t_prefill_start = time.perf_counter()
        logits = model(input_ids, use_cache=True, past_key_values=cache, fwd_cfg=prefill_cfg)
        _sync_cuda()
        t_prefill_end = time.perf_counter()
        model._deer_save_prefill_hs = False

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out = torch.cat([input_ids, next_token], dim=1)

        _sync_cuda()
        t_decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            logits = model(next_token, use_cache=True, past_key_values=cache, fwd_cfg=fwd_cfg)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
                break
        _sync_cuda()
        t_decode_end = time.perf_counter()

        num_generated = out.shape[1] - num_prompt_tokens
        return GenerationStats(
            output_ids=out,
            num_prompt_tokens=num_prompt_tokens,
            num_generated_tokens=num_generated,
            prefill_time=t_prefill_end - t_prefill_start,
            decode_time=t_decode_end - t_decode_start,
        )

    out = input_ids
    _sync_cuda()
    t_start = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = model(out, logits_to_keep=1, fwd_cfg=fwd_cfg)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out = torch.cat([out, next_token], dim=1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
            break
    _sync_cuda()
    t_end = time.perf_counter()

    num_generated = out.shape[1] - num_prompt_tokens
    total = t_end - t_start
    return GenerationStats(
        output_ids=out,
        num_prompt_tokens=num_prompt_tokens,
        num_generated_tokens=num_generated,
        prefill_time=0.0,
        decode_time=total,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone DEER-over-layers Qwen demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", type=str, default="Hi")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache and recompute the full prefix")
    parser.add_argument("--no-chat-template", action="store_true", help="Skip chat template and use the raw prompt")

    parser.add_argument("--layer-forward", type=str, default="recurrent", choices=["recurrent", "deer"])
    parser.add_argument("--parallel-iters", type=int, default=4, help="DEER iterations")
    parser.add_argument("--parallel-damping", type=float, default=1.0, help="Damping in (0,1]")
    parser.add_argument(
        "--hidden-init",
        type=str,
        default="warm_start",
        choices=["warm_start", "batched_fwd"],
        help="Reuse previous hidden states when compatible, otherwise use batched_fwd",
    )
    parser.add_argument(
        "--init-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to the selected hidden-state initializer",
    )
    parser.add_argument("--cache-update", type=str, default="kv_only", choices=["kv_only", "full_fwd"])
    parser.add_argument(
        "--jacobian",
        type=str,
        default="diagonal",
        choices=["diagonal", "diagonal_batched", "diagonal_batched_cached", "full", "full_scan"],
    )
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--run-layers", type=str, default=None)
    parser.add_argument("--skip-layers", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    args.trust_remote_code = True
    args.local_files_only = True

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    base.config.use_cache = True

    model = Qwen2ExplicitLoop(base).eval()

    if not args.no_chat_template and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = args.prompt

    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device)

    fwd_cfg = ForwardConfig(
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        run_layers=_parse_int_list(args.run_layers),
        skip_layers=_parse_int_list(args.skip_layers),
        layer_forward=args.layer_forward,
        parallel_iters=args.parallel_iters,
        parallel_damping=args.parallel_damping,
        hidden_init=args.hidden_init,
        init_noise_std=args.init_noise_std,
        cache_update=args.cache_update,
        jacobian=args.jacobian,
    )

    stats = generate_greedy(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        fwd_cfg=fwd_cfg,
        use_cache=not args.no_cache,
    )

    text = tokenizer.decode(stats.output_ids[0], skip_special_tokens=False)
    print(text)
    print("\n--- Stats ---")
    stats.print_stats()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
