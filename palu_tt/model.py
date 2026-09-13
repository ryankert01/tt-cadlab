"""Build stock / Palu full models on top of tt_transformers' Transformer."""
from typing import Optional

import torch
import ttnn
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

from palu_tt import convert, hf
from palu_tt.attention import PaluAttention


def build_model(mesh_device, kind: str, palu_dir: Optional[str] = None, max_batch_size: int = 1, max_seq_len: int = 2048,
                n_layers: Optional[int] = None, dtype=ttnn.bfloat8_b, latent_dtype=None,
                use_fused_k: bool = False, cs_recurrence: bool = False, fused_k_chunk: Optional[int] = None,
                group_heads: Optional[int] = None, rot_mode: str = "legacy", table_mcast: bool = False):
    """latent_dtype: optional ttnn dtype for the Palu latent K/V caches (None = the model's KV cache
    dtype, i.e. bfp8). use_fused_k / cs_recurrence / fused_k_chunk: stage-2 fused decode kernel
    (Phase 1, Phase 2, chunk override). group_heads: stage-2c KV heads per core in the fused op's
    group-shared mode (None = pick the largest that fits L1 per layer, 1 = the stage 2b per-head
    kernel). Only meaningful for kind="palu"."""
    assert kind in ("stock", "palu")
    args = ModelArgs(mesh_device, instruct=True, max_batch_size=max_batch_size, max_seq_len=max_seq_len, cache_hf=False)
    args.use_qk_fused = False          # same RoPE path for both kinds
    if n_layers is not None:
        args.n_layers = n_layers
    state_dict = args.load_state_dict()
    cfg = None
    if kind == "palu":
        cfg = convert.add_palu_to_state_dict(state_dict, hf.snapshot_dir(), palu_dir, args.n_layers,
                                             args.n_heads, args.n_kv_heads, args.head_dim)
        cfg.latent_dtype = latent_dtype
        cfg.runtime.use_fused_k, cfg.runtime.cs_recurrence, cfg.runtime.fused_k_chunk = use_fused_k, cs_recurrence, fused_k_chunk
        cfg.runtime.group_heads_per_core = group_heads
        cfg.runtime.rot_mode, cfg.runtime.table_mcast = rot_mode, bool(table_mcast)
        args.palu = cfg
    model = Transformer(args=args, mesh_device=mesh_device, dtype=dtype, state_dict=state_dict,
                        weight_cache_path=args.weight_cache_path(dtype),
                        attention_class=PaluAttention if kind == "palu" else None)
    return args, model, cfg


def set_recon_len(cfg: Optional[convert.PaluConfig], current_pos: torch.Tensor):
    """Reconstruction length for the next decode step, rounded up to 256.

    Every distinct value changes the decode op shapes and recompiles the decode graph; rounding to
    256 (rather than the tile 32) keeps that to one recompile per 256 generated tokens. The
    attention caps the value at the cache length.
    """
    if cfg is not None:
        cfg.runtime.recon_len = ((int(current_pos.max()) + 1 + 255) // 256) * 256


def kv_bytes_per_token_per_layer(model, args) -> float:
    """Per chip. Stock: K and V full head_dim; Palu: from the attention module.

    This is **layer 0**, not the model: the calibrated dumps have per-layer ranks, so the consumer
    that reports a whole-model figure (``report._kv_model_mean``) rescales this by the mean
    per-layer rank sum read from the dump's ``head_wise_ranks``.
    """
    attn = model.layers[0].attention
    if hasattr(attn, "kv_bytes_per_token"):
        return attn.kv_bytes_per_token()
    bpe = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0, ttnn.bfloat4_b: 0.5625}[attn.kv_cache_dtype]
    return attn.n_local_kv_heads * 2 * attn.head_dim * bpe
