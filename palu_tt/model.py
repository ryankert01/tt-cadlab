"""Build stock / Palu full models on top of tt_transformers' Transformer."""
from typing import Optional

import torch
import ttnn
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

from palu_tt import convert, hf
from palu_tt.attention import PaluAttention


def build_model(mesh_device, kind: str, palu_dir: Optional[str] = None, max_batch_size: int = 1, max_seq_len: int = 2048,
                n_layers: Optional[int] = None, dtype=ttnn.bfloat8_b):
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
    """Per chip. Stock: K and V full head_dim; Palu: from the attention module."""
    attn = model.layers[0].attention
    if hasattr(attn, "kv_bytes_per_token"):
        return attn.kv_bytes_per_token()
    bpe = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0}[attn.kv_cache_dtype]
    return attn.n_local_kv_heads * 2 * attn.head_dim * bpe
