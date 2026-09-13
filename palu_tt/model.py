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
                k_latent_dtype=None, v_latent_dtype=None, kv_cache_dtype=None,
                use_fused_k: bool = False, cs_recurrence: bool = False, fused_k_chunk: Optional[int] = None,
                group_heads: Optional[int] = None, rot_mode: str = "legacy", table_mcast: bool = False,
                latent_scale: str = "balanced", latent_quant_in_prefill: bool = True):
    """latent_dtype: optional ttnn dtype for the Palu latent K/V caches (None = the model's KV cache
    dtype, i.e. bfp8). k_latent_dtype / v_latent_dtype: stage 2f, per-cache overrides of latent_dtype
    (None = follow it), so the K and V latents can take different block-float formats.
    kv_cache_dtype: stage 2f, the *model's* KV_CACHE precision -- the stock path's whole KV cache, and on
    the Palu path the dtype the naive decode's reconstructed K/V temporaries are cast to. None leaves
    tt_transformers' own choice (bfp8) alone. use_fused_k / cs_recurrence / fused_k_chunk: stage-2 fused decode kernel
    (Phase 1, Phase 2, chunk override). group_heads: stage-2c KV heads per core in the fused op's
    group-shared mode (None = pick the largest that fits L1 per layer, 1 = the stage 2b per-head
    kernel). latent_scale: stage 2f, "balanced" (Palu's split, default) or "unit" (every latent
    component rescaled to unit variance -- see convert.unit_scale_lowrank). latent_quant_in_prefill:
    stage 2f, round-trip the prefill latents through the cache format before reconstructing K and V, so
    the prefill-only instruments (eval_ppl, eval_tasks) score what decode reads back; False restores the
    stage 1..2e behaviour. Only meaningful for kind="palu"."""
    assert kind in ("stock", "palu")
    args = ModelArgs(mesh_device, instruct=True, max_batch_size=max_batch_size, max_seq_len=max_seq_len, cache_hf=False)
    args.use_qk_fused = False          # same RoPE path for both kinds
    if n_layers is not None:
        args.n_layers = n_layers
    set_kv_cache_dtype(args, kv_cache_dtype)
    state_dict = args.load_state_dict()
    cfg = None
    if kind == "palu":
        cfg = convert.add_palu_to_state_dict(state_dict, hf.snapshot_dir(), palu_dir, args.n_layers,
                                             args.n_heads, args.n_kv_heads, args.head_dim, latent_scale=latent_scale)
        apply_palu_flags(cfg, latent_dtype=latent_dtype, k_latent_dtype=k_latent_dtype,
                         v_latent_dtype=v_latent_dtype, use_fused_k=use_fused_k, cs_recurrence=cs_recurrence,
                         fused_k_chunk=fused_k_chunk, group_heads=group_heads, rot_mode=rot_mode,
                         table_mcast=table_mcast, latent_quant_in_prefill=latent_quant_in_prefill)
        args.palu = cfg
    model = Transformer(args=args, mesh_device=mesh_device, dtype=dtype, state_dict=state_dict,
                        weight_cache_path=args.weight_cache_path(dtype),
                        attention_class=PaluAttention if kind == "palu" else None)
    return args, model, cfg


def set_kv_cache_dtype(args, kv_cache_dtype) -> None:
    """Override the model's KV_CACHE precision on a built ``ModelArgs`` (stage 2f's ``--kv-dtype``).

    This is the stock path's whole KV cache -- the format-only competitor the rank/format frontier is
    measured against -- and, on the Palu path, the dtype ``PaluAttention`` casts the naive decode's
    reconstructed K/V temporaries to. tt-transformers reads it per layer from the ``DecodersPrecision``
    object, so the override rebuilds that object with the KV_CACHE group replaced. ``None`` leaves
    tt-transformers' own choice (bfp8 for Llama-3) alone.

    Asking for the format the model already has is a no-op rather than an equivalent rebuild: the object
    tt-transformers built for itself may carry per-decoder settings loaded from
    ``model_params/<model>/accuracy_decoder_config.json`` (Llama-3.1-8B-Instruct has no such file today,
    so the rebuild below would match it exactly -- but a tt-metal bump could add one), and it alone
    carries the ``__name__`` the demo flows print. So --kv-dtype's default costs every existing row
    nothing at all.

    Must be called before the modules are constructed (``Attention.__init__`` reads the dtype once) and
    after any ``args.n_layers`` override (the precision object is sized by the layer count).
    """
    if kv_cache_dtype is None:
        return
    from models.tt_transformers.tt.model_config import DecodersPrecision, ModelOptimizations, PrecisionSetting, TensorGroup
    if all(args.optimizations.get_tensor_dtype(decoder_id=i, tensor=TensorGroup.KV_CACHE) == kv_cache_dtype
           for i in range(args.n_layers)):
        return
    setting = {ttnn.bfloat8_b: PrecisionSetting.BFP8, ttnn.bfloat4_b: PrecisionSetting.BFP4,
               ttnn.bfloat16: PrecisionSetting.BF16}[kv_cache_dtype]
    conf = ModelOptimizations.accuracy(args.model_name)
    conf.tensor_dtype_settings[TensorGroup.KV_CACHE] = setting
    args.optimizations = DecodersPrecision(args.n_layers, args.model_name, conf)
    args.model_config["DECODERS_OPTIMIZATIONS"] = args.optimizations


def apply_palu_flags(cfg, latent_dtype=None, k_latent_dtype=None, v_latent_dtype=None,
                     use_fused_k: bool = False, cs_recurrence: bool = False, fused_k_chunk=None,
                     group_heads=None, rot_mode: str = "legacy", table_mcast: bool = False,
                     latent_quant_in_prefill: bool = True, latent_scale=None, kv_cache_dtype=None):
    """Write one run's Palu switches onto ``cfg``: the single place ``build_model``'s keyword arguments
    become ``PaluConfig`` state.

    ``palu_tt.bench``'s single-layer mode builds its attention by hand instead of through
    ``build_model`` and used to re-copy these switches itself, so every switch added since (rot_mode,
    table_mcast, the latent formats, latent_quant_in_prefill) had to be remembered in two places -- and
    some were not. Both callers go through here now, so a new switch is either handled for both or a
    ``TypeError`` for both.

    ``latent_scale`` and ``kv_cache_dtype`` are accepted only so a caller can splat the whole
    ``build_model`` keyword set: neither is Palu runtime state. The scale is applied by the loader and
    recorded on the config when the factors are built (asserted here), and the KV precision lives on
    the model's ``DecodersPrecision`` -- see ``set_kv_cache_dtype``.
    """
    assert latent_scale is None or cfg.latent_scale == latent_scale, \
        f"latent_scale {latent_scale!r} does not match the loaded factors ({cfg.latent_scale!r})"
    cfg.latent_dtype = latent_dtype
    cfg.k_latent_dtype, cfg.v_latent_dtype = k_latent_dtype, v_latent_dtype
    rt = cfg.runtime
    rt.use_fused_k, rt.cs_recurrence, rt.fused_k_chunk = use_fused_k, cs_recurrence, fused_k_chunk
    rt.group_heads_per_core = group_heads
    rt.rot_mode, rt.table_mcast = rot_mode, bool(table_mcast)
    rt.latent_quant_in_prefill = bool(latent_quant_in_prefill)
    return cfg


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
