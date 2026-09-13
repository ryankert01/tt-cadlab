"""Turn Palu low-rank factors into the tensors PaluAttention loads (see plan header for key layout)."""
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

from models.tt_transformers.tt.load_checkpoints import reverse_permute
from palu_tt import fused, hf

# Bumped whenever palu_layer_tensors changes the *content* of a tensor without changing its name,
# shape, dtype or the weight-set id -- the on-disk ttnn weight cache keys on those alone, so an old
# entry would otherwise be served for the new tensor. attention.cache_tag() puts it in every
# Palu-specific cache file name. v2: full-rank V is collapsed into the QKV weight (see below).
# v3: adds w_rot = [u_k | u_k T] for the fused decode kernel (stage 2).
# v4: group-aware layout (G-LRD): the group latent occupies its heads' slots in the fused QKV weight and
#     u_k/u_v/w_rot carry the group rank per head; group size 1 is unchanged.
CONVERT_VERSION = 4


@dataclass
class PaluRuntime:
    recon_len: Optional[int] = None
    check: bool = False
    # stage 2: decode through ttnn.transformer.palu_sdpa_decode (K reconstructed and rotated on
    # core) instead of matmul + prefill-mode RoPE + the MLA op. Only applies to layers with
    # r_k < head_dim; the V-only rows keep the stock K path either way.
    use_fused_k: bool = False
    # stage 2 Phase 2: chunk-relative rotation with the query advanced on core per chunk, so the
    # cos/sin table is read once per core instead of once per chunk. Needs use_fused_k.
    cs_recurrence: bool = False
    # stage 2: override the fused op's K chunk (positions; power of two, multiple of 32). None = the
    # op's own choice (largest power of two dividing max_seq_len, capped at 512), which is what the
    # stock decode op uses too.
    fused_k_chunk: Optional[int] = None
    # stage 2c: how many KV heads one core owns in the fused op's group-shared mode, so that a core's
    # heads share one latent read. None = pick the largest H that fits L1 per layer
    # (fused.group_heads_per_core); 1 = the stage 2b per-head kernel with the stock work split. Only
    # meaningful with grouped latents (group_size > 1).
    group_heads_per_core: Optional[int] = None
    # stage 2e: how the fused op rebuilds a K row tile. "legacy" = AB through c_10 and an SFPU add
    # (stage 2); "pack" = the packer's L1 accumulate does the add, two column tiles per DST trip;
    # "dst" = matmul straight into DST, dest-reuse multiplies, no AB scratch buffer.
    rot_mode: str = "legacy"
    # stage 2e: read the Phase 2 angle table once per grid column and multicast it to the column.
    table_mcast: bool = False
    # stage 2f: round-trip the prefill latents through the cache format before reconstructing K and V, so
    # perplexity and task scores (prefill-only) see what decode will read back. Off = the stage 1..2e
    # behaviour, where the cache format never entered a prefill score.
    latent_quant_in_prefill: bool = True


@dataclass
class PaluConfig:
    rank_k: List[int]
    rank_v: List[int]
    # Identifies the *weight set* these factors came from. The on-disk weight cache is keyed by
    # name/shape/dtype only, so two dumps with the same per-layer ranks would otherwise silently
    # share cache entries (and one would be loaded with the other's tensors). No default: every
    # producer must say which weights it built.
    cache_id: str
    runtime: PaluRuntime = field(default_factory=PaluRuntime)
    # Optional ttnn dtype override for the latent K/V caches (default: the model's KV_CACHE dtype,
    # bfp8). Kept as a plain object so this module stays ttnn-free.
    latent_dtype: Optional[object] = None
    # stage 2f: per-cache overrides of ``latent_dtype`` -- the K and V latent caches may take different
    # block-float formats (the rank/format frontier sweeps bfp8 against bfp4 on each side on its own).
    # None = follow ``latent_dtype``. Same plain-object convention as above.
    k_latent_dtype: Optional[object] = None
    v_latent_dtype: Optional[object] = None
    # Palu head group size (1 = M-LRD per head, 4 = G-LRD one group per chip on the N300). rank_k/rank_v
    # are the per-layer *group* latent widths; at group size 1 those are the per-head ranks as before.
    group_size: int = 1
    # stage 2f: which latent scale the factors were loaded with ("balanced" = Palu's split, "unit" =
    # rescaled so every latent component has unit variance). Record-only; the rescale itself already
    # happened in load_palu_dir before these tensors were built.
    latent_scale: str = "balanced"
    # Device tensors shared by every layer of one model on the fused path (the [cos | sin] table,
    # the Phase-2 step constants and transformation tile), keyed by attention.py; never compared
    # or printed.
    device_tables: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass
class LowRank:
    """Per-head view of the factors: W_h ≈ U_h @ VT_h with k_VT [n_kv, r, dim], k_U [n_kv, hd, r], same for v.

    group_size > 1 (Palu's G-LRD): the gs heads of a group share one latent, so k_VT[h] is the group's
    VT (identical for the group's heads) and k_U[h] is the head's hd-row slice of the group's U
    [gs*hd, r]. Every consumer that reconstructs a head sees the same per-head interface."""
    k_VT: torch.Tensor
    k_U: torch.Tensor
    v_VT: torch.Tensor
    v_U: torch.Tensor
    group_size: int = 1

    @property
    def r_k(self):
        return self.k_VT.shape[1]

    @property
    def r_v(self):
        return self.v_VT.shape[1]


def unit_scale_lowrank(lr: LowRank) -> LowRank:
    """Move each latent component's scale into the up-projection.

    Palu's whitened decomposition splits the singular values evenly (U sqrt(S), sqrt(S) VT), so latent
    component j has variance proportional to sigma_j over the whitening distribution. Scaling column j of
    U by c_j = ||U[:, j]|| (the norm over the whole group's rows, = sqrt(sigma_j)) and row j of VT by 1/c_j
    leaves U @ VT unchanged and makes every latent component unit variance -- what a block-floating-point
    cache wants. Zero columns (rank padding) are left alone.
    """
    def scale(VT, U, gs):
        n_kv = U.shape[0]
        VT2, U2 = VT.clone(), U.clone()
        for g in range(n_kv // gs):
            heads = slice(g * gs, (g + 1) * gs)
            c = torch.sqrt((U[heads] ** 2).sum(dim=(0, 1)))          # [r], over the group's gs*hd rows
            c = torch.where(c > 0, c, torch.ones_like(c))
            U2[heads] = U[heads] * c[None, None, :]
            VT2[heads] = VT[heads] / c[None, :, None]
        return VT2, U2
    k_VT, k_U = scale(lr.k_VT, lr.k_U, lr.group_size)
    v_VT, v_U = scale(lr.v_VT, lr.v_U, lr.group_size)
    return LowRank(k_VT=k_VT, k_U=k_U, v_VT=v_VT, v_U=v_U, group_size=lr.group_size)


def identity_lowrank(hf_attn: Dict[str, torch.Tensor], n_kv_heads: int, head_dim: int) -> LowRank:
    """r == head_dim, VT = W rows of the head, U = I. Makes the Palu path bit-for-bit the stock model."""
    eye = torch.eye(head_dim).expand(n_kv_heads, head_dim, head_dim).contiguous()
    return LowRank(
        k_VT=hf_attn["k"].float().reshape(n_kv_heads, head_dim, -1).contiguous(), k_U=eye.clone(),
        v_VT=hf_attn["v"].float().reshape(n_kv_heads, head_dim, -1).contiguous(), v_U=eye.clone(),
    )


def _read_palu_layer(tensors, ranks, layer, proj, n_kv_heads, head_dim, rank_multiple):
    """One projection of one layer -> per-head (VT, U) with the group's VT repeated over its heads.

    Palu stores one U per *group* (`U.{g}.weight`, [gs*hd, r_g]), one stacked VT ([sum r_g, dim]) and one
    rank per group; group size 1 is the head-wise case."""
    key = f"model.layers.{layer}.self_attn.{proj}"
    r_list = ranks[key]
    n_groups = len(r_list)
    assert n_kv_heads % n_groups == 0, f"{key}: {n_groups} groups do not divide {n_kv_heads} KV heads"
    gs = n_kv_heads // n_groups
    r_max = int(math.ceil(max(r_list) / rank_multiple) * rank_multiple)
    VT_all = tensors[f"{key}.VT.weight"]
    dim = VT_all.shape[1]
    VT = torch.zeros(n_kv_heads, r_max, dim)
    U = torch.zeros(n_kv_heads, head_dim, r_max)
    off = 0
    for g, r in enumerate(r_list):
        U_g = tensors[f"{key}.U.{g}.weight"]                       # [gs*hd, r]
        assert U_g.shape == (gs * head_dim, r), f"{key}.U.{g}: {tuple(U_g.shape)} != {(gs * head_dim, r)}"
        for hh in range(gs):
            h = g * gs + hh
            VT[h, :r] = VT_all[off:off + r]
            U[h, :, :r] = U_g[hh * head_dim:(hh + 1) * head_dim]
        off += r
    return VT, U, gs


def load_palu_dir(palu_dir: Path, n_layers: int, n_kv_heads: int, head_dim: int, rank_multiple: int = 32,
                  latent_scale: str = "balanced") -> Dict[int, LowRank]:
    """Reads a factors-only file (palu_lowrank.safetensors) or a full Palu HF dump (same key names). Group
    ranks are zero-padded to the layer max rounded up to rank_multiple, which changes nothing numerically.

    latent_scale: "balanced" (Palu's split, default) or "unit" (stage 2f: rescaled so every latent
    component has unit variance -- see unit_scale_lowrank).
    """
    assert latent_scale in ("balanced", "unit"), f"latent_scale must be 'balanced' or 'unit', got {latent_scale!r}"
    palu_dir = Path(palu_dir)
    ranks = json.loads((palu_dir / "config.json").read_text())["head_wise_ranks"]
    tensors = {}
    for f in sorted(palu_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                if ".k_proj." in k or ".v_proj." in k:
                    tensors[k] = sf.get_tensor(k).float()
    out = {}
    for i in range(n_layers):
        k_VT, k_U, gs_k = _read_palu_layer(tensors, ranks, i, "k_proj", n_kv_heads, head_dim, rank_multiple)
        v_VT, v_U, gs_v = _read_palu_layer(tensors, ranks, i, "v_proj", n_kv_heads, head_dim, rank_multiple)
        assert gs_k == gs_v, f"layer {i}: K group size {gs_k} != V group size {gs_v}"
        lr = LowRank(k_VT, k_U, v_VT, v_U, group_size=gs_k)
        out[i] = unit_scale_lowrank(lr) if latent_scale == "unit" else lr
    return out


def palu_dir_cache_id(palu_dir: Path) -> str:
    """Filesystem-safe id for a Palu dump: directory name + 8 hex chars of a cheap content hash.

    The hash covers the bytes of config.json (the head-wise ranks) and the name/size/mtime of every
    shard -- enough to tell a re-generated dump from the one whose tensors are already cached,
    without reading tens of GB.
    """
    palu_dir = Path(palu_dir)
    h = hashlib.sha256()
    h.update((palu_dir / "config.json").read_bytes())
    for f in sorted(palu_dir.glob("*.safetensors")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    name = re.sub(r"[^A-Za-z0-9._-]", "_", palu_dir.resolve().name)
    return f"{name}-{h.hexdigest()[:8]}"


def palu_cache_id_for(palu_dir: Path, latent_scale: str) -> str:
    """The weight-cache id for a dump under a latent scale: rescaled factors have the same shapes and
    dtypes as the balanced ones, and the on-disk weight cache is keyed by name/shape/dtype only, so the
    id has to differ (see the ttnn weight-cache note in the README)."""
    base = palu_dir_cache_id(palu_dir)
    return base if latent_scale == "balanced" else f"{base}-{latent_scale}"


def palu_layer_tensors(hf_attn: Dict[str, torch.Tensor], lr: LowRank, n_heads: int, n_kv_heads: int, head_dim: int):
    dim = hf_attn["q"].shape[1]
    r_k, r_v = lr.r_k, lr.r_v
    wk_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    wv_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    u_k = torch.zeros(n_kv_heads, r_k, head_dim)
    u_v = torch.zeros(n_kv_heads, r_v, head_dim)
    # r_k == head_dim (the V-only rows and the identity row): the K factorisation saves nothing, so
    # PaluAttention runs the *stock* K path -- stock RoPE straight into a head_dim cache. That path
    # needs the K block of the fused QKV weight to be the full Meta-permuted K weight of the head,
    # so build it from the factors themselves: reverse_permute(U_h @ VT_h). When the factorisation
    # really is full rank this equals the stock permuted wk block byte-for-byte (what
    # tt_transformers' convert_hf_qkv_to_meta_format produces for k_proj.weight); when the ranks are
    # head-wise and merely *padded* up to head_dim it stays the correct truncated weight instead of
    # silently restoring uncompressed K. `u_k` is then dead on device but still emitted so every
    # rank produces the same tensor shapes.
    # r_v == head_dim: the V factorisation saves nothing either, so PaluAttention takes the "fast V"
    # path -- the V half of the QKV head split goes straight into SDPA as V (prefill) and into the V
    # cache, and decode runs the MLA op on that cache with head_dim_v == head_dim. All of that is
    # only correct when the tensor the fused QKV weight produces really *is* V, i.e. when U_v == I.
    # identity_lowrank satisfies that; a genuine full-rank factorisation (plain SVD's
    # U = U_r*sqrt(S), VT = sqrt(S)*Vt_r, or Palu's whitened decomposition) does not, and the fast
    # path would then attend over x @ VT_v^T instead of V. So collapse the factorisation here: emit
    # the per-head product U_v @ VT_v as the V block (the factorised V weight -- the *original* wv
    # when the factors are exact, the truncated one when the ranks were merely padded up to
    # head_dim) and hand the identity to everything downstream. `u_v` is then dead on device but
    # still emitted so every rank produces the same tensor shapes, and wo_fused == wo.
    gs = lr.group_size
    # Full-rank collapses (the stock-K and fast-V paths) exist for the per-head case only: a grouped
    # latent is always kept as a latent and reconstructed per head (in prefill, in the naive decode
    # and in the fused kernel), so wo_fused is plain wo there.
    fast_v = gs == 1 and r_v == head_dim
    # The V-side up-projection *as the device will see it*: the identity on the fast path (U_v has
    # just been folded into wv_lat_pad below), the factor itself otherwise. Deliberately not called
    # `v_U` -- it differs from `lr.v_U` exactly when fast_v, and that shadowing is a trap.
    u_v_eff = torch.eye(head_dim).expand(n_kv_heads, head_dim, head_dim) if fast_v else lr.v_U
    for h in range(n_kv_heads):
        g, hh = h // gs, h % gs
        if gs == 1 and r_k == head_dim:
            wk_lat_pad[h * head_dim:(h + 1) * head_dim] = reverse_permute(lr.k_U[h] @ lr.k_VT[h], 1, head_dim, dim)
        elif hh == 0:
            # the group's VT goes once, into the first r_k rows of the group's gs*head_dim slot
            wk_lat_pad[g * gs * head_dim:g * gs * head_dim + r_k] = lr.k_VT[h]
        if fast_v:
            # no reverse_permute on the V side: V never sees RoPE, so it stays in HF head order
            wv_lat_pad[h * head_dim:(h + 1) * head_dim] = lr.v_U[h] @ lr.v_VT[h]
        elif hh == 0:
            wv_lat_pad[g * gs * head_dim:g * gs * head_dim + r_v] = lr.v_VT[h]
        # K leaves the latent in Meta-permuted head order (RoPE convention of tt_transformers)
        u_k[h] = reverse_permute(lr.k_U[h], 1, head_dim, r_k).T
        u_v[h] = u_v_eff[h].T
    wo = hf_attn["o"].float()                       # [dim, n_heads*hd]
    if gs > 1:
        wo_fused = wo.clone()
    else:
        wo_fused = torch.zeros_like(wo)
        rep = n_heads // n_kv_heads
        for qh in range(n_heads):
            kv = qh // rep
            wo_fused[:, qh * head_dim:qh * head_dim + r_v] = wo[:, qh * head_dim:(qh + 1) * head_dim] @ u_v_eff[kv]
    return {"wk_lat_pad": wk_lat_pad, "wv_lat_pad": wv_lat_pad, "u_k": u_k, "u_v": u_v, "wo_fused": wo_fused,
            "w_rot": fused.w_rot_from_u_k(u_k)}


def hf_attn_weights(hf_dir: Path, layer: int) -> Dict[str, torch.Tensor]:
    p = f"model.layers.{layer}.self_attn."
    t = hf.load_tensors(hf_dir, lambda k: k.startswith(p) and k.endswith("proj.weight"))
    return {"q": t[p + "q_proj.weight"], "k": t[p + "k_proj.weight"], "v": t[p + "v_proj.weight"], "o": t[p + "o_proj.weight"]}


def add_palu_to_state_dict(state_dict: dict, hf_dir: Path, palu_dir: Optional[Path], n_layers: int,
                           n_heads: int, n_kv_heads: int, head_dim: int, latent_scale: str = "balanced") -> PaluConfig:
    """palu_dir None -> identity factorisation (stock model through the Palu path)."""
    lrs = None if palu_dir is None else load_palu_dir(palu_dir, n_layers, n_kv_heads, head_dim, latent_scale=latent_scale)
    rank_k, rank_v = [], []
    for i in range(n_layers):
        w = hf_attn_weights(hf_dir, i)
        lr = identity_lowrank(w, n_kv_heads, head_dim) if lrs is None else lrs[i]
        for name, t in palu_layer_tensors(w, lr, n_heads, n_kv_heads, head_dim).items():
            state_dict[f"layers.{i}.attention.palu.{name}"] = t
        rank_k.append(lr.r_k)
        rank_v.append(lr.r_v)
    cache_id = "identity" if palu_dir is None else palu_cache_id_for(palu_dir, latent_scale)
    gs = {lr.group_size for lr in (lrs.values() if lrs else [])} or {1}
    assert len(gs) == 1, f"mixed group sizes across layers: {gs}"
    return PaluConfig(rank_k=rank_k, rank_v=rank_v, cache_id=cache_id, group_size=gs.pop(), latent_scale=latent_scale)
