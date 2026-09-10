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
CONVERT_VERSION = 3


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
    # Device tensors shared by every layer of one model on the fused path (the [cos | sin] table,
    # the Phase-2 step constants and transformation tile), keyed by attention.py; never compared
    # or printed.
    device_tables: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass
class LowRank:
    """Head-major factors: W_h ≈ U_h @ VT_h. Shapes k_VT [n_kv, r_k, dim], k_U [n_kv, hd, r_k], same for v."""
    k_VT: torch.Tensor
    k_U: torch.Tensor
    v_VT: torch.Tensor
    v_U: torch.Tensor

    @property
    def r_k(self):
        return self.k_VT.shape[1]

    @property
    def r_v(self):
        return self.v_VT.shape[1]


def identity_lowrank(hf_attn: Dict[str, torch.Tensor], n_kv_heads: int, head_dim: int) -> LowRank:
    """r == head_dim, VT = W rows of the head, U = I. Makes the Palu path bit-for-bit the stock model."""
    eye = torch.eye(head_dim).expand(n_kv_heads, head_dim, head_dim).contiguous()
    return LowRank(
        k_VT=hf_attn["k"].float().reshape(n_kv_heads, head_dim, -1).contiguous(), k_U=eye.clone(),
        v_VT=hf_attn["v"].float().reshape(n_kv_heads, head_dim, -1).contiguous(), v_U=eye.clone(),
    )


def _read_palu_layer(tensors, ranks, layer, proj, n_kv_heads, head_dim, rank_multiple):
    key = f"model.layers.{layer}.self_attn.{proj}"
    r_list = ranks[key]
    assert len(r_list) == n_kv_heads, f"{key}: expected {n_kv_heads} groups (group size 1), got {len(r_list)}"
    r_max = int(math.ceil(max(r_list) / rank_multiple) * rank_multiple)
    VT_all = tensors[f"{key}.VT.weight"]
    dim = VT_all.shape[1]
    VT = torch.zeros(n_kv_heads, r_max, dim)
    U = torch.zeros(n_kv_heads, head_dim, r_max)
    off = 0
    for h, r in enumerate(r_list):
        VT[h, :r] = VT_all[off:off + r]
        U[h, :, :r] = tensors[f"{key}.U.{h}.weight"]
        off += r
    return VT, U


def load_palu_dir(palu_dir: Path, n_layers: int, n_kv_heads: int, head_dim: int, rank_multiple: int = 32) -> Dict[int, LowRank]:
    """Reads our delta file or a full Palu HF dump (same key names). Per-head ranks are zero-padded to the
    layer max rounded up to rank_multiple, which changes nothing numerically."""
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
        k_VT, k_U = _read_palu_layer(tensors, ranks, i, "k_proj", n_kv_heads, head_dim, rank_multiple)
        v_VT, v_U = _read_palu_layer(tensors, ranks, i, "v_proj", n_kv_heads, head_dim, rank_multiple)
        out[i] = LowRank(k_VT, k_U, v_VT, v_U)
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
    fast_v = r_v == head_dim
    # The V-side up-projection *as the device will see it*: the identity on the fast path (U_v has
    # just been folded into wv_lat_pad below), the factor itself otherwise. Deliberately not called
    # `v_U` -- it differs from `lr.v_U` exactly when fast_v, and that shadowing is a trap.
    u_v_eff = torch.eye(head_dim).expand(n_kv_heads, head_dim, head_dim) if fast_v else lr.v_U
    for h in range(n_kv_heads):
        if r_k == head_dim:
            wk_lat_pad[h * head_dim:(h + 1) * head_dim] = reverse_permute(lr.k_U[h] @ lr.k_VT[h], 1, head_dim, dim)
        else:
            wk_lat_pad[h * head_dim:h * head_dim + r_k] = lr.k_VT[h]
        if fast_v:
            # no reverse_permute on the V side: V never sees RoPE, so it stays in HF head order
            wv_lat_pad[h * head_dim:(h + 1) * head_dim] = lr.v_U[h] @ lr.v_VT[h]
        else:
            wv_lat_pad[h * head_dim:h * head_dim + r_v] = lr.v_VT[h]
        # K leaves the latent in Meta-permuted head order (RoPE convention of tt_transformers)
        u_k[h] = reverse_permute(lr.k_U[h], 1, head_dim, r_k).T
        u_v[h] = u_v_eff[h].T
    wo = hf_attn["o"].float()                       # [dim, n_heads*hd]
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
                           n_heads: int, n_kv_heads: int, head_dim: int) -> PaluConfig:
    """palu_dir None -> identity factorisation (stock model through the Palu path)."""
    lrs = None if palu_dir is None else load_palu_dir(palu_dir, n_layers, n_kv_heads, head_dim)
    rank_k, rank_v = [], []
    for i in range(n_layers):
        w = hf_attn_weights(hf_dir, i)
        lr = identity_lowrank(w, n_kv_heads, head_dim) if lrs is None else lrs[i]
        for name, t in palu_layer_tensors(w, lr, n_heads, n_kv_heads, head_dim).items():
            state_dict[f"layers.{i}.attention.palu.{name}"] = t
        rank_k.append(lr.r_k)
        rank_v.append(lr.r_v)
    cache_id = "identity" if palu_dir is None else palu_dir_cache_id(palu_dir)
    return PaluConfig(rank_k=rank_k, rank_v=rank_v, cache_id=cache_id)
