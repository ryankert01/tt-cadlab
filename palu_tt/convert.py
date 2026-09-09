"""Turn Palu low-rank factors into the tensors PaluAttention loads (see plan header for key layout)."""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

from models.tt_transformers.tt.load_checkpoints import reverse_permute
from palu_tt import hf


@dataclass
class PaluRuntime:
    recon_len: Optional[int] = None
    check: bool = False


@dataclass
class PaluConfig:
    rank_k: List[int]
    rank_v: List[int]
    runtime: PaluRuntime = field(default_factory=PaluRuntime)


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


def palu_layer_tensors(hf_attn: Dict[str, torch.Tensor], lr: LowRank, n_heads: int, n_kv_heads: int, head_dim: int):
    dim = hf_attn["q"].shape[1]
    r_k, r_v = lr.r_k, lr.r_v
    wk_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    wv_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    u_k = torch.zeros(n_kv_heads, r_k, head_dim)
    u_v = torch.zeros(n_kv_heads, r_v, head_dim)
    for h in range(n_kv_heads):
        wk_lat_pad[h * head_dim:h * head_dim + r_k] = lr.k_VT[h]
        wv_lat_pad[h * head_dim:h * head_dim + r_v] = lr.v_VT[h]
        # K leaves the latent in Meta-permuted head order (RoPE convention of tt_transformers)
        u_k[h] = reverse_permute(lr.k_U[h], 1, head_dim, r_k).T
        u_v[h] = lr.v_U[h].T
    wo = hf_attn["o"].float()                       # [dim, n_heads*hd]
    wo_fused = torch.zeros_like(wo)
    rep = n_heads // n_kv_heads
    for qh in range(n_heads):
        kv = qh // rep
        wo_fused[:, qh * head_dim:qh * head_dim + r_v] = wo[:, qh * head_dim:(qh + 1) * head_dim] @ lr.v_U[kv]
    return {"wk_lat_pad": wk_lat_pad, "wv_lat_pad": wv_lat_pad, "u_k": u_k, "u_v": u_v, "wo_fused": wo_fused}


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
    return PaluConfig(rank_k=rank_k, rank_v=rank_v)
