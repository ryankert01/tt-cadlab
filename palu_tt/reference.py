"""CPU ground truth. HF conventions (rotate_half RoPE, unpermuted weights); fp32."""
import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from palu_tt import convert


class PaluRefAttention:
    def __init__(self, hf_cfg: dict, w: Dict[str, torch.Tensor], lr: convert.LowRank):
        self.cfg = LlamaConfig(**{k: v for k, v in hf_cfg.items() if k != "architectures"})
        self.nh = self.cfg.num_attention_heads
        self.nkv = self.cfg.num_key_value_heads
        self.hd = self.cfg.hidden_size // self.nh
        self.wq, self.wo = w["q"].float(), w["o"].float()
        self.lr = lr
        self.rotary = LlamaRotaryEmbedding(config=self.cfg)
        self.cache_k_lat = None  # [B, nkv, T, r_k]
        self.cache_v_lat = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        B, S, _ = x.shape
        q = (x @ self.wq.T).view(B, S, self.nh, self.hd).transpose(1, 2)              # [B, nh, S, hd]
        k_lat = torch.einsum("bsd,hrd->bhsr", x, self.lr.k_VT)                        # [B, nkv, S, r_k]
        v_lat = torch.einsum("bsd,hrd->bhsr", x, self.lr.v_VT)
        if self.cache_k_lat is None or start_pos == 0:
            self.cache_k_lat, self.cache_v_lat = k_lat, v_lat
        else:
            assert self.cache_k_lat.shape[2] == start_pos
            self.cache_k_lat = torch.cat([self.cache_k_lat, k_lat], dim=2)
            self.cache_v_lat = torch.cat([self.cache_v_lat, v_lat], dim=2)
        T = self.cache_k_lat.shape[2]
        K = torch.einsum("bhtr,hdr->bhtd", self.cache_k_lat, self.lr.k_U)              # [B, nkv, T, hd]
        V = torch.einsum("bhtr,hdr->bhtd", self.cache_v_lat, self.lr.v_U)
        pos_q = torch.arange(start_pos, start_pos + S)[None].expand(B, -1)
        pos_k = torch.arange(0, T)[None].expand(B, -1)
        cos_q, sin_q = self.rotary(x, pos_q)
        cos_k, sin_k = self.rotary(x, pos_k)
        q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        _, K = apply_rotary_pos_emb(K, K, cos_k, sin_k)
        rep = self.nh // self.nkv
        K = K.repeat_interleave(rep, dim=1)
        V = V.repeat_interleave(rep, dim=1)
        scores = q @ K.transpose(-1, -2) / math.sqrt(self.hd)                          # [B, nh, S, T]
        mask = torch.ones(S, T, dtype=torch.bool).tril(diagonal=start_pos)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, S, self.nh * self.hd)
        return out @ self.wo.T


class LowRankProj(nn.Module):
    """x -> concat_h (x @ VT_h^T @ U_h^T). Drop-in for HF k_proj / v_proj."""

    def __init__(self, VT: torch.Tensor, U: torch.Tensor):
        super().__init__()
        n_kv, r, dim = VT.shape
        self.n_kv, self.r = n_kv, r
        self.VT = nn.Parameter(VT.reshape(n_kv * r, dim).contiguous(), requires_grad=False)  # [n_kv*r, dim]
        self.U = nn.Parameter(U.contiguous(), requires_grad=False)                          # [n_kv, hd, r]

    def forward(self, x):
        lat = (x @ self.VT.T).reshape(*x.shape[:-1], self.n_kv, self.r)
        out = torch.einsum("...hr,hdr->...hd", lat, self.U)
        return out.reshape(*x.shape[:-1], -1)


def build_cpu_lowrank_llama(hf_dir: Path, palu_dir: Optional[Path]) -> LlamaForCausalLM:
    torch.set_num_threads(12)
    model = LlamaForCausalLM.from_pretrained(str(hf_dir), torch_dtype=torch.float32).eval()
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    # rank_multiple=1: no TT tile-alignment padding here (that's a device-path concern in
    # convert.add_palu_to_state_dict); the CPU reference should use the exact stored ranks.
    lrs = None if palu_dir is None else convert.load_palu_dir(
        palu_dir, cfg.num_hidden_layers, cfg.num_key_value_heads, hd, rank_multiple=1)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        w = {"q": attn.q_proj.weight.data, "k": attn.k_proj.weight.data, "v": attn.v_proj.weight.data, "o": attn.o_proj.weight.data}
        lr = convert.identity_lowrank(w, cfg.num_key_value_heads, hd) if lrs is None else lrs[i]
        attn.k_proj = LowRankProj(lr.k_VT, lr.k_U)
        attn.v_proj = LowRankProj(lr.v_VT, lr.v_U)
    return model
