"""Palu-style per-head SVD of k_proj / v_proj (group size 1, plain SVD, uniform rank)."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from palu_tt import hf


def svd_per_head(W: torch.Tensor, n_heads: int, rank: int):
    """W: [n_heads*hd, dim] -> U [n_heads, hd, rank], VT [n_heads, rank, dim] with W_h ≈ U_h @ VT_h.
    Same balanced split as Palu: U = U_r·sqrt(S), VT = sqrt(S)·Vt_r."""
    hd = W.shape[0] // n_heads
    Wh = W.float().reshape(n_heads, hd, -1)
    Us, VTs = [], []
    for h in range(n_heads):
        U, S, Vt = torch.linalg.svd(Wh[h], full_matrices=False)
        s = torch.sqrt(S[:rank])
        Us.append(U[:, :rank] * s[None, :])
        VTs.append(s[:, None] * Vt[:rank, :])
    return torch.stack(Us), torch.stack(VTs)


def compress_attention(hf_dir: Path, out_dir: Path, rank_k: int, rank_v: int, layers=None):
    hf_dir, out_dir = Path(hf_dir), Path(out_dir)
    cfg = hf.load_config(hf_dir)
    n_kv = cfg["num_key_value_heads"]
    n_layers = cfg["num_hidden_layers"] if layers is None else layers
    tensors = hf.load_tensors(hf_dir, lambda k: ("k_proj.weight" in k or "v_proj.weight" in k) and "layers." in k)
    out, ranks = {}, {}
    for i in range(n_layers):
        for proj, r in (("k_proj", rank_k), ("v_proj", rank_v)):
            key = f"model.layers.{i}.self_attn.{proj}"
            U, VT = svd_per_head(tensors[f"{key}.weight"], n_kv, r)
            out[f"{key}.VT.weight"] = VT.reshape(n_kv * r, -1).contiguous()
            for h in range(n_kv):
                out[f"{key}.U.{h}.weight"] = U[h].contiguous()
            ranks[key] = [r] * n_kv
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "palu_attn.safetensors"))
    (out_dir / "config.json").write_text(
        json.dumps({"head_wise_ranks": ranks, "palu_tt": {"method": "svd", "group_size": 1,
                    "rank_k": rank_k, "rank_v": rank_v, "source": str(hf_dir)}}, indent=1)
    )
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank-k", type=int, required=True)
    p.add_argument("--rank-v", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=hf.MODEL_ID)
    a = p.parse_args()
    torch.set_num_threads(12)
    print(compress_attention(hf.snapshot_dir(a.model), a.out, a.rank_k, a.rank_v))
