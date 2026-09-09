import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from palu_tt import compress


def test_svd_per_head_full_rank_is_exact():
    W = torch.randn(4 * 32, 64)
    U, VT = compress.svd_per_head(W, n_heads=4, rank=32)
    assert U.shape == (4, 32, 32) and VT.shape == (4, 32, 64)
    rec = torch.cat([U[h] @ VT[h] for h in range(4)], dim=0)
    assert torch.allclose(rec, W, atol=1e-4)


def test_svd_per_head_error_decreases_with_rank():
    W = torch.randn(2 * 64, 256)
    errs = []
    for r in (16, 32, 64):
        U, VT = compress.svd_per_head(W, n_heads=2, rank=r)
        rec = torch.cat([U[h] @ VT[h] for h in range(2)], dim=0)
        errs.append((rec - W).norm().item())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3


def test_compress_attention_writes_palu_layout(tmp_path):
    # tiny fake HF checkpoint: 2 layers, dim 64, 4 q heads, 2 kv heads, head_dim 16
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    cfg = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2}
    (hf_dir / "config.json").write_text(json.dumps(cfg))
    tensors = {}
    for i in range(2):
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(32, 64)
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(32, 64)
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(64, 64)
    save_file(tensors, str(hf_dir / "model.safetensors"))
    (hf_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    out = tmp_path / "out"
    compress.compress_attention(hf_dir, out, rank_k=8, rank_v=16)
    t = load_file(str(out / "palu_attn.safetensors"))
    assert t["model.layers.0.self_attn.k_proj.VT.weight"].shape == (2 * 8, 64)
    assert t["model.layers.0.self_attn.k_proj.U.1.weight"].shape == (16, 8)
    assert t["model.layers.1.self_attn.v_proj.VT.weight"].shape == (2 * 16, 64)
    assert t["model.layers.1.self_attn.v_proj.U.0.weight"].shape == (16, 16)
    c = json.loads((out / "config.json").read_text())
    assert c["head_wise_ranks"]["model.layers.0.self_attn.k_proj"] == [8, 8]
    assert c["head_wise_ranks"]["model.layers.1.self_attn.v_proj"] == [16, 16]
    # reconstruction of v (full rank 16 == head_dim) is exact
    W = tensors["model.layers.1.self_attn.v_proj.weight"]
    VT = t["model.layers.1.self_attn.v_proj.VT.weight"]
    rec = torch.cat([t[f"model.layers.1.self_attn.v_proj.U.{h}.weight"] @ VT[h * 16:(h + 1) * 16] for h in range(2)])
    assert torch.allclose(rec, W, atol=1e-4)
