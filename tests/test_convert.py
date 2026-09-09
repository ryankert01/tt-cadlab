import torch

from models.tt_transformers.tt.load_checkpoints import reverse_permute
from palu_tt import convert

DIM, NH, NKV, HD = 64, 4, 2, 16


def _hf_attn():
    return {
        "q": torch.randn(NH * HD, DIM), "k": torch.randn(NKV * HD, DIM),
        "v": torch.randn(NKV * HD, DIM), "o": torch.randn(DIM, NH * HD),
    }


def test_identity_lowrank_reproduces_stock_k_and_v():
    w = _hf_attn()
    lr = convert.identity_lowrank(w, NKV, HD)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    x = torch.randn(5, DIM)
    # r_k == head_dim -> the K block of the fused QKV weight *is* the stock Meta-permuted wk
    # (V-only fast path: PaluAttention runs the stock K path and never touches u_k).
    assert torch.allclose(t["wk_lat_pad"], reverse_permute(w["k"], NKV, NKV * HD, DIM), atol=1e-6)
    lat_v = x @ t["wv_lat_pad"].T
    v = torch.cat([lat_v[:, h * HD:(h + 1) * HD] @ t["u_v"][h] for h in range(NKV)], dim=-1)
    assert torch.allclose(v, x @ w["v"].T, atol=1e-5)


def test_full_rank_k_emits_stock_permuted_wk():
    """r_k == head_dim (identity row and any Palu dump with rank 128): the K block must be the
    stock permuted wk, exactly as convert_hf_qkv_to_meta_format builds it for k_proj.weight."""
    w = _hf_attn()
    r_v = 8
    lr = convert.LowRank(
        k_VT=torch.randn(NKV, HD, DIM), k_U=torch.randn(NKV, HD, HD),      # full-rank, non-identity
        v_VT=torch.randn(NKV, r_v, DIM), v_U=torch.randn(NKV, HD, r_v),
    )
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert lr.r_k == HD
    assert torch.allclose(t["wk_lat_pad"], reverse_permute(w["k"], NKV, NKV * HD, DIM), atol=1e-6)
    assert t["u_k"].shape == (NKV, HD, HD)   # still emitted so device shapes stay uniform
    # the V side is untouched by the fast path
    x = torch.randn(5, DIM)
    lat_v = x @ t["wv_lat_pad"].T
    for h in range(NKV):
        assert torch.allclose(lat_v[:, h * HD:h * HD + r_v], x @ lr.v_VT[h].T, atol=1e-5)


def test_low_rank_k_block_is_padded_latent():
    """r_k < head_dim keeps the Palu latent branch: rows are VT_k, the rest zero, and the latent
    times u_k reproduces the Meta-permuted low-rank K."""
    w = _hf_attn()
    r_k = 8
    lr = convert.LowRank(
        k_VT=torch.randn(NKV, r_k, DIM), k_U=torch.randn(NKV, HD, r_k),
        v_VT=torch.randn(NKV, HD, DIM), v_U=torch.randn(NKV, HD, HD),
    )
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    x = torch.randn(5, DIM)
    lat_k = x @ t["wk_lat_pad"].T
    for h in range(NKV):
        assert torch.allclose(t["wk_lat_pad"][h * HD:h * HD + r_k], lr.k_VT[h], atol=1e-6)
        assert torch.count_nonzero(t["wk_lat_pad"][h * HD + r_k:(h + 1) * HD]) == 0
        w_h = reverse_permute(lr.k_U[h] @ lr.k_VT[h], 1, HD, DIM)          # Meta-order low-rank K weight
        assert torch.allclose(lat_k[:, h * HD:h * HD + r_k] @ t["u_k"][h], x @ w_h.T, atol=1e-4)


def test_lowrank_tensors_shapes_and_padding():
    w = _hf_attn()
    r_k, r_v = 8, 16
    lr = convert.LowRank(
        k_VT=torch.randn(NKV, r_k, DIM), k_U=torch.randn(NKV, HD, r_k),
        v_VT=torch.randn(NKV, r_v, DIM), v_U=torch.randn(NKV, HD, r_v),
    )
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert t["wk_lat_pad"].shape == (NKV * HD, DIM) and t["wv_lat_pad"].shape == (NKV * HD, DIM)
    assert t["u_k"].shape == (NKV, r_k, HD) and t["u_v"].shape == (NKV, r_v, HD)
    assert t["wo_fused"].shape == (DIM, NH * HD)
    # padded rows are zero
    assert torch.count_nonzero(t["wk_lat_pad"][r_k:HD]) == 0
    assert torch.count_nonzero(t["wo_fused"][:, r_v:HD]) == 0


def test_wo_fused_matches_attn_times_v_then_o():
    w = _hf_attn()
    r_v = 16  # full rank so V is exact and the check is tight
    lr = convert.identity_lowrank(w, NKV, HD)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    S, T = 3, 7
    x = torch.randn(T, DIM)
    lat_v = (x @ t["wv_lat_pad"].T).reshape(T, NKV, HD)          # [T, NKV, r_v(=HD)]
    attn = torch.softmax(torch.randn(NH, S, T), dim=-1)
    # Palu fused path: (attn @ latent) padded to HD per head, then wo_fused
    fused_in = torch.zeros(S, NH * HD)
    for h in range(NH):
        fused_in[:, h * HD:h * HD + r_v] = attn[h] @ lat_v[:, h // (NH // NKV), :r_v]
    y_fused = fused_in @ t["wo_fused"].T
    # stock path: reconstruct V, attn @ V, o_proj
    V = (x @ w["v"].T).reshape(T, NKV, HD)
    stock_in = torch.cat([attn[h] @ V[:, h // (NH // NKV)] for h in range(NH)], dim=-1)
    y_stock = stock_in @ w["o"].T
    assert torch.allclose(y_fused, y_stock, atol=1e-4)


def test_load_palu_dir_pads_ranks_to_layer_max(tmp_path):
    import json
    from safetensors.torch import save_file
    r = [8, 4]  # non-uniform across heads -> padded to 8
    tensors = {
        "model.layers.0.self_attn.k_proj.VT.weight": torch.randn(sum(r), DIM),
        "model.layers.0.self_attn.k_proj.U.0.weight": torch.randn(HD, r[0]),
        "model.layers.0.self_attn.k_proj.U.1.weight": torch.randn(HD, r[1]),
        "model.layers.0.self_attn.v_proj.VT.weight": torch.randn(2 * 16, DIM),
        "model.layers.0.self_attn.v_proj.U.0.weight": torch.randn(HD, 16),
        "model.layers.0.self_attn.v_proj.U.1.weight": torch.randn(HD, 16),
    }
    save_file(tensors, str(tmp_path / "palu_attn.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"head_wise_ranks": {
        "model.layers.0.self_attn.k_proj": r, "model.layers.0.self_attn.v_proj": [16, 16]}}))
    lrs = convert.load_palu_dir(tmp_path, n_layers=1, n_kv_heads=NKV, head_dim=HD, rank_multiple=4)
    lr = lrs[0]
    assert lr.k_VT.shape == (NKV, 8, DIM) and lr.k_U.shape == (NKV, HD, 8)
    assert torch.count_nonzero(lr.k_VT[1, 4:]) == 0 and torch.count_nonzero(lr.k_U[1, :, 4:]) == 0
    # head 1 still reconstructs the same product
    x = torch.randn(3, DIM)
    ref = x @ tensors["model.layers.0.self_attn.k_proj.VT.weight"][8:12].T @ tensors["model.layers.0.self_attn.k_proj.U.1.weight"].T
    assert torch.allclose(x @ lr.k_VT[1].T @ lr.k_U[1].T, ref, atol=1e-5)
