import json

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
    """r_k == head_dim with a factorisation that is genuinely full rank (identity row, or a Palu
    dump whose every head rank is 128): U_h @ VT_h reproduces the head's K weight, so the K block
    must come out as the stock permuted wk, exactly as convert_hf_qkv_to_meta_format builds it for
    k_proj.weight."""
    w = _hf_attn()
    r_v = 8
    # non-identity but exact: U_h random invertible, VT_h = U_h^-1 @ wk_h  ->  U_h @ VT_h == wk_h
    k_U = torch.randn(NKV, HD, HD).double() + 3 * torch.eye(HD).double()
    wk_heads = w["k"].double().reshape(NKV, HD, DIM)
    k_VT = torch.linalg.solve(k_U, wk_heads)
    lr = convert.LowRank(
        k_VT=k_VT.float(), k_U=k_U.float(),
        v_VT=torch.randn(NKV, r_v, DIM), v_U=torch.randn(NKV, HD, r_v),
    )
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert lr.r_k == HD
    assert torch.allclose(t["wk_lat_pad"], reverse_permute(w["k"], NKV, NKV * HD, DIM), atol=1e-4)
    assert t["u_k"].shape == (NKV, HD, HD)   # still emitted so device shapes stay uniform
    # the V side is untouched by the fast path
    x = torch.randn(5, DIM)
    lat_v = x @ t["wv_lat_pad"].T
    for h in range(NKV):
        assert torch.allclose(lat_v[:, h * HD:h * HD + r_v], x @ lr.v_VT[h].T, atol=1e-5)


def test_padded_full_rank_k_uses_the_factors_not_the_original_wk():
    """r_k == head_dim can also mean "head-wise ranks zero-padded up to head_dim" (load_palu_dir
    rounds the layer max rank up to a multiple of 32, so a dump whose max head rank is in
    (96, 128] lands here with genuinely truncating factors). The K block must then be the
    factorised weight reverse_permute(U_h @ VT_h), never the original wk."""
    w = _hf_attn()
    r_true = 8
    k_VT = torch.zeros(NKV, HD, DIM)
    k_U = torch.zeros(NKV, HD, HD)
    k_VT[:, :r_true] = torch.randn(NKV, r_true, DIM)      # padded to HD columns/rows -> r_k == HD
    k_U[:, :, :r_true] = torch.randn(NKV, HD, r_true)
    lr = convert.LowRank(k_VT=k_VT, k_U=k_U, v_VT=torch.randn(NKV, HD, DIM), v_U=torch.randn(NKV, HD, HD))
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert lr.r_k == HD
    for h in range(NKV):
        blk = t["wk_lat_pad"][h * HD:(h + 1) * HD]
        assert torch.allclose(blk, reverse_permute(k_U[h] @ k_VT[h], 1, HD, DIM), atol=1e-6)
    # ... and emphatically not the uncompressed K weight (that would silently disable compression)
    assert not torch.allclose(t["wk_lat_pad"], reverse_permute(w["k"], NKV, NKV * HD, DIM), atol=1e-3)


def test_full_rank_v_emits_the_product_and_an_identity_u_v():
    """r_v == head_dim is PaluAttention's "fast V" path: prefill feeds the V head-split output
    straight into SDPA and applies the *original* wo, the V cache stores that same tensor and decode
    runs the MLA op on it with head_dim_v == head_dim followed by wo_fused. That is only correct if
    the tensor the fused QKV weight produces really is V, i.e. if U_v == I. A genuine full-rank
    factorisation has U_v != I, so the converter must collapse it: emit the per-head product
    U_v @ VT_v as the V block, an identity u_v, and therefore wo_fused == wo."""
    w = _hf_attn()
    r_k = 8
    # non-identity but exact: U_h random invertible, VT_h = U_h^-1 @ wv_h  ->  U_h @ VT_h == wv_h
    v_U = torch.randn(NKV, HD, HD).double() + 3 * torch.eye(HD).double()
    wv_heads = w["v"].double().reshape(NKV, HD, DIM)
    v_VT = torch.linalg.solve(v_U, wv_heads)
    lr = convert.LowRank(
        k_VT=torch.randn(NKV, r_k, DIM), k_U=torch.randn(NKV, HD, r_k),
        v_VT=v_VT.float(), v_U=v_U.float(),
    )
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert lr.r_v == HD
    # no permutation on the V side: the block is the plain (unpermuted) V weight of the head
    assert torch.allclose(t["wv_lat_pad"], w["v"], atol=1e-4)
    eye = torch.eye(HD).expand(NKV, HD, HD)
    assert torch.allclose(t["u_v"], eye, atol=1e-6)
    assert torch.allclose(t["wo_fused"], w["o"], atol=1e-6)
    # and the fast path is then exactly stock: the head-split output *is* V
    x = torch.randn(5, DIM)
    assert torch.allclose(x @ t["wv_lat_pad"].T, x @ w["v"].T, atol=1e-4)


def test_padded_full_rank_v_uses_the_factors_not_the_original_wv():
    """Same collapse when r_v == head_dim only because head-wise ranks were zero-padded up to it:
    the V block must be the *factorised* weight U_v @ VT_v, never the uncompressed wv."""
    w = _hf_attn()
    r_true = 4
    v_VT = torch.zeros(NKV, HD, DIM)
    v_U = torch.zeros(NKV, HD, HD)
    v_VT[:, :r_true] = torch.randn(NKV, r_true, DIM)
    v_U[:, :, :r_true] = torch.randn(NKV, HD, r_true)
    lr = convert.LowRank(k_VT=torch.randn(NKV, 8, DIM), k_U=torch.randn(NKV, HD, 8), v_VT=v_VT, v_U=v_U)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert lr.r_v == HD
    for h in range(NKV):
        assert torch.allclose(t["wv_lat_pad"][h * HD:(h + 1) * HD], v_U[h] @ v_VT[h], atol=1e-6)
    assert not torch.allclose(t["wv_lat_pad"], w["v"], atol=1e-3)
    assert torch.allclose(t["wo_fused"], w["o"], atol=1e-6)


def test_full_rank_v_from_plain_svd_reproduces_wv():
    """The real producer: compress.svd_per_head at rank == head_dim (U = U_r*sqrt(S),
    VT = sqrt(S)*Vt_r) is exact but has U != I -- the case that made compressed/svd_rk32_rv128
    score PCC 0.05 against the reference before this collapse existed."""
    from palu_tt import compress
    w = _hf_attn()
    v_U, v_VT = compress.svd_per_head(w["v"], NKV, HD)
    lr = convert.LowRank(k_VT=torch.randn(NKV, 8, DIM), k_U=torch.randn(NKV, HD, 8), v_VT=v_VT, v_U=v_U)
    assert lr.r_v == HD
    prod = torch.stack([v_U[h] @ v_VT[h] for h in range(NKV)]).reshape(NKV * HD, DIM)
    assert torch.allclose(prod, w["v"], atol=1e-4)                       # the factorisation is exact
    assert not torch.allclose(v_U, torch.eye(HD).expand(NKV, HD, HD), atol=1e-3)   # ... but U != I
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert torch.allclose(t["wv_lat_pad"], prod, atol=1e-5)
    assert torch.allclose(t["wv_lat_pad"], w["v"], atol=1e-4)
    assert torch.allclose(t["u_v"], torch.eye(HD).expand(NKV, HD, HD), atol=1e-6)
    assert torch.allclose(t["wo_fused"], w["o"], atol=1e-6)


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


def test_wo_fused_matches_attn_times_v_then_o_low_rank():
    """r_v < head_dim keeps the real Palu fusion (u_v = V_U^T, wo_fused = wo @ v_U): the fast-V
    collapse must not touch it. Exact low-rank wv so the stock comparison is tight."""
    w = _hf_attn()
    r_v = 8
    v_VT = torch.randn(NKV, r_v, DIM)
    v_U = torch.randn(NKV, HD, r_v)
    w["v"] = torch.cat([v_U[h] @ v_VT[h] for h in range(NKV)], dim=0)     # wv is exactly the product
    lr = convert.LowRank(k_VT=torch.randn(NKV, HD, DIM), k_U=torch.randn(NKV, HD, HD), v_VT=v_VT, v_U=v_U)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert torch.allclose(t["u_v"], v_U.transpose(1, 2), atol=1e-6)       # untouched by the collapse
    S, T = 3, 7
    x = torch.randn(T, DIM)
    lat_v = (x @ t["wv_lat_pad"].T).reshape(T, NKV, HD)
    attn = torch.softmax(torch.randn(NH, S, T), dim=-1)
    fused_in = torch.zeros(S, NH * HD)
    for h in range(NH):
        fused_in[:, h * HD:h * HD + r_v] = attn[h] @ lat_v[:, h // (NH // NKV), :r_v]
    y_fused = fused_in @ t["wo_fused"].T
    V = (x @ w["v"].T).reshape(T, NKV, HD)
    stock_in = torch.cat([attn[h] @ V[:, h // (NH // NKV)] for h in range(NH)], dim=-1)
    assert torch.allclose(y_fused, stock_in @ w["o"].T, atol=1e-4)


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


# --- weight-set identity in cache names (Task 13) --------------------------------------------------

# load_palu_dir rounds ranks up to a multiple of 32, so these use head_dim 32 (not the module's 16).
HDR = 32


def _tiny_hf_dir(tmp_path, n_layers=1):
    """Minimal HF snapshot: one shard, no index, just the attention projections."""
    from safetensors.torch import save_file
    d = tmp_path / "hf"
    d.mkdir()
    t = {}
    for i in range(n_layers):
        p = f"model.layers.{i}.self_attn."
        t[p + "q_proj.weight"] = torch.randn(NH * HDR, DIM)
        t[p + "k_proj.weight"] = torch.randn(NKV * HDR, DIM)
        t[p + "v_proj.weight"] = torch.randn(NKV * HDR, DIM)
        t[p + "o_proj.weight"] = torch.randn(DIM, NH * HDR)
    save_file(t, str(d / "model.safetensors"))
    return d


def _tiny_palu_dir(path, r=HDR, n_layers=1):
    import json
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    tensors, ranks = {}, {}
    for i in range(n_layers):
        key = f"model.layers.{i}.self_attn."
        for proj in ("k_proj", "v_proj"):
            tensors[key + proj + ".VT.weight"] = torch.randn(NKV * r, DIM)
            for h in range(NKV):
                tensors[key + proj + f".U.{h}.weight"] = torch.randn(HDR, r)
            ranks[key + proj] = [r] * NKV
    save_file(tensors, str(path / "palu_attn.safetensors"))
    (path / "config.json").write_text(json.dumps({"head_wise_ranks": ranks}))
    return path


def test_cache_id_is_identity_without_a_palu_dir(tmp_path):
    cfg = convert.add_palu_to_state_dict({}, _tiny_hf_dir(tmp_path), None, 1, NH, NKV, HDR)
    assert cfg.cache_id == "identity"


def test_cache_id_is_dirname_plus_content_hash(tmp_path):
    import re
    hf_dir = _tiny_hf_dir(tmp_path)
    pd = _tiny_palu_dir(tmp_path / "my_dump_0.5")
    cfg = convert.add_palu_to_state_dict({}, hf_dir, pd, 1, NH, NKV, HDR)
    assert re.fullmatch(r"my_dump_0\.5-[0-9a-f]{8}", cfg.cache_id), cfg.cache_id
    # stable across calls when nothing changed
    again = convert.add_palu_to_state_dict({}, hf_dir, pd, 1, NH, NKV, HDR)
    assert again.cache_id == cfg.cache_id


def test_cache_id_changes_when_a_byte_of_config_json_changes(tmp_path):
    hf_dir = _tiny_hf_dir(tmp_path)
    pd = _tiny_palu_dir(tmp_path / "dump")
    before = convert.add_palu_to_state_dict({}, hf_dir, pd, 1, NH, NKV, HDR).cache_id
    cj = pd / "config.json"
    cj.write_text(cj.read_text() + " ")          # one byte, still valid JSON
    after = convert.add_palu_to_state_dict({}, hf_dir, pd, 1, NH, NKV, HDR).cache_id
    assert after != before, f"{after} == {before}"
    assert after.startswith("dump-")


def test_cache_id_differs_between_two_dumps_with_the_same_ranks(tmp_path):
    hf_dir = _tiny_hf_dir(tmp_path)
    a = convert.add_palu_to_state_dict({}, hf_dir, _tiny_palu_dir(tmp_path / "a"), 1, NH, NKV, HDR)
    b = convert.add_palu_to_state_dict({}, hf_dir, _tiny_palu_dir(tmp_path / "b"), 1, NH, NKV, HDR)
    assert a.rank_k == b.rank_k and a.rank_v == b.rank_v
    assert a.cache_id != b.cache_id


def test_layer_tensors_include_w_rot_built_from_u_k():
    from palu_tt import fused
    w = _hf_attn()
    lr = convert.identity_lowrank(w, NKV, HD)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert t["w_rot"].shape == (NKV, HD, 2 * HD)
    torch.testing.assert_close(t["w_rot"], fused.w_rot_from_u_k(t["u_k"]))


def test_convert_version_is_4_and_runtime_flags_default_off():
    assert convert.CONVERT_VERSION == 4
    rt = convert.PaluRuntime()
    assert rt.use_fused_k is False and rt.cs_recurrence is False
    assert rt.fused_k_chunk is None
    cfg = convert.PaluConfig(rank_k=[32], rank_v=[64], cache_id="x")
    assert cfg.device_tables == {}


def _group_lowrank(w, gs, r):
    """A synthetic group factorisation: one U_g [gs*HD, r] and VT_g [r, DIM] per group of gs KV heads."""
    n_groups = NKV // gs
    VT = torch.randn(n_groups, r, DIM)
    U = torch.randn(n_groups, gs * HD, r)
    return VT, U


def _write_group_dump(tmp_path, VT, U, gs):
    """Palu's on-disk layout for head_group_size gs: U.{g}.weight [gs*hd, r_g], one stacked VT.weight, one rank per group."""
    from safetensors.torch import save_file
    d = tmp_path / "glrd"; d.mkdir()
    n_groups, r, _ = VT.shape
    tensors, ranks = {}, {}
    for proj, (vt, u) in {"k_proj": (VT, U), "v_proj": (VT * 0.5, U * 2.0)}.items():
        key = f"model.layers.0.self_attn.{proj}"
        tensors[f"{key}.VT.weight"] = vt.reshape(n_groups * r, DIM).contiguous()
        for g in range(n_groups):
            tensors[f"{key}.U.{g}.weight"] = u[g].contiguous()
        ranks[key] = [r] * n_groups
    save_file(tensors, str(d / "palu_lowrank.safetensors"))
    (d / "config.json").write_text(json.dumps({"head_wise_ranks": ranks}))
    return d


def test_load_palu_dir_expands_groups_to_heads(tmp_path):
    gs, r = 2, 8                                   # NKV = 2 -> one group of two heads
    VT, U = _group_lowrank(_hf_attn(), gs, r)
    d = _write_group_dump(tmp_path, VT, U, gs)
    lr = convert.load_palu_dir(d, 1, NKV, HD, rank_multiple=8)[0]
    assert lr.group_size == gs and lr.r_k == r
    x = torch.randn(5, DIM)
    for h in range(NKV):
        g, hh = h // gs, h % gs
        assert torch.equal(lr.k_VT[h], VT[g])                          # the group's VT, shared by its heads
        assert torch.equal(lr.k_U[h], U[g, hh * HD:(hh + 1) * HD])      # the head's slice of the group's U
        k_h = x @ VT[g].T @ U[g, hh * HD:(hh + 1) * HD].T             # what Palu computes for head h
        assert torch.allclose(x @ lr.k_VT[h].T @ lr.k_U[h].T, k_h, atol=1e-5)
        assert torch.allclose(lr.v_U[h], 2.0 * U[g, hh * HD:(hh + 1) * HD]) and torch.allclose(lr.v_VT[h], 0.5 * VT[g])


def test_group_size_one_dump_still_loads_as_before(tmp_path):
    d = _tiny_palu_dir(tmp_path / "p", r=HDR)
    lr = convert.load_palu_dir(d, 1, NKV, HDR)[0]
    assert lr.group_size == 1


def test_layer_tensors_lay_the_group_latent_into_the_groups_head_slots():
    gs, r = 2, 8
    w = _hf_attn()
    VT, U = _group_lowrank(w, gs, r)
    heads = []
    for h in range(NKV):
        g, hh = h // gs, h % gs
        heads.append((VT[g], U[g, hh * HD:(hh + 1) * HD]))
    lr = convert.LowRank(k_VT=torch.stack([a for a, _ in heads]), k_U=torch.stack([b for _, b in heads]),
                         v_VT=torch.stack([a for a, _ in heads]), v_U=torch.stack([b for _, b in heads]), group_size=gs)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    # K block: group g's VT sits at rows [g*gs*HD, g*gs*HD + r), the rest of the group's rows are zero
    assert torch.equal(t["wk_lat_pad"][:r], VT[0]) and torch.count_nonzero(t["wk_lat_pad"][r:gs * HD]) == 0
    assert torch.equal(t["wv_lat_pad"][:r], VT[0])
    # per-head up-projections carry the group rank, in Meta order for K
    assert t["u_k"].shape == (NKV, r, HD) and t["u_v"].shape == (NKV, r, HD) and t["w_rot"].shape == (NKV, r, 2 * HD)
    lat = torch.randn(3, r)
    for h in range(NKV):
        hh = h % gs
        v_h = lat @ U[0, hh * HD:(hh + 1) * HD].T
        assert torch.allclose(lat @ t["u_v"][h], v_h, atol=1e-5)
        k_h_meta = reverse_permute(U[0, hh * HD:(hh + 1) * HD], 1, HD, r).T
        assert torch.allclose(t["u_k"][h], k_h_meta, atol=1e-6)
    # grouped rows never absorb V: wo_fused is wo
    assert torch.equal(t["wo_fused"], w["o"].float())


def test_convert_version_is_4_and_group_size_default_is_1():
    assert convert.CONVERT_VERSION == 4
    cfg = convert.PaluConfig(rank_k=[32], rank_v=[64], cache_id="x")
    assert cfg.group_size == 1


def test_runtime_has_stage2e_switches_off_by_default():
    from palu_tt.convert import PaluRuntime
    rt = PaluRuntime()
    assert rt.rot_mode == "legacy"
    assert rt.table_mcast is False
