"""Op-level tests for ttnn.transformer.palu_sdpa_decode (stage 2 fused K-reconstruction decode).

Runs on the N300 mesh fixture; every tensor is replicated to both chips and chip 0 is read back.
References: (a) palu_tt.fused.reference_decode in fp32 on bf16-rounded inputs, (b) the stock MLA
flash-decode op fed the host-reconstructed, host-rotated K -- (b) isolates kernel error from the
bf16 rounding both paths share."""
import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

from palu_tt import fused

HD, NH, NKV = 128, 16, 4
THETA = 500000.0


def _cfg(k_chunk):
    return dict(
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=k_chunk,
                                              exp_approx_mode=False),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                               fp32_dest_acc_en=True, packer_l1_acc=False),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dev(md, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, device=md, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(md))


def _pos(md, pos):
    return ttnn.from_torch(pos.to(torch.int32), device=md, dtype=ttnn.int32, mesh_mapper=ttnn.ReplicateTensorToMesh(md))


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _bf(t):
    return t.to(torch.bfloat16).float()


def _inputs(batch, S, r_k, r_v):
    q = _bf(torch.randn(1, batch, NH, HD))
    k_lat = _bf(torch.randn(batch, NKV, S, r_k))
    u_k = _bf(torch.randn(NKV, r_k, HD) / r_k ** 0.5)
    v = _bf(torch.randn(batch, NKV, S, r_v))
    pos = torch.randint(0, S, (batch,))
    pos[0] = S - 1                                   # full cache
    if batch >= 4:
        pos[1], pos[2], pos[3] = 0, 127, 128         # first row only; chunk boundary both sides
    cos, sin = fused.cos_sin_tables(HD, S, THETA, None)
    return q, k_lat, u_k, v, pos, _bf(cos), _bf(sin)


def _stock_on_reconstructed_k(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk):
    S = k_lat.shape[2]
    K = torch.einsum("bhsr,hrd->bhsd", k_lat, u_k)
    K = _bf(K * cos[None, None, :S] + (K @ fused.rope_transformation_matrix(HD)) * sin[None, None, :S])
    out = ttnn.transformer.flash_multi_latent_attention_decode(
        _dev(md, q), _dev(md, K), _dev(md, v), head_dim_v=r_v, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5,
        **_cfg(k_chunk))
    return _host(out)


def _fused(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk, phase2):
    S = k_lat.shape[2]
    extra = {}
    if phase2:
        c = k_chunk or fused.chunk_size(S)
        extra = dict(rot_step=_dev(md, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                     trans_mat=_dev(md, fused.trans_mat_tile()))
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(md, q), _dev(md, k_lat), _dev(md, fused.w_rot_from_u_k(u_k)[None]), _dev(md, fused.cs_table(cos, sin, S)[None, None]),
        _dev(md, v), head_dim_v=r_v, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5, **extra, **_cfg(k_chunk))
    return _host(out)


def _check(md, batch, S, r_k, r_v, k_chunk, phase2, pcc_ref=0.99, pcc_stock=0.995):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(batch, S, r_k, r_v)
    ref = fused.reference_decode(q, k_lat, u_k, v, pos, cos, sin, HD ** -0.5)
    out = _fused(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk, phase2)
    assert out.shape == (1, batch, NH, r_v)
    ok, msg = comp_pcc(ref, out, pcc_ref)
    assert ok, f"vs fp32 reference: {msg}"
    stock = _stock_on_reconstructed_k(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk)
    ok, msg = comp_pcc(stock, out, pcc_stock)
    assert ok, f"vs stock MLA on host-rotated K: {msg}"


@pytest.mark.parametrize("r_k", [32, 64, 96])
@pytest.mark.parametrize("batch", [1, 32])
def test_phase1_matches_reference(mesh_device, r_k, batch):
    # Phase 1 streams a [c, 256] cos/sin block per chunk; at the op's default chunk (512) that CB is
    # 512 KB and does not fit next to the stock CBs, so Phase 1 always runs with k_chunk_size=128.
    _check(mesh_device, batch, 1024, r_k, 64, k_chunk=128, phase2=False)


def test_phase1_r_v_full_width(mesh_device):
    _check(mesh_device, 32, 1024, 32, 128, k_chunk=128, phase2=False)


def test_envelope_is_enforced(mesh_device):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(1, 256, 64, 64)
    good = dict(head_dim_v=64, cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5, **_cfg(128))
    w = _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None])
    cs = _dev(mesh_device, fused.cs_table(cos, sin, 256)[None, None])
    with pytest.raises(RuntimeError, match="r_k"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, torch.randn(1, NKV, 256, 48)), w, cs,
                                          _dev(mesh_device, v), **good)
    with pytest.raises(RuntimeError, match="cs_tab"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w,
                                          _dev(mesh_device, fused.cs_table(cos, sin, 128)[None, None]), _dev(mesh_device, v), **good)
    with pytest.raises(RuntimeError, match="w_rot"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat),
                                          _dev(mesh_device, u_k[None]), cs, _dev(mesh_device, v), **good)
    with pytest.raises(RuntimeError, match="optional together"):     # trans_mat without rot_step
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs, _dev(mesh_device, v),
                                          trans_mat=_dev(mesh_device, fused.trans_mat_tile()), **good)
    with pytest.raises(RuntimeError, match="Phase 1"):               # streamed table needs the small chunk
        bad = dict(good, **_cfg(256))
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs,
                                          _dev(mesh_device, v), **bad)


@pytest.mark.parametrize("r_k", [32, 64, 96])
@pytest.mark.parametrize("batch", [1, 32])
def test_phase2_matches_reference_default_chunk(mesh_device, r_k, batch):
    _check(mesh_device, batch, 1024, r_k, 64, k_chunk=0, phase2=True)      # the op picks 512


@pytest.mark.parametrize("k_chunk", [128, 512])
def test_phase2_chunk_override(mesh_device, k_chunk):
    _check(mesh_device, 32, 1024, 64, 64, k_chunk=k_chunk, phase2=True)


@pytest.mark.parametrize("S", [4096, 8192])
def test_phase2_long_context(mesh_device, S):
    # batch 1 -> 16 cores per KV head -> every core starts at a different chunk (k_chunk_start steps)
    _check(mesh_device, 1, S, 64, 64, k_chunk=0, phase2=True)


def test_phase2_long_context_full_batch(mesh_device):
    _check(mesh_device, 32, 8192, 32, 64, k_chunk=0, phase2=True)


def test_phase1_and_phase2_agree(mesh_device):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 1024, 64, 64)
    p1 = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 64, 128, phase2=False)
    p2 = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 64, 128, phase2=True)
    ok, msg = comp_pcc(p1, p2, 0.998)
    assert ok, msg


# ---------------- Stage 2b: grouped latents (G-LRD) and in-kernel V reconstruction ----------------


def _inputs_group(batch, S, r_k, r_v, gs):
    """Grouped latents: n_groups = NKV // gs share one K latent and one V latent; per-head factors of width r_g."""
    ng = NKV // gs
    q = _bf(torch.randn(1, batch, NH, HD))
    k_lat = _bf(torch.randn(batch, ng, S, r_k))
    u_k = _bf(torch.randn(NKV, r_k, HD) / r_k ** 0.5)
    v_lat = _bf(torch.randn(batch, ng, S, r_v))
    u_v = _bf(torch.randn(NKV, r_v, HD) / r_v ** 0.5)
    pos = torch.randint(0, S, (batch,))
    pos[0] = S - 1
    if batch >= 4:
        pos[1], pos[2], pos[3] = 0, 127, 128
    cos, sin = fused.cos_sin_tables(HD, S, THETA, None)
    return q, k_lat, u_k, v_lat, u_v, pos, _bf(cos), _bf(sin)


def _expand(lat, gs):
    return lat.repeat_interleave(gs, dim=1)


def _stock_on_reconstructed_kv(md, q, k_full_rot, v_full, pos, k_chunk):
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        _dev(md, q), _dev(md, k_full_rot), _dev(md, v_full), cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5, **_cfg(k_chunk))
    return _host(out)


def _fused_group(md, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, k_chunk, phase2):
    S = k_lat.shape[2]
    extra = dict(u_v=_dev(md, u_v[None]))
    if phase2:
        # u_v is always given here, so the op's default chunk is the in-kernel-V rule's 128.
        c = k_chunk or 128
        extra.update(rot_step=_dev(md, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                     trans_mat=_dev(md, fused.trans_mat_tile()))
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(md, q), _dev(md, k_lat), _dev(md, fused.w_rot_from_u_k(u_k)[None]), _dev(md, fused.cs_table(cos, sin, S)[None, None]),
        _dev(md, v_lat), head_dim_v=HD, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5, **extra, **_cfg(k_chunk))
    return _host(out)


def _check_group(md, batch, S, r_k, r_v, gs, k_chunk, phase2, pcc_ref=0.99, pcc_stock=0.995):
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(batch, S, r_k, r_v, gs)
    k_h, v_h = _expand(k_lat, gs), _expand(v_lat, gs)
    ref = fused.reference_decode(q, k_h, u_k, v_h, pos, cos, sin, HD ** -0.5, u_v=u_v)
    out = _fused_group(md, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, k_chunk, phase2)
    assert out.shape == (1, batch, NH, HD)
    ok, msg = comp_pcc(ref, out, pcc_ref)
    assert ok, f"vs fp32 reference: {msg}"
    K = torch.einsum("bhsr,hrd->bhsd", k_h, u_k)
    K = _bf(K * cos[None, None, :S] + (K @ fused.rope_transformation_matrix(HD)) * sin[None, None, :S])
    V = _bf(torch.einsum("bhsr,hrd->bhsd", v_h, u_v))
    stock = _stock_on_reconstructed_kv(md, q, K, V, pos, k_chunk)
    ok, msg = comp_pcc(stock, out, pcc_stock)
    assert ok, f"vs stock decode on host-rebuilt K and V: {msg}"


@pytest.mark.parametrize("r_k,r_v", [(128, 256), (256, 512), (384, 512)])
@pytest.mark.parametrize("batch", [1, 32])
def test_grouped_latents_with_in_kernel_v(mesh_device, r_k, r_v, batch):
    _check_group(mesh_device, batch, 1024, r_k, r_v, gs=4, k_chunk=0, phase2=True)   # host default chunk = 128 here


def test_grouped_latents_phase1_and_phase2_agree(mesh_device):
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(32, 1024, 128, 256, 4)
    p1 = _fused_group(mesh_device, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, 128, phase2=False)
    p2 = _fused_group(mesh_device, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, 128, phase2=True)
    ok, msg = comp_pcc(p1, p2, 0.998)
    assert ok, msg


def test_grouped_latents_long_context_batch8(mesh_device):
    _check_group(mesh_device, 8, 8192, 64, 512, gs=4, k_chunk=0, phase2=True)


def test_in_kernel_v_with_per_head_latents(mesh_device):
    """u_v without grouping: the V mode alone, heads_per_group = 1."""
    _check_group(mesh_device, 32, 1024, 64, 64, gs=1, k_chunk=0, phase2=True)


def test_group_envelope_is_enforced(mesh_device):
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(1, 256, 128, 256, 4)
    w = _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None])
    cs = _dev(mesh_device, fused.cs_table(cos, sin, 256)[None, None])
    good = dict(head_dim_v=HD, cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5, **_cfg(128))
    with pytest.raises(RuntimeError, match="heads_per_group"):      # 3 latent heads do not divide 4 KV heads
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, torch.randn(1, 3, 256, 128)), w, cs,
                                          _dev(mesh_device, v_lat), u_v=_dev(mesh_device, u_v[None]), **good)
    with pytest.raises(RuntimeError, match="u_v"):                  # grouped V latent without u_v
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs, _dev(mesh_device, v_lat), **good)
    with pytest.raises(RuntimeError, match="head_dim_v"):           # u_v given but head_dim_v is not 128
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs, _dev(mesh_device, v_lat),
                                          u_v=_dev(mesh_device, u_v[None]), **{**good, "head_dim_v": 256})


def test_grouped_latents_full_rank_group(mesh_device):
    """glrd_0.7 has layers at K group rank 512 and V group rank 512: the widest latents the envelope allows."""
    _check_group(mesh_device, 32, 1024, 512, 512, gs=4, k_chunk=0, phase2=True)
