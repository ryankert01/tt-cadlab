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
