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


def _cfg(k_chunk, grid=(8, 8)):
    return dict(
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=k_chunk,
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


# Sentinel for "do not pass this keyword at all", so a test can exercise the op's *own* default
# instead of whatever these helpers happen to pass (stage 2e Task 9 made the defaults dst + multicast).
_OP_DEFAULT = object()


def _switches(rot_mode, table_mcast):
    return {k: v for k, v in (("rot_mode", rot_mode), ("table_mcast", table_mcast)) if v is not _OP_DEFAULT}


def _fused(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk, phase2, rot_mode="legacy", table_mcast=False):
    S = k_lat.shape[2]
    extra = {}
    if phase2:
        c = k_chunk or fused.chunk_size(S)
        extra = dict(rot_step=_dev(md, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                     trans_mat=_dev(md, fused.trans_mat_tile()))
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(md, q), _dev(md, k_lat), _dev(md, fused.w_rot_from_u_k(u_k)[None]), _dev(md, fused.cs_table(cos, sin, S)[None, None]),
        _dev(md, v), head_dim_v=r_v, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5,
        **_switches(rot_mode, table_mcast), **extra, **_cfg(k_chunk))
    return _host(out)


def _check(md, batch, S, r_k, r_v, k_chunk, phase2, rot_mode="legacy", pcc_ref=0.99, pcc_stock=0.995):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(batch, S, r_k, r_v)
    ref = fused.reference_decode(q, k_lat, u_k, v, pos, cos, sin, HD ** -0.5)
    out = _fused(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk, phase2, rot_mode=rot_mode)
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


ROT_MODES = ["legacy", "pack", "dst"]


@pytest.mark.parametrize("rot_mode", ROT_MODES)
@pytest.mark.parametrize("r_k", [32, 64, 96])
@pytest.mark.parametrize("batch", [1, 32])
def test_phase2_matches_reference_default_chunk(mesh_device, r_k, batch, rot_mode):
    _check(mesh_device, batch, 1024, r_k, 64, k_chunk=0, phase2=True, rot_mode=rot_mode)      # the op picks 512


def test_rot_mode_dst_and_table_mcast_are_the_defaults_and_bogus_is_rejected(mesh_device):
    """Stage 2e Task 9 ships `dst` + table multicast as the op's own defaults, so a Phase 2 call that
    names neither must be exactly the call that names both. Phase 1 has no resident angle table to
    broadcast, so the shipped multicast default has to fold away there rather than be rejected."""
    q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 1024, 32, 128)
    default = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True,
                     rot_mode=_OP_DEFAULT, table_mcast=_OP_DEFAULT)
    shipped = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode="dst", table_mcast=True)
    assert torch.equal(default, shipped)
    p1_default = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 128, False,
                        rot_mode=_OP_DEFAULT, table_mcast=_OP_DEFAULT)
    p1_explicit = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 128, False,
                         rot_mode="dst", table_mcast=False)
    assert torch.equal(p1_default, p1_explicit)
    with pytest.raises(RuntimeError, match="rot_mode must be"):
        _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode="bogus")


@pytest.mark.parametrize("rot_mode", ["pack", "dst"])
def test_rot_mode_tracks_legacy_closely(mesh_device, rot_mode):
    """The new modes round K_rot twice (bf16 pack, then the accumulating pack); against legacy's single
    rounding the outputs must still agree to PCC 0.999 at every calibrated rank."""
    for r_k in (32, 64, 96):
        q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 2048, r_k, 128)
        a = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode="legacy")
        b = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode=rot_mode)
        ok, msg = comp_pcc(a, b, 0.999)
        assert ok, f"r_k={r_k} {rot_mode} vs legacy: {msg}"


@pytest.mark.parametrize("k_chunk", [128, 512])
def test_phase2_chunk_override(mesh_device, k_chunk):
    _check(mesh_device, 32, 1024, 64, 64, k_chunk=k_chunk, phase2=True)


@pytest.mark.parametrize("rot_mode", ROT_MODES)
@pytest.mark.parametrize("S", [4096, 8192])
def test_phase2_long_context(mesh_device, S, rot_mode):
    # batch 1 -> 16 cores per KV head -> every core starts at a different chunk (k_chunk_start steps)
    _check(mesh_device, 1, S, 64, 64, k_chunk=0, phase2=True, rot_mode=rot_mode)


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


def _fused_group(md, q, k_lat, u_k, v_lat, u_v, pos, cos, sin, k_chunk, phase2, heads_per_core=None, grid=(8, 8),
                  rot_mode="legacy", table_mcast=False):
    S = k_lat.shape[2]
    extra = dict(u_v=_dev(md, u_v[None]))
    if phase2:
        # u_v is always given here, so the op's default chunk is the in-kernel-V rule's 128.
        c = k_chunk or 128
        extra.update(rot_step=_dev(md, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                     trans_mat=_dev(md, fused.trans_mat_tile()))
    if heads_per_core is not None:
        extra["heads_per_core"] = heads_per_core
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(md, q), _dev(md, k_lat), _dev(md, fused.w_rot_from_u_k(u_k)[None]), _dev(md, fused.cs_table(cos, sin, S)[None, None]),
        _dev(md, v_lat), head_dim_v=HD, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5,
        **_switches(rot_mode, table_mcast), **extra, **_cfg(k_chunk, grid))
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
    # r_v = 128 = head_dim_v so the "V latent width == head_dim_v" absorbed-V check does not fire before
    # the "grouped latents need u_v" check the second case below is meant to exercise.
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(1, 256, 128, 128, 4)
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


# ---------------- Stage 2c: group-shared mode (a core's heads share one latent read) ----------------


def _group_ref_pcc(q, k_lat, u_k, v_lat, u_v, pos, cos, sin, out, gs=4, pcc=0.99):
    ref = fused.reference_decode(q, _expand(k_lat, gs), u_k, _expand(v_lat, gs), pos, cos, sin, HD ** -0.5, u_v=u_v)
    return comp_pcc(ref, out, pcc)


# The bit-identity bar cannot apply to any comparison in which a child is merged. The group-shared
# branch does the softmax correction with the plain max / sub_exp / mul / add primitives instead of the
# fused correction_block the per-head branch uses -- correction_block needs five dst tiles and this
# program's window is four under fp32 dest accumulate, and in the group branch that overflow drops the
# child's contribution outright (see the compute kernel for the measurements). The primitives are the
# same values but *not* the same precision: correction_block keeps its six steps in fp32 dst across one
# acquisition, the primitives pack to bf16 after each, so six extra quantisations per child merge. On
# top of that, an H that changes the work split also changes the order the chunks are accumulated in.
# Either alone puts the two outputs a few bf16 ulps apart, so those comparisons are PCC (spec: 0.9999).
SPLIT_PCC = 0.9999
# The same cases against the fp32 reference: 0.99996-0.99998 measured, so 0.9995 catches a regression
# in the merge while leaving room for the bf16 noise.
SPLIT_REF_PCC = 0.9995


def _assert_matches(per_head, shared, msg, bitwise):
    if bitwise:
        assert torch.equal(per_head, shared), f"{msg}: max |diff| = {(per_head - shared).abs().max()}"
    else:
        ok, detail = comp_pcc(per_head, shared, SPLIT_PCC)
        assert ok, f"{msg}: {detail}"


@pytest.mark.parametrize("r_k,r_v", [(64, 512), (128, 256), (288, 512)], ids=["rk64_rv512", "rk128_rv256", "rk288_rv512"])
@pytest.mark.parametrize("phase2", [False, True], ids=["p1", "p2"])
def test_group_shared_matches_per_head(mesh_device, r_k, r_v, phase2):
    """Group-shared mode against the per-head mode at batch 16.

    A child is merged here, so the bar is PCC: the group branch's correction is the primitive sequence,
    not the fused correction_block, and it packs to bf16 six more times per merge (see SPLIT_PCC).
    heads_per_core=1 is the stock split at batch 16 -- one (batch, KV head) item per core, 64 cores,
    the whole sequence on each -- while heads_per_core=2 halves the item count and gives each item two
    cores, so the chunk order differs too. test_group_shared_batch32_matches_stock_split is the
    bit-identical comparison: same layout, and no reduction to merge."""
    args = _inputs_group(16, 2048, r_k, r_v, 4)
    per_head = _fused_group(mesh_device, *args, 128, phase2, heads_per_core=1)
    shared = _fused_group(mesh_device, *args, 128, phase2, heads_per_core=2)
    _assert_matches(per_head, shared, "group-shared vs per-head", bitwise=False)
    ok, msg = _group_ref_pcc(*args, shared, pcc=SPLIT_REF_PCC)
    assert ok, f"vs fp32 reference: {msg}"


def test_group_shared_whole_group_on_one_core(mesh_device):
    """H = 4 with the whole sequence on the core (Phase A): the state CBs hold four blocks and the FIFO
    rotation has to hand each head its own block back three extra times per chunk. No child is merged,
    so the correction never runs and the output must be bit-identical.

    A 4x4 grid at batch 16 is the layout that gives one core per (batch, group of 4 heads) and no
    sequence split: 16 cores, one item each. On the full 8x8 grid the same H would additionally split
    the sequence over 4 cores, which is the Phase B path that
    test_group_shared_with_sequence_split_matches_per_head covers."""
    args = _inputs_group(16, 2048, 128, 256, 4)
    per_head = _fused_group(mesh_device, *args, 128, True, heads_per_core=1, grid=(4, 4))
    shared = _fused_group(mesh_device, *args, 128, True, heads_per_core=4, grid=(4, 4))
    _assert_matches(per_head, shared, "H=4 vs per-head", bitwise=True)
    ok, msg = _group_ref_pcc(*args, shared)
    assert ok, f"vs fp32 reference: {msg}"


def test_group_shared_envelope_is_enforced(mesh_device):
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(1, 256, 128, 128, 4)
    w = _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None])
    cs = _dev(mesh_device, fused.cs_table(cos, sin, 256)[None, None])
    good = dict(head_dim_v=HD, cur_pos_tensor=_pos(mesh_device, pos), u_v=_dev(mesh_device, u_v[None]),
                scale=HD ** -0.5, **_cfg(128))
    with pytest.raises(RuntimeError, match="heads_per_core"):      # 3 does not divide heads_per_group 4
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs,
                                          _dev(mesh_device, v_lat), heads_per_core=3, **good)


# ---------------- Stage 2c Phase B: H heads per core with the sequence split over cores ----------------


@pytest.mark.parametrize("batch,H", [(8, 4), (16, 4), (32, 4), (8, 2)], ids=["b8_h4", "b16_h4", "b32_h4", "b8_h2"])
def test_group_shared_with_sequence_split_matches_per_head(mesh_device, batch, H):
    """Phase B: a core owns H heads *and* only part of the sequence, so the tree reduction runs once
    per head, against ``heads_per_core=None`` -- the stock split with the per-head kernel.

    PCC, not bit identity, for two independent reasons: the group branch's softmax correction is the
    primitive sequence rather than the fused correction_block (six extra bf16 quantisations per child
    merge -- see SPLIT_PCC), and H also changes the work split, so the chunks are accumulated in a
    different order.

    The layouts on the 64-core grid (4 KV heads, one group; the factory caps a group-mode split at 4
    cores per item): batch 32 / H=4 is 32 items over 2 cores each, batch 16 / H=4 is 16 items over 4,
    batch 8 / H=4 is 8 items over 4 (32 cores busy -- the cap costs the other half of the grid there),
    batch 8 / H=2 is 16 items over 4. All four exercise the per-head tree reduction."""
    args = _inputs_group(batch, 4096, 128, 256, 4)
    per_head = _fused_group(mesh_device, *args, 128, True, heads_per_core=None)
    shared = _fused_group(mesh_device, *args, 128, True, heads_per_core=H)
    _assert_matches(per_head, shared, f"batch={batch} H={H}: group-shared vs the stock split", bitwise=False)
    ok, msg = _group_ref_pcc(*args, shared, pcc=SPLIT_REF_PCC)
    assert ok, f"vs fp32 reference: {msg}"


@pytest.mark.parametrize("rot_mode", ROT_MODES)
@pytest.mark.parametrize("r_k,r_v", [(64, 512), (128, 256), (288, 512)], ids=["rk64_rv512", "rk128_rv256", "rk288_rv512"])
@pytest.mark.parametrize("phase2", [False, True], ids=["p1", "p2"])
def test_group_shared_batch32_matches_stock_split(mesh_device, r_k, r_v, phase2, rot_mode):
    """``heads_per_core=None`` at batch 32 is the stock split -- two heads per core, one core per head,
    per-head kernel. ``heads_per_core=2`` is the group kernel on exactly that layout -- two heads per
    core, one core per head, no sequence split -- so this isolates the shared latent read from any
    change of work assignment. No core has a child here, so the branch that differs in precision (the
    softmax correction) never runs and the bit-identity bar applies."""
    args = _inputs_group(32, 2048, r_k, r_v, 4)
    stock = _fused_group(mesh_device, *args, 128, phase2, heads_per_core=None, rot_mode=rot_mode)
    shared = _fused_group(mesh_device, *args, 128, phase2, heads_per_core=2, rot_mode=rot_mode)
    _assert_matches(stock, shared, "group-shared vs the stock split", bitwise=True)
    ok, msg = _group_ref_pcc(*args, shared)
    assert ok, f"vs fp32 reference: {msg}"


@pytest.mark.parametrize("rot_mode", ROT_MODES)
def test_heads_per_core_1_is_the_stock_split(mesh_device, rot_mode):
    """heads_per_core=1 must mean 'per-head kernel, stock work split' -- not 'pin one head per core'.
    At batch 32 the stock split is two heads per core, so pinning 1 would ask for 128 items on the
    64-core grid; the op must instead build the same program as heads_per_core=None."""
    args = _inputs_group(32, 1024, 128, 256, 4)
    unset = _fused_group(mesh_device, *args, 128, True, heads_per_core=None, rot_mode=rot_mode)
    one = _fused_group(mesh_device, *args, 128, True, heads_per_core=1, rot_mode=rot_mode)
    assert torch.equal(unset, one)


def test_dst_mode_three_inner_tiles_group(mesh_device):
    """r_k 96 (three latent tiles) in group-shared mode: the two matmul_block loops per trip run three
    accumulating steps each, with the per-head weight-block base applied to both halves."""
    args = _inputs_group(32, 2048, 96, 256, 4)
    per_head = _fused_group(mesh_device, *args, 128, True, heads_per_core=None, rot_mode="dst")
    shared = _fused_group(mesh_device, *args, 128, True, heads_per_core=2, rot_mode="dst")
    _assert_matches(per_head, shared, "dst: group-shared vs the stock split", bitwise=True)
    ok, msg = _group_ref_pcc(*args, shared)
    assert ok, f"vs fp32 reference: {msg}"


def test_group_shared_phase_b_envelope_is_enforced(mesh_device):
    """The validation messages Phase B makes reachable."""
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(1, 256, 128, 128, 4)
    w = _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None])
    cs = _dev(mesh_device, fused.cs_table(cos, sin, 256)[None, None])
    good = dict(head_dim_v=HD, cur_pos_tensor=_pos(mesh_device, pos), u_v=_dev(mesh_device, u_v[None]),
                scale=HD ** -0.5, **_cfg(128))
    with pytest.raises(RuntimeError, match="KV head count"):        # 8 > the 4 KV heads
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs,
                                          _dev(mesh_device, v_lat), heads_per_core=8, **good)
    # heads_per_group 2 (2 latent heads, 4 KV heads): 4 divides the KV heads but not the group
    k_lat2 = _bf(torch.randn(1, 2, 256, 128))
    v_lat2 = _bf(torch.randn(1, 2, 256, 128))
    with pytest.raises(RuntimeError, match="heads_per_group"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat2), w, cs,
                                          _dev(mesh_device, v_lat2), heads_per_core=4, **good)
    # per-head latents: sharing a latent between heads is meaningless
    k_lat1 = _bf(torch.randn(1, NKV, 256, 128))
    v_lat1 = _bf(torch.randn(1, NKV, 256, 128))
    with pytest.raises(RuntimeError, match="grouped latents"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat1), w, cs,
                                          _dev(mesh_device, v_lat1), heads_per_core=2, **good)


def test_pinned_heads_per_core_that_does_not_fit_the_grid_is_rejected(mesh_device):
    """The core-count fatal still fires for a layout that genuinely does not fit: batch 4 on a 2x2
    grid is one core per batch, and heads_per_core=2 wants two items on it."""
    q, k_lat, u_k, v_lat, u_v, pos, cos, sin = _inputs_group(4, 256, 128, 128, 4)
    tiny = dict(_cfg(128))
    tiny["program_config"] = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(2, 2), q_chunk_size=0,
                                                    k_chunk_size=128, exp_approx_mode=False)
    good = dict(head_dim_v=HD, cur_pos_tensor=_pos(mesh_device, pos), u_v=_dev(mesh_device, u_v[None]),
                scale=HD ** -0.5, **tiny)
    w = _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None])
    cs = _dev(mesh_device, fused.cs_table(cos, sin, 256)[None, None])
    with pytest.raises(RuntimeError, match="only 4 available"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w, cs,
                                          _dev(mesh_device, v_lat), heads_per_core=2, **good)


def test_per_head_reduction_is_accurate_under_fp32_dest_acc(mesh_device):
    """The per-head branch keeps the fused ``correction_block``, which stages five operands through dst
    registers 0..4 while this program's window is four tiles (``fp32_dest_acc_en`` is on, and Phase 2
    requires it). That overflow is what corrupts the merged max in the group branch. This pins the
    per-head branch, which is the stage 2b decode path: batch 8 is two cores per (batch, KV head), so
    ``correction_block`` runs, and the result has to stay tight against the fp32 reference.

    Measured 0.99998 here (and 0.99935 at batch 1, where the tree is three rounds deep and the bf16
    merge error genuinely accumulates), against 0.334 for the same helper in the group branch."""
    args = _inputs_group(8, 4096, 128, 256, 4)
    out = _fused_group(mesh_device, *args, 128, True, heads_per_core=None)
    ok, msg = _group_ref_pcc(*args, out, pcc=0.9999)
    assert ok, f"per-head tree reduction under fp32 dest accumulate: {msg}"


@pytest.mark.parametrize("batch,S,grid", [(32, 2048, (8, 8)), (8, 2048, (8, 8)), (1, 2048, (8, 8)),
                                          (32, 2048, (8, 6)), (5, 512, (8, 8))],
                         ids=["b32_full", "b8_split", "b1_split16", "b32_partial_columns", "b5_idle_leader"])
def test_table_mcast_is_bit_identical(mesh_device, batch, S, grid):
    """One leader per grid column reads the Phase 2 table and multicasts it; every layout (full columns,
    sequence splits, and an 8x6 grid at batch 32 where 32 of 48 cores are active so columns are partial)
    must produce exactly the output of the per-core read.

    b5_idle_leader is the layout that pins where the multicast has to sit in the reader: batch 5 is 3
    cores per (batch, KV head), a 512-row cache is one chunk, so only the first core of each triple has
    work and the other two return early -- and 3 does not divide the row stride, so in five of the eight
    columns the leader is one of the cores with no chunk. It must still read and multicast the table, or
    the receivers below it wait on the semaphore forever."""
    q, k_lat, u_k, v, pos, cos, sin = _inputs(batch, S, 32, 128)
    c = fused.chunk_size(S)
    extra = dict(rot_step=_dev(mesh_device, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                 trans_mat=_dev(mesh_device, fused.trans_mat_tile()))
    def run(mcast):
        out = ttnn.transformer.palu_sdpa_decode(
            _dev(mesh_device, q), _dev(mesh_device, k_lat), _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None]),
            _dev(mesh_device, fused.cs_table(cos, sin, S)[None, None]), _dev(mesh_device, v), head_dim_v=128,
            cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5, table_mcast=mcast, **extra, **_cfg(0, grid))
        return _host(out)
    # Multicast run FIRST: the per-core read would otherwise leave the correct table in every core's c_13 /
    # c_14 at the same L1 address, and a multicast that silently did nothing would still read as correct.
    with_mcast = run(True)
    assert torch.equal(with_mcast, run(False))


def test_bfp4_latents_are_accepted_and_close(mesh_device):
    """bfp4 K and V latents (stage 2e L4 / stage 2f): the op must accept the format and the output must
    still track the fp32 reference, though at a looser bar than bfp8 (4-bit mantissas)."""
    q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 1024, 64, 128)
    S = 1024
    c = fused.chunk_size(S)
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(mesh_device, q), _dev(mesh_device, k_lat, ttnn.bfloat4_b), _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None]),
        _dev(mesh_device, fused.cs_table(cos, sin, S)[None, None]), _dev(mesh_device, v, ttnn.bfloat4_b), head_dim_v=128,
        cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5,
        rot_step=_dev(mesh_device, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
        trans_mat=_dev(mesh_device, fused.trans_mat_tile()), **_cfg(0))
    ref = fused.reference_decode(q, k_lat, u_k, v, pos, cos, sin, HD ** -0.5)
    ok, msg = comp_pcc(ref, _host(out), 0.95)
    assert ok, msg
