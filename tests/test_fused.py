import torch

from palu_tt import fused

HD = 128


def test_transformation_matrix_is_the_rope_pair_rotation():
    # rotary_embedding_llama computes rotated = x @ T; for interleaved pairs that is (x0, x1) -> (-x1, x0)
    T = fused.rope_transformation_matrix(HD)
    x = torch.randn(3, HD)
    r = x @ T
    torch.testing.assert_close(r[:, 0::2], -x[:, 1::2])
    torch.testing.assert_close(r[:, 1::2], x[:, 0::2])
    t = fused.trans_mat_tile()
    assert t.shape == (1, 1, 32, 32) and torch.equal(t[0, 0], T[:32, :32])


def test_w_rot_is_u_k_and_u_k_times_T():
    u_k = torch.randn(4, 64, HD)
    w = fused.w_rot_from_u_k(u_k)
    assert w.shape == (4, 64, 2 * HD)
    assert torch.equal(w[..., :HD], u_k)
    lat = torch.randn(4, 7, 64)
    torch.testing.assert_close(lat @ w[..., HD:], (lat @ u_k) @ fused.rope_transformation_matrix(HD))


def test_chunk_size_is_the_ops_static_choice():
    # ttnn sdpa_decode (unpaged): largest power of two dividing S, capped at 512
    assert [fused.chunk_size(s) for s in (96, 256, 1024, 4096, 8192, 8192 + 32)] == [32, 256, 512, 512, 512, 32]


def test_tables_are_meta_interleaved_and_long_enough():
    cos, sin = fused.cos_sin_tables(HD, 1024, 500000.0, None)
    assert cos.shape == (1024 + 513, HD) and cos.dtype == torch.float32
    assert torch.equal(cos[:, 0::2], cos[:, 1::2]) and torch.equal(sin[:, 0::2], sin[:, 1::2])
    torch.testing.assert_close(cos[0], torch.ones(HD))
    tab = fused.cs_table(cos, sin, 1024)
    assert tab.shape == (1024, 2 * HD)
    assert torch.equal(tab[:, :HD], cos[:1024]) and torch.equal(tab[:, HD:], sin[:1024])


def test_rot_step_is_the_table_row_at_the_chunk_length_on_every_row():
    cos, sin = fused.cos_sin_tables(HD, 1024, 500000.0, None)
    for c in (32, 128, 512):
        step = fused.rot_step_table(cos, sin, c)
        assert step.shape == (32, 2 * HD) and step.dtype == torch.float32
        assert torch.equal(step[0, :HD], cos[c]) and torch.equal(step[31, HD:], sin[c])
        assert torch.equal(step[0], step[17])


def test_angle_addition_advances_the_table_by_a_chunk():
    cos, sin = fused.cos_sin_tables(HD, 1024, 500000.0, None)
    for c in (32, 128, 512):
        p = torch.arange(0, 700)
        cos2 = cos[p] * cos[c] - sin[p] * sin[c]
        sin2 = sin[p] * cos[c] + cos[p] * sin[c]
        # fp32 angle rounding in the shared rope generator: ~1e-4 at position ~1000, well under bf16
        torch.testing.assert_close(cos2, cos[p + c], atol=2e-4, rtol=0)
        torch.testing.assert_close(sin2, sin[p + c], atol=2e-4, rtol=0)


def test_relative_rotation_identity():
    """q R_p (k R_{s0+t})^T == (q R_{p-s0}) (k R_t)^T -- the identity Phase 2 rests on (spec §5)."""
    cos, sin = fused.cos_sin_tables(HD, 2048, 500000.0, None)
    T = fused.rope_transformation_matrix(HD)
    rot = lambda x, s: x * cos[s] + (x @ T) * sin[s]          # R_s applied to a row vector
    q, k = torch.randn(HD), torch.randn(HD)
    p, s0, t = 1500, 1024, 37
    lhs = rot(q, p) @ rot(k, s0 + t)
    q_rel = rot(q, p) * cos[s0] - (rot(q, p) @ T) * sin[s0]    # R_{-s0}: cos even, sin odd
    rhs = q_rel @ rot(k, t)
    torch.testing.assert_close(lhs, rhs, atol=1e-3, rtol=1e-5)


def test_reference_decode_reduces_to_causal_sdpa_without_rotation():
    B, nkv, nh, S, r_v = 2, 2, 4, 64, 32
    q = torch.randn(1, B, nh, HD)
    k = torch.randn(B, nkv, S, HD)
    v = torch.randn(B, nkv, S, r_v)
    cur = torch.tensor([10, 63])
    cos, sin = torch.ones(S, HD), torch.zeros(S, HD)
    out = fused.reference_decode(q, k, torch.eye(HD).expand(nkv, HD, HD), v, cur, cos, sin, HD ** -0.5)
    assert out.shape == (1, B, nh, r_v)
    for b in range(B):
        for h in range(nh):
            kk, vv = k[b, h // 2, : cur[b] + 1], v[b, h // 2, : cur[b] + 1]
            att = torch.softmax(q[0, b, h] @ kk.T * HD ** -0.5, -1) @ vv
            torch.testing.assert_close(out[0, b, h], att, atol=1e-5, rtol=1e-4)


def test_reference_decode_rotates_k_like_rotary_embedding_llama():
    B, nkv, nh, S, r_k = 1, 1, 1, 8, 32
    q = torch.randn(1, B, nh, HD)
    lat = torch.randn(B, nkv, S, r_k)
    u = torch.randn(nkv, r_k, HD)
    v = torch.randn(B, nkv, S, 16)
    cos, sin = fused.cos_sin_tables(HD, S, 500000.0, None)
    out = fused.reference_decode(q, lat, u, v, torch.tensor([S - 1]), cos, sin, 1.0)
    K = lat[0, 0] @ u[0]
    K = K * cos[:S] + (K @ fused.rope_transformation_matrix(HD)) * sin[:S]
    exp = torch.softmax(q[0, 0, 0] @ K.T, -1) @ v[0, 0]
    torch.testing.assert_close(out[0, 0, 0], exp, atol=1e-5, rtol=1e-4)


def test_reference_decode_reconstructs_v_when_u_v_is_given():
    B, nkv, nh, S, r_k, r_v = 1, 2, 4, 16, 32, 48
    q = torch.randn(1, B, nh, HD); k = torch.randn(B, nkv, S, r_k); u_k = torch.randn(nkv, r_k, HD)
    v_lat = torch.randn(B, nkv, S, r_v); u_v = torch.randn(nkv, r_v, HD)
    cos, sin = fused.cos_sin_tables(HD, S, 500000.0, None)
    out = fused.reference_decode(q, k, u_k, v_lat, torch.tensor([S - 1]), cos, sin, 1.0, u_v=u_v)
    v_full = torch.einsum("bhsr,hrd->bhsd", v_lat, u_v)
    exp = fused.reference_decode(q, k, u_k, v_full, torch.tensor([S - 1]), cos, sin, 1.0)
    assert out.shape == (1, B, nh, HD)
    torch.testing.assert_close(out, exp, atol=1e-5, rtol=1e-4)


# ---------------- Stage 2c: how many KV heads one core may own ----------------


def test_group_heads_per_core_picks_the_widest_h_that_fits_l1():
    """The three layer shapes the spec calls out (chunk 128, Phase 2, group size 4, batch 32). H = 4
    would fit L1 for the first, but at batch 32 it is 32 items over 2 cores (one reduction round) and
    the stage 2c measurements showed the merge costs more than the reads it saves, so the reduction-free
    H = 2 wins; the L1 order still shows through the other two."""
    assert fused.group_heads_per_core(128, 416, 128, True, 4) == 2
    assert fused.group_heads_per_core(288, 512, 128, True, 4) == 2
    assert fused.group_heads_per_core(512, 512, 128, True, 4) == 1
    # the L1-only order is visible when reductions are ignored: raise the batch so no layout splits
    fits = [h for h in (4, 2, 1) if h == 1 or fused.palu_l1_cb_bytes(128, 416, 128, True, h, cores_per_head=1,
                                                                        tree_rounds=0, batch=32) <= fused.L1_CB_BUDGET_BYTES]
    assert fits == [4, 2, 1]


def test_group_heads_per_core_rejects_an_h_that_leaves_the_grid_idle():
    """128 / 256 fits L1 at H = 4 at either batch, but only fills the 64-core grid at batch 32.

    At batch 8 a four-head group is 8 items and the factory gives each of them 4 cores (its cap), so
    32 cores work and 32 idle; H = 2 is 16 items over 4 cores each, fills the grid, and still halves
    the latent reads. At batch 32 H = 4 is 32 items over 2 cores each -- full, but one reduction round
    -- so the reduction-free H = 2 wins there; at batch 8 every layout reduces (per-head: one round,
    H = 2: two rounds) and the fewest rounds win: H = 1."""
    assert fused.group_heads_per_core(128, 256, 128, True, 4, batch=32) == 2
    assert fused.group_heads_per_core(128, 256, 128, True, 4, batch=8) == 1
    # ... and it really is occupancy, not L1, that rejects H = 4 at batch 8
    cores, rounds, active = fused._split_for(4, 8, 4, 64, 16)
    assert (cores, active) == (4, 32)
    assert fused.palu_l1_cb_bytes(128, 256, 128, True, 4, cores_per_head=cores, tree_rounds=rounds,
                                  batch=8) <= fused.L1_CB_BUDGET_BYTES
    assert fused._split_for(2, 8, 4, 64, 16)[2] == 64
    # a tie on occupancy is broken by the reduction rounds (fewest wins), then the larger H
    assert fused._split_for(4, 32, 4, 64, 16)[2] == fused._split_for(2, 32, 4, 64, 16)[2] == 64
    assert fused._split_for(4, 32, 4, 64, 16)[1] == 1 and fused._split_for(2, 32, 4, 64, 16)[1] == 0
    assert fused._split_for(1, 8, 4, 64, 16)[1] == 1 and fused._split_for(2, 8, 4, 64, 16)[1] == 2


def test_split_for_matches_the_stock_rule_at_h_1():
    """H = 1 is the split the op picks on its own, and it always fills the grid -- which is why the
    occupancy rule reduces to "a group layout has to fill the grid"."""
    for batch, cores in ((32, 1), (16, 1), (8, 2), (4, 4), (1, 16)):
        got_cores, _, active = fused._split_for(1, batch, 4, 64, 16)
        assert (got_cores, active) == (cores, 64), (batch, got_cores, active)


def test_group_heads_per_core_divides_the_group_and_never_returns_zero():
    for gs in (1, 2, 4):
        for r_k, r_v in ((64, 128), (512, 512)):
            h = fused.group_heads_per_core(r_k, r_v, 128, True, gs)
            assert 1 <= h <= gs and gs % h == 0
    # a budget below even H = 1 still falls back to the per-head kernel rather than 0
    assert fused.group_heads_per_core(512, 512, 128, True, 4, l1_budget_bytes=1) == 1


def test_group_heads_per_core_is_monotone_in_the_budget():
    small = fused.group_heads_per_core(128, 416, 128, True, 4, l1_budget_bytes=700_000)
    big = fused.group_heads_per_core(128, 416, 128, True, 4)
    assert small < big


def test_l1_cb_model_matches_the_factory_allocator_report():
    """Calibration point: the factory refused r_k = r_v = 512, chunk 128, Phase 2, H = 2, bf16 latents,
    one core per head with 'circular buffers ... grow to 1813984 B'. Task 1's state-CB sizing gave every
    state CB H blocks; task 2 keeps only O/m/l per head, worth (H-1) * (2*out_tiles + 5*PNHt) tiles."""
    cb = fused.palu_l1_cb_bytes(512, 512, 128, True, 2, cores_per_head=1, latent_tile_bytes=2048)
    task1_extra = (2 - 1) * (2 * 4 + 5 * 1) * 2048
    assert cb + task1_extra + fused.L1_RUNTIME_BASE_BYTES == 1_813_984
    assert fused.L1_CB_BUDGET_BYTES + fused.L1_RUNTIME_BASE_BYTES < fused.L1_MAX_BYTES


def test_group_heads_per_core_falls_back_to_the_per_head_kernel_below_batch_8():
    """The occupancy gate turns group mode off at batch 4 and below, which is what a batch-1 harness
    (decode_check) would get from ``auto``: a four-head group is 4 items over 4 cores there (16 of 64),
    H = 2 reaches 32, and only the per-head kernel fills the grid."""
    for batch, occ in ((4, (16, 32, 64)), (2, (8, 16, 64)), (1, (4, 8, 64))):
        assert tuple(fused._split_for(h, batch, 4, 64, 16)[2] for h in (4, 2, 1)) == occ
        assert fused.group_heads_per_core(64, 192, 128, True, 4, batch=batch) == 1
    # ... while batch 32 gets the reduction-free group layout for the same (narrow) layer, and batch 8
    # and 16 stay per-head (every group layout there adds a reduction round)
    assert fused.group_heads_per_core(64, 192, 128, True, 4, batch=32) == 2
    assert fused.group_heads_per_core(64, 192, 128, True, 4, batch=16) == 1
    assert fused.group_heads_per_core(64, 192, 128, True, 4, batch=8) == 1


def test_record_group_heads_indexes_by_layer_and_last_write_wins():
    tables = {}
    assert fused.group_heads_used(_Cfg(tables)) == []
    fused.record_group_heads(tables, 2, 4)
    assert fused.group_heads_used(_Cfg(tables)) == [1, 1, 4]      # layers not recorded read as 1
    fused.record_group_heads(tables, 0, 2)
    fused.record_group_heads(tables, 2, 2)                        # the allocator fallback halving H
    assert fused.group_heads_used(_Cfg(tables)) == [2, 1, 2]
    assert fused.group_heads_used(object()) == []                 # a model with no Palu config


class _Cfg:
    def __init__(self, tables):
        self.device_tables = tables
