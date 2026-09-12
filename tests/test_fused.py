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
