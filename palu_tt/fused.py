"""Host-side pieces of the fused K-reconstruction decode kernel (stage 2).

``ttnn.transformer.palu_sdpa_decode`` never materialises K. Per K chunk (positions s0..s0+c-1 of one
KV head) the kernel computes

    AB   = lat @ [u_k | u_k @ T]                      (one small matmul; T = pair-rotation matrix)
    K~_t = A_t * cos(t f) + B_t * sin(t f)             (rows t = 0..c-1, chunk-relative angles)

and, in Phase 2, rotates the query instead of the chunk base:  q~_k = q R_{-k c}, so that
q~_k . K~_t == (q R_p) . (k R_{s0+t})  -- the score the stock path computes -- because 2-D rotations
commute: R_{s0+t} = R_t R_{s0}. The chunk-relative table [cos(t f) | sin(t f)] for t < c is the
first c rows of the ordinary table, resident in L1; q~ advances by the constant R_{-c} once per
chunk (fp32). Phase 1 (the stepping stone) streams the absolute rows [cos((s0+t) f) | sin(..)] per
chunk instead and leaves q alone. Everything the kernel needs besides the two latent caches is
built here:

* ``w_rot``     = [u_k | u_k @ T]   per KV head,  [n_kv, r_k, 2*hd]   (bf16 on device)
* ``cs_tab``    = [cos | sin]       per position, [S, 2*hd]           (bf16)
* ``rot_step``  = [cos(c f) | sin(c f)] on all 32 rows, [32, 2*hd]   (fp32; Phase 2 only)
* ``trans_mat`` = the 32x32 pair-rotation tile                        (bf16; Phase 2 only)

cos/sin come from the same generator tt_transformers' ``get_rot_mats`` uses (so they match the
tables the naive stage-1 path rotates with), Meta-interleaved: every frequency occupies two adjacent
columns. ``reference_decode`` is the fp32 torch model of the whole op.
"""
import torch
from models.common.tensor_utils import get_rot_transformation_mat
from models.tt_transformers.tt.rope import compute_gather_cos_sin

MAX_CHUNK = 512  # ttnn sdpa_decode's cap on the static (unpaged) K chunk


def rope_transformation_matrix(head_dim: int) -> torch.Tensor:
    """[head_dim, head_dim] pair-rotation matrix: (x @ T) turns (x0, x1) into (-x1, x0) for every adjacent pair.

    The RoPE op applies get_rot_transformation_mat(32) per 32-wide tile; pairs never cross a tile
    boundary, so the full-width matrix is the same construction at head_dim (any even size)."""
    return get_rot_transformation_mat(head_dim)[0, 0]


def trans_mat_tile() -> torch.Tensor:
    """[1, 1, 32, 32]: the tile the kernel uses for x @ T on one 32-column tile."""
    return get_rot_transformation_mat(32)


def w_rot_from_u_k(u_k: torch.Tensor) -> torch.Tensor:
    """u_k [n_kv, r_k, hd] -> [n_kv, r_k, 2*hd] = [u_k | u_k @ T]."""
    return torch.cat([u_k, u_k @ rope_transformation_matrix(u_k.shape[-1])], dim=-1)


def chunk_size(seq_len: int) -> int:
    """The K chunk ttnn's unpaged flash decode picks for a cache of seq_len rows (sdpa_decode.cpp
    get_chunk_size): the largest power of two dividing seq_len, capped at MAX_CHUNK."""
    c = seq_len & -seq_len
    return min(MAX_CHUNK, c)


def cos_sin_tables(head_dim: int, seq_len: int, theta: float, rope_scaling):
    """Meta-interleaved cos/sin, fp32, each [seq_len + MAX_CHUNK + 1, head_dim].

    The extra rows are where the step constants come from (row c = angle c*f)."""
    n = seq_len + MAX_CHUNK + 1
    cos, sin = compute_gather_cos_sin(dhead=head_dim, end=2 * n, theta=theta, rope_scaling=rope_scaling)
    return cos[0, 0, :n].float(), sin[0, 0, :n].float()


def cs_table(cos: torch.Tensor, sin: torch.Tensor, seq_len: int) -> torch.Tensor:
    """[seq_len, 2*hd] = [cos | sin] for positions 0..seq_len-1."""
    return torch.cat([cos[:seq_len], sin[:seq_len]], dim=-1)


def rot_step_table(cos: torch.Tensor, sin: torch.Tensor, chunk: int) -> torch.Tensor:
    """[32, 2*hd] fp32: [cos(chunk f) | sin(chunk f)] repeated on every row (one full tile row, any
    row-broadcast convention works)."""
    return torch.cat([cos[chunk], sin[chunk]]).expand(32, -1).contiguous().float()


def reference_decode(q, k_lat, u_k, v_lat, cur_pos, cos, sin, scale: float) -> torch.Tensor:
    """fp32 torch model of palu_sdpa_decode.

    q [1, B, nh, hd] (already rotated), k_lat [B, nkv, S, r_k], u_k [nkv, r_k, hd], v_lat [B, nkv, S, r_v],
    cur_pos [B] ints (inclusive last valid position), cos/sin [>= S, hd] Meta-interleaved.
    GQA: query head h attends KV head h // (nh // nkv). Returns [1, B, nh, r_v]."""
    B, nkv, S, _ = k_lat.shape
    nh = q.shape[2]
    T = rope_transformation_matrix(u_k.shape[-1])
    K = torch.einsum("bhsr,hrd->bhsd", k_lat.float(), u_k.float())
    K = K * cos[None, None, :S] + (K @ T) * sin[None, None, :S]
    rep = nh // nkv
    K = K.repeat_interleave(rep, dim=1)                       # [B, nh, S, hd]
    V = v_lat.float().repeat_interleave(rep, dim=1)           # [B, nh, S, r_v]
    Q = q.float().permute(1, 2, 0, 3)                         # [B, nh, 1, hd]
    mask = torch.zeros(B, 1, 1, S)
    for b in range(B):
        mask[b, :, :, int(cur_pos[b]) + 1:] = float("-inf")
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask, scale=scale)
    return out.permute(2, 0, 1, 3)                            # [1, B, nh, r_v]
