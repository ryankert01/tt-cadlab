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
import os
from typing import Optional

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
    assert seq_len > 0, f"chunk_size requires seq_len > 0, got {seq_len} (chunk 0 gives an identity rot_step and a silently wrong Phase 2)"
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


def reference_decode(q, k_lat, u_k, v_lat, cur_pos, cos, sin, scale: float, u_v=None) -> torch.Tensor:
    """fp32 torch model of palu_sdpa_decode.

    q [1, B, nh, hd] (already rotated), k_lat [B, nkv, S, r_k], u_k [nkv, r_k, hd], v_lat [B, nkv, S, r_v],
    cur_pos [B] ints (inclusive last valid position), cos/sin [>= S, hd] Meta-interleaved.
    GQA: query head h attends KV head h // (nh // nkv). Returns [1, B, nh, r_v].

    u_v [nkv, r_v, hd] (optional): V is reconstructed per head as v_lat @ u_v (the kernel's in-kernel V
    mode); without it v_lat is attended over directly (the absorbed-V path)."""
    B, nkv, S, _ = k_lat.shape
    nh = q.shape[2]
    T = rope_transformation_matrix(u_k.shape[-1])
    K = torch.einsum("bhsr,hrd->bhsd", k_lat.float(), u_k.float())
    K = K * cos[None, None, :S] + (K @ T) * sin[None, None, :S]
    rep = nh // nkv
    K = K.repeat_interleave(rep, dim=1)                       # [B, nh, S, hd]
    V = v_lat.float()
    if u_v is not None:
        V = torch.einsum("bhsr,hrd->bhsd", V, u_v.float())    # in-kernel V mode: rebuild V per head
    V = V.repeat_interleave(rep, dim=1)                       # [B, nh, S, r_v (or hd)]
    Q = q.float().permute(1, 2, 0, 3)                         # [B, nh, 1, hd]
    mask = torch.zeros(B, 1, 1, S)
    for b in range(B):
        mask[b, :, :, int(cur_pos[b]) + 1:] = float("-inf")
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask, scale=scale)
    return out.permute(2, 0, 1, 3)                            # [1, B, nh, r_v]


# ---------------- Stage 2c: how many KV heads one core may own (group-shared mode) ----------------
#
# ``palu_sdpa_decode(heads_per_core=H)`` keeps H per-head weight blocks and H flash states resident on
# every core so the H heads of a group share one latent read. What bounds H is L1, and the bound is
# tight enough that H has to be picked per layer: the weight block alone is
# (r_k/32 * 8 + r_v/32 * 4 + phase2) tiles of 2 KB -- 174 KB at r_k 128 / r_v 416, 395 KB at 512/512.
#
# ``palu_l1_cb_bytes`` is a byte-exact model of the program factory's circular-buffer sum for the Palu
# envelope (unpaged, causal, no sink / sliding window / block padding, DRAM Q and output, cur_pos as a
# tensor, in-kernel V). It is calibrated against the allocator: the factory reports "circular buffers
# ... grow to 1813984 B" for r_k = r_v = 512, chunk 128, Phase 2, H = 2, bf16 latents, one core per head
# -- with stage 2c task 1's state-CB sizing (every state CB H blocks). This model, with the same sizing,
# sums to 1708288 B, so the runtime's own L1 base is exactly 1813984 - 1708288 = 105696 B; it sits below
# the CBs and is charged against the same 1499136 B limit. (Task 2 shrank the transient state CBs back
# to one block, worth 26624 B at H = 2, so the model's CB sum for that configuration is 1681664 B now.)
L1_MAX_BYTES = 1_499_136          # the allocator's ceiling in the factory's error message
L1_RUNTIME_BASE_BYTES = 105_696   # measured: reported total minus this model's CB sum
# The op does not get the whole of L1 inside a model: tt_transformers keeps L1-sharded activations at
# the top of the bank and the CB region has to end below them, so 1499136 is not the ceiling that
# matters. Measured on Llama-3.1-8B decode (glrd_0.5 layer 0, H = 4 forced):
#
#   batch  8 : clash, "L1 buffer allocated at 1413120 ... region ends at 1415456"
#   batch 16 : no clash with the region ending at 1415648, so the ceiling is above that
#   batch 32 : no clash with the region ending at 1366496 (a lower bound only)
#
# The smallest of those is the batch-8 one, and that is what this uses. The margin below is the
# estimator's own slack on top of it -- the model's buffer address is not something this can predict,
# so PaluAttention also halves H and retries when the allocator does refuse (see attention.py).
# Raising the margin past ~22.5 KB would push the spec's H = 4 layer (128 / 416 at batch 32, which
# needs the region to end at 1390560) below the bar, so the retry carries the residual risk.
L1_MODEL_CEILING_BYTES = 1_413_120
L1_SAFETY_BYTES = 16_384
L1_CB_BUDGET_BYTES = L1_MODEL_CEILING_BYTES - L1_RUNTIME_BASE_BYTES - L1_SAFETY_BYTES


def palu_l1_cb_bytes(r_k: int, r_v: int, chunk: int, phase2: bool, heads_per_core: int,
                     cores_per_head: int = 4, tree_rounds: int = 2, head_dim: int = 128,
                     q_head_tiles: int = 1, latent_tile_bytes: int = 1088,
                     k_latent_tile_bytes: Optional[int] = None, v_latent_tile_bytes: Optional[int] = None,
                     batch: int = 32, rot_mode: str = "legacy") -> int:
    """Bytes of circular buffer the program factory allocates per core, for one fused-decode program.

    ``cores_per_head`` / ``tree_rounds`` describe the sequence split; the default is the widest one
    group-shared mode allows (the program factory caps it at 4 cores per item, i.e. two tree rounds),
    because the receive slots for the tree reduction are sized per head in group-shared mode.
    ``latent_tile_bytes`` is 1088 for a bfp8 latent cache (the default), 2048 for bf16, 576 for bfp4;
    ``k_latent_tile_bytes`` / ``v_latent_tile_bytes`` override it for one side only (stage 2f: the two
    latent caches may take different block-float formats), so c_9 is sized from the K cache's tile and
    c_2 from the V cache's.
    ``rot_mode`` is the stage 2e K-rebuild mode: "dst" never stages the AB block through c_10, so the
    factory shrinks that slot to the group-mode peel scratch (PNHt tiles); "legacy" and "pack" keep it.
    """
    kt = k_latent_tile_bytes or latent_tile_bytes
    vt_ = v_latent_tile_bytes or latent_tile_bytes
    t, f = 2048, 4096                       # bf16 / fp32 32x32 tile
    dht = head_dim // 32                    # head dim in tiles
    wt = 2 * dht                            # [u_k | u_k T] width
    sk = chunk // 32                        # K chunk in row tiles
    rt, rvt = r_k // 32, r_v // 32          # latent widths in tiles
    p2 = 1 if phase2 else 0
    h = max(1, int(heads_per_core))
    out_tiles = q_head_tiles * dht          # PNHt * vDHt
    cur_pos_stick = max(32, ((batch * 4 + 31) // 32) * 32)
    n = 0
    n += q_head_tiles * dht * t                                   # c_0  Q
    n += sk * dht * 2 * t                                         # c_1  K_rot (double buffered)
    n += sk * rvt * 2 * vt_                                       # c_2  V latent
    n += sk * q_head_tiles * t                                    # c_3  causal mask
    n += 3 * t                                                    # c_5 scale, c_6 m_in, c_7 l_in
    n += 2 * cur_pos_stick                                        # c_8 / c_15 cur_pos
    n += sk * rt * 2 * kt                                         # c_9  K latent
    n += h * (rt * wt + rvt * dht + p2) * t                       # c_4  H resident weight blocks
    n += (q_head_tiles if rot_mode == "dst" else 2 * wt) * t      # c_10 AB block (dst mode: peel scratch only)
    n += (sk * wt if phase2 else 2 * sk * wt) * t                 # c_13 [cos | sin]
    if phase2:
        n += 4 * q_head_tiles * dht * f                           # c_12 rotated-query state
        n += wt * f                                               # c_14 R_{-c} constants
    n += t                                                        # c_11 column identity
    n += sk * q_head_tiles * t                                    # c_24 QK
    n += out_tiles * t                                            # c_25 out_im       (transient)
    n += h * out_tiles * t                                        # c_26 out_acc      (per head)
    n += q_head_tiles * t                                         # c_27 cur_max      (transient)
    n += h * q_head_tiles * t                                     # c_28 prev_max     (per head)
    n += q_head_tiles * t                                         # c_29 cur_sum      (transient)
    n += h * q_head_tiles * t                                     # c_30 prev_sum     (per head)
    n += 3 * q_head_tiles * t                                     # c_31, c_21, c_22  (transient)
    n += out_tiles * t                                            # c_23 out_acc_2    (transient)
    n += out_tiles * t + 2 * q_head_tiles * t                     # c_16 / c_17 / c_18 payload
    slots = 0 if cores_per_head <= 1 else (tree_rounds if h > 1 else cores_per_head - 1)
    n += (out_tiles + 2 * q_head_tiles) * slots * h * t           # c_19 receive slots
    n += out_tiles * t                                            # c_20 output
    return n


def _group_cap() -> int:
    """Cores per item that group mode may split over: the factory's cap of 4, or the debug override
    PALU_GROUP_MAX_CORES_PER_ITEM (the factory reads the same variable), so the L1 model and the
    program agree on the tree depth during the eight-core sweep."""
    v = os.environ.get("PALU_GROUP_MAX_CORES_PER_ITEM", "").strip()
    return max(1, int(v)) if v.isdigit() else 4


def _split_for(h: int, batch: int, num_kv_heads: int, grid_cores: int, max_cores_per_head: int):
    """(cores per item, tree rounds, active cores) the program factory derives for heads_per_core = h.

    Mirror of the factory: an item is (batch, group of h heads), the cores per batch come from the grid
    and max_cores_per_head_batch, and group mode caps the split at 4 cores per item. h = 1 is the stock
    rule instead -- the split the op picks on its own, which is what heads_per_core=None builds.
    """
    uncapped = max(1, min(grid_cores, max_cores_per_head * batch * num_kv_heads) // max(1, batch))
    if h > 1:
        items = max(1, num_kv_heads // h)
        cores = max(1, min(_group_cap(), uncapped // items))
    else:
        heads_per_core = max(1, -(-num_kv_heads // uncapped))       # the stock ceil, then round up
        while num_kv_heads % heads_per_core:
            heads_per_core += 1
        items = num_kv_heads // heads_per_core
        cores = max(1, uncapped // num_kv_heads)
    rounds = 0 if cores <= 1 else (cores - 1).bit_length()
    return cores, rounds, batch * items * cores


def group_heads_per_core(r_k: int, r_v: int, chunk: int, phase2: bool, heads_per_group: int,
                         l1_budget_bytes: int = L1_CB_BUDGET_BYTES, batch: int = 32,
                         num_kv_heads: int = 0, grid_cores: int = 64, max_cores_per_head: int = 16,
                         rot_mode: str = "legacy", **kwargs) -> int:
    """Largest H dividing ``heads_per_group`` whose program fits L1 (spec §3.3).

    ``batch`` (and the grid) matter because the sequence split follows from them, and in group mode the
    tree-reduction receive slots are sized per head: the same layer needs one round of slots at batch
    32 and two at batch 8. ``num_kv_heads`` defaults to ``heads_per_group`` (one group per chip, which
    is the N300 G-LRD layout). ``**kwargs`` -- ``latent_tile_bytes`` and the per-cache
    ``k_latent_tile_bytes`` / ``v_latent_tile_bytes`` -- is forwarded to ``palu_l1_cb_bytes``.

    The choice is occupancy- and reduction-aware, in that order, from the stage 2c measurements
    (docs/results/stage2c.md §2): halving the item count halves the cores an item can be split over
    once the grid is full, so an H whose layout does not fill the grid is rejected when a smaller one
    does (at batch 8 a four-head group is 8 items over 4 cores each, 32 of 64 cores). Among the layouts
    that fill the grid, the one with the fewest tree-reduction rounds wins: every cross-core merge in
    group mode runs the unfused correction sequence and costs more than the latent reads it saves
    (batch 32: H = 2 on 64 items with no reduction beat both per-head and H = 4 with one round; batch
    8: per-head with one round beat H = 2 with two). Ties keep the larger H. In practice: batch 32 ->
    H = 2, batch 16 and 8 -> H = 1.

    **Below batch 8 this returns H = 1 for every layer**: at batch 4 a four-head group is 4 items over
    4 cores (16 of 64) and even H = 2 only reaches 32, while the per-head kernel spreads a single head
    over 4 cores and fills the grid; at batch 1 it is 4 / 8 / 64. A batch-1 harness
    (``palu_tt.decode_check``) therefore runs the stage 2b per-head kernel unless it passes an explicit
    H -- nobody has measured whether trading three quarters of the grid for a 4x cut in latent reads
    pays off there -- so ``auto`` is not a way to exercise the group kernel at small batch.

    H = 1 is always the fallback -- it is the stage 2b per-head kernel, which the op runs with
    ``heads_per_core=None`` and which always fits (it is the program stage 2b already shipped).
    """
    assert heads_per_group >= 1
    n_kv = num_kv_heads or heads_per_group
    fits = []                                  # (h, active cores, reduction rounds), largest h first
    for h in range(heads_per_group, 0, -1):
        if heads_per_group % h != 0 or n_kv % h != 0:
            continue
        cores, rounds, active = _split_for(h, batch, n_kv, grid_cores, max_cores_per_head)
        if h == 1 or palu_l1_cb_bytes(r_k, r_v, chunk, phase2, h, cores_per_head=cores,
                                      tree_rounds=rounds, batch=batch, rot_mode=rot_mode,
                                      **kwargs) <= l1_budget_bytes:
            fits.append((h, active, rounds))
    best_active = max(a for _, a, _ in fits)
    full = [(h, r) for h, a, r in fits if a == best_active]
    fewest_rounds = min(r for _, r in full)
    return max(h for h, r in full if r == fewest_rounds)


# ---------------- Stage 2c: recording which H each layer actually ran with ----------------
#
# The estimate above is per layer, and PaluAttention can still fall back to a smaller H when the
# allocator refuses, so the only authority on what a run used is the run itself. The attention module
# records it here (keyed on the PaluConfig's device_tables, which lives exactly as long as one model
# build) and the harnesses copy it into their result JSON, so a row's numbers can be read against the
# H that produced them. Both are pure dict operations -- no device, no ttnn.


def record_group_heads(device_tables: dict, layer: int, heads_per_core: int) -> None:
    """Record the heads_per_core layer ``layer`` ended up using. Idempotent per layer (last wins)."""
    used = device_tables.setdefault("group_heads", [])
    while len(used) <= layer:
        used.append(1)
    used[layer] = int(heads_per_core)


def group_heads_used(config) -> list:
    """Per-layer heads_per_core for the model built from ``config`` (index = layer).

    ``[]`` when nothing recorded -- a non-grouped model, or a row that never ran the fused decode.
    """
    tables = getattr(config, "device_tables", None) or {}
    return list(tables.get("group_heads", []))
