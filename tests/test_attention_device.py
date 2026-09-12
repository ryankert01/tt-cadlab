import pytest
import torch
import ttnn
from loguru import logger
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup

from palu_tt import convert, hf, reference
from palu_tt.attention import PaluAttention

SEQ = 256


def _setup(mesh_device, palu_dir, batch, max_seq_len):
    args = ModelArgs(mesh_device, instruct=True, max_batch_size=batch, max_seq_len=max_seq_len, cache_hf=False)
    args.use_qk_fused = False
    args.n_layers = 1
    sd = args.load_state_dict()
    hf_dir = hf.snapshot_dir()
    cfg = convert.add_palu_to_state_dict(sd, hf_dir, palu_dir, 1, args.n_heads, args.n_kv_heads, args.head_dim)
    args.palu = cfg
    rope = RotarySetup(mesh_device, batch, args.head_dim, args.max_seq_len, args.rope_theta, args.rope_scaling, False)
    tt_ccl = TT_CCL(mesh_device)
    w = convert.hf_attn_weights(hf_dir, 0)
    lr = convert.identity_lowrank(w, args.n_kv_heads, args.head_dim) if palu_dir is None else \
        convert.load_palu_dir(palu_dir, 1, args.n_kv_heads, args.head_dim)[0]
    ref = reference.PaluRefAttention(hf.load_config(hf_dir), w, lr)
    return args, sd, rope, tt_ccl, ref


def _tt_attention(cls, mesh_device, args, sd, rope, tt_ccl):
    return cls(mesh_device, tt_ccl, args, sd, weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b), layer_num=0,
               dtype=ttnn.bfloat8_b, transformation_mats=rope.get_both_trans_mats(), configuration=args)


def _to_torch(mesh_device, args, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=args.cluster_shape))


def _ref_k_cache(ref, tt, S):
    """What the reference says the module's K cache should hold at position 0..S-1.

    r_k < head_dim: the K latent, verbatim. r_k == head_dim (V-only fast path): the stock K cache,
    i.e. RoPE'd K in tt_transformers' Meta interleaved head order (meta[2i] = hf[i],
    meta[2i+1] = hf[hd/2+i], the transpose reverse_permute applies to the weight rows)."""
    if tt.recon_k:
        return ref.cache_k_lat[:, :, :S, :]
    K = torch.einsum("bhtr,hdr->bhtd", ref.cache_k_lat[:, :, :S, :], ref.lr.k_U)
    cos, sin = ref.rotary(K, torch.arange(S)[None])
    _, K = apply_rotary_pos_emb(K, K, cos, sin)
    hd = K.shape[-1]
    return torch.stack([K[..., : hd // 2], K[..., hd // 2:]], dim=-1).flatten(-2)


def _ref_v_cache(ref, tt, S):
    """What the reference says the module's V cache should hold at position 0..S-1.

    r_v < head_dim: the V latent, verbatim. r_v == head_dim (fast-V path): full V -- the converter
    collapses U_v into the QKV weight there, so the head-split output that lands in the cache is V
    itself, whatever the factorisation (identity_lowrank's U_v = I, or a full-rank SVD's U_v != I)."""
    lat = ref.cache_v_lat[:, :, :S, :]
    return lat if tt.recon_v else torch.einsum("bhtr,hdr->bhtd", lat, ref.lr.v_U)


@pytest.mark.parametrize("palu_dir,pcc", [(None, 0.999), ("compressed/svd_r64", 0.99),
                                          ("compressed/svd_rk128_rv64", 0.99),
                                          ("compressed/svd_rk32_rv128", 0.99)],
                         ids=["identity", "r64", "rk128_rv64", "rk32_rv128"])
def test_prefill_matches_reference(mesh_device, palu_dir, pcc):
    args, sd, rope, tt_ccl, ref = _setup(mesh_device, palu_dir, batch=1, max_seq_len=SEQ)
    tt = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    x = torch.randn(1, SEQ, args.dim) * 0.05
    y_ref = ref.forward(x, start_pos=0)
    x_tt = args.prepare_residual_tensor_prefill(x, force_replicated=True)
    rot = [rope.cos_matrix_prefill[:, :, :SEQ, :], rope.sin_matrix_prefill[:, :, :SEQ, :]]
    y = tt.forward(x_tt, None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    y = _to_torch(mesh_device, args, y)[:, 0:1, :SEQ, : args.dim].view(1, SEQ, args.dim)
    ok, msg = comp_pcc(y_ref, y, pcc)
    logger.info(msg)
    assert ok, msg
    # K cache holds what the reference holds (latent, or stock rotated K on the V-only fast path)
    k_cache = ttnn.to_torch(tt.layer_past[0], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
    k_cache = k_cache[:1, : args.n_kv_heads, :SEQ, :]
    ok, msg = comp_pcc(_ref_k_cache(ref, tt, SEQ), k_cache, 0.99)
    logger.info(f"k cache {msg}")
    assert ok, f"k cache {msg}"


def test_identity_prefill_equals_stock_attention(mesh_device):
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, None, batch=1, max_seq_len=SEQ)
    stock = _tt_attention(Attention, mesh_device, args, sd, rope, tt_ccl)
    palu = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    x = torch.randn(1, SEQ, args.dim) * 0.05
    rot = [rope.cos_matrix_prefill[:, :, :SEQ, :], rope.sin_matrix_prefill[:, :, :SEQ, :]]
    ys = []
    for m in (stock, palu):
        x_tt = args.prepare_residual_tensor_prefill(x, force_replicated=True)
        y = m.forward(x_tt, None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
        ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :SEQ, : args.dim].view(1, SEQ, args.dim))
    ok, msg = comp_pcc(ys[0], ys[1], 0.999)
    assert ok, msg


def _decode_inputs(mesh_device, args, rope, x_bd, current_pos):
    x_tt = args.prepare_residual_tensor_decode(x_bd, args.get_attn_input_mem_config(Mode.DECODE, None), force_replicated=True)
    pos_tt = ttnn.from_torch(current_pos, device=mesh_device, dtype=ttnn.int32,
                             mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=args.cluster_shape))
    return x_tt, pos_tt, rope.get_rot_mats(current_pos)


@pytest.mark.parametrize("palu_dir,pcc", [(None, 0.999), ("compressed/svd_r64", 0.99),
                                          ("compressed/svd_rk128_rv64", 0.99),
                                          ("compressed/svd_rk32_rv128", 0.99)],
                         ids=["identity", "r64", "rk128_rv64", "rk32_rv128"])
def test_decode_after_prefill_matches_reference(mesh_device, palu_dir, pcc):
    batch = 32
    args, sd, rope, tt_ccl, ref = _setup(mesh_device, palu_dir, batch=batch, max_seq_len=SEQ)
    args.palu.runtime.check = True
    tt = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    # prefill user 0 with 128 tokens
    P = 128
    x_pre = torch.randn(1, P, args.dim) * 0.05
    ref.forward(x_pre, start_pos=0)
    rot = [rope.cos_matrix_prefill[:, :, :P, :], rope.sin_matrix_prefill[:, :, :P, :]]
    tt.forward(args.prepare_residual_tensor_prefill(x_pre, force_replicated=True), None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    # 8 decode steps, all users get user 0's input so the reference (batch 1) applies to slot 0
    for step in range(8):
        pos = P + step
        x = torch.randn(1, 1, args.dim) * 0.05
        y_ref = ref.forward(x, start_pos=pos)
        current_pos = torch.full((batch,), pos)
        args.palu.runtime.recon_len = ((pos + 1 + 31) // 32) * 32
        x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x.expand(batch, 1, -1).contiguous(), current_pos)
        y = tt.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
        y = _to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim)
        ok, msg = comp_pcc(y_ref, y[0:1], pcc)
        logger.info(f"pos={pos} {msg}")
        assert ok, f"pos={pos} {msg}"
    # both caches hold what the reference holds after prefill + 8 decode steps: the latents for a
    # reconstructed side, and (K only) the stock Meta-order rotated K on the r_k == head_dim path,
    # which decode fills with the decode-mode RoPE rather than prefill's.
    def _cache(i):
        c = ttnn.to_torch(tt.layer_past[i], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
        return c[:1, : args.n_kv_heads, : P + 8, :]
    ok, msg = comp_pcc(_ref_k_cache(ref, tt, P + 8), _cache(0), 0.99)
    assert ok, f"k cache {msg}"
    ok, msg = comp_pcc(_ref_v_cache(ref, tt, P + 8), _cache(1), 0.99)
    assert ok, f"v cache {msg}"


def test_identity_decode_equals_stock_attention(mesh_device):
    batch = 32
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, None, batch=batch, max_seq_len=SEQ)
    stock = _tt_attention(Attention, mesh_device, args, sd, rope, tt_ccl)
    palu = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    for step in range(4):
        x = torch.randn(batch, 1, args.dim) * 0.05
        current_pos = torch.full((batch,), step)
        args.palu.runtime.recon_len = 32
        ys = []
        for m in (stock, palu):
            x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x, current_pos)
            y = m.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
            ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim))
        ok, msg = comp_pcc(ys[0], ys[1], 0.999)
        assert ok, f"step={step} {msg}"


def test_decode_at_full_context_keeps_prefill_cos_sin(mesh_device):
    """recon_len == max_seq_len: the cos/sin slice covers the whole tensor, and a full-range ttnn
    slice returns the *same* tensor, so deallocating it frees the module's persistent prefill
    cos/sin and every later decode step dies with "Input Tensor is not allocated". Two identical
    steps at the last position, which is what a decode at the end of the context window does."""
    batch = 32
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, "compressed/svd_r64", batch=batch, max_seq_len=SEQ)
    tt = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    args.palu.runtime.recon_len = SEQ
    x = torch.randn(batch, 1, args.dim) * 0.05
    current_pos = torch.full((batch,), SEQ - 1)
    ys = []
    for _ in range(2):
        x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x, current_pos)
        y = tt.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
        ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim))
    ok, msg = comp_pcc(ys[0], ys[1], 0.999)  # same input at the same position -> same output
    assert ok, msg


def _cache_torch(mesh_device, args, m, i):
    return ttnn.to_torch(m.layer_past[i], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))


@pytest.mark.parametrize("phase", [1, 2])
@pytest.mark.parametrize("palu_dir", ["compressed/svd_r64", "compressed/svd_rk32_rv128"], ids=["r64", "rk32_rv128"])
def test_fused_decode_matches_naive(mesh_device, palu_dir, phase):
    """Stage 2 acceptance at the layer level: fused K path == stage-1 naive path (spec §7.2)."""
    batch = 32
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, palu_dir, batch=batch, max_seq_len=SEQ)
    args.palu.runtime.check = True
    naive = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    args.palu.runtime.use_fused_k, args.palu.runtime.cs_recurrence = True, phase == 2
    fusedm = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    assert not naive.use_fused_k and fusedm.use_fused_k and fusedm.phase2 == (phase == 2)
    assert not hasattr(fusedm, "u_k_decode") and not hasattr(fusedm, "cos_prefill")
    P = 128
    x_pre = torch.randn(1, P, args.dim) * 0.05
    rot = [rope.cos_matrix_prefill[:, :, :P, :], rope.sin_matrix_prefill[:, :, :P, :]]
    for m in (naive, fusedm):
        m.forward(args.prepare_residual_tensor_prefill(x_pre, force_replicated=True), None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    for step in range(6):
        pos = P + step
        x = torch.randn(batch, 1, args.dim) * 0.05
        current_pos = torch.full((batch,), pos)
        args.palu.runtime.recon_len = ((pos + 1 + 31) // 32) * 32
        ys = []
        for m in (naive, fusedm):
            x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x, current_pos)
            y = m.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
            ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim))
        ok, msg = comp_pcc(ys[0], ys[1], 0.99)
        logger.info(f"phase={phase} pos={pos} {msg}")
        assert ok, f"step={step} {msg}"
    for i in range(2):
        assert torch.equal(_cache_torch(mesh_device, args, naive, i), _cache_torch(mesh_device, args, fusedm, i)), f"cache {i}"


def test_fused_flag_is_ignored_on_the_v_only_row(mesh_device):
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, "compressed/svd_rk128_rv64", batch=1, max_seq_len=SEQ)
    args.palu.runtime.use_fused_k = True
    m = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    assert not m.recon_k and not m.use_fused_k and not hasattr(m, "w_rot")



def test_glrd_prefill_and_naive_decode_match_reference(mesh_device):
    """Group size 4: one K latent and one V latent per chip; prefill and the naive decode rebuild every
    head from the shared latent. Reference = the per-head CPU module on the expanded factors."""
    batch = 32
    args, sd, rope, tt_ccl, ref = _setup(mesh_device, "compressed/glrd_0.5", batch=batch, max_seq_len=SEQ)
    args.palu.runtime.check = True
    tt = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    assert tt.gs == 4 and tt.n_local_groups == 1 and not tt.use_fused_k
    assert tuple(tt.layer_past[0].shape)[:2] == (batch, 1) and tuple(tt.layer_past[1].shape)[:2] == (batch, 1)
    assert tt.kv_bytes_per_token() == 1 * (tt.r_k + tt.r_v) * 1.0625
    P = 128
    x_pre = torch.randn(1, P, args.dim) * 0.05
    y_ref = ref.forward(x_pre, start_pos=0)
    rot = [rope.cos_matrix_prefill[:, :, :P, :], rope.sin_matrix_prefill[:, :, :P, :]]
    y = tt.forward(args.prepare_residual_tensor_prefill(x_pre, force_replicated=True), None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    y = _to_torch(mesh_device, args, y)[0, 0, :P, : args.dim]
    ok, msg = comp_pcc(y_ref[0], y, 0.99)
    logger.info(f"prefill {msg}")
    assert ok, f"prefill {msg}"
    for step in range(4):
        pos = P + step
        x = torch.randn(1, 1, args.dim) * 0.05
        y_ref = ref.forward(x, start_pos=pos)
        current_pos = torch.full((batch,), pos)
        args.palu.runtime.recon_len = ((pos + 1 + 31) // 32) * 32
        x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x.expand(batch, 1, -1).contiguous(), current_pos)
        y = tt.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
        y = _to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim)
        ok, msg = comp_pcc(y_ref, y[0:1], 0.99)
        logger.info(f"pos={pos} {msg}")
        assert ok, f"pos={pos} {msg}"
    # the group latent is the per-head latent of any head in the group (the reference keeps it per head)
    for i, ref_cache in ((0, ref.cache_k_lat), (1, ref.cache_v_lat)):
        c = ttnn.to_torch(tt.layer_past[i], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
        got = c[:1, :, : P + 4, :]                                  # [1, n_groups(2), P+4, r]
        want = ref_cache[:, ::tt.gs, : P + 4, :]                    # head 0 of each group
        ok, msg = comp_pcc(want, got, 0.99)
        assert ok, f"cache {i} {msg}"


def test_glrd_fused_decode_matches_naive(mesh_device):
    batch = 32
    args, sd, rope, tt_ccl, _ = _setup(mesh_device, "compressed/glrd_0.5", batch=batch, max_seq_len=SEQ)
    args.palu.runtime.check = True
    naive = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    args.palu.runtime.use_fused_k, args.palu.runtime.cs_recurrence = True, True
    fusedm = _tt_attention(PaluAttention, mesh_device, args, sd, rope, tt_ccl)
    assert fusedm.gs == 4 and fusedm.use_fused_k and fusedm.phase2 and hasattr(fusedm, "u_v_kernel") and not hasattr(fusedm, "u_v_decode")
    P = 128
    x_pre = torch.randn(1, P, args.dim) * 0.05
    rot = [rope.cos_matrix_prefill[:, :, :P, :], rope.sin_matrix_prefill[:, :, :P, :]]
    for m in (naive, fusedm):
        m.forward(args.prepare_residual_tensor_prefill(x_pre, force_replicated=True), None, rot_mats=rot, user_id=0, mode=Mode.PREFILL)
    for step in range(6):
        pos = P + step
        x = torch.randn(batch, 1, args.dim) * 0.05
        current_pos = torch.full((batch,), pos)
        args.palu.runtime.recon_len = ((pos + 1 + 31) // 32) * 32
        ys = []
        for m in (naive, fusedm):
            x_tt, pos_tt, rot_mats = _decode_inputs(mesh_device, args, rope, x, current_pos)
            y = m.forward(x_tt, pos_tt, rot_mats=rot_mats, mode=Mode.DECODE)
            ys.append(_to_torch(mesh_device, args, y)[:, 0:1, :batch, : args.dim].view(batch, 1, args.dim))
        ok, msg = comp_pcc(ys[0], ys[1], 0.99)
        logger.info(f"pos={pos} {msg}")
        assert ok, f"step={step} {msg}"
    for i in range(2):
        assert torch.equal(_cache_torch(mesh_device, args, naive, i), _cache_torch(mesh_device, args, fusedm, i)), f"cache {i}"
