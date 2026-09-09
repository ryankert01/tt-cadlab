import os

import pytest
import torch
import ttnn
from loguru import logger

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


@pytest.mark.parametrize("palu_dir,pcc", [(None, 0.999), ("compressed/svd_r64", 0.99)], ids=["identity", "r64"])
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
    # latent caches hold what the reference holds
    k_lat = ttnn.to_torch(tt.layer_past[0], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
    k_lat = k_lat[:1, : args.n_kv_heads, :SEQ, :]
    ok, msg = comp_pcc(ref.cache_k_lat, k_lat, 0.99)
    assert ok, f"k latent cache {msg}"


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


@pytest.mark.parametrize("palu_dir,pcc", [(None, 0.999), ("compressed/svd_r64", 0.99)], ids=["identity", "r64"])
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
    v_lat = ttnn.to_torch(tt.layer_past[1], mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=args.cluster_shape))
    ok, msg = comp_pcc(ref.cache_v_lat, v_lat[:1, : args.n_kv_heads, : P + 8, :], 0.99)
    assert ok, f"v latent cache {msg}"


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
