"""Op-level tests for ttnn.transformer.palu_sdpa_decode (stage 2 fused K-reconstruction decode).

Runs on the N300 mesh fixture; every tensor is replicated to both chips and chip 0 is read back."""
import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

HD = 128


def _cfg():
    return dict(
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0,
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


@pytest.mark.parametrize("batch", [1, 32])
def test_clone_matches_stock_mla(mesh_device, batch):
    """Task 1 checkpoint: the renamed copy is byte-for-byte the stock MLA flash decode."""
    nh, nkv, S, r_v = 16, 4, 1024, 64
    q = torch.randn(1, batch, nh, HD)
    k = torch.randn(batch, nkv, S, HD)
    v = torch.randn(batch, nkv, S, r_v)
    pos = torch.randint(0, S, (batch,))
    pos[0] = S - 1
    args = (_dev(mesh_device, q), _dev(mesh_device, k), _dev(mesh_device, v))
    kw = dict(head_dim_v=r_v, cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5, **_cfg())
    ref = _host(ttnn.transformer.flash_multi_latent_attention_decode(*args, **kw))
    out = _host(ttnn.transformer.palu_sdpa_decode(*args, **kw))
    assert ref.shape == (1, batch, nh, r_v) and out.shape == ref.shape
    ok, msg = comp_pcc(ref, out, 0.9999)
    assert ok, msg
    assert torch.equal(ref, out), "clone must be bit-identical to the stock op"
