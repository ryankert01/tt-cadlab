#!/usr/bin/env python3
"""Reproducer, issue 002: ttnn.transformer.palu_sdpa_decode in group-shared mode (heads_per_core > 1) hangs
whenever a work item is split over 8 cores, i.e. a three-round reduction tree. Two-round layouts pass.

Run through the device gate with the n300 idle and scripts/env.sh sourced:
  scripts/run_device.sh python report_issue/repro/group_reduction_three_round_hang_repro.py         # controls only (4 cores per item): all pass
  scripts/run_device.sh python report_issue/repro/group_reduction_three_round_hang_repro.py --hang  # adds the 8-core layouts: expect a device timeout
                                                                                                  # after ~60 s, then reset the chip (tt-smi -r) before any
                                                                                                  # other device work.
The cap of 4 cores per item in the program factory is lifted through PALU_GROUP_MAX_CORES_PER_ITEM (debug override).
"""
import os, sys
os.environ.setdefault("PALU_GROUP_MAX_CORES_PER_ITEM", "8")       # read by the program factory at program build
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "60")  # a hang raises instead of blocking forever; read at process start
import torch, ttnn
from models.common.utility_functions import comp_pcc
from palu_tt import device as dev, fused

HD, NH, NKV, THETA = 128, 16, 4, 500000.0
R_K, R_V, GS, CHUNK = 128, 256, 4, 128
GRID_CORES = 64


def _dev(md, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, device=md, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(md))


def run(md, batch, H, S, legacy_rot):
    torch.manual_seed(0)
    bf = lambda t: t.to(torch.bfloat16).float()
    q = bf(torch.randn(1, batch, NH, HD))
    k_lat, v_lat = bf(torch.randn(batch, NKV // GS, S, R_K)), bf(torch.randn(batch, NKV // GS, S, R_V))
    u_k, u_v = bf(torch.randn(NKV, R_K, HD) / R_K ** 0.5), bf(torch.randn(NKV, R_V, HD) / R_V ** 0.5)
    pos = torch.full((batch,), S - 1)                       # every user attends the whole sequence: all cores get chunks
    cos, sin = (bf(t) for t in fused.cos_sin_tables(HD, S, THETA, None))
    cores_per_item = min(int(os.environ["PALU_GROUP_MAX_CORES_PER_ITEM"]), (GRID_CORES // batch) // (NKV // H))
    rounds = (cores_per_item - 1).bit_length()
    label = f"batch {batch:2d}  H {H}  S {S:5d}  cores/item {cores_per_item}  rounds {rounds}"
    print(f"{label} ... ", end="", flush=True)

    switches = dict(rot_mode="legacy", table_mcast=False) if legacy_rot else {}
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(md, q), _dev(md, k_lat), _dev(md, fused.w_rot_from_u_k(u_k)[None]), _dev(md, fused.cs_table(cos, sin, S)[None, None]),
        _dev(md, v_lat), head_dim_v=HD, scale=HD ** -0.5, heads_per_core=H,
        cur_pos_tensor=ttnn.from_torch(pos.to(torch.int32), device=md, dtype=ttnn.int32, mesh_mapper=ttnn.ReplicateTensorToMesh(md)),
        u_v=_dev(md, u_v[None]), rot_step=_dev(md, fused.rot_step_table(cos, sin, CHUNK)[None, None], ttnn.float32),
        trans_mat=_dev(md, fused.trans_mat_tile()), **switches,
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=CHUNK, exp_approx_mode=False),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                               fp32_dest_acc_en=True, packer_l1_acc=False),
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ref = fused.reference_decode(q, k_lat.repeat_interleave(GS, dim=1), u_k, v_lat.repeat_interleave(GS, dim=1), pos, cos, sin, HD ** -0.5, u_v=u_v)
    ok, msg = comp_pcc(ref, got, 0.99)
    print(f"{'pass' if ok else 'WRONG'}  ({msg})", flush=True)


if __name__ == "__main__":
    hang = "--hang" in sys.argv
    legacy_rot = "--shipped-rot" not in sys.argv   # default: the rotation mode the hang was measured with (stage 2c)
    md = dev.open_n300()
    try:
        if "--skip-controls" not in sys.argv:
            print("--- two-round trees (controls)")
            run(md, 16, 4, 1024, legacy_rot)
            run(md, 8, 2, 1024, legacy_rot)
        if hang:
            print("--- three-round trees: expect a device timeout; reset the chip afterwards")
            run(md, 4, 2, 1024, legacy_rot)
            run(md, 8, 4, 1024, legacy_rot)
    except Exception as e:
        msg = str(e)
        print(f"\nEXCEPTION: {msg.splitlines()[0][:200]}", flush=True)
        if "imeout" in msg or "hang" in msg.lower():
            print("The device is most likely hung: reset it (tt-smi -r 0) before any other device work.", flush=True)
            os._exit(2)                              # closing a hung device would block; leave without it
        dev.close(md)                                # a host-side failure before launch: close normally
        sys.exit(1)
    dev.close(md)
