# Issue 002 (internal, `palu-sdpa` branch): three-round reduction tree hangs in group-shared mode

**Status (2026-09-24): fixed, ours.** Root cause found with tt-exalens on the hung chip and fixed in the kernels; the two eight-core layouts that hung now complete at PCC 0.9997 (reproducer below, all four cases pass). Not an upstream bug: the stock decode op completes three- and five-round trees on this build and its decode kernels are byte-identical to upstream main. The stock kernel carries the same latent hazard but never triggers it, because each core emits once per launch.

### Root cause

`c_16` served two roles on a node that both receives from children and sends to a parent: the writer pushed each child's O into it for the compute to merge (writer produces, unpacker consumes), and the compute's packer pushed the node's own merged O into it for the writer to send (packer produces, writer consumes). The two sides count differently. The dataflow API increments the shared stream registers (`pages_received_ptr[0] += n`, `pages_acked_ptr[0] += n`); the LLK pack and unpack sides keep private cumulative counts and **store** them into the same registers (`llk_push_to_brisc`, `llk_pop_tiles`). After the first emit the packer's store overwrote the writer's pushes, so the next child O the writer pushed was invisible to the unpacker (`received - private_acked == 0`), the compute waited forever for it at the O merge, and the writer then blocked on the next round's `cb_m_in` reserve because the compute never popped the previous max.

Measured on the hung intermediate core (8 cores per item, head 1, round 0): `c_16` register received 8, acked 12, unpacker private acked 8; `c_6` 3 / 2; `c_7` 4 / 3. Writer PC at `cb_m.reserve_back` (writer line 337), unpack thread PC in `move_block(cb_out_o, ...)` (compute line 604). Root cores were merely waiting for that node's head-1 payload. One-child nodes survive two heads because the writer's own push is counted before the packer's first store; two-child nodes do not.

### Fix (submodule, 2026-09-24)

Non-root cores emit their payload O through `c_20` (`cb_out_final`, the root's output CB, which a non-root core never uses) instead of `c_16`, in both the group-shared and the per-head compute paths; the writer sends from `c_20`. `c_16` is now only the child-O inbox (one producer, one consumer), `c_20` only the outbox. No numeric change. A `TT_FATAL` in the factory requires the output tile size to equal the stats tile size when a sequence is split, since the parent's receive slot assumes one size. The four-core cap in the factory stays until the eight-core layouts are re-measured for L1 and speed; `PALU_GROUP_MAX_CORES_PER_ITEM` lifts it.

Reproducer after the fix (n300, cap lifted to 8):

```
batch 16  H 4  S  1024  cores/item 4  rounds 2 ... pass  (0.99973)
batch  8  H 2  S  1024  cores/item 4  rounds 2 ... pass  (0.99973)
batch  4  H 2  S  1024  cores/item 8  rounds 3 ... pass  (0.99972)
batch  8  H 4  S  1024  cores/item 8  rounds 3 ... pass  (0.99973)
```

### Symptom

`ttnn.transformer.palu_sdpa_decode` in group-shared mode (`heads_per_core > 1`) hangs, with no error and no watcher output, whenever a work item is split over 8 cores, which is a three-round reduction tree. One- and two-round trees complete and are correct. The device must be reset afterwards.

### Reproduce

The factory's cap is lifted through `PALU_GROUP_MAX_CORES_PER_ITEM` (debug override, submodule 7ed8432), so no source edit is needed. The script runs two two-round control layouts and, with `--hang`, the two eight-core layouts under `TT_METAL_OPERATION_TIMEOUT_SECONDS=60`, so the hang surfaces as a device timeout instead of a stuck process. **Reset the chip after a `--hang` run before any other device work.**

```bash
source scripts/env.sh
scripts/run_device.sh python report_issue/repro/group_reduction_three_round_hang_repro.py          # controls only
scripts/run_device.sh python report_issue/repro/group_reduction_three_round_hang_repro.py --hang   # adds the three-round layouts
```

```python
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
        print("--- two-round trees (controls)")
        run(md, 16, 4, 1024, legacy_rot)
        run(md, 8, 2, 1024, legacy_rot)
        if hang:
            print("--- three-round trees: expect a device timeout; reset the chip afterwards")
            run(md, 4, 2, 1024, legacy_rot)
            run(md, 8, 4, 1024, legacy_rot)
    except Exception as e:
        print(f"\nEXCEPTION: {str(e).splitlines()[0][:200]}\nThe device is most likely hung: reset it (tt-smi -r) before any other device work.", flush=True)
        os._exit(2)                                  # closing a hung device would block; leave without it
    dev.close(md)
```

Control run, 2026-09-23 (n300, submodule 7ed8432, cap lifted to 8):

```
--- two-round trees (controls)
batch 16  H 4  S  1024  cores/item 4  rounds 2 ... pass  (0.99973)
batch  8  H 2  S  1024  cores/item 4  rounds 2 ... pass  (0.99973)
```

The `--hang` layouts (batch 4 / H 2 and batch 8 / H 4 at S 1024, 8 cores per item) have not been re-run with this script; the stage 2c measurements of the same layouts (with the cap edited in source) hung at S 1024 and 4096 and passed at S 512, where only four cores receive a chunk.

### What we know

- Tree tables, reader, writer and the receive protocol are the stock ones. Group mode changes two things: receive slots and semaphore nibbles are per (head, round), and the child merge uses unfused primitives instead of `correction_block`.
- Checked without finding the cycle: slot indexing, semaphore counts per round, CB capacity on a node that has both two children and a parent, the `cb_out_o == cb_out_worker` aliasing on intermediate nodes, pop/push accounting on the group receive path (second reviewer agreed; `docs/results/stage2c.md` section 6.3).
- The watcher does not link on this build (`idle_erisc.elf ... overflows region`), so there is no instruction-level trace.

### Mitigation that was in place

`group_max_cores_per_head = 4` in the program factory: group mode never built a three-round tree. Cost: at batch 8 with H 4 half the grid idles, which is why the shipped selector uses the per-head kernel below batch 32.

### Next step (done; kept for the record)

1. Run the script with `--hang` to confirm the hang still reproduces with the shipped rotation mode (`--shipped-rot`) as well as the legacy one.
2. Add per-round debug prints (DPRINT server, which does not need the watcher) to the group receive path in the writer and compute kernels of an intermediate node, to see which core stops at which wait.
3. Test the one live hypothesis: on an intermediate node the second child's payload overwrites the first child's slot before it is consumed. Give each child its own slot for the round and re-run.
4. If it completes, sweep the layouts above with the cap lifted, raise the cap, and re-measure the group-mode cells it constrained.

### References

`docs/results/stage2c.md` sections 6.2 and 6.3; `docs/kernel_guide.md` B5 and C5; `docs/audit/reports/G_kernel.md` item 7.
