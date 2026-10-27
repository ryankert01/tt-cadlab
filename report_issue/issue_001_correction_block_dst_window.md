# Issue draft 001 (upstream, tt-metal)

**Status (2026-09-23): superseded, do not file.** Upstream fixed this bug on 2026-09-14 in PR #56225 ("[Bug fix] Fit SDPA decode correction in FP32 half-sync DST", closes issue #56171), after v0.78.0 and not yet in a release tag. Their fix keeps the five-tile layout in the other DEST modes and reuses tile 2 for worker_sum only under FP32 half-sync; it ships an op-level regression test (`test_sdpa_decode_fp32_half_sync_cross_core_reduction`) and the LLK four-slot tests. Our own fix (submodule `palu-sdpa` a41a23f, four tiles in every mode) is verified on the n300 and stays until the next tt-metal upgrade, when it should be dropped in favour of upstream's. What our reproducers add to #56171: the >32-query-heads-per-KV-head case is always wrong on any split (PCC 0.11 on Wormhole), and with one head tile the outcome flips with acquisition parity. Everything below the line is the issue text as it was drafted.

---

**Title:** SDPA decode returns wrong output (PCC 0.11) with `fp32_dest_acc_en=True` when there are more than 32 query heads per KV head and the sequence is split across cores

### Describe the bug

`ttnn.transformer.scaled_dot_product_attention_decode` with 64 query heads on 1 KV head, `fp32_dest_acc_en=True` (default `dst_full_sync_en=False`) and a sequence split across cores returns garbage: PCC 0.11 against an fp32 reference. The same call with `dst_full_sync_en=True`, or with `fp32_dest_acc_en=False`, or with 32 query heads, gives PCC 0.9997. No error or warning.

Cause: `correction_block` (`sdpa/device/kernels/compute/compute_common.hpp:718`), the fused softmax merge used in the tree reduction, stages five tiles in DST registers 0 to 4 inside one `tile_regs_acquire`. With `fp32_dest_acc_en=True` and half sync the DST window is four fp32 tiles (`tech_reports/matrix_engine/matrix_engine.md:83`; the decode factory itself uses `dst_size = fp32_dest_acc_en ? 4 : 8`). Register 4 aliases register 0 on one of the two DST halves, so the merged max comes back equal to `worker_sum`. With two query-head tiles one tile always lands on that half, hence the always-wrong result above. With one query-head tile the outcome depends on how many DST acquisitions preceded the merge; we confirmed this on a single core by running the helper alone with 0/1/2/3 empty acquisitions in front of it (wrong/ok/wrong/ok). Happy to attach that harness.

### To reproduce

```python
# ttnn.transformer.scaled_dot_product_attention_decode returns wrong output with fp32_dest_acc_en=True
# when there are more than 32 query heads per KV head and the sequence is split across cores.
import torch, ttnn

def run(dev, nqh, fp32_acc, full_sync, B=1, S=4096, D=128):
    torch.manual_seed(0)
    q, k, v = [t.to(torch.bfloat16) for t in (torch.randn(1, B, nqh, D), torch.randn(B, 1, S, D), torch.randn(B, 1, S, D))]
    tq, tk, tv = [ttnn.from_torch(t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for t in (q, k, v)]
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq, tk, tv, cur_pos=[S - 1] * B, scale=D ** -0.5,
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc, packer_l1_acc=False, dst_full_sync_en=full_sync))
    got = ttnn.to_torch(out).float()[0, :, :nqh]
    p = torch.softmax(torch.einsum("bhd,bsd->bhs", q[0].float(), k[:, 0].float()) * D ** -0.5, dim=-1)
    ref = torch.einsum("bhs,bsd->bhd", p, v[:, 0].float())            # fp32 reference over the whole sequence
    pcc = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
    print(f"q_heads={nqh} fp32_dest_acc_en={str(fp32_acc):5s} dst_full_sync_en={str(full_sync):5s} | PCC {pcc:.4f}", flush=True)

dev = ttnn.open_device(device_id=0)
run(dev, 64, True, False)   # 2 query-head tiles, 4-tile fp32 DST window
run(dev, 64, True, True)    # 8-tile window
run(dev, 64, False, False)  # bf16 DST, 8-tile window
run(dev, 32, True, False)   # 1 query-head tile
ttnn.close_device(dev)
```

Output on a Wormhole n300, tt-metal v0.78.0:

```
q_heads=64 fp32_dest_acc_en=True  dst_full_sync_en=False | PCC 0.1086
q_heads=64 fp32_dest_acc_en=True  dst_full_sync_en=True  | PCC 0.9997
q_heads=64 fp32_dest_acc_en=False dst_full_sync_en=False | PCC 0.9997
q_heads=32 fp32_dest_acc_en=True  dst_full_sync_en=False | PCC 0.9997
```

### Expected behavior

PCC about 0.9997 in all four rows.

### Environment

- tt-metal v0.78.0 (`23b8a7a`), built from source with clang-20
- Wormhole n300, Ubuntu 22.04, Python 3.10

### Fix (PR to follow)

The SFPU body `calculate_fused_max_sub_exp_add_tile` (`ckernel_sfpu_sdpa.h`, wormhole_b0 and blackhole) now uses four DEST tiles instead of five: each output is written over the input consumed in the same iteration (tile 0 prev_max -> exp_max_diff, 1 worker_max -> exp_max_diff_2, 2 prev_sum -> cur_max, 3 worker_sum -> cur_sum), and `correction_block` loads prev_sum / worker_sum into registers 2 / 3. Same arithmetic and the same DEST round-trips as before, so results in the configurations that already worked are unchanged. The LLK unit test `test_sfpu_sdpa_correction` is updated to the four-tile layout and now also runs the fp32 half-sync configuration. With the fix, the script above prints:

```
q_heads=64 fp32_dest_acc_en=True  dst_full_sync_en=False | PCC 0.9997
q_heads=64 fp32_dest_acc_en=True  dst_full_sync_en=True  | PCC 0.9997
q_heads=64 fp32_dest_acc_en=False dst_full_sync_en=False | PCC 0.9997
q_heads=32 fp32_dest_acc_en=True  dst_full_sync_en=False | PCC 0.9997
```

---

## Internal notes (not part of the issue)

- `report_issue/repro/sdpa_decode_fp32acc_repro.py`: the script above, byte-identical. Run with `scripts/run_device.sh` after sourcing `scripts/env.sh`.
- `report_issue/repro/correction_block_repro.py`: the single-core `generic_op` harness behind the parity claim (helper alone, three DST configurations, 0 to 3 empty acquisitions). Supporting evidence only; offer it if asked.
- Sweep behind the "split across cores" statement (same script, more shapes): with 64 query heads and `fp32_dest_acc_en=True`, batch 1/4/16 (64/16/4 cores per head) at S 1024 and 4096 all give PCC 0.17 to 0.45; batch 64 (1 core per head, no merge) gives 0.9998. With 8 query heads every layout is 0.9997.
- Our own kernel (`third_party/tt-metal`, branch `palu-sdpa`): the group branch hit the bad half (PCC 0.334, stage 2c section 6.1); the per-head branch and the stock op at <= 32 query heads per KV head are correct by acquisition parity, not by design. Workaround shipped at `palu_sdpa_flash_decode.cpp:560-590` (unfused primitives). See also `docs/audit/reports/G_kernel.md`.
- Llama-3 class models have at most 32 query heads per KV head, so `models/tt_transformers` is in the parity-dependent case, not the always-wrong one.
