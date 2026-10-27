---
name: correction-block-dst-fix
description: tt-metal sdpa correction_block five-DST-tile overflow, root cause, fix on palu-sdpa a41a23f (2026-09-23), and the device-gate gotchas hit while verifying
metadata:
  type: project
---

The stock sdpa softmax-merge SFPU body (`calculate_fused_max_sub_exp_add_tile`, dst offsets 0..128 = five tiles) overflowed the four-tile fp32 half-sync DEST window; tile 4 aliased tile 0 on one of the two DEST halves, so the result depended on acquisition parity. Public op `scaled_dot_product_attention_decode` with >32 q heads per KV head and a split sequence returned PCC 0.11. Fixed on submodule branch `palu-sdpa` commit a41a23f (superproject a4559a6): four tiles, each output written over the input it consumed; LLK test/golden updated but not run here. Upstream fixed the same bug independently in PR #56225 (issue #56171, merged 2026-09-14, after v0.78.0, unreleased as of 2026-09-23) with a different layout (tile 2 reused only under FP32 half-sync) plus an op-level test; no upstream PR needed. Drop our a41a23f in favour of 70f840f46ad at the next tt-metal upgrade. Issue drafts in `report_issue/` (marked superseded, uncommitted), reproducers in `report_issue/repro/`.

**Why:** the stage 2c "per-head branch works, group branch fails" mystery was parity luck, not design; the audit's I33 "timing property" is this.

**How to apply:** when editing shared tt-metal headers (sdpa `compute_common.hpp`, `ckernel_sfpu_sdpa.h`), purge `~/.cache/tt-metal-cache/*/kernels/{sdpa,sdpa_flash_decode,Kernel_Source_Code,palu_sdpa_flash_decode}` by hand: `run_device.sh` only hashes the palu kernel dir. Keep SFPU vFloat live ranges short across `_ckernel_sfpu_exp_accurate_` calls or the SFPI compiler fails with "cannot store sfpu register (register spill)". Device gate: `scripts/env.sh` must be sourced before `run_device.sh` (else `python` not found), and a `pgrep -f "pytest tests/"` in the same command line self-matches, so write `pytest ./tests/...`. Upstream regression: `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py` (12 tests, ~40 s). See [[palu-tt-project]], [[code-audit-2026-09-19]].
