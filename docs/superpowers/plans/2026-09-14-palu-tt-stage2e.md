# Stage 2e Implementation Plan: DST-resident K rebuild, packer-accumulate add, angle-table multicast, bfp4 latents

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `ttnn.transformer.palu_sdpa_decode` two new K-rebuild modes and a multicast angle table behind switches, measure every variant with the stage 2d profile fit and whole-step decode at batch 32, ship the fastest mode that passes the correctness bars, and record the bfp4-latent measurement.

**Architecture:** The op gains two attributes, `rot_mode` (`legacy` / `pack` / `dst`, a compute-kernel define) and `table_mcast` (a reader define plus the five multicast runtime args the reader already receives). `palu_build_k_chunk` in `palu_compute_common.hpp` grows two `#if PALU_ROT_MODE` branches; the reader grows one multicast helper; the factory computes column leaders and sizes `c_10` per mode. The Python side threads the two flags from the CLIs through `PaluRuntime` to the op call, `palu_tt.profile_fit` turns profile JSONs into the fixed / per-chunk table, and the report gains `--stage 2e`.

**Tech Stack:** tt-metal v0.78.0 submodule branch `palu-sdpa` (C++ program factory, nanobind, JIT-compiled RISC-V kernels), ttnn, PyTorch, pytest, the tracy device profiler, Python 3.

## Global Constraints

Copied from the spec (`docs/superpowers/specs/2026-09-14-palu-tt-stage2e-design.md`) and the project's standing rules:

- Op attribute `rot_mode` takes exactly the values `legacy`, `pack`, `dst`; compute define `PALU_ROT_MODE` = 0 / 1 / 2. Op attribute `table_mcast` is a bool; reader define `PALU_TABLE_MCAST`. Both default off (`legacy`, `false`) until Task 9 sets the shipped default.
- Correctness bars (S1), unchanged for every variant: op PCC >= 0.99 vs the fp32 reference and >= 0.995 vs stock MLA on host-rotated K; attention-level fused-vs-naive PCC >= 0.999; `decode_check` min PCC >= 0.99 over 63 steps; group-mode `torch.equal` where no child merges; `table_mcast` on must be bit-identical to off.
- Measurement bars: S2 fixed term <= 78 us per layer (`r_k` 32 / `r_v` 128); S3 per-chunk term <= 101 us per layer; S4 `palu_full_50` shipped variant at batch 32: 4k <= 60.0 ms, 1k <= 40.5 ms (stretch 55.4 / 38.0); S5 bfp4 table (perplexity, `decode_check`, bytes) for `palu_full_50` and `palu_full_70`.
- Result row suffix: `_fk2` + `{p, d}` (rot_mode pack / dst) + `{m}` (table_mcast on), e.g. `palu_full_50_fk2dm`; bfp4 latent rows `_l4`. Stage 2 `_fk2` rows are never overwritten.
- `pack` and `dst` modes need `DHt % 2 == 0` and `dst_size >= 4` (both true here: `DHt` 4, fp32 dest accumulate); the factory rejects anything else with `TT_FATAL`.
- In `dst` mode `c_10` is sized `PNHt` tiles (group mode's peel scratch); in the other modes `2 * Wt` as today.
- Every device command runs through `scripts/run_device.sh <cmd>` **and** only after `pgrep -f "palu_tt\."` shows nothing of ours running; never run bare `pytest tests/` or `tests/*_device.py` outside the gate. Shared server: nice CPU work, `CMAKE_BUILD_PARALLEL_LEVEL=10` is what `scripts/build_ttnn.sh` already sets; no sudo, no firmware or driver changes, never touch `/home/user01/tt-data`.
- tt-metal changes are committed inside `third_party/tt-metal` on branch `palu-sdpa` with `git commit --no-verify` (the pre-commit clang-format hook reformats copied kernels); the superproject commit records the new submodule pointer. `third_party/palu` is never committed. Never commit `docs/results/raw/*.log`, profiler CSVs, `.superpowers/`, or smoke JSONs.
- `git add` by explicit path only. Commit with `git -c user.name=ryan -c user.email=ryankert01@gmail.com`; every commit message ends with the two trailer lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs` (verbatim; do not substitute another model name).
- Host-side C++ edits (factory, op, nanobind) need `scripts/build_ttnn.sh`; kernel edits under `device/kernels/` are JIT-compiled at first op run.
- Python env: `source scripts/env.sh` (activates `.venv`, sets `PYTHONPATH`, `MESH_DEVICE=N300`).

---

## File map

| file | responsibility in this stage |
|---|---|
| `third_party/tt-metal/ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/device/palu_sdpa_decode_device_operation_types.hpp` | the two new attributes |
| `.../palu_sdpa_decode/device/palu_sdpa_decode_device_operation.cpp` | `prim::palu_sdpa_decode` signature, attribute init |
| `.../palu_sdpa_decode/palu_sdpa_decode.hpp`, `.cpp`, `palu_sdpa_decode_nanobind.cpp` | host wrapper and Python binding (`rot_mode: str`, `table_mcast: bool`) |
| `.../palu_sdpa_decode/device/palu_sdpa_decode_program_factory.cpp` | `PALU_ROT_MODE` / `PALU_TABLE_MCAST` defines, `c_10` sizing, column-leader multicast args |
| `.../palu_sdpa_decode/device/kernels/compute/palu_compute_common.hpp` | `palu_build_k_chunk` modes 1 and 2 |
| `.../palu_sdpa_decode/device/kernels/compute/palu_sdpa_flash_decode.cpp` | the `c_10` static_assert made mode-aware |
| `.../palu_sdpa_decode/device/kernels/dataflow/palu_dataflow_common.hpp`, `palu_reader_decode.cpp` | `palu_read_table_mcast`, its call under the define |
| `palu_tt/convert.py`, `palu_tt/model.py`, `palu_tt/attention.py` | `PaluRuntime.rot_mode` / `table_mcast` to the op call |
| `palu_tt/fused.py` | `palu_l1_cb_bytes(..., rot_mode)` |
| `palu_tt/bench.py`, `palu_tt/decode_check.py`, `palu_tt/profile_decode.py`, `palu_tt/eval_ppl.py` | `--rot-mode`, `--table-mcast`, `--latent-dtype` |
| `palu_tt/profile_fit.py` (new) | fixed + per-chunk fit over profile JSONs |
| `palu_tt/report.py`, `palu_tt/profile_plots.py` | `--stage 2e` tables, variant figure |
| `tests/test_palu_sdpa_device.py`, `tests/test_attention_device.py`, `tests/test_fused.py`, `tests/test_profile_fit.py` (new), `tests/test_convert.py` | coverage |
| `docs/results/stage2e.md`, `docs/results/raw/*.json`, `docs/results/profile_attention_ctx_2e.png` | results |

---

### Task 1: `rot_mode` and `table_mcast` plumbed end to end, behaviour unchanged

**Files:**
- Modify: `third_party/tt-metal/ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/device/palu_sdpa_decode_device_operation_types.hpp:48-52`
- Modify: `.../device/palu_sdpa_decode_device_operation.cpp:624-680` (`prim::palu_sdpa_decode`)
- Modify: `.../device/palu_sdpa_decode_device_operation.hpp:121` (the `prim` declaration that lists `heads_per_core`)
- Modify: `.../palu_sdpa_decode.hpp:14-31`, `.../palu_sdpa_decode.cpp:40-115`, `.../palu_sdpa_decode_nanobind.cpp:53-95`
- Modify: `.../device/palu_sdpa_decode_program_factory.cpp:958-972` (compute defines), `:997-1000` (dataflow defines)
- Modify: `palu_tt/convert.py:27-46`, `palu_tt/model.py:13-35`, `palu_tt/attention.py:290-291` and `:566-575`
- Modify: `palu_tt/bench.py:57,364-395`, `palu_tt/decode_check.py:64-95`, `palu_tt/profile_decode.py:26-40`
- Test: `tests/test_convert.py`, `tests/test_palu_sdpa_device.py`

**Interfaces:**
- Produces: op kwargs `rot_mode: str = "legacy"`, `table_mcast: bool = False` on `ttnn.transformer.palu_sdpa_decode`; `PaluRuntime.rot_mode: str`, `PaluRuntime.table_mcast: bool`; `build_model(..., rot_mode="legacy", table_mcast=False)`; CLI flags `--rot-mode {legacy,pack,dst}`, `--table-mcast {off,on}` on `bench`, `decode_check`, `profile_decode`; compute define `PALU_ROT_MODE` (0/1/2) and dataflow define `PALU_TABLE_MCAST` (only when `table_mcast && phase2`).

- [ ] **Step 1: Host test for the runtime defaults (fails first)**

Append to `tests/test_convert.py`:

```python
def test_runtime_has_stage2e_switches_off_by_default():
    from palu_tt.convert import PaluRuntime
    rt = PaluRuntime()
    assert rt.rot_mode == "legacy"
    assert rt.table_mcast is False
```

Run: `source scripts/env.sh && python -m pytest tests/test_convert.py -q -k stage2e`
Expected: FAIL with `AttributeError: 'PaluRuntime' object has no attribute 'rot_mode'`

- [ ] **Step 2: Python runtime fields and `build_model` kwargs**

`palu_tt/convert.py`, after `group_heads_per_core: Optional[int] = None` inside `PaluRuntime`:

```python
    # stage 2e: how the fused op rebuilds a K row tile. "legacy" = AB through c_10 and an SFPU add
    # (stage 2); "pack" = the packer's L1 accumulate does the add, two column tiles per DST trip;
    # "dst" = matmul straight into DST, dest-reuse multiplies, no AB scratch buffer.
    rot_mode: str = "legacy"
    # stage 2e: read the Phase 2 angle table once per grid column and multicast it to the column.
    table_mcast: bool = False
```

`palu_tt/model.py` `build_model` signature: add `rot_mode: str = "legacy", table_mcast: bool = False` after `group_heads`, and after `cfg.runtime.group_heads_per_core = group_heads` add:

```python
        cfg.runtime.rot_mode, cfg.runtime.table_mcast = rot_mode, bool(table_mcast)
```

- [ ] **Step 3: Attention passes the flags to the op**

`palu_tt/attention.py`, right after `self.fused_compute_cfg = ...` (line 291):

```python
            # stage 2e: the rebuild mode and the table multicast are per-run switches; multicast only
            # exists in Phase 2 (Phase 1 streams the table per chunk).
            self.rot_mode = rt.rot_mode
            self.table_mcast = bool(rt.table_mcast and self.phase2)
```

and in `_fused_call` (line 570) add the two kwargs to the op call:

```python
                    scale=self.scale, heads_per_core=hpc if hpc > 1 else None,
                    rot_mode=self.rot_mode, table_mcast=self.table_mcast,
```

- [ ] **Step 4: CLI flags**

`palu_tt/bench.py`: `FUSED = {"use_fused_k": False, "cs_recurrence": False, "fused_k_chunk": None, "group_heads": None, "rot_mode": "legacy", "table_mcast": False}`. In `main()` after `--group-heads`:

```python
    p.add_argument("--rot-mode", choices=["legacy", "pack", "dst"], default="legacy",
                   help="stage-2e K rebuild mode in the fused op (see docs/superpowers/specs/2026-09-14-palu-tt-stage2e-design.md).")
    p.add_argument("--table-mcast", choices=["off", "on"], default="off",
                   help="stage-2e: multicast the Phase 2 angle table down each grid column (Phase 2 only).")
```

and in the `FUSED.update(...)` call add `rot_mode=a.rot_mode, table_mcast=a.table_mcast == "on"`; in the result dict `r` add `"rot_mode": a.rot_mode, "table_mcast": a.table_mcast`.

`palu_tt/decode_check.py`: same two `add_argument`s; pass `rot_mode=a.rot_mode, table_mcast=a.table_mcast == "on"` in the fused `_logits(...)` call; add both to the result dict.

`palu_tt/profile_decode.py`: same two `add_argument`s; in `flags = dict(...)` add `rot_mode=a.rot_mode, table_mcast=a.table_mcast == "on"`.

- [ ] **Step 5: Run the host test**

Run: `source scripts/env.sh && python -m pytest tests/test_convert.py tests/test_bench_utils.py -q`
Expected: all PASS (the new test included).

- [ ] **Step 6: C++ attributes**

`palu_sdpa_decode_device_operation_types.hpp`, after `uint32_t heads_per_core = 0;`:

```cpp
    // Palu stage 2e: how palu_build_k_chunk rotates one K row tile. 0 = legacy (AB block through
    // c_10, SFPU add), 1 = pack (packer L1-accumulate add, two column tiles per DST trip), 2 = dst
    // (matmul into DST, dest-reuse multiplies, packer-accumulate add, no c_10 block).
    uint32_t rot_mode = 0;
    // Palu stage 2e: the Phase 2 angle table and rot_step are read by one core per grid column and
    // multicast to the column's other active cores.
    bool table_mcast = false;
```

`palu_sdpa_decode_device_operation.hpp` (the `prim::palu_sdpa_decode` declaration at line ~121) and `.cpp` (line ~635): add `uint32_t rot_mode, bool table_mcast,` immediately after `std::optional<uint32_t> heads_per_core,`; in the attribute initialiser after `.heads_per_core = heads_per_core.value_or(0),` add `.rot_mode = rot_mode, .table_mcast = table_mcast,`.

`palu_sdpa_decode.hpp`: after the `heads_per_core` parameter add

```cpp
    const std::string& rot_mode = "legacy",                       // stage 2e: legacy | pack | dst
    bool table_mcast = false,                                     // stage 2e: multicast the Phase 2 table per column
```

(add `#include <string>` at the top). `palu_sdpa_decode.cpp`: same two parameters in the definition; before the `prim` call:

```cpp
    uint32_t rot_mode_id = 0;
    if (rot_mode == "legacy") {
        rot_mode_id = 0;
    } else if (rot_mode == "pack") {
        rot_mode_id = 1;
    } else if (rot_mode == "dst") {
        rot_mode_id = 2;
    } else {
        TT_THROW("palu_sdpa_decode: rot_mode must be legacy, pack or dst, got {}", rot_mode);
    }
    TT_FATAL(!table_mcast || rot_step.has_value(), "palu_sdpa_decode: table_mcast needs Phase 2 (rot_step)");
```

and pass `heads_per_core, rot_mode_id, table_mcast, scale, ...` to `ttnn::prim::palu_sdpa_decode`.

`palu_sdpa_decode_nanobind.cpp`: after `nb::arg("heads_per_core") = nb::none(),` add `nb::arg("rot_mode") = "legacy", nb::arg("table_mcast") = false,`; in the docstring after the `heads_per_core` paragraph add:

```
        rot_mode (stage 2e): "legacy" (stage 2 rebuild), "pack" (packer L1-accumulate add) or "dst"
        (matmul into DST with dest-reuse multiplies). table_mcast (stage 2e, Phase 2 only): one core
        per grid column reads the chunk-relative angle table and multicasts it to the column.
```

- [ ] **Step 7: Factory defines and envelope checks**

`palu_sdpa_decode_program_factory.cpp`, in the compute defines block after `compute_defines["PALU_DST_TILES"] = ...`:

```cpp
    const uint32_t rot_mode = operation_attributes.rot_mode;
    TT_FATAL(rot_mode <= 2, "palu_sdpa_decode: rot_mode {} out of range", rot_mode);
    if (rot_mode != 0) {
        TT_FATAL(Sk_chunk_t > 0, "palu_sdpa_decode: rot_mode {} needs a pinned K chunk", rot_mode);
        TT_FATAL(DHt % 2 == 0 && dst_size >= 4,
                 "palu_sdpa_decode: rot_mode {} rotates two column tiles per DST trip (DHt {}, dst_size {})",
                 rot_mode, DHt, dst_size);
    }
    compute_defines["PALU_ROT_MODE"] = std::to_string(rot_mode);
```

and in the dataflow defines block:

```cpp
    const bool table_mcast = operation_attributes.table_mcast && phase2;
    if (table_mcast) {
        dataflow_defines.emplace_back("PALU_TABLE_MCAST", "1");
    }
```

(Task 4 fills in the runtime args; until then the define is set but every core has `do_k_mcast = 0` and `num_dests = 0`, so the reader helper of Task 4 falls back to the plain read. Do **not** add the define's consumer yet.)

- [ ] **Step 8: Rebuild and op-level test that `legacy` is the default**

Run: `scripts/build_ttnn.sh` (5 to 15 minutes incremental).

Append to `tests/test_palu_sdpa_device.py` (next to `test_phase2_matches_reference_default_chunk`):

```python
def test_rot_mode_legacy_is_the_default_and_bogus_is_rejected(mesh_device):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 1024, 32, 128)
    default = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True)
    legacy = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode="legacy")
    assert torch.equal(default, legacy)
    with pytest.raises(RuntimeError, match="rot_mode must be"):
        _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode="bogus")
```

and change `_fused`'s signature to `def _fused(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk, phase2, rot_mode="legacy", table_mcast=False):` passing `rot_mode=rot_mode, table_mcast=table_mcast` into the op call. Same two keyword parameters on `_fused_group` (forwarded the same way).

Run (gate first): `pgrep -f "palu_tt\." ; scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k "rot_mode_legacy or phase2_matches_reference_default_chunk"`
Expected: PASS.

- [ ] **Step 9: Commit (submodule first, then superproject)**

```bash
cd third_party/tt-metal && git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode && git -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m "palu_sdpa_decode: rot_mode and table_mcast attributes (stage 2e), legacy behaviour unchanged" && cd ../..
git add third_party/tt-metal palu_tt/convert.py palu_tt/model.py palu_tt/attention.py palu_tt/bench.py palu_tt/decode_check.py palu_tt/profile_decode.py tests/test_convert.py tests/test_palu_sdpa_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: rot_mode / table_mcast switches threaded from the CLIs to the op (no behaviour change)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

(the submodule commit body also needs the two trailer lines; add them with additional `-m` arguments as above.)

---

### Task 2: `pack` mode: packer L1-accumulate replaces the SFPU add

**Files:**
- Modify: `.../device/kernels/compute/palu_compute_common.hpp:59-76` (the rotation phase of `palu_build_k_chunk`)
- Test: `tests/test_palu_sdpa_device.py`

**Interfaces:**
- Consumes: `PALU_ROT_MODE` define (Task 1).
- Produces: `palu_build_k_chunk` behaves identically in mode 0, and in mode 1 packs `A cos` then `B sin` with L1 accumulate, two column tiles per DST trip.

- [ ] **Step 1: Parametrise the reference tests over `rot_mode` (fails first for `pack`)**

In `tests/test_palu_sdpa_device.py` change `_check` to accept `rot_mode="legacy"` and forward it to `_fused`, then replace the two Phase 2 reference tests with:

```python
ROT_MODES = ["legacy", "pack", "dst"]


@pytest.mark.parametrize("rot_mode", ROT_MODES)
@pytest.mark.parametrize("r_k", [32, 64, 96])
@pytest.mark.parametrize("batch", [1, 32])
def test_phase2_matches_reference_default_chunk(mesh_device, r_k, batch, rot_mode):
    _check(mesh_device, batch, 1024, r_k, 64, k_chunk=0, phase2=True, rot_mode=rot_mode)


@pytest.mark.parametrize("rot_mode", ROT_MODES)
@pytest.mark.parametrize("S", [4096, 8192])
def test_phase2_long_context(mesh_device, S, rot_mode):
    _check(mesh_device, 1, S, 64, 64, k_chunk=0, phase2=True, rot_mode=rot_mode)


@pytest.mark.parametrize("rot_mode", ["pack", "dst"])
def test_rot_mode_tracks_legacy_closely(mesh_device, rot_mode):
    """The new modes round K_rot twice (bf16 pack, then the accumulating pack); against legacy's single
    rounding the outputs must still agree to PCC 0.999 at every calibrated rank."""
    for r_k in (32, 64, 96):
        q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 2048, r_k, 128)
        a = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode="legacy")
        b = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 128, 0, True, rot_mode=rot_mode)
        ok, msg = comp_pcc(a, b, 0.999)
        assert ok, f"r_k={r_k} {rot_mode} vs legacy: {msg}"
```

(keep the original bodies of the two replaced tests, which use `r_v` 64: `r_v` 128 at the op's default chunk does not fit L1 at batch 1 and `r_k` >= 64; only the parametrisation is new.) The `dst` cases fail until Task 3; run with `-k "pack"` for now.

Run: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k "pack and (default_chunk or tracks_legacy)"`
Expected: FAIL (mode 1 not implemented: the kernel ignores the define, so `tracks_legacy` passes trivially but the reference test with `rot_mode="pack"` must fail once the `#if` branch exists; write the branch, then confirm the tests pass).

- [ ] **Step 2: Implement mode 1 in `palu_build_k_chunk`**

Replace the `// ---- K[:, i] = A_i * cos + B_i * sin ----` block of `palu_build_k_chunk` (lines 59-76) with:

```cpp
        // ---- K[:, i] = A_i * cos + B_i * sin ----
        ab.wait_front(Wt);
        reconfig_data_format(cb_ab, cb_cs);  // eltwise: in0 -> SrcA, in1 -> SrcB
        pack_reconfig_data_format(cb_k);
        mul_init(cb_ab, cb_cs);
#if PALU_ROT_MODE == 1
        // pack mode (stage 2e): two column tiles per DST trip, the packer's L1 accumulate does the add.
        // dst0 = A_j cos_j, dst1 = B_j sin_j, dst2 = A_j+1 cos_j+1, dst3 = B_j+1 sin_j+1; the A products
        // are packed with accumulate off, the B products onto the same L1 tiles with it on.
        static_assert(DHt % 2 == 0 && dst_tiles >= 4, "pack mode rotates two column tiles per DST trip");
        for (uint32_t j = 0; j < DHt; j += 2) {
            const uint32_t c = cs_tile_base + i * Wt + j;
            tile_regs_acquire();
            mul_tiles(cb_ab, cb_cs, j, c, 0);
            mul_tiles(cb_ab, cb_cs, j + DHt, c + DHt, 1);
            mul_tiles(cb_ab, cb_cs, j + 1, c + 1, 2);
            mul_tiles(cb_ab, cb_cs, j + 1 + DHt, c + 1 + DHt, 3);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(0);
            pack_tile<true>(0, cb_k, j * Sk_chunk_t + i);
            pack_tile<true>(2, cb_k, (j + 1) * Sk_chunk_t + i);
            pack_reconfig_l1_acc(1);
            pack_tile<true>(1, cb_k, j * Sk_chunk_t + i);
            pack_tile<true>(3, cb_k, (j + 1) * Sk_chunk_t + i);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
#else
        add_binary_tile_init();
        for (uint32_t j = 0; j < DHt; ++j) {
            const uint32_t c = cs_tile_base + i * Wt + j;
            tile_regs_acquire();
            mul_tiles(cb_ab, cb_cs, j, c, 0);
            mul_tiles(cb_ab, cb_cs, j + DHt, c + DHt, 1);
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_k, j * Sk_chunk_t + i);
            tile_regs_release();
        }
#endif
        ab.pop_front(Wt);
```

Add `#include "api/compute/pack.h"` next to the existing include at the top of the header, and `#ifndef PALU_ROT_MODE / #define PALU_ROT_MODE 0 / #endif` right after the includes so the header compiles when the define is absent.

- [ ] **Step 3: Run the pack tests on device**

Run: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k "pack and (default_chunk or long_context or tracks_legacy)"`
Expected: PASS. If `test_rot_mode_tracks_legacy_closely` fails on PCC but the reference tests pass, record the PCC in the task report and continue (the spec's fallback is decided in Task 9); if the reference tests fail, the accumulate is not doing what the API says: check that `pack_reconfig_l1_acc(0)` precedes the first pack and that the two packs of a pair target the same `c_1` index, then report BLOCKED with the PCC numbers.

- [ ] **Step 4: Group mode under pack (bit identity where no child merges)**

Parametrise `test_group_shared_batch32_matches_stock_split` and `test_heads_per_core_1_is_the_stock_split` over `rot_mode in ["legacy", "pack"]` (add a `rot_mode` parameter forwarded through `_fused_group` to both calls of each test).

Run: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k "pack and (batch32_matches_stock_split or heads_per_core_1)"`
Expected: PASS (`torch.equal` holds because both sides run the same mode).

- [ ] **Step 5: Commit**

```bash
cd third_party/tt-metal && git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/device/kernels/compute/palu_compute_common.hpp && git -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m "palu_sdpa_decode: rot_mode pack -- packer L1-accumulate add, two column tiles per DST trip" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs" && cd ../..
git add third_party/tt-metal tests/test_palu_sdpa_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: pack rebuild mode and its op tests" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 3: `dst` mode: matmul into DST, dest-reuse multiplies, no AB block

**Files:**
- Modify: `.../device/kernels/compute/palu_compute_common.hpp` (`palu_build_k_chunk`)
- Modify: `.../device/kernels/compute/palu_sdpa_flash_decode.cpp:526-531` (the `c_10` static_assert)
- Modify: `.../device/palu_sdpa_decode_program_factory.cpp:721` (`ab_tiles`)
- Modify: `palu_tt/fused.py:142-190` (`palu_l1_cb_bytes`), `:213` (`group_heads_per_core`), `palu_tt/attention.py:305-308`
- Test: `tests/test_palu_sdpa_device.py`, `tests/test_fused.py`

**Interfaces:**
- Consumes: `PALU_ROT_MODE == 2`.
- Produces: `palu_l1_cb_bytes(..., rot_mode: str = "legacy")` and `group_heads_per_core(..., rot_mode: str = "legacy")` (keyword, default preserves every existing result).

- [ ] **Step 1: Host test for the L1 model (fails first)**

Append to `tests/test_fused.py`:

```python
def test_l1_model_drops_the_ab_block_in_dst_mode():
    legacy = fused.palu_l1_cb_bytes(32, 128, 512, True, 1, cores_per_head=1)
    dst = fused.palu_l1_cb_bytes(32, 128, 512, True, 1, cores_per_head=1, rot_mode="dst")
    # c_10 goes from 2 * Wt = 16 bf16 tiles to PNHt = 1 tile
    assert legacy - dst == (16 - 1) * 2048
    assert fused.palu_l1_cb_bytes(32, 128, 512, True, 1, cores_per_head=1, rot_mode="pack") == legacy
```

Run: `source scripts/env.sh && python -m pytest tests/test_fused.py -q -k dst_mode`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'rot_mode'`

- [ ] **Step 2: L1 model**

`palu_tt/fused.py`: add `rot_mode: str = "legacy"` as the last keyword of `palu_l1_cb_bytes` and replace `n += 2 * wt * t  # c_10 AB` with:

```python
    n += (q_head_tiles if rot_mode == "dst" else 2 * wt) * t    # c_10 AB block (dst mode: peel scratch only)
```

Add `rot_mode: str = "legacy"` to `group_heads_per_core`'s keywords and forward it into every `palu_l1_cb_bytes(...)` call inside it. In `palu_tt/attention.py` the `fused.group_heads_per_core(...)` call (line 305) gains `rot_mode=rt.rot_mode`.

Run: `source scripts/env.sh && python -m pytest tests/test_fused.py -q`
Expected: PASS (all, including the allocator-report test, which uses the default).

- [ ] **Step 3: Factory `c_10` sizing and the kernel static_assert**

`palu_sdpa_decode_program_factory.cpp` line 721: replace `const uint32_t ab_tiles = 2 * Wt;` with

```cpp
    // rot_mode 2 (dst) never stages AB through c_10; the slot only has to hold one statistics block for
    // the group-mode peel (cb_l_head), which is PNHt tiles.
    const uint32_t ab_tiles = operation_attributes.rot_mode == 2 ? PNHt : 2 * Wt;
```

`palu_sdpa_flash_decode.cpp` line 531: replace the static_assert with

```cpp
    static_assert(PALU_ROT_MODE == 2 || 2 * Wt >= Sq_chunk_t, "c_10 (ab_tiles = 2 * Wt) must hold one statistics block");
```

(in mode 2 the factory sizes `c_10` to exactly `Sq_chunk_t` tiles.)

- [ ] **Step 4: Implement mode 2**

In `palu_compute_common.hpp`, wrap the existing per-row-tile loop of `palu_build_k_chunk` (from `for (uint32_t i = 0; i < Sk_chunk_t; ++i) {` to the closing brace before `k.push_back`) in `#if PALU_ROT_MODE != 2 ... #else ... #endif`, with this `#else` branch:

```cpp
    // dst mode (stage 2e): per row tile i and column-tile pair (j, j+1), one DST trip:
    //   dst0..1 = lat_i @ w[:, j..j+1]            (the A half)
    //   dst2..3 = lat_i @ w[:, j+DHt..j+DHt+1]    (the B half)
    //   dst0..1 *= cos_j, cos_j+1 ; dst2..3 *= sin_j, sin_j+1   (dest-reuse: DST -> SrcA, table -> SrcB)
    //   pack dst0, dst1 into c_1; pack dst2, dst3 onto the same tiles with L1 accumulate on.
    // No AB block in L1, no SFPU pass; two trips per row tile.
    static_assert(DHt % 2 == 0 && dst_tiles >= 4, "dst mode needs four DST tiles: A_j, A_j+1, B_j, B_j+1");
    pack_reconfig_data_format(cb_k);
    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        lat.wait_front((i + 1) * Rt);
        for (uint32_t j = 0; j < DHt; j += 2) {
            const uint32_t c = cs_tile_base + i * Wt + j;
            reconfig_data_format(cb_w, cb_lat);  // matmul: in1 -> SrcA, in0 -> SrcB
            matmul_block_init(cb_lat, cb_w, false /*transpose*/, 2 /*ct_dim*/, 1 /*rt_dim*/, Rt /*kt_dim*/);
            tile_regs_acquire();
            uint32_t in0_index = i * Rt;
            uint32_t in1_index = w_block_base + j;
            for (uint32_t kt = 0; kt < Rt; ++kt) {          // A_j, A_j+1 -> dst 0, 1
                matmul_block(cb_lat, cb_w, in0_index, in1_index, 0, false, 2, 1, Rt);
                in0_index++;
                in1_index += Wt;
            }
            in0_index = i * Rt;
            in1_index = w_block_base + j + DHt;
            for (uint32_t kt = 0; kt < Rt; ++kt) {          // B_j, B_j+1 -> dst 2, 3
                matmul_block(cb_lat, cb_w, in0_index, in1_index, 2, false, 2, 1, Rt);
                in0_index++;
                in1_index += Wt;
            }
            reconfig_data_format(cb_cs, cb_cs);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, c, 0);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, c + 1, 1);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, c + DHt, 2);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, c + DHt + 1, 3);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(0);
            pack_tile<true>(0, cb_k, j * Sk_chunk_t + i);
            pack_tile<true>(1, cb_k, (j + 1) * Sk_chunk_t + i);
            pack_reconfig_l1_acc(1);
            pack_tile<true>(2, cb_k, j * Sk_chunk_t + i);
            pack_tile<true>(3, cb_k, (j + 1) * Sk_chunk_t + i);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
    }
```

`cb_ab` stays a template parameter (unused in this branch; `[[maybe_unused]]` on the local `ab` CircularBuffer). `w.wait_front(...)`, `cs.wait_front(...)`, `k.reserve_back(...)`, `k.push_back(...)` and the `pop_latent` pop stay outside the `#if`, shared by all modes.

- [ ] **Step 5: Rebuild the host side (factory changed) and run the dst tests**

Run: `scripts/build_ttnn.sh`, then
`scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k "dst and (default_chunk or long_context or tracks_legacy)"`
Expected: PASS. Diagnostics if not: (a) PCC near 0 at every rank means the dest-reuse multiply read zeroes, i.e. the DST index was cleared before the multiply: try `reconfig_data_format_srcb(cb_cs)` instead of the two-operand reconfig, and if that does not help, replace the four `mul_reuse_dest_tiles` calls by packing `dst0..3` to `cb_ab` (sized `2 * Wt` again) and running the `pack`-mode multiplies from there (that is `pack` mode with the matmul trips unchanged, and the report records `dst` as not viable on this LLK); (b) PCC ~0.5 with `r_k` 96 only means the second `matmul_block` loop did not accumulate at `dst_index` 2: split the B half into its own acquisition (three trips per row tile) and record it.

- [ ] **Step 6: Group mode under dst, and the `r_k` 96 group shape**

Extend the `rot_mode` parametrisation of Task 2 Step 4 to `["legacy", "pack", "dst"]` and add:

```python
def test_dst_mode_three_inner_tiles_group(mesh_device):
    """r_k 96 (three latent tiles) in group-shared mode: the two matmul_block loops per trip run three
    accumulating steps each, with the per-head weight-block base applied to both halves."""
    args = _inputs_group(32, 2048, 96, 256, 4)
    per_head = _fused_group(mesh_device, *args, 128, True, heads_per_core=None, rot_mode="dst")
    shared = _fused_group(mesh_device, *args, 128, True, heads_per_core=2, rot_mode="dst")
    _assert_matches(per_head, shared, "dst: group-shared vs the stock split", bitwise=True)
    ok, msg = _group_ref_pcc(*args, shared)
    assert ok, f"vs fp32 reference: {msg}"
```

Run: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k "dst"`
Expected: PASS.

- [ ] **Step 7: Full op suite once, all modes**

Run: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q`
Expected: PASS (report the count; stage 2c's suite was 53 tests before this stage).

- [ ] **Step 8: Commit**

```bash
cd third_party/tt-metal && git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode && git -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m "palu_sdpa_decode: rot_mode dst -- matmul into DST, dest-reuse multiplies, packer-accumulate add; c_10 shrinks to the peel scratch" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs" && cd ../..
git add third_party/tt-metal palu_tt/fused.py palu_tt/attention.py tests/test_fused.py tests/test_palu_sdpa_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: dst rebuild mode, L1 model and tests" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 4: angle-table multicast per grid column

**Files:**
- Modify: `.../device/kernels/dataflow/palu_dataflow_common.hpp` (new helper)
- Modify: `.../device/kernels/dataflow/palu_reader_decode.cpp:263-269`
- Modify: `.../device/palu_sdpa_decode_program_factory.cpp:1042-1070` (runtime args)
- Test: `tests/test_palu_sdpa_device.py`

**Interfaces:**
- Consumes: `PALU_TABLE_MCAST` define (Task 1); the reader's existing runtime args `do_k_mcast, mcast_x, mcast_y0, mcast_y1, num_dests` and compile-time `k_mcast_semaphore_id`.
- Produces: `palu_read_table_mcast<cb_cs, cb_rs, barrier_threshold>(cs_tiles, rs_tiles, cs_reader, rs_reader, leader, num_dests, mcast_x, mcast_y0, mcast_y1, sem_id)`.

- [ ] **Step 1: Bit-identity test (fails first: multicast on must equal off, and today the leader flags are all zero, so it passes trivially; make the test also assert the factory reports leaders)**

Append to `tests/test_palu_sdpa_device.py`:

```python
@pytest.mark.parametrize("batch,grid", [(32, (8, 8)), (8, (8, 8)), (1, (8, 8)), (32, (8, 6))],
                         ids=["b32_full", "b8_split", "b1_split16", "b32_partial_columns"])
def test_table_mcast_is_bit_identical(mesh_device, batch, grid):
    """One leader per grid column reads the Phase 2 table and multicasts it; every layout (full columns,
    sequence splits, and an 8x6 grid at batch 32 where 32 of 48 cores are active so columns are partial)
    must produce exactly the output of the per-core read."""
    q, k_lat, u_k, v, pos, cos, sin = _inputs(batch, 2048, 32, 128)
    S = k_lat.shape[2]
    c = fused.chunk_size(S)
    extra = dict(rot_step=_dev(mesh_device, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                 trans_mat=_dev(mesh_device, fused.trans_mat_tile()))
    def run(mcast):
        out = ttnn.transformer.palu_sdpa_decode(
            _dev(mesh_device, q), _dev(mesh_device, k_lat), _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None]),
            _dev(mesh_device, fused.cs_table(cos, sin, S)[None, None]), _dev(mesh_device, v), head_dim_v=128,
            cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5, table_mcast=mcast, **extra, **_cfg(0, grid))
        return _host(out)
    assert torch.equal(run(False), run(True))
```

Run: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k table_mcast`
Expected: PASS trivially today (no leaders); it becomes the real check after Steps 2 and 3.

- [ ] **Step 2: Reader helper**

Append to `palu_dataflow_common.hpp`:

```cpp
// Phase 2 angle table and rot_step, multicast down a grid column (stage 2e). The column's leader reads
// both from DRAM into its own CB region and writes them to the same L1 addresses on the num_dests
// cores in the vertical range (x, y0..y1), then bumps their multicast semaphore; the receivers wait
// for it. Both CBs are empty on every core at this point, so every write pointer is the CB base and
// the addresses coincide. A core whose column has no other active core (num_dests == 0) reads for
// itself. Pieces of at most 32 KB per multicast call keep each NoC transaction well inside the
// hardware's burst limit.
template <uint32_t cb_cs, uint32_t cb_rs, uint32_t barrier_threshold, typename CsReaderT, typename RsReaderT>
void palu_read_table_mcast(
    uint32_t cs_tiles, uint32_t rs_tiles, const CsReaderT& cs_reader, const RsReaderT& rs_reader,
    bool leader, uint32_t num_dests, uint32_t mcast_x, uint32_t mcast_y0, uint32_t mcast_y1, uint32_t sem_id) {
    Noc noc;
    CircularBuffer cs(cb_cs), rs(cb_rs);
    cs.reserve_back(cs_tiles);
    rs.reserve_back(rs_tiles);
    const uint32_t cs_wp = cs.get_write_ptr();
    const uint32_t rs_wp = rs.get_write_ptr();
    if (leader || num_dests == 0) {
        palu_read_tiles_at<cb_cs, barrier_threshold>(cs_wp, 0, cs_tiles, cs_reader);
        palu_read_tiles_at<cb_rs, barrier_threshold>(rs_wp, 0, rs_tiles, rs_reader);
        if (num_dests > 0) {
            auto mcast = [&](uint32_t addr, uint32_t bytes) {
                constexpr uint32_t piece = 32 * 1024;
                for (uint32_t off = 0; off < bytes; off += piece) {
                    const uint32_t n = (bytes - off < piece) ? (bytes - off) : piece;
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(addr + off),
                        MulticastEndpoint{},
                        n,
                        num_dests,
                        {},
                        {.noc_x_start = mcast_x, .noc_y_start = mcast_y0, .noc_x_end = mcast_x, .noc_y_end = mcast_y1,
                         .addr = addr + off},
                        false);
                }
            };
            mcast(cs_wp, cs_tiles * get_tile_size(cb_cs));
            mcast(rs_wp, rs_tiles * get_tile_size(cb_rs));
            noc.async_write_barrier();
            constexpr uint32_t VALID = 1;
            Semaphore<> sem(sem_id);
            sem.set(VALID);
            sem.set_multicast(noc, mcast_x, mcast_y0, mcast_x, mcast_y1, num_dests);
            noc.async_write_barrier();
        }
    } else {
        Semaphore<> sem(sem_id);
        sem.wait(1);
        sem.set(0);
        noc.async_atomic_barrier();
    }
    cs.push_back(cs_tiles);
    rs.push_back(rs_tiles);
}
```

(`MulticastEndpoint`, `Semaphore<>` and `noc.async_write_multicast` are the same calls the stock K multicast in `sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:632-700` uses; copy their exact spelling if the compiler disagrees with the above.)

`palu_reader_decode.cpp`, replace the Phase 2 block (lines 263-269) with:

```cpp
    if constexpr (phase2) {
        const auto rot_step_reader = TensorAccessor(rot_step_args, rot_step_addr);
#ifdef PALU_TABLE_MCAST
        // stage 2e: the reader's multicast args (do_k_mcast, mcast_x/y0/y1, num_dests) describe the
        // table multicast here; the K multicast they were copied for never runs in the Palu envelope.
        palu_read_table_mcast<cb_cs_in, cb_rot_step, barrier_threshold>(
            Sk_chunk_t_dynamic * Wt, Wt, cs_reader, rot_step_reader, do_k_mcast, num_dests, mcast_x, mcast_y0,
            mcast_y1, k_mcast_semaphore_id);
#else
        // Chunk-relative [cos | sin] rows t = 0..c-1: the first Sk_chunk_t row tiles of cs_tab, and the
        // per-chunk rotation constants. Both are read once per core and stay resident for the run.
        palu_read_tiles<cb_cs_in, barrier_threshold>(0, Sk_chunk_t_dynamic * Wt, cs_reader);
        palu_read_tiles<cb_rot_step, barrier_threshold>(0, Wt, rot_step_reader);
#endif
    }
```

- [ ] **Step 3: Factory: column leaders**

In `palu_sdpa_decode_program_factory.cpp`, before the runtime-argument loop (`for (uint32_t i = 0; i < num_active_cores; ++i)`), add:

```cpp
    // ========== Stage 2e: table multicast leaders ==========
    // Active cores fill the grid row-major (core_group[i] = {i % grid_x, i / grid_x}), so the active
    // cores of a column are a prefix from the top: the leader is y == 0 and the receivers are y = 1..k-1.
    std::vector<uint32_t> column_active(grid_size.x, 0);
    if (table_mcast) {
        for (uint32_t i = 0; i < num_active_cores; ++i) {
            column_active[core_group[i].x] += 1;
        }
    }
```

and inside the loop, in the `else` branch of `if (use_col_major_group_indexing)` (after `core_num_in_output = ...;`):

```cpp
            if (table_mcast) {
                const uint32_t k = column_active[core.x];
                num_dests = k > 1 ? k - 1 : 0;
                do_k_mcast = (core.y == 0);
                if (num_dests > 0) {
                    auto phys_start = device->worker_core_from_logical_core(CoreCoord{core.x, 1});
                    auto phys_end = device->worker_core_from_logical_core(CoreCoord{core.x, k - 1});
                    mcast_x = phys_start.x;
                    mcast_y0 = phys_start.y;
                    mcast_y1 = phys_end.y;
                }
            }
```

(`table_mcast` is the `const bool` computed in Task 1 Step 7; move its definition above this block if it sits after it.) Confirm with `grep -n "k_mcast_semaphore_id" palu_sdpa_decode_program_factory.cpp` that the semaphore is created in `desc.semaphores` as in the stock factory (`SemaphoreDescriptor{.id = k_mcast_semaphore_id, ...}`); it is, because the factory is a copy of stock.

- [ ] **Step 4: Rebuild, run the multicast tests, then the whole op suite**

Run: `scripts/build_ttnn.sh` then `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k table_mcast`
Expected: PASS on all four layouts. A hang on `b32_partial_columns` or `b1_split16` means the multicast rectangle or `num_dests` is wrong for that layout: print `column_active`, the leader flags and the physical range from the factory with `log_debug` and compare with `device->worker_core_from_logical_core` for each receiver. A hang on every layout means the semaphore never arrives: check that receivers `reserve_back` before waiting and that `sem_id` is `k_mcast_semaphore_id` on both sides. If the multicast cannot be made to work on the partial-column layout within a day, restrict `table_mcast` in the factory to full columns (`TT_FATAL(column_active[x] == grid_size.y)` when on) and record it.

Then: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd third_party/tt-metal && git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode && git -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m "palu_sdpa_decode: table_mcast -- one Phase 2 table read per grid column, multicast to the column" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs" && cd ../..
git add third_party/tt-metal tests/test_palu_sdpa_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: angle-table multicast and its layout tests" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 5: `palu_tt.profile_fit`: the stage 2d fit as a script

**Files:**
- Create: `palu_tt/profile_fit.py`
- Test: `tests/test_profile_fit.py` (new)

**Interfaces:**
- Produces: `fit_signatures(raw: Path, rows: list[str], contexts: list[int], chunk: int = 512) -> list[dict]` with keys `signature` (`"<op> r_k=<in1 last dim> out=<out last dim>"`), `rows` (which rows contributed), `points` (`[(chunks, us_per_layer), ...]`), `fixed_us`, `per_chunk_us`, `per_item_per_chunk_us` (= `per_chunk_us / 2`, the batch-32 layout runs two head-items per core); CLI `python -m palu_tt.profile_fit --row A --row B --contexts 1024 4096 8192 [--chunk 512] [--json out.json]` printing a markdown table.

- [ ] **Step 1: Test against the committed stage 2d JSONs (fails first)**

Create `tests/test_profile_fit.py`:

```python
from pathlib import Path

from palu_tt import profile_fit

RAW = Path("docs/results/raw")


def _by_sig(rows):
    return {r["signature"]: r for r in rows}


def test_fit_reproduces_the_stage2d_table():
    """docs/results/stage2d.md section 4: stock 47.7 + 88.4 x chunks; fused r_k 32 / r_v 128
    118.1 + 114.3 x chunks; fused r_k 64 / r_v 128 129.5 + 123.1 x chunks (two significant figures)."""
    rows = profile_fit.fit_signatures(RAW, ["baseline", "palu_full_50", "palu_full_50_fk2", "palu_full_70_fk2"],
                                      [1024, 4096, 8192])
    s = _by_sig(rows)
    stock = s["SdpaDecodeDeviceOperation r_k=128 out=128"]
    assert abs(stock["fixed_us"] - 47.7) < 1.5 and abs(stock["per_chunk_us"] - 88.4) < 1.0
    f32 = s["PaluSdpaDecodeDeviceOperation r_k=32 out=128"]
    assert abs(f32["fixed_us"] - 118.1) < 1.5 and abs(f32["per_chunk_us"] - 114.3) < 1.0
    f64 = s["PaluSdpaDecodeDeviceOperation r_k=64 out=128"]
    assert abs(f64["fixed_us"] - 129.5) < 1.5 and abs(f64["per_chunk_us"] - 123.1) < 1.0
    assert abs(f32["per_item_per_chunk_us"] - f32["per_chunk_us"] / 2) < 1e-9


def test_fit_skips_missing_contexts_and_needs_two_points():
    rows = profile_fit.fit_signatures(RAW, ["baseline"], [1024, 4096, 8192])   # no stock JSON at 8192
    stock = _by_sig(rows)["SdpaDecodeDeviceOperation r_k=128 out=128"]
    assert [p[0] for p in stock["points"]] == [2, 8]
```

Run: `source scripts/env.sh && python -m pytest tests/test_profile_fit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'palu_tt.profile_fit'`

- [ ] **Step 2: Implement**

Create `palu_tt/profile_fit.py`:

```python
"""Least-squares fit of per-layer attention-op time as fixed + chunks x per_chunk (stage 2d section 4).

Reads the profile JSONs ``docs/results/raw/profile_<row>_<ctx>.json`` written by ``palu_tt.profile_report``
and fits every attention-category op *signature* (op name, K width = in1's last dim, output width) over
the contexts given. A signature can be fed by several rows: the stock op at 8k only exists inside the
naive Palu rows (a whole stock step does not allocate there), and the fused rows carry both r_k 32 and
r_k 64 layers of the same model.

    python -m palu_tt.profile_fit --row baseline --row palu_full_50 --row palu_full_50_fk2 --contexts 1024 4096 8192
"""
import argparse
import json
from pathlib import Path

RAW = Path("docs/results/raw")


def _signature(op):
    return f"{op['op']} r_k={op['in1'][-1]} out={op['out'][-1]}"


def _lstsq(points):
    """points: [(x, y), ...] -> (intercept, slope) by ordinary least squares."""
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return intercept, slope


def fit_signatures(raw, rows, contexts, chunk=512):
    per_sig = {}
    for row in rows:
        for ctx in contexts:
            p = Path(raw) / f"profile_{row}_{ctx}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for op in d["ops"]:
                if op["category"] != "attention" or op["count"] <= 0:
                    continue
                sig = per_sig.setdefault(_signature(op), {"signature": _signature(op), "rows": [], "points": []})
                if row not in sig["rows"]:
                    sig["rows"].append(row)
                sig["points"].append((ctx // chunk, 1000.0 * op["ms"] / op["count"]))   # us per layer
    out = []
    for sig in per_sig.values():
        sig["points"].sort()
        if len({x for x, _ in sig["points"]}) < 2:
            continue
        fixed, per_chunk = _lstsq(sig["points"])
        sig.update(fixed_us=fixed, per_chunk_us=per_chunk, per_item_per_chunk_us=per_chunk / 2)
        out.append(sig)
    return sorted(out, key=lambda s: s["signature"])


def table(rows):
    lines = ["| signature | rows | points (chunks: us/layer) | fixed us | per chunk us | per item per chunk us |",
             "|---|---|---|---|---|---|"]
    for s in rows:
        pts = ", ".join(f"{x}: {y:.0f}" for x, y in s["points"])
        lines.append(f"| {s['signature']} | {', '.join(s['rows'])} | {pts} | {s['fixed_us']:.1f} | "
                     f"{s['per_chunk_us']:.1f} | {s['per_item_per_chunk_us']:.1f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--row", action="append", required=True)
    p.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 8192])
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--json", type=Path, default=None, help="also write the fitted rows to this JSON file")
    a = p.parse_args()
    rows = fit_signatures(RAW, a.row, a.contexts, a.chunk)
    print(table(rows))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the test**

Run: `source scripts/env.sh && python -m pytest tests/test_profile_fit.py -q`
Expected: PASS. If the stock intercept is off by more than 1.5 us, the stage 2d fit used a different point set; print `table(rows)` and compare with `docs/results/stage2d.md` section 4; widen the tolerance only if the difference is explained by the duplicated 4k point (baseline and naive both contribute) and note it in the test.

- [ ] **Step 4: Commit**

```bash
git add palu_tt/profile_fit.py tests/test_profile_fit.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: profile_fit reproduces the stage 2d fixed + per-chunk fit from the profile JSONs" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 6: bfp4 latent flag and the L4 rows

**Files:**
- Modify: `palu_tt/bench.py`, `palu_tt/eval_ppl.py`, `palu_tt/decode_check.py` (flag), `palu_tt/model.py:63` (byte accounting)
- Test: `tests/test_palu_sdpa_device.py` (op accepts bfp4 latents), `tests/test_bench_utils.py`

**Interfaces:**
- Produces: `--latent-dtype {bfp8,bfp4}` (default `bfp8`) on the three CLIs, mapped to `latent_dtype=ttnn.bfloat8_b | ttnn.bfloat4_b` in `build_model`; result JSONs record `"latent_dtype"`.

- [ ] **Step 1: Op test with bfp4 latents (fails first if the op rejects the format)**

Append to `tests/test_palu_sdpa_device.py`:

```python
def test_bfp4_latents_are_accepted_and_close(mesh_device):
    """bfp4 K and V latents (stage 2e L4 / stage 2f): the op must accept the format and the output must
    still track the fp32 reference, though at a looser bar than bfp8 (4-bit mantissas)."""
    q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 1024, 64, 128)
    S = 1024
    c = fused.chunk_size(S)
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(mesh_device, q), _dev(mesh_device, k_lat, ttnn.bfloat4_b), _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None]),
        _dev(mesh_device, fused.cs_table(cos, sin, S)[None, None]), _dev(mesh_device, v, ttnn.bfloat4_b), head_dim_v=128,
        cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5,
        rot_step=_dev(mesh_device, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
        trans_mat=_dev(mesh_device, fused.trans_mat_tile()), **_cfg(0))
    ref = fused.reference_decode(q, k_lat, u_k, v, pos, cos, sin, HD ** -0.5)
    ok, msg = comp_pcc(ref, _host(out), 0.95)
    assert ok, msg
```

Run: `scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q -k bfp4`
Expected: PASS, or a `TT_FATAL` naming a dtype check. If the latter, relax that check in `palu_sdpa_decode_device_operation.cpp` to accept `DataType::BFLOAT4_B` alongside `BFLOAT8_B` / `BFLOAT16` for `k` and `v` only (never for `w_rot`, `cs_tab`, `rot_step`, `trans_mat`), rebuild with `scripts/build_ttnn.sh`, re-run.

- [ ] **Step 2: CLI flag and byte accounting**

In `bench.py`, `eval_ppl.py`, `decode_check.py`:

```python
    p.add_argument("--latent-dtype", choices=["bfp8", "bfp4"], default="bfp8",
                   help="tile format of the Palu latent K/V caches (default bfp8, the model's KV cache format).")
```

and map `LATENT_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}` into `build_model(..., latent_dtype=LATENT_DTYPES[a.latent_dtype])` on every `pm.build_model` call of the Palu path (bench: add `"latent_dtype": None` to `FUSED` and set it from the flag, since `bench_*` pass `**FUSED`; eval_ppl and decode_check pass it explicitly). Record `"latent_dtype": a.latent_dtype` in each result JSON. In `palu_tt/model.py` line 63 extend the stock `bpe` map with `ttnn.bfloat4_b: 0.5625`.

Host check: `source scripts/env.sh && python -m pytest tests/test_bench_utils.py tests/test_ppl_harness.py -q` → PASS.

- [ ] **Step 3: The two L4 measurements (device, gated)**

```bash
pgrep -f "palu_tt\." || true
scripts/run_device.sh python -m palu_tt.eval_ppl --row palu_full_50_l4 --kind palu --palu-dir compressed/palu_full_0.5 --latent-dtype bfp4
scripts/run_device.sh python -m palu_tt.eval_ppl --row palu_full_70_l4 --kind palu --palu-dir compressed/palu_full_0.7 --latent-dtype bfp4
scripts/run_device.sh python -m palu_tt.decode_check --row palu_full_50_fk2_l4 --palu-dir compressed/palu_full_0.5 --fused p2 --latent-dtype bfp4
scripts/run_device.sh python -m palu_tt.decode_check --row palu_full_70_fk2_l4 --palu-dir compressed/palu_full_0.7 --fused p2 --latent-dtype bfp4
scripts/run_device.sh python -m palu_tt.bench --row palu_full_50_fk2_l4 --kind palu --palu-dir compressed/palu_full_0.5 --what decode --fused p2 --latent-dtype bfp4
```

Expected: five JSONs under `docs/results/raw/` (`ppl_palu_full_50_l4.json`, `ppl_palu_full_70_l4.json`, `decode_check_palu_full_50_fk2_l4.json`, `decode_check_palu_full_70_fk2_l4.json`, `bench_decode_palu_full_50_fk2_l4.json`); about 35 minutes. Perplexity is a result either way; report the numbers as they come.

- [ ] **Step 4: Commit**

```bash
git add palu_tt/bench.py palu_tt/eval_ppl.py palu_tt/decode_check.py palu_tt/model.py tests/test_palu_sdpa_device.py docs/results/raw/ppl_palu_full_50_l4.json docs/results/raw/ppl_palu_full_70_l4.json docs/results/raw/decode_check_palu_full_50_fk2_l4.json docs/results/raw/decode_check_palu_full_70_fk2_l4.json docs/results/raw/bench_decode_palu_full_50_fk2_l4.json
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: --latent-dtype bfp4 and the L4 rows (perplexity, decode fidelity, decode step)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

(if the op needed a dtype-check change, commit the submodule first as in Task 1 Step 9.)

---

### Task 7: the measurement matrix

**Files:**
- Create (scratchpad, not committed): `stage2e_matrix.sh`
- Produce: `docs/results/raw/bench_decode_<row>.json` and `docs/results/raw/profile_<row>_<ctx>.json` for every variant

**Interfaces:**
- Consumes: the CLI flags of Tasks 1 and 6, `palu_tt.profile_report` (unchanged), `palu_tt.profile_fit`.
- Produces: rows `palu_full_{50,70}_fk2{,p,d}{,m}` (six variants x two dumps) with decode benches (1k / 4k / 8k) and profiles (1k / 4k / 8k).

- [ ] **Step 1: The queue script**

Write to the session scratchpad (`$SCRATCH/stage2e_matrix.sh`; the profiler output directory is `$SCRATCH/tracy_2e/<row>_<ctx>`):

```bash
#!/usr/bin/env bash
# Stage 2e measurement matrix. Run as: nohup bash stage2e_matrix.sh > stage2e_matrix.log 2>&1 &
set -uo pipefail
cd /home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9
source scripts/env.sh
SCRATCH="$(dirname "$0")"
gate() { while pgrep -f "palu_tt\." >/dev/null; do sleep 30; done; scripts/device_free.sh; }
for dump in 50 70; do
  for rot in legacy pack dst; do
    for mc in off on; do
      sfx="_fk2"; [[ $rot == pack ]] && sfx="${sfx}p"; [[ $rot == dst ]] && sfx="${sfx}d"; [[ $mc == on ]] && sfx="${sfx}m"
      row="palu_full_${dump}${sfx}"
      if [[ ! -f docs/results/raw/bench_decode_${row}.json ]]; then
        gate; python -m palu_tt.bench --row "$row" --kind palu --palu-dir "compressed/palu_full_0.${dump}" \
          --what decode --fused p2 --rot-mode "$rot" --table-mcast "$mc"
      fi
      for ctx in 1024 4096 8192; do
        if [[ ! -f docs/results/raw/profile_${row}_${ctx}.json ]]; then
          gate; python -m tracy -r -p -o "$SCRATCH/tracy_2e/${row}_${ctx}" --op-support-count 6000 -t 5000 \
            -m palu_tt.profile_decode --row "$row" --kind palu --palu-dir "compressed/palu_full_0.${dump}" \
            --fused p2 --rot-mode "$rot" --table-mcast "$mc" --ctx "$ctx" --steps 3
          python -m palu_tt.profile_report --dir "$SCRATCH/tracy_2e/${row}_${ctx}" --steps 3 --name "${row}_${ctx}"
        fi
      done
    done
  done
done
echo MATRIX FINISHED
```

(`palu_tt.profile_report`'s CLI is the stage 2d one: `--dir <tracy artifacts dir> --steps N --name <row>_<ctx>` writes `docs/results/raw/profile_<name>.json`; check `python -m palu_tt.profile_report --help` and adjust the flag names to what it prints before launching.) The `legacy` / `off` cells duplicate the stage 2d `_fk2` rows on purpose: a same-day re-measurement is the noise reference.

- [ ] **Step 2: Launch and monitor**

Run: `nohup bash $SCRATCH/stage2e_matrix.sh > $SCRATCH/stage2e_matrix.log 2>&1 &` then poll `tail -3 $SCRATCH/stage2e_matrix.log` until `MATRIX FINISHED` (about 2.5 hours: 12 benches x ~4 min, 36 profiles x ~3 min). A cell that fails (a hang or `TT_FATAL`) is skipped by the `set -uo pipefail` without `-e`; list the missing JSONs at the end and re-run those cells by hand through the gate.

- [ ] **Step 3: Fit every variant**

```bash
for v in "" p d m pm dm; do
  python -m palu_tt.profile_fit --row baseline --row palu_full_50 --row palu_full_50_fk2$v --row palu_full_70_fk2$v --contexts 1024 4096 8192 --json docs/results/raw/profile_fit_fk2$v.json
done
```

Expected: six fit JSONs; the `legacy`/`off` one (`profile_fit_fk2.json`) reproduces the stage 2d numbers to within the day's noise (about 1 %).

- [ ] **Step 4: Commit the JSONs (never the CSVs or logs)**

```bash
git add docs/results/raw/bench_decode_palu_full_50_fk2*.json docs/results/raw/bench_decode_palu_full_70_fk2*.json docs/results/raw/profile_palu_full_50_fk2*_*.json docs/results/raw/profile_palu_full_70_fk2*_*.json docs/results/raw/profile_fit_fk2*.json
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: decode benches and per-op profiles for the six kernel variants, with the fixed / per-chunk fits" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 8: `report --stage 2e` and the variant figure

**Files:**
- Modify: `palu_tt/report.py` (new `ROWS2E`, `_stage2e_lines`, `--stage 2e`), `palu_tt/profile_plots.py` (new `attention_ctx_variants_figure`)
- Test: `tests/test_report.py`, `tests/test_plots.py`

**Interfaces:**
- Consumes: `profile_fit.fit_signatures`, the raw JSONs of Tasks 6 and 7, `report._row_data` / `_timing` / `_fidelity` / `_load` (existing helpers at `palu_tt/report.py:143-330`).
- Produces: `docs/results/stage2e.md` generated tables above the hand-written marker; `docs/results/profile_attention_ctx_2e.png`.

- [ ] **Step 1: Host test (fails first)**

Append to `tests/test_report.py`:

```python
def test_stage2e_tables_render_from_the_raw_dir(tmp_path):
    from palu_tt import report
    out = tmp_path / "stage2e.md"
    report.main(raw=report.RAW, out=out, rows=report.ROWS2E, stage="2e")
    text = out.read_text()
    assert "| variant |" in text and "fixed us" in text          # the ablation table
    assert "palu_full_50_fk2" in text and "baseline" in text     # whole-step table
    assert "bfp4" in text                                        # the L4 table
```

Run: `source scripts/env.sh && python -m pytest tests/test_report.py -q -k stage2e`
Expected: FAIL (`AttributeError: module 'palu_tt.report' has no attribute 'ROWS2E'`).

- [ ] **Step 2: Implement**

In `palu_tt/report.py`:

```python
VARIANTS2E = [("", "legacy, table per core (stage 2 = 2d baseline)"), ("p", "pack"), ("d", "dst"),
              ("m", "legacy + table multicast"), ("pm", "pack + multicast"), ("dm", "dst + multicast")]
ROWS2E = {"baseline": "stock `tt_transformers` attention"}
for _sfx, _label in VARIANTS2E:
    ROWS2E[f"palu_full_50_fk2{_sfx}"] = f"Palu M-LRD ratio 0.5, fused Phase 2: {_label}"
    ROWS2E[f"palu_full_70_fk2{_sfx}"] = f"Palu M-LRD ratio 0.7, fused Phase 2: {_label}"
L4_ROWS2E = (("palu_full_50", "palu_full_50_l4", "palu_full_50_fk2", "palu_full_50_fk2_l4"),
             ("palu_full_70", "palu_full_70_l4", "palu_full_70_fk2", "palu_full_70_fk2_l4"))


def _stage2e_lines(raw, rows, loaded):
    from palu_tt import profile_fit
    lines = [f"# Stage 2e results: idle-resource kernel variants and bfp4 latents",
             "",
             f"Generated by `python -m palu_tt.report --stage 2e` from `docs/results/raw/*.json` at commit `{_commit()}`.",
             ""]
    # 1. ablation: the fit per variant
    lines += ["## Kernel variants: fixed and per-chunk cost per layer (batch 32)", "",
              "| variant | signature | fixed us | per chunk us | per item per chunk us | points (chunks: us/layer) |",
              "|---|---|---|---|---|---|"]
    for sfx, label in VARIANTS2E:
        fits = profile_fit.fit_signatures(raw, ["baseline", "palu_full_50", f"palu_full_50_fk2{sfx}", f"palu_full_70_fk2{sfx}"],
                                          [1024, 4096, 8192])
        for s in fits:
            if s["signature"].startswith("PaluSdpaDecode") or sfx == "":
                pts = ", ".join(f"{x}: {y:.0f}" for x, y in s["points"])
                lines.append(f"| {label} | {s['signature']} | {s['fixed_us']:.1f} | {s['per_chunk_us']:.1f} | "
                             f"{s['per_item_per_chunk_us']:.1f} | {pts} |")
    lines.append("")
    # 2. whole step
    stock_dec = _cell(loaded["baseline"]["decode"], "decode") if "baseline" in loaded else None
    lines += ["## Decode step, batch 32 (min ms; `vs stock` = stock / row at the same context)", "",
              "| row | 1024 | 4096 | 8192 | vs stock 1k | vs stock 4k |", "|---|---|---|---|---|---|"]
    for row in rows:
        if row not in loaded or "decode" not in loaded[row]:
            continue
        cells = [_min_only(_cell(loaded[row]["decode"], ctx)) for ctx in (1024, 4096, 8192)]
        vs = [_vs_stock(stock_dec, loaded[row]["decode"], ctx) for ctx in (1024, 4096)]
        lines.append(f"| {row} | {' | '.join(cells)} | {' | '.join(vs)} |")
    lines.append("")
    # 3. fidelity of the shipped variant and the L4 rows
    lines += ["## Decode fidelity (`decode_check`, 63 teacher-forced steps)", "",
              "| row | PCC min / mean | argmax agree |", "|---|---|---|"]
    for row in sorted(r for r in loaded if "decode_check" in loaded[r] and ("fk2" in r)):
        lines.append(f"| {row} | {_fidelity(loaded[row]['decode_check'])} |")
    lines.append("")
    # 4. bfp4 latents
    lines += ["## bfp4 latents (L4): the same dumps with the latent caches stored in bfp4", "",
              "| dump | perplexity bfp8 | perplexity bfp4 | KV B/token/layer/chip bfp8 -> bfp4 | decode PCC min bfp8 -> bfp4 | decode 4k ms bfp8 -> bfp4 |",
              "|---|---|---|---|---|---|"]
    for base, l4, fk2, fk2_l4 in L4_ROWS2E:
        ppl8 = _load(raw, f"ppl_{base}"); ppl4 = _load(raw, f"ppl_{l4}")
        chk8 = _load(raw, f"decode_check_{fk2}"); chk4 = _load(raw, f"decode_check_{fk2_l4}")
        b8 = _load(raw, f"bench_decode_{fk2}"); b4 = _load(raw, f"bench_decode_{fk2_l4}")
        kv8 = _kv_bytes({"bench": b8}) if b8 else None
        kv4 = _kv_bytes({"bench": b4}) if b4 else None
        lines.append(f"| {base} | {_fmt(ppl8['ppl'], 3) if ppl8 else '-'} | {_fmt(ppl4['ppl'], 3) if ppl4 else '-'} | "
                     f"{kv8 if kv8 else '-'} -> {kv4 if kv4 else '-'} | "
                     f"{_fmt(chk8['pcc_min'], 4) if chk8 else '-'} -> {_fmt(chk4['pcc_min'], 4) if chk4 else '-'} | "
                     f"{_min_only(_cell(b8, 4096)) if b8 else '-'} -> {_min_only(_cell(b4, 4096)) if b4 else '-'} |")
    lines.append("")
    return lines
```

Wire it like `_stage2c_lines`: in `main()` add a `stage == "2e"` branch that loads the rows of `ROWS2E` plus the L4 rows with the existing `_row_data`, calls `_stage2e_lines`, and hands the lines to `_finish(lines, out)` (which preserves everything below the hand-written marker); add `"2e"` to the `--stage` choices with `OUT` = `docs/results/stage2e.md` and `rows = ROWS2E`. The helper signatures used above (`_cell(bench, key)`, `_min_only(cell)`, `_vs_stock(stock_dec, dec, ctx)`, `_fidelity(chk)`, `_kv_bytes(row_files)`, `_fmt(v, nd)`, `_load(raw, name)`) are the existing ones at `palu_tt/report.py:143-383`; read them before wiring and adapt the argument shapes to what they take (`_kv_bytes` reads the layer-0 KV bytes out of a bench JSON).

`palu_tt/profile_plots.py`: add

```python
def attention_ctx_variants_figure(out_png, dump="palu_full_50"):
    """Attention block per step against context for the six stage 2e variants of one dump, with stock."""
    variants = [("", "legacy"), ("p", "pack"), ("d", "dst"), ("m", "legacy+mcast"), ("pm", "pack+mcast"), ("dm", "dst+mcast")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, color in (("baseline", ROW_COLOR["baseline"]),):
        xs = [ctx // 1024 for ctx in CONTEXTS if _load(name, ctx)]
        ys = [_cat_ms(_load(name, ctx), "attention") for ctx in CONTEXTS if _load(name, ctx)]
        ax.plot(xs, ys, marker="o", color=color, label="stock", linewidth=2)
    for i, (sfx, label) in enumerate(variants):
        row = f"{dump}_fk2{sfx}"
        xs = [ctx // 1024 for ctx in CONTEXTS if _load(row, ctx)]
        ys = [_cat_ms(_load(row, ctx), "attention") for ctx in CONTEXTS if _load(row, ctx)]
        if xs:
            ax.plot(xs, ys, marker="o", label=label, linewidth=1.6, linestyle="-" if "m" not in sfx else "--",
                    color=plt.cm.viridis(i / 6))
    ax.set_xticks([c // 1024 for c in CONTEXTS]); ax.set_xticklabels([f"{c // 1024}k" for c in CONTEXTS])
    ax.set_ylabel("attention ms per decode step (device)"); ax.set_xlabel("context")
    ax.set_title(f"{dump}: attention per step, six kernel variants vs stock"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
```

and call it from `main()` for `profile_attention_ctx_2e.png`. Add to `tests/test_plots.py` a call that renders it into `tmp_path` and asserts the file exists (mirroring the existing figure tests there).

- [ ] **Step 3: Generate**

Run: `source scripts/env.sh && python -m pytest tests/test_report.py tests/test_plots.py tests/test_profile_report.py -q && python -m palu_tt.report --stage 2e && python -m palu_tt.profile_plots`
Expected: tests PASS; `docs/results/stage2e.md` written with the four generated tables; `docs/results/profile_attention_ctx_2e.png` written.

- [ ] **Step 4: Commit**

```bash
git add palu_tt/report.py palu_tt/profile_plots.py tests/test_report.py tests/test_plots.py docs/results/stage2e.md docs/results/profile_attention_ctx_2e.png
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: report --stage 2e (ablation fit, whole step, fidelity, bfp4) and the variants figure" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 9: choose the shipped variant, verify it, write the results document

**Files:**
- Modify: `palu_tt/convert.py` (`PaluRuntime` defaults), `.../palu_sdpa_decode.hpp` and `_nanobind.cpp` (op defaults), `palu_tt/bench.py` / `decode_check.py` / `profile_decode.py` (flag defaults), `README.md` (flags)
- Modify: `docs/results/stage2e.md` (hand-written analysis below the marker)
- Produce: `docs/results/raw/decode_check_palu_full_50_fk2<sfx>.json`, `..._70_fk2<sfx>.json`, `docs/results/raw/decode_check_glrd_50_fk2g<sfx>.json`, `docs/results/raw/bench_decode_glrd_50_fk2g<sfx>.json`

**Interfaces:**
- Consumes: the ablation table (Task 8), S1 to S5 from the spec.

- [ ] **Step 1: Pick**

From `docs/results/stage2e.md`'s ablation and whole-step tables choose the variant with the lowest `palu_full_50` decode step at 4k among those whose op tests passed in Tasks 2 to 4 (the matrix rows exist only for variants that ran). Record the choice and the runner-up with their 1k / 4k / 8k numbers in the task report. If no variant improves on `legacy` by more than 1 % at 4k, the shipped default stays `legacy` and the report says so.

- [ ] **Step 2: Defaults**

Set the chosen `rot_mode` and `table_mcast` as the defaults in: `PaluRuntime` (`convert.py`), `build_model` (`model.py`), the three CLIs' `default=` values, and the op (`palu_sdpa_decode.hpp` default argument and `nanobind` `nb::arg(...) = ...`), then `scripts/build_ttnn.sh`. Update `tests/test_convert.py`'s Task 1 test to the new defaults and `test_rot_mode_legacy_is_the_default_and_bogus_is_rejected` to compare `rot_mode=<shipped>` against the default. `README.md`: document `--rot-mode`, `--table-mcast`, `--latent-dtype` next to `--group-heads` and state the shipped defaults.

- [ ] **Step 3: S1 on the shipped variant (device, gated)**

```bash
scripts/run_device.sh python -m pytest tests/test_palu_sdpa_device.py -q
scripts/run_device.sh python -m pytest tests/test_attention_device.py -q -k "fused or glrd"
scripts/run_device.sh python -m palu_tt.decode_check --row palu_full_50_fk2<sfx> --palu-dir compressed/palu_full_0.5 --fused p2 --group-heads 1
scripts/run_device.sh python -m palu_tt.decode_check --row palu_full_70_fk2<sfx> --palu-dir compressed/palu_full_0.7 --fused p2 --group-heads 1
scripts/run_device.sh python -m palu_tt.decode_check --row glrd_50_fk2g<sfx> --palu-dir compressed/glrd_0.5_f256 --fused p2 --group-heads 2
scripts/run_device.sh python -m palu_tt.bench --row glrd_50_fk2g<sfx> --kind palu --palu-dir compressed/glrd_0.5_f256 --what decode --fused p2 --group-heads auto
```

(with the shipped flags now the defaults, `<sfx>` names the rows; `--rot-mode`/`--table-mcast` need not be passed.) Expected: op and attention suites PASS; `decode_check` `pcc_min >= 0.99` on all three; the G-LRD bench at 4k within noise of `bench_decode_glrd_50_fk2g.json` or better.

- [ ] **Step 4: Regenerate the tables and write the analysis**

`python -m palu_tt.report --stage 2e` (the fidelity table now includes the shipped rows). Below the hand-written marker in `docs/results/stage2e.md` write, in this order and with the numbers from the tables: (1) the verdict against S1 to S5, one line each, met or missed with the number; (2) the attribution: which hypothesis held: the fixed term with and without multicast, the per-chunk term per rotation mode, in microseconds and as a share of the stage 2d excess; (3) what the levers did not deliver and why, including any mode that failed its tests; (4) the bfp4 result in two sentences and the pointer to stage 2f; (5) caveats: same-day noise from the duplicated `legacy` cell, the attention-only bench never used, the double rounding of the accumulating pack quantified from the op-level PCC vs legacy. Under 900 words; every number must appear in a generated table or a raw JSON.

- [ ] **Step 5: Commit, then merge**

```bash
cd third_party/tt-metal && git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode && git -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m "palu_sdpa_decode: ship rot_mode <shipped> / table_mcast <shipped> as the defaults (stage 2e)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs" && cd ../..
git add third_party/tt-metal palu_tt/convert.py palu_tt/model.py palu_tt/bench.py palu_tt/decode_check.py palu_tt/profile_decode.py README.md tests/test_convert.py tests/test_palu_sdpa_device.py docs/results/stage2e.md docs/results/raw/decode_check_palu_full_50_fk2*.json docs/results/raw/decode_check_palu_full_70_fk2*.json docs/results/raw/decode_check_glrd_50_fk2g*.json docs/results/raw/bench_decode_glrd_50_fk2g*.json
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2e: shipped variant, S1 verification, results document" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

The controller then runs the final whole-branch review and merges (fast-forward of `main`, as every stage so far), and publishes the results page from `docs/results/stage2e.md` with the figure.

---

## Self-review

- **Spec coverage.** L1 and L2: Tasks 2 and 3 (both modes, tests at every calibrated rank, group mode, the `r_k` 96 two-loop case). L3: Task 4 (helper, reader, factory leaders, four layouts, partial columns). L4: Task 6. S1: Tasks 2 to 4 and 9 Step 3. S2 / S3: Tasks 5, 7, 8 (fit per variant). S4: Tasks 7 and 9. S5: Tasks 6 and 8. Report items 1 to 5 of spec section 6: Task 8 tables 1 to 4 and Task 9 Step 4. Risks: dest-reuse probe = Task 3 Step 5's diagnostics and fallbacks; fixed-cost hypothesis = the `m` variants in Task 7 read against S2; precision = `test_rot_mode_tracks_legacy_closely` and the fidelity table; L1 = Task 3's model and factory sizing; multicast addressing = Task 4 Step 4's diagnostics and the full-column fallback. The spec's optional chunk-level init hoisting for `pack` mode is deliberately not implemented (it needs 224 KB more L1 and the row-tile form already removes the SFPU pass); Task 9's analysis names it as the next lever if `pack` wins on time.
- **Placeholders.** None: every code step has its code; the `<sfx>` in Task 9 is the variant chosen in its Step 1, and the report-helper signatures are pinned to `palu_tt/report.py` line ranges for the implementer to read.
- **Type consistency.** `rot_mode` is a `str` in Python and the host wrapper, `uint32_t` in the op attributes and the define; `table_mcast` is `bool` everywhere; `palu_l1_cb_bytes(..., rot_mode="legacy")` and `group_heads_per_core(..., rot_mode="legacy")` are keyword parameters used with those names in Tasks 3 and 9; `profile_fit.fit_signatures(raw, rows, contexts, chunk=512)` is called with the same shape in Tasks 5, 7 and 8; `_fused(..., rot_mode=, table_mcast=)` and `_fused_group(..., rot_mode=, table_mcast=)` are extended in Task 1 and used in Tasks 2 to 4 and 6.
