# Palu on TT — Stage 2: Fused K-Reconstruction Decode Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new ttnn op, `ttnn.transformer.palu_sdpa_decode`, that reconstructs and RoPE-rotates K from the Palu latent cache inside the flash-decode kernel, wired into `PaluAttention` behind a runtime flag, with the stage-1 harness re-run to show calibrated Palu rows decoding no slower than stock at 1k/4k and running at 8k.

**Architecture:** The op is a verbatim copy of tt-metal's `sdpa_decode` (flash decode) in the tt-metal submodule, renamed, with one addition: per K chunk the compute kernel does `AB = lat @ [u_k | u_k·T]` (one small matmul), then `K_rot[i,j] = AB[i,j]⊙cos + AB[i,j+4]⊙sin` straight into the K circular buffer, after which the stock QK/softmax/QK·V/tree-reduction code runs untouched. Phase 1 streams the absolute `[cos | sin]` rows per chunk from DRAM (the checkpoint). Phase 2 uses chunk-relative angles for K (the first `c` table rows, resident in L1) and instead rotates the query by `R_{-c}` once per chunk in fp32 on core, so the table is read once per core; the scores are identical because 2-D rotations commute. The host side (`palu_tt/fused.py`, `convert.py`, `attention.py`) builds the packed weight `w_rot`, the tables, and switches `forward_decode` to the fused op when `PaluRuntime.use_fused_k` is set.

**Tech Stack:** tt-metal v0.78.0 (C++ program factory + RISC-V kernels, JIT-compiled), nanobind, ttnn Python, torch (host reference), pytest. Spec: `docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md` (read it first; §3 op interface, §4 CB plan, §5 recurrence, §6 host integration).

## Global Constraints

Copied from the spec and the stage-1 working rules. Every task inherits these.

- **Shared server.** No sudo; no firmware/driver/hugepage changes. Every device command goes through `scripts/run_device.sh <cmd>` (it refuses to run when the n300 is busy). CPU work is `nice -n 10` with at most 12 threads; builds use `CMAKE_BUILD_PARALLEL_LEVEL=10`. Never read or write anything under `/home/user01/tt-data`.
- **Environment.** `source scripts/env.sh && source .venv/bin/activate` before anything (sets `TT_METAL_HOME`, `PYTHONPATH`, `MESH_DEVICE=N300`, `HF_HUB_OFFLINE=1`). Work from the repo root `/home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9`.
- **Two git repos.** tt-metal work is committed *inside the submodule* on branch `palu-sdpa` (cut from the pinned `v0.78.0`, commit `a3a9fb4229a`): `git -C third_party/tt-metal add <files> && git -C third_party/tt-metal -c user.name=ryan -c user.email=ryankert01@gmail.com commit`. The superproject then records the new submodule pointer with `git add third_party/tt-metal`. **Never** `git add third_party/palu` (its working tree is patched at run time and must stay uncommitted). Never commit `docs/results/raw/*.log` or `.superpowers/`.
- **Commit trailers** (both repos), verbatim as the last two lines of every commit message:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs`. Commit as `git -c user.name=ryan -c user.email=ryankert01@gmail.com commit ...`.
- **Build loop.** Kernels (`*.cpp` under `device/kernels/`) are JIT-compiled at first op run: no host rebuild. Host C++ (program factory, device operation, nanobind, CMake lists) needs `scripts/build_ttnn.sh` (Task 1 creates it; ~10-25 min incremental). A kernel compile error surfaces as a Python exception whose message names the failing kernel and its log file.
- **Op envelope (spec §3), asserted on the host:** head_dim 128; `r_k ∈ {32, 64, 96}`; `r_v ≤ 128`; batch ≤ 32; causal; non-paged; `cur_pos_tensor` required; no mask tensor, sink, sliding window, block padding, share_cache; Q in TILE layout; `w_rot` `[1, n_kv, r_k, 256]` bf16 TILE; `cs_tab` `[1, 1, S_cs ≥ S, 256]` bf16 TILE; `rot_step` `[1, 1, 32, 256]` fp32 TILE and `trans_mat` `[1, 1, 32, 32]` bf16 TILE, optional together, their presence selects Phase 2; Phase 2 requires `fp32_dest_acc_en=True`; the 16-row Q tile optimisation is off.
- **CB plan (spec §4):** `c_1` K_rot (bf16, stock size); `c_3` causal mask (stock, untouched); `c_4` w_rot resident (`Rt × 8` bf16 tiles, `+1` T tile in Phase 2); `c_9` latent chunk (`Sk_chunk_t_cb_size × Rt × 2` tiles, latent dtype); `c_10` one AB row block, ping-pong (`2 × 8` bf16 tiles); `c_12` Phase 2 rotated-query state (`3 × PNHt × DHt` fp32 tiles; the stock zero tile is dropped); `c_13` Phase 1: streamed cs chunk ×2 / Phase 2: resident chunk-relative rows (`Sk_chunk_t_cb_size × 8` bf16 tiles); `c_14` Phase 2 `rot_step` (8 fp32 tiles). Everything else exactly as stock.
- **K tile order in `c_1`:** tile for (seq-row-tile i, head-dim-tile j) sits at index `j * Sk_chunk_t + i` (column-major over the chunk) — the order `matmul_blocks(cb_q_in, cb_k_in, ..., transpose=true)` expects and the stock reader produces.
- **Names.** Python op: `ttnn.transformer.palu_sdpa_decode`. C++: `ttnn::transformer::palu_sdpa_decode`, `ttnn::prim::palu_sdpa_decode`, `ttnn::prim::PaluSdpaDecodeDeviceOperation`, `PaluSdpaDecodeParams`, `PaluSdpaDecodeInputs`. State-dict key `layers.{i}.attention.palu.w_rot`. `CONVERT_VERSION = 3`. Runtime flags `PaluRuntime.use_fused_k`, `PaluRuntime.cs_recurrence`. Compute defines `PALU_DST_TILES`, `PALU_PHASE2`.
- **Accuracy bars.** Op-level PCC ≥ 0.99 vs the fp32 reference and ≥ 0.995 vs the stock MLA op on host-rotated K; attention-level PCC ≥ 0.99 vs the naive path with byte-identical caches; model-level decode fidelity (`decode_check`) min PCC ≥ 0.99 and argmax agreement ≥ 0.95; decode-step timings from `palu_tt.bench` (`min_ms` is the quoted number).
- **Tests are run with the same commands as stage 1:** CPU `nice -n 10 pytest tests/<file> -q`; device `scripts/run_device.sh pytest tests/<file> -q` (the `mesh_device` fixture in `tests/conftest.py` opens the N300 mesh once per session; use `-k` to select tests).

## File structure

**tt-metal submodule** (`third_party/tt-metal/ttnn/cpp/ttnn/operations/transformer/`):
- `palu_sdpa_decode/palu_sdpa_decode.{hpp,cpp}` — public op (`ttnn::transformer::palu_sdpa_decode`): chunk-size choice, compute-config defaults, prim call.
- `palu_sdpa_decode/palu_sdpa_decode_nanobind.{hpp,cpp}` — Python binding `ttnn.transformer.palu_sdpa_decode`.
- `palu_sdpa_decode/device/palu_sdpa_decode_device_operation_types.hpp` — `PaluSdpaDecodeParams` (+ `cs_recurrence`), `PaluSdpaDecodeInputs` (+ `w_rot`, `cs_tab`, `rot_step`, `trans_mat`).
- `palu_sdpa_decode/device/palu_sdpa_decode_device_operation.{hpp,cpp}` — validation (envelope), output spec, prim wrapper.
- `palu_sdpa_decode/device/palu_sdpa_decode_program_factory.cpp` — CBs, compile-time/runtime args, kernel descriptors.
- `palu_sdpa_decode/device/kernels/dataflow/palu_reader_decode.cpp` — reads Q, latent chunks, w_rot (+ T tile), cs rows (streamed or resident), rot_step, V.
- `palu_sdpa_decode/device/kernels/dataflow/palu_dataflow_common.hpp` — `palu_read_tiles`, `read_w_rot`, `read_palu_chunks`.
- `palu_sdpa_decode/device/kernels/dataflow/palu_writer_decode.cpp` — verbatim stock writer (mask generation, output, tree reduction traffic).
- `palu_sdpa_decode/device/kernels/compute/palu_sdpa_flash_decode.cpp` — stock compute loop plus the K_rot build per chunk.
- `palu_sdpa_decode/device/kernels/compute/palu_compute_common.hpp` — `palu_build_k_chunk`, `palu_init_q_state`, `palu_step_q`.
- `sources.cmake`, `CMakeLists.txt`, `transformer_nanobind.cpp` — registration.

**This repo:**
- `scripts/build_ttnn.sh` — niced incremental tt-metal rebuild + install of `_ttnn.so`.
- `palu_tt/fused.py` — `rope_transformation_matrix`, `trans_mat_tile`, `w_rot_from_u_k`, `chunk_size`, `cos_sin_tables`, `cs_table`, `rot_step_table`, `reference_decode`.
- `palu_tt/convert.py` — `w_rot` in `palu_layer_tensors`, `CONVERT_VERSION = 3`, runtime flags, `PaluConfig.device_tables`.
- `palu_tt/attention.py` — fused branch in `forward_decode`, `w_rot`/tables upload.
- `palu_tt/model.py`, `bench.py`, `decode_check.py` (new), `report.py` — flags, rows, decode fidelity, `docs/results/stage2.md`.
- `tests/test_palu_sdpa_device.py` (new), `tests/test_fused.py` (new, CPU), `tests/test_convert.py`, `tests/test_attention_device.py`, `tests/test_report.py` (extended).
- `docs/results/stage2.md`, `docs/results/raw/*.json` (new rows), `README.md`.

---

### Task 1: Copy flash-decode into a registered `palu_sdpa_decode` op (verbatim clone)

Outcome: `ttnn.transformer.palu_sdpa_decode(q, k, v, head_dim_v=..., cur_pos_tensor=..., ...)` exists, is built, and returns exactly what `ttnn.transformer.flash_multi_latent_attention_decode` returns. No behaviour change yet; this is the checkpoint every later kernel change is bisected against.

**Files:**
- Create (submodule): everything under `ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/` listed in *File structure* except the two `palu_*_common.hpp` headers.
- Modify (submodule): `ttnn/cpp/ttnn/operations/transformer/sources.cmake`, `.../transformer/CMakeLists.txt`, `.../transformer/transformer_nanobind.cpp`.
- Create: `scripts/build_ttnn.sh`, `tests/test_palu_sdpa_device.py`.
- Modify: `README.md` (one paragraph on the submodule branch and build script).

**Interfaces:**
- Produces: Python `ttnn.transformer.palu_sdpa_decode(input_tensor_q, input_tensor_k, input_tensor_v, head_dim_v, *, is_causal=True, attn_mask=None, cur_pos=[], cur_pos_tensor=None, attention_sink=None, scale=None, sliding_window_size=None, memory_config=None, program_config=None, compute_kernel_config=None)` — the MLA signature, renamed. Task 3 changes it.
- Produces: `scripts/build_ttnn.sh` (no args).

- [ ] **Step 1: Branch the submodule**

```bash
cd /home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9
git -C third_party/tt-metal status --short | head   # expect clean (ignore untracked build dirs)
git -C third_party/tt-metal checkout -b palu-sdpa    # from the detached v0.78.0 checkout (a3a9fb4229a)
```

- [ ] **Step 2: Copy the op directory and rename every symbol**

```bash
cd third_party/tt-metal/ttnn/cpp/ttnn/operations/transformer
mkdir -p palu_sdpa_decode/device/kernels/compute palu_sdpa_decode/device/kernels/dataflow
cp sdpa_decode/sdpa_decode.hpp                                   palu_sdpa_decode/palu_sdpa_decode.hpp
cp sdpa_decode/sdpa_decode.cpp                                   palu_sdpa_decode/palu_sdpa_decode.cpp
cp sdpa_decode/sdpa_decode_nanobind.hpp                          palu_sdpa_decode/palu_sdpa_decode_nanobind.hpp
cp sdpa_decode/sdpa_decode_nanobind.cpp                          palu_sdpa_decode/palu_sdpa_decode_nanobind.cpp
cp sdpa_decode/device/sdpa_decode_device_operation.hpp           palu_sdpa_decode/device/palu_sdpa_decode_device_operation.hpp
cp sdpa_decode/device/sdpa_decode_device_operation.cpp           palu_sdpa_decode/device/palu_sdpa_decode_device_operation.cpp
cp sdpa_decode/device/sdpa_decode_device_operation_types.hpp     palu_sdpa_decode/device/palu_sdpa_decode_device_operation_types.hpp
cp sdpa_decode/device/sdpa_decode_program_factory.cpp            palu_sdpa_decode/device/palu_sdpa_decode_program_factory.cpp
cp sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp      palu_sdpa_decode/device/kernels/compute/palu_sdpa_flash_decode.cpp
cp sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp     palu_sdpa_decode/device/kernels/dataflow/palu_reader_decode.cpp
cp sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp     palu_sdpa_decode/device/kernels/dataflow/palu_writer_decode.cpp
cd palu_sdpa_decode
# types and classes
grep -rl "SdpaDecode" . | xargs sed -i 's/SdpaDecodeDeviceOperation/PaluSdpaDecodeDeviceOperation/g; s/SdpaDecodeParams/PaluSdpaDecodeParams/g; s/SdpaDecodeInputs/PaluSdpaDecodeInputs/g'
# local includes
grep -rl '#include "' . | xargs sed -i 's|#include "sdpa_decode_device_operation|#include "palu_sdpa_decode_device_operation|; s|#include "device/sdpa_decode_device_operation|#include "device/palu_sdpa_decode_device_operation|; s|#include "sdpa_decode\.hpp"|#include "palu_sdpa_decode.hpp"|; s|#include "sdpa_decode_nanobind\.hpp"|#include "palu_sdpa_decode_nanobind.hpp"|'
# the prim function
sed -i 's/^Tensor sdpa_decode(/Tensor palu_sdpa_decode(/' device/palu_sdpa_decode_device_operation.hpp device/palu_sdpa_decode_device_operation.cpp
sed -i 's/ttnn::prim::sdpa_decode(/ttnn::prim::palu_sdpa_decode(/g' palu_sdpa_decode.cpp
# kernel path and file names
sed -i 's|"ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/"|"ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/device/kernels/"|; s|dataflow/reader_decode_all.cpp|dataflow/palu_reader_decode.cpp|; s|dataflow/writer_decode_all.cpp|dataflow/palu_writer_decode.cpp|; s|compute/sdpa_flash_decode.cpp|compute/palu_sdpa_flash_decode.cpp|' device/palu_sdpa_decode_program_factory.cpp
# kernels keep sharing the stock helper headers, by full path
sed -i 's|#include "dataflow_common.hpp"|#include "ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"|' device/kernels/dataflow/*.cpp
# nanobind entry point
sed -i 's/bind_sdpa_decode/bind_palu_sdpa_decode/' palu_sdpa_decode_nanobind.hpp palu_sdpa_decode_nanobind.cpp
grep -rn "sdpa_decode" . | grep -v "palu_sdpa_decode\|operations/transformer/sdpa_decode/device/kernels" ; echo "(the list above must be empty)"
```

- [ ] **Step 3: Keep only the MLA entry point, renamed**

In `palu_sdpa_decode/palu_sdpa_decode.hpp` delete the declarations of `scaled_dot_product_attention_decode`, `paged_scaled_dot_product_attention_decode` and `paged_flash_multi_latent_attention_decode`; rename `flash_multi_latent_attention_decode` to `palu_sdpa_decode`. The header then declares exactly:

```cpp
namespace ttnn::transformer {

ttnn::Tensor palu_sdpa_decode(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const std::optional<const Tensor>& input_tensor_v,
    uint32_t head_dim_v,
    bool is_causal = true,
    const std::optional<const Tensor>& attn_mask = std::nullopt,
    const std::vector<uint32_t>& cur_pos = std::vector<uint32_t>(),
    const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
    const std::optional<const Tensor>& attention_sink = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
```

In `palu_sdpa_decode.cpp` delete the three other function definitions (keep `get_chunk_size`, the `kDefault*` constants may go too) and rename the remaining `flash_multi_latent_attention_decode` definition to `palu_sdpa_decode`. Its body is unchanged and ends in `return ttnn::prim::palu_sdpa_decode(...)` with the same 19 arguments as the MLA function passed.

In `palu_sdpa_decode_nanobind.cpp` keep one `bind_function` (the MLA one), renamed:

```cpp
void bind_palu_sdpa_decode(nb::module_& mod) {
    const auto* const doc =
        R"doc(
        Palu fused-K flash decode (stage 2 of the Palu-on-TT study). Task 1 state: a verbatim clone of
        flash_multi_latent_attention_decode; the K-reconstruction inputs are added in Task 3.
        )doc";

    ttnn::bind_function<"palu_sdpa_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::palu_sdpa_decode,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v") = nb::none(),
        nb::arg("head_dim_v").noconvert(),
        nb::kw_only(),
        nb::arg("is_causal").noconvert() = true,
        nb::arg("attn_mask") = nb::none(),
        nb::arg("cur_pos") = nb::cast(std::vector<uint32_t>()),
        nb::arg("cur_pos_tensor") = nb::none(),
        nb::arg("attention_sink") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("sliding_window_size") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}
```

- [ ] **Step 4: Register the sources**

`sources.cmake`: inside `set(TTNN_OP_TRANSFORMER_SRCS ...)` add after the `sdpa_decode/sdpa_decode.cpp` line:

```cmake
    palu_sdpa_decode/device/palu_sdpa_decode_device_operation.cpp
    palu_sdpa_decode/device/palu_sdpa_decode_program_factory.cpp
    palu_sdpa_decode/palu_sdpa_decode.cpp
```

and inside `set(TTNN_OP_TRANSFORMER_NANOBIND_SRCS ...)` after `sdpa_decode/sdpa_decode_nanobind.cpp`:

```cmake
    palu_sdpa_decode/palu_sdpa_decode_nanobind.cpp
```

`CMakeLists.txt`: in the `file(GLOB_RECURSE kernels ...)` list add `palu_sdpa_decode/device/kernels/*.hpp`; in the `FILE_SET api ... FILES` list add `palu_sdpa_decode/palu_sdpa_decode.hpp` after `sdpa_decode/sdpa_decode.hpp`.

`transformer_nanobind.cpp`: add `#include "palu_sdpa_decode/palu_sdpa_decode_nanobind.hpp"` next to the sdpa_decode include, and `bind_palu_sdpa_decode(mod);` right after `bind_sdpa_decode(mod);`.

- [ ] **Step 5: Build script and build**

Create `scripts/build_ttnn.sh`:

```bash
#!/usr/bin/env bash
# Incremental tt-metal rebuild after host-side C++ edits (program factory, device op, nanobind,
# CMake lists). Kernels under device/kernels/ are JIT-compiled at first op run and need no rebuild.
# The `install` target is what build_metal.sh runs: it also copies the Python extension into the
# source tree (third_party/tt-metal/ttnn/ttnn/_ttnn.so), which is the file the venv imports.
# Shared server: niced, 10 parallel jobs.
set -euo pipefail
cd "$(dirname "$0")/../third_party/tt-metal"
nice -n 10 env CMAKE_BUILD_PARALLEL_LEVEL=10 cmake --build build --target install
```

```bash
chmod +x scripts/build_ttnn.sh
source scripts/env.sh && source .venv/bin/activate
scripts/build_ttnn.sh 2>&1 | tail -20      # first run after adding files reconfigures cmake; 10-25 min
python -c "import ttnn; print(ttnn.transformer.palu_sdpa_decode)"
```
Expected: a bound function object is printed (no AttributeError). If the build fails on the `INTERFACE_HEADER_SETS_TO_VERIFY` step, the new public header is missing an include it needs standalone; fix the header rather than the verification.

- [ ] **Step 6: Write the clone test**

Create `tests/test_palu_sdpa_device.py`:

```python
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
```

- [ ] **Step 7: Run it**

```bash
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q
```
Expected: 2 passed. The first call JIT-compiles the three copied kernels (a minute or two).

- [ ] **Step 8: README and commits**

Add to `README.md` under the build/setup section:

```markdown
### Stage 2: fused kernel (tt-metal submodule branch)

The fused K-reconstruction decode op lives in the tt-metal submodule on branch `palu-sdpa`
(cut from `v0.78.0`); the superproject pins that branch's commit. Until the fork
`ryankert01/tt-metal` exists and the pointer is pushed there, a fresh clone cannot fetch the
submodule commit -- the working environment on the shared box has it. After editing host-side
C++ (program factory, op, bindings) run `scripts/build_ttnn.sh`; kernel-only edits are picked up
at the next op run (JIT).
```

```bash
cd third_party/tt-metal
git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode ttnn/cpp/ttnn/operations/transformer/sources.cmake ttnn/cpp/ttnn/operations/transformer/CMakeLists.txt ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "palu_sdpa_decode: verbatim copy of flash decode (MLA entry point) as a separate op

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
cd ../..
git add third_party/tt-metal scripts/build_ttnn.sh tests/test_palu_sdpa_device.py README.md
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2 Task 1: palu_sdpa_decode clone op, build script, clone test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 2: Host-side builders and torch reference (`palu_tt/fused.py`, converter v3)

Outcome: everything the kernel consumes besides the caches can be built on the host and checked on CPU: `w_rot`, the `[cos | sin]` table, the Phase-2 chunk-step constants and transformation tile, and an fp32 torch model of the op that Tasks 3-5 test against.

**Files:**
- Create: `palu_tt/fused.py`, `tests/test_fused.py`
- Modify: `palu_tt/convert.py` (`CONVERT_VERSION`, `PaluRuntime`, `PaluConfig`, `palu_layer_tensors`), `tests/test_convert.py`

**Interfaces:**
- Produces (`palu_tt/fused.py`):
  - `rope_transformation_matrix(head_dim: int) -> torch.Tensor  # [hd, hd]`
  - `trans_mat_tile() -> torch.Tensor  # [1, 1, 32, 32], the tile the kernel multiplies with`
  - `w_rot_from_u_k(u_k: torch.Tensor) -> torch.Tensor  # [n_kv, r_k, hd] -> [n_kv, r_k, 2*hd]`
  - `chunk_size(seq_len: int) -> int  # the K chunk the op picks for an unpaged cache of this length`
  - `cos_sin_tables(head_dim, seq_len, theta, rope_scaling) -> (cos, sin)  # each [seq_len + 513, hd] fp32, Meta-interleaved`
  - `cs_table(cos, sin, seq_len) -> torch.Tensor  # [seq_len, 2*hd]`
  - `rot_step_table(cos, sin, chunk: int) -> torch.Tensor  # [32, 2*hd] fp32: every row = [cos(chunk*f) | sin(chunk*f)]`
  - `reference_decode(q, k_lat, u_k, v_lat, cur_pos, cos, sin, scale) -> torch.Tensor  # [1, B, nh, r_v] fp32`
- Produces (`palu_tt/convert.py`): `CONVERT_VERSION == 3`; `PaluRuntime(recon_len, check, use_fused_k=False, cs_recurrence=False)`; `PaluConfig.device_tables: dict` (per-model device-tensor cache, excluded from equality/repr); state-dict key `layers.{i}.attention.palu.w_rot` of shape `[n_kv, r_k, 2*head_dim]`.

- [ ] **Step 1: Write the failing CPU tests**

Create `tests/test_fused.py`:

```python
import torch

from palu_tt import fused

HD = 128


def test_transformation_matrix_is_the_rope_pair_rotation():
    # rotary_embedding_llama computes rotated = x @ T; for interleaved pairs that is (x0, x1) -> (-x1, x0)
    T = fused.rope_transformation_matrix(HD)
    x = torch.randn(3, HD)
    r = x @ T
    torch.testing.assert_close(r[:, 0::2], -x[:, 1::2])
    torch.testing.assert_close(r[:, 1::2], x[:, 0::2])
    t = fused.trans_mat_tile()
    assert t.shape == (1, 1, 32, 32) and torch.equal(t[0, 0], T[:32, :32])


def test_w_rot_is_u_k_and_u_k_times_T():
    u_k = torch.randn(4, 64, HD)
    w = fused.w_rot_from_u_k(u_k)
    assert w.shape == (4, 64, 2 * HD)
    assert torch.equal(w[..., :HD], u_k)
    lat = torch.randn(4, 7, 64)
    torch.testing.assert_close(lat @ w[..., HD:], (lat @ u_k) @ fused.rope_transformation_matrix(HD))


def test_chunk_size_is_the_ops_static_choice():
    # ttnn sdpa_decode (unpaged): largest power of two dividing S, capped at 512
    assert [fused.chunk_size(s) for s in (96, 256, 1024, 4096, 8192, 8192 + 32)] == [32, 256, 512, 512, 512, 32]


def test_tables_are_meta_interleaved_and_long_enough():
    cos, sin = fused.cos_sin_tables(HD, 1024, 500000.0, None)
    assert cos.shape == (1024 + 513, HD) and cos.dtype == torch.float32
    assert torch.equal(cos[:, 0::2], cos[:, 1::2]) and torch.equal(sin[:, 0::2], sin[:, 1::2])
    torch.testing.assert_close(cos[0], torch.ones(HD))
    tab = fused.cs_table(cos, sin, 1024)
    assert tab.shape == (1024, 2 * HD)
    assert torch.equal(tab[:, :HD], cos[:1024]) and torch.equal(tab[:, HD:], sin[:1024])


def test_rot_step_is_the_table_row_at_the_chunk_length_on_every_row():
    cos, sin = fused.cos_sin_tables(HD, 1024, 500000.0, None)
    for c in (32, 128, 512):
        step = fused.rot_step_table(cos, sin, c)
        assert step.shape == (32, 2 * HD) and step.dtype == torch.float32
        assert torch.equal(step[0, :HD], cos[c]) and torch.equal(step[31, HD:], sin[c])
        assert torch.equal(step[0], step[17])


def test_angle_addition_advances_the_table_by_a_chunk():
    cos, sin = fused.cos_sin_tables(HD, 1024, 500000.0, None)
    for c in (32, 128, 512):
        p = torch.arange(0, 700)
        cos2 = cos[p] * cos[c] - sin[p] * sin[c]
        sin2 = sin[p] * cos[c] + cos[p] * sin[c]
        torch.testing.assert_close(cos2, cos[p + c], atol=2e-5, rtol=0)
        torch.testing.assert_close(sin2, sin[p + c], atol=2e-5, rtol=0)


def test_relative_rotation_identity():
    """q R_p (k R_{s0+t})^T == (q R_{p-s0}) (k R_t)^T -- the identity Phase 2 rests on (spec §5)."""
    cos, sin = fused.cos_sin_tables(HD, 2048, 500000.0, None)
    T = fused.rope_transformation_matrix(HD)
    rot = lambda x, s: x * cos[s] + (x @ T) * sin[s]          # R_s applied to a row vector
    q, k = torch.randn(HD), torch.randn(HD)
    p, s0, t = 1500, 1024, 37
    lhs = rot(q, p) @ rot(k, s0 + t)
    q_rel = rot(q, p) * cos[s0] - (rot(q, p) @ T) * sin[s0]    # R_{-s0}: cos even, sin odd
    rhs = q_rel @ rot(k, t)
    torch.testing.assert_close(lhs, rhs, atol=1e-3, rtol=1e-5)


def test_reference_decode_reduces_to_causal_sdpa_without_rotation():
    B, nkv, nh, S, r_v = 2, 2, 4, 64, 32
    q = torch.randn(1, B, nh, HD)
    k = torch.randn(B, nkv, S, HD)
    v = torch.randn(B, nkv, S, r_v)
    cur = torch.tensor([10, 63])
    cos, sin = torch.ones(S, HD), torch.zeros(S, HD)
    out = fused.reference_decode(q, k, torch.eye(HD).expand(nkv, HD, HD), v, cur, cos, sin, HD ** -0.5)
    assert out.shape == (1, B, nh, r_v)
    for b in range(B):
        for h in range(nh):
            kk, vv = k[b, h // 2, : cur[b] + 1], v[b, h // 2, : cur[b] + 1]
            att = torch.softmax(q[0, b, h] @ kk.T * HD ** -0.5, -1) @ vv
            torch.testing.assert_close(out[0, b, h], att, atol=1e-5, rtol=1e-4)


def test_reference_decode_rotates_k_like_rotary_embedding_llama():
    B, nkv, nh, S, r_k = 1, 1, 1, 8, 32
    q = torch.randn(1, B, nh, HD)
    lat = torch.randn(B, nkv, S, r_k)
    u = torch.randn(nkv, r_k, HD)
    v = torch.randn(B, nkv, S, 16)
    cos, sin = fused.cos_sin_tables(HD, S, 500000.0, None)
    out = fused.reference_decode(q, lat, u, v, torch.tensor([S - 1]), cos, sin, 1.0)
    K = lat[0, 0] @ u[0]
    K = K * cos[:S] + (K @ fused.rope_transformation_matrix(HD)) * sin[:S]
    exp = torch.softmax(q[0, 0, 0] @ K.T, -1) @ v[0, 0]
    torch.testing.assert_close(out[0, 0, 0], exp, atol=1e-5, rtol=1e-4)
```

Append to `tests/test_convert.py` (its header defines `DIM, NH, NKV, HD = 64, 4, 2, 16` and `_hf_attn()`; head dim 16 is fine, the transformation matrix is defined for any even size):

```python
def test_layer_tensors_include_w_rot_built_from_u_k():
    from palu_tt import fused
    w = _hf_attn()
    lr = convert.identity_lowrank(w, NKV, HD)
    t = convert.palu_layer_tensors(w, lr, NH, NKV, HD)
    assert t["w_rot"].shape == (NKV, HD, 2 * HD)
    torch.testing.assert_close(t["w_rot"], fused.w_rot_from_u_k(t["u_k"]))


def test_convert_version_is_3_and_runtime_flags_default_off():
    assert convert.CONVERT_VERSION == 3
    rt = convert.PaluRuntime()
    assert rt.use_fused_k is False and rt.cs_recurrence is False
    cfg = convert.PaluConfig(rank_k=[32], rank_v=[64], cache_id="x")
    assert cfg.device_tables == {}
```

- [ ] **Step 2: Run them, expect failure**

```bash
nice -n 10 pytest tests/test_fused.py tests/test_convert.py -q 2>&1 | tail -5
```
Expected: `ModuleNotFoundError: No module named 'palu_tt.fused'` and `KeyError: 'w_rot'`.

- [ ] **Step 3: Write `palu_tt/fused.py`**

```python
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
```

- [ ] **Step 4: Converter changes**

In `palu_tt/convert.py`:

```python
# (replace the CONVERT_VERSION block)
# Bumped whenever palu_layer_tensors changes the *content* of a tensor without changing its name,
# shape, dtype or the weight-set id -- the on-disk ttnn weight cache keys on those alone, so an old
# entry would otherwise be served for the new tensor. attention.cache_tag() puts it in every
# Palu-specific cache file name. v2: full-rank V is collapsed into the QKV weight (see below).
# v3: adds w_rot = [u_k | u_k T] for the fused decode kernel (stage 2).
CONVERT_VERSION = 3


@dataclass
class PaluRuntime:
    recon_len: Optional[int] = None
    check: bool = False
    # stage 2: decode through ttnn.transformer.palu_sdpa_decode (K reconstructed and rotated on
    # core) instead of matmul + prefill-mode RoPE + the MLA op. Only applies to layers with
    # r_k < head_dim; the V-only rows keep the stock K path either way.
    use_fused_k: bool = False
    # stage 2 Phase 2: chunk-relative rotation with the query advanced on core per chunk, so the
    # cos/sin table is read once per core instead of once per chunk. Needs use_fused_k.
    cs_recurrence: bool = False
```

Add to `PaluConfig` (after `latent_dtype`):

```python
    # Device tensors shared by every layer of one model on the fused path (the [cos | sin] table,
    # the Phase-2 step constants and transformation tile), keyed by attention.py; never compared
    # or printed.
    device_tables: dict = field(default_factory=dict, repr=False, compare=False)
```

Add `from palu_tt import fused` to the imports and change the `return` of `palu_layer_tensors` to:

```python
    return {"wk_lat_pad": wk_lat_pad, "wv_lat_pad": wv_lat_pad, "u_k": u_k, "u_v": u_v, "wo_fused": wo_fused,
            "w_rot": fused.w_rot_from_u_k(u_k)}
```

(`u_k` here is already the Meta-permuted, transposed factor the decode path multiplies the latent with, so `w_rot` needs no further permutation.)

- [ ] **Step 5: Run the CPU suite**

```bash
nice -n 10 pytest tests/test_fused.py tests/test_convert.py tests/test_attention_tag.py tests/test_reference.py tests/test_report.py tests/test_bench_utils.py -q 2>&1 | tail -3
```
Expected: all pass (the tag test picks up `palu_v3_` automatically).

- [ ] **Step 6: Commit**

```bash
git add palu_tt/fused.py palu_tt/convert.py tests/test_fused.py tests/test_convert.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2 Task 2: fused-kernel host builders (w_rot, cos/sin tables, step constants, torch reference); converter v3

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 3: Phase 1 kernel — on-core K reconstruction with streamed cos/sin

Outcome: `palu_sdpa_decode` takes the latent cache, `w_rot` and `cs_tab`, builds every K chunk on core, and matches the fp32 reference (PCC ≥ 0.99) and the stock MLA op fed the host-reconstructed K. Phase 1 streams the absolute `[cos | sin]` rows per chunk; it is the correctness checkpoint for the K-side math and stays selectable (no `rot_step`) as the reference path.

**Files:**
- Modify (submodule, all under `ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/`): `palu_sdpa_decode.hpp`, `palu_sdpa_decode.cpp`, `palu_sdpa_decode_nanobind.cpp`, `device/palu_sdpa_decode_device_operation_types.hpp`, `device/palu_sdpa_decode_device_operation.{hpp,cpp}`, `device/palu_sdpa_decode_program_factory.cpp`, `device/kernels/dataflow/palu_reader_decode.cpp`, `device/kernels/compute/palu_sdpa_flash_decode.cpp`.
- Create (submodule): `device/kernels/dataflow/palu_dataflow_common.hpp`, `device/kernels/compute/palu_compute_common.hpp`.
- Rewrite: `tests/test_palu_sdpa_device.py` (the Task 1 clone test is superseded: the signature changes).

**Interfaces:**
- Consumes: `palu_tt.fused` (Task 2): `w_rot_from_u_k`, `cos_sin_tables`, `cs_table`, `reference_decode`, `rope_transformation_matrix`.
- Produces: Python `ttnn.transformer.palu_sdpa_decode(q, k_latent, w_rot, cs_tab, v_latent, head_dim_v, cur_pos_tensor, *, rot_step=None, trans_mat=None, scale=None, memory_config=None, program_config=None, compute_kernel_config=None) -> Tensor[1, B, nh, head_dim_v]`. `rot_step`/`trans_mat` are accepted and validated here but Phase 2 (their use) is Task 4; passing them in Task 3 raises "Phase 2 not implemented" from validation.
- Produces (C++): `ttnn::prim::palu_sdpa_decode(q, k_latent, w_rot, cs_tab, v_latent, cur_pos_tensor, rot_step, trans_mat, head_dim_v, scale, output_mem_config, program_config, compute_kernel_config, k_chunk_size)`; `PaluSdpaDecodeInputs{..., w_rot, cs_tab, rot_step, trans_mat}`; `PaluSdpaDecodeParams{..., cs_recurrence}`.
- Produces (kernels): `palu_build_k_chunk<cb_lat, cb_w, cb_ab, cb_cs, cb_k, Rt, DHt, dst_tiles>(Sk_chunk_t, cs_tile_base)`, `read_w_rot<cb_w, Rt, Wt, barrier>(head, reader)`, `read_palu_chunks<Rt, vDHt, Wt, barrier, cb_lat, cb_v, cb_cs, phase2>(...)`, `palu_read_tiles<cb, barrier>(first_tile, n, reader)`; reader compile-time args 37 = `Rt`, 38 = `phase2`, accessor args from 39 in the order q, k, v, mask, pos, page_table, sink, w_rot, cs_tab, rot_step, trans_mat; reader runtime args gain `w_rot_addr, cs_addr, rot_step_addr, trans_mat_addr` right after `attention_sink_addr`; compute compile-time arg 32 = `Rt`; compute defines `PALU_DST_TILES` (4 or 8) and `PALU_PHASE2` (Task 4).

- [ ] **Step 1: Write the failing test**

Replace the whole of `tests/test_palu_sdpa_device.py` with:

```python
"""Op-level tests for ttnn.transformer.palu_sdpa_decode (stage 2 fused K-reconstruction decode).

Runs on the N300 mesh fixture; every tensor is replicated to both chips and chip 0 is read back.
References: (a) palu_tt.fused.reference_decode in fp32 on bf16-rounded inputs, (b) the stock MLA
flash-decode op fed the host-reconstructed, host-rotated K -- (b) isolates kernel error from the
bf16 rounding both paths share."""
import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

from palu_tt import fused

HD, NH, NKV = 128, 16, 4
THETA = 500000.0


def _cfg(k_chunk):
    return dict(
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=k_chunk,
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


def _bf(t):
    return t.to(torch.bfloat16).float()


def _inputs(batch, S, r_k, r_v):
    q = _bf(torch.randn(1, batch, NH, HD))
    k_lat = _bf(torch.randn(batch, NKV, S, r_k))
    u_k = _bf(torch.randn(NKV, r_k, HD) / r_k ** 0.5)
    v = _bf(torch.randn(batch, NKV, S, r_v))
    pos = torch.randint(0, S, (batch,))
    pos[0] = S - 1                                   # full cache
    if batch >= 4:
        pos[1], pos[2], pos[3] = 0, 127, 128         # first row only; chunk boundary both sides
    cos, sin = fused.cos_sin_tables(HD, S, THETA, None)
    return q, k_lat, u_k, v, pos, _bf(cos), _bf(sin)


def _stock_on_reconstructed_k(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk):
    S = k_lat.shape[2]
    K = torch.einsum("bhsr,hrd->bhsd", k_lat, u_k)
    K = _bf(K * cos[None, None, :S] + (K @ fused.rope_transformation_matrix(HD)) * sin[None, None, :S])
    out = ttnn.transformer.flash_multi_latent_attention_decode(
        _dev(md, q), _dev(md, K), _dev(md, v), head_dim_v=r_v, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5,
        **_cfg(k_chunk))
    return _host(out)


def _fused(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk, phase2):
    S = k_lat.shape[2]
    extra = {}
    if phase2:
        c = k_chunk or fused.chunk_size(S)
        extra = dict(rot_step=_dev(md, fused.rot_step_table(cos, sin, c)[None, None], ttnn.float32),
                     trans_mat=_dev(md, fused.trans_mat_tile()))
    out = ttnn.transformer.palu_sdpa_decode(
        _dev(md, q), _dev(md, k_lat), _dev(md, fused.w_rot_from_u_k(u_k)[None]), _dev(md, fused.cs_table(cos, sin, S)[None, None]),
        _dev(md, v), head_dim_v=r_v, cur_pos_tensor=_pos(md, pos), scale=HD ** -0.5, **extra, **_cfg(k_chunk))
    return _host(out)


def _check(md, batch, S, r_k, r_v, k_chunk, phase2, pcc_ref=0.99, pcc_stock=0.995):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(batch, S, r_k, r_v)
    ref = fused.reference_decode(q, k_lat, u_k, v, pos, cos, sin, HD ** -0.5)
    out = _fused(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk, phase2)
    assert out.shape == (1, batch, NH, r_v)
    ok, msg = comp_pcc(ref, out, pcc_ref)
    assert ok, f"vs fp32 reference: {msg}"
    stock = _stock_on_reconstructed_k(md, q, k_lat, u_k, v, pos, cos, sin, r_v, k_chunk)
    ok, msg = comp_pcc(stock, out, pcc_stock)
    assert ok, f"vs stock MLA on host-rotated K: {msg}"


@pytest.mark.parametrize("r_k", [32, 64, 96])
@pytest.mark.parametrize("batch", [1, 32])
def test_phase1_matches_reference(mesh_device, r_k, batch):
    # Phase 1 streams a [c, 256] cos/sin block per chunk; at the op's default chunk (512) that CB is
    # 512 KB and does not fit next to the stock CBs, so Phase 1 always runs with k_chunk_size=128.
    _check(mesh_device, batch, 1024, r_k, 64, k_chunk=128, phase2=False)


def test_phase1_r_v_full_width(mesh_device):
    _check(mesh_device, 32, 1024, 32, 128, k_chunk=128, phase2=False)


def test_envelope_is_enforced(mesh_device):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(1, 256, 64, 64)
    good = dict(head_dim_v=64, cur_pos_tensor=_pos(mesh_device, pos), scale=HD ** -0.5, **_cfg(128))
    w = _dev(mesh_device, fused.w_rot_from_u_k(u_k)[None])
    cs = _dev(mesh_device, fused.cs_table(cos, sin, 256)[None, None])
    with pytest.raises(RuntimeError, match="r_k"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, torch.randn(1, NKV, 256, 48)), w, cs,
                                          _dev(mesh_device, v), **good)
    with pytest.raises(RuntimeError, match="cs_tab"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat), w,
                                          _dev(mesh_device, fused.cs_table(cos, sin, 128)[None, None]), _dev(mesh_device, v), **good)
    with pytest.raises(RuntimeError, match="w_rot"):
        ttnn.transformer.palu_sdpa_decode(_dev(mesh_device, q), _dev(mesh_device, k_lat),
                                          _dev(mesh_device, u_k[None]), cs, _dev(mesh_device, v), **good)
```

(In the first `raises`, the latent of width 48 is what trips the `r_k` check. Task 4 adds the Phase-2 tests to this file.)

- [ ] **Step 2: Run to see it fail**

```bash
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q -k "phase1_matches and 64-1" 2>&1 | tail -3
```
Expected: `TypeError` (unexpected keyword / wrong positional types) from the Task 1 signature.

- [ ] **Step 3: Types and prim wrapper**

`device/palu_sdpa_decode_device_operation_types.hpp` — append to `PaluSdpaDecodeParams` (after `cache_position_modulo`):

```cpp
    // Palu: chunk-relative rotation with the query advanced on core (Phase 2). Set iff rot_step is given.
    bool cs_recurrence = false;
```

and to `PaluSdpaDecodeInputs` (after `attention_sink`):

```cpp
    // Palu fused-K inputs (spec §3)
    Tensor w_rot;                      // [1, n_kv, r_k, 2*hd] bf16: [u_k | u_k T]
    Tensor cs_tab;                     // [1, 1, S_cs >= S, 2*hd] bf16: [cos | sin] per position
    std::optional<Tensor> rot_step;    // [1, 1, 32, 2*hd] fp32: [cos(c f) | sin(c f)] (Phase 2)
    std::optional<Tensor> trans_mat;   // [1, 1, 32, 32] bf16 pair-rotation tile (Phase 2)
```

`device/palu_sdpa_decode_device_operation.hpp` — replace the `Tensor palu_sdpa_decode(...)` declaration with:

```cpp
Tensor palu_sdpa_decode(
    const Tensor& input_tensor_q,
    const Tensor& k_latent,
    const Tensor& w_rot,
    const Tensor& cs_tab,
    const Tensor& v_latent,
    const Tensor& cur_pos_tensor,
    const std::optional<const Tensor>& rot_step,
    const std::optional<const Tensor>& trans_mat,
    uint32_t head_dim_v,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t k_chunk_size);
```

`device/palu_sdpa_decode_device_operation.cpp` — replace the definition at the bottom with:

```cpp
Tensor palu_sdpa_decode(
    const Tensor& input_tensor_q,
    const Tensor& k_latent,
    const Tensor& w_rot,
    const Tensor& cs_tab,
    const Tensor& v_latent,
    const Tensor& cur_pos_tensor,
    const std::optional<const Tensor>& rot_step,
    const std::optional<const Tensor>& trans_mat,
    uint32_t head_dim_v,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t k_chunk_size) {
    using OperationType = PaluSdpaDecodeDeviceOperation;
    // Stock attributes pinned to the envelope: causal, unpaged, MLA-style separate V, cur_pos from tensor.
    auto operation_attributes = OperationType::operation_attributes_t{
        .is_causal = true,
        .paged_attention = false,
        .cur_pos = {},
        .scale = scale,
        .sliding_window_size = std::nullopt,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .compute_kernel_config = compute_kernel_config,
        .k_chunk_size = k_chunk_size,
        .share_cache = std::nullopt,
        .use_mla = true,
        .head_dim_v = head_dim_v,
        .paged_cache_geometry = ttnn::operations::transformer::PagedCacheGeometryOverride{},
        .cache_position_modulo = std::nullopt,
        .cs_recurrence = rot_step.has_value(),
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = input_tensor_q,
        .k = k_latent,
        .v = v_latent,
        .cur_pos_tensor = cur_pos_tensor,
        .page_table_tensor = std::nullopt,
        .attn_mask = std::nullopt,
        .attention_sink = std::nullopt,
        .w_rot = w_rot,
        .cs_tab = cs_tab,
        .rot_step = rot_step,
        .trans_mat = trans_mat,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
```

- [ ] **Step 4: Validation**

In `validate_on_program_cache_miss` (same file), the unpaged branch checks `k_shape[-1] == D` ("K tensor hidden dimension must equal Q tensor hidden dimension"). Replace that one `TT_FATAL` with:

```cpp
        // Palu: K in DRAM is the latent [B, nkv, S, r_k]; the head dim comes from Q.
        TT_FATAL(D == 128, "palu_sdpa_decode: head_dim must be 128, got {}", D);
        TT_FATAL(
            k_shape[-1] == 32 || k_shape[-1] == 64 || k_shape[-1] == 96,
            "palu_sdpa_decode: r_k (latent width) must be 32, 64 or 96, got {}",
            k_shape[-1]);
```

and append this block at the very end of the function (after the attention-sink checks):

```cpp
    // ---------------- Palu envelope (stage 2 spec §3) ----------------
    TT_FATAL(operation_attributes.is_causal && !operation_attributes.paged_attention, "palu_sdpa_decode: causal, unpaged only");
    TT_FATAL(tensor_args.cur_pos_tensor.has_value(), "palu_sdpa_decode: cur_pos_tensor is required");
    TT_FATAL(
        !tensor_args.attn_mask.has_value() && !tensor_args.attention_sink.has_value() &&
            !tensor_args.page_table_tensor.has_value() && !operation_attributes.sliding_window_size.has_value() &&
            !operation_attributes.share_cache.value_or(false),
        "palu_sdpa_decode: mask, sink, page table, sliding window and share_cache are outside the envelope");
    TT_FATAL(tensor_args.q.layout() == Layout::TILE, "palu_sdpa_decode: Q must be TILE layout (c_10 is reused)");
    TT_FATAL(q_shape[1] <= 32, "palu_sdpa_decode: batch must be <= 32, got {}", q_shape[1]);
    const uint32_t r_k = k_shape[-1];
    const uint32_t n_kv = k_shape[1];
    const uint32_t S = k_shape[2];
    auto expect = [](const Tensor& t, const char* name, std::initializer_list<uint32_t> shape, DataType dt) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr, "palu_sdpa_decode: {} must be on device", name);
        TT_FATAL(t.layout() == Layout::TILE, "palu_sdpa_decode: {} must be TILE layout", name);
        TT_FATAL(t.dtype() == dt, "palu_sdpa_decode: {} has the wrong dtype", name);
        TT_FATAL(t.padded_shape().rank() == 4, "palu_sdpa_decode: {} must be rank 4", name);
        uint32_t i = 0;
        for (uint32_t want : shape) {
            TT_FATAL(want == 0 || t.padded_shape()[i] == want, "palu_sdpa_decode: {} dim {} is {}, expected {}", name, i, t.padded_shape()[i], want);
            i++;
        }
    };
    expect(tensor_args.w_rot, "w_rot", {1, n_kv, r_k, 2 * D}, DataType::BFLOAT16);
    expect(tensor_args.cs_tab, "cs_tab", {1, 1, 0, 2 * D}, DataType::BFLOAT16);
    TT_FATAL(tensor_args.cs_tab.padded_shape()[2] >= S, "palu_sdpa_decode: cs_tab has {} rows, cache has {}", tensor_args.cs_tab.padded_shape()[2], S);
    if (operation_attributes.cs_recurrence) {
        TT_FATAL(tensor_args.trans_mat.has_value(), "palu_sdpa_decode: rot_step needs trans_mat");
        expect(*tensor_args.rot_step, "rot_step", {1, 1, 32, 2 * D}, DataType::FLOAT32);
        expect(*tensor_args.trans_mat, "trans_mat", {1, 1, 32, 32}, DataType::BFLOAT16);
        auto [fid, approx, fp32_dest, l1_acc, full_sync] =
            get_compute_kernel_config_args(tensor_args.q.device()->arch(), operation_attributes.compute_kernel_config);
        TT_FATAL(fp32_dest, "palu_sdpa_decode: Phase 2 (rot_step) requires fp32_dest_acc_en");
        TT_FATAL(false, "palu_sdpa_decode: Phase 2 not implemented yet (Task 4)");
    }
```

(`D` and `q_shape`, `k_shape` are the locals the stock function already has in scope at that point; if the stock code declared `D` inside the unpaged `else` block, hoist `const auto D = q_shape[-1];` above the `if (paged)` so both places see it. `get_compute_kernel_config_args` comes from `ttnn/operations/core/compute_kernel/compute_kernel_config.hpp`, already included via the types header.)

- [ ] **Step 5: Public op and binding**

`palu_sdpa_decode.hpp`:

```cpp
namespace ttnn::transformer {

// Palu fused-K flash decode (stage 2). K is never materialised: each K chunk is rebuilt on core
// from the latent cache as RoPE(lat @ u_k). See docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md.
ttnn::Tensor palu_sdpa_decode(
    const ttnn::Tensor& input_tensor_q,          // [1, B, nh, 128] bf16 TILE, already rotated
    const ttnn::Tensor& k_latent,                // [B, nkv, S, r_k]   (r_k in {32, 64, 96})
    const ttnn::Tensor& w_rot,                   // [1, nkv, r_k, 256] bf16: [u_k | u_k T]
    const ttnn::Tensor& cs_tab,                  // [1, 1, >=S, 256] bf16: [cos | sin]
    const ttnn::Tensor& v_latent,                // [B, nkv, S, r_v]
    uint32_t head_dim_v,
    const ttnn::Tensor& cur_pos_tensor,          // [B] int32
    const std::optional<const Tensor>& rot_step = std::nullopt,   // [1, 1, 32, 256] fp32 -> Phase 2
    const std::optional<const Tensor>& trans_mat = std::nullopt,  // [1, 1, 32, 32] bf16 (Phase 2)
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
```

`palu_sdpa_decode.cpp` — the body keeps the MLA chunk logic verbatim, with `s` taken from `k_latent` and the V check against `v_latent`, and ends with:

```cpp
    return ttnn::prim::palu_sdpa_decode(
        input_tensor_q, k_latent, w_rot, cs_tab, v_latent, cur_pos_tensor, rot_step, trans_mat, head_dim_v, scale,
        memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG), program_config, kernel_config_val, k_chunk_size);
```

`palu_sdpa_decode_nanobind.cpp` — the binding becomes:

```cpp
    ttnn::bind_function<"palu_sdpa_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::palu_sdpa_decode,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("k_latent").noconvert(),
        nb::arg("w_rot").noconvert(),
        nb::arg("cs_tab").noconvert(),
        nb::arg("v_latent").noconvert(),
        nb::arg("head_dim_v").noconvert(),
        nb::arg("cur_pos_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("rot_step") = nb::none(),
        nb::arg("trans_mat") = nb::none(),
        nb::arg("scale") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
```

with a docstring that lists the tensor table from spec §3.

- [ ] **Step 6: Program factory**

Edits to `device/palu_sdpa_decode_program_factory.cpp`, in file order:

(a) After `const auto& attention_sink = tensor_args.attention_sink;`:

```cpp
    // ---------------- Palu inputs (stage 2) ----------------
    const auto& w_rot = tensor_args.w_rot;
    const auto& cs_tab = tensor_args.cs_tab;
    const bool phase2 = operation_attributes.cs_recurrence;  // rot_step + trans_mat present (Task 4)
    Buffer* rot_step_buffer = phase2 ? tensor_args.rot_step->buffer() : nullptr;
    Buffer* trans_mat_buffer = phase2 ? tensor_args.trans_mat->buffer() : nullptr;
```

(b) Replace `uint32_t DH = apply_geometry_override ? q_shape[3] : k_shape[3];` with

```cpp
    uint32_t DH = q_shape[3];  // Palu: k_shape[3] is the latent width r_k, the head dim comes from Q
```

and after `const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;` add

```cpp
    const uint32_t r_k = k_shape[3];
    const uint32_t Rt = r_k / TILE_WIDTH;   // latent width in tiles
    const uint32_t Wt = 2 * DHt;            // [u_k | u_k T] / [cos | sin] width in tiles
```

(c) Data formats: after `const tt::DataFormat v_df = ...;` add

```cpp
    const tt::DataFormat lat_df = k_df;                        // the latent cache dtype (bfp8 by default)
    const tt::DataFormat krot_df = tt::DataFormat::Float16_b;  // K_rot is built on core in bf16
    const tt::DataFormat w_df = tt_metal::datatype_to_dataformat_converter(w_rot.dtype());
    const tt::DataFormat cs_df = tt_metal::datatype_to_dataformat_converter(cs_tab.dtype());
```

Replace `const bool use_half_tile = is_causal && num_q_heads <= 16 && q_df == tt::DataFormat::Float16_b;` with

```cpp
    // Palu: Phase 2 keeps the rotated query in a full-tile fp32 CB that feeds the QK matmul, and every
    // Q-side intermediate must share that geometry, so the 16-row Q tile optimisation is off here.
    const bool use_half_tile = false;
```

Replace `const uint32_t k_tile_size = k_tile.get_tile_size(k_df);` with

```cpp
    const uint32_t k_tile_size = k_tile.get_tile_size(krot_df);
    const uint32_t lat_tile_size = full_tile.get_tile_size(lat_df);
    const uint32_t w_tile_size = full_tile.get_tile_size(w_df);
    const uint32_t cs_tile_size = full_tile.get_tile_size(cs_df);
    const uint32_t bf16_tile_size = full_tile.get_tile_size(tt::DataFormat::Float16_b);
```

(d) CBs: change the `c_1` line to `add_cb(CBIndex::c_1, k_tiles * k_tile_size, krot_df, k_tile_size);  // K_rot, built on core`. Replace the unconditional `add_cb(CBIndex::c_10, q_tiles * q_tile_size, q_df, q_tile_size, &q_tile);  // tilized Q` line with:

```cpp
    // ---------------- Palu CBs (spec §4) ----------------
    // The stock owners of these slots (row-major Q c_10, sink c_4, page table c_9, sliding window c_13,
    // block padding c_14) are outside the envelope and are never allocated here.
    const uint32_t lat_tiles = Sk_chunk_t_cb_size * Rt * 2;                                  // latent chunk, double buffered
    const uint32_t w_rot_tiles = Rt * Wt + 1;                                                 // [u_k | u_k T] + the T tile (Task 4)
    const uint32_t ab_tiles = 2 * Wt;                                                         // one AB row block, ping-pong
    const uint32_t cs_tiles = phase2 ? Sk_chunk_t_cb_size * Wt : 2 * Sk_chunk_t_cb_size * Wt;  // resident table / streamed x2
    add_cb(CBIndex::c_9, lat_tiles * lat_tile_size, lat_df, lat_tile_size, &full_tile);
    add_cb(CBIndex::c_4, w_rot_tiles * w_tile_size, w_df, w_tile_size, &full_tile);
    add_cb(CBIndex::c_10, ab_tiles * bf16_tile_size, tt::DataFormat::Float16_b, bf16_tile_size, &full_tile);
    add_cb(CBIndex::c_13, cs_tiles * cs_tile_size, cs_df, cs_tile_size, &full_tile);
```

(e) Compile-time args. In `reader_compile_time_args_common` append after `capacity_t,`:

```cpp
        Rt,
        static_cast<uint32_t>(phase2),
```

and after the attention-sink `TensorAccessorArgs(...).append_to(reader_compile_time_args_common);` lines add:

```cpp
    tt_metal::TensorAccessorArgs(w_rot.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(cs_tab.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(rot_step_buffer).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(trans_mat_buffer).append_to(reader_compile_time_args_common);
```

In `compute_compile_time_args_common` append `Rt,` after `original_block_size,`. After `compute_defines["LOG2_DHT_GRANULARITY"] = ...;` add:

```cpp
    compute_defines["PALU_DST_TILES"] = std::to_string(dst_size);
    if (phase2) {
        compute_defines["PALU_PHASE2"] = "1";
    }
```

(f) Runtime args: after `reader_rt_args.push_back(attention_sink_buffer);` add

```cpp
        reader_rt_args.push_back(w_rot.buffer());
        reader_rt_args.push_back(cs_tab.buffer());
        reader_rt_args.push_back(rot_step_buffer);
        reader_rt_args.push_back(trans_mat_buffer);
```

and in the idle-core block change `KernelDescriptor::CoreRuntimeArgs reader_rt_args(20, 0);` to `(24, 0)` (comment: "+4 Palu buffers").

- [ ] **Step 7: Reader kernel and its helper header**

Create `device/kernels/dataflow/palu_dataflow_common.hpp`:

```cpp
// SPDX-License-Identifier: Apache-2.0
// Dataflow helpers for palu_sdpa_decode (Palu fused-K flash decode).
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

// n consecutive DRAM tiles [first_tile, first_tile + n) -> one push of n tiles into cb_id.
template <uint32_t cb_id, uint32_t barrier_threshold, typename ReaderT>
void palu_read_tiles(uint32_t first_tile, uint32_t n, const ReaderT& reader) {
    Noc noc;
    CircularBuffer cb(cb_id);
    constexpr uint32_t tile_bytes = get_tile_size(cb_id);
    cb.reserve_back(n);
    uint32_t wp = cb.get_write_ptr();
    uint32_t barrier_count = 0;
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(reader, CoreLocalMem<uint32_t>(wp), tile_bytes, {.page_id = first_tile + i}, {});
        if (++barrier_count == barrier_threshold) {
            noc.async_read_barrier();
            barrier_count = 0;
        }
        wp += tile_bytes;
    }
    noc.async_read_barrier();
    cb.push_back(n);
}

// w_rot for one KV head: Rt x Wt tiles, row-major in DRAM ([1, n_kv, r_k, 2*hd] TILE), pushed once per head.
template <uint32_t cb_w, uint32_t Rt, uint32_t Wt, uint32_t barrier_threshold, typename ReaderT>
void read_w_rot(uint32_t head, const ReaderT& reader) {
    palu_read_tiles<cb_w, barrier_threshold>(head * Rt * Wt, Rt * Wt, reader);
}

// Per K chunk, in this order: the latent rows [Sk_chunk_t x Rt] (contiguous in DRAM), the absolute
// [cos | sin] rows for the same positions [Sk_chunk_t x Wt] (Phase 1 only; Phase 2 uses a resident
// chunk-relative table read once by the caller), then V [Sk_chunk_t x vDHt] exactly as stock.
template <
    uint32_t Rt,
    uint32_t vDHt,
    uint32_t Wt,
    uint32_t barrier_threshold,
    uint32_t cb_lat,
    uint32_t cb_v,
    uint32_t cb_cs,
    bool phase2,
    typename LatReaderT,
    typename VReaderT,
    typename CsReaderT>
void read_palu_chunks(
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t lat_start_tile_id,
    uint32_t v_start_tile_id,
    uint32_t Sk_chunk_t,
    const LatReaderT& lat_reader,
    const VReaderT& v_reader,
    const CsReaderT& cs_reader) {
    const uint32_t lat_chunk_tiles = Sk_chunk_t * Rt;
    const uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    const uint32_t cs_chunk_tiles = Sk_chunk_t * Wt;
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        palu_read_tiles<cb_lat, barrier_threshold>(lat_start_tile_id, lat_chunk_tiles, lat_reader);
        lat_start_tile_id += lat_chunk_tiles;
        if constexpr (!phase2) {
            palu_read_tiles<cb_cs, barrier_threshold>(k_chunk * cs_chunk_tiles, cs_chunk_tiles, cs_reader);
        }
        palu_read_tiles<cb_v, barrier_threshold>(v_start_tile_id, v_chunk_tiles, v_reader);
        v_start_tile_id += v_chunk_tiles;
    }
}
```

Edits to `device/kernels/dataflow/palu_reader_decode.cpp`:

(a) Add `#include "palu_dataflow_common.hpp"` after the dataflow_common include.

(b) After `constexpr uint32_t capacity_t = get_compile_time_arg_val(36);` add

```cpp
    // ---------------- Palu (stage 2) ----------------
    constexpr uint32_t Rt = get_compile_time_arg_val(37);       // latent width in tiles
    constexpr bool phase2 = get_compile_time_arg_val(38) == 1;  // chunk-relative rotation (Task 4)
    constexpr uint32_t Wt = 2 * DHt;
```

change `constexpr auto q_args = TensorAccessorArgs<37>();` to `TensorAccessorArgs<39>()`, and after the `attention_sink_args` line add

```cpp
    constexpr auto w_rot_args = TensorAccessorArgs<attention_sink_args.next_compile_time_args_offset()>();
    constexpr auto cs_args = TensorAccessorArgs<w_rot_args.next_compile_time_args_offset()>();
    constexpr auto rot_step_args = TensorAccessorArgs<cs_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<rot_step_args.next_compile_time_args_offset()>();
```

(c) After the stock CB constants add

```cpp
    // Palu CBs (spec §4); the stock owners of these slots are compiled out in the envelope.
    constexpr uint32_t cb_lat_in = tt::CBIndex::c_9;
    constexpr uint32_t cb_w_rot = tt::CBIndex::c_4;
    constexpr uint32_t cb_cs_in = tt::CBIndex::c_13;
```

(d) After `const uint32_t attention_sink_addr = get_arg_val<uint32_t>(arg_idx++);` add

```cpp
    const uint32_t w_rot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cs_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rot_step_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t trans_mat_addr = get_arg_val<uint32_t>(arg_idx++);
```

(e) After `const auto mask_reader = TensorAccessor(mask_args, mask_addr);` add

```cpp
    const auto w_rot_reader = TensorAccessor(w_rot_args, w_rot_addr);
    const auto cs_reader = TensorAccessor(cs_args, cs_addr);
```

(f) In the per-head loop, the `else` (unpaged) branch: delete everything from `// Offset for current batch` through the closing `);` of `read_kv_mask_chunks<...>(...)` and put in its place:

```cpp
            // ---------------- Palu: latent chunks + rotation tables instead of K chunks ----------------
            const uint32_t lat_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * Rt;
            const uint32_t lat_head_offset = cur_head * St * Rt;
            const uint32_t lat_start_tile_id = lat_batch_offset + lat_head_offset + k_chunk_start * Sk_chunk_t_dynamic * Rt;
            const uint32_t v_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * vDHt;
            const uint32_t v_head_offset = cur_head * St * vDHt;
            const uint32_t v_start_tile_id = v_batch_offset + v_head_offset + k_chunk_start * Sk_chunk_t_dynamic * vDHt;
            read_w_rot<cb_w_rot, Rt, Wt, barrier_threshold>(cur_head, w_rot_reader);
            read_palu_chunks<Rt, vDHt, Wt, barrier_threshold, cb_lat_in, cb_v_in, cb_cs_in, phase2>(
                k_chunk_start, k_chunk_end, lat_start_tile_id, v_start_tile_id, Sk_chunk_t_dynamic,
                k_reader, v_reader, cs_reader);
```

(`k_reader` is the stock accessor over the K buffer, which is now the latent cache; `rot_step_addr`/`trans_mat_addr`/their accessor args are consumed in Task 4.)

- [ ] **Step 8: Compute kernel and its helper header**

Create `device/kernels/compute/palu_compute_common.hpp`:

```cpp
// SPDX-License-Identifier: Apache-2.0
// Compute helpers for palu_sdpa_decode (Palu fused-K flash decode). Tile indices are relative to the
// CB read pointer. Assumes compute_kernel_hw_startup<SrcOrder::Reverse> already ran (stock does it).
#pragma once
#include "api/compute/eltwise_binary_sfpu.h"

// K_rot for one chunk of Sk_chunk_t row tiles. For each row tile i:
//   AB_i    = lat_i @ [u_k | u_k T]            (1 x Rt) @ (Rt x Wt) -> Wt tiles into cb_ab
//   K[j, i] = AB_i[j] * cs[i, j] + AB_i[j + DHt] * cs[i, j + DHt]        for j < DHt
// packed into cb_k at index j * Sk_chunk_t + i: column-major over the chunk, which is the layout the
// stock QK matmul_blocks(..., transpose=true) reads K in. cs tiles are read at
// cs_tile_base + i*Wt + j and are NOT popped (the caller decides: Phase 1 pops the streamed chunk,
// Phase 2 keeps the resident table). The latent rows are popped here; cb_w stays resident.
template <
    uint32_t cb_lat,
    uint32_t cb_w,
    uint32_t cb_ab,
    uint32_t cb_cs,
    uint32_t cb_k,
    uint32_t Rt,
    uint32_t DHt,
    uint32_t dst_tiles>
ALWI void palu_build_k_chunk(uint32_t Sk_chunk_t, uint32_t cs_tile_base) {
    constexpr uint32_t Wt = 2 * DHt;
    constexpr uint32_t ab_w = Wt > dst_tiles ? dst_tiles : Wt;  // output columns per dest acquisition
    constexpr uint32_t ab_subblocks = Wt / ab_w;
    CircularBuffer lat(cb_lat), w(cb_w), ab(cb_ab), cs(cb_cs), k(cb_k);
    w.wait_front(Rt * Wt);
    cs.wait_front(cs_tile_base + Sk_chunk_t * Wt);
    k.reserve_back(Sk_chunk_t * DHt);
    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        // ---- AB_i = lat_i @ w_rot ----
        lat.wait_front((i + 1) * Rt);
        reconfig_data_format(cb_w, cb_lat);  // matmul: in1 -> SrcA, in0 -> SrcB
        pack_reconfig_data_format(cb_ab);
        matmul_block_init(cb_lat, cb_w, false /*transpose*/, ab_w /*ct_dim*/, 1 /*rt_dim*/, Rt /*kt_dim*/);
        ab.reserve_back(Wt);
        for (uint32_t sub = 0; sub < ab_subblocks; ++sub) {
            tile_regs_acquire();
            uint32_t in0_index = i * Rt;
            uint32_t in1_index = sub * ab_w;
            for (uint32_t kt = 0; kt < Rt; ++kt) {
                matmul_block(cb_lat, cb_w, in0_index, in1_index, 0, false, ab_w, 1, Rt);
                in0_index++;
                in1_index += Wt;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < ab_w; ++c) {
                pack_tile<true>(c, cb_ab, sub * ab_w + c);
            }
            tile_regs_release();
        }
        ab.push_back(Wt);
        // ---- K[:, i] = A_i * cos + B_i * sin ----
        ab.wait_front(Wt);
        reconfig_data_format(cb_ab, cb_cs);  // eltwise: in0 -> SrcA, in1 -> SrcB
        pack_reconfig_data_format(cb_k);
        mul_init(cb_ab, cb_cs);
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
        ab.pop_front(Wt);
    }
    k.push_back(Sk_chunk_t * DHt);
    lat.pop_front(Sk_chunk_t * Rt);
}
```

Edits to `device/kernels/compute/palu_sdpa_flash_decode.cpp`:

(a) After the `compute_common.hpp` include add

```cpp
#include "api/compute/eltwise_binary_sfpu.h"
#include "ttnn/operations/transformer/palu_sdpa_decode/device/kernels/compute/palu_compute_common.hpp"
```

(b) After `constexpr uint32_t original_block_size = get_compile_time_arg_val(31);` add

```cpp
    // ---------------- Palu (stage 2) ----------------
    constexpr uint32_t Rt = get_compile_time_arg_val(32);  // latent width in tiles
    constexpr uint32_t Wt = 2 * DHt;
```

(c) After the stock CB constants add

```cpp
    // Palu CBs (spec §4)
    constexpr uint32_t cb_lat_in = tt::CBIndex::c_9;
    constexpr uint32_t cb_w_rot = tt::CBIndex::c_4;
    constexpr uint32_t cb_ab = tt::CBIndex::c_10;
    constexpr uint32_t cb_cs_in = tt::CBIndex::c_13;
```

(d) At the top of the K-chunk loop body (the first statement inside `for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {`, before `// Reconfig register DF`):

```cpp
            // ---------------- Palu: rebuild this chunk's K on core into cb_k_in ----------------
            palu_build_k_chunk<cb_lat_in, cb_w_rot, cb_ab, cb_cs_in, cb_k_in, Rt, DHt, PALU_DST_TILES>(
                Sk_chunk_t_dynamic, /*cs_tile_base=*/0);
#ifndef PALU_PHASE2
            CircularBuffer(cb_cs_in).pop_front(Sk_chunk_t_dynamic * Wt);  // Phase 1: streamed rows, consumed
#endif
```

(e) Right after the `/* END OF FLASH ATTENTION LOOP */` comment add

```cpp
        CircularBuffer(cb_w_rot).pop_front(Rt * Wt);  // this head's [u_k | u_k T]
```

- [ ] **Step 9: Build and test**

```bash
scripts/build_ttnn.sh 2>&1 | tail -5
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q -k "phase1_matches and 64-1" 2>&1 | tail -15
```
Expected: pass. Debugging aids if it does not: a hang means a CB count mismatch between reader/compute (check every wait/pop pair above against the reader's pushes); an L1 allocation error names the CB, reduce `ab_tiles`/`cs_tiles` per the constraints table; a PCC near 0 with the right shape usually means the K tile order — re-check `j * Sk_chunk_t + i`. Then the whole file:

```bash
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q 2>&1 | tail -5
```
Expected: 8 passed (6 grid + r_v full width + envelope).

- [ ] **Step 10: Commit both repos**

```bash
cd third_party/tt-metal
git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "palu_sdpa_decode: Phase 1 -- K rebuilt on core from the latent cache, cos/sin streamed per chunk

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
cd ../..
git add third_party/tt-metal tests/test_palu_sdpa_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2 Task 3: Phase 1 fused kernel and op-level tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 4: Phase 2 kernel — chunk-relative rotation, query advanced on core

Outcome: with `rot_step` and `trans_mat` passed, the op reads the `[cos | sin]` table once per core (rows 0..c-1, chunk-relative) instead of once per chunk, rotates the query by `R_{-c}` per chunk in fp32 on core, and matches the references at 1k/4k/8k. This is the configuration the success bar is judged on.

Why this and not the K-side recurrence the approved design sketched: at the op's default chunk (512 positions) a K-side cos/sin state is `16 × 8` fp32 tiles = 512 KB and the step costs ~100 SFPU ops per chunk; the Q-side state is `PNHt × 4` tiles and one step is 4 tiny matmuls, 8 broadcast multiplies and 4 subtractions. The math (spec §5, `test_relative_rotation_identity`): `q R_p · (k R_{s0+t})ᵀ = (q R_{p-s0}) · (k R_t)ᵀ`.

**Files:**
- Modify (submodule): `device/palu_sdpa_decode_device_operation.cpp` (drop the "not implemented" fatal), `device/palu_sdpa_decode_program_factory.cpp` (c_12/c_14), `device/kernels/dataflow/palu_writer_decode.cpp` (no zero tile), `device/kernels/dataflow/palu_reader_decode.cpp`, `device/kernels/compute/palu_compute_common.hpp`, `device/kernels/compute/palu_sdpa_flash_decode.cpp`.
- Modify: `tests/test_palu_sdpa_device.py` (Phase-2 cases).

**Interfaces:**
- Consumes: everything Task 3 produced; `fused.rot_step_table`, `fused.trans_mat_tile`, `fused.chunk_size`.
- Produces: Phase 2 selected by `rot_step=` (+ `trans_mat=`); CBs `c_12` = rotated query state (fp32, `3 × PNHt × DHt` tiles), `c_14` = `rot_step` (fp32, `Wt` tiles); `c_4` now holds `Rt × Wt + 1` tiles per head (T tile last); kernel helpers `palu_init_q_state<cb_q_in, cb_q, n>()`, `palu_step_q<cb_q, cb_w, cb_step, DHt, PNHt>(t_idx)`; compute define `PALU_PHASE2`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_palu_sdpa_device.py`:

```python
@pytest.mark.parametrize("r_k", [32, 64, 96])
@pytest.mark.parametrize("batch", [1, 32])
def test_phase2_matches_reference_default_chunk(mesh_device, r_k, batch):
    _check(mesh_device, batch, 1024, r_k, 64, k_chunk=0, phase2=True)      # the op picks 512


@pytest.mark.parametrize("k_chunk", [128, 512])
def test_phase2_chunk_override(mesh_device, k_chunk):
    _check(mesh_device, 32, 1024, 64, 64, k_chunk=k_chunk, phase2=True)


@pytest.mark.parametrize("S", [4096, 8192])
def test_phase2_long_context(mesh_device, S):
    # batch 1 -> 16 cores per KV head -> every core starts at a different chunk (k_chunk_start steps)
    _check(mesh_device, 1, S, 64, 64, k_chunk=0, phase2=True)


def test_phase2_long_context_full_batch(mesh_device):
    _check(mesh_device, 32, 8192, 32, 64, k_chunk=0, phase2=True)


def test_phase1_and_phase2_agree(mesh_device):
    q, k_lat, u_k, v, pos, cos, sin = _inputs(32, 1024, 64, 64)
    p1 = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 64, 128, phase2=False)
    p2 = _fused(mesh_device, q, k_lat, u_k, v, pos, cos, sin, 64, 128, phase2=True)
    ok, msg = comp_pcc(p1, p2, 0.998)
    assert ok, msg
```

Run one to see it fail on the Task-3 guard:

```bash
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q -k "phase2_matches and 64-1" 2>&1 | tail -3
```
Expected: `RuntimeError: ... Phase 2 not implemented yet (Task 4)`.

- [ ] **Step 2: Host side**

`device/palu_sdpa_decode_device_operation.cpp`: delete the line `TT_FATAL(false, "palu_sdpa_decode: Phase 2 not implemented yet (Task 4)");`.

`device/palu_sdpa_decode_program_factory.cpp`:
- Delete the stock line `add_cb(CBIndex::c_12, scale_tiles * scalar_tile_size, scalar_df, scalar_tile_size, &scalar_tile);` (the zero tile only feeds the dynamic-chunk mask fusion, which the unpaged path never takes).
- After the Palu CB block from Task 3 add:

```cpp
    if (phase2) {
        const uint32_t fp32_tile_size = full_tile.get_tile_size(tt::DataFormat::Float32);
        // rotated-query state: [state | state @ T | next state] FIFO (palu_step_q), fp32
        add_cb(CBIndex::c_12, 3 * q_tiles * fp32_tile_size, tt::DataFormat::Float32, fp32_tile_size, &full_tile);
        // [cos(c f) | sin(c f)] on every row, resident for the run
        add_cb(CBIndex::c_14, Wt * fp32_tile_size, tt::DataFormat::Float32, fp32_tile_size, &full_tile);
    }
```

`device/kernels/dataflow/palu_writer_decode.cpp`: delete the line `dataflow_kernel_lib::prepare_zero_tile<cb_zero_in>();` (and the now-unused `cb_zero_in` constant if the compiler complains).

- [ ] **Step 3: Reader**

In `palu_reader_decode.cpp`, add the constant `constexpr uint32_t cb_rot_step = tt::CBIndex::c_14;` next to the other Palu CB constants. After the `read_q<...>(...)` call and the accessor definitions (before the per-head `for (uint32_t cur_head = ...)` loop) add:

```cpp
    if constexpr (phase2) {
        // Chunk-relative [cos | sin] rows t = 0..c-1: the first Sk_chunk_t row tiles of cs_tab, resident for the run.
        palu_read_tiles<cb_cs_in, barrier_threshold>(0, Sk_chunk_t_dynamic * Wt, cs_reader);
        const auto rot_step_reader = TensorAccessor(rot_step_args, rot_step_addr);
        palu_read_tiles<cb_rot_step, barrier_threshold>(0, Wt, rot_step_reader);
    }
```

and inside the per-head loop, right after `read_w_rot<...>(cur_head, w_rot_reader);`:

```cpp
            if constexpr (phase2) {
                // the T tile rides behind w_rot in the same CB (index Rt * Wt)
                const auto trans_mat_reader = TensorAccessor(trans_mat_args, trans_mat_addr);
                palu_read_tiles<cb_w_rot, barrier_threshold>(0, 1, trans_mat_reader);
            }
```

- [ ] **Step 4: Compute helpers**

Append to `palu_compute_common.hpp`:

```cpp
// Phase 2: the rotated query lives in cb_q as n = PNHt*DHt fp32 tiles at the front.
// palu_init_q_state copies the (already position-rotated) bf16 Q into that state.
template <uint32_t cb_q_in, uint32_t cb_q, uint32_t n>
ALWI void palu_init_q_state() {
    CircularBuffer src(cb_q_in), q(cb_q);
    src.wait_front(n);
    q.reserve_back(n);
    reconfig_data_format_srca(cb_q_in);
    pack_reconfig_data_format(cb_q);
    copy_init(cb_q_in);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile(cb_q_in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_q);
        tile_regs_release();
    }
    q.push_back(n);
}

// state <- state R_{-c} = state * cos(c f) - (state @ T) * sin(c f).
// cb_step holds Wt fp32 tiles [cos(c f) | sin(c f)] with every row identical (row broadcast);
// the T tile is cb_w[t_idx]. cb_q is used as a 3n FIFO: [state | state @ T | new state], then pop 2n.
template <uint32_t cb_q, uint32_t cb_w, uint32_t cb_step, uint32_t DHt, uint32_t PNHt>
ALWI void palu_step_q(uint32_t t_idx) {
    constexpr uint32_t n = PNHt * DHt;
    CircularBuffer q(cb_q), step(cb_step);
    q.wait_front(n);
    step.wait_front(2 * DHt);
    // qT = state @ T, one tile at a time (T is block-diagonal per 32 columns)
    q.reserve_back(n);
    reconfig_data_format(cb_w, cb_q);  // matmul: in1 (T) -> SrcA, in0 (state) -> SrcB
    pack_reconfig_data_format(cb_q);
    matmul_init(cb_q, cb_w);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        matmul_tiles(cb_q, cb_w, i, t_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_q);
        tile_regs_release();
    }
    q.push_back(n);
    // new state = state * cos_c - qT * sin_c
    q.reserve_back(n);
    reconfig_data_format(cb_q, cb_step);
    mul_bcast_rows_init(cb_q, cb_step);
    sub_binary_tile_init();
    for (uint32_t r = 0; r < PNHt; ++r) {
        for (uint32_t j = 0; j < DHt; ++j) {
            const uint32_t i = r * DHt + j;
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_q, cb_step, i, j, 0);            // state[i] * cos_c[j]
            mul_tiles_bcast_rows(cb_q, cb_step, n + i, DHt + j, 1);  // qT[i]    * sin_c[j]
            sub_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_q);
            tile_regs_release();
        }
    }
    q.push_back(n);
    q.pop_front(2 * n);
}
```

- [ ] **Step 5: Compute kernel**

In `palu_sdpa_flash_decode.cpp`:

(a) Add the constants `constexpr uint32_t cb_qrot = tt::CBIndex::c_12;` and `constexpr uint32_t cb_rot_step = tt::CBIndex::c_14;` next to the Palu CB constants, and right after them:

```cpp
#ifdef PALU_PHASE2
    constexpr uint32_t cb_q_src = cb_qrot;   // QK reads the on-core rotated query
    constexpr uint32_t w_rot_cb_tiles = Rt * Wt + 1;
#else
    constexpr uint32_t cb_q_src = cb_q_in;
    constexpr uint32_t w_rot_cb_tiles = Rt * Wt;
#endif
```

(b) Inside `for (uint32_t cur_head_work = 0; ...) {`, after the ping-pong buffer reset lines and before `/* START OF FLASH ATTENTION LOOP */`:

```cpp
#ifdef PALU_PHASE2
        // q~ = q R_{-k_chunk_start * c}: copy Q into the fp32 state, then one R_{-c} step per skipped chunk
        palu_init_q_state<cb_q_in, cb_qrot, q_chunk_tiles>();
        for (uint32_t s = 0; s < k_chunk_start; ++s) {
            palu_step_q<cb_qrot, cb_w_rot, cb_rot_step, DHt, Sq_chunk_t>(Rt * Wt);
        }
#endif
```

(c) In the K-chunk loop, before the `palu_build_k_chunk` call from Task 3:

```cpp
#ifdef PALU_PHASE2
            if (k_chunk > k_chunk_start) {
                palu_step_q<cb_qrot, cb_w_rot, cb_rot_step, DHt, Sq_chunk_t>(Rt * Wt);  // q~ <- q~ R_{-c}
            }
#endif
```

(d) The stock QK matmul: change `reconfig_data_format(cb_k_in, cb_q_in);` (the first statement after the Palu block) to `reconfig_data_format(cb_k_in, cb_q_src);`, and the first argument of the `matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, ...)` call to `cb_q_src`. Nothing else in the loop changes.

(e) Replace the Task-3 line after `/* END OF FLASH ATTENTION LOOP */` with:

```cpp
        CircularBuffer(cb_w_rot).pop_front(w_rot_cb_tiles);  // this head's [u_k | u_k T] (+ T)
#ifdef PALU_PHASE2
        CircularBuffer(cb_qrot).pop_front(q_chunk_tiles);    // the last q~ state
#endif
```

- [ ] **Step 6: Build, test, commit**

```bash
scripts/build_ttnn.sh 2>&1 | tail -3
scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q 2>&1 | tail -5
```
Expected: all pass (8 from Task 3 + 6 + 2 + 2 + 1 + 1 = 20). If `test_phase2_long_context_full_batch` fails with an L1 allocation error at the default chunk, record the CB sizes it prints in the report file, and make the test pass `k_chunk=128` for that case; Task 5 then defaults `fused_k_chunk` to 128 and the results note it.

```bash
cd third_party/tt-metal
git add ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "palu_sdpa_decode: Phase 2 -- chunk-relative rotation, query advanced on core in fp32

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
cd ../..
git add third_party/tt-metal tests/test_palu_sdpa_device.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2 Task 4: Phase 2 fused kernel and long-context op tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 5: `PaluAttention` fused decode path and attention-level equivalence test

Outcome: with `PaluRuntime.use_fused_k` set, a layer with `r_k < head_dim` decodes through `palu_sdpa_decode` (Phase 1 or, with `cs_recurrence`, Phase 2) and produces the same output and identical caches as the stage-1 naive path. The V-only and identity rows are untouched.

**Files:**
- Modify: `palu_tt/convert.py` (`PaluRuntime.fused_k_chunk`), `palu_tt/attention.py`, `palu_tt/model.py`, `tests/test_attention_device.py`, `tests/test_convert.py` (one assertion).

**Interfaces:**
- Consumes: `ttnn.transformer.palu_sdpa_decode` (Tasks 3-4), `palu_tt.fused` (Task 2), state-dict key `layers.{i}.attention.palu.w_rot`.
- Produces: `PaluRuntime(recon_len, check, use_fused_k, cs_recurrence, fused_k_chunk: Optional[int] = None)`; `PaluAttention.use_fused_k`, `.phase2`, `.w_rot`, `.cs_tab`, `.rot_step`, `.trans_mat`, `.fused_program_config`, `.fused_compute_cfg`; `attention._fused_tables(configuration, mesh_device, chunk) -> (cs_tab, rot_step | None, trans_mat | None)`; `model.build_model(..., use_fused_k=False, cs_recurrence=False, fused_k_chunk=None)`.

- [ ] **Step 1: Failing tests**

Append to `tests/test_attention_device.py`:

```python
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
```

Append to `tests/test_convert.py`'s `test_convert_version_is_3_and_runtime_flags_default_off`: `assert rt.fused_k_chunk is None`.

- [ ] **Step 2: Runtime flag**

In `palu_tt/convert.py` `PaluRuntime`, after `cs_recurrence`:

```python
    # stage 2: override the fused op's K chunk (positions; power of two, multiple of 32). None = the
    # op's own choice (largest power of two dividing max_seq_len, capped at 512), which is what the
    # stock decode op uses too.
    fused_k_chunk: Optional[int] = None
```

- [ ] **Step 3: `attention.py`**

Add `from palu_tt import fused` to the imports and this module-level helper after `cache_tag`:

```python
def _fused_tables(configuration, mesh_device, chunk):
    """Device tensors the fused kernel shares across layers: the [cos | sin] table (bf16) and, for
    Phase 2, the R_{-chunk} step constants (fp32) and the 32x32 pair-rotation tile (bf16). Built once
    per model and kept on PaluConfig.device_tables; a second layer with the same key reuses them."""
    rt = configuration.palu.runtime
    key = ("fused", id(mesh_device), configuration.max_seq_len, int(chunk), bool(rt.cs_recurrence))
    tabs = configuration.palu.device_tables
    if key not in tabs:
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
        cos, sin = fused.cos_sin_tables(configuration.head_dim, configuration.max_seq_len, configuration.rope_theta,
                                        configuration.rope_scaling)
        cs = up(fused.cs_table(cos, sin, configuration.max_seq_len)[None, None], ttnn.bfloat16)
        step = tm = None
        if rt.cs_recurrence:
            step = up(fused.rot_step_table(cos, sin, chunk)[None, None], ttnn.float32)
            tm = up(fused.trans_mat_tile(), ttnn.bfloat16)
        tabs[key] = (cs, step, tm)
    return tabs[key]
```

In `__init__`, right after `self.recon_v = self.r_v < self.head_dim`:

```python
        rt = self.palu.runtime
        # stage 2: fused K reconstruction. Only where K really is low rank; on the V-only and identity
        # rows the stock K path stays, flag or no flag.
        self.use_fused_k = self.recon_k and bool(rt.use_fused_k)
        self.phase2 = self.use_fused_k and bool(rt.cs_recurrence)
```

Replace the `if self.recon_k:` up-projection block with:

```python
        if self.recon_k:
            u_k = state_dict[f"{P}.u_k"]  # [n_kv, r_k, hd]
            self.u_k_prefill = ttnn.as_tensor(pad_u(u_k, self.r_k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_u_k_prefill"))
            if self.use_fused_k:
                # decode: the kernel's packed rotation weight [1, n_kv, r_k, 2*hd] = [u_k | u_k T], head-sharded like u_k
                self.w_rot = ttnn.as_tensor(
                    state_dict[f"{P}.w_rot"].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads, cache_file_name=cache_name(f"{tag}_w_rot"))
            else:
                # decode: batched (non-broadcast) matmul needs matching batch dims -> replicate over batch
                self.u_k_decode = ttnn.as_tensor(
                    u_k.unsqueeze(0).expand(B, -1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard_heads,
                    cache_file_name=cache_name(f"{tag}_u_k_decode_b{B}"),
                )
```

Replace the unconditional `self.cos_prefill, self.sin_prefill = get_rot_mats(...)` block at the end of `__init__` with:

```python
        if self.recon_k and not self.use_fused_k:
            # prefill-mode cos/sin for the naive K reconstruction in decode
            self.cos_prefill, self.sin_prefill = get_rot_mats(
                head_dim=self.head_dim, device=mesh_device, seq_len=self.max_seq_len, theta=configuration.rope_theta,
                rope_scaling=configuration.rope_scaling, datatype=ttnn.bfloat16, rot_mats_layout=ttnn.TILE_LAYOUT,
            )
        if self.use_fused_k:
            pc = self.args.get_attn_sdpa_decode_program_config(None)
            if rt.fused_k_chunk:
                pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                            q_chunk_size=0, k_chunk_size=int(rt.fused_k_chunk))
            self.fused_program_config = pc
            chunk = pc.k_chunk_size or fused.chunk_size(self.max_seq_len)   # what the op will pick
            self.cs_tab, self.rot_step, self.trans_mat = _fused_tables(configuration, mesh_device, chunk)
            cfg = self.sdpa_decode_compute_kernel_cfg
            # Phase 2 keeps the rotated query in fp32 and needs an fp32 dest; the HiFi4 preset has it
            self.fused_compute_cfg = cfg if (not self.phase2 or cfg.fp32_dest_acc_en) else configuration.compute_kernel_config_hifi4
```

In `forward_decode`, replace the block from `if self.recon_k:` (the `# K reconstruction (the naive Palu path; ...)` one) through the `if self.recon_k: ttnn.deallocate(k_att) ...` cleanup after the MLA call with:

```python
        if self.use_fused_k:
            # stage 2: K rebuilt and rotated inside the kernel; the caches go in as they are
            attn = ttnn.transformer.palu_sdpa_decode(
                q, keys, self.w_rot, self.cs_tab, values, self.r_v, current_pos,
                rot_step=self.rot_step, trans_mat=self.trans_mat, scale=self.scale,
                program_config=self.fused_program_config, compute_kernel_config=self.fused_compute_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)                                        # [1, B, nh, r_v]
            ttnn.deallocate(q)
        else:
            if self.recon_k:
                ... (the existing naive reconstruction block, verbatim) ...
            else:
                S, k_att, v_att = self.max_seq_len, keys, values
            attn = ttnn.transformer.flash_multi_latent_attention_decode(... existing call ...)
            ttnn.deallocate(q)
            if self.recon_k:
                ttnn.deallocate(k_att)
                if S < self.max_seq_len:
                    ttnn.deallocate(v_att)
```

(Indent the existing code one level; do not change it.) The padding of `attn` to `head_dim` and the stock tail stay as they are.

- [ ] **Step 4: `model.py`**

```python
def build_model(mesh_device, kind: str, palu_dir: Optional[str] = None, max_batch_size: int = 1, max_seq_len: int = 2048,
                n_layers: Optional[int] = None, dtype=ttnn.bfloat8_b, latent_dtype=None,
                use_fused_k: bool = False, cs_recurrence: bool = False, fused_k_chunk: Optional[int] = None):
    """latent_dtype: optional ttnn dtype for the Palu latent K/V caches (None = the model's KV cache
    dtype, i.e. bfp8). use_fused_k / cs_recurrence / fused_k_chunk: stage-2 fused decode kernel
    (Phase 1, Phase 2, chunk override). Only meaningful for kind="palu"."""
    ...
    if kind == "palu":
        cfg = convert.add_palu_to_state_dict(...)
        cfg.latent_dtype = latent_dtype
        cfg.runtime.use_fused_k, cfg.runtime.cs_recurrence, cfg.runtime.fused_k_chunk = use_fused_k, cs_recurrence, fused_k_chunk
        args.palu = cfg
```

- [ ] **Step 5: Run**

```bash
nice -n 10 pytest tests/test_convert.py tests/test_attention_tag.py -q 2>&1 | tail -2
scripts/run_device.sh pytest tests/test_attention_device.py -q 2>&1 | tail -5
```
Expected: CPU pass; device: the 11 stage-1 tests plus 4 fused + 1 flag test pass. Record in the report file which compute config the fused op ran with (`fusedm.fused_compute_cfg`) and the stock SDPA decode config (`fusedm.sdpa_decode_compute_kernel_cfg`).

- [ ] **Step 6: Commit**

```bash
git add palu_tt/convert.py palu_tt/attention.py palu_tt/model.py tests/test_attention_device.py tests/test_convert.py
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2 Task 5: fused K decode path in PaluAttention behind PaluRuntime.use_fused_k

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

### Task 6: Evaluation — decode fidelity, benchmarks, `docs/results/stage2.md`

Outcome: the stage-1 harness re-run with the fused rows, a decode-fidelity check (perplexity is a prefill measurement and cannot see the decode kernel), and a rendered results document that answers the success bar.

**Files:**
- Create: `palu_tt/decode_check.py`, `docs/results/stage2.md` (rendered + hand-written section), `docs/results/raw/{bench_decode,bench_attention}_*_fk*.json`, `docs/results/raw/decode_check_*.json`.
- Modify: `palu_tt/bench.py`, `palu_tt/report.py`, `tests/test_report.py`, `README.md`.

**Interfaces:**
- Consumes: `model.build_model(..., use_fused_k, cs_recurrence, fused_k_chunk)`, `generate.prefill_last_logits`, `generate.decode_step`, `eval_ppl.wikitext2_ids`.
- Produces: `python -m palu_tt.bench --row R --kind palu --palu-dir D --what decode|attention --fused off|p1|p2 [--fused-k-chunk N]`; `python -m palu_tt.decode_check --row R --palu-dir D --fused p1|p2 [--steps 64] [--fused-k-chunk N]` -> `docs/results/raw/decode_check_R.json` with `{"row", "palu_dir", "fused", "steps", "pcc_mean", "pcc_min", "argmax_agree"}`; `python -m palu_tt.report --stage 2` -> `docs/results/stage2.md`. Row naming: `<stage-1 row>_fk1` / `_fk2`.

- [ ] **Step 1: Harness flags**

`palu_tt/bench.py`:
- `main()`: add `p.add_argument("--fused", choices=["off", "p1", "p2"], default="off")` and `p.add_argument("--fused-k-chunk", type=int, default=None)`; store `FUSED = {"use_fused_k": a.fused != "off", "cs_recurrence": a.fused == "p2", "fused_k_chunk": a.fused_k_chunk}` in a module global before dispatch and write `r["fused"] = a.fused; r["fused_k_chunk"] = a.fused_k_chunk` into the JSON.
- `_decode_one`, `bench_prefill`, `bench_maxctx`: pass `**FUSED` to `pm.build_model(...)`.
- `_attention_one`: after `args.palu = convert.add_palu_to_state_dict(...)` set `args.palu.runtime.use_fused_k, args.palu.runtime.cs_recurrence, args.palu.runtime.fused_k_chunk = FUSED["use_fused_k"], FUSED["cs_recurrence"], FUSED["fused_k_chunk"]`.
- `--kind stock` with `--fused` other than `off` is an argparse error.

- [ ] **Step 2: `palu_tt/decode_check.py`**

```python
"""Decode-path fidelity of the fused kernel: naive vs fused Palu model, teacher-forced.

Perplexity in this study is measured through prefill windows, so it cannot see the decode kernel
at all. This harness feeds both models the same prompt (the first 512 wikitext-2 tokens), then the
same next `steps` tokens one decode step at a time, and compares the per-step logits: PCC per step
and how often the argmax agrees. The two models are built one after the other (two 8B models do not
both fit next to their caches).

Usage: python -m palu_tt.decode_check --row palu_full_50_fk2 --palu-dir compressed/palu_full_0.5 --fused p2
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from models.common.utility_functions import comp_pcc

from palu_tt import device as dev, eval_ppl, generate, hf, model as pm

RAW = Path("docs/results/raw")
PROMPT = 512


def _logits(md, palu_dir, ids, steps, **flags):
    args, model, cfg = pm.build_model(md, "palu", palu_dir, max_batch_size=1, max_seq_len=2048, **flags)
    prompt, teacher = ids[0, :PROMPT], ids[0, PROMPT:PROMPT + steps]
    out = [generate.prefill_last_logits(args, model, prompt[None]).float()]
    pos = PROMPT
    for t in teacher[:-1].tolist():
        out.append(generate.decode_step(args, model, cfg, t, pos).float())
        pos += 1
    del model
    gc.collect()
    return torch.stack(out)  # [steps, vocab]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--palu-dir", required=True)
    p.add_argument("--fused", choices=["p1", "p2"], required=True)
    p.add_argument("--fused-k-chunk", type=int, default=None)
    p.add_argument("--steps", type=int, default=64)
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    md = dev.open_n300()
    try:
        args_tok = pm.ModelArgs(md, instruct=True, max_batch_size=1, max_seq_len=2048, cache_hf=False)
        ids = eval_ppl.wikitext2_ids(args_tok.tokenizer)
        naive = _logits(md, a.palu_dir, ids, a.steps)
        fused = _logits(md, a.palu_dir, ids, a.steps, use_fused_k=True, cs_recurrence=a.fused == "p2",
                        fused_k_chunk=a.fused_k_chunk)
    finally:
        dev.close(md)
    pccs = [comp_pcc(naive[i], fused[i], 0.0)[1] for i in range(a.steps)]
    pccs = [float(s.split("PCC: ")[1]) for s in pccs]
    r = {"row": a.row, "palu_dir": a.palu_dir, "fused": a.fused, "fused_k_chunk": a.fused_k_chunk, "steps": a.steps,
         "pcc_mean": sum(pccs) / len(pccs), "pcc_min": min(pccs),
         "argmax_agree": float((naive.argmax(-1) == fused.argmax(-1)).float().mean())}
    (RAW / f"decode_check_{a.row}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
```

(`comp_pcc` returns `(bool, "PCC: 0.9987")`-style messages; if the message format differs, parse whatever it prints -- check one call interactively first. `pm.ModelArgs` is `models.tt_transformers.tt.model_config.ModelArgs`, imported in `palu_tt/model.py`.)

- [ ] **Step 3: `report.py` for stage 2**

- Add `--stage {1,2}` (default 1). Stage 2 renders to `docs/results/stage2.md` with:

```python
ROWS2 = {
    "baseline": "stock `tt_transformers` attention",
    "svd_r64": "plain per-head SVD, r_k = r_v = 64 (naive K reconstruction)",
    "svd_r64_fk1": "  same, fused kernel Phase 1 (cos/sin streamed)",
    "svd_r64_fk2": "  same, fused kernel Phase 2 (chunk-relative rotation)",
    "svd_r96": "plain per-head SVD, r_k = r_v = 96 (naive)",
    "svd_r96_fk2": "  same, fused Phase 2",
    "palu_full_50": "Palu full recipe, ratio 0.5 (naive)",
    "palu_full_50_fk1": "  same, fused Phase 1",
    "palu_full_50_fk2": "  same, fused Phase 2",
    "palu_full_70": "Palu full recipe, ratio 0.7 (naive)",
    "palu_full_70_fk1": "  same, fused Phase 1",
    "palu_full_70_fk2": "  same, fused Phase 2",
}
```

- `_row_data(raw, row)`: for a row ending in `_fk1`/`_fk2`, load `bench_decode_{row}` and `bench_attention_{row}` from the row's own files, and `ppl_{base}`, `ppl_cpu_{base}`, `bench_prefill_{base}`, `bench_maxctx_{base}` from the base row (`base = row.rsplit("_fk", 1)[0]`): prefill, perplexity, KV bytes and max context are identical by construction. Also load `decode_check_{row}` (None for base rows).

```python
def _base_row(row):
    """'palu_full_50_fk2' -> 'palu_full_50'; a stage-1 row maps to itself."""
    return row.rsplit("_fk", 1)[0] if "_fk" in row else row


def _row_data(raw, row):
    base = _base_row(row)
    ppl = _load(raw, f"ppl_{row}") or _load(raw, f"ppl_{base}")           # prefill measurement: same for fused rows
    cpu = _load(raw, f"ppl_cpu_{base}")
    dec = _load(raw, f"bench_decode_{row}")
    att = _load(raw, f"bench_attention_{row}")
    pre = _load(raw, f"bench_prefill_{row}") or _load(raw, f"bench_prefill_{base}")
    mx = _load(raw, f"bench_maxctx_{row}") or _load(raw, f"bench_maxctx_{base}")
    chk = _load(raw, f"decode_check_{row}")
    return ppl, cpu, dec, att, pre, mx, chk


def _vs_stock(stock_dec, dec, ctx):
    """stock min_ms / row min_ms at one context, or '-' when either cell is missing or errored."""
    a, b = _cell(stock_dec, ctx), _cell(dec, ctx)
    if not a or not b or "min_ms" not in a or "min_ms" not in b:
        return "-"
    return f"{a['min_ms'] / b['min_ms']:.2f}x"


def _fidelity(chk):
    if not chk:
        return "-", "-"
    return f"{chk['pcc_min']:.4f} / {chk['pcc_mean']:.4f}", f"{chk['argmax_agree']:.3f}"
```

Stage 1's `main` unpacks six values from `_row_data`; give it the seventh (`chk`) and ignore it there, or route both stages through one `main(raw, out, rows, title, intro)` that adds the two fidelity columns and the `vs stock` column whenever `rows is ROWS2`.
- Quality table gains two columns after the perplexities: `decode PCC (min / mean, 64 steps)` and `argmax agree` from the decode-check JSON (dashes when absent); the intro line says perplexity is a prefill measurement.
- Decode/attention latency table and prefill table as in stage 1; add a `vs stock` column to the latency table = `stock min_ms / row min_ms` at the same context (dash when either is missing/errored).
- Title: "Stage 2 results: fused K-reconstruction decode kernel on a Wormhole n300". Same `MARKER` convention for the hand-written tail.
- `tests/test_report.py`: add a fixture-directory test that renders stage 2 with one fk row present (decode + attention + decode_check JSONs) and asserts the row line contains the PCC and the `vs stock` ratio, and that a missing `ppl_palu_full_50_fk2.json` falls back to the base row's perplexity.

```bash
nice -n 10 pytest tests/test_report.py tests/test_bench_utils.py -q 2>&1 | tail -2
```

- [ ] **Step 4: Run the device matrix (idle host only; `device_free.sh` gates every command)**

```bash
for row in svd_r64:compressed/svd_r64:p1 svd_r64:compressed/svd_r64:p2 svd_r96:compressed/svd_r96:p2 \
           palu_full_50:compressed/palu_full_0.5:p1 palu_full_50:compressed/palu_full_0.5:p2 \
           palu_full_70:compressed/palu_full_0.7:p1 palu_full_70:compressed/palu_full_0.7:p2; do
  IFS=: read base dir ph <<< "$row"; r="${base}_fk${ph#p}"
  scripts/run_device.sh python -m palu_tt.bench --row $r --kind palu --palu-dir $dir --what decode --fused $ph 2>&1 | tee docs/results/raw/bench_decode_$r.log | tail -1
  scripts/run_device.sh python -m palu_tt.bench --row $r --kind palu --palu-dir $dir --what attention --fused $ph 2>&1 | tee docs/results/raw/bench_attention_$r.log | tail -1
done
for row in palu_full_50_fk1:compressed/palu_full_0.5:p1 palu_full_50_fk2:compressed/palu_full_0.5:p2 \
           palu_full_70_fk2:compressed/palu_full_0.7:p2 svd_r64_fk2:compressed/svd_r64:p2; do
  IFS=: read r dir ph <<< "$row"
  scripts/run_device.sh python -m palu_tt.decode_check --row $r --palu-dir $dir --fused $ph 2>&1 | tee docs/results/raw/decode_check_$r.log | tail -1
done
```

If Task 4 ended up needing `fused_k_chunk=128`, add `--fused-k-chunk 128` to every fused command above and also measure the stock sensitivity row `baseline_k128` (`--what attention` with `args.sdpa_decode_k_chunk_size = 128` set in `_attention_one` behind a new `--stock-k-chunk` flag) so the table can show what chunk 128 costs stock.

Re-measure any row whose timing was taken while another job of ours was on the host (compare the run's wall-clock with the calibration/ppl logs), exactly as in stage 1. Never commit the `.log` files.

- [ ] **Step 5: Render, write up, commit**

```bash
python -m palu_tt.report --stage 2
```

Below the marker in `docs/results/stage2.md`, write (hand-written, preserved on re-render):
1. the success-bar verdict, one line per criterion with the numbers (fused-vs-naive PCC at op/attention/model level; calibrated rows' decode `min_ms` vs stock at 1k/4k; 8k runs);
2. what Phase 1 vs Phase 2 shows (DRAM traffic reasoning from spec §4, with the measured attention-only numbers);
3. the compute-config and chunk-size caveats (fp32 dest, `fused_k_chunk` if used, stock sensitivity row);
4. what did not work or was not measured.

Update `README.md` (stage-2 commands: build script, `--fused`, `decode_check`, `report --stage 2`) and the spec's status line. Then:

```bash
git add palu_tt/bench.py palu_tt/decode_check.py palu_tt/report.py tests/test_report.py docs/results/stage2.md docs/results/raw/*.json README.md docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md
git -c user.name=ryan -c user.email=ryankert01@gmail.com commit -m "Stage 2 Task 6: fused-kernel benchmarks, decode fidelity check, stage 2 results document

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs"
```

---

## After the last task

Whole-branch review, then `superpowers:finishing-a-development-branch` (stage 1 merged to `main` locally; do the same). Publishing the submodule branch needs the fork `ryankert01/tt-metal` and credentials on this box; until then `.gitmodules` keeps the upstream URL and the README says the pinned commit is local-only.
