# Stage 2c Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) score the two G-LRD dumps re-calibrated with 256 Fisher samples; (2) make the fused op read each group latent chunk once per core for all the heads the core owns, with a core able to own all four heads of a group while the sequence is still split over cores.

**Architecture:** The op keeps flash decode's structure. A new compile-time mode `PALU_GROUP_SHARED` swaps the head/chunk loop order in the compute kernel (chunk outer, head inner), keeps `H = num_heads_per_core` flash states and rotated-query states as FIFO-rotated blocks in the existing state CBs, holds `H` per-head weight blocks resident in `c_4`, and lets the reader push every latent chunk once. Phase B lifts the "one head per core when the sequence is split" restriction with a per-head-counting reduction semaphore, and the host picks `H` per layer from an L1 estimate. Spec: `docs/superpowers/specs/2026-09-12-palu-tt-stage2c-design.md`.

**Tech Stack:** tt-metal submodule branch `palu-sdpa` (C++ host op, RISC-V kernels), ttnn, PyTorch, lm-eval 0.4.2, pytest.

## Global Constraints

- **Shared server.** No sudo. Every device command through `scripts/run_device.sh <cmd>` (refuses when the n300 is busy; wait and retry, never bypass). `source scripts/env.sh && source .venv/bin/activate` first, from the repo root `/home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9`. CPU work `nice -n 10`, <= 12 threads. Never touch `/home/user01/tt-data`. A 256-sample Fisher calibration (CPU, 12 threads) runs in the background until about 22:40 on 2026-09-12; keep builds at `JOBS=8`. Never run bare `pytest` or `pytest tests/`; name the test file.
- **Two repos.** tt-metal changes are committed inside the submodule on branch `palu-sdpa` with `--no-verify`: `git -C third_party/tt-metal add <files>; git -C third_party/tt-metal -c user.name=ryan -c user.email=ryankert01@gmail.com commit --no-verify -m ...`; the superproject then records the pointer with `git add third_party/tt-metal`. Never `git add third_party/palu`, `scripts/env.sh` or `.superpowers/`. Host C++ edits need `scripts/build_ttnn.sh` (10-25 min); kernel edits are JIT. `.log` files are never committed.
- **Commit trailers**, verbatim last two lines of every commit in both repos: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01SEbH3ediiqrAP9zmP59kKs`. Commit as `git -c user.name=ryan -c user.email=ryankert01@gmail.com commit`. Do not substitute another model name.
- **Kernel contract carried from stage 2b (must not change):** `heads_per_group = w_rot.shape[1] / k_latent.shape[1]`; `num_kv_heads` for the work layout is `w_rot.shape[1]`; `r_k` any multiple of 32 in 32..512; optional `u_v [1, n_kv, r_v_g, 128]` selects in-kernel V; per-head weight block in `c_4` is `[w_rot (Rt*Wt) | u_v (Rvt*DHt) | T (Phase 2)]`; CB sizes are whole numbers of push blocks; compute-side operand reads never wrap. With `heads_per_core = 1` the op must be bit-identical to stage 2b (`tests/test_palu_sdpa_device.py`, 31 tests, unchanged).
- **New op parameter:** `heads_per_core: Optional[int]` on `palu_sdpa_decode` (Python keyword, default `None` = the stock rule, per-head kernel). When given it must divide `heads_per_group` and `num_kv_heads`. Group-shared mode is **opt-in**: `group_shared = heads_per_core > 1` (Task 1 finding: keying it off the stock split would turn it on at batch 32 and exceed L1 on the widest existing tests, breaking the no-regression bar); `heads_per_core == 1` means "per-head kernel, stock split" (it does not pin one head per core). A core's heads in group mode are `[cur_head_group*H, cur_head_group*H + H)` and all lie in one group.
- **Bit-identity bar:** group-shared output equals the per-head output (`torch.equal` on the bf16 result) for the same inputs. If a legitimate rounding difference appears, the test bar becomes PCC >= 0.9999 and the report file records why.
- **Rows:** `glrd_50_f256`, `glrd_70_f256` (dumps `compressed/glrd_0.5_f256`, `compressed/glrd_0.7_f256`, produced by the background calibration; group size 4); kernel rows `glrd_50_fk2g`, `glrd_70_fk2g` (fused Phase 2 with automatic `H`), compared with the stage 2b `_fk2` rows.
- **Time budget:** ~3.5 h of device time (spec §4).

## File structure

- Submodule `third_party/tt-metal/ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/`:
  - `palu_sdpa_decode.{hpp,cpp}`, `palu_sdpa_decode_nanobind.cpp`, `device/palu_sdpa_decode_device_operation{,_types}.{hpp,cpp}` -- `heads_per_core` attribute, validation.
  - `device/palu_sdpa_decode_program_factory.cpp` -- work layout with `H`, CB sizing (`H x` state, `H` weight blocks, `H x q`), `PALU_GROUP_SHARED` define, per-head-counting semaphore plumbing, restriction lift.
  - `device/kernels/compute/palu_sdpa_flash_decode.cpp`, `device/kernels/compute/palu_compute_common.hpp` -- loop swap, helper `pop`/block-base parameters.
  - `device/kernels/dataflow/palu_reader_decode.cpp`, `palu_dataflow_common.hpp` -- read Q and weights for `H` heads, latent chunks once.
  - `device/kernels/dataflow/palu_writer_decode.cpp` -- per-head reduction loop without the `H == 1` assert; semaphore nibble compared against `h + 1`.
- `palu_tt/fused.py` -- `group_heads_per_core(r_k, r_v, chunk, phase2, heads_per_group, l1_budget)` L1 estimate; `palu_tt/convert.py` -- `PaluRuntime.group_heads_per_core`; `palu_tt/attention.py` -- passes `heads_per_core`; `palu_tt/bench.py`, `decode_check.py`, `eval_tasks.py`, `model.py` -- `--group-heads`; `palu_tt/report.py` -- `--stage 2c`.
- Tests: `tests/test_palu_sdpa_device.py` (new group-shared cases), `tests/test_attention_device.py`, `tests/test_fused.py`, `tests/test_report.py`.

---

### Task 1: Group-shared kernel mode, Phase A (a core's heads share the latent reads)

**Files:**
- Modify (submodule): `palu_sdpa_decode.hpp/.cpp`, `palu_sdpa_decode_nanobind.cpp`, `device/palu_sdpa_decode_device_operation_types.hpp`, `device/palu_sdpa_decode_device_operation.cpp`, `device/palu_sdpa_decode_program_factory.cpp`, `device/kernels/compute/palu_sdpa_flash_decode.cpp`, `device/kernels/compute/palu_compute_common.hpp`, `device/kernels/dataflow/palu_reader_decode.cpp`, `device/kernels/dataflow/palu_dataflow_common.hpp`.
- Test: `tests/test_palu_sdpa_device.py`, `palu_tt/fused.py` (`reference_decode` unchanged; only the call helper gains `heads_per_core`).

**Interfaces:**
- Produces: `ttnn.transformer.palu_sdpa_decode(..., heads_per_core=None)`; op attribute `heads_per_core` (0 = stock rule); compile-time define `PALU_GROUP_SHARED`; compute helpers `palu_build_k_chunk<..., pop_latent>(Sk_chunk_t, cs_tile_base, w_block_base)` and `palu_build_v_chunk<..., pop_latent>(Sk_chunk_t, w_block_base)`; `palu_init_q_state(t_tile_idx + w_block_base)`.
- Phase A scope: `num_cores_per_head == 1` only (the factory still rejects `H > 1` with a sequence split; Task 2 lifts it). At batch 32 the stock rule already gives `H = 2`, so this task is measurable at batch 32.

- [ ] **Step 1: Failing op test.** Append to `tests/test_palu_sdpa_device.py` (reuse the file's `_run_op` / reference helpers; adapt names to what exists):

```python
@pytest.mark.parametrize("r_k,r_v", [(64, 512), (128, 256), (288, 512)], ids=["rk64_rv512", "rk128_rv256", "rk288_rv512"])
@pytest.mark.parametrize("phase2", [False, True], ids=["p1", "p2"])
def test_group_shared_matches_per_head_batch32(mesh_device, r_k, r_v, phase2):
    """Group-shared mode (the two heads a core owns at batch 32 share one latent read) must give the
    same bf16 output as the per-head mode, bit for bit: same per-head math, same chunk order."""
    inputs = _grouped_inputs(mesh_device, batch=32, S=2048, r_k=r_k, r_v=r_v, heads_per_group=4, phase2=phase2)
    per_head = _run_op(mesh_device, inputs, heads_per_core=1)
    shared = _run_op(mesh_device, inputs, heads_per_core=2)
    assert torch.equal(per_head, shared)
    assert _pcc(per_head, _reference(inputs)) >= 0.99
```

Run: `scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q -k group_shared` -> fails (`heads_per_core` unknown).

- [ ] **Step 2: Host op parameter.** Add `uint32_t heads_per_core` (0 = stock rule) to the operation attributes, the public op signature and the nanobind keyword (`heads_per_core: Optional[int] = None`). Validation: if given, `heads_per_group % heads_per_core == 0` and `num_kv_heads % heads_per_core == 0`; `heads_per_core > 1` requires `heads_per_group > 1` (otherwise it is meaningless; reject with a message naming both).

- [ ] **Step 3: Factory.** After the stock work-split computes `num_heads_per_core`, override it with the attribute when non-zero and, in Phase A, `TT_FATAL(num_cores_per_head == 1 || num_heads_per_core == 1)` (existing rule). Set `group_shared = heads_per_group > 1 && num_heads_per_core > 1`; add `compute_defines["PALU_GROUP_SHARED"] = "1"` and the same define for the reader and writer when set. CB sizing when `group_shared`:
  - `c_4`: `w_rot_tiles = H * w_rot_block` (whole blocks, no prefetch buffer).
  - `c_0`: `H * q_tiles` (the reader pushes Q for the H heads up front; the Phase-1 QK reads head `h`'s block at offset `h * q_tiles`; Phase 2 seeds head `h`'s q-state from it).
  - `c_12` (Phase 2 q-state): `(H + 1) * 2 * q_tiles` fp32 tiles -- each head's `[q~ | q~ T]` is a `2n` block, and `palu_step_q` pushes the new block before popping the old, so `H + 1` slots are needed and a block never straddles the end.
  - State CBs `c_25, c_26, c_27..c_31, c_21, c_22, c_23`: `H x` their current sizes. Statistics CBs are `H * PNHt`, out accumulators `H * out_tiles`.
  - Latent CBs `c_9`, `c_2`, cs `c_13`, AB `c_10`, K/V `c_1`: unchanged (transient per chunk / per head).
  - Extend the existing L1 sum / whole-block `TT_FATAL` to the new sizes; the message names `H`.

- [ ] **Step 4: Compute kernel loop swap** (`#ifdef PALU_GROUP_SHARED` around the new structure; the existing code stays the `#else` branch verbatim). Skeleton:

```cpp
#ifdef PALU_GROUP_SHARED
    // seed: for each head h, Phase 2 q-state (palu_init_q_state reads head h's Q block and head h's T tile:
    // t_tile_idx + h * w_rot_cb_tiles) and the k_chunk_start skips; ping-pong bookkeeping starts identical
    for (uint32_t h = 0; h < num_heads_per_core; ++h) { ... }
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        for (uint32_t h = 0; h < num_heads_per_core; ++h) {
            const bool last_head = h + 1 == num_heads_per_core;
            // Phase 2: advance head h's q-state (its block is at the FIFO front: every head steps once per chunk in order)
            palu_build_k_chunk<..., /*pop_latent=*/false>(Sk_chunk_t_dynamic, 0, h * w_rot_cb_tiles);  // pop after last head
            ... QK, softmax update, (V) palu_build_v_chunk<..., false>(Sk_chunk_t_dynamic, h * w_rot_cb_tiles), PV, accumulate ...
            if (last_head) { CircularBuffer(cb_lat_in).pop_front(Sk_chunk_t_dynamic * Rt); if (recon_v) CircularBuffer(cb_v_in).pop_front(Sk_chunk_t_dynamic * Rvt); /* Phase 1 cs pop as today */ }
        }
    }
    for (uint32_t h = 0; h < num_heads_per_core; ++h) { /* tree reduction + output for head h: the existing code block, unchanged */ }
    CircularBuffer(cb_w_rot).pop_front(num_heads_per_core * w_rot_cb_tiles);
#else
    ... existing per-head loop ...
#endif
```

  **State rotation rule (verify for every state CB, and write the verification in the report):** every head performs the identical sequence of `wait_front / pop_front / reserve_back / push_back` on each state CB per chunk, so a CB holding `H` blocks hands head `h`'s block back to head `h` at the FIFO front each chunk. The first-chunk special case (`k_chunk == k_chunk_start`: `cb_out_mm = cb_out_im` aliasing, no `prev` state) happens for every head on that chunk in the same way. Where an in-place helper (`mul_block_bcast_cols_inplace`, `add_block_inplace`, `mul_block_inplace`, `move_block`) reads and rewrites the *front* block, confirm it leaves the block at the *back* (pop + push); if any helper rewrites in place without rotating, replace that use with the explicit `move_block` to the back so the rotation invariant holds. The Phase-2 q-state follows the same rule (`palu_step_q` pushes then pops). Keep the `cb_cur_max/cb_prev_max/cb_cur_sum/cb_prev_sum` ping-pong variables per head (an array of `H` small structs or recompute per head), since the swap pattern differs between the first and later chunks.

- [ ] **Step 5: Helpers.** `palu_build_k_chunk` and `palu_build_v_chunk` gain a template `bool pop_latent` (default true, existing behaviour) and a `w_block_base` argument added to every weight-tile index (`w.wait_front(w_block_base + Rt*Wt)` becomes a wait on the whole resident set: in group mode wait once for `H * w_rot_cb_tiles` at kernel start instead, and pass the base for indexing only). `palu_init_q_state` takes the T tile index as today (caller adds the base).

- [ ] **Step 6: Reader.** Under `PALU_GROUP_SHARED`: read Q for heads `[base, base+H)` into `c_0` in order (one push of `H * q_tiles`, or `H` pushes of `q_tiles` -- match what the compute waits for), call `read_w_rot` for each head in order (H pushes of one block each into `c_4`), then `read_palu_chunks` **once** with the group's latent offsets (they are the same for all H heads because `lat_head = cur_head / heads_per_group` is constant on the core; assert that in the kernel with `ASSERT`). The `#else` branch keeps the existing per-head loop.

- [ ] **Step 7: Rebuild and test.** `nice -n 10 JOBS=8 scripts/build_ttnn.sh` (host changes). Then `scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q` -> the 31 existing tests plus the 6 new cases pass. If `torch.equal` fails with a tiny difference, investigate before relaxing: the expected cause is an in-place helper that does not rotate (Step 4), not rounding.

- [ ] **Step 8: Commit** (submodule first, then superproject with the pointer and the test file).

---

### Task 2: Phase B -- `H` heads per core with the sequence split over cores; host picks `H`

**Files:**
- Modify (submodule): `device/palu_sdpa_decode_program_factory.cpp` (restriction lift, item-based layout formulas, `num_reducer_cores = B * num_kv_heads / H`), `device/kernels/dataflow/palu_writer_decode.cpp` (per-head loop in the reduction path; semaphore nibble compared with `h + 1`; the sender increments once per head), `device/kernels/compute/palu_sdpa_flash_decode.cpp` (nothing new beyond Task 1 if the per-head reduction loop is already in place).
- Modify: `palu_tt/fused.py` (`group_heads_per_core`), `palu_tt/convert.py` (`PaluRuntime.group_heads_per_core: Optional[int] = None`), `palu_tt/attention.py` (choose `H` per layer, log once, pass `heads_per_core`), `palu_tt/model.py` (`build_model(..., group_heads=None)`), `palu_tt/bench.py`, `palu_tt/decode_check.py`, `palu_tt/eval_tasks.py` (`--group-heads {auto,1,2,4}`; row suffix `_fk2g` is the caller's choice of `--row`).
- Test: `tests/test_palu_sdpa_device.py`, `tests/test_attention_device.py`, `tests/test_fused.py`.

**Interfaces:**
- `fused.group_heads_per_core(r_k, r_v, chunk, phase2, heads_per_group, l1_budget_bytes=1_200_000) -> int` returns the largest `H` in `{heads_per_group, ..., 2, 1}` dividing `heads_per_group` such that `H * block_bytes(r_k, r_v, phase2) + latent_bytes(chunk, r_k, r_v) + state_bytes(H) + fixed <= l1_budget_bytes`, where `block_bytes = 2048 * (Rt*8 + Rvt*4 + phase2)`, `latent_bytes = 2 * chunk/32 * (Rt + Rvt) * 1088` (bfp8 tiles, double buffered), `state_bytes(H) = H * 2048 * (2*4 + 7*1) + (H+1) * 4096 * 8` (out accumulators, statistics, fp32 q-state), `fixed = 300_000`. Unit-test it in `tests/test_fused.py` with the three layer shapes from the spec (128/416 -> 4, 288/512 -> 2, 512/512 -> 1 at chunk 128, Phase 2).
- `PaluAttention` calls it per layer when `gs > 1` and `runtime.group_heads_per_core is None`; a non-None value is used as is (1 = stage 2b behaviour); the chosen `H` is stored as `self.group_heads` and logged once per distinct value.

- [ ] **Step 1: Failing tests.**

```python
# tests/test_palu_sdpa_device.py
@pytest.mark.parametrize("batch,H", [(8, 4), (16, 4), (32, 4), (8, 2)], ids=["b8_h4", "b16_h4", "b32_h4", "b8_h2"])
def test_group_shared_with_sequence_split_matches_per_head(mesh_device, batch, H):
    inputs = _grouped_inputs(mesh_device, batch=batch, S=4096, r_k=128, r_v=256, heads_per_group=4, phase2=True)
    per_head = _run_op(mesh_device, inputs, heads_per_core=1)
    shared = _run_op(mesh_device, inputs, heads_per_core=H)
    assert torch.equal(per_head, shared)

# tests/test_attention_device.py
@pytest.mark.parametrize("batch", [32, 8], ids=["b32", "b8"])
def test_glrd_group_shared_decode_matches_per_head(mesh_device, batch):
    """PaluAttention with automatic H (group-shared) against H = 1 (stage 2b kernel): equal outputs and caches."""
```

- [ ] **Step 2: Factory.** Replace the `(num_cores_per_head >= 1 && num_heads_per_core == 1) || (num_cores_per_head == 1 && num_heads_per_core >= 1)` `TT_FATAL` by: allowed when `group_shared`, otherwise the old rule. Recompute `num_cores_per_batch`, `num_reducer_cores`, `num_active_cores`, reducer/output core selection and tree tables with items = `B * num_kv_heads / H` (the existing formulas already divide by `num_heads_per_core`; check every use of `num_heads_per_core` and `num_cores_per_head` in the core-assignment loops and the per-core runtime args -- in particular `cur_head_group` indexing and the reducer spatial index comment at line ~361 "For multi KV head, would need: batch * num_kv_heads + head_within_batch").

- [ ] **Step 3: Writer.** In the reduction branch, replace `ASSERT(num_heads_per_core == 1)` by the per-head loop that already wraps it, and change the receive poll from `step_sem >= 1` to `step_sem >= head_index_on_core + 1` (the sender's semaphore increment per round is per head, so the nibble counts heads); confirm the 4-bit nibble cannot overflow (`H <= 4`, rounds <= 4). The sender path runs once per head as it does today.

- [ ] **Step 4: Host selection and flags.** Implement `group_heads_per_core` in `fused.py` with its unit test; thread `PaluRuntime.group_heads_per_core` -> `PaluAttention` -> op call; CLI `--group-heads` on bench/decode_check/eval_tasks/model.build_model (default `auto`; `1` reproduces stage 2b). The attention module logs `"palu G-LRD layer {i}: heads_per_core={H}"` once per distinct `H`.

- [ ] **Step 5: Rebuild, run** `scripts/run_device.sh pytest tests/test_palu_sdpa_device.py -q` and `scripts/run_device.sh pytest tests/test_attention_device.py -q -k glrd` and `nice -n 10 pytest tests/test_fused.py -q`.

- [ ] **Step 6: Commit** (submodule, then superproject).

---

### Task 3: Stage 2c report

**Files:** `palu_tt/report.py`, `tests/test_report.py`.

**Interfaces:** `report.ROWS2C` (`glrd_50`, `glrd_50_f256`, `glrd_70`, `glrd_70_f256` for the calibration table; `baseline`, `palu_full_50_fk2`, `glrd_50_fk2`, `glrd_50_fk2g`, `palu_full_70_fk2`, `glrd_70_fk2`, `glrd_70_fk2g` for the kernel tables), `report.OUT2C = Path("docs/results/stage2c.md")`, `--stage 2c`.

- [ ] **Step 1: Failing test** (fixture JSONs in `tmp_path` as in the stage 2b test): asserts the calibration table renders `acc (acc_norm)` cells and the paper-protocol average for `glrd_50` and `glrd_50_f256` side by side with a `delta` column; the kernel table renders decode `min_ms` at 1k/4k/8k for batch 32 and batch 8 (`bench_decode_<row>.json` and `bench_decode_<row>_b8.json`) and the attention slope `(att_8k - att_1k) / 7`; missing files render dashes.
- [ ] **Step 2: Implement** by reusing `_quality_row`, `_latency_rows`, `_task_display`, `_avg5` and the paper-protocol average (add `_avg5_paper(tasks)` = mean of `acc_norm` for openbookqa/arc_challenge and `acc` for the rest, shared with the stage 2b renderer if it is not already a helper).
- [ ] **Step 3:** `nice -n 10 pytest tests/test_report.py -q`; stage 1/2/2b renders unchanged apart from the stamp; commit.

---

### Task 4: Runs and write-up

**Files:** `docs/results/stage2c.md`, `docs/results/raw/*.json` (new rows), `README.md`, the spec's status line.

- [ ] **Step 1: Calibration rows** (after `calib_f256.sh` reports FINISHED and `compressed/glrd_0.5_f256`, `compressed/glrd_0.7_f256` exist): one gated background loop, sequential, skip-if-JSON-exists, logs under `docs/results/raw/<name>.log`:

```bash
for r in glrd_50_f256:compressed/glrd_0.5_f256 glrd_70_f256:compressed/glrd_0.7_f256; do IFS=: read row dir <<< "$r"
  python -m palu_tt.eval_ppl   --row $row --kind palu --palu-dir $dir
  python -m palu_tt.eval_tasks --row $row --kind palu --palu-dir $dir
  python -m palu_tt.bench      --row ${row}_fk2 --kind palu --palu-dir $dir --what decode --fused p2 --group-heads 1
  python -m palu_tt.decode_check --row ${row}_fk2 --palu-dir $dir --fused p2 --group-heads 1
done
```

- [ ] **Step 2: Kernel rows** (device; can run before Step 1's inputs exist):

```bash
for r in glrd_50:compressed/glrd_0.5 glrd_70:compressed/glrd_0.7; do IFS=: read row dir <<< "$r"
  python -m palu_tt.bench --row ${row}_fk2g --kind palu --palu-dir $dir --what attention --fused p2 --group-heads auto
  python -m palu_tt.bench --row ${row}_fk2g --kind palu --palu-dir $dir --what decode    --fused p2 --group-heads auto
  python -m palu_tt.bench --row ${row}_fk2g --kind palu --palu-dir $dir --what decode    --fused p2 --group-heads auto --batch 8
  python -m palu_tt.bench --row ${row}_fk2  --kind palu --palu-dir $dir --what decode    --fused p2 --group-heads 1    --batch 8
  python -m palu_tt.decode_check --row ${row}_fk2g --palu-dir $dir --fused p2 --group-heads auto
  python -m palu_tt.bench --row ${row}_fk2g --kind palu --palu-dir $dir --what maxctx --fused p2 --group-heads auto --batch 8
done
python -m palu_tt.bench --row baseline --kind stock --what decode --batch 8   # if bench_decode_baseline_b8.json is missing
```

  Record the per-layer `H` the attention module logged (how many layers at 4 / 2 / 1) for each dump.

- [ ] **Step 3: Render and write up** (`python -m palu_tt.report --stage 2c`; prose below the marker): (1) calibration sensitivity -- did the paper-protocol gap move, per task and on average, against the bootstrap noise; perplexity and rank means; conclusion for the paper-gap hypothesis; (2) kernel -- slope and decode step before/after at batch 32 and 8 against the spec's targets (slope <= 0.35, 4k step <= 80 ms), the `H` distribution, and where the remaining time goes; (3) fidelity of `_fk2g`; (4) capacity unchanged or not; (5) caveats and what was not measured. README: `--group-heads`, `run_calibration.sh` third argument, `report --stage 2c`. Spec status line -> "executed".

- [ ] **Step 4: Commit** `docs/results/stage2c.md docs/results/raw/*.json README.md docs/superpowers/specs/2026-09-12-palu-tt-stage2c-design.md`.

---

## After the last task

Whole-branch review (base `fa1b17d`), then merge to `main` locally as before; the controller publishes the stage 2c results page.
