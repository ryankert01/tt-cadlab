# Palu on Tenstorrent — Stage 2: Fused K-Reconstruction Decode Kernel

Date: 2026-09-10
Status: **implemented and measured 2026-09-10** -- op, host integration, tests and the evaluation
matrix are done; results and the success-bar verdict are in `docs/results/stage2.md`.
Approved 2026-09-10 ("Lgtm"); amended at planning time -- §3 (`rot_step`/`trans_mat`), §4 CB table,
§5 rewritten (query-side rotation), §6-§9. Deviations found while implementing are recorded below
the marker in `docs/results/stage2.md`.
Depends on: stage 1 (`docs/superpowers/specs/2026-09-09-palu-tt-stage1-design.md`, merged at 44f24a9)

## 1. Goal and success bar

Stage 1 runs Palu-compressed Llama-3.1-8B-Instruct on the n300 but reconstructs K
naively each decode step: a matmul over the whole latent cache, a RoPE pass over the
result, then the stock flash-decode op. That path decodes 2.5–9× slower than stock
(88–100 ms vs 38 ms at 1k; 263–283 ms vs 55 ms at 4k). Stage 2 moves reconstruction
and rotation inside the attention kernel so the full K is never materialised.

**Success bar ("beat stock decode"):**

1. The fused op is numerically equivalent to the stage-1 naive path (PCC ≥ 0.99 on
   attention output at op level and attention level; identical cache contents).
2. The calibrated rows (`palu_full_0.5`, `palu_full_0.7`) decode **no slower than
   stock** at 1k and 4k context, batch 32, and **run at 8k** where stock is out of
   memory.
3. Decode fidelity: per-step logits PCC ≥ 0.99 (min over 64 teacher-forced steps) and
   argmax agreement ≥ 0.95 between the fused and the naive Palu model.

All numbers come from the stage-1 harness (`palu_tt.bench`, `palu_tt.eval_ppl`) with
new rows, on the same device, same batch, same contexts.

## 2. Repository and build loop

- The kernel lives in the tt-metal submodule on a local branch `palu-sdpa` cut from
  the pinned `v0.78.0`. The superproject records that branch's commit hash. Nothing
  under `third_party/` is committed to this repo's history except the submodule
  pointer.
- Publishing: the user forks `tenstorrent/tt-metal` to `ryankert01/tt-metal` and
  provides push credentials; `.gitmodules` then switches to the fork URL. Until then
  the repo is not cloneable by others, and the README says so.
- Kernels are JIT-compiled from source at first op invocation, so kernel edits need
  no host rebuild. Host-side edits (program factory, op registration, nanobind) need
  a niced `ninja` in `third_party/tt-metal/build` (≤ 12 threads, per the shared-server
  rules in the stage-1 spec §3).
- Every device run still passes through `scripts/device_free.sh`.

## 3. Op interface

A new op, copied from `sdpa_decode` (flash decode) and registered under
`ttnn.transformer.palu_sdpa_decode`:

```
palu_sdpa_decode(q, k_latent, w_rot, cs_tab, v_latent, head_dim_v, cur_pos_tensor,
                 rot_step=None, trans_mat=None, scale=None, program_config=None,
                 compute_kernel_config=None, memory_config=None) -> [1, B, n_q, head_dim_v]
```

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `q` | `[1, B, n_q_pad, 128]` | bf16 | already RoPE'd, as for the stock op |
| `k_latent` | `[B, n_kv, max_seq_len, r_k]` | bf16 | the stage-1 K latent cache, unchanged |
| `w_rot` | `[1, n_kv, r_k, 256]` | bf16 | `[u_k \| u_k·T]` per KV head; `T` is the 128×128 interleaved-RoPE transformation matrix the stock `rotary_embedding_llama` op uses |
| `cs_tab` | `[1, 1, max_seq_len, 256]` | bf16 | `[cos \| sin]` per position, Meta-interleaved layout (each frequency in two adjacent columns); Phase 2 reads only rows 0..c-1 |
| `v_latent` | `[B, n_kv, max_seq_len, r_v]` | bf16 | the stage-1 V latent cache, as consumed by `flash_multi_latent_attention_decode` today |
| `cur_pos_tensor` | `[B]` | int32 | current position per batch row |
| `rot_step` | `[1, 1, 32, 256]` | fp32 | optional; `[cos(c·f) \| sin(c·f)]` on every row, c = the op's K chunk. Passing it selects Phase 2. |
| `trans_mat` | `[1, 1, 32, 32]` | bf16 | required with `rot_step`; the pair-rotation tile the RoPE op uses |

Envelope (asserted on the host, `TT_FATAL` on the device op):

- head_dim = 128, `r_k ∈ {32, 64, 96}`, `r_v ≤ 128`, batch ≤ 32
- causal, non-paged, no attention mask tensor, no sink, no sliding window, no
  block padding, `cur_pos_tensor` required
- `compute_kernel_config` with `fp32_dest_acc_en=True` for Phase 2 (the HiFi4 preset has
  it); the K chunk follows the stock unpaged rule (largest power of two dividing the cache
  length, capped at 512) unless the program config overrides it; the 16-row Q tile
  optimisation is off (all Q-side tiles are full tiles)

Semantics: for every valid position `p ≤ cur_pos`, `K[p] = RoPE_p(k_latent[p] · u_k)`
and the output is `softmax(q·Kᵀ·scale) · v_latent`, exactly what stage 1 computes with
matmul + `rotary_embedding_llama` + `flash_multi_latent_attention_decode`.

## 4. Kernel internals

Per (batch, KV head) work item the compute kernel loops over K chunks of `c`
positions (c = 128 at fp32 dest). Per chunk:

1. Row tile by row tile: `AB_i = lat_i @ w_rot` (inner dim `r_k/32` tiles, 8 output tiles)
   into a small ping-pong CB; `AB_i = [A_i | Bm_i]` where `A = lat·u_k` and `Bm = lat·u_k·T`.
2. For row tile `i` and column tile `j < 4`, in one dest acquisition: `A[i,j]⊙cos[i,j]` and
   `Bm[i,j]⊙sin[i,j]` with two FPU `mul_tiles`, summed with the SFPU `add_binary_tile`, and
   packed into the K slot (`c_1`) at index `j·Sk_chunk_t + i` -- the column-major tile order the
   stock QK matmul reads K in. `w_rot` is the matmul's in1 and must survive the chunk, so the
   matmul is a copy of `matmul_blocks` without its final in1 pop.
3. (merged into 2)
4. Stock code, unchanged: QK matmul, on-core causal mask on the last chunk, online
   softmax, QK·V with the V latent, tree reduction across cores when a head spans
   several cores.

This reproduces the stock RoPE identity `out = x⊙cos + (x·T)⊙sin` with `x = lat·u_k`,
using `(lat·u_k)·T = lat·(u_k·T)`.

**Circular buffers.** The stock op uses all 32 indices. In the envelope above five are
idle -- but `c_3` is *not* one of them: the writer generates the causal mask into `c_3` for
the last chunk, and that stays. The idle five are the sink, page-table, row-major-Q,
sliding-window and block-padding slots. Sizes are at c = 128 in tiles:

| Slot | Stock use | Palu use | Tiles (c = 512, r_k = 96) |
|---|---|---|---|
| `c_1` | K chunk | receives `K_rot` (bf16) | unchanged |
| `c_3` | causal mask | causal mask (unchanged) | unchanged |
| `c_4` | attention sink | `[u_k \| u_k·T]` for the current KV head (+ the T tile in Phase 2), resident (bf16) | `3 × 8 + 1` |
| `c_9` | page table | latent chunk `[c, r_k]` (cache dtype, bfp8 by default), double buffered | `2 × 16 × 3` |
| `c_10` | row-major Q (`tilize_q`) | one `AB` row block, ping-pong (bf16) | `2 × 8` |
| `c_12` | zero scalar (mask fusion only) | Phase 2: rotated-query state `[q̃ \| q̃·T \| q̃']` (fp32) | `3 × PNHt × 4` |
| `c_13` | sliding window | Phase 1: streamed `[cos \| sin]` chunk ×2; Phase 2: chunk-relative table rows 0..c-1, resident (bf16) | `16 × 8` (P2) |
| `c_14` | block padding | Phase 2: `rot_step` (fp32) | 8 |

Using `c_10` means Q must arrive in TILE layout (it does from `rotary_embedding_llama`);
the host asserts it. `c_12` is free because the zero tile only feeds the dynamic-chunk mask
fusion, which the unpaged path never takes; the copied writer no longer produces it.

**Annotations (final review).** `c_4` is sized `2 × (Rt·Wt + 1)` in Phase 2 (`w_rot_block = Rt·Wt +
1`, doubled so the reader's per-head push always lands back on a block boundary -- see the CB-size
`TT_FATAL` in `palu_sdpa_decode_program_factory.cpp`) rather than `Rt·Wt + 1`; Phase 1 drops the `+1`
T tile since it never rotates the query. `c_12` is `4n` tiles (`n = PNHt·DHt`), not `2n`: the CB's
push/pop granule is the `2n`-tile `[q~ | q~ T]` block, and `4n` keeps both the block being read and
the block being written contiguous so an operand index never wraps the CB -- the same "whole number
of reader push blocks" constraint the `TT_FATAL` above checks for `c_4`/`c_9`/`c_13` (reader-pushed
CBs); `c_12` is compute-pushed but sized by the identical rule. Compute-operand reads (the CBs above
that only the compute kernel indexes into, `c_1`/`c_9`/`c_13`'s resident form) do not need to avoid
wrapping the way a reader's linear push does; only the push/pop granule matters.

Extra L1 at c = 512, r_k = 96 is about 500 KB in Phase 2 on top of the stock footprint
(Wormhole L1 is 1.5 MB); Phase 1's streamed table needs 512 KB by itself at c = 512, so
Phase 1 runs at c = 128. The program factory fails loudly on overflow, so the fit is verified
at first run; if Phase 2 does not fit at 8k, the fused op runs at c = 128 (`fused_k_chunk`)
and the results include a stock row at c = 128 for comparison. The last-chunk masking code keeps reading the mask from `c_3` exactly as in stock.
**Correction (final review):** measured Phase 2 L1 at c = 512, r_k = 96 is about **990 KB total**
new/changed CBs, not the ~500 KB estimated here -- roughly 700 KB of new Palu CBs plus ~120 KB from
`c_1` widening (bfp8 K in the stock op vs bf16 `K_rot` here). The estimate above was a lower bound on
the new CBs alone and did not include the `c_1` dtype change; see `docs/results/stage2.md` §3.

**Reader.** Per KV head: read `w_rot` once into `c_4`. Per chunk: read the latent
chunk `[c, r_k]` into `c_9` and (Phase 1) the `cs` chunk `[c, 256]` into `c_13`. Phase 2
reads the chunk-relative table (`cs_tab` rows 0..c-1) and `rot_step` once per core. The V
path, writer, and reduction are untouched. K-side DRAM traffic per position goes from
128 elements (stock K, bfp8: 136 B) to `r_k + 256` in Phase 1 (worse than stock) and
`r_k` in Phase 2 (bfp8: 34/68/102 B at r_k 32/64/96), so Phase 1 is only a checkpoint and
Phase 2 is what the success bar is judged on.

**Compute.** Per K row tile: one matmul with inner dim ≤ 3 tiles and 8 output tiles, then 8
multiplies and 4 SFPU adds. Phase 2 adds, per chunk, 4 single-tile matmuls, 8 broadcast
multiplies and 4 subtractions on the query. Whether this hides behind DRAM time is what the
attention-only benchmark measures.

## 5. Phase 2: chunk-relative rotation, query advanced on core

Phase 1 streams `[cos | sin]` rows per chunk: 256 values per position, twice the size of
the K it replaces, so it cannot win. Phase 2 removes that traffic with the relative-position
property of RoPE. Two-dimensional rotations commute, so for a chunk starting at `s0` and a
position `s0 + t` inside it

```
q R_p · (k R_{s0+t})ᵀ = (q R_{p−s0}) · (k R_t)ᵀ
```

The K side therefore uses the *chunk-relative* angles `t = 0..c−1`: the first `c` rows of the
ordinary table, read once per core and resident in L1 (`c_13`). The chunk base moves to the
query: `q̃_k = q R_{−k·c}`, advanced once per chunk by the constant rotation `R_{−c}`:

```
q̃ ← q̃ ⊙ cos(c·f) − (q̃ · T) ⊙ sin(c·f)
```

`cos(c·f)` and `sin(c·f)` are one table row (`rot_step`, position `c`), applied as a row
broadcast; `q̃ · T` is four single-tile matmuls with the same 32×32 pair-rotation tile the
RoPE op uses (`trans_mat`). The state is `PNHt × 4` tiles kept in fp32 (`c_12`), so 16 steps
at 8k drift by well under a bf16 ulp. The query already carries its own position `p` from
the stock decode RoPE, so the first state is a copy of Q rotated `k_chunk_start` times
(cores that own a later slice of the context start at their first chunk). The QK matmul reads
its in0 from this state instead of `c_0`; nothing downstream changes.

**Caveat (final review):** "fp32" here describes the CB's tile format, not the multiply's actual
precision. Wormhole's FPU unpacks fp32 operands into 19-bit Src registers -- about 11 mantissa bits
per multiply, close to bf16 -- so each `palu_step_q` update is not full fp32 accuracy; keeping the
*state* in fp32 tiles is what keeps rounding error from compounding additively across the 16 steps at
8k, not per-step precision. The same caveat applies to the query step's matmuls in §4 (`palu_step_q`
in `palu_compute_common.hpp`, which now carries this note) and is why the measured drift, not the
tile dtype, is the thing to trust (Task 4; also `docs/results/stage2.md` §3).

Cost per chunk on the query side: 4 tiny matmuls, 8 broadcast multiplies, 4 subtractions.
The chunk length `c` is static for an unpaged cache (`fused.chunk_size(max_seq_len)`, or the
program-config override), so the host builds `rot_step` for exactly that `c`.

Phase 2 is a compile-time kernel define (`PALU_PHASE2`), selected by passing `rot_step` (and
`trans_mat`) to the op, so Phase 1 remains the reference path and both are covered by the
same tests. (The approved design described a K-side cos/sin recurrence instead; at the op's
default chunk that state is 512 KB of fp32 tiles and ~100 SFPU ops per chunk, so it was
replaced by the query-side formulation at planning time.)

## 6. Host integration

- `palu_tt/convert.py`: emit `w_rot` per layer under
  `layers.{i}.attention.palu.w_rot` (built from `u_k` and the stock transformation
  matrix) for every rank, like `u_k` itself; the attention uploads it only on the fused
  path. `CONVERT_VERSION` becomes 3. The builders (`w_rot`, `cs_tab`, `rot_step`, `trans_mat`, and an fp32
  torch reference of the op) live in a new `palu_tt/fused.py`.
- `palu_tt/convert.py` `PaluRuntime` gains `use_fused_k: bool = False`,
  `cs_recurrence: bool = False` (Phase 2 switch) and `fused_k_chunk: Optional[int] = None`
  (K-chunk override for the fused op).
- `palu_tt/attention.py`: `forward_decode` takes a fused branch when `recon_k` and
  `runtime.use_fused_k` are both set. The branch skips the reconstruction matmul, the
  RoPE call, and the cache slices, and calls `palu_sdpa_decode` on `keys`, `values`,
  the resident `w_rot`, `cs_tab` and, in Phase 2, `rot_step` and `trans_mat`. No cache-tag suffix is needed: `w_rot` is a new
  tensor name, every other cached tensor has identical content on both paths, and the
  `CONVERT_VERSION` bump already separates the dumps from stage 1. `cs_tab` (and
  `rot_step`/`trans_mat` in Phase 2) are built once per model from the existing rope generator and
  shared by all layers through `PaluConfig.device_tables`; the fused path does not build
  `cos_prefill`/`sin_prefill` or `u_k_decode`. Prefill, the V path, and the naive path are unchanged; the naive
  path remains the correctness reference.
- `recon_len` is irrelevant on the fused path: the op reads `cur_pos_tensor` and
  stops at the last valid chunk.
- `palu_tt/model.py`, `bench.py`, `report.py`: thread the two runtime flags through
  and add "fused K (P1)" and "fused K (P2)" decode rows/columns beside the naive
  decode column.

## 7. Tests

1. **Op level, device** (`tests/test_palu_sdpa_device.py`): random latents, `w_rot`,
   `cs_tab`, `q`; fused op vs an fp32 torch reference that reconstructs K, applies
   RoPE, and runs attention. Grid: `r_k ∈ {32, 64, 96}` × context `{1024, 4096, 8192}`
   × batch `{1, 32}` × Phase `{1, 2}`. Pass: PCC ≥ 0.99 against the reference and ≥ 0.995
   against the stock MLA op fed the host-rotated K. Batch-1 cases are the default smoke set
   so a run holds the board for seconds.
   **Deviation (final review):** the implemented suite is 20 tests, not this full grid --
   `r_k ∈ {32, 64, 96}` is only exercised at `S = 1024` (both phases, both batches); the long
   contexts (`4096`, `8192`) run Phase 2 only, at `r_k = 64` (batch 1) and, at `8192`, `r_k = 32`
   (batch 32). `r_k = 96` at `S ∈ {4096, 8192}` is not covered at the op level (see
   `docs/results/stage2.md` §4 for how it is covered empirically instead). See §7.3 below for the
   corresponding correction to the decode-fidelity step count.
2. **Attention level, device** (`tests/test_attention_device.py`): one decode step on
   the same `PaluAttention` with the naive and the fused path; output PCC ≥ 0.99 and
   byte-identical caches.
3. **Model level**: perplexity in this study is measured through prefill windows and cannot
   see the decode kernel, so the model-level check is `palu_tt.decode_check`: naive vs fused
   Palu model, same 512-token prompt, 64 teacher-forced decode steps, per-step logits PCC
   and argmax agreement. `palu_tt.bench` produces the fused timing rows.
   **Correction (final review):** the shared prefill logit that seeds the 64 steps is
   bit-identical on both paths (the fused kernel only replaces decode), so it was inflating
   agreement by 1/64; PCC and argmax are now computed over the 63 *decode* steps only, and the
   JSON's `"steps"` field records that count.
4. **CPU**: `tests/test_convert.py` covers `w_rot` construction (`w_rot[:, :128] == u_k`,
   `w_rot[:, 128:] == u_k·T`, and that `(lat·u_k)·T == lat·w_rot[:, 128:]`) and the
   version bump.

## 8. Evaluation

Re-run the stage-1 harness rows for `palu_full_0.5`, `palu_full_0.7`, and the synthetic
`svd_r64`/`svd_r96` rows with `use_fused_k` on, Phase 1 and Phase 2, at contexts 1k/4k/8k
and batch 32; `decode_check` for the calibrated rows and `svd_r64`; perplexity, prefill,
KV bytes and max context are unchanged from stage 1 by construction and are not re-measured. Results go to `docs/results/stage2.md`
with raw JSON in `docs/results/raw/`, rendered by the existing `report.py`.
**Correction (final review):** perplexity and prefill hold *by construction* (the fused kernel only
changes decode), but KV bytes and max context turned out not to always be stage-1's numbers — two
rows were re-probed at 16k and the reconstruction-scratch ceiling stage 1 predicted was found to have
moved, not disappeared (`docs/results/stage2.md` §4). The rule actually implemented is: KV bytes and
max context are inherited from the base row unless the row has its own `bench_maxctx_*` probe, in
which case that measurement wins.

## 9. Risks

1. **CB reassignment breaks a stock path we still use.** Mitigated by a separate
   mask-CB define and host-side envelope asserts.
2. **L1 at the default chunk (512).** Phase 2 adds ~500 KB of CBs at r_k = 96; if it does
   not fit at 8k the fused op runs at c = 128 and the stock sensitivity row at c = 128 is
   measured too, so the comparison stays honest.
3. **Phase 1 slower than stock at r_k = 96.** Expected; both phases are reported and
   the success bar is judged on Phase 2.
4. **Program-factory complexity.** The copy is kept *verbatim* rather than stripped: the
   unused feature branches (paged, sink, mask tensor, sliding window, block padding, mcast)
   stay compiled out behind their existing flags, the envelope is enforced by host asserts,
   and `diff -r sdpa_decode palu_sdpa_decode` then shows only the fusion. (Decision taken at
   planning time; the approved design said "strip up front".)
5. **Fork logistics.** No push credentials on the box; the submodule branch is local
   until the fork exists.
6. **Shared server.** All device work through the `device_free.sh` gate; smoke tests
   default to batch 1.
