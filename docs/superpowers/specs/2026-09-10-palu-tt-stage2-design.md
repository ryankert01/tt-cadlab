# Palu on Tenstorrent — Stage 2: Fused K-Reconstruction Decode Kernel

Date: 2026-09-10
Status: design approved in conversation, pending written review
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
3. Perplexity of the fused rows matches the stage-1 device numbers within ±0.02.

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
                 scale=None, program_config=None, compute_kernel_config=None,
                 memory_config=None) -> [1, B, n_q_pad, head_dim_v_pad]
```

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `q` | `[1, B, n_q_pad, 128]` | bf16 | already RoPE'd, as for the stock op |
| `k_latent` | `[B, n_kv, max_seq_len, r_k]` | bf16 | the stage-1 K latent cache, unchanged |
| `w_rot` | `[1, n_kv, r_k, 256]` | bf16 | `[u_k \| u_k·T]` per KV head; `T` is the 128×128 interleaved-RoPE transformation matrix the stock `rotary_embedding_llama` op uses |
| `cs_tab` | `[1, 1, max_seq_len, 256]` | bf16 | `[cos \| sin]` per position, Meta-interleaved layout (each frequency in two adjacent columns) |
| `v_latent` | `[B, n_kv, max_seq_len, r_v]` | bf16 | the stage-1 V latent cache, as consumed by `flash_multi_latent_attention_decode` today |
| `cur_pos_tensor` | `[B]` | int32 | current position per batch row |

Envelope (asserted on the host, `TT_FATAL` on the device op):

- head_dim = 128, `r_k ∈ {32, 64, 96}`, `r_v ≤ 128`, batch ≤ 32
- causal, non-paged, no attention mask tensor, no sink, no sliding window, no
  block padding, `cur_pos_tensor` required
- `compute_kernel_config` with `fp32_dest_acc_en=True` (what tt_transformers uses
  for SDPA decode at HiFi4); chunk size follows the stock dynamic rule (4 tiles with
  fp32 dest)

Semantics: for every valid position `p ≤ cur_pos`, `K[p] = RoPE_p(k_latent[p] · u_k)`
and the output is `softmax(q·Kᵀ·scale) · v_latent`, exactly what stage 1 computes with
matmul + `rotary_embedding_llama` + `flash_multi_latent_attention_decode`.

## 4. Kernel internals

Per (batch, KV head) work item the compute kernel loops over K chunks of `c`
positions (c = 128 at fp32 dest). Per chunk:

1. `AB = lat_chunk @ w_rot` with `matmul_blocks` (inner dim `r_k/32` tiles, N = 8
   tiles). `AB` is `[c, 256] = [A | Bm]` where `A = lat·u_k` and `Bm = lat·u_k·T`.
2. `AB *= cs_chunk` in place (`mul_block_inplace`), giving `[A⊙cos | Bm⊙sin]`.
3. For row tile `i` and column tile `j < 4`: `K_rot[i,j] = AB[i,j] + AB[i,j+4]`,
   packed into the K slot (`c_1`) with `add_tiles` from one CB at two indices.
4. Stock code, unchanged: QK matmul, on-core causal mask on the last chunk, online
   softmax, QK·V with the V latent, tree reduction across cores when a head spans
   several cores.

This reproduces the stock RoPE identity `out = x⊙cos + (x·T)⊙sin` with `x = lat·u_k`,
using `(lat·u_k)·T = lat·(u_k·T)`.

**Circular buffers.** The stock op uses all 32 indices; in the envelope above five
are idle and four are reassigned. Sizes are at c = 128 in bf16 tiles (2 KB each):

| Slot | Stock use | Palu use | Tiles |
|---|---|---|---|
| `c_1` | K chunk | receives `K_rot` from step 3 | unchanged |
| `c_3` | attention mask | latent chunk `[c, r_k]` | `4 × r_k/32`, double buffered |
| `c_4` | attention sink | `w_rot` for the current KV head, resident | `r_k/32 × 8` |
| `c_9` | page table | `AB` | `4 × 8` |
| `c_13` | sliding window | `cs_chunk` (Phase 1: bf16; Phase 2: fp32) | `4 × 8`, double buffered |
| `c_14` | block padding | Phase 2 rotation constants `[cosΔ \| sinΔ]` | 8 |

Extra L1 at r_k = 96 is about 240 KB on top of the stock footprint (Wormhole L1 is
1.5 MB); the program factory fails loudly on overflow, so this is verified at first
run. The mask CB index inside the copied compute kernel becomes its own compile-time
define so the last-chunk masking code never reads `c_3`.

**Reader.** Per KV head: read `w_rot` once into `c_4`. Per chunk: read the latent
chunk `[c, r_k]` into `c_3` and (Phase 1) the `cs` chunk `[c, 256]` into `c_13`. The V
path, writer, and reduction are untouched. DRAM traffic per chunk goes from
`c × 128` (stock K) to `c × (r_k + 256)` in Phase 1 and `c × r_k` in Phase 2, so
Phase 1 only wins at r_k = 32 and Phase 2 is what the success bar is judged on.

**Compute.** The added work per chunk is one matmul with inner dim ≤ 3 tiles and
32 output tiles, 32 multiplies, and 16 adds. With two work items per core at batch 32
this hides behind DRAM time.

## 5. Phase 2: on-core cos/sin recurrence

Phase 1 streams `cs_tab` per chunk. Phase 2 reads only the first chunk's rows and
advances on core. For column `j` the chunk-to-chunk angle step `Δ_j = c·θ_j` is the
same for every row, so the constants are eight tiles `[cosΔ | sinΔ]`, resident in
`c_14`, built on the host from the same rope generator. Per chunk:

```
cos' = cos·cosΔ − sin·sinΔ
sin' = sin·cosΔ + cos·sinΔ
```

Four multiplies, one subtract, one add per tile pair, computed in an fp32 dest and
packed back into `c_13` as fp32 tiles. At 8k context that is 64 steps; the fp32 drift
(~64 × 2⁻²⁴) vanishes under the bf16 rounding that already applies when `cs`
multiplies `AB`. bf16 storage would not be accurate enough, which is why `c_13` is fp32
in Phase 2 and why the envelope requires `fp32_dest_acc_en`. The `cs` chunk read in
step 2 of §4 then uses a bf16→fp32 data-format reconfigure, exactly as the stock
kernel does for its fp32 statistics CBs.

The partial last chunk needs no special case: the recurrence advances by a full chunk
regardless of how many rows are valid, and the causal mask zeroes the invalid rows.

Phase 2 is a compile-time kernel define (`PALU_CS_RECURRENCE`) selected from the
program config, so Phase 1 remains the reference path and both are covered by the same
tests. The reader in Phase 2 skips the `cs` read after the first chunk.

## 6. Host integration

- `palu_tt/convert.py`: emit `w_rot` per layer under
  `layers.{i}.attention.palu.w_rot` (built from `u_k` and the stock transformation
  matrix), only when `r_k < head_dim`. `CONVERT_VERSION` becomes 3.
- `palu_tt/convert.py` `PaluRuntime` gains `use_fused_k: bool = False` and
  `cs_recurrence: bool = False` (Phase 2 switch).
- `palu_tt/attention.py`: `forward_decode` takes a fused branch when `recon_k` and
  `runtime.use_fused_k` are both set. The branch skips the reconstruction matmul, the
  RoPE call, and the cache slices, and calls `palu_sdpa_decode` on `keys`, `values`,
  the resident `w_rot`, and `cs_tab`. `cache_tag` gains an `_fk` suffix so the
  weight cache never serves stage-1 dumps (see the stage-1 cache lessons). `cs_tab`
  is built once per model from the existing rope generator and lives on device next
  to `cos_prefill`. Prefill, the V path, and the naive path are unchanged; the naive
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
   × batch `{1, 32}` × Phase `{1, 2}`. Pass: PCC ≥ 0.99 (the bar the stock
   flash-decode tests use). Batch-1 cases are the default smoke set so a run holds the
   board for seconds.
2. **Attention level, device** (`tests/test_attention_device.py`): one decode step on
   the same `PaluAttention` with the naive and the fused path; output PCC ≥ 0.99 and
   byte-identical caches.
3. **Model level**: `tests/test_ppl_harness.py` gains a fused row; `palu_tt.bench`
   and `palu_tt.eval_ppl` produce the fused rows for the results table.
4. **CPU**: `tests/test_convert.py` covers `w_rot` construction (`w_rot[:, :128] == u_k`,
   `w_rot[:, 128:] == u_k·T`, and that `(lat·u_k)·T == lat·w_rot[:, 128:]`) and the
   version bump.

## 8. Evaluation

Re-run the stage-1 harness rows for `palu_full_0.5`, `palu_full_0.7`, and the synthetic
`svd_r64`/`svd_r96` rows with `use_fused_k` on, Phase 1 and Phase 2, at contexts 1k/4k/8k
and batch 32; device perplexity for the two calibrated rows; KV bytes and max context
are unchanged from stage 1 and are not re-measured. Results go to `docs/results/stage2.md`
with raw JSON in `docs/results/raw/`, rendered by the existing `report.py`.

## 9. Risks

1. **CB reassignment breaks a stock path we still use.** Mitigated by a separate
   mask-CB define and host-side envelope asserts.
2. **fp32 dest halves the chunk to 128 positions**, doubling per-chunk overhead. This
   is stock HiFi4 behaviour; if the fused path is slower than stock at 1k the first
   knob is HiFi2 with bf16 dest for Phase 1, measured against perplexity.
3. **Phase 1 slower than stock at r_k = 96.** Expected; both phases are reported and
   the success bar is judged on Phase 2.
4. **Program-factory complexity.** The copy strips paged, sink, sliding-window, block
   padding, and mcast branches up front so the diff stays reviewable.
5. **Fork logistics.** No push credentials on the box; the submodule branch is local
   until the fork exists.
6. **Shared server.** All device work through the `device_free.sh` gate; smoke tests
   default to batch 1.
