# Stage 2e: spend the idle hardware — a DST-resident K rebuild, a multicast angle table, and bfp4 latents

Status: proposed (2026-09-14). Follows stage 2d (`docs/results/stage2d.md`) and the kernel guide
(`docs/kernel_guide.md`), which together established *where* the fused Palu op's time goes.

## 0. What stage 2d and the resource inventory say

The fused op (`ttnn.transformer.palu_sdpa_decode`, Phase 2) is 1.36x the stock flash-decode op per
layer at 4k and batch 32 (1.55x at 1k, 1.33x at 8k) while reading 0.63x the bytes. Fitting per-layer
time as `fixed + chunks x per_chunk` over 1k / 4k / 8k:

| per layer, batch 32 | fixed | per chunk | per head-item per chunk |
|---|---|---|---|
| stock (`r_v` 128) | 47.7 us | 88.4 us | 44.2 us |
| fused `r_k` 32 / `r_v` 128 | 118.1 us | 114.3 us | 57.2 us |
| fused `r_k` 64 / `r_v` 128 | 129.5 us | 123.1 us | 61.6 us |

Two conclusions drive this stage:

1. **The per-chunk excess is choreography, not arithmetic.** Per K row tile the rebuild costs about
   810 ns over stock, of which the rank-dependent `AB` matmul is about 275 ns (the marginal cost of
   doubling `r_k`; it also matches the FPU cycle count for 128 tile products at HiFi2, 4.1 us per
   chunk against 4.4 measured). The other 535 ns is six DST round trips, three engine inits, four
   format reconfigurations, eight FPU multiplies and four SFPU full-tile adds per row tile, most of
   which move data through L1 that could stay in DST. The FPU is idle about 75 % of the time and DRAM
   about 60 % in the fused op; the bottleneck is the serial instruction stream on the three compute
   threads.
2. **The fixed excess is the size of the angle table read.** Phase 2 reads 256 KB of chunk-relative
   `[cos | sin]` rows plus 32 KB of `rot_step` into every core before its first chunk. Over 64 cores
   that is 18.4 MB of DRAM traffic per layer per chip, 78 us at the measured 237 GB/s, on the
   critical path because compute cannot start until the table, the weight block and the first latent
   chunk have all landed. The fit's unexplained fixed excess is 60 to 80 us per layer. This is a
   hypothesis from arithmetic; the stage tests it directly.

Two hardware features the kernel does not use today make both fixable without new arithmetic: the
FPU's *dest-reuse* element-wise ops (one operand taken from the tile already in DST) and the packer's
*L1 accumulate* (a pack that adds onto the tile already in L1); and one it uses only in prefill, NoC
multicast, delivers one DRAM read to a whole column of cores. Separately, the unpacker converts
`bfp4_b` for free, so latent bytes can shrink through format instead of rank.

## 1. Goal and success criteria

Close the fused op's gap to stock at batch 32 using only idle resources, and attribute each lever's
effect with the stage 2d fit. Four levers, three of them kernel changes to the same op behind
switches, one a measurement:

- **L1, DST-resident rotation.** Rebuild each K row tile without the `AB` scratch buffer or the SFPU.
- **L2, packer-accumulate add.** Let the packer sum `A cos + B sin` while writing `K_rot`.
- **L3, angle-table multicast.** One DRAM read of the table and `rot_step` per column, multicast to
  the column's cores.
- **L4, bfp4 latents.** Measure perplexity and decode fidelity with the latent caches stored in
  `bfp4_b` at the existing ranks.

Success bars, all at batch 32 on `palu_full_50` and `palu_full_70` (M-LRD, Phase 2), whole-step
decode `min_ms` from `palu_tt.bench` (which reproduces to 0.1 %) and per-layer op time from the
stage 2d profile pipeline:

| # | criterion | must | stretch |
|---|---|---|---|
| S1 | correctness | every existing op, attention and decode test passes unchanged in its bars for each shipped variant (op PCC >= 0.99 vs the fp32 reference and >= 0.995 vs stock with host-rotated K; attention fused-vs-naive PCC >= 0.999; `decode_check` min PCC >= 0.99 over 63 steps; group-mode bit identity where no child merges) | same |
| S2 | fixed cost (L3) | fitted fixed term for `r_k` 32 / `r_v` 128 from 118 us to <= 78 us per layer, or the hypothesis is recorded as refuted with the prologue re-attributed from the profile | <= 65 us |
| S3 | per-chunk cost (L1 + L2) | fitted per-chunk term from 114 us to <= 101 us per layer (excess over stock halved, 26 -> 13 us) | <= 95 us |
| S4 | whole step, `palu_full_50` with the shipped variant | 4k: 65.0 -> <= 60.0 ms; 1k: 42.8 -> <= 40.5 ms; 8k reported (stage 2 `_fk2` row: 65.0 / 42.8 / 94.5) | parity: <= 55.4 ms at 4k and <= 38.0 at 1k |
| S5 | bfp4 latents (L4) | a table of perplexity, `decode_check` PCC and KV bytes for both ratios in bfp8 and bfp4; adoption is a separate decision, recommended only if perplexity is within +0.2 of bfp8 and PCC >= 0.99 | – |

Out of scope: half-tile Q math in the copied op (the next fixed-cost lever if S4's stretch is
missed; it changes every Q-side operand and the fp32 query state), LoFi fidelity for the rebuild,
absorbing the position-free low-frequency K tile, MLA-style model conversion, the group-mode
three-round deadlock, paged caches. Group mode is not changed but inherits L1 to L3 through the shared
helpers and is re-verified (S1).

## 2. L1 and L2: the K rebuild without L1 round trips or the SFPU

### 2.1 What the hardware allows

Checked against `tt_metal/hw/inc/api/compute/eltwise_binary.h`, `pack.h` and the Wormhole LLK
(`tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h`):

- `mul_reuse_dest_tiles<DEST_TO_SRCA>(cb, tile, idst)`: `DST[idst] = DST[idst] * cb[tile]`. The
  unpacker moves the DST tile into SrcA and the CB tile into SrcB; the math thread overwrites
  `DST[idst]`. Pair with `mul_reuse_dest_init<DEST_TO_SRCA>(cb)`. `add_reuse_dest_tiles` exists in the
  same form.
- `pack_reconfig_l1_acc(1)`: subsequent `pack_tile` calls add the DST tile onto the tile already at the
  destination L1 address instead of overwriting it (the packer reads L1, adds in fp32, writes back in
  the CB's format). Used today by prefill's `sub_exp` row sums.
- The FPU multiply **cannot** accumulate into DST on Wormhole: the ELWMUL microcode is emitted with
  `acc_to_dest = 0` unconditionally (only ELWADD and ELWSUB honour it). So "two multiplies summed in
  DST by the FPU" is not available, and the sum must come from the SFPU (today), the packer (L2) or a
  dest-reuse add from a scratch tile.
- A matmul and element-wise ops can share one DST acquisition with an init switch in between: the
  stock kernel's dynamic-chunk mask fusion does exactly this (`matmul_block`, `add_init`,
  `add_tiles`, `matmul_block_init` inside one acquisition).

### 2.2 The two rebuild modes

`palu_build_k_chunk` gains a compile-time rotation mode, selected by an op attribute `rot_mode`
with values `legacy`, `pack` and `dst` (defaults to `legacy` until the results choose). Per K row
tile `i` (`Rt` latent tiles in, 4 rotated output tiles out, `dst_size` 4):

**`legacy` (today).** Two matmul trips producing `[A | B]` (8 tiles) into `c_10`; then per column tile
`j` one trip: `mul_tiles(A_j, cos)`, `mul_tiles(B_j, sin)`, SFPU `add_binary_tile`, one pack. Six trips,
four SFPU passes, eight packs and eight unpacks through `c_10`.

**`pack` (L2 alone).** Same two matmul trips into `c_10`; then per *pair* of column tiles one trip:
`mul_tiles(A_j, cos_j) -> dst0`, `mul_tiles(B_j, sin_j) -> dst1`, `mul_tiles(A_j+1, cos_j+1) -> dst2`,
`mul_tiles(B_j+1, sin_j+1) -> dst3`; pack `dst0 -> c_1[idx_j]` with accumulate off, `dst1 -> c_1[idx_j]`
with accumulate on, likewise `dst2`/`dst3`. Four trips, no SFPU, the same `c_10` traffic. `mul_init` and
`matmul_block_init` are hoisted so they run once per phase, not per row tile: either per row tile
(matmul phase then rotation phase) or, if L1 allows, per chunk with `c_10` widened to hold a whole
chunk's `AB` (128 tiles bf16 = 256 KB at `r_k` 32; the plan decides from `palu_l1_cb_bytes`).

**`dst` (L1 + L2).** Per pair of column tiles one trip: two `matmul_block` calls with `ct_dim` 2 put
`A_j, A_j+1` in `dst0..1` and `B_j, B_j+1` in `dst2..3` (the `[u_k | u_k T]` block's columns `j, j+1` and
`j+4, j+5`); `mul_reuse_dest_init` on the table CB, then `mul_reuse_dest_tiles` with `cos_j`, `cos_j+1`,
`sin_j`, `sin_j+1` in place; pack `dst0 -> c_1[idx_j]` and `dst2 -> c_1[idx_j]` with accumulate on for
the second, likewise `dst1`/`dst3`. **Two trips per row tile, no SFPU, no `c_10` at all.** Per trip
there are two init switches (matmul to dest-reuse multiply) and two format reconfigurations
(bfp8 latent + bf16 weights for the matmul, bf16 table for the multiply); these cannot be hoisted
across the switch.

Both new modes change rounding: `K_rot` is bf16 in `c_1`, so the packer-accumulate add rounds twice
(after `A cos`, after the sum) where `legacy` rounds once after an fp32 SFPU add. Two bf16 roundings
are within the precision of the bfp8 K cache the stock op reads, and S1's bars decide.

Per chunk of 16 row tiles the expected instruction-stream cost:

| per chunk (`r_k` 32) | legacy | pack | dst |
|---|---|---|---|
| DST round trips | 96 | 64 | 32 |
| SFPU full-tile passes | 64 | 0 | 0 |
| pack + unpack through `c_10` | 128 + 128 | 128 + 128 | 0 |
| packs into `c_1` | 64 | 128 (half accumulating) | 128 (half accumulating) |
| engine inits | 48 | 2 to 32 | 64 |
| L1 for `c_10` | 32 KB | 32 KB (or 256 KB, hoisted) | 0 |

Neither number is a prediction of time; the profile fit is the measurement, and both modes ship
behind the switch so the fit can compare them on the same day.

### 2.3 Group mode and V

`palu_build_v_chunk` (in-kernel V for G-LRD) is a plain matmul and is unchanged. Group mode calls
`palu_build_k_chunk` with the same template and inherits the mode; its state rings and reduction are
untouched. The `c_10` slot stays allocated in `dst` mode only where the group-mode peel uses it as
`cb_l_head` scratch (one statistics block), so the factory sizes it to `max(ab_tiles, Sq_chunk_t)`
per mode.

## 3. L3: multicast the angle table

### 3.1 Design

`cs_tab` rows `0 .. c-1` (`Sk_chunk_t x Wt` tiles bf16) and `rot_step` (`Wt` fp32 tiles) are identical
on every core and read once per core per call. A new op attribute `table_mcast` (default off until
measured) makes the reader on the **top active core of each grid column** read both from DRAM into its
own `c_13` / `c_14` and `noc.async_write_multicast` them to the same L1 addresses on the active cores
below it, then signal them through the existing multicast semaphore (`k_mcast_semaphore_id`, already
created by the factory; `Semaphore<>::set_multicast` as the stock K-multicast does). Receivers
`reserve_back`, wait on the semaphore, `push_back`. Both CBs are empty at that point on every core, so
the write pointers coincide and no reservation can be outstanding.

The factory computes per core: `do_table_mcast` (leader), the physical `mcast_x`, `mcast_y0`,
`mcast_y1` of the receivers and `num_dests`, from the active core list (row-major, so a column's active
cores are a prefix from the top). Columns with a single active core skip the multicast. The
per-head weight block is **not** multicast: different cores in a column hold different heads and the
block is 18 KB per head, 1.2 MB per layer per chip, 5 us.

Expected effect if the fixed-cost hypothesis holds: DRAM traffic for the table from 18.4 MB to
2.3 MB per layer per chip; the leaders' reads overlap the multicast fan-out; fixed term down by 50 to
70 us per layer. S2 measures it. If the fixed term does not move, the profile's per-op breakdown of
the first chunk is used to re-attribute the prologue (Q read, query-state seed, first latent chunk)
and the result is recorded either way.

### 3.2 Risks specific to L3

- **Multicast rectangles over non-worker rows.** A logical column maps to physical rows with gaps
  (harvested row, Ethernet row). The stock K-multicast issues the same kind of vertical rectangle
  through `worker_core_from_logical_core`, so the implementation copies its addressing exactly and
  tests every layout in S1 (batch 32, 16, 8, 1; full and partial columns). If a layout proves unsafe,
  the fallback is a two-level unicast tree from the leader's writer over NoC 1 (leader -> 1 -> 2 -> 4),
  which cuts DRAM traffic the same 8x at about three multicast latencies.
- **Semaphore reuse.** The multicast semaphore is set once per call; receivers reset it after the
  wait as the stock code does. It is not otherwise used in the Palu envelope (no column-major Q
  parallelism).

## 4. L4: bfp4 latents (measurement only)

`palu_tt.model.build_model` already takes `latent_dtype`; `bench`, `eval_ppl` and `decode_check`
gain `--latent-dtype {bfp8,bfp4}` (default bfp8, the existing behaviour) and a row suffix `_l4`. The
fused op accepts the tensor's dtype through its existing tile-size plumbing; the factory's
`lat_tile_size` follows the format, and `LATENT_TILE_BYTES` already lists `bfloat4_b` (576 B).
Measured for `palu_full_50` and `palu_full_70`: perplexity (141 windows), `decode_check` (63 steps,
PCC and argmax agreement), KV bytes per token, and decode `min_ms` at 4k (expected unchanged: the
op is not byte-bound). Reported next to the bfp8 rows. No kernel change; if the op refuses bfp4 at
some check, the check is relaxed only where the format is genuinely irrelevant.

## 5. Host, harness, tests

- Op attributes `rot_mode` and `table_mcast` on `palu_sdpa_decode` (nanobind + `operation_attributes_t`
  + hash), mapped to compute defines `PALU_ROT_MODE` = 0/1/2 and a reader define + runtime args.
  `PaluRuntime` gains `rot_mode` and `table_mcast`; `bench`, `decode_check`, `profile_decode` gain
  `--rot-mode {legacy,pack,dst}` and `--table-mcast {off,on}`; result row suffix `_fk2` + `{p,d}` +
  `{m}` (e.g. `palu_full_50_fk2dm`). The stage 2 `_fk2` numbers stay as the baseline. After the
  measurements, the fastest mode that passes S1 becomes the op's default (`legacy` stays
  selectable), and the report names the shipped variant by its suffix.
- `palu_tt/profile_fit.py`: the stage 2d least-squares fit as a script over the profile JSONs
  (`fixed + chunks x per_chunk` per op signature, with the +-3 us caveat), so every variant's
  attribution is one command.
- Tests. Op level (`tests/test_palu_sdpa_device.py`): the existing grid parametrised over
  `rot_mode` for the calibrated shapes (`r_k` 32/64/96 at S 1024, batch 1 and 32; S 4096/8192 at
  `r_k` 64) and, for `dst`, an `r_k` 96 case (three inner tiles across two `matmul_block` calls);
  `table_mcast` on and off at batch 32, 8 and 1 (full, half and quarter columns) with bit-identical
  output to `table_mcast` off; group mode (`heads_per_core` 2) under each `rot_mode` for the
  `torch.equal` layouts. Attention level: `test_fused_decode_matches_naive` under the shipped mode.
  Host: `profile_fit` on the committed stage 2d JSONs reproduces the table in §0 to two figures;
  L1 model updated for `dst` mode (`c_10` sizing) and pinned by the existing test.
- Device time: three rotation modes x two multicast settings x (bench 2 rows x 3 contexts + profile
  2 rows x 3 contexts) ~ 1.6 h; op and attention suites per variant ~ 40 min; `decode_check` for the
  shipped variant on `palu_full_50/70` and `glrd_50` ~ 10 min; bfp4 ~ 40 min. About 3.5 h, run
  through the device gate as always.

## 6. Evaluation and report

`docs/results/stage2e.md` and a results page:

1. The ablation: the fit table of §0 extended with one row per variant (`legacy`/`pack`/`dst` x
   multicast off/on) for `r_k` 32 / `r_v` 128 and `r_k` 64 / `r_v` 128, and the per-op time at
   1k / 4k / 8k next to stock, so each lever's effect on the fixed and per-chunk terms is read off
   directly. A figure like `profile_attention_ctx.png` with the variants as series.
2. Whole-step decode at batch 32 for the shipped variant against stock and the stage 2 `_fk2` rows
   (1k / 4k / 8k), and the batch-8 and batch-1 rows at 4k to confirm nothing regressed on split
   layouts.
3. Fidelity: `decode_check` for the shipped variant, and the op-level PCC of each mode against the
   fp32 reference side by side, so the double-rounding cost of L2 is a number.
4. The bfp4 table (S5).
5. Verdict against S1 to S5, and the honest attribution: which of the "idle resource" hypotheses
   held (table traffic as the fixed cost; choreography as the per-chunk cost) and by how much.

## 7. Risks

- **Dest-reuse under fp32 dest accumulate.** The dest-reuse ops are documented and used elsewhere
  in tt-metal, but not on a matmul result under fp32 dest in this kernel. The first task of the plan
  is a one-tile op-level test of each primitive combination (`matmul_block` then
  `mul_reuse_dest_tiles`, and pack with L1 accumulate onto a bf16 tile), so a wrong assumption
  costs an hour, not the stage. Fallback for the multiply: `pack` mode. Fallback for the add: keep the
  SFPU `add_binary_tile` inside the `dst` trip (still two trips per row tile).
- **The fixed-cost hypothesis is wrong.** Then L3 buys little and S2 records it; the stage still
  delivers L1/L2 and the bfp4 table. The profile per-op breakdown of a one-chunk call (1k context)
  is the diagnostic.
- **Precision.** Two bf16 roundings per `K_rot` element; if S1's op bar (0.995 vs stock with
  host-rotated K) fails for `pack`/`dst`, the SFPU-add fallback above restores single rounding at the
  cost of the SFPU passes, and the report states the trade.
- **L1.** `dst` mode frees `c_10`; `pack` with chunk-level hoisting needs +224 KB and is only taken
  where `palu_l1_cb_bytes` says it fits (it does at `r_k` <= 64 per stage 2's 990 KB at `r_k` 96).
- **Multicast addressing** (§3.2).
- **Attention-only bench noise** (up to 20 %): as in stage 2c, conclusions rest on whole-step decode
  and the profile fit, never on the attention-only cells.
