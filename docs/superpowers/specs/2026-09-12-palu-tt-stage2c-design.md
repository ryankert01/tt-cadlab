# Stage 2c: 256-sample calibration for G-LRD, and a shared-latent (group) decode kernel

Status: executed (2026-09-12/13). Results and the analysis are in `docs/results/stage2c.md`; the
recorded deviations from this spec are §1's bit-identity bar (met where the work split is the same,
PCC >= 0.9995 where a child merges — see that document's §6.2 for why it provably cannot be met
otherwise), the group split capped at 4 cores per item instead of §3.3's 8 and 16 (an unresolved
three-round-reduction deadlock, §6.3), and the host's choice of H being occupancy-aware as well as
L1-aware, which turns group mode off below batch 8.

Stage 2b left two follow-ups at the top of its list: (1) the leading explanation for our G-LRD
shortfall against the paper is the 32-sample Fisher calibration, untested; (2) fused G-LRD decode
runs at 0.46-0.70x stock because each of the four head work-items of a group re-reads the same
latent chunk. This stage does both and reports them side by side with the stage 2b rows.

## 1. Goal and success criteria

1. **Calibration sensitivity.** Re-run the Palu recipe for both G-LRD ratios with 256 Fisher
   samples (rows `glrd_50_f256`, `glrd_70_f256`), score them on the five tasks and on perplexity,
   and answer: does the five-task gap to stock move toward the paper's G-LRD gap (-3.17 on the
   paper protocol; ours -5.09 at ratio 0.5)? Any movement of the average by more than the
   bootstrap noise (~0.5 points on 22k sequences) is a result either way.
2. **Shared-latent kernel.** Make the fused op read each group latent chunk once per core for
   all the heads that core owns, and let a core own all four heads of a group while still
   splitting the sequence over several cores. Success bar: (a) bit-identical output to the
   per-head mode for the same inputs (same per-head math, same chunk order); (b) attention-only
   slope at batch 32 for `glrd_50_fk2` from 0.583 ms per 1k toward M-LRD's 0.259 -- the target is
   <= 0.35; (c) decode step at 4k, batch 32, from 105.5 ms to <= 80 ms; (d) no regression of the
   per-head path (`heads_per_core = 1` numbers unchanged within noise).

Out of scope: J-LRD, HellaSwag, changes to the M-LRD (`gs = 1`) kernel path, paged caches.

## 2. Calibration (step 1)

- `scripts/run_calibration.sh <ratio> <gs> [n_fisher]` gains the third argument; the output
  directory gets a `_f<n>` suffix when `n_fisher != 32`. `scripts/patch_palu_cpu.sh` keys Palu's
  Fisher cache file by the calibration set size (`..._calib_fisher_info_n{len(calib_loader)}.pt`);
  the stage-1 cache was renamed to the `_n32` name. The whitening cache is unchanged: Palu's
  whitening already used its default 256 samples (`n_whiten_calib_samples`), so the only
  calibration difference between the paper and us is the Fisher pass.
- Measured cost: 66 s per Fisher sample in fp32 on 12 threads -> 4.7 h for 256 samples, once
  (both ratios share the cache); rank search and decomposition take minutes. Started 17:50 on
  2026-09-12 as a background job (`calib_f256.sh` in the session scratchpad).
- Dumps: `compressed/glrd_0.5_f256`, `compressed/glrd_0.7_f256` (factors only, like the others).
- Device runs (rows `glrd_50_f256`, `glrd_70_f256`, fused Phase 2 variants `_fk2`): `eval_ppl`,
  `eval_tasks` (five tasks, full), `bench --what decode --fused p2`, `decode_check --fused p2`.
  Budget: 2 x (10 min ppl + 53 min tasks + 5 min bench/check) ~ 2.3 h of device time.
- Report: a `Calibration sensitivity` table in the stage 2c document: per-task accuracy, the
  paper-protocol five-task average, perplexity, and the K/V rank means, for `glrd_*` (f32) next to
  `glrd_*_f256` and the paper's G-LRD gap.

## 3. Shared-latent kernel (step 2)

### 3.1 Today's work layout

The op inherits flash decode's layout: a work item is (batch, KV head); `num_heads_per_core`
items run sequentially on a core (`for cur_head_work`), each with its own Q block, weight block,
flash state (`m`, `l`, `O` in `c_27..c_31`, `c_25/c_26`) and, in Phase 2, its own rotated-query
state (`c_12`). When a batch has more cores than heads the item's K chunks are split over
`num_cores_per_head` cores and merged by a tree reduction (writer moves `l, m, O` payloads via
`c_6/c_7/c_16`, compute merges with `correction_block`). The factory forbids both
`num_heads_per_core > 1` and `num_cores_per_head > 1` at once (writer asserts
`num_heads_per_core == 1` in the reduction). On the n300 (64 cores, 4 KV heads per chip):

| batch | items | heads per core | cores per head | latent reads per chunk |
|---|---|---|---|---|
| 32 | 128 | 2 | 1 | 4 (the two heads on a core read it twice; the other two heads on another core) |
| 16 | 64 | 1 | 1 | 4 |
| 8 | 32 | 1 | 2 | 4 (each half of the sequence) |
| 1 | 4 | 1 | 16 | 4 |

The reader reads, per head, the weight block once and then every latent chunk; the four heads of
a group therefore read each latent chunk four times in total. Stage 2b measured a 2.25x
attention slope over M-LRD for a 3.6x element-count ratio.

### 3.2 Group-shared mode

New compile-time mode `PALU_GROUP_SHARED`, opt-in per call (host: `group_shared =
heads_per_core > 1`, where `heads_per_core` is the new op parameter; `None`/`1` keep the per-head
kernel and the stock split -- implicit activation would exceed L1 at batch 32 on wide latents). The host guarantees `num_heads_per_core` divides `heads_per_group` and
that a core's heads are contiguous within one group, so all heads on a core share the same
latent (`lat_head = cur_head / heads_per_group` is constant on the core).

**Compute kernel, loop order swapped.** Per core:

```
H = num_heads_per_core
seed the H rotated-query states (Phase 2) and H flash states
for k_chunk in [start, end):
    wait one K-latent chunk (c_9) and one V-latent chunk (c_2)     # pushed once per chunk
    for h in 0..H-1:
        Phase 2: advance q-state h by R_{-c} (except at the first chunk)
        K_h   = build_k_chunk(latent, w_rot block h)   -> c_1
        S_h   = Q_h K_h^T ; online-softmax update of state h (m_h, l_h, O_h)
        V_h   = build_v_chunk(v latent, u_v block h)   -> c_1
        O_h  += P_h V_h
    pop the two latent chunks
for h in 0..H-1:
    tree reduction for head h (unchanged code, per head), then output head h
```

State is replicated per head as FIFO-rotated blocks in the existing CBs (`c_26`, `c_28`, `c_30` and
the reduction scratch CBs sized `H x`); Q and the Phase-2 rotated-query state are per *core*, not per
head (one Q tile holds every q head of the core, as in stock), so `c_0`/`c_12` keep one block. The
per-head math, chunk order and accumulation order are unchanged, so head `h`'s result is
bit-identical to the per-head mode.

**Weight blocks resident for all H heads.** `c_4` holds `H` whole blocks `[w_rot | u_v | T]`
pushed once by the reader in head order; the compute indexes block `h` (today it pops the single
block after each head). This is what bounds `H`: per head `Rt*Wt + Rvt*DHt (+1)` tiles bf16
(2 KB each): 168 KB at the ratio-0.5 layer means (r_k 128, r_v 416), 274 KB at 288/512,
390 KB at 512/512.

**Reader.** In group mode: read Q for the H heads (c_0 sized `H x q_tiles`), the H weight blocks,
then the latent chunks once each (K latent to `c_9`, V latent to `c_2`, cs table in Phase 1 to
`c_13`), double-buffered by chunk as today so the next chunk streams while the H heads compute.
The per-head `read_w_rot` / `read_palu_chunks` helpers are reused; the loop nesting changes.

**Writer.** Unchanged protocol: after the chunk loop the compute emits head 0's payload, then
head 1's, ...; the writer's existing per-head loop (send to parent / receive from children / write
output) runs `H` times in the same order on every core, so payloads for head `h` never interleave
with head `h+1`'s. The `ASSERT(num_heads_per_core == 1)` in the writer's reduction branch is
replaced by the per-head loop.

### 3.3 Work layout with sequence split (Phase B)

The factory restriction "not both `num_heads_per_core > 1` and `num_cores_per_head > 1`" is
lifted for the group-shared mode: an item is (batch, head-group of `H` heads), split over
`num_cores_per_head` cores; `num_reducer_cores = B * num_kv_heads / H`, `num_active_cores =
num_cores_per_head * B * num_kv_heads / H`, reducer/output core selection and the tree tables are
the existing formulas with the item count. Target layouts on the n300 with `H = 4`:

| batch | items | cores per item | latent reads per chunk | cores busy |
|---|---|---|---|---|
| 32 | 32 | 2 | 1 (per half sequence) | 64 |
| 16 | 16 | 4 | 1 | 64 |
| 8 | 8 | 8 | 1 | 64 |
| 1 | 1 | 16 | 1 | 16 (as today) |

`H` is chosen on the host per layer: the largest of {4, 2, 1} dividing `heads_per_group` whose
L1 estimate fits (stock CBs ~300 KB + `H` weight blocks + double-buffered latent chunks at the
default chunk + `H x` state + AB + cs). With chunk 128: `H = 4` fits at the ratio-0.5 layer means
(~672 KB of weights), not at the widest layers (288/512: 1.1 MB); those fall back to `H = 2`
(548 KB) or `H = 1`. The estimate is the factory's own CB sum, exposed to Python as
`palu_sdpa_decode_l1_bytes(...)` (or computed in `palu_tt/fused.py` from the same formula) so the
attention module can pick `H` when it builds its program config; the op accepts
`heads_per_core` as an explicit program parameter (default: the stock rule, i.e. today's
behaviour), and the reduced-`H` fallback is logged once per layer.

### 3.4 Host and harness

- `PaluRuntime.group_heads_per_core: Optional[int]` (None = automatic per layer; 1 = today's
  per-head kernel). `bench`, `decode_check`, `eval_tasks` gain `--group-heads {auto,1,2,4}`; row
  suffix `_fk2g` for the automatic group-shared runs, so the stage 2b `_fk2` rows stay as the
  baseline.
- Tests: op-level `test_group_shared_matches_per_head` (bit-identical output for `H in {2, 4}` vs
  `H = 1`, batch 32 and 8, `r_k in {64, 128, 288}`, `r_v in {256, 512}`); attention-level
  `test_glrd_group_shared_matches_naive` at batch 32 and 8; the existing 31 op tests and the
  G-LRD attention tests must pass unchanged with `H = 1`.

## 4. Evaluation and report

`docs/results/stage2c.md` rendered by `python -m palu_tt.report --stage 2c`:

1. Calibration sensitivity table (§2).
2. Kernel: attention-only 1k/4k/8k and slope, decode step 1k/4k/8k at batch 32 and batch 8, for
   `glrd_50_fk2` / `glrd_70_fk2` (per-head, stage 2b) against `glrd_50_fk2g` / `glrd_70_fk2g`
   (group-shared), with stock and fused M-LRD for reference; the per-layer `H` chosen.
3. Decode fidelity for the `_fk2g` rows (`decode_check`, PCC and argmax against naive), and the
   bit-identity check against `_fk2` (from the op tests).
4. Batch-8 max context for `_fk2g` (expected unchanged: the caches set the ceiling).
5. Caveats: `H < 4` fallback layers, unchanged calibration caveats, what was not measured.

Device budget: step 1 ~2.3 h; step 2 op tests + attention benches + decode benches (2 rows x 2
batches) + decode_check + maxctx ~1 h; total ~3.5 h plus the 4.7 h CPU calibration.

## 5. Risks

- **L1 at `H = 4`.** The widest layers will not fit four weight blocks; the per-layer fallback is
  part of the design, and the report states how many layers ran at each `H`.
- **Writer/compute ordering in the multi-head reduction.** Both sides iterate heads in the same
  order; the risk is a hang if a payload CB is consumed out of order. Mitigation: the per-head
  reduction reuses the single-head code path verbatim inside a head loop; a hang shows up in the
  op test at batch 8 immediately.
- **Program-cache and compile-time explosion.** `H` becomes a compile-time argument, so a model
  with three distinct `H` values compiles three kernel variants; acceptable (one-off JIT).
- **Bit-identity claim.** The per-head accumulation order is unchanged, but the DST/`c_1` reuse
  across heads inside one chunk is new; the op test asserts `torch.equal`, and if a legitimate
  difference appears (e.g. packer rounding from a different intermediate CB) the bar drops to
  PCC >= 0.9999 with the reason recorded.
- **Calibration result may be null.** If `f256` does not move the gap, the paper-gap explanation
  shifts to the model revision; the report says so plainly.
