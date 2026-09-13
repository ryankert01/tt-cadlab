# Stage 2f: spend rank where time is free and bytes where format is cheap

Status: proposed (2026-09-14). Companion to stage 2e (`2026-09-14-palu-tt-stage2e-design.md`), which
changes the kernel; this stage changes nothing on device and asks which (K rank, V rank, cache
format) allocation gives the best quality per KV byte under the cost model stage 2d measured.

## 0. Premise: the cost model, and what Palu's default allocation ignores

Stage 2d measured, for the fused Phase 2 op at batch 32:

- Decode time is nearly independent of rank. Doubling `r_k` from 32 to 64 costs +8 % per layer;
  halving `r_v` from 128 to 64 saves 2 %. The op is bound by instruction-stream overhead, not by
  bytes (22 to 42 % of the DRAM roofline).
- V compression is free at decode for per-head Palu: the V latent is used directly as V and `u_v`
  folds into the output projection. V costs bytes and nothing else.
- Bytes still decide capacity (max context at a batch) and, at long context, time.

Palu allocates rank by Fisher information under one global budget, which at ratio 0.5 puts K at
rank 32 almost everywhere and V at about 102. Its cost model is bytes only. On this hardware the
right objective is quality per byte with time held nearly constant, and bytes can be bought two
ways: rank, or the cache's tile format. The unpacker converts `bfp4_b` (4 bits per element, one
8-bit exponent shared by 16 consecutive elements) into the engine format for free, so a bfp4 latent
at rank 64 costs the same bytes as a bfp8 latent at rank 32 and, per stage 2d, about 8 % more time.

Two properties make block floating point plausible for latents, and one of them needs a fix:

1. **Whitened latents are decorrelated.** Palu whitens the projection input, so any orthonormal
   projection of the whitened input has identity covariance over the calibration distribution.
2. **But Palu's default split leaves them far from unit scale.** The dump stores `L = U sqrt(S)` as
   the up-projection and `R = sqrt(S) V` on the latent side (`svd_linear.py`), so latent component `j`
   has variance proportional to the singular value `sigma_j`, which decays across the rank index. A
   16-element block-float group spanning components `j .. j+15` therefore spans a magnitude range of
   `sqrt(sigma_j / sigma_j+15)`, and bfp4's 3-bit mantissa zeroes anything below one eighth of the
   block maximum. Moving the singular values entirely into the up-projection (`L' = U S`, `R' = V`)
   makes every latent component unit variance at no runtime cost: it is a diagonal rescale of the
   stored factors, exact in real arithmetic (`L'[:, j] = L[:, j] * c_j`, `R'[j, :] = R[j, :] / c_j`
   with `c_j = ||L[:, j]||`, which equals `sqrt(sigma_j)` when `U` has orthonormal columns).

A third point sets the bar: **the stock model with a bfp4 KV cache** is the format-only competitor.
It stores 576 B per token per layer per chip, the same bytes as Palu ratio 0.5 in bfp8 (574 B), with
no reconstruction at all. If its quality is acceptable it beats every Palu row measured so far on
this hardware, and the study has to say so.

## 1. Goal and success criteria

Produce the quality-per-byte frontier over K rank, V rank and cache format for Llama-3.1-8B-Instruct
on the n300, with the time axis predicted from stage 2d and confirmed on the finalists, and pick the
allocation to carry into stage 3.

| # | criterion | must | stretch |
|---|---|---|---|
| Q1 | frontier | a table and figure of wikitext perplexity against KV bytes per token for every cell in §2, with the stock bfp8 and bfp4 points and the existing Fisher-searched rows on the same axes | five-task accuracy for the finalists on the same figure |
| Q2 | unit-scale hypothesis | the bfp4 perplexity of the existing ratio-0.5 and 0.7 dumps with balanced vs unit-scaled latents, reported as a difference with its window-to-window noise | – |
| Q3 | format-only baseline | stock with a bfp4 KV cache: perplexity, five tasks, decode step at 1k / 4k / 8k, max context at batch 32 | – |
| Q4 | a recommendation | one allocation with perplexity <= 9.1 (the ratio-0.7 M-LRD level, 9.03) at <= 37 KB per token for the whole model (the ratio-0.5 M-LRD bytes, 36.7 KB), or a statement that none exists on the grid and where the frontier lies instead | perplexity <= 8.3 (the G-LRD 0.7 level) at <= 37 KB per token |
| Q5 | time | decode `min_ms` at 1k / 4k / 8k for the finalists within 10 % of the stage 2d prediction (`r_k` 32 row +8 % per doubling of `r_k`, independent of `r_v` and format) | – |

Out of scope: kernel changes (stage 2e), Fisher-searched split budgets unless §2's trigger fires,
J-LRD, HellaSwag, calibration-set changes, paged caches, bf16 caches.

## 2. The grid

Per token, per layer, per chip, bytes are `4 heads x (r_k x bpe_k + r_v x bpe_v)` with `bpe` 1.0625
for bfp8 and 0.5625 for bfp4; whole-model KB per token is that times 64 layer-chips divided by 1024.
Stock is 1088 B (69.6 KB per token).

**Grid A, the afternoon (existing dumps, no calibration).** `palu_full_50`, `palu_full_70`,
`glrd_50_f256`, `glrd_70_f256` x cache format {bfp8, bfp4} x latent scale {balanced, unit}: 16
cells, plus stock {bfp8, bfp4}: 18 cells. bfp8 balanced is the existing row for the four dumps and
stock bfp8 is the existing baseline, so 13 new runs. This answers Q2 and Q3 and gives the first bfp4 points on the frontier. The
two bfp4 balanced cells of `palu_full_*` are stage 2e's L4 rows; whichever stage runs first records
them and the other reuses the JSON.

**Grid B, the factorial (new uniform-rank dumps).** Calibrated per-head factors (256-sample whitening,
the cached scale matrices) at uniform ranks `r_k` in {32, 64, 96} x `r_v` in {32, 64, 96, 128}, unit
scale, in {bfp8, bfp4}: 12 dumps, 24 cells. This is the clean instrument for the K-versus-V question,
because the existing Fisher dumps confound rank allocation with per-head variation. The stock and
Grid A points anchor it: uniform (32, 102) in bfp8 should land near `palu_full_50`, and the
comparison says how much the Fisher allocation is worth (uniform (32, 96) in bfp8 should land near
`palu_full_50`, whose Fisher ranks average 33 / 102).

**Trigger for a split-budget Fisher search** (a conditional third grid, not run by default): if
Grid B shows the K-versus-V asymmetry the premise predicts (perplexity more sensitive to `r_v` than
to `r_k` at equal bytes, or the reverse), one Fisher-uniform search per matrix with separate targets
(`k_proj` at the K ratio, `v_proj` at the V ratio) at the two best Grid B corners, to see whether
per-head allocation adds on top of the uniform result. Two dumps, four cells.

**Screening and finalists.** Every cell is screened at 40 windows (about 3 minutes each; 82k tokens
give perplexity to about +-0.05, enough for differences of 0.2). The four best cells by bytes and
perplexity, plus the stock bfp4 baseline, are then measured in full: 141-window perplexity, the five
tasks (52 minutes per row), `decode_check` (fused Phase 2; PCC and argmax agreement, since bfp4
latents change what the kernel reads), decode `min_ms` at 1k / 4k / 8k and max context at batch 32
(`bench --what maxctx`), and, where bfp4 doubles the reach, the step at 16k.

## 3. Pipeline changes (host only)

- **Uniform-rank dumps.** `scripts/run_calibration_uniform.sh <r_k> <r_v> [gs]` -> `compressed/uni_rk<K>_rv<V>`:
  Palu's `compress.py` gains a `--ranks_json` path (a `patch_palu_cpu.sh` patch) that replaces the
  rank search with a supplied per-module rank list; the script writes that list (`r_k` for every
  `k_proj` group, `r_v` for every `v_proj` group), runs the whitened decomposition with `--use_cache`
  (the 256-sample whitening cache is keyed by model and reused), and keeps only the low-rank factors
  as the other dumps do. CPU, niced, about ten minutes per dump; Grid B's twelve run as one
  background queue.
- **Latent scale.** `palu_tt.convert.load_palu_dir` gains `latent_scale in {"balanced", "unit"}`
  (default `balanced`, the existing rows): `unit` rescales each head's factors by the column norms
  of the up-projection as in §0. A test checks the rescaled reconstruction `L' R'` equals `L R` to
  bf16 precision, and an empirical check on two layers computes the latent's per-component standard
  deviation on two wikitext windows through the CPU reference model, before and after, expecting a
  decaying profile to become flat within a factor of two.
- **Cache formats.** `build_model` gains `k_latent_dtype`, `v_latent_dtype` (Palu rows; default the
  existing single `latent_dtype`) and `kv_cache_dtype` (stock rows, through the decoder precision
  settings, `PrecisionSetting.BFP4` exists), with `LATENT_TILE_BYTES` and the byte accounting
  extended to `bfloat4_b` on both paths. `eval_ppl`, `eval_tasks`, `bench`, `decode_check` gain
  `--latent-dtype {bfp8,bfp4}`, `--k-latent-dtype`, `--v-latent-dtype`, `--kv-dtype`, `--latent-scale`.
  Row naming: `<dump>[_u][_lk4|_lv4|_l4]`; stock bfp4 is `baseline_kv4`.
- **Frontier report.** `palu_tt.report --stage 2f`: the table (row, `r_k`, `r_v`, formats, scale, B per
  token per layer per chip, KB per token, reduction vs stock, perplexity with windows, tasks where
  measured, predicted and measured decode at 4k, max context) and `palu_tt.plots --frontier`: bytes on
  x, perplexity on y, format as marker, `r_k` as colour, the stock points and the Pareto front drawn.

## 4. What "free" means, precisely

Time is not re-derived per cell. The stage 2d fit gives per-layer fused time as
`118 + 114 x chunks` us at `r_k` 32 and `130 + 123 x chunks` at `r_k` 64; `r_k` 96 is extrapolated at
+16 % over 32 and confirmed on a finalist. `r_v` and format do not enter. Q5 checks the prediction on
the finalists so the frontier's time axis is trustworthy without benchmarking every cell. Bytes enter
time only through capacity: max context at batch 32 for the finalists, which is where bfp4 should pay
(the ratio-0.5 bfp8 row reaches 8k at batch 32; a bfp4 row of the same rank should reach 16k).

## 5. Evaluation and report

`docs/results/stage2f.md` and a results page:

1. The frontier figure and table (Q1), with the stock bfp8 and bfp4 points marked and the Pareto
   front drawn.
2. Q2 as a four-row table: ratio 0.5 and 0.7, bfp4 balanced vs unit, perplexity and delta, next to
   the bfp8 numbers.
3. Q3: the format-only baseline in full, against `palu_full_50` (same bytes) and `palu_full_70`.
4. Grid B read as a factorial: perplexity as a function of `r_k` at fixed `r_v` and vice versa, and
   the equal-bytes diagonals (bfp4 at rank 2r vs bfp8 at rank r).
5. Finalists in full (perplexity, tasks, `decode_check`, decode 1k / 4k / 8k / 16k, max context) and
   the recommendation (Q4) with its bytes, time and quality stated against stock.
6. Caveats: uniform ranks are not Fisher ranks; wikitext perplexity is not task accuracy (stage 2c);
   40-window screening noise; bfp4 rounding is applied at cache write time on device, so the CPU
   reference is not a bfp4 reference.

Device time: Grid A 13 x 3 min, Grid B 24 x 3 min, finalists 5 x (10 + 52 + 10) min, stock bfp4
bench and maxctx 10 min: about 8 hours through the device gate, in three sessions (the afternoon,
the factorial, the finalists). CPU: 12 dumps, about 2 hours niced.

## 6. Risks

- **bfp4 support on the stock path.** The stock flash-decode and paged-update-cache ops read what
  the precision settings give them; `PrecisionSetting.BFP4` exists for the KV cache group in
  tt-transformers, but this model's decode may never have run with it. If an op refuses, the
  baseline is recorded as unavailable with the error, not worked around.
- **bfp4 on the fused op.** The op reads the latent through its tile-size plumbing; stage 2e's L4
  task establishes it. If the factory's dtype checks refuse `bfloat4_b`, they are relaxed only where
  the format is genuinely irrelevant.
- **The unit rescale assumes `U` has orthonormal columns** (column norms of `L` equal `sqrt(sigma_j)`).
  If the dump folds anything else into `L`, the empirical latent-std check in §3 catches it and the
  rescale switches to the measured standard deviations.
- **Uniform ranks may under-represent Palu.** The comparison to the Fisher rows and the conditional
  split-budget search bound that; the report never claims a uniform cell beats Palu's allocation
  without the anchor next to it.
- **Perplexity can hide task damage** (stage 2c: tasks moved four times more than perplexity). Tasks
  are measured on the finalists before any recommendation.
- **Quantisation cliff.** bfp4 may be unusable for K regardless of scale (rotation-sensitive), in
  which case the frontier's answer may be "bfp4 V only", which the separate `_lv4` cells cover.
