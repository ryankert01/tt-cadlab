# Palu on Tenstorrent — Stage 2b: G-LRD and Task Accuracy

Date: 2026-09-12
Status: executed 2026-09-12 (approved 2026-09-12, "Lgtm"), including the in-kernel V decision and the device-only evaluation budget
Depends on: stage 2 (`docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md`, merged at 668e250)

## 0. Two findings that shape this stage

- **Our calibration used 32 Fisher samples; the paper used 2048.** Palu's upstream rank search
  hard-codes 2048 calibration samples; stage 1 patched it to the command-line default (32)
  because 2048 samples in fp32 on this CPU was estimated at more than a day. This is a real
  difference from the paper and a plausible part of why our M-LRD degradation (1.78x the
  baseline perplexity) is worse than theirs (1.50x).
- **Paper reference numbers** (arXiv 2407.21118, Table 1, Llama-3-8B-Instruct, 50 % KV
  compression, zero-shot accuracy via lm-evaluation-harness):

  | method | OBQA | HellaSwag | PIQA | ARC-e | ARC-c | WinoGrande | avg (6) | wikitext-2 ppl (seq 4096) |
  |---|---|---|---|---|---|---|---|---|
  | baseline | 43.20 | 75.80 | 78.62 | 81.61 | 56.83 | 71.90 | 67.99 | 8.28 |
  | J-LRD | 43.40 | 73.20 | 76.50 | 79.63 | 51.96 | 72.45 | 66.19 | 9.12 |
  | G-LRD (gs 4) | 42.60 | 70.36 | 76.06 | 76.30 | 48.99 | 72.38 | 64.45 | 10.11 |
  | M-LRD | 38.80 | 63.04 | 73.67 | 69.78 | 42.58 | 62.51 | 58.40 | 12.38 |

  The paper's calibration is 2048 Wikitext-2 samples of length 1024. Our perplexity windows
  are 2048 tokens, which is why our baseline reads 7.25 against their 8.28.

## 1. Goal and success criteria

Add G-LRD with group size 4 (one group per chip on the N300: each chip's 4 KV heads share
one K latent and one V latent per token) and evaluate every row the way the paper does:
zero-shot accuracy on **openbookqa, piqa, arc_easy, arc_challenge, winogrande**, in full, on
the n300, alongside the existing perplexity, KV-memory and decode rows. HellaSwag is dropped
(two thirds of the paper's sequences) per the user's budget choice; J-LRD is deferred.

Rows: `baseline`, `palu_full_50`, `palu_full_70` (M-LRD, existing dumps), `glrd_50`, `glrd_70`
(new calibrations, group size 4).

Success is descriptive rather than a bar:
1. all five rows have accuracy on all five tasks, measured on device;
2. the G-LRD rows have perplexity, KV bytes, decode timing (naive and fused) and decode
   fidelity, produced by the stage-1/2 harnesses;
3. the write-up reports our accuracy gap to stock next to the paper's gap, with the caveats
   in §8 attached. The expectation from the paper is that G-LRD recovers about half of
   M-LRD's accuracy loss.

## 2. Task-evaluation harness

- `palu_tt/lm_eval_tt.py`: `TTLM(lm_eval.api.model.LM)` implementing `loglikelihood` only.
  For each (context, continuation) request: tokenise context and context+continuation with
  the model tokenizer (BOS once), pad to a multiple of 128, `generate.prefill_hidden_states`
  on device, apply the final RMSNorm and `lm_head` on the host (the perplexity harness's
  `_nll_from_hidden` path), return the summed log-probability of the continuation tokens and
  `is_greedy`. `loglikelihood_rolling` and `generate_until` raise `NotImplementedError` (the
  five tasks need neither). Requests are processed one at a time; identical
  context+continuation strings are memoised within a run.
- `palu_tt/eval_tasks.py`: CLI `--row R --kind stock|palu [--palu-dir D] [--fused off|p1|p2]
  [--tasks ...] [--limit N]`; builds the model at batch 1, `max_seq_len` 2048; runs
  `lm_eval.simple_evaluate` with `TTLM`; writes `docs/results/raw/tasks_<row>.json`
  containing, per task, `acc`, `acc_norm` where lm-eval defines it, `n`, and lm-eval's full
  `results`/`versions`/`configs` dictionaries, plus `row`, `kind`, `palu_dir`, `fused`,
  `limit`, `lm_eval_version`, and the task-directory overrides in use.
- **lm-eval stays at 0.4.2** (the Palu repo imports it). Its `openbookqa`, `piqa` and
  `winogrande` task configs point at dataset ids the Hub no longer serves; the harness ships
  `palu_tt/lm_eval_tasks/` with override configs (`include:` the built-in yaml, override
  `dataset_path` to `allenai/openbookqa`, `ybisk/piqa`, `allenai/winogrande`) registered
  through `TaskManager(include_path=...)`. If a name clash with the built-ins forces distinct
  task names, the harness maps them back to the paper's names in the JSON.
- **Network.** Datasets download once behind the proxy with the system CA bundle
  (`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`) and
  `HF_HUB_OFFLINE=0` for that step only; `scripts/env.sh` keeps offline mode by default and
  documents the one-time download command. Verified today: arc_easy (2376 docs, 9501
  requests), arc_challenge (1172 / 4687) and winogrande (1267) load; openbookqa and piqa are
  the override cases.
- **Budget**, measured today with the stock model: `prefill_hidden_states` takes 76 ms at
  128 tokens, 91 ms at 256, 129 ms at 512. About 22,400 scored sequences per row → roughly
  35 minutes per row, ~3 hours for five rows.

## 3. G-LRD in the converter and the attention module

- **Dump format.** For group size `g`, Palu stores one `U.{k}.weight` per group of shape
  `[g·head_dim, r_k]`, one stacked `VT.weight` of shape `[Σ r_k, hidden]`, and
  `head_wise_ranks[proj]` with one rank per group. `load_palu_dir` generalises from "one entry
  per head" to "one entry per group": `n_groups = len(r_list)`, `heads_per_group = n_kv /
  n_groups`; head `h` gets the group's VT and rows `(h mod g)·head_dim … +head_dim` of the
  group's U. Ranks are padded to a multiple of 32 (so `r_g ∈ 32..512`). `LowRank` gains
  `group_size`; `PaluConfig` gains `group_size` and per-layer group ranks.
- **Fused QKV weight.** Per chip the K section is `n_local_kv_heads × head_dim = 512`
  columns, which holds one group latent of up to 512; the group's VT rows go there
  zero-padded. The stock head-split op yields four 128-wide pieces whose concatenation is the
  latent; the module reshapes `[1, B, 4, 128] → [1, B, 1, 512]` and slices `:r_g` for the
  cache write (decode) and the `fill_cache` (prefill). Same on the V side.
- **Caches** become `[B, n_groups_local, S, r_k_g]` and `[B, n_groups_local, S, r_v_g]`
  (`n_groups_local = 1` here). `kv_bytes_per_token` becomes
  `n_groups_local · (r_k_g + r_v_g) · 1.0625`.
- **Prefill** reconstructs each head from the shared latent with its own U slice
  (`u_k_prefill`/`u_v_prefill` become `[1, n_kv, r_g_pad, 128]` per head; the group latent is
  expanded over the heads before the matmul).
- **V is not absorbed on the G-LRD decode path.** The absorbed-V trick makes the attention
  output as wide as the V latent; with a group latent of up to 512 that would make the fused
  output projection four times wider (about 1.6 GB more weight per chip and an estimated 6-8
  ms per step) and would cost context capacity. Instead the fused kernel reconstructs V per
  head on core (§4) and the stock `wo` is used; `wo_fused` and the V-absorption remain the
  M-LRD path (`r_v ≤ 128`, group size 1). The calibration result confirms the need: at ratio
  0.5 the G-LRD V group ranks are 416-512 (V nearly uncompressed), K group ranks 64-288.
- **Naive decode** reconstructs K and V from the group latents per head the same way
  (`u_k_decode`/`u_v_decode` per head, latents expanded over heads, then the stock
  `scaled_dot_product_attention_decode` on full K and V), so the G-LRD rows have a naive
  timing row too.
- **Latent caches are no longer written to the weight cache.** Stage 1 passed a
  `cache_file_name` for the zero-initialised latent caches; every batch/context shape then
  left a file of up to 273 MB behind (212 GB found and deleted on 2026-09-12). The caches are
  created with `ttnn.from_torch`/zeros directly.
- **Group size 1** is the same code with `n_groups = n_kv`; the stage-1/2 rows must stay
  bit-identical (tests assert the converter output and the device caches).

## 4. G-LRD in the fused kernel

Four changes to `palu_sdpa_decode`:

- **Group indexing.** New op attribute `heads_per_group` (derived on the host as
  `w_rot.shape[1] / k_latent.shape[1]`, must divide evenly). The reader offsets the K-latent
  and V-latent reads by `cur_head / heads_per_group` instead of `cur_head`; the rotation
  weight stays per head. The factory's `num_kv_heads` (work-item layout, output shape) comes
  from `w_rot.shape[1]`, not from the cache's group dimension; validation checks
  `k_latent.shape[1] == v_latent.shape[1] == n_kv / heads_per_group`.
- **Envelope.** `r_k` widens from {32, 64, 96} to any multiple of 32 in 32..512; `r_v ≤ 512`
  as `head_dim_v`. L1 rule in the host op: when `r_k > 128`, the default chunk becomes 128
  (the latent chunk at chunk 512 and `Rt = 12` is ~418 KB) and the rotation-weight CB holds
  one block instead of two (~194 KB at `r_k = 384`); the factory's existing size assert
  remains the backstop. `PaluAttention` passes the chunk it computes to `_fused_tables` as
  today, so `rot_step` matches.
- **In-kernel V reconstruction (new mode, selected by passing `u_v`).** Inputs `v_latent
  [B, n_groups, S, r_v_g]` (read into the stock V slot `c_2`, sized for the latent width) and
  `u_v [1, n_kv, r_v_g, 128]` bf16, appended to the per-head resident weight block in `c_4`
  (`[w_rot | u_v | T]`, one push per head, CB a whole number of blocks). Per chunk, after the
  QK matmul has consumed `c_1`, the compute kernel builds `V_h = lat_v_g @ u_v[h]` row tile by
  row tile into `c_1` in row-major tile order (`k·vDHt + n`) with the same helper structure as
  `palu_build_k_chunk` minus the rotation, and the PV `matmul_blocks` reads in1 from `c_1`
  instead of `c_2`. `head_dim_v` stays 128, the output shape and the downstream tail are the
  stock ones. When `u_v` is absent the op behaves exactly as in stage 2 (absorbed V from
  `c_2`). L1 at chunk 128, r_k_g = 288, r_v_g = 512: weight block ≈ (9·8 + 16·4 + 1) tiles bf16
  ≈ 274 KB single-buffered, latent chunks ≈ 80 + 140 KB, AB 32 KB; fits with the stock ~300 KB.
- **Tests.** Op-level: `heads_per_group = 4` with `r_k ∈ {128, 256, 384}` and `u_v` given
  (`r_v ∈ {256, 512}`) against the fp32 reference (which reconstructs V too) and the stock
  decode op on host-reconstructed K and V; `heads_per_group = 1` without `u_v` unchanged
  (bit-identical outputs to stage 2 on the stage-2 test inputs). Attention-level:
  naive-vs-fused on a G-LRD dump, byte-identical caches.

**Expected decode behaviour, stated up front.** Each of the four head work-items rebuilds its
head from the full group latents, so per position the K reconstruction arithmetic is ~4x
M-LRD's, V reconstruction (absent on the M-LRD path) adds `r_v_g × 128` per head, and each
latent is read four times: at ratio 0.5 about 4 × (107 + 404) ≈ 2000 elements per position
versus M-LRD's ~540 and stock's 1024. G-LRD decode will be slower than M-LRD and probably slower than
stock, while remaining far ahead of the naive path. The paper reports the same ordering of
reconstruction cost. Removing the fourfold read requires one core to build all four heads of
a group from one latent read, which inverts the kernel's head/chunk loop nesting and needs
four sets of softmax state in L1; that is a follow-up, not this stage. The memory and
accuracy rows are the payoff of G-LRD here.

## 5. Calibration

Two CPU runs with the stage-1 recipe (`scripts/run_calibration.sh` gains a group-size
argument): `--head_group_size 4`, ratios 0.5 and 0.7, Fisher-uniform rank search with 32
samples, whitened decomposition in float64. Outputs `compressed/glrd_0.5`, `compressed/glrd_0.7`.
They start first, in the background, while the code is written (each is hours of 12-thread
CPU time and ~55 GB of host RAM; nothing else CPU-heavy runs alongside). If CPU time allows,
one extra run `glrd_0.5_f256` with 256 Fisher samples as a sensitivity check on the
32-vs-2048 gap, reported as its own row and never substituted for the 32-sample row.

## 6. Tests

- CPU: converter round-trip on a synthetic group-size-4 dump (`K_h = lat_g @ U_g[h]`
  reproduces the group weight per head; padding to 32 preserved); group size 1 output
  unchanged byte-for-byte against the current converter on the stage-1 fixtures; `TTLM` on a
  tiny fake model (continuation boundary, BOS handling, `is_greedy`); the report renderer on
  fixture JSONs.
- Device: the op tests in §4; the attention-level test in §4; a 20-example lm-eval run on
  `arc_easy` for stock asserting accuracy in a plausible range and the JSON schema.

## 7. Evaluation and report

`docs/results/stage2b.md`, rendered by `palu_tt.report --stage 2b` from the JSONs:
1. accuracy per task and five-task average for the five rows, with the paper's
   Llama-3-8B-Instruct numbers as a reference block averaged over the same five tasks;
2. perplexity (device, 141 windows) and KV bytes for the five rows;
3. decode and attention timing at batch 32 (1k/4k/8k) for `glrd_50`/`glrd_70`, naive and fused
   Phase 2, next to the stage-2 rows;
4. decode fidelity (`decode_check`) for the fused G-LRD rows;
5. max context at batch 8 for the fused G-LRD rows.
**Time budget (user requirement: evaluation runs on the n300, not the CPU; device time is the
budget to manage).** All evaluation is device-side; the CPU is used only for calibration
(done) and unit tests. No CPU fp32 perplexity column for the new rows.

| activity | rows | estimate |
|---|---|---|
| zero-shot accuracy, 5 tasks in full | 5 | ~35 min each, ~3 h |
| perplexity, 141 windows | glrd_50, glrd_70 | ~5 min each |
| decode + attention benchmarks, 1k/4k/8k, batch 32 | 2 rows × naive/fused | ~40 min |
| decode fidelity (`decode_check`) | 2 fused rows | ~20 min |
| batch-8 max context probes | 2 rows | ~20 min |
| device tests during development | | ~30 min, spread out |

About 5 hours of board time. The accuracy block runs as one queued job (one process at a
time, gated) overnight so the board is never idle between rows. The optional 256-sample
Fisher calibration is a CPU background job and costs no device time. The results page
(artifact) gets the same tables.

## 8. Risks and caveats

1. lm-eval task overrides may need task names distinct from the built-ins to avoid a
   registration clash; the harness maps them back.
2. Palu's group-size-4 calibration path is less exercised than group size 1: the rank-search
   output (`head_wise_ranks` per group) and the float64 whitening patch need checking on the
   first run.
3. Wide latents at chunk 128 stress the kernel's L1 rule; the factory assert fails loudly
   rather than silently.
4. The paper compares a different model revision (Llama-3 vs Llama-3.1 Instruct), 2048
   calibration samples against our 32, and perplexity at 4096 against our 2048, so the
   comparison is directional; every table carries the caveat.
5. Task evaluation runs prefill only and therefore does not see the fused kernel; decode
   fidelity remains the `decode_check` measurement.
6. **Disk.** Each full Palu HF dump is 30 GB; the box's root disk filled during the first
   G-LRD calibration (6 GB free). The dumps are now stored as factors-only files
   (`palu_lowrank.safetensors`, ~0.6 GB, same key names; `load_palu_dir` reads either form)
   and the calibration script strips the full dump right after writing it. Anything that
   writes large regenerable files must not go through the weight cache (see §3).
