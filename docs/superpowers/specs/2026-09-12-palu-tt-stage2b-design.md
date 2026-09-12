# Palu on Tenstorrent — Stage 2b: G-LRD and Task Accuracy

Date: 2026-09-12
Status: approved in conversation 2026-09-12, pending written review
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
  (`u_k_prefill` stays `[1, n_kv, 128, 128]`-shaped per head; the latent is broadcast to the
  heads with a `repeat`/expand before the matmul). The absorbed-V trick and `wo_fused` are
  unchanged: each query head already carries its own `U_v` slice.
- **Naive decode** reconstructs K from the group latent per head the same way (`u_k_decode`
  per head, latent expanded over heads), so the G-LRD rows have a naive timing row too.
- **Group size 1** is the same code with `n_groups = n_kv`; the stage-1/2 rows must stay
  bit-identical (tests assert the converter output and the device caches).

## 4. G-LRD in the fused kernel

Three changes to `palu_sdpa_decode`:

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
- **Tests.** Op-level: `heads_per_group = 4` with `r_k ∈ {128, 256, 384}` against the fp32
  reference and the stock MLA op on host-rotated K; `heads_per_group = 1` unchanged.
  Attention-level: naive-vs-fused on a G-LRD dump, byte-identical caches.

**Expected decode behaviour, stated up front.** Each of the four head work-items rebuilds its
head from the full group latent, so per position the reconstruction arithmetic is 4x
M-LRD's and the latent is read four times: 1024 elements per position at ratio 0.5 versus
M-LRD's 256 and stock's 512. G-LRD decode will be slower than M-LRD and probably slower than
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
Device time: ~3 h task evaluation, ~1.5 h G-LRD benchmarks, ~1 h fidelity and capacity. The
results page (artifact) gets the same tables.

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
