# Palu on Tenstorrent — Stage 1: Functional Port and On-Device Comparison

Date: 2026-09-09
Status: design approved in conversation, pending written review

## 1. Goal and staging

Bring Palu (low-rank KV-cache compression, https://github.com/shadowpa0327/palu,
arXiv 2407.21118) to a Tenstorrent Wormhole n300, as a paper-style study in three
stages. Every stage gets its own spec and plan. This document covers stage 1 only.

| Stage | Deliverable | Depends on |
|---|---|---|
| 1 (this spec) | Palu-compressed Llama-3.1-8B-Instruct running on the n300 using existing ttnn ops; on-device comparison of original vs Palu model: perplexity, KV memory, decode/prefill speed | — |
| 2 | Fused tt-metal decode kernel that reconstructs and rotates K latents on-core, removing the per-step reconstruction cost | stage 1 harness and baseline numbers |
| 3 | Serving: smaller KV blocks in the TT vLLM backend; capacity and throughput benchmarks | stage 2 kernel (paged) |

The headline comparison is **original model on TT vs Palu(model) on TT**, both measured
through the same harness on the same device.

## 2. Palu in one paragraph

For each attention layer Palu factorises only `k_proj` and `v_proj`. Per group of KV
heads, the weight slice `W_g` (group_dim × hidden) is truncated-SVD'd into
`U_g` (group_dim × r) and `VT_g` (r × hidden). The KV cache stores the r-dimensional
latents `h = x · VT_gᵀ` instead of full K and V. At decode:

- Values need no reconstruction: `softmax · V = softmax · h_v · U_v`, and `U_v` is
  folded into `o_proj`.
- Keys must be reconstructed (`K = h_k · U_k`) and RoPE-rotated before `q · Kᵀ`,
  because RoPE is position-dependent and sits after `k_proj`. Palu's speedup comes
  from a fused GPU kernel doing this on-chip. Stage 1 does it with ttnn ops
  (matmul + RoPE) each step; stage 2 fuses it.
- Prefill reconstructs K in full and runs ordinary attention.

## 3. Environment and repository

Hardware and host: one Wormhole n300 (two chips, TP=2), Intel i7-14700 (20 cores,
AVX2 only, no AVX-512/bf16), 125 GB RAM, Ubuntu 22.04, Python 3.10, clang-20,
cmake 4.2, ninja, no sudo. **The server is shared** with several other users; the
device is world-accessible and other users' processes are not visible to us.

Rules baked into scripts and followed throughout:

- Everything lives in this repo, the user's own HF cache
  (`~/.cache/huggingface`, read-only), and the session scratch dir. Nothing under
  `/home/user01/tt-data` is read at runtime, imported, or modified.
- No sudo, no firmware/driver/hugepage changes.
- Every device run goes through `scripts/device_free.sh`, which refuses to start
  when telemetry shows the n300 busy (AICLK above idle 500 MHz or power well above
  the ~13 W idle) or when another of our jobs holds it, and fails fast rather than
  contending.
- CPU-heavy work (build, calibration, CPU perplexity) is niced and capped at
  ~10–12 threads.

Layout:

```
tt-cadlab/
  third_party/tt-metal/        git submodule, tag v0.78.0, built from source
                               (build_metal.sh, clang-20, ninja, -j10). Not edited.
  third_party/palu/            git submodule (our fork) used only for the full
                               calibration recipe (Fisher + whitening) with CPU patches
  .venv/                       gitignored; ttnn from the local build, torch cpu,
                               transformers, safetensors, datasets
  palu_tt/
    compress.py                plain-SVD per-head export -> HF folder in Palu key format
    convert.py                 HF Palu folder -> TT state dict (permutes, fuses U_v into wo)
    attention.py               PaluAttention(tt_transformers Attention)
    model.py                   build_palu_model(): class substitution + rank config
    reference.py               CPU torch Palu attention/model used as ground truth
    eval_ppl.py                on-device wikitext2 perplexity harness
    bench.py                   decode/prefill timing, KV bytes/token, max context
  scripts/
    setup_env.sh               submodules, venv, build, deps
    device_free.sh             device idle check (see rules)
    run_calibration.sh         Fisher + whitening via third_party/palu, niced
    run_ppl_cpu.sh             CPU HF perplexity (sanity check only)
  tests/
  docs/superpowers/specs/
  docs/results/                result tables produced by the harness
```

## 4. Compression and export (CPU)

Input: `meta-llama/Llama-3.1-8B-Instruct` from the HF cache (32 layers,
32 query heads, 8 KV heads, head_dim 128, hidden 4096).

Decomposition (stage 1 default): plain SVD, **group size 1** (one group per KV
head), uniform ranks. For each layer and each KV head `i`, the 128×4096 slices of
`k_proj` and `v_proj` are SVD'd and truncated to `r_k` and `r_v`; `U = U_r·√S`,
`VT = √S·VTr` as in Palu. Ranks must be multiples of 32 (device tile size; Palu rounds
to 32 as well). Study points: `r=64` (50 % of head_dim), `r=96` (75 %), and
`r_k=128, r_v=64` (V-only compression). `r_k=r_v=128` is the baseline path
(identity factorisation, used for parity tests).

Why group size 1: the MLA decode op requires K and V to have the same number of
heads, and reconstructed K is per KV head; per-head latents also shard across the
two chips exactly like the original KV heads. Palu's medium-grained grouping
(group size 4) is deferred to stage 2, where a custom kernel can index V by group.

Output format: an HF folder in Palu's own layout (`k_proj.VT.weight`,
`k_proj.U.{i}.weight`, config field `head_wise_ranks`), so that a folder produced by
Palu's full recipe (Fisher rank search + whitening, run in the background on CPU
in fp32 from `third_party/palu`) is a drop-in replacement. Non-uniform ranks from
Fisher search are supported by the converter only if every rank is a multiple of 32
and identical across KV heads within a layer (per-layer ranks may differ); other
results are rounded up to satisfy this and the rounding is logged.

**As built (deviation):** `compress.py` writes a *delta* file, not a full HF dump — `compressed/<row>/palu_attn.safetensors` (only the factorised `k_proj`/`v_proj` tensors, in Palu's key naming) plus `config.json` (`head_wise_ranks` and a `palu_tt` block recording method, group size, `rank_k`, `rank_v` and the source snapshot). Every other weight is read from the read-only HF cache, so a row costs ~0.5 GB instead of a 16 GB copy. A full Palu HF dump still loads, because `convert.py` reads the same key names from either layout.

Cost notes (measured): CPU fp32 matmul ≈ 1.3 TFLOP/s; bf16 on this CPU is ~1000×
slower, so all CPU work runs in fp32. Plain SVD export: minutes. Full recipe:
Fisher ~1.5–3 h, whitening ~3–4 h, both cached per model.

`convert.py` produces, from either folder, the tensors the TT side loads:

| Tensor | Shape (full model) | Notes |
|---|---|---|
| `wq` | 4096 × 4096 | HF `q_proj`, with tt_transformers' Meta-style head permutation, as stock |
| `w_kv_lat` | 4096 × (8·r_k + 8·r_v) | fused `[VT_k ; VT_v]`, column-blocked per KV head so TP sharding by head works |
| `U_k` | 8 × r_k × 128 | per-head up-projection; the Meta-style permutation is applied to its 128 output columns |
| `wo` | 4096 × 4096 | original, used in prefill |
| `wo_fused` | 4096 × 4096 | `wo[:, head h] · U_v[kv(h)]` per query head, used in decode. **As built:** each query head's 128-column block holds the `r_v` fused columns and is zero-padded to 128, so the tensor is always 4096 × 4096 (one device shape for every rank) |

The converter is validated by a CPU test: for random x, applying `U_k` to the latent
must equal the permuted stock K; `attn · h_v · wo_fusedᵀ` must equal
`attn · V · woᵀ`.

## 5. PaluAttention on device

Per chip (TP=2): 16 query heads, 4 KV heads, head_dim 128, batch ≤ 32.

### Decode, per layer

1. `ttnn.linear(x, wq_kvlat)` gives q (16×128) and latents `h_k` (4·r_k),
   `h_v` (4·r_v) for the new token.
2. Head split: existing `nlp_create_qkv_heads_decode` for q; reshape for the latents
   (they are already per KV head).
3. RoPE on q only (`rotary_embedding_llama`, decode mode).
4. Write `h_k`, `h_v` into the latent caches with `paged_update_cache`
   (non-paged usage, as stock). Cache shapes `[32, 4, max_seq, r_k]`,
   `[32, 4, max_seq, r_v]`.
5. K reconstruction: `K = h_k[:, :, :S] @ U_k` with `S = max(current_pos)+1`
   rounded up to 256 (**as built**; rounding to the 32-element tile changes the decode op shapes
   every 32 tokens and thrashes the program cache), then `rotary_embedding_llama` in prefill mode
   over positions 0..S-1 (cos/sin from the stock RoPE setup).
   **Deviation, as built:** there is no reused scratch buffer. ttnn ops return new tensors rather
   than writing into a caller-owned one, so every step allocates a fresh `k_rec` (bf16 matmul
   output), `k_rot` (bf16 RoPE output) and `k_att` (the bfp8 typecast the MLA op consumes), each
   deallocated as soon as the next has consumed it. At B=32 / S=8k that is ≈ 268 + 268 + 143 MB
   live per layer-step, all freed before the next layer runs — which is why the compressed rows
   allocate at 8k where stock does not, despite the reconstruction traffic.
   This is Palu's non-fused path and the cost stage 2 removes.
6. Attention: `flash_multi_latent_attention_decode(q, k_att, v=h_v cache,
   head_dim_v=r_v, cur_pos_tensor=current_pos)`. Output `[1, 32, 16, r_v]`.
7. `nlp_concat_heads_decode`, `ttnn.linear(·, wo_fused)`, stock all-reduce.

### Prefill, per layer

Same projection; reconstruct **both** K and V in full (`h @ U`), RoPE on q and K,
stock `scaled_dot_product_attention` with head_dim 128, original `wo`. Only the
latents are written to the cache (`fill_cache`). Prefill is compute-bound, so
reconstructing V costs little and avoids `flash_mla_prefill`, which takes V only as
a slice of K.

### Full-rank fast paths (as built)

A rank equal to `head_dim` is not a compression, and forcing it through the latent
machinery costs latency (K) or correctness (V). Both sides therefore short-circuit:

- **`r_k == head_dim` — stock K path.** The converter emits `reverse_permute(U_k · VT_k)`
  as the K block of the fused QKV weight (built from the *factors*, so a dump whose
  head-wise ranks were merely padded up to 128 stays truncated), and the module writes
  RoPE'd K straight into a full-width cache with the stock decode-mode rotation. No
  `latent @ U_k` matmul, no per-step reconstruction tensors, no prefill-mode RoPE over the window, no
  `recon_len` dependence. This is what makes V-only compression free (section 8's
  `r_k=128/r_v=64` row decodes at stock speed and runs at 8k, where stock does not
  allocate).
- **`r_v == head_dim` — V collapsed into the QKV weight.** On this path prefill feeds
  the V block of the QKV output directly to SDPA, the cache stores that tensor as V,
  and decode applies `wo_fused` — all of which are correct **only if `U_v == I`**. That
  is true for the identity factorisation but false for any genuine full-rank SVD or
  whitened factorisation, i.e. for most layers of a calibrated dump. The converter
  therefore emits `U_v · VT_v` as the V block, `u_v = I` and `wo_fused = wo` whenever
  `r_v == head_dim`, restoring the invariant. Violating it cost prefill PCC 0.05 and
  device perplexities of 2433/3399 before it was found.

Because both changes alter the *content* of tensors whose name, shape and dtype are
unchanged, and ttnn's on-disk weight cache keys only on those, every Palu cache file
name carries a converter version (`CONVERT_VERSION`) and a weight-set id.

### Risks and fallbacks

- **MLA decode op with GQA.** Upstream exercises it with one KV head; we use 4 KV
  heads and 16 query heads. If it fails validation or PCC, fallback: reconstruct V
  per step like K and use stock `scaled_dot_product_attention_decode`. Capacity
  saving stays; V bandwidth saving is lost (documented in results).
- **Reconstruction cost.** At batch 32, context 16k, step 5 writes ~1 GB of K per
  layer per step. Expected; it is the stage 2 motivation. Reconstruction is bounded
  by the longest active sequence, never `max_seq`.
- **Latent dtype — resolved, fallback struck.** Caches keep the base model's KV dtype (bfp8).
  Moving the latents to bf16 was tested and does **not** help: prefill never reads the latent cache
  (it reconstructs in flight), so the prefill PCC is unchanged, and the decode teacher-forced
  agreement moved by one token. The `r=64` fidelity gap (prefill logit PCC 0.963 against the fp32
  CPU low-rank reference, versus 0.999 for stock against its own) lives in the reconstruction path
  — the bf16/HiFi2 `latent @ U` matmul plus bfp8 requantisation. The lever, if a later stage needs
  tighter fidelity, is higher matmul fidelity (HiFi4 / fp32 accumulate) on the two reconstruction
  matmuls, not the cache dtype.

### Not in stage 1

Paged cache (per-step reconstruction over a paged pool would reconstruct the whole
pool), group size > 1, latent quantisation, tracing of the decode step (S varies).

## 6. Loading, cache allocation, injection

- `PaluAttention.__init__` mirrors the stock class, taking the converted state dict.
  Device weights: `wq_kvlat` column-sharded across the two chips by head;
  `U_k` `[4, r_k, 128]` per chip in bf16 (its error enters scores directly);
  `wo` and `wo_fused` row-sharded like stock `wo`. Other dtypes follow the model's
  optimisation settings.
- Latent caches replace `layer_past`. KV bytes per token per layer per chip go from
  `4 · 128 · 2` (K and V, 1 byte each in bfp8) to `4 · (r_k + r_v)`.
- `build_palu_model()` constructs tt_transformers' `Transformer` with the attention
  class swapped for `PaluAttention` and the converted state dict, reading ranks from
  the Palu config. Baseline (`r=128`) goes through the same function. If class
  substitution needs a hook in the decoder, it is a minimal patch file applied to the
  submodule by `setup_env.sh` and recorded here.
- Generator, demo, RoPE setup, CCL are reused unchanged.

## 7. Tests (correctness, not headline results)

1. CPU: SVD export round-trips; per-head reconstruction error decreases with rank;
   converter shapes; permutation correctness; `wo_fused` equivalence (section 4).
2. Device, attention block: `PaluAttention` decode and prefill vs `reference.py`
   for one layer with real ranks, PCC ≥ 0.99. First milestone; surfaces the MLA/GQA
   and dtype risks. **As built**, the parametrisation must cover all four combinations
   of the full-rank fast paths: `identity` (fast K + fast V), `r64` (reconstruct both),
   `rk128_rv64` (fast K + reconstruct V) and `rk32_rv128` (reconstruct K + fast V) —
   the last is the only one that exercises a genuine, non-identity `U_v` on the fast-V
   path, and it is the case that was wrong on device until it was added (PCC 0.05 →
   0.9996). The prefill test must also assert the **contents of the K cache**, not only
   the block output: at the tests' input scale the softmax is nearly uniform, so output
   PCC stayed at 0.9994 with a completely wrong K cache.
3. Device, baseline parity: the `r=128` path must match stock tt_transformers
   outputs (PCC ≥ 0.999 on logits), proving the plumbing adds no error. **Measured: 0.9994**
   prefill logit PCC identity-vs-stock, with 32/32 identical teacher-forced tokens. Since the
   full-rank fast paths landed, the identity attention block is **bit-identical** to stock
   (max-abs difference 0.0 on caches and outputs), which is why the 32/32 equality is safe to
   assert; before that it sat on knife-edge logit ties and was not a fidelity measurement.
4. Device, model: full 8B Palu model vs the CPU Palu model on a ~128-token prompt
   plus 32 greedy tokens; logit PCC and top-1 match reported, compared against the
   same numbers for stock-vs-HF so the acceptance bar is "no worse than
   tt_transformers' own fidelity". **Bars as built:** prefill logit PCC ≥ 0.95 against the fp32
   CPU low-rank reference and teacher-forced token agreement ≥ 0.85 (`r=64` measured 0.963 and
   0.906; the stock control measured 0.999 and 0.969). The originally written 0.99 PCC and 0.75
   *free-running* greedy match were both wrong: 0.99 is unreachable even for stock against an fp32
   reference at bfp8 weights, and free-running greedy diverges after the first differing token, so
   it measures divergence time rather than per-step fidelity (stock itself scores 0.19 on it).

## 8. Evaluation: TT baseline vs TT Palu

All headline numbers are measured on the n300 through `palu_tt/eval_ppl.py` and
`palu_tt/bench.py`. **Rows as built (seven):** `baseline` (stock attention), `identity`
(the Palu path at `r=128`), the uncalibrated plain-SVD rows `svd_r96`, `svd_r64` and
`svd_rk128_rv64`, and the two calibrated rows from Palu's full recipe (Fisher-uniform
rank search + whitening, group size 1) at target ratios 0.5 and 0.7 — `palu_full_50`
and `palu_full_70`. The calibrated dumps have **per-layer** ranks, so the results table
reports a range and a mean, and the KV-bytes column is the mean over the 32 layers
(`bench` reports layer 0). Stage 2's kernel is a later row.

| Metric | Method |
|---|---|
| Wikitext2 perplexity (device) | Model forward in "all positions" mode on 2048-token windows; NLL on host from full logits (or from final hidden states with the HF `lm_head` in fp32 if the device path returns hidden states). CPU HF perplexity run once per row as a sanity check that the device harness agrees with HF within noise. |
| KV cache bytes/token | From allocated latent caches; cross-checked with device memory reports |
| Max context at batch 32 | Largest `max_seq` that allocates |
| Decode latency | Per step at batch 32, contexts 1k/4k/8k, plus the max-context probe. **As built:** 16k is not measured — it does not allocate at batch 32 for any row (stock and the identity path stop at 4k), so the sweep is 1k/4k/8k and the ceiling is reported separately. Whole step and attention-only (a separate single-layer bench, not the op profiler) |
| Prefill time | TTFT at 2k and 8k tokens, batch 1 |
| Generation samples | Same prompts, greedy, side by side |

Results go to `docs/results/stage1.md` with the exact commit, ranks, dtypes and
device state recorded.

## 9. Success criteria

- Tests 1–4 pass; the `r=128` path is indistinguishable from stock.
  **Met** — identity 7.255 vs stock 7.254 perplexity over 141 windows, prefill logit
  PCC 0.9994, 32/32 greedy tokens, and the attention block bit-identical to stock.
- All rows produce a complete results table on device. **Met** — seven rows.
- Perplexity degradation at `r=64` is in the same range as Palu reports for 50 %
  compression without calibration (their ablation baseline); with calibrated weights
  it should approach their headline numbers. **Partly met** — uncalibrated plain SVD is
  catastrophic here (`r64` 703.6, `r96` 79.4), and the calibrated recipe recovers the
  model: 12.93 at 52.7 % of stock KV bytes and 9.03 at 69.9 %, against the paper's
  per-head (M-LRD) 12.38 vs its own 8.28 baseline at 50 %. Our 50 % row is a 1.78x
  degradation against the paper's 1.50x; closing that gap needs group size > 1 (the
  paper's G-LRD reaches 10.11), which is out of scope for stage 1.
- **V-only compression must not cost decode latency (restated as measured).** The
  criterion as originally written was ambiguous: a rank-128 K that is still *stored as a
  latent and reconstructed every step* pays the full reconstruction cost, and the first
  measurement of `svd_rk128_rv64` was 105.5/278.3/500.1 ms — far above stock. With the
  `r_k == head_dim` fast path (section 5) the row keeps K in the stock cache, and the
  criterion is **met**: decode min 38.7 / 52.0 / 69.2 ms at 1k/4k/8k against stock
  38.0 / 55.4 / OOM (faster at 4k, since the V cache is 25 % smaller; and it runs at 8k,
  where stock does not allocate), with attention-only minima 1.52 / 1.27 / 1.85 against
  stock 0.81 / 1.31 / 2.07.
- **The calibrated recipe must reproduce on device, not just on CPU (added).** The device
  perplexity of each calibrated row must agree with the fp32 CPU low-rank reference on the
  same corpus to within the spread the harness already shows on the stock row (~2.0 % from
  the window count alone: device **7.400** over the same 20 windows the CPU column uses against
  7.254 over all 141 — `docs/results/raw/ppl_baseline_20w.json`). **Met**: `palu_full_50` device
  12.931 vs CPU 12.759 (1.3 %),
  `palu_full_70` device 9.028 vs CPU 9.059 (0.3 %).
- The full compression rows quantify the reconstruction cost that stage 2 targets.
  **Met** — 2.3–2.6x the stock decode step at 1k and 4.7–5.1x at 4k, with 32 x the
  attention-only time accounting for essentially the whole step.
