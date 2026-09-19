# Audit C: model path

## Scope read

Repo files (read in full unless noted):

- `palu_tt/attention.py`: 1-740 (whole file).
- `palu_tt/convert.py`: 1-312 (whole file).
- `palu_tt/fused.py`: 1-296 (whole file).
- `palu_tt/model.py`: 1-193 (whole file).
- `palu_tt/compress.py`: 1-58; `palu_tt/uniform_ranks.py`: 1-50; `palu_tt/latent_stats.py`: 1-68; `palu_tt/generate.py`: 1-62; `palu_tt/hf.py`: 1-43; `palu_tt/device.py`: 1-28 (whole files).
- `palu_tt/eval_ppl.py`: 1-151; `palu_tt/eval_tasks.py`: 1-156; `palu_tt/reference.py`: 1-89; `palu_tt/decode_check.py`: 1-138 (whole files — needed for the CLI defaults and the prefill/decode instrument question).
- `palu_tt/bench.py`: 1-60, 100-215, 240-275, 285-330, 425-442 (only the parts that build a Palu model / set `recon_len` / build `FUSED`). **Skimmed/skipped**: the JSON-writing and context-scan bodies (another auditor's area).
- `palu_tt/report.py`: 1145-1190, 1267-1280, 1360-1460 only (`_row_is_blind`, `frontier_rows`, the frontier prose). **Skipped** the rest.
- `palu_tt/plots.py`: lines 16, 288-294 only (the `blind` filter). **Skipped** the rest.
- `tests/test_convert.py`: 1-492; `tests/test_fused.py`: 1-235; `tests/test_compress.py`: 1-57; `tests/test_uniform_ranks.py`: 1-29; `tests/test_attention_tag.py`: 1-39; `tests/test_attention_device.py`: 1-474; `tests/test_model_device.py`: 1-97; `tests/test_kv_cache_bfp4_device.py`: 1-37; `tests/conftest.py`: 1-22 (whole files).
- `tests/test_bench_utils.py`: 235-279 only (the `apply_palu_flags` coverage test).
- **Skipped entirely**: `tests/test_palu_sdpa_device.py` (the op-level tests — that is the kernel auditor's area), `tests/test_report.py`, `tests/test_plots.py`, `tests/test_lm_eval_tt.py`, `tests/test_reference.py`, `tests/test_profile_*.py`, `tests/test_patch_palu_cpu.py`, `tests/test_kernel_figures.py`, `tests/test_report_figures.py`.

Docs: `README.md` 1-30, 85-230, 290-300; `docs/report_palu_on_tt.md` 1-70 (§1-§4); `docs/results/stage1.md` 1-30, 145-175, 320-381; `docs/results/stage2b.md` 1-10 plus a grep of every G-LRD/group line; `docs/results/stage2f.md` 1-164 (whole file); `docs/superpowers/specs/2026-09-14-palu-tt-stage2f-design.md` §0b only (grep); `docs/superpowers/specs/2026-09-09-palu-tt-stage1-design.md` — **skimmed via grep only** (`recon_len`), not read end to end.

Stock reference: `third_party/tt-metal/models/tt_transformers/tt/attention.py` 250-470, 620-710, 1020-1330; `.../tt/load_checkpoints.py` 440-500, 885-920; `.../tt/rope.py` 30-80, 225-265, 310-390; `.../tt/model_config.py` 730-760, 3150-3200, 4700-4835; `third_party/tt-metal/models/common/tensor_utils.py` 70-90; `ttnn/cpp/.../sdpa_decode/sdpa_decode.cpp` 18-40; `ttnn/cpp/.../palu_sdpa_decode/palu_sdpa_decode.cpp` 55-130.

Data checked (read-only): safetensors header of `compressed/palu_full_0.7/palu_lowrank.safetensors`; `config.json` of 11 dumps; all 79 `ppl_*.json` and 9 `tasks_*.json` in `docs/results/raw/`; a listing of `model_cache/`.

Device / ttnn: **nothing was executed.** I did not run pytest at all: `tests/conftest.py:7` does `from palu_tt import device as dev` and `palu_tt/device.py:4` is `import ttnn`, so any `pytest tests/...` imports ttnn into the process, which the brief forbids. Instead I re-typed `reverse_permute`, `get_rot_transformation_mat` and `convert.palu_layer_tensors` into a standalone torch script and checked the whole prefill/decode algebra numerically against an independent HF-convention reference (`scratchpad/audit/math_check.py`, plus a mutation control `math_check_mut.py`).

**Independent numeric check (the result behind Finding "math verified" below).** Six shapes, TP=2 emulated, `NH=16, NKV=8, HD=32, DIM=256`, max relative error against the reference:

| case | gs | r_k | r_v | prefill | decode |
|---|---|---|---|---|---|
| M-LRD low rank | 1 | 8 | 16 | 7.1e-07 | 6.8e-07 |
| V-only fast K (r_k == hd) | 1 | 32 | 16 | 2.2e-06 | 2.1e-06 |
| fast V (r_v == hd, U_v != I) | 1 | 8 | 32 | 1.1e-06 | 1.1e-06 |
| identity-shaped (both == hd, U != I) | 1 | 32 | 32 | 1.1e-06 | 1.1e-06 |
| G-LRD gs=2 | 2 | 16 | 48 | 3.5e-07 | 3.5e-07 |
| G-LRD gs=4, one group/chip | 4 | 32 | 96 | 4.8e-07 | 4.8e-07 |

Mutation control: changing `kv = qh // rep` to `kv = qh % n_kv_heads` in the `wo_fused` fold drives the decode error to 1.2 on the low-rank-V rows, so the check is not vacuous.

---

## Findings

### [Critical] The `identity` perplexity row and the "bit-identical to stock" claim cannot both be true

**File:line** — `docs/results/raw/ppl_identity.json` (ppl 7.2552432062377985, nll 571979.1708068848) vs `docs/results/raw/ppl_baseline.json` (7.253966155267361, nll 571928.3628845215); claim at `README.md:9`, `docs/results/stage1.md:10` and `:158-160`, `docs/report_palu_on_tt.md:59`; code at `palu_tt/attention.py:143-153` (`recon_k`/`recon_v`), `:268-271` (`wo_fused` alias), `palu_tt/convert.py:258-286` (`fast_v` collapse).

**What** — For the identity factorisation (`palu_dir=None`) every fast path fires: `r_k == r_v == head_dim`, `gs == 1`, so `recon_k` and `recon_v` are both `False`, the converter writes `reverse_permute(U_k @ VT_k)` = the stock Meta-permuted `wk` and `U_v @ VT_v` = the stock `wv` into the fused QKV weight, `u_v = I`, and `self.wo_fused` is aliased to `self.wo`. I walked `PaluAttention.forward_prefill` (`attention.py:336-513`) against `Attention.forward_prefill` (`third_party/.../attention.py:1020-1318`) op by op for that configuration: same pad/reshape rule, same `minimal_matmul`/`linear` branch, same `tt_all_reduce(cluster_axis=1, dtype=ccl_dtype)`, same `nlp_create_qkv_heads`, same `rotary_embedding_llama` with `transformation_mats["prefill"]`, the same two typecasts to `keys.dtype`/`values.dtype` feeding SDPA (`attention.py:487-489`, with `faithful and not recon_k` selecting exactly `keys.dtype`), the same SDPA program config and scale, and the same `nlp_concat_heads` → `linear(wo)` → `tt_all_reduce(cluster_axis=0, dim=3)` tail. I found no op, dtype, memory config or CCL argument that differs. The only Palu-specific difference in prefill is the K/V *cache-fill* shard config (`attention.py:325-334`), which cannot change the SDPA inputs.

Perplexity is a prefill-only measurement (`eval_ppl.device_ppl` → `generate.prefill_hidden_states`, `generate.py:204-213`), so the decode differences (`flash_multi_latent_attention_decode` instead of `scaled_dot_product_attention_decode`) cannot enter this number. Therefore, with today's code, `identity` should equal `baseline` to the last digit. It does not: 50.8 nats over 288,627 tokens.

So exactly one of these is true:
1. `ppl_identity.json` is a stale measurement of the pre-Task-12 code (before `03e52da` "V-only fast path" and `b3d9afb` "Collapse full-rank V into the QKV weight"), in which case stage1.md's headline `identity 7.255` row and report §3's "reproduces the baseline to three decimals (7.255 vs 7.254) … which is how the port itself was validated" describe a code path that no longer exists; or
2. the current code is **not** bit-identical, and `stage1.md:159-160` ("after Task 12's full-rank shortcuts the identity attention block is bit-identical to stock (max-abs difference 0.0 on caches and outputs)") and `README.md:9` are wrong.

Nothing in the repo settles it: there is no artefact behind the "max-abs difference 0.0" claim, and no test asserts it (see the next finding). stage1.md is internally inconsistent on its own terms — it reports prefill logit PCC **0.9994** and "bit-identical" in the same paragraph.

**Why it matters** — This is the row that validates the whole port; it is quoted in the external report (`report_palu_on_tt.md:59`). If (1), a presented number describes dead code. If (2), the port's central correctness claim is false as stated.

**How to verify** — One gated device run, ~40 min:
`scripts/run_device.sh python -m palu_tt.eval_ppl --row identity_recheck --kind palu` and compare `docs/results/raw/ppl_identity_recheck.json`'s `nll` against `ppl_baseline.json`'s `571928.3628845215`. Equal to the last bit ⇒ the committed row is stale and must be re-measured. Not equal ⇒ drop "bit-identical" from README/stage1/report and state the measured PCC instead.

**Suggested fix** — Re-measure `identity` with the current code and replace the row; and in `tests/test_attention_device.py` change `test_identity_prefill_equals_stock_attention` (line 95) from `comp_pcc(..., 0.999)` to `torch.equal(ys[0], ys[1])` (or assert `(ys[0]-ys[1]).abs().max() == 0`), which is what the docs claim. If bit-equality does not hold, weaken the docs to the measured bound.

---

### [Important] No test asserts the bit-identity the docs claim; both identity-vs-stock tests bar at PCC 0.999

**File:line** — `tests/test_attention_device.py:95-110` (`test_identity_prefill_equals_stock_attention`), `:156-173` (`test_identity_decode_equals_stock_attention`), `tests/test_model_device.py:23-34` (`test_identity_path_matches_stock_model`, `comp_pcc(..., 0.999)` + greedy-token equality).

**What** — `stage1.md:159` states "max-abs difference 0.0 on caches and outputs". The three tests that compare the identity path against stock all use `comp_pcc` with a 0.999 bar; none compares tensors for equality, and none compares the *caches* at all (the cache-equality assertions in this file, `test_attention_device.py:227-228` and `:308-309`, compare naive-vs-fused Palu builds, not Palu-vs-stock). A regression that moved the identity path off stock by ~1e-3 relative would pass every one of them — which is exactly the size of the discrepancy in the committed `ppl_identity` row.

**Why it matters** — The claim that "whatever the compressed rows lose is lost in the factorisation, not in the port" (`stage1.md:161`) rests on a measurement no test reproduces.

**How to verify** — `grep -n "torch.equal\|comp_pcc" tests/test_attention_device.py tests/test_model_device.py`; every identity-vs-stock comparison is `comp_pcc(..., 0.999)`.

**Suggested fix** — Add the stock-vs-Palu cache comparison and switch the identity comparisons to exact equality (see previous finding).

---

### [Important] The stage-1/2b/2c perplexity and task rows were measured with a prefill the current default no longer runs, and no documented command reproduces them

**File:line** — `palu_tt/convert.py:58` (`latent_quant_in_prefill: bool = True`), `palu_tt/model.py:57` (`build_model(..., latent_quant_in_prefill: bool = True)`), `palu_tt/eval_ppl.py:96-99` and `palu_tt/eval_tasks.py:264-267` (`--no-latent-quant-in-prefill` is `store_false`, so the CLI default is **on**); the affected prefill block is `palu_tt/attention.py:424-444` and the SDPA casts at `:483-489`.

**What** — `latent_quant_in_prefill` round-trips the prefill latents through the cache dtype (`_roundtrip`, `attention.py:24-34`) before the up-projection, and on the fast-path sides makes the SDPA cast follow `keys.dtype`/`values.dtype` instead of `self.kv_cache_dtype`. It is the only switch in `PaluRuntime` that defaults **on**, and it did not exist when stages 1, 2b and 2c were measured. The raw JSONs confirm the split: every 141-window row from those stages carries `"latent_quant_in_prefill": null` and `"latent_dtype": null` (`ppl_baseline`, `ppl_identity`, `ppl_palu_full_50/70`, `ppl_glrd_50/70`, `ppl_glrd_*_f256`, `tasks_baseline`, `tasks_palu_full_50/70`, `tasks_glrd_*`), while every stage-2f row carries `true`.

`README.md:96-97`, `:168-172`, `:179`, `:196-197` give the reproduction commands for exactly those rows, and none passes `--no-latent-quant-in-prefill`. Running them today measures a different numeric path. `stage1.md` and `stage2b.md` carry no note about it — even though `stage2b.md:5` does carry the equivalent note for the stage-2e rotation defaults ("reproduce them with `--rot-mode legacy --table-mcast off`").

**Why it matters** — Every headline quality number in stage 1 and stage 2b (`baseline` 7.254, `palu_full_70` 9.028, `glrd_70_f256` 8.135, `avg5` 65.44 / 61.50 / 64.53 …) is irreproducible from the repo as documented.

**How to verify** — `python3 -c "import json;print(json.load(open('docs/results/raw/ppl_palu_full_70.json')))"` (no `latent_quant_in_prefill` key) against `ppl_palu_full_70_w40.json` (`true`); then `grep -n "eval_ppl\|eval_tasks" README.md` — no `--no-latent-quant-in-prefill` anywhere.

**Suggested fix** — Add the same one-line note `stage2b.md:5` already uses, to `stage1.md`, `stage2b.md` and `stage2c.md`, and append `--no-latent-quant-in-prefill` to the README repro blocks for those rows (or re-measure them faithfully).

---

### [Important] stage 2f's headline "cost of bfp4" deltas compare a faithful row against a blind one, and the task-metric half has no measured correction at all

**File:line** — `docs/results/stage2f.md:84-97` (the recommendation), `:108-118` ("Stock bfp4 against Palu at equal bytes"), `:128-130`; raw rows `ppl_palu_full_70_lv4.json` (`lqip=true`, 9.2369) vs `ppl_palu_full_70.json` (`lqip=null`, 9.0277); `tasks_palu_full_70_lv4.json` (`lqip=true`, avg5 61.08) vs `tasks_palu_full_70.json` (`lqip=null`, avg5 61.50).

**What** — The recommendation reads "vs `palu_full_70` (47.55 KB, 9.028, 61.50, 56.9 ms): V moved to bfp4 — 32 % fewer bytes at 0.99x the decode step for **+0.209 ppl and 0.42 points**." Both deltas subtract a cache-blind prefill measurement from a cache-faithful one. The document's Caveats (`:160-162`) disclose this for perplexity and bound it at −0.033, so the true ppl delta is nearer +0.24 — the stated figure understates the cost of bfp4-V by ~15 %, and the correction is never applied. For the five-task mean there is no disclosure and **no measurement anywhere in the repo** of what the bfp8 round trip does to `avg5`, so "0.42 points" has an unquantified bias of unknown sign. The same mixing runs through §"Stock bfp4 against Palu at equal bytes" (`baseline_kv4`, faithful by construction — stock prefill casts K/V to `keys_BKSD.dtype` before SDPA at `third_party/.../attention.py:1120,1127,1232-1238` — against `palu_full_50`, blind).

**Why it matters** — These are the numbers the recommendation and the "format-only compression wins" conclusion are argued from.

**How to verify** — Compare the `latent_quant_in_prefill` field of the two JSONs behind each quoted pair (script in the Scope section, or `grep latent_quant docs/results/raw/tasks_palu_full_70*.json`).

**Suggested fix** — Re-measure the four bfp8 reference rows (`palu_full_50/70`, `glrd_50/70_f256`) faithfully at 141 windows and at `avg5`, or annotate every cross-instrument delta in §"Findings" the way the table's `blind` cells already are.

---

### [Important] The −0.033 bfp8 round-trip shift — the constant that licenses mixing the two instruments — has no artefact in the repo

**File:line** — `palu_tt/report.py:1274` (docstring of `_row_is_blind`) and `:1447-1449` (the generated prose), rendered into `docs/results/stage2f.md:69` and `:162`.

**What** — `_row_is_blind` deliberately renders a pre-2f **bfp8** row as a number rather than `blind`, justified by "the bfp8 round trip was measured at -0.033 ppl (2f Task 1b)". I searched every JSON in `docs/results/raw/`: exactly one row has `latent_quant_in_prefill: false` (`ppl_palu_full_50_l4.json`, 141 windows, bfp4, and identical to its bfp8 sibling by construction). There is no blind/faithful bfp8 pair anywhere, so −0.033 cannot be recomputed from the repo; it is a hard-coded literal in the report generator.

**Why it matters** — That constant is the sole quantitative basis for (a) showing pre-2f bfp8 rows as numbers in the stage-2f frontier table beside faithful bfp4 rows, and (b) the caveat that bounds the previous finding.

**How to verify** — `python3 -c "import json,glob; print([f for f in glob.glob('docs/results/raw/*.json') if json.load(open(f)).get('latent_quant_in_prefill') is False])"` → one file, bfp4.

**Suggested fix** — Run one `--no-latent-quant-in-prefill` bfp8 row against its faithful sibling at the same window count, commit the pair, and have `report.py` compute the shift from those two JSONs instead of quoting a literal.

---

### [Important] The round-trip idempotency that makes "prefill and decode see one latent" true is asserted only for bfp8, never for bfp4

**File:line** — `tests/test_attention_device.py:449-452`; the code it guards is `palu_tt/attention.py:424-444` plus the cache write at `:466-473`.

**What** — `test_prefill_sees_the_latent_cache_format` builds six variants including `bfp4` and `bfp4_blind`, captures both caches for every one of them (`caches[name] = [...]`, line 438), and then asserts idempotency only for the bfp8 pair:
```
for i, side in ((0, "K"), (1, "V")):
    assert torch.equal(caches["bfp8"][i], caches["bfp8_blind"][i]), f"{side} cache moved under the round trip"
```
The `bfp4` / `bfp4_blind` caches are already in hand and are never compared. The test's own comment says idempotency "is what makes 'prefill and decode see one latent' true — without it prefill would attend over q(x) while the cache held q(q(x))" — and bfp4 is the format the entire stage-2f frontier (every `_l4`, `_lk4`, `_lv4` row) is scored in.

**Why it matters** — If bfp4 quantisation were not idempotent on the r-slice (e.g. if a rank were ever not a multiple of 32, or a future shared-exponent block size crossed the slice), every bfp4 perplexity would describe a latent the cache does not hold, and this test would still pass.

**How to verify** — `sed -n '445,455p' tests/test_attention_device.py`: only `"bfp8"`/`"bfp8_blind"` appear in the equality loop.

**Suggested fix** — One line: iterate the loop over `("bfp8", "bfp8_blind")` and `("bfp4", "bfp4_blind")`.

---

### [Important] `test_apply_palu_flags_writes_every_paluruntime_switch…` supplies the very key it is supposed to prove `bench.FUSED` carries

**File:line** — `tests/test_bench_utils.py:244-279`; `palu_tt/bench.py:63-65` (`FUSED`), `palu_tt/model.py:141-168` (`apply_palu_flags`).

**What** — The docstring says: "`bench.FUSED` — the dict every `build_model` / `_attention_one` caller splats — must set every `PaluRuntime` switch, or a switch added to the dataclass without a matching `apply_palu_flags` parameter would silently keep its class default". The body then does
```
flags = dict(bench.FUSED, use_fused_k=True, ..., latent_quant_in_prefill=False, latent_dtype="bfp4", ...)
```
i.e. it *adds* `latent_quant_in_prefill` before splatting. `bench.FUSED` itself does not contain that key (`bench.py:63-65`), so the property the docstring claims to guard is false today and the test cannot detect it. The same overstatement is in `apply_palu_flags`' own docstring (`model.py:149-152`: "a new switch is either handled for both or a `TypeError` for both").

**Why it matters** — It is a test that asserts against a fixed-up copy of the object under test. No reported number is affected (bench's decode/attention/maxctx modes never run prefill, and `bench_prefill` passes `latent_quant_in_prefill=False` explicitly at `bench.py:254`), but the guard the repo believes it has is not there.

**How to verify** — `python3 -c "from ast import *" ` not needed: `grep -n "latent_quant_in_prefill" palu_tt/bench.py` returns only line 254 (the explicit prefill override), never the `FUSED` dict.

**Suggested fix** — Assert `set(bench.FUSED) | {"group_heads"} ⊇ switch_fields` directly, and add `latent_quant_in_prefill` to `bench.FUSED` (keeping `bench_prefill`'s explicit `False`).

---

### [Important] A reported row is tied to its factorisation only by a path into an untracked directory

**File:line** — `.gitignore:3` (`compressed/`), `palu_tt/convert.py:202-224` (`palu_dir_cache_id` / `palu_cache_id_for`), `palu_tt/eval_ppl.py:144-146` and `palu_tt/eval_tasks.py:294-301` (the result-JSON block).

**What** — `palu_dir_cache_id` computes a weight-set id from `config.json`'s bytes plus each shard's `name:size:mtime_ns`. That id is used for the device weight-cache key (correct, and it is what fixed the 2397.32 collision described in `stage1.md:321-330`) but it is **never written to any result JSON**: `grep -l cache_id docs/results/raw/*.json` returns nothing, while 244 files record only `"palu_dir": "compressed/palu_full_0.7"`. `compressed/` is gitignored, so the factors themselves are not in the repo. Two further consequences of keying on `mtime_ns`: the id is not stable across a copy/restore of the tree (so it cannot be used after the fact to check *which* bytes a row used), and the README already records the known collision window ("regenerating a dump in place within the same clock tick, at identical sizes").

**Why it matters** — Reproducibility and provenance: nothing in the committed artefacts ties `ppl_palu_full_70 = 9.028` to a specific set of factors, and the factors cannot be obtained from the repo (only re-derived by a ~9 h calibration).

**How to verify** — `git ls-files compressed | wc -l` → 0; `grep -l cache_id docs/results/raw/*.json | wc -l` → 0.

**Suggested fix** — Write `cfg.cache_id` (and the dump's `head_wise_ranks` hash) into every result JSON from `eval_ppl`, `eval_tasks`, `bench` and `decode_check`; switch `palu_dir_cache_id` to hashing shard *content* (or at least the safetensors header) rather than mtime.

---

### [Minor] `recon_len` is caller-supplied global state and the guard against it being stale is off by default

**File:line** — `palu_tt/attention.py:518-530` (`_recon_len`), `palu_tt/model.py:171-179` (`set_recon_len`), `palu_tt/convert.py:28-29` (`recon_len`, `check`).

**What** — On the naive K path the number of reconstructed cache rows comes from `self.palu.runtime.recon_len`, which the *caller* must set before every decode step (`generate.decode_step:220`, `bench.py:123,199`, `profile_decode.py:54`). If a caller forgets, the field keeps the previous step's value; a value smaller than `max(current_pos)+1` silently hands the decode op a K/V tensor shorter than the positions it is told to attend. The assertion that would catch it (`attention.py:527-529`) only runs when `runtime.check` is True, which no CLI sets and which `bench.py:9` explicitly documents as off for timing. Every shipped caller does set it, so nothing reported is affected.

I verified the rest of the `_recon_len` contract and it is correct: `S = min(ceil(recon_len/64)*64, max_seq_len)` is applied on exactly one path (`attention.py:660`, the naive `recon_k` branch) and nowhere else (the fused path passes the whole cache and pins its chunk explicitly, `:628-640`; the stock-K branch uses `S = max_seq_len`, `:689`). It does **not** change the number of positions attended — `flash_multi_latent_attention_decode` masks by `cur_pos_tensor` — and the 64 rounding does defeat the chunk-32 defect: `ttnn`'s `get_chunk_size` (`sdpa_decode.cpp:22-36`) is `min(512, s & -s)`, so a 64-multiple `S` can never yield a 32-row chunk. `fused.chunk_size` (`fused.py:53-58`) reproduces that formula exactly.

**Why it matters** — Latent hazard for any future harness that drives decode directly; not a current numeric error.

**How to verify** — `grep -rn "set_recon_len\|runtime.check" palu_tt/` — four setters, one guard, guard default `False`.

**Suggested fix** — Make `_recon_len` assert unconditionally against a cheap host-side copy of `current_pos` kept by `set_recon_len`, or have `set_recon_len` stamp a monotone step counter the attention can check.

---

### [Minor] `--kv-dtype bfp4` on `--kind palu` penalises prefill with a quantisation the fused decode never performs

**File:line** — `palu_tt/attention.py:487-489` (`k8`/`v8` cast to `self.kv_cache_dtype` on reconstructed sides) and `:678,690` (the naive decode's `typecast(..., self.kv_cache_dtype)`); flag help at `eval_ppl.py:87-92`.

**What** — On a reconstructed side the prefill SDPA sees K/V cast to the model's `KV_CACHE` precision. That mirrors stock and mirrors the *naive* decode. The *fused* decode (`attention.py:628-640`) never materialises K and never applies `kv_cache_dtype` at all, so a `--kind palu --kv-dtype bfp4 --fused p2` row would score a prefill penalty its own decode path does not pay. No reported row uses that combination — every `palu` row in `docs/results/raw` has `kv_dtype: "bfp8"` or null — so nothing published is affected.

**Why it matters** — A future frontier sweep that varies `--kv-dtype` on the Palu path would produce a number that does not describe the shipped kernel.

**How to verify** — `python3 -c "import json,glob;print({json.load(open(f)).get('kv_dtype') for f in glob.glob('docs/results/raw/ppl_*.json') if json.load(open(f)).get('kind')=='palu'})"` → `{None, 'bfp8'}`.

**Suggested fix** — Reject `--kv-dtype bfp4` together with `--kind palu --fused p1|p2`, or document the asymmetry in the flag help.

---

### [Minor] `report_palu_on_tt.md` §4 describes a cache layout the calibrated dumps mostly do not use

**File:line** — `docs/report_palu_on_tt.md:67-69`; code at `palu_tt/attention.py:143-153`, `:466-473`, `palu_tt/convert.py:258-274`.

**What** — §4 says "only the latents are written to the caches (`fill_cache`)" and "caches of shape `[32, 4, max_seq, r]`". Both are wrong for a substantial share of the shipped rows. From the dumps' own `config.json`: `palu_full_0.7` has `r_v == 128` in **25 of 32 layers** (so `recon_v` is False and the V cache holds full V, with `U_v` folded into the QKV weight) and `r_k == 128` in 1 layer (stock K path: the cache holds *rotated full K*, not a latent); `palu_full_0.5` has 18 such V layers; `uni_rk*_rv128` has 32. And for every G-LRD row the cache is `[B, 1, max_seq, r]`, not `[B, 4, ...]` (`attention.py:274-283`, `n_local_groups == 1`). Byte accounting is unaffected (`r == head_dim` costs the same as a head).

**Why it matters** — A reader of the external report will mis-picture what `palu_full_0.7` actually stores; it also obscures that 1 of 32 layers never enters the fused kernel at all (`attention.py:157`, `use_fused_k = recon_k and rt.use_fused_k`).

**How to verify** — `python3 -c "import json;hw=json.load(open('compressed/palu_full_0.7/config.json'))['head_wise_ranks'];print(sum(1 for i in range(32) if max(hw[f'model.layers.{i}.self_attn.v_proj'])==128))"` → 25.

**Suggested fix** — One sentence in §4 noting the two full-rank fast paths and that G-LRD keeps one group per chip.

---

### [Minor] "A 20-window fp32 CPU run … for every row (it agrees within 2 %)" is not what the repo contains

**File:line** — `docs/report_palu_on_tt.md:17`; data in `docs/results/raw/ppl_cpu_*.json`.

**What** — There are four CPU rows (`baseline`, `palu_full_50`, `palu_full_70`, `svd_r64`), not "every row" (stage 1 alone has seven), and `svd_r64` is 703.6 (device) against 776.4 (CPU) — **10 %**, not 2 %. `stage1.md:163-175` states this honestly ("same order; both degenerate"); the report generalises it.

**How to verify** — `ls docs/results/raw/ppl_cpu_*.json` and the stage1.md table at lines 165-172.

**Suggested fix** — Restate as "the two calibrated rows agree within 2 %; the degenerate SVD rows agree only in order of magnitude".

---

### [Minor] `wo_fused = U_v W_o` is written transposed relative to the code

**File:line** — `docs/report_palu_on_tt.md:47` and `:63`; code at `palu_tt/convert.py:282-286`.

**What** — The code computes `wo_fused[:, qh*hd : qh*hd+r_v] = wo[:, qh*hd:(qh+1)*hd] @ u_v_eff[kv]`, i.e. `W_o U_v` in the `[dim, n_heads*hd]` convention HF stores `o_proj` in. The report writes `W_o' = U_v W_o` and `softmax . h_v . (U_v^T W_o)`. The algebra is right, the notation is not. (I verified the fold numerically; see the Scope table.)

**Suggested fix** — Write it in the same convention as the code, or state the convention.

---

### [Minor] The L1 circular-buffer model is tested almost entirely against itself

**File:line** — `tests/test_fused.py:127-235`; model at `palu_tt/fused.py:144-200`, constants at `:124-141`.

**What** — Of the ten tests around `palu_l1_cb_bytes` / `group_heads_per_core`, nine compare `palu_l1_cb_bytes(...)` to `palu_l1_cb_bytes(...)` with one argument changed and assert the difference equals the tile arithmetic re-typed in the test (e.g. `assert legacy - dst == (16 - 1) * 2048`), or restate `_split_for`'s branch structure. The single tie to a device is `test_l1_cb_model_matches_the_factory_allocator_report`, and `L1_RUNTIME_BASE_BYTES = 105_696` was *defined* as `1_813_984 − palu_l1_cb_bytes(...)` (the comment at `fused.py:118-123` says so). It is a valid regression anchor for the model as a whole, but it is one measured point, and the correction term `task1_extra` is re-derived inside the test rather than measured.

**Why it matters** — The estimator picks `heads_per_core` per layer, i.e. which fused-kernel layout every G-LRD timing row ran. The runtime does have a real safety net (`attention.py:641-656` halves H and retries on a circular-buffer failure, and records the H that actually ran), so a wrong estimate degrades to a slower-but-correct layout, and the recorded `group_heads_per_layer` makes it visible in the JSONs. That is why this is Minor rather than Important.

**How to verify** — Read `tests/test_fused.py:127-235`; `grep -c "palu_l1_cb_bytes" tests/test_fused.py`.

**Suggested fix** — Record one more allocator-reported CB total at a different `(r_k, r_v, chunk, H)` and pin the model at both points.

---

### [Minor] Stale weight-cache generations left on a shared disk

**File:line** — `model_cache/meta-llama/Llama-3.1-8B-Instruct/N300/tensor_cache_instruct_bfp8/` (5,806 files, 50 GB), naming from `palu_tt/attention.py:37-49`.

**What** — The directory still holds `palu_v3_*` entries (dead — `CONVERT_VERSION` is 4) and `..._latcache_k_(1, 4, 2048, 64)_...` entries from before `de661c6` took the latent caches off the weight cache (`attention.py:275-283` now uses `ttnn.from_torch` with no `cache_file_name`). They cannot be served to the current code — the tag and the tensor names both changed — so this is disk hygiene, not a correctness risk. `README.md` notes 212 GB of such files were cleaned up during stage 2b; 50 GB remain.

**How to verify** — `find model_cache -name '*palu_v3*' | wc -l`; `find model_cache -name '*latcache*' | wc -l`.

**Suggested fix** — Prune generations below `CONVERT_VERSION` and the `latcache_*` entries.

---

## Tests in this area

**`tests/test_convert.py` (492 lines) — the strongest file in this area.** It is genuinely behavioural, not a restatement: the fast-path tests construct *non-identity but exact* factorisations (`U` random invertible, `VT = U⁻¹ W`) and assert the converter's output equals the stock permuted weight; `test_padded_full_rank_k/v_uses_the_factors_not_the_original_wk/wv` assert the *negative* (`not torch.allclose(..., w["k"])`), which is the bug that produced ppl 2433/3399; `test_wo_fused_matches_attn_times_v_then_o[_low_rank]` computes the absorbed path and the stock path independently and compares, and uses the real `h // (NH // NKV)` GQA mapping on both sides. `test_load_palu_dir_expands_groups_to_heads` checks the G-LRD `U_g[hh*hd:(hh+1)*hd]` slicing against what Palu computes for head `h`. **Could it pass with a bug?** A `wo_fused` head-mapping bug would be caught (I confirmed with the mutation control). A bug in the Meta permutation would *not* be caught by `test_identity_lowrank_reproduces_stock_k_and_v` alone, because it compares `reverse_permute` per head against `reverse_permute` over the whole matrix — same function both sides; but `test_low_rank_k_block_is_padded_latent` closes that by checking `lat_k @ u_k == x @ reverse_permute(U@VT)ᵀ`, and nothing in the file ties `reverse_permute` to what tt-transformers actually does to `k_proj` (I verified that separately against `convert_hf_qkv_to_meta_format`, `load_checkpoints.py:458-461`). `test_unit_scale_keeps_the_product_and_flattens_the_latent_variance` is behavioural; `test_unit_scale_is_group_aware` restates the formula.

**`tests/test_fused.py` (235 lines).** The first half is real: `test_transformation_matrix_is_the_rope_pair_rotation` checks the `(x0,x1) → (−x1,x0)` property rather than the matrix's construction; `test_angle_addition_advances_the_table_by_a_chunk` and `test_relative_rotation_identity` check the trigonometric identities Phase 2 rests on; the `reference_decode` tests reduce it to plain causal SDPA and to an explicitly rotated K. `test_chunk_size_is_the_ops_static_choice` restates the implementation but happens to be correct (I checked `sdpa_decode.cpp:22-36`). The L1-model half is largely self-referential — see the Minor finding.

**`tests/test_compress.py` (57 lines).** Behavioural: full-rank SVD is exact, error decreases with rank, and the written layout reconstructs `W`. It does not check that `compress.py`'s output is loadable by `convert.load_palu_dir` (different file name: `palu_attn.safetensors` vs the dumps' `palu_lowrank.safetensors` — both are globbed, so this is fine, but untested end to end).

**`tests/test_uniform_ranks.py` (29 lines).** Small and behavioural, including the negative case (`split_dump_name` rejects non-whole percentages, with the collision it prevents named).

**`tests/test_attention_tag.py` (39 lines).** Pure string assertions on `cache_tag`, including the two collisions that actually happened (same ranks/different weight set; converter version). Real, but it only covers the *tag*; nothing asserts that the tag is *complete* — i.e. that no input that changes a cached tensor's content is missing from it. (I checked this by hand and found no gap: `cache_id` folds in the dump hash and `latent_scale`, `gs` and both ranks are in the tag, `_kstock2` covers the `recon_k` fork, `_b{B}` covers the batch expansion, and dtype/layout are in the filename by `as_tensor`.)

**`tests/test_attention_device.py` (474 lines) — read only, not run.** What it would catch: (a) prefill and decode against an fp32 CPU reference for four latent shapes including the mixed `rk32_rv128`, at PCC 0.99/0.999, plus the *cache contents* against what the reference says they should hold, with a separate expectation for the fast paths (`_ref_k_cache`/`_ref_v_cache`) — this is the test that would have caught the fast-V bug; (b) fused-vs-naive at PCC 0.99 **plus `torch.equal` on both caches**, for Phase 1 and Phase 2, per-head and group-shared, at batch 32 and batch 1; (c) the G-LRD group/head arithmetic against the per-head reference at two batches; (d) two genuine regressions with the failure mode in the docstring (the full-range-slice dealloc at `recon_len == max_seq_len`; the 512-token grouped prefill shard fallback); (e) `test_prefill_sees_the_latent_cache_format`, which asserts *inequality* of outputs across K-only/V-only/both bfp4 and across the blind switch — this is the one that would catch the SDPA cast on a fast path going to the wrong dtype. What it would **not** catch: bit-identity against stock (see the Important finding), bfp4 round-trip idempotency (see the Important finding), and any kernel-layout difference finer than PCC 0.9999 in the group-shared comparison (documented in the test).

**`tests/test_model_device.py` (97 lines) — read only.** `test_identity_path_matches_stock_model` is the whole-model identity check: PCC 0.999 on prefill logits **and 32/32 identical greedy tokens**, the latter being a fairly strong signal. It would pass with the ~1e-3 difference the committed `ppl_identity` row implies. `test_r64_model_tracks_cpu_lowrank_model` is `@slow`, bars 0.95 PCC and 0.85 teacher-forced agreement, and the bars are justified against tt-transformers' own full-model bar (0.94) — the file explicitly documents why free-running greedy match was rejected as a metric (stock scores 0.19 on it), which is unusually careful.

**`tests/test_kv_cache_bfp4_device.py` (37 lines) — read only.** Narrow and good: `fill_cache` into a bfp4 cache copies tiles byte-for-byte (`torch.equal`), neighbours untouched, and the **negative** case (bfp4 input into a bfp8 cache still raises). It pins the one submodule relaxation the bfp4 latent rows depend on.

**CPU-test run status.** Not run — `tests/conftest.py:7` imports `palu_tt.device`, which imports `ttnn` at module level, so the permitted pytest invocation would have imported ttnn into the process. The math those tests cover was verified independently instead (Scope section).

---

## Claims traced

| documented claim | code implementing it | status |
|---|---|---|
| Latent is `h = x @ VTᵀ` (README:6) | `convert.palu_layer_tensors:269,274` writes `VT` into the QKV weight rows; `attention.py:197-209` transposes per device | **verified** (numerically, 6 shapes) |
| K rebuilt as `latent @ U_k` with prefill-mode RoPE over every cached position (README:7) | `attention.py:449-455` (prefill), `:660-684` (decode: matmul, `cos_prefill[:, :, :S]`, `is_decode_mode=False`) | **verified** — cache row *i* holds position *i*, RoPE table sliced to `[0, S)` |
| `u_k` carries the Meta head permutation so stock RoPE applies (`convert.py:275-276`) | `reverse_permute(k_U[h], 1, hd, r_k).T` | **verified** — matches `convert_hf_qkv_to_meta_format` (`load_checkpoints.py:458-461`), the same function on the same axis |
| V up-projection folded into `wo`, `U_v` of KV head *g* applied to the 4 query heads that read it (report §3) | `convert.py:282-286`, `kv = qh // rep` | **verified** numerically; mutation control fails as expected |
| The fold is consistent with TP=2 head sharding (16 q / 4 kv per chip) | `attention.py:197-209` (`chunk(dim=0)`), `:263-266` (`ShardTensorToMesh(dim=2)` on `woᵀ`), `:222` (`ShardTensorToMesh(dim=1)` on `u_*`) | **verified** — all three shard the same global head order |
| Attention scale unchanged | `attention.py:126` `self.scale = head_dim ** -0.5`; passed at `:494` (prefill SDPA), `:637` (fused), `:700` (MLA decode); stock `attention.py:375-377` | **verified** — identical, and `query_pre_attn_scalar` is asserted `None` at `:99` |
| "a rank-128 identity factorisation is bit-identical to stock" (README:9, stage1.md:159) | `attention.py:143-153`, `:271`; `convert.py:258-286` | **contradicted** by `ppl_identity` vs `ppl_baseline` (Δnll 50.8); **not verified** by any test |
| G-LRD: one shared latent per group, per-head `U` slices | `convert.py:164-172` (`U_g[hh*hd:(hh+1)*hd]`), `:267-274` (VT once per group slot) | **verified** numerically at gs=2 and gs=4 |
| G-LRD: V not absorbed; vanilla path uses a second matmul; fused kernel rebuilds V in-kernel | `convert.py:279-280` (`wo_fused = wo.clone()`), `attention.py:685-694` (`u_v_decode` matmul), `:632-637` (`u_v=self.u_v_kernel`, `head_dim_v=head_dim`) | **verified** |
| G-LRD decode uses the same MLA op at `head_dim_v == head_dim` | `attention.py:699-702` | **verified** in the host call; the op's own equivalence is the kernel auditor's area |
| "one group per chip" | `attention.py:316,324` — `assert self.n_local_groups == 1`; refuses any other mapping | **verified** (fails loudly rather than mis-indexing) |
| Prefill attends over the quantised latent when `latent_quant_in_prefill` | `attention.py:424-444` (`_roundtrip` before `recon`), `:483-489` (fast-path casts follow `keys/values.dtype`) | **verified** in code and by `test_prefill_sees_the_latent_cache_format` |
| `latent_quant_in_prefill` defaults on in every CLI | `convert.py:58`, `model.py:57`, `eval_ppl.py:96`, `eval_tasks.py:264` (`store_false`) | **verified** |
| stage 1 / 2b ppl rows describe a different prefill from stage 2f rows | raw JSON `latent_quant_in_prefill` field: `null` vs `true` | **verified** — they do |
| `latent_dtype` / `k_/v_latent_dtype` select the latent caches' tile format, independent of the model's KV precision | `attention.py:164-168`, `:274-283`; `model.py:20-30` | **verified** |
| `kv_dtype` is the model's `KV_CACHE` precision (whole cache on stock; reconstruction temporaries on Palu) | `model.py:98-138`; `attention.py:487-489`, `:678,690` | **verified**; `set_kv_cache_dtype`'s rebuild is a true no-op for this model (no `accuracy_decoder_config.json` for Llama-3.1-8B-Instruct, so the default and the rebuild are the same object shape) |
| `latent_scale="unit"` is an exact reparametrisation (`U@VT` unchanged) and changes the cache id | `convert.py:115-136`, `:219-224` | **verified** (scale factors are inverse per component; group-aware via `lr.group_size`) |
| `CONVERT_VERSION` bumped whenever a cached tensor's content changed under an unchanged name (report §4: "three times") | `convert.py:16-23` (v1→v4), `attention.py:37-49` | **verified**; I found no input that changes a cached tensor's content and is absent from the tag |
| `_recon_len` keeps `S` a multiple of 64 so the decode op never picks a 32-row chunk | `attention.py:518-530` vs `sdpa_decode.cpp:22-36` (`min(512, s & -s)`) | **verified**; applied on the naive path only, which is the only path that needs it |
| bytes/token = `groups_per_chip × (r_k·bpe_k + r_v·bpe_v)` (stage2f.md:3) | `attention.py:297-303` | **verified**; and the dumps' ranks are uniform within a layer and already multiples of 32, so `load_palu_dir`'s zero-padding to `r_max` adds no unaccounted bytes (checked on 6 dumps) |
| "the bfp8 round trip was measured at −0.033 ppl (2f Task 1b)" | `report.py:1274,1448`; `stage2f.md:69,162` | **not verified** — no artefact in `docs/results/raw/` |
| "the 40-window screens sit about 0.5 ppl below the 141-window instrument" (stage2f.md:156) | — | **verified** from raw: `palu_full_50` 12.366 (40w) vs 12.931 (141w) |
| "only the latents are written to the caches" (report §4) | `attention.py:466-473`, `convert.py:258-274` | **contradicted** for the 25/32 fast-V layers of `palu_full_0.7` and the 1 stock-K layer |
| "A 20-window fp32 CPU run … for every row (agrees within 2 %)" (report §1) | `ppl_cpu_*.json` | **contradicted** — 4 rows, one at 10 % |

---

## Verdict

The mathematics is right. I re-derived the whole model path independently — latent orientation, the Meta permutation on `U_k`, prefill-mode RoPE over cached positions, the `U_v`-into-`wo` fold under GQA on TP=2, and the G-LRD group/head index arithmetic — and every one of the six shapes agrees with a from-scratch reference to fp32 noise, with a mutation control proving the check is not vacuous; the attention scale is untouched, the fast-path collapses are correct for non-identity factors, and the `_recon_len` workaround is applied on exactly the one path that needs it and does not change the positions attended.

What is weak is the provenance around the numbers, not the arithmetic. The `identity` row — the row the port's correctness is argued from — is inconsistent with the "bit-identical to stock" claim that appears in the README, stage 1 and the external report, and nothing in the repo settles which of the two is wrong; one 40-minute device run would. The stage-1/2b/2c perplexity and task rows were measured with a prefill that the current default no longer runs, no documented command reproduces them, and stage 2f's headline "cost of bfp4" deltas subtract those blind rows from faithful ones on the strength of a −0.033 correction that has no artefact behind it (and no equivalent correction at all for the task metric).

Test quality is above average for this kind of codebase — `test_convert.py` and the device attention tests assert against independently computed references and include the negative cases that the historical bugs needed — but three guards are weaker than they look: nothing asserts the bit-identity the docs claim, the bfp4 round-trip idempotency that every bfp4 row's meaning depends on is computed and then not compared, and `test_apply_palu_flags_…` supplies the key it exists to prove is present.
