# Audit F: calibration pipeline and Palu patches

Repo: `/home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9` @ 40b9a15.
Submodule `third_party/palu` @ `bb22666e2ef96707e8dd21d93fc00146c2e0d615` ("Merge pull request #11…", 2025-02-19), dirty by design (7 modified files, untracked `cache/`).
Read-only audit: no git mutation, no edit, nothing written inside the repo. **`tests/test_patch_palu_cpu.py` was NOT run** — see Finding 5.

## Scope read

- `scripts/patch_palu_cpu.sh`: 1–216 (whole file).
- `scripts/patch_rank_search_split.py`: 1–113 (whole file).
- `scripts/run_calibration.sh` 1–39, `run_calibration_uniform.sh` 1–39, `run_calibration_split.sh` 1–65, `run_calibration_from_ranks.sh` 1–52 (all whole).
- `scripts/env.sh` 1–15, `scripts/setup_env.sh` 1–18, `scripts/build_ttnn.sh` 1–9, `scripts/run_ppl_cpu.sh` 1–7 (all whole).
- `requirements.txt`, `.gitmodules`, `.gitignore`, `pytest.ini` (whole).
- `tests/test_patch_palu_cpu.py`: 1–196 (whole file, read only — not executed).
- `docs/remote/patch_palu_gpu.sh`: 1–116 (whole). `docs/remote/goal.md`: 1–289 (whole).
- `docs/remote/results/gpu_report_2026-09-18.md`: preflight/versions/wall-clock/quality blocks read in full; the four inline rank JSON lines were parsed programmatically rather than read by eye.
- Live `git -C third_party/palu diff`: all 308 lines, saved to the scratchpad as `palu_live.diff`.
- Upstream (`git show HEAD:`) `compress.py` 1–135, `palu/rank_search.py` 1–110 (plus the rest via the diff's context), `palu/decomposition.py` 1–268, `palu/data_utils.py` 1–43, `palu/model/modules/svd_linear.py` 1–34 and 80–205, `palu/model/svd_llama/modeling_palu_llama.py` 38–55, `utils.py` 1–110.
- `README.md` 18–26 and 53–92. `docs/report_palu_on_tt.md` 9–15, 30–59, 110–118 (sections 4–7, the kernel chapters, skimmed only — out of area). `docs/results/stage1.md` 310–325 plus grep-targeted lines. `docs/results/stage2c.md` 701–800 (§9 in full) plus grep-targeted lines; §§1–8 **skimmed only**. `docs/results/stage2b.md` and `stage2f.md`: **grep-targeted only**, tables skimmed.
- `palu_tt/uniform_ranks.py` 1–50 (whole), `palu_tt/convert.py` 170–265, `palu_tt/hf.py` 1–43, `palu_tt/report.py` grep-targeted (1212–1260).
- `.venv/.../transformers/models/llama/modeling_llama.py`, `LlamaDecoderLayer` (installed transformers 5.12.1) — return type check.
- Data verified programmatically: every `compressed/*/config.json` and `*.ranks.json`, `docs/remote/results/*.json`, every `compressed/calibration_*.log` (grepped for cache lines and rank-search lines), `calibration_0.5.nohup` / `calibration_0.5_redo.nohup` (head/tail), the safetensors **headers** of all 27 dumps, and layer-0/1/2/3/16/31 `k_proj.VT.weight` tensors of five dumps.
- Not read (out of area): the kernel/bench/eval code and their tests, `scripts/kernel_figures.py`, `scripts/report_figures.py`, `scripts/run_device.sh`, `scripts/device_free.sh`.

## Patch inventory

Classes: **a** environment/compat · **b** recipe numerics · **c** algorithm · **d** other.
"Changes the factorisation?" = does it change which U/VT come out for the same nominal recipe.

| file | change | class | changes factorisation? | disclosed |
|---|---|---|---|---|
| `utils.py:86` | `torch_dtype=torch.float16` → `torch.float32` | **b** | yes — Fisher gradients, whitening activations and the decomposed weights are all fp32 instead of fp16 | yes: `report:12` "on the host CPU in fp32", `README:63` |
| `utils.py:88` | drop `device_map="auto"` | a | no | yes: `patch_palu_gpu.sh:5` ("CPU-only changes … deliberately NOT here") |
| `palu/model/modules/svd_linear.py:14,17` | whitening inverse + `W·S` product fp32 → **float64** | **b** | yes — without it layers 0–1 are destroyed (ppl 2397) | yes: `report:12`, `report:55`, `stage1:318–320` |
| `palu/rank_search.py:37` | `model.half()` → `model.float()` on CPU | **b** | yes — Fisher is `grad²` accumulated in the model dtype; upstream's fp16 backward has no loss scaling | partly: "fp32" is stated, but not that it makes CPU ranks non-comparable with the fp16 GPU/paper search |
| `palu/rank_search.py:39–40` | `requires_grad_('attn' in n)` | d | no (grads of attn weights are unchanged) | no (code comment only) — harmless |
| `palu/rank_search.py:41` | Fisher cache key gains `_n{len(calib_loader)}` | **c** | no by itself; prevents silent cross-count reuse | yes: `README:82–88`, `stage2c:65` |
| `palu/rank_search.py:109,173` | literal `2048` → `args.n_fisher_calib_samples` (both call sites) | **c** | yes — this is what makes 32/256-sample searches possible | yes: `stage1:312–314`, `README:80–88`, `report:12` |
| `palu/rank_search.py:184–243` | split-budget `--k_ratio_target` / `--v_ratio_target` (one Fisher pass per matrix family) | **c** | only when a flag is set (the `split_*` dumps); no-flags path is upstream's arithmetic | script/test comments only — **not in `stage2f.md`, `README.md` or the report** |
| `compress.py:20–26` | `--ranks_json` short-circuits `rank_search()` | **c** | no — whitening + SVD unchanged; it only supplies the ranks | `stage2c:713` names the script; the patch itself is not described anywhere |
| `compress.py:34–38` | the three new CLI flags | c | no | as above |
| `palu/data_utils.py:29` | `"wikitext"` → `"Salesforce/wikitext"` | a | no (mirror is byte-identical; the calib cache file name is unchanged) | script comment only |
| `palu/decomposition.py:88,96–101` | Catcher captures `position_embeddings`; `is None` → `i == 1`; None-safe mask concat | a | no | `stage1:315–316` (as "transformers 5.x needed position_embeddings") |
| `palu/decomposition.py:139` | layer replay forwards `position_embeddings`, None-safe `position_ids`, drops upstream's `[0]` | a (+d) | no on tf 5.12.1 (layer returns a Tensor — verified); upstream's `[0]` would have taken batch row 0 | partly, same bullet |
| `palu/decomposition.py:150,163,175` | three `.cuda()` → `.to(dev)` | a | no | script comment only |
| `palu/model/modules/hadamard_utils.py:2` | optional `fast_hadamard_transform` import | a | no (`hadamard=False` throughout this recipe — verified) | script comment only |
| untracked `third_party/palu/cache/` | 9.7 GB of Fisher/whitening/calib-data caches | d | — | `.gitignore` does not cover it; `README:91` mentions the directory |

GPU patch set (`docs/remote/patch_palu_gpu.sh`) = items 3, 6, 7, 11, 12, 13 above **only**. It deliberately omits the two `utils.py` edits and the `model.float()` edit, so the GPU ran upstream fp16 with `device_map="auto"`, under transformers 4.46.3.

## Findings

### [Critical] The report states the Palu reference implementation was "used unchanged"; it was patched in ways that change the factorisation

- **File:** `docs/report_palu_on_tt.md:118`; contradicted by the live `git -C third_party/palu diff` (7 files, 100 insertions / 61 deletions) and by `scripts/patch_palu_cpu.sh:14–216`.
- **What:** the bibliography bullet reads "`third_party/palu` is its reference implementation, **used unchanged** for the offline factorisation." Two patches change the numerics of the factorisation (`svd_linear.py` fp32→float64 inverse and SVD; `utils.py` fp16→fp32 model load, which also makes the Fisher backward and the whitening activations fp32 via `rank_search.py`'s `model.float()`), and four change the algorithm (honouring `--n_fisher_calib_samples`, the Fisher cache key, `--ranks_json`, `--k_ratio_target`/`--v_ratio_target`).
- **Why it matters:** this is the one sentence a reader consults to judge "is the recipe Palu's?". It directly contradicts line 12 and line 55 of the same document, which do describe the float64 whitening and the fp32 CPU run. For an externally presented study this misstates what was run.
- **How to verify:** `git -C third_party/palu diff` (all seven files); `scripts/patch_palu_cpu.sh` lines 14–216; compare with `docs/report_palu_on_tt.md:118`.
- **Suggested fix:** "…is its reference implementation; it is patched at run time by `scripts/patch_palu_cpu.sh` — environment/compat fixes plus the float64 whitening and the rank-search flags described in §3 and the README. The recipe (`fisher_uniform` + `whiten`) is Palu's."

### [Important] §9.3's "same factorisations" comparison compares two different sets of factors, so the quoted "port cost" is not only the port

- **File:** `docs/results/stage2c.md:755–780`; `docs/report_palu_on_tt.md:10`; `docs/remote/goal.md:231–236`; `docs/remote/patch_palu_gpu.sh:5,8–9`; upstream `utils.py:86` and `palu/rank_search.py:37`.
- **What:** the GPU fp16 perplexities and accuracies were measured on the **GPU's own dumps** (`goal.md:235` scores `"$dump"`). Those dumps were produced by `compress.py` on the GPU with the *unpatched* `utils.py` (fp16 load, `device_map="auto"`), and `calib_fisher_info`'s upstream `model.half()` converts the model in place before `get_whiten_scale_matrix` runs — so the GPU whitening statistics, the weights fed to the SVD and the emitted L/R are all fp16, under transformers 4.46.3. The n300 rows use dumps rebuilt locally from **only the rank lists**, with the CPU fp32 whitening cache and fp32 weights (`run_calibration_from_ranks.sh`). Same ranks, different U/VT.
- **Why it matters:** §9.3 says explicitly "set against the n300 rows they isolate the port (bfp8 weights and latents, bf16 activations, TP 2, our harness)", and `report:10` carries the numbers forward as "puts the whole port at 0.04 perplexity on the baseline and 0.09 to 0.22 on the compressed models". The compressed-row deltas also contain fp16-vs-fp32 decomposition and transformers 4.46.3-vs-5.12.1. §9.4 ("what this addendum does not settle") lists three caveats and not this one. The baseline row (+0.041) is unaffected — there is no factorisation there.
- **How to verify:** `docs/remote/patch_palu_gpu.sh` contains no `utils.py` patch and no `model.half()` patch; upstream `utils.py:86` is `torch_dtype=torch.float16`; upstream `rank_search.py:37` is `model.half()`; `goal.md:16` says only the ranks come back; `stage2c:710–713` confirms the local rebuild uses the cached (CPU) whitening.
- **Suggested fix:** state that the two sides share ranks, not factors, and relabel the delta "port + decomposition precision", or add it to §9.4.

### [Important] `run_calibration.sh` and `run_calibration_uniform.sh` lack the stale-output and existing-target guards their two siblings have

- **File:** `scripts/run_calibration.sh:21,38`; `scripts/run_calibration_uniform.sh:61,78`. Compare `scripts/run_calibration_split.sh:31–34,63` and `scripts/run_calibration_from_ranks.sh:93–95`.
- **What:** `compress.py`'s output folder is named only from `param_ratio_target`, `head_group_size`, `search_method`, `decompose_method` (upstream `compress.py:25`). `--n_fisher_calib_samples` is **not** in it, so `run_calibration.sh 0.5 4 256` and `run_calibration.sh 0.5 4` produce the identical directory name, and all twelve `uni_*` runs share `*_ratio-1.0_gs-1-uniform-whiten`. Both scripts then do `out="$(ls -d <glob> | head -1)"` with no stale check, and `mv "$out" "compressed/<name>"` with no existing-target check. `mv` into an existing directory **nests** rather than replaces; `convert.load_palu_dir` (`palu_tt/convert.py:185–192`) globs only the top level, so it would keep reading the *old* `config.json` and the *old* `palu_lowrank.safetensors`, and `convert.palu_dir_cache_id` (`:202–216`, hashes `config.json` bytes plus each shard's name/size/mtime) would be unchanged, so the ttnn weight cache would be reused too.
- **Why it matters:** this is exactly the failure mode `stage1.md:323–327` records ("the re-generated ratio-0.5 dump kept loading the broken dump's tensors and reproduced its perplexity bit-for-bit, 2397.3228530751553 twice"), and the sequence did occur: `calibration_0.5.nohup` ends `2026-09-10 07:19:53 … done -> compressed/palu_full_0.5`, then `calibration_0.5_redo.nohup` (the first run **with** the float64 patch, committed 07:35) ends `2026-09-10 07:40:43 … done -> compressed/palu_full_0.5`. The second `mv` into an already-existing directory only avoided nesting because someone deleted the first output by hand — an unrecorded manual step.
- **Evidence the shipped dumps are in fact clean (checked):** `find compressed -mindepth 2 -type d` is empty; every dump's `VT.weight` first dimension equals `sum(head_wise_ranks[module])` for all 64 modules in all 27 dumps (0 mismatches); the three `_f2048` dumps' `head_wise_ranks` are exactly their `.ranks.json`, which are exactly `docs/remote/results/ranks_*_f2048.json`; `max|VT_k|` for layers 0/1 of `palu_full_0.5/0.7`, `glrd_0.7`, `glrd_0.5_f256`, `palu_full_0.7_f2048` is 0.013–0.053, i.e. the same order as layers 2–31, so no dump carries fp32-whitening damage.
- **Suggested fix:** lift the two guards from `run_calibration_split.sh` verbatim into `run_calibration.sh` and `run_calibration_uniform.sh`, and put `nf` (and `rk`/`rv`) into the `compress.py` output folder name or into a pre-run `mktemp -d` working directory.

### [Important] The CPU patch script has no post-condition for the two `sed`s that decide the Fisher sample count and the Fisher cache key

- **File:** `scripts/patch_palu_cpu.sh:40` and `:45` (bare `sed -i`, no verification). Contrast `docs/remote/patch_palu_gpu.sh:114–115`, which greps for both `float64` and `args.n_fisher_calib_samples, seqlen` and exits non-zero.
- **What:** the script asserts anchors or greps for the `decomposition.py` patch (`assert` ×3), `svd_linear.py` (`assert` + grep), `--ranks_json` (grep), and the split patch (grep ×2). The two `sed -i` lines that (a) replace the hard-coded `2048` with `args.n_fisher_calib_samples` at both call sites and (b) append `_n{len(calib_loader)}` to the Fisher cache file name are unchecked; `sed` exits 0 when nothing matches.
- **Why it matters:** failure of (a) is loud (a 39 h Fisher pass). Failure of (b) is **silent and corrupting**: a `--n_fisher 256` run would load the 32-sample cache and be published as a 256-sample dump, which would invalidate the whole calibration-size line of argument in `stage2c` §1 and §9. `README:87–88` asserts "The cache key is what makes that safe".
- **How to verify:** read `scripts/patch_palu_cpu.sh:35–45` and `docs/remote/patch_palu_gpu.sh:110–116` side by side.
- **Suggested fix:** add `grep -q "args.n_fisher_calib_samples, seqlen" palu/rank_search.py || { echo "n_fisher patch missing"; exit 1; }` and `grep -q "calib_fisher_info_n{len(calib_loader)}" palu/rank_search.py || { echo "fisher cache-key patch missing"; exit 1; }`.

### [Important] `tests/test_patch_palu_cpu.py` mutates `third_party/palu`, contradicting its own docstring

- **File:** `tests/test_patch_palu_cpu.py:8` ("nothing here writes under `third_party/`") versus `:184–196`, which runs `subprocess.run(["bash", "scripts/patch_palu_cpu.sh"], cwd=root)`; `scripts/patch_palu_cpu.sh:9–10` then runs `git checkout -- …` inside the submodule and rewrites six of its files.
- **What:** four of the five tests are genuinely hermetic (they work on `git show HEAD:palu/rank_search.py` copied into `tmp_path`). The fifth is a smoke test that re-applies the live patch set in place.
- **Why it matters:** the submodule's working tree *is* the artefact — it is patched at run time and never committed. If the script aborts between the `git checkout --` and the re-application (anchor drift after a submodule bump, a Ctrl-C, a concurrent calibration reading the tree), the runtime patches are silently gone and the next calibration runs upstream fp32 whitening — the documented 2397-perplexity failure. It also makes `pytest` non-hermetic on a shared checkout. I did not execute this file for exactly this reason, so the four hermetic tests are unverified by execution here (they were read and their arithmetic checked against the calibration logs, which do print `k_proj target ratio 0.25: 8192 of 32768 rank` and `v_proj target ratio 0.75: 24576 of 32768 rank`, i.e. the numbers `test_split_path_hits_each_familys_target_exactly` asserts).
- **Suggested fix:** run the smoke test against a copy (`cp -a third_party/palu $tmp` or `git worktree add`), or split `patch_palu_cpu.sh` so the `git checkout --` reset is a separate, explicitly requested step.

### [Important] The per-layer ranks of every shipped dump except the four GPU ones are not in the repo, and the published tables are generated from them

- **File:** `.gitignore:3` (`compressed/`); `palu_tt/report.py:1258–1260` reads `compressed/<dump>/config.json`; `docs/results/stage2f.md:3` documents that the byte columns are "read from the dump's own `head_wise_ranks`".
- **What:** only `docs/remote/results/ranks_{mlrd,glrd}_{0.5,0.7}_f2048.json` are committed. The `head_wise_ranks` of `palu_full_0.5`, `palu_full_0.7`, `glrd_0.5`, `glrd_0.7`, `glrd_0.5_f256`, `glrd_0.7_f256`, `split_k25_v75`, `split_k50_v75` and all twelve `uni_*` dumps exist only in untracked `compressed/*/config.json`.
- **Why it matters:** the r_k / r_v / bytes columns of `stage1`, `stage2b`, `stage2c` and `stage2f` and of `report:40–51` cannot be checked or regenerated from a clean clone; `python -m palu_tt.report` would fail or silently skip those rows. Every number I verified in this audit came from files an external reader will not receive.
- **Suggested fix:** commit the `head_wise_ranks` block of each dump (a few kB each) alongside `docs/remote/results/`, as was already done for the GPU ranks.

### [Important] The whitening calibration ignores every CLI knob, and the docs describe it with the Fisher pass's sequence length

- **File:** upstream `third_party/palu/palu/decomposition.py:24–30` — `get_calib_data("wikitext2", tokenizer, model_id, nsamples=256, seqlen=2048)`, all three literals; `compress.py:70–75` declares `--n_whiten_calib_samples` (default 256) and `grep -rn n_whiten_calib_samples third_party/palu` matches only that declaration, so the flag is dead. `--calib_dataset` and `--calib_seqlen` likewise never reach the whitening pass (the `[whiten] Calibration dataset: {args.calib_dataset}` log line at `:48` prints a value the code does not use).
- **What the docs say:** `report:12` "wikitext-2 calibration sequences of 1024 tokens" — true of the Fisher pass, not of the whitening pass, which uses 2048. `README:84–85` "Palu's whitening already uses its own default of 256 samples at every setting" — right in value, but implies a knob that does not exist. `stage2c:783` lists "Whitening samples stay at Palu's default 256 in every dump … The whitening limb of stage 2b's hypothesis remains untested" without noting that testing it requires another patch.
- **Evidence:** two distinct calibration-data caches are on disk — `cache/wikitext2_…_256_2048_3.pt` (the whitening set, 2026-09-09 21:30) and `cache/wikitext2_…_{32,256}_1024_3.pt` (the Fisher sets).
- **Suggested fix:** in `report:12` and `README:84`, say "the whitening pass hard-codes 256 wikitext-2 sequences of 2048 tokens (`--n_whiten_calib_samples` is inert in upstream); the Fisher pass uses 32/256 sequences of 1024 tokens".

### [Minor] Fisher cache key covers the sample count but not sequence length, dataset or model dtype

- **File:** patched `palu/rank_search.py:41` → `cache/{model_id}_calib_fisher_info_n{len(calib_loader)}.pt`.
- **What:** `--calib_seqlen`, `--calib_dataset` and the model dtype are absent from the key, although `get_calib_data`'s own cache (`palu/data_utils.py:10–12`) keys on all of `name/model/nsamples/seqlen/seed`. Nothing collided in this study (every run used wikitext2 / 1024 / fp32 on CPU), but `docs/remote/goal.md:216` explicitly authorises halving `--calib_seqlen` to 512 on OOM, which would silently reuse the 1024-token Fisher tensor, and a CPU (fp32) and GPU (fp16) run sharing one tree would collide.
- **Fix:** include seqlen, dataset and `next(model.parameters()).dtype` in the key.

### [Minor] The `_n32` Fisher cache was renamed by hand and nothing in the repo ties it to 32 samples

- **File:** `scripts/patch_palu_cpu.sh:44` ("the stage-1 32-sample cache was renamed to the `_n32` name by hand").
- **What:** every stage-1/2b calibration log (`compressed/calibration_{0.5,0.7,glrd_0.5,glrd_0.7}.log`) loads the *unsuffixed* `…_calib_fisher_info.pt`, so the file now named `…_n32.pt` was renamed some time after 2026-09-12 09:38 and before the first `_n256` run at 22:44. The only corroboration is timing: `cache/wikitext2_…_32_1024_3.pt` at 09-09 20:50 → `…_n32.pt` at 09-09 21:30 (≈75 s/sample), consistent with `…_256_1024_3.pt` 09-12 17:50 → `…_n256.pt` 22:44 (≈69 s/sample). Plausible, but it is a manual step with no record.
- **Fix:** record the rename (and the two caches' sha256 and provenance) in `stage1.md`.

### [Minor] "float64 end to end" overstates the whitening precision

- **File:** `scripts/patch_palu_cpu.sh:210` comment; upstream `palu/decomposition.py:152,164` `torch.linalg.cholesky(raw_scaling_diag_matrix).float()`.
- **What:** the Cholesky itself is already double upstream, but its result is rounded to **fp32** before being cached (`cache/whiten/…_fp16.pt`) and before `svd_linear.py` casts it back to float64. The inverse and the SVD are genuinely fp64 — which is what fixes the reconstruction, since inverting an fp32 matrix accurately in fp64 is well-posed — but the whitening *basis* is an fp32-rounded Cholesky factor, not an fp64 one. (The cache file name also still says `fp16` although the statistics here came from an fp32 model.)
- **Fix:** "the inverse and the SVD are taken in float64" rather than "float64 end to end".

### [Minor] The CPU `decomposition.py` patch is hard-bound to transformers 5.x, with no version check

- **File:** `scripts/patch_palu_cpu.sh:126–134` — drops upstream's `[0]` and assigns the layer's return value straight into `outs[j]`.
- **What:** verified correct for the pinned `transformers==5.12.1` (`LlamaDecoderLayer.forward` is annotated `-> torch.Tensor` and returns `hidden_states`). Under transformers 4.x — the version `docs/remote/goal.md:41` pins for the GPU box — the same patch would assign a tuple to a tensor slice and raise. The GPU script handles both shapes (`patch_palu_gpu.sh:87`); the CPU one does not, and nothing checks the installed version at patch time.
- **Fix:** adopt the GPU script's `_out[0] if isinstance(_out, (tuple, list)) else _out` form in the CPU script too; the two scripts would then differ only where they must.

### [Minor] `--ranks_json` and the split-budget search are undocumented as departures from Palu

- **File:** `docs/results/stage2f.md` (no occurrence of `ranks_json`, `k_ratio_target`, `run_calibration_split.sh`, `run_calibration_uniform.sh`, "patch" or "upstream"); `README.md:53–91` documents only `run_calibration.sh`; `docs/report_palu_on_tt.md` mentions neither.
- **What:** `stage2f:125–127` attributes 0.274 perplexity at equal bytes to "per-layer allocation" (`split_k50_v75` 9.472 vs uniform (64, 96) 9.746) without saying that the allocation is a locally modified `fisher_uniform` running two independent per-family budgets, not Palu's pooled one. The twelve `uni_*` dumps likewise never get a provenance line.
- **Fix:** one paragraph in `stage2f.md` and three lines in the README's "Producing the weight sets".

### [Minor] "picks the same ranks within one 32-step in at most 7 of 64 modules" is true, but the M-LRD 0.5 "bit-identical" result is largely a rounding floor

- **File:** `docs/report_palu_on_tt.md:12,53`; `docs/results/stage2c.md:716–728`; upstream `palu/rank_search.py:11–17` (`rounding_search_result`, `max(1, round(r/32))*32`).
- **Verified exactly:** `palu_full_0.7` vs `_f2048` 1/64 modules; `glrd_0.5_f256` vs `_f2048` 5/64; `glrd_0.7` vs `_f2048` 7/64; `glrd_0.7_f256` vs `_f2048` 4/64; every delta exactly ±32; `ranks_mlrd_0.5_f2048.json` is byte-for-byte the ranks of `palu_full_0.5`. So the claim stands.
- **The caveat:** in `palu_full_0.5`, 248 of 256 k-groups sit at the rank-32 floor and the v-groups are 144×128 / 96×64 / 16×96 — the allocation is saturated, so insensitivity to the sample count is partly arithmetic, not evidence about Fisher. Also, `split_values` at group size 1 divides one pooled per-layer budget by 8, so the eight "per-head" ranks of a module are always equal: these are per-layer ranks, not per-head ones.
- **Fix:** add a clause noting the 32-step floor; consider calling the M-LRD numbers per-layer ranks.

### [Minor] Script and doc nits

- `scripts/run_calibration.sh:13` installs `loguru click lm-eval==0.4.2` with `2>/dev/null || true`; `run_calibration_uniform.sh`, `run_calibration_split.sh` and `run_calibration_from_ranks.sh` install nothing and silently depend on a previous run having done it. `requirements.txt` pins only `transformers==5.12.1` and `torch==2.11.0`; `datasets`, `huggingface_hub`, `safetensors`, `numpy`, `loguru`, `click` are unpinned or absent; `scripts/setup_env.sh:32` installs only `requirements.txt`.
- `scripts/patch_palu_cpu.sh:193` prints "applied cleanly" **before** the float64 patch at 195–216 runs; `tests/test_patch_palu_cpu.py:196` asserts on that string.
- `scripts/patch_palu_cpu.sh:9–10`'s `git checkout --` list omits `compress.py`, so a half-applied `compress.py` is never repaired (its three edits are guarded by `if … not in s`, so they no-op instead).
- `docs/results/stage1.md:313` says 66 s/Fisher sample, `README.md:85` says 69 s; cache mtimes imply ≈69–75 s. Harmless but inconsistent.
- `scripts/run_calibration_from_ranks.sh:88` copies the ranks file into `compressed/<name>.ranks.json` **before** the two refusal guards at `:93–95`, so a refused run still overwrites it.
- `scripts/env.sh:8` exports `OMP_NUM_THREADS=12` unconditionally on a shared host.

## Tests in this area

`tests/test_patch_palu_cpu.py` is the only test touching the calibration pipeline. It covers the split-budget transform well and nothing else.

- Covered: (1) the patch is a fixed point under a second application and the result compiles; (2) with neither ratio flag the patched `fisher_uniform` allocation block produces the same `select_result` and `total_rank` as the pristine block over four synthetic seeds; (3) with both flags each family hits its target rank exactly (8192/24576 and 16384/24576); (4) with only `--k_ratio_target` the other family falls back to `--param_ratio_target`. The synthetic shape (`num_lr_groups=1`, `lr_group_dims=1024`, 32 layers) matches the real model — I confirmed this independently from `get_kv_info(model, num_key_value_heads=8)` in `palu/model/svd_llama/modeling_palu_llama.py:38–59`, and the targets match the real logs in `compressed/calibration_split_k{25,50}_v75.log`.
- (5) is a smoke test that `bash scripts/patch_palu_cpu.sh` exits 0 — and it mutates the submodule (Finding 5).
- **Not covered at all:** the float64 `svd_linear.py` patch; the `--ranks_json` short-circuit (no test that it yields the same factors as the searched path — the load-bearing claim of §9); the two `sed`s for the Fisher sample count and the cache key; the `decomposition.py` Catcher/replay patch; the `utils.py` dtype patch; any property of `run_calibration*.sh` (naming, guards, the safetensors trim). `tests/test_uniform_ranks.py` (1.2 kB) covers `palu_tt/uniform_ranks.py` only.
- **Not executed here.** See Finding 5; the pre-condition in my brief ("it must operate on a temporary copy, never on `third_party/palu` itself") is not met by test (5).

## Claims traced

| documented claim | code / data | verdict |
|---|---|---|
| `report:118` "`third_party/palu` … used unchanged for the offline factorisation" | live `git diff`: 7 files, 100 insertions | **contradicted** |
| `report:12` / `README:63` "on the host CPU in fp32" | `utils.py:86` patched to `float32`; every dump's `config.json` has `"dtype": "float32"`; all 27 safetensors are `F32` | verified |
| `report:12,55` "the whitening runs in float64; in float32 the first two layers do not invert" | `svd_linear.py:14,17` patched; `calibration_0.5_redo.nohup` head shows the float64 lines; layer-0/1 `max\|VT_k\|` (0.047/0.013–0.016) is in family with layers 2–31 → no fp32 damage in any shipped dump | verified |
| `report:12` "wikitext-2 calibration sequences of 1024 tokens" | true of Fisher (`--calib_seqlen 1024`); the whitening pass hard-codes `seqlen=2048` (`decomposition.py:29`) | **contradicted** (for the whitening limb) |
| `report:12` "32 Fisher samples for the per-head dumps and 256 for the grouped dump" | `glrd_0.5` and `glrd_0.7` were also 32-sample (logs load `…_calib_fisher_info.pt`, i.e. the n32 cache); the table at `:44–51` shows both | imprecise, not wrong in the table |
| `report:12,53` / `stage2c:716` "2048 samples change at most 7 of 64 per-module ranks, each by one 32-step" | computed: 1, 5, 7, 4 modules changed; all deltas ±32 | verified |
| `stage2c:714` "`mlrd_0.5` was not rebuilt because its ranks are identical to `palu_full_0.5`'s" | `ranks_mlrd_0.5_f2048.json == compressed/palu_full_0.5/config.json['head_wise_ranks']` | verified |
| `stage2c:710–713` the `_f2048` dumps are "the paper's ranks with our whitening" (rebuilt from ranks, cached whitening, no rank search) | `compress.py` ranks_json branch skips only `rank_search`; `compress_model_whiten` → `get_whiten_scale_matrix` loads the same `cache/whiten/…_fp16.pt` (keyed on model id only) and `from_linear_whiten` derives group count from `len(ranks)`, so `--head_group_size` never enters; the three `calibration_*_f2048.log` contain **no** `[Fisher]` lines and do contain `Load scaling diag matrix from cache` | verified |
| `stage2c:755–780` / `report:10` GPU fp16 numbers are "the same factorisations", isolating the port | same ranks, different factors (GPU fp16 whitening + fp16 weights, tf 4.46.3) | **contradicted** as worded |
| `stage1:312–314` "both `rank_search.py` call sites pass a literal 2048 and silently ignore `--n_fisher_calib_samples`" | upstream `rank_search.py:107` and `:171` | verified |
| `stage2c:783` / `README:84` "whitening stays at Palu's default 256" | 256 is hard-coded at `decomposition.py:28`; `--n_whiten_calib_samples` is never read | verified in value, misleading in mechanism |
| `README:82–88` the `_n<n>` cache key "is what makes that safe" | key patched at `rank_search.py:41`, and the n256 log does say `No cache_file=…_n256.pt`; but the sed is unguarded and the key omits seqlen/dataset/dtype | partly verified |
| `README:86` "4 h 57 min for the ratio-0.5 run; the second ratio took 3 minutes" | `…_256_1024_3.pt` 17:50 → `…_n256.pt` 22:44 → `glrd_0.5_f256` 22:46 → `glrd_0.7_f256` 22:50 | verified |
| `patch_rank_search_split.py:11–13` "with neither flag set … the arithmetic is upstream's, unchanged" | read: `passes = [(("k_proj","v_proj"), args.param_ratio_target)]`, one pass, same expressions; upstream's `dif` over all of `select_result` equals the one-pass `dif` over `fisher_info_dict` | verified by reading (test not executed) |
| `run_calibration.sh:5–6` "only the k_proj/v_proj factors are kept" is enough for the loader | `convert.load_palu_dir:185–192` needs `config.json['head_wise_ranks']` plus `*.safetensors` keys containing `.k_proj.`/`.v_proj.`; VT row counts match `sum(ranks)` in all 27 dumps; no `model.safetensors.index.json` is left behind | verified |
| Fisher cache reuse across ratios and group sizes is valid | `calib_fisher_info` stores per-`nn.Linear` `grad²` (weight-shaped), independent of ratio; `fisher_uniform` calls `get_kv_info(model, num_key_value_heads)` — **not** `head_group_size` — so `num_lr_groups=1`, `lr_group_dims=1024` regardless of `--head_group_size`, which enters only at `split_values(…, 8//gs)` | verified |
| `stage2f:125–127` "0.274 for per-layer allocation" | the allocation is the patched split-budget search, disclosed nowhere in stage2f | unverified as presented |
| Reproducibility: `compressed/palu_full_0.7` rebuildable | see below | rank-for-rank plausible, bit-for-bit not guaranteed |

**Reproducibility (question 7), in detail.** Rank-for-rank is plausible: the submodule commit is pinned by the gitlink (`bb22666`), `transformers==5.12.1` and `torch==2.11.0` are pinned, the calibration sample draw is deterministic (`palu/data_utils.py:13` `random.seed(3)`, fixed, not exposed on the CLI — and because the offsets are drawn from one stream, the 32-sample set is a strict prefix of the 256- and 2048-sample sets, which makes the sample-count comparison clean), and the rank arithmetic is integer. Bit-for-bit is not guaranteed: the Fisher pass sums fp32 `grad²` over 32 backward passes and the whitening accumulates a 4096×4096 fp32 second moment over 256×2048 tokens, both in BLAS/thread-count-dependent order (`scripts/env.sh:8` sets `OMP_NUM_THREADS=12`, but nothing records the value used), and a small perturbation can flip `math.floor(rank_float)` across an integer boundary and move a rank by one 32-step. Missing from the repo: the caches under `third_party/palu/cache` (9.7 GB — without them a rebuild costs a ~40 min Fisher pass plus a ~10 h whitening pass), the rank lists of every non-`_f2048` dump (Finding 6), the provenance of the hand-renamed `_n32` cache (Finding 8), pins for `datasets`/`huggingface_hub`/`safetensors`/`numpy`/`loguru`/`click`, and any model revision pin (`palu_tt/hf.py:15` calls `snapshot_download(model_id, local_files_only=True)` with no `revision=`; the snapshot hash `0e9e39f2…` appears only inside the untracked dump configs and the GPU report).

## Verdict

The recipe that produced the weight sets **is** Palu's `fisher_uniform` + `whiten`, run through upstream's own `compress.py`, and the artefacts on disk are internally consistent: every dump's tensors match its own `head_wise_ranks`, the `_f2048` dumps carry exactly the GPU rank lists, the `--ranks_json` rebuild path provably reuses the same whitening cache and the same decomposition code, and no shipped dump shows the fp32-whitening damage. But the reference implementation was **not** used unchanged, and the report's pointer sentence says it was — that single sentence contradicts two other passages of the same document and is the most serious thing I found. Two further claims are weaker than they read: §9.3's GPU-versus-n300 "port cost" compares factors built from fp16 statistics against factors built from fp32 statistics (only the ranks are shared), and the docs attribute the Fisher pass's 1024-token calibration to the whitening pass, which hard-codes 256 sequences of 2048 tokens and ignores every CLI knob. Reproducibility is the weakest limb overall: the headline dumps' rank lists and the caches they were built from are untracked, two of the four calibration scripts lack the stale-output guards their siblings have, and the one test in this area mutates the very submodule tree the study depends on.
