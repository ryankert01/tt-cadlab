# Audit A: perplexity path

Auditor note on method: nothing was executed on the Tenstorrent device and `ttnn` was never imported.
Everything below is from reading the code, plus three CPU-only checks that touch no repo file:
(1) re-tokenising wikitext-2-raw-v1 exactly as `eval_ppl.wikitext2_ids` does, (2) replaying
`_nll_from_hidden`'s chunking against a one-shot cross-entropy on synthetic tensors, (3) an internal
consistency sweep of `docs/results/raw/ppl*.json`. Scripts are in the scratchpad next to this file
(`tok.py`, `nll.py`).

## Scope read

- `palu_tt/eval_ppl.py`: 1-151 (whole file).
- `palu_tt/reference.py`: 1-89 (whole file).
- `palu_tt/hf.py`: 1-43 (whole file).
- `palu_tt/device.py`: 1-28 (whole file).
- `palu_tt/model.py`: 1-193 (whole file).
- `palu_tt/generate.py`: 1-66 (whole file; `prefill_hidden_states` is the device half of the ppl path).
- `palu_tt/attention.py`: 18-52 (`_roundtrip`, `cache_tag`), 150-230 (dtype resolution, weight setup),
  275-360 (latent cache creation, `kv_bytes_per_token`), 382-520 (`forward_prefill`), 585-620 (start of
  `forward_decode`, to confirm prefill/decode split). **Skimmed/skipped**: 52-150, 230-275, 520-585,
  620-740 (fused-kernel decode path, group-latent slicing) — decode-only, does not affect perplexity.
- `palu_tt/report.py`: 220-232, 347-390, 526-545, 594-680, 811-880, 1140-1180, 1267-1310, 1340-1450
  (how a ppl JSON becomes a table cell). **Skipped** the rest (bench/profile/figure rendering).
- `palu_tt/eval_tasks.py`: 115-135 only. `palu_tt/decode_check.py`: 40-60, 100-105 only. `palu_tt/bench.py`:
  grep for `build_model` call sites only. All three read solely to compare model-build arguments.
- `scripts/run_ppl_cpu.sh`, `scripts/env.sh`: whole files.
- `tests/conftest.py` (1-22), `tests/test_reference.py` (1-63), `tests/test_ppl_harness_device.py` (1-12),
  `tests/test_model_device.py` (1-97): whole files. `tests/test_report.py`: grepped for `ppl` only.
- `third_party/tt-metal/models/tt_transformers/tt/model.py`: 620-660, 940-1035 (`get_last_token=-1`
  semantics, `concat_host_output`). `.../model_config.py`: 590-630, 3770-3890 (`TOKENIZER_PATH`,
  `create_tokenizer`). Nothing else from tt-metal.
- `third_party/palu/run_ppl_eval.py`: whole file (to compare protocols).
- Docs: `docs/report_palu_on_tt.md` 1-70, `docs/results/stage1.md` 1-200, `docs/results/stage2f.md` 1-100
  and 155-170, `docs/results/stage2c.md` 740-800, `README.md` 70-130.
- Data: all 78 `docs/results/raw/ppl*.json` (tabulated), the HF snapshot's `config.json` and
  `model.safetensors.index.json`.

## Findings

### [Critical] The report's CPU cross-check claim is contradicted by the repo's own raw data

**File:** `docs/report_palu_on_tt.md:14` ("A 20-window fp32 CPU run of the same low-rank model serves as
a harness cross-check **for every row** (it agrees **within 2 %**)"), against `palu_tt/eval_ppl.py:59-70`
and `docs/results/raw/ppl_cpu_*.json`.

**What.** Only four CPU rows exist in the whole repo: `ppl_cpu_baseline` (7.3573), `ppl_cpu_palu_full_50`
(12.7592), `ppl_cpu_palu_full_70` (9.0588), `ppl_cpu_svd_r64` (776.4337), plus a 5-window
`palu_full_70_quick`. There are ~20 device rows at 141 windows and ~45 more at 40 windows; none of the
G-LRD rows, none of the `_f2048` rows, none of the format/scale sweep rows, and none of the 40-window
frontier rows has a CPU counterpart. Second, the agreement is not within 2 % everywhere: `svd_r64` is
776.43 (CPU) against 703.65 (device) — **10.3 %**. Third, and worst, where the claim does hold the
comparison is a 20-window CPU number against a 141-window device number, and the repo itself measured
the window-count artifact at **2.0 %** (`ppl_baseline_20w.json` = 7.3996 vs `ppl_baseline.json` = 7.2540).
A "cross-check" whose tolerance equals its own confound has no resolving power.

`docs/results/stage1.md:120-170` states all of this correctly (it names the 20-window device number, the
2.0 % artifact, and marks `svd_r64` as "same order; both degenerate"). The top-level report summarised
it into a sentence that the stage document does not support.

**Why it matters.** This is the only independent-implementation check offered for every perplexity in
the report, including the headline 7.25 / 9.03 / 12.93 / 8.14 table in section 3 and section 6. As
written it claims validation coverage that does not exist.

**How to verify.**
`python3 -c "import json,glob;[print(f, json.load(open(f))['ppl'], json.load(open(f))['windows']) for f in sorted(glob.glob('docs/results/raw/ppl_cpu_*.json'))]"`
then compare against `ppl_svd_r64.json` (703.648) and `ppl_baseline_20w.json` (7.3996).

**Suggested fix.** Replace the sentence with the stage-1 wording: "Four rows were cross-checked against
a 20-window fp32 CPU run of the same low-rank model; on the same 20 windows the stock row agrees to
0.57 % (device 7.400 vs CPU 7.357). The window-count difference alone moves the stock row by 2.0 %, so
the 141-window device numbers and the 20-window CPU numbers must not be differenced directly."

---

### [Important] The headline 141-window Palu perplexities are prefill-blind and the documented command no longer reproduces them (and would overwrite them)

**File:** `palu_tt/eval_ppl.py:96-99` (`--no-latent-quant-in-prefill`, `dest=latent_quant_in_prefill`,
`action="store_false"`, so the **default is True**), `:116-121`, `:124` (`out = RAW / f"ppl_{a.row}.json"`);
`palu_tt/attention.py:425-449` (the round trip); `README.md:96-97`.

**What.** Every 141-window Palu row that feeds report section 3 and section 6 — `ppl_palu_full_50.json`
(12.931), `ppl_palu_full_70.json` (9.028), `ppl_glrd_50.json`, `ppl_glrd_70.json`, `ppl_glrd_50_f256.json`,
`ppl_glrd_70_f256.json` (8.135), `ppl_identity.json`, `ppl_svd_r*.json` — carries **no**
`latent_quant_in_prefill` key, i.e. it was produced before stage 2f, when prefill never round-tripped the
latents through the cache format. Today that flag defaults to `True`. Running the command the README
gives (`python -m palu_tt.eval_ppl --row svd_r64 --kind palu --palu-dir compressed/svd_r64`) therefore
measures a *different* model than the recorded number, and writes to the same filename, destroying the
recorded number. The flag to reproduce the old behaviour exists (`--no-latent-quant-in-prefill`) but
appears in no documented reproduction command.

**Why it matters.** Affects reproducibility of every headline Palu perplexity in the report
(12.93 / 9.03 / 8.22 / 8.14 / 10.72 / 10.56 / 703.6 / 79.4 / 502.7) and of `ppl (device, 141 windows)`
in `stage1.md`, `stage2.md`, `stage2b.md`. `stage2f.md:69` says "Rows still carrying a 141-window pre-2f
number are re-measured in Task 6/7" — that re-measurement is not in the raw directory at HEAD.

**How to verify.** `grep -L latent_quant_in_prefill docs/results/raw/ppl_*.json` lists the pre-2f rows;
cross-check against the report's tables.

**Suggested fix.** Add `--no-latent-quant-in-prefill` to the README/stage reproduction commands for the
pre-2f rows, or re-measure them and keep the old files under a `_blind` row name. Long term: make
`eval_ppl` refuse to overwrite an existing `ppl_<row>.json` whose recorded flags differ from the current
run's (or fold the flags into the filename).

---

### [Important] Blind and cache-faithful perplexities are mixed in one column, and the headline "bfp4 V costs +0.209 ppl" is a cross-protocol delta

**File:** `docs/results/stage2f.md:5-63` (the `ppl (windows)` column) and `:88-91`; `docs/report_palu_on_tt.md:64`
("for a perplexity of 9.24" set against 9.03 in the same table); `palu_tt/report.py:1267-1282` (`_row_is_blind`),
`:1420-1432`.

**What.** `_row_is_blind` renders a row as `blind` only when it is bfp4-latent-suffixed *and* either
declares `latent_quant_in_prefill: false` or predates the field. A pre-2f **bfp8** row is rendered as a
plain number. So in one column the frontier table puts `palu_full_70` 9.028 (blind, 141) next to
`palu_full_70_lv4` 9.237 (faithful, 141) and `palu_full_70_l4` 9.923 (faithful, 141), and the
recommendation section differences them: "vs `palu_full_70` … for **+0.209 ppl**". By the document's own
correction (`stage2f.md:69`, the bfp8 round trip is −0.033 ppl), the same-protocol delta would be ≈ +0.242,
about 16 % larger. Report section 6's "9.24" vs "9.03" repeats the same cross-protocol pair.

**Why it matters.** The stage 2f recommendation ("bfp4 on V is nearly free") and the report's closing
paragraph rest on this delta.

**How to verify.** `python3 -c "import json;print(json.load(open('docs/results/raw/ppl_palu_full_70.json')).keys())"`
(no `latent_quant_in_prefill`) versus `ppl_palu_full_70_lv4.json` (`true`).

**Suggested fix.** Either re-measure the pre-2f bfp8 141-window rows with the faithful prefill (the
cheapest fix, and the one `stage2f.md:69` already promises), or mark them `(blind)` in the column the way
the bfp4 rows are, so no reader differences two protocols.

---

### [Important] The −0.033 "bfp8 round trip" correction has no committed raw measurement

**File:** `docs/results/stage2f.md:69`, `:162`; `docs/report_palu_on_tt.md:12` ("the bfp8 storage of the
latents costs 0.03 perplexity (stage 2f measured the round trip)").

**What.** Exactly one raw file in the repo declares `latent_quant_in_prefill: false`
(`ppl_palu_full_50_l4.json`, 141 windows), and there is no 40-window blind/faithful bfp8 pair anywhere in
`docs/results/raw/`. The −0.033 figure quoted three times — and used to justify the protocol mixing in
the previous finding, and quoted in the report as the cost of bfp8 latent storage — cannot be recomputed
from the repo. `stage1.md:147` states that "the raw JSONs … are the only provenance these tables have".

Secondary oddity: the stated sign says the bfp8 round trip *lowers* perplexity by 0.033. Quantising the
latents to bfp8 and back (`attention.py:24-35`) should not improve the model; at 40 windows ±0.03 is
plausibly noise, but nothing in the repo lets a reader check which.

**Why it matters.** It is the numeric justification for mixing blind and faithful rows, and it appears in
report section 1 as a measured property of the port.

**How to verify.** `grep -l latent_quant_in_prefill docs/results/raw/ppl_*_w40.json | xargs grep -h latent_quant`
— every 40-window file says `true`.

**Suggested fix.** Commit the blind/faithful 40-window pair (two files, e.g. `ppl_palu_full_70_w40_blind.json`),
or drop the numeric claim.

---

### [Important] The output filename is `ppl_<--row>.json` with no window count, seq-len or format component

**File:** `palu_tt/eval_ppl.py:110`, `:124`.

**What.** `--row` is a free-form string and is the only thing that determines which file is written.
Nothing ties the filename to `--windows`, `--seq-len`, `--latent-dtype`, `--latent-scale` or
`--latent-quant-in-prefill`. `stage1.md:126-128` records that a 20-window run already overwrote the
141-window `ppl_baseline.json` once, and the workaround was a hand-chosen row name (`baseline_20w`). The
naming convention that keeps `_w40`, `_l4`, `_u`, `_lv4` rows apart is entirely a human habit.
`palu_tt/report.py:1305` warns when a *frontier* row's name disagrees with its own format fields, but
nothing checks window count, and nothing at all checks the stage 1/2/2b rows.

**Why it matters.** Any headline number can be silently replaced by a shorter or differently-configured
run. Combined with the previous finding (the current default differs from the recorded runs), this is a
live hazard, not a hypothetical.

**How to verify.** Read `eval_ppl.py:124`; then note `ppl_palu_full_70.json` (141 windows) and
`ppl_palu_full_70_w40.json` (40 windows) are distinguished only by the string the operator typed.

**Suggested fix.** Derive the suffix in code (`_w{windows}` whenever `--windows` is given) and refuse to
overwrite an existing file whose recorded `windows`/format block differs, unless `--force` is passed.

---

### [Important] No test exercises the scoring math; the one test that looks like it does asserts against a copy of the code under test

**File:** `tests/test_model_device.py:37-57`, `tests/test_ppl_harness_device.py:1-12`; there is no
`tests/test_eval_ppl.py`.

**What.** `eval_ppl.wikitext2_ids`, `_head`, `_nll_from_hidden` and `cpu_ppl` — the functions that turn a
device forward pass into every reported perplexity — have no CPU test at all.
`test_prefill_hidden_states_feed_the_lm_head` is documented as guarding exactly this ("Task 8's
perplexity harness consumes `prefill_hidden_states` exactly this way"), but it **re-implements** the
RMSNorm and the LM head inline (lines 48-52) instead of calling `eval_ppl._head` / `_nll_from_hidden`. A
wrong epsilon, a wrong norm weight key, a wrong chunk boundary, a wrong target shift or a wrong token
count inside `eval_ppl` cannot fail it. It also checks only the last of 128 positions, at PCC 0.99.
The only test that calls `device_ppl` (`test_ppl_harness_device.py`) asserts `windows == 2`,
`tokens == 2*2047` and `3.0 < ppl < 20.0`; that band would pass with a wrong `rms_norm_eps`, a changed
join string in `wikitext2_ids`, a tokenizer that stopped emitting BOS, or `--windows n` changed to mean
"the last n windows" (which would silently break the `_w40` comparability claim). It requires the device,
so it cannot run in CI.

**Why it matters.** Every number in the "ppl (device, 141 windows)" and "ppl (windows)" columns of four
results documents and the report comes out of untested host code.

**How to verify.** `grep -rn "_nll_from_hidden\|wikitext2_ids\|cpu_ppl" tests/` returns only
`tests/test_ppl_harness_device.py`.

**Suggested fix.** A pure-CPU test file: (a) `_nll_from_hidden` on random `[2048, 64]` hidden states
against a one-shot `cross_entropy` (I ran this; they agree to 4.6e-8 relative); (b) a shift test where
`h[s]` predicts `targets[s]`, built so predicting the wrong offset fails; (c) `wikitext2_ids` asserting
`shape[1] == 289077`, `ids[0,0] == 128000` and exactly one special token in the stream; (d) `cpu_ppl` on a
tiny `LlamaForCausalLM` against a hand-rolled loop, including that `--windows k` takes the *first* k.

---

### [Important] The CPU cross-check cannot run without tt-metal, and neither can its tests

**File:** `palu_tt/eval_ppl.py:12` (`from palu_tt import generate, hf` at module scope; `generate.py:3`
does `import ttnn`), `tests/conftest.py:6` (`from palu_tt import device as dev`; `device.py:3` does
`import ttnn`).

**What.** `python -m palu_tt.eval_ppl --cpu` imports `ttnn` before it parses arguments, and collecting
*any* file under `tests/` imports `ttnn` through `conftest.py`, including the four CPU-only tests in
`tests/test_reference.py`. The device import is otherwise carefully deferred (`eval_ppl.py:112` imports
`palu_tt.device` and `palu_tt.model` lazily inside the device branch), so the coupling looks accidental.

**Why it matters.** The fp32 CPU reference is the study's independent implementation; a reviewer on a
machine without a Tenstorrent stack cannot run it, or the tests that guard it. (For the same reason I did
not execute `tests/test_reference.py` under the no-ttnn-import rule for this audit — its assertions are
reviewed by reading below.)

**How to verify.** `head -12 palu_tt/eval_ppl.py; head -8 palu_tt/generate.py; head -8 tests/conftest.py`.

**Suggested fix.** Move `from palu_tt import generate` into `device_ppl` (it is the only user), and make
`conftest.py`'s `mesh_device` fixture import `palu_tt.device` inside the fixture body.

---

### [Important] `ppl_palu_full_50_l4.json` records a format that provably did not affect its number, and the raw JSONs are hand-edited

**File:** `docs/results/raw/ppl_palu_full_50_l4.json`; `palu_tt/eval_ppl.py:134-147`.

**What.** The file records `latent_dtype: "bfp4"` with `latent_quant_in_prefill: false`, and its `ppl`
and `nll` are **bit-identical** to `ppl_palu_full_50.json` (12.930530567327075 / 738767.13671875) — as
they must be, since with the blind prefill the latent cache format cannot reach the forward pass. So a
device row's `latent_dtype` field is an echo of a CLI flag, not a property of the measurement; the only
thing that stops a reader treating it as bfp4 evidence is a `note` field. `eval_ppl` never writes `note`
(see the `r.update(...)` key list at `:144-145`) — that field, and the six `note`s in
`profile_*_fk2r_*.json`, were added by hand, and one of the profile notes says the file was renamed. The
raw directory is described as the tables' only provenance, yet nothing in the schema distinguishes a
machine-written field from a hand-written one, and there is no timestamp, git commit, tt-metal version or
dump hash in any ppl JSON. `report._commit()` stamps the commit at which the *markdown was rendered*, not
at which the number was measured.

**Why it matters.** Provenance of every ppl row. `report.py` handles this specific row correctly (it
renders `blind`), so no published number is currently wrong because of it — but the mechanism that
protects it is a hand-written annotation.

**How to verify.** `diff <(jq -S 'del(.row,.latent_dtype,.latent_quant_in_prefill,.note)' docs/results/raw/ppl_palu_full_50_l4.json) <(jq -S 'del(.row)' docs/results/raw/ppl_palu_full_50.json)`.

**Suggested fix.** Have `eval_ppl` write `note: null`, a UTC timestamp, `git rev-parse HEAD` and the tt-metal
version into every result, and emit `latent_dtype: null` (or a `latent_dtype_effective` field) whenever
`latent_quant_in_prefill` is false, so the flag echo cannot be mistaken for a measurement.

---

### [Minor] `tokens` is assumed, not measured

**File:** `palu_tt/eval_ppl.py:52-53` (device), `:68-69` (CPU).

**What.** `_nll_from_hidden` sums over `h.shape[0] - 1` positions; the caller then does
`tokens += seq_len - 1` unconditionally. The two are independent. If `prefill_hidden_states` ever
returned fewer than `seq_len` rows (it slices `[:ids.shape[1]]` with no assertion,
`generate.py:36`), the NLL would cover fewer positions while the denominator still counted 2047, and the
perplexity would be silently *understated*. Not reachable today — I verified all 78 raw files satisfy
`tokens == windows*(seq_len-1)` and `exp(nll/tokens) == ppl` exactly — but the invariant is unasserted.

**How to verify.** Read the two lines; `assert h.shape[0] == seq_len` in `device_ppl` would close it.

**Suggested fix.** Have `_nll_from_hidden` return `(nll, n_positions)` and accumulate both.

---

### [Minor] `device_ppl` runs outside `torch.no_grad()`, unlike `cpu_ppl`

**File:** `palu_tt/eval_ppl.py:43` vs `:59`. No graph is actually built (the device tensor and the
safetensors weights all have `requires_grad=False`), so this is not a correctness bug, only an asymmetry
that would become one if `_head` ever loaded a parameter. Add `@torch.no_grad()` to `device_ppl`.

---

### [Minor] Hard-coded `torch.set_num_threads(12)` on a shared host

**File:** `palu_tt/eval_ppl.py:44`, `:61`; `palu_tt/reference.py:75`; `tests/conftest.py:90`. Overrides
`OMP_NUM_THREADS` from `scripts/env.sh`. The host side of a 141-window device run is roughly 76 TFLOP of
fp32 `[512,4096] @ [4096,128256]` matmuls; on this shared box it will occupy 12 cores for the whole run.
It cannot corrupt a perplexity (nothing is timed), but it can corrupt someone else's untraced-decode
benchmark, which `stage1.md:139-146` documents as having happened once. Read the thread count from the
environment.

---

### [Minor] `_head` assumes untied embeddings; no guard

**File:** `palu_tt/eval_ppl.py:26-29`. `t["lm_head.weight"]` raises `KeyError` on a checkpoint with
`tie_word_embeddings: true`. Correct for this model — I confirmed the snapshot's `config.json` has
`tie_word_embeddings: false`, `rms_norm_eps: 1e-05`, and that `lm_head.weight` and `model.norm.weight`
are both in `model.safetensors.index.json` — but a fallback to `model.embed_tokens.weight` plus an
explicit assertion would document the assumption.

---

### [Minor] Reproducibility footguns in the harness

**File:** `palu_tt/eval_ppl.py:14` (`RAW = Path("docs/results/raw")` is relative to the *cwd*; only
`scripts/env.sh`'s `cd "$ROOT"` makes it correct), `:21` (`load_dataset` with no `revision=` pin),
`:79` (`--seq-len` accepted unvalidated, although `prefill_hidden_states` requires a multiple of 128 —
`attention.py:387` asserts it, so a bad value fails loudly rather than silently), `:112`+`:137` (the same
`from palu_tt import model as pm` import written twice in one function).

---

### [Minor] Two small wording overstatements in the report

**File:** `docs/report_palu_on_tt.md:59` — "The identity factorisation … reproduces the baseline **to
three decimals** (7.255 vs 7.254)". The raw values are 7.2552 and 7.2540; they agree to two decimals and
differ in the third. `stage1.md:155` states it correctly (+0.0013, 0.018 %).
**File:** `docs/report_palu_on_tt.md:14` — "Three instruments, all run on the device through the same
model build". `eval_ppl` builds `max_batch_size=1, max_seq_len=--seq-len` with `use_fused_k=False`;
`eval_tasks` builds `max_batch_size=1, max_seq_len=2048` and *does* expose `--fused`; `bench` builds
`max_batch_size=32, max_seq_len=ctx`. The builds differ; what is true (and is what the sentence wants to
say) is that prefill numerics are identical across them, which I verified — `forward_prefill`
(`attention.py:382-517`) never reads `use_fused_k`, `rot_mode`, `table_mcast`, `group_heads` or
`recon_len`.

---

### [Minor] `test_cpu_lowrank_llama_uses_palu_factors` cannot fail on a wrong factorisation

**File:** `tests/test_reference.py:48-57`. It asserts `isinstance(kp, LowRankProj)`, `kp.VT.shape == (16, 64)`
and `torch.isfinite(lm(ids).logits).all()`. A transposed `U`, a swapped head order in the
`reshape(n_kv*r, dim)`, or a wrong einsum index would all produce finite logits of the right shape and
pass. The identity case (`test_cpu_lowrank_llama_identity_equals_hf`) does pin the einsum for `U = I`,
which is a real constraint, but nothing pins the non-identity path against a reference.
Suggested fix: compare against an explicit `x @ (U_h @ VT_h).T` per head.

---

### [Minor] stage 2c's caveat about Palu's own protocol is over-cautious (the two protocols are arithmetically identical)

**File:** `docs/results/stage2c.md:789-791` — "the GPU perplexity protocol is Palu's script, whose window
construction is not byte-identical to ours; the 0.04 baseline offset absorbs that difference together
with the hardware's". Reading `third_party/palu/run_ppl_eval.py:40-116`: same `"\n\n".join(testdata["text"])`,
same `tokenizer(..., return_tensors="pt")`, same `nsamples = numel // 2048` non-overlapping windows, same
`logits[:, :-1]` / `labels[:, 1:]` shift. Its aggregation is `exp(Σ_w mean_w · 2048 / (N·2048))` =
`exp(mean of per-window mean NLL)`; ours is `exp(Σ_w nll_w / Σ_w 2047)`. Because every window scores
exactly 2047 tokens, these are **identical** (I checked the algebra numerically). So the 7.213 vs 7.254
gap is not absorbing a protocol difference — it is the port, and it agrees to 0.001 with the independent
CPU-vs-device measurement below. Worth correcting *in the study's favour*.

## Tests in this area

| test file | what it really asserts | could it pass with the bug? |
|---|---|---|
| `tests/test_reference.py::test_ref_attention_identity_matches_hf_layer` | `PaluRefAttention` at identity rank equals HF's `LlamaAttention` output (`atol 1e-4`), plus incremental-decode consistency and cache shape. A genuine assertion against an independent implementation. | No — but `PaluRefAttention` is **not on the perplexity path at all**; `cpu_ppl` goes through `build_cpu_lowrank_llama`/`LowRankProj`. It guards `tests/test_attention_device.py` only. |
| `…::test_cpu_lowrank_llama_identity_equals_hf` | `LowRankProj` with identity factors reproduces stock HF logits (`atol 1e-4`). This is the real guard on the CPU cross-check's plumbing (head ordering, reshape, einsum) at `U = I`. | No, for the identity case. Yes for anything that only shows up at `r < head_dim` (e.g. a rank-slicing error). |
| `…::test_cpu_lowrank_llama_uses_palu_factors` | `isinstance`, one shape, `isfinite`. | **Yes.** Would pass with a transposed or mis-ordered up-projection. See Minor finding above. |
| `…::test_identity_lowrank_reports_group_size_one` | `group_size == 1` on a 3-line helper. | Trivially strong; guards nothing numeric. |
| `tests/test_model_device.py::test_identity_path_matches_stock_model` | Prefill-logit PCC 0.999 stock-vs-identity plus 32/32 identical greedy tokens. Strong, and it is the evidence behind the "identity reproduces the baseline" claim. | No. (Device test; not run here.) |
| `…::test_prefill_hidden_states_feed_the_lm_head` | `h.shape == (128, args.dim)`, then PCC 0.99 between a *locally re-written* norm+lm_head over `h[-1]` and `prefill_last_logits`. Guards the N300 residual sharding, which is real and valuable. | **Yes** for the perplexity path: it never calls `eval_ppl._head` or `_nll_from_hidden`, so it is an assertion against a copy of the code under test. A wrong `eps`, a wrong key in `_head`, or a wrong chunk/target offset in `_nll_from_hidden` all survive it. Also only one position, PCC 0.99. |
| `…::test_r64_model_tracks_cpu_lowrank_model` (`slow`) | Prefill-logit PCC ≥ 0.95 device-vs-CPU on `svd_r64`, plus teacher-forced argmax ≥ 0.85. Real device-vs-reference check with a justified bar. | No, for what it covers — but it is a *single 6-token prompt*, not the perplexity windows, and it is `@pytest.mark.slow` (needs the 32 GB fp32 model), so it is not part of any routine run. |
| `tests/test_ppl_harness_device.py::test_device_ppl_two_windows_is_sane` | `ids.shape[1] > 200_000`, `windows == 2`, `tokens == 2*2047`, `3.0 < ppl < 20.0`. The `tokens` assertion does pin the 2047-per-window convention. | **Yes** for most of what matters: the ppl band is ~7x wide, so a changed join string, a lost BOS, a wrong `rms_norm_eps`, or `--windows n` meaning "last n" all pass. Device-only, so it never runs in CI. |
| `tests/conftest.py` | Seeds torch/numpy/random; session-scoped `mesh_device`. | Imports `ttnn` at collection time for every test file, including the CPU-only ones (Important finding above). |
| *(absent)* | No test of `wikitext2_ids`, `_head`, `_nll_from_hidden`, `cpu_ppl`, the window slicing, or the `--windows k` = "first k" contract. | — |

## Claims traced

| documented claim | code implementing it | status |
|---|---|---|
| "wikitext-2-raw-v1 test split, joined and tokenised once" (`report:14`, `stage1:114`) | `eval_ppl.wikitext2_ids:20-23`, `"\n\n".join(ds["text"])`, one `tokenizer(...)` call | **verified** — reproduced on CPU: 4358 rows, 1,294,336 chars |
| "141 non-overlapping windows of 2048 tokens (288,627 scored tokens)" | `device_ppl:47-53`, `windows = N // seq_len`, `chunk = ids[:, w*2048:(w+1)*2048]`, `tokens += 2047` | **verified** — my tokenisation gives N = 289,077 → 141 windows, 141×2047 = 288,627, matching every 141-window JSON exactly; 309 tail tokens dropped |
| "Only window 0 carries a BOS token" | tokenizer post-processor, not code in this repo | **verified** — exactly one special token in the whole stream, id 128000 at index 0; no EOS, no PAD |
| "`--windows 40` takes the first 40 windows in order" (implicit in every `_w40` comparison) | `device_ppl:47-49`, `min(n_windows, N//seq_len)` then `for w in range(windows)` with offset `w*seq_len` | **verified** — `_w40` rows are scored on identical windows and are comparable across dumps |
| "The device forward … returns the pre-final-norm hidden states" | `generate.prefill_hidden_states:29-38` with `get_last_token=-1`; tt-metal `model.py:1010-1017` returns `x` *before* `self.norm(...)` | **verified** |
| "the final RMSNorm, the LM head and the cross-entropy run on the host in fp32" | `eval_ppl._head:26-29` (`model.norm.weight`, `lm_head.weight`, `rms_norm_eps` from the real `config.json`), `_nll_from_hidden:32-40` (`h.float()`, `rsqrt(mean(h²)+eps)*norm_w`, `@ lm_w.T`, `cross_entropy`) | **verified** — real weights, real eps (1e-05), fp32 throughout. The LM head is the checkpoint's own `lm_head.weight` (`tie_word_embeddings: false`; the key exists in the index) — note this is the *host* fp32 head, never the device's bfp8 one |
| Token-weighted loss, `ppl = exp(Σnll / Σtokens)` computed once at the end | `device_ppl:48-56` / `cpu_ppl:64-70`; `math.exp` appears once outside the loop (the in-loop `math.exp` at `:55` is a progress print only) | **verified** — and every raw file satisfies `exp(nll/tokens) == ppl` and `tokens == windows*(seq_len-1)` exactly |
| Logit/target alignment (no off-by-one) | `_nll_from_hidden` chunk loop: `h[s:min(s+512,S-1)]` against `targets[s:s+len]`, `targets = chunk[0,1:]` | **verified** — 2047 positions scored, position `s` predicts token `s+1`; the 512-chunking equals a one-shot cross-entropy to 4.6e-8 relative on synthetic data |
| Perplexity "cannot see a decode kernel at all" (`report:16`, `README:112`) | `attention.forward_prefill:382-517` reads none of `use_fused_k`, `rot_mode`, `table_mcast`, `group_heads`, `recon_len`; prefill runs SDPA on freshly computed K/V and only *writes* the caches (`fill_cache`), never reads them, so windows cannot contaminate each other | **verified** |
| The JSON's `k_latent_dtype`/`v_latent_dtype`/`kv_dtype` describe the caches the run actually held | `eval_ppl:113`/`:138-142` both go through `model.resolve_latent_dtype_names`; `attention.py:167-169` resolves the same way and `:275-283` creates `layer_past` with exactly those dtypes; `set_kv_cache_dtype` (`model.py:98-138`) makes `TensorGroup.KV_CACHE` match `--kv-dtype` | **verified** — one resolution rule, used for both the build and the record |
| …but a row's `latent_dtype` can name a format the number did not depend on | `latent_quant_in_prefill=False` ⇒ `_roundtrip` is skipped (`attention.py:425`) ⇒ prefill is blind to the cache format | **contradicted as a measurement claim** — `ppl_palu_full_50_l4.json` (bfp4) is bit-identical to the bfp8 row |
| "A 20-window fp32 CPU run … for every row (it agrees within 2 %)" (`report:14`) | `cpu_ppl:59-70` + `scripts/run_ppl_cpu.sh` (default `windows=20`) | **contradicted** — 4 rows only; `svd_r64` off by 10.3 %; comparison mixes 20- and 141-window runs |
| "the window-count difference alone moves the stock row by 2.0 % — device 7.400 on those same 20 windows" (`stage1:125-128`) | `ppl_baseline_20w.json` vs `ppl_baseline.json` | **verified** — 7.3996 vs 7.2540 = 2.007 % |
| "the bfp8 storage of the latents costs 0.03 perplexity" (`report:12`) | none in `docs/results/raw/` | **not verified** — no blind/faithful pair is committed |
| "puts the whole port at 0.04 perplexity on the baseline (7.21 against 7.25)" (`report:12`) | `third_party/palu/run_ppl_eval.py:40-116` vs `eval_ppl` | **verified as like-for-like** — the two protocols are arithmetically identical (see the Minor finding); and the figure is independently corroborated (below) |
| CPU reference uses "the same windows and scoring" as the device | `cpu_ppl` reuses `wikitext2_ids` and the same slicing/shift; the device tokenizer is `AutoTokenizer.from_pretrained(TOKENIZER_PATH=HF_MODEL)` (tt-metal `model_config.py:598`, `:3816-3823`) and the CPU tokenizer is `AutoTokenizer.from_pretrained(snapshot_dir)` — the same files under `HF_HUB_OFFLINE=1` | **verified**, with one difference: the *window count* (20 vs 141), which is where the whole 7.36-vs-7.254 discrepancy lives |

### Question 4 answered explicitly: 7.36 (CPU) vs 7.254 (device) vs 7.213 (Palu fp16 GPU)

It is fully explained from the code and the data, and it is **not** a CPU-vs-device gap.
`ppl_cpu_baseline.json` records `windows: 20, tokens: 40940` — `scripts/run_ppl_cpu.sh:5` defaults
`windows="${4:-20}"` and the README invokes it with no fourth argument. `ppl_baseline.json` is 141
windows. The repo contains the controlled comparison: `ppl_baseline_20w.json`, the **device** on the same
first 20 windows, is 7.3996.

- window count, device only: 7.3996 (20w) → 7.2540 (141w) = **−0.146, 2.01 %** — this is the entire
  "7.36 vs 7.25" appearance.
- device vs fp32 CPU, same 20 windows, same scoring code path: 7.3996 − 7.3573 = **+0.0423 (0.57 %)**.
- device vs Palu's fp16 GPU script, 141 windows: 7.254 − 7.213 = **+0.041**.

The two independent estimates of the port's arithmetic cost (bfp8 weights, bf16 activations, TP 2) agree
to 0.001 ppl, which is a genuinely strong result the documents currently bury. Note the device row's LM
head is host fp32 (`_head`/`_nll_from_hidden`), so this 0.042 is the transformer body alone; the GPU
number carries an fp16 head, making the comparison mildly conservative in the study's favour.

## Verdict

The perplexity measurement itself is sound: I verified the window construction (141 × 2048, 288,627
scored tokens, single BOS) against the actual dataset, the shift and the 512-chunked host head against a
one-shot reference, the pre-final-norm hidden states against tt-metal's own code, the token-weighted
aggregation, the first-k semantics of `--windows`, and the internal consistency of all 78 raw files —
and Palu's own evaluation script turns out to be arithmetically the same protocol, so the 7.213/7.254
comparison is like-for-like. The problems are in what the documents say *about* those numbers and in how
the numbers are stored: the report's "cross-check for every row, within 2 %" is contradicted by the
repo's own four CPU files (one of which is 10.3 % out), the headline 141-window Palu rows were measured
under a prefill semantics that is no longer the default and that the documented command would silently
overwrite, and blind and cache-faithful rows are differenced against each other in stage 2f and in
report section 6 on the strength of a −0.033 correction with no committed raw measurement. Test coverage
is the weakest link: the entire host scoring path has no CPU test, and the one device test that claims to
guard it asserts against an inline copy of the formula rather than against `eval_ppl` itself.
