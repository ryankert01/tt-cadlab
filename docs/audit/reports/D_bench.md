# Audit D: benchmarks and fidelity harnesses

Repo: `/home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9` @ 40b9a15. Read-only audit; no device
touched, no test executed (see note at the end of "Scope read"), no file written inside the repo.

## Scope read

Primary (my area), read in full:
- `palu_tt/bench.py`: 1-475 (all).
- `palu_tt/decode_check.py`: 1-138 (all).
- `palu_tt/profile_decode.py`: 1-78 (all).
- `palu_tt/profile_report.py`: 1-165 (all).
- `palu_tt/profile_fit.py`: 1-85 (all).
- `palu_tt/profile_plots.py`: 1-297 (all).
- `palu_tt/device.py`: 1-28 (all).
- `scripts/device_free.sh`: 1-23 (all); `scripts/run_device.sh`: 1-24 (all); `scripts/env.sh`: 1-14 (all).
- `tests/test_bench_utils.py`: 1-279 (all); `tests/test_profile_report.py`: 1-57 (all);
  `tests/test_profile_fit.py`: 1-38 (all); `tests/conftest.py`: 1-21 (all); `pytest.ini`.

Supporting, read in part (to verify what the harnesses actually drive):
- `palu_tt/attention.py`: 505-560, 585-740 (decode path, `_recon_len`, cache init at 274-284,
  `kv_bytes_per_token` 352-358, `record_group_heads` site 330-350). The prefill path (382-505) was
  skimmed only for the cache-write/`latent_quant_in_prefill` block (419-440).
- `palu_tt/model.py`: `resolve_latent_dtype*`, `resolve_rot_switches`, `build_model`,
  `set_kv_cache_dtype`, `apply_palu_flags`, `set_recon_len`, `kv_bytes_per_token_per_layer` (lines 20-193).
- `palu_tt/generate.py`: 1-62 (all). `palu_tt/fused.py`: `record_group_heads` / `group_heads_used` only.
- `palu_tt/report.py`: grep only (`_timing`, `_vs_stock`, `_sweep_cell`, `_scaling_decode_cell`) to see
  which JSON field reaches the tables. I did **not** audit report.py itself (another auditor's area).
- `third_party/tt-metal/models/common/utility_functions.py`: `comp_pcc` 488-533.
- Docs: `docs/report_palu_on_tt.md` (all 118 lines), `README.md` 20-60 / 93-300,
  `docs/results/stage1.md` (relevant sections + 120-160), `stage2.md` 1-150 / 280-370,
  `stage2b.md` 318-340 / 395-411, `stage2d.md` 1-140 + section 4, `stage2e.md` 1-70 + 95-160,
  `stage2f.md` grepped. `docs/superpowers/specs/*`: grepped for timing/protocol claims only
  (no spec states iteration counts) - **skimmed, not read in full**.
- Data: all 132 `bench_*.json`, all 19 `decode_check_*.json`, all `profile_*.json` headers plus the
  full op lists of `profile_baseline_4096`, `profile_palu_full_50_4096`, `profile_palu_full_50_fk2_4096`.
- Not read at all: `docs/kernel_guide.md`, `docs/results/stage2c.md`, the tt-metal C++ op and kernels,
  the device tests (`tests/*_device.py`).

Note on the permitted pytest command: I did **not** run it. `tests/conftest.py:7` does
`from palu_tt import device as dev`, and `palu_tt/device.py:3` imports `ttnn` at module level; in
addition `tests/test_bench_utils.py:7` imports `palu_tt.bench`, which imports `ttnn` and
`models.tt_transformers.*` at module level. Any `pytest tests/...` invocation therefore imports ttnn
into the Python process, which the task's hard rules forbid. All test claims below are from reading.

## Findings

### [Critical] The report's decode timing protocol ("20 timed iterations") is not what the code does
**File:** `palu_tt/bench.py:101-109` (`_time(fn, iters=10, warmup=3)`), `:136` (`row = _time(step)`),
`:215` (`_time(step, iters=20)`), `:270` (`_time(step, iters=5, warmup=2)`);
`docs/report_palu_on_tt.md:17`.

**What.** The report states: "Whole decode step at batch 32 ... 3 warm-up and 20 timed iterations,
device sync inside the timed region; the minimum is reported". The whole-step decode bench calls
`_time(step)` with the defaults, i.e. **3 warm-up and 10 timed iterations**. 20 is the count used by
the *attention-only* microbench; prefill uses 2 warm-up and 5 timed iterations at batch 1. Every
committed decode JSON records `"iters": 10` (e.g. `bench_decode_baseline.json`,
`bench_decode_palu_full_50_fk2dm.json`); every attention JSON records `"iters": 20`; every prefill
JSON records `"iters": 5`. `docs/results/stage1.md:129-131` states the counts correctly
("decode 3+10, attention-only 3+20, prefill 2+5 at batch 1"), so the externally-presented report
contradicts both the code and its own stage document.

**Why it matters.** It is the stated protocol for every decode number in the report (38.0 / 55.4,
42.8 / 65.0 / 94.5, 39.0 / 55.5 / 77.4, the 339 tok/s figure). A reader who tries to reproduce with
20 iterations gets a different (slightly lower) minimum, because min over 20 samples <= min over 10.

**How to verify.** `python3 -c "import json;print({c:r.get('iters') for c,r in json.load(open('docs/results/raw/bench_decode_baseline.json'))['data'].items()})"`
-> `{'1024': 10, '4096': 10, '8192': None}`; then read `bench.py:136` against `bench.py:101`.

**Suggested fix.** Change the report sentence to "3 warm-up and 10 timed iterations for the whole
step, 3+20 for the attention-only microbench, 2+5 at batch 1 for prefill", or pass `iters=20` to
`_decode_one`'s `_time` and re-measure (the latter changes every published decode number).

### [Critical] "the attention op itself is 8 ms" is the layer-0 microbench, not the profile it cites
**File:** `docs/report_palu_on_tt.md:46` (section 4, last paragraph); source data
`docs/results/raw/profile_palu_full_50_4096.json` and `docs/results/raw/bench_attention_palu_full_50.json`.

**What.** The report says "The stage 2d profile put numbers on why: at 4k the reconstruction block
costs 207 ms of the 265 ms step (matmul 91, RoPE 80, typecast 33) while the attention op itself is
8 ms." In the profile JSON it cites, the `attention` category at 4k for `palu_full_50` is
**22.118 ms** per step (SdpaDecode signatures 13.676 + 7.105 + 1.337), and `docs/results/stage2d.md:86`
and `:236` both print 22.12 ("The naive rows' attention op is ... 22.12 vs 24.24 ms at 4k").
The number 8 is `bench_attention_palu_full_50.json` at ctx 4096: `min_ms = 8.307` - the **layer-0
attention-module microbench** (which also contains QKV, cache writes, the K reconstruction, concat,
`wo` and two collectives; `palu_tt/attention.py:718-733`), i.e. a per-layer figure for the whole
block, not the attention op. It is also numerically ~= 265.34 / 32 layers = 8.29.

**Why it matters.** The sentence is the report's single quantitative account of where the vanilla
path's 265 ms goes. As written it understates the attention op by 2.7x and mixes two instruments in
one sentence (207 and 22 are device time over 32 layers; 8 is wall time for one layer; 265 is wall
time for the step).

**How to verify.**
`python3 -c "import json;d=json.load(open('docs/results/raw/profile_palu_full_50_4096.json'));print(d['categories']['attention'])"`
-> `{'ms': 22.118..., 'count': 32.0}`; and
`python3 -c "import json;print(json.load(open('docs/results/raw/bench_attention_palu_full_50.json'))['data']['4096']['min_ms'])"`
-> `8.307`.

**Suggested fix.** "...while the attention op itself is 22 ms (0.69 ms per layer)", or, if the
per-layer microbench is meant, say "the whole attention block is 8.3 ms per layer".

### [Critical] The grouped-dump decode-fidelity claim contradicts the raw `decode_check` JSONs
**File:** `docs/report_palu_on_tt.md:64` (section 5, "Speed" paragraph); data
`docs/results/raw/decode_check_glrd_*.json`.

**What.** The report says the grouped dumps run "with decode fidelity PCC 0.995 or better and 0.95
to 0.98 argmax agreement". Measured values:

| row | pcc_min | pcc_mean | argmax_agree |
|---|---|---|---|
| `glrd_50_fk2` (per-head layout, the row the sentence is about) | 0.99761 | 0.99910 | **0.9048** |
| `glrd_70_fk2` | 0.99783 | 0.99891 | 0.9841 |
| `glrd_50_fk2g` | **0.99473** | 0.99908 | 0.9524 |
| `glrd_50_fk2gdm` | 0.99790 | 0.99910 | 0.9365 |
| `glrd_50_f256_fk2` | 0.99750 | 0.99908 | 0.9365 |
| `glrd_70_f256_fk2` | 0.99774 | 0.99892 | 0.9365 |

The argmax range across grouped rows is **0.90 to 0.98**, not 0.95 to 0.98, and `glrd_50_fk2g`'s
`pcc_min` is 0.9947, below "0.995 or better". `docs/results/stage2b.md:318-323` states the 0.905
figure explicitly, so the report contradicts its own stage document.

**Why it matters.** Decode fidelity is one of the three evaluation protocols the report advertises
(section 1), and it is the *only* instrument that sees the fused decode kernel at all. Rounding
0.905 up into a "0.95 to 0.98" range overstates it.

**How to verify.**
`python3 -c "import json,glob;[print(f.split('/')[-1], json.load(open(f))['pcc_min'], json.load(open(f))['argmax_agree']) for f in sorted(glob.glob('docs/results/raw/decode_check_glrd*.json'))]"`

**Suggested fix.** Quote the measured range ("PCC_min 0.995 to 0.998 and 0.90 to 0.98 argmax
agreement"), and say which statistic (min or mean) "PCC 0.998 or better" in the same paragraph
refers to - the shipped `_fk2dm` rows have `pcc_min` 0.99485 (`palu_full_50_fk2dm`), which is not
"0.998 or better" either, although their `pcc_mean` is 0.9993.

### [Critical] The "max context at batch 32" column mixes power-of-two probes with 1k-bisected ones
**File:** `docs/report_palu_on_tt.md:70-77` (section 6 table) and `:17` (section 1 "Capacity: ...
found by bisection"); `palu_tt/bench.py:313-369`; data `docs/results/raw/bench_maxctx_*.json`.

**What.** `bench_maxctx_baseline.json`, `bench_maxctx_palu_full_50.json`,
`bench_maxctx_palu_full_70.json`, `bench_maxctx_palu_full_50_fk2.json`, `bench_maxctx_svd_*.json`
were produced by the *pre-bisection* harness: their `data` has no `probes`, no `max_context` and no
`limit`, only `max_context_batch32` = the largest power of two that allocated (4096 / 8192). The
G-LRD rows were probed with the current bisecting harness: `bench_maxctx_glrd_70_fk2.json` = 10240,
`bench_maxctx_glrd_50_fk2.json` = 14336. The report's section 6 prints 4096 / 8192 / 8192 / 8192 /
8192 / **10240** in one column, and section 1 says the capacity is "found by bisection".
`docs/results/stage2b.md:403-407` records this exact limitation ("`baseline`, `palu_full_50` and
`palu_full_70` still carry the older power-of-two probe's coarse figures ... at batch 32 the G-LRD
rows cannot be placed against them to better than a factor of two; they were not re-run"); the
report drops the caveat.

**Why it matters.** Printed side by side, 10240 (G-LRD 0.7) vs 8192 (Palu 0.5) reads as "the grouped
dump holds more context". The true ordering is the opposite: at batch 8, where both were bisected,
`palu_full_50_fk2` reaches 54272 and `glrd_70_fk2` 40960, and `glrd_50_fk2` (543 B/token) reaches
14336 at batch 32 - so `palu_full_50` (574 B/token) almost certainly sits near 13-14k, not 8192.
Section 4's "every compressed row allocates 8192 tokens of context at batch 32" is a true lower
bound; the section 6 table is not a comparison.

**How to verify.**
`python3 -c "import json;print(json.load(open('docs/results/raw/bench_maxctx_palu_full_50.json'))['data'])"`
(no `probes`/`limit` key, `max_context_batch32: 8192`, `failed_at: 16384`) against
`python3 -c "import json;print(json.load(open('docs/results/raw/bench_maxctx_glrd_70_fk2.json'))['data']['probes'])"`.

**Suggested fix.** Either re-run `--what maxctx` at batch 32 for `baseline`, `palu_full_50`,
`palu_full_70` (about 10 min each per the 2f spec) or mark the coarse cells, e.g. ">= 8192
(power-of-two probe only)", and repeat stage2b's caveat in the report.

### [Important] The timed "whole decode step" excludes host input prep, logits readback and sampling
**File:** `palu_tt/bench.py:124` (`host = model.prepare_decode_inputs_host(...)`, outside the loop),
`:129-133` (`step`), module docstring `:1-12`.

**What.** The timed region is exactly: `copy_host_to_device(host)` (a pre-built host tensor tuple),
`model.ttnn_decode_forward(...)` (32 layers + final norm + LM head + the vocab all-gather - the
profile confirms `lm_head` and the 128256-wide all-gather are inside), `dev.sync(md)`
(`ttnn.synchronize_device`, so device time *is* included), `ttnn.deallocate(logits)`. Excluded:
`prepare_decode_inputs_host` (hoisted to line 124, once for the whole loop), the logits readback
(`process_output_decode`, which `palu_tt/generate.py:48` uses in real generation) and any
argmax/sampling. The module docstring at `bench.py:6` claims the opposite: "host-side input
preparation is inside the timed region too (it is part of a decode step)". Only the H2D copy is.

**Why it matters.** The tokens/s numbers derived from these times (report section 5: "At 8k and
batch 32 that is 339 tokens per second"; `report.py:493` computes `batch * 1000 / min_ms`) are
device+dispatch only and will not be reached by a generation loop. It also means the same host input
(same token id, same position) is replayed for all 10 iterations, so the per-256-token graph
recompile that `set_recon_len` (`model.py:171-179`) forces in real generation never happens inside a
timed loop - deliberate and documented at `bench.py:123`, but it is another reason the number is a
floor, not a decode-loop rate.

**How to verify.** Read `bench.py:117-143`; compare with `palu_tt/generate.py:40-50`, which adds
`prepare_decode_inputs_host` and `process_output_decode` per step.

**Suggested fix.** Fix the docstring, and state in the report that the figure is "device step time
(H2D of the token, forward, device sync), excluding host input preparation, logits readback and
sampling".

### [Important] `profile_report --steps` is a hand-typed constant; the docstring claims a `meta.json` that does not exist
**File:** `palu_tt/profile_report.py:7-8` (docstring), `:147` (`p.add_argument("--steps", type=int,
default=3)`), `:151` (`aggregate(latest_csv(...), a.steps)`); `palu_tt/profile_decode.py:179`
(`--steps`, default 3).

**What.** The docstring says the aggregator "divides by the number of profiled steps (recorded by
`palu_tt.profile_decode` as `steps` in the run's `meta.json`, default 3)". `profile_decode` writes no
file at all, and `find . -name meta.json` returns nothing anywhere in the repo or under
`third_party/tt-metal/generated/profiler`. The divisor is whatever the operator types into
`profile_report --steps` (default 3), with no cross-check against the run.

**Why it matters.** If a profile is taken with `--steps 5` and reported with the default 3, every
`device_ms_per_step`, every category `ms` and every `count` is inflated by 5/3 with no error and no
trace - and the stage 2d/2e tables, the `profile_fit` per-chunk fits and the stage 2e "85.8 vs 88.5
us per chunk" parity claim all ride on those numbers. The profile JSON records only `name`, `steps`
and the CSV path: not the row, the context, the batch (`profile_decode.BATCH` is hard-coded 32), the
`--fused` phase, `--rot-mode`, `--table-mcast`, `--group-heads` or the dump. `profile_fit.py:204`
recovers the context from the *file name*.

**How to verify.** `grep -rn "meta.json" palu_tt/ ; find . -name meta.json` (both empty); then
`python3 -c "import json;d=json.load(open('docs/results/raw/profile_baseline_4096.json'));print(sorted(d))"`
-> only `categories, csv, device_ms_per_step, device_ms_per_step_chip1, name, ops, ops_per_step, steps`.

**Suggested fix.** Have `profile_decode` write `<artifacts>/meta.json` with its full argv (steps,
row, kind, ctx, fused/rot switches, batch) and have `profile_report` read it, falling back to
`--steps` only with a warning; copy that provenance into the profile JSON.

### [Important] `device_free.sh`'s "our own process holds /dev/tenstorrent" check can never fire
**File:** `scripts/device_free.sh:4` (`set -euo pipefail`), `:20-23`.

**What.** The check is
`if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q tenstorrent; then ... exit 1; fi`.
On a shared host `ls -l /proc/[0-9]*/fd` cannot read other users' `/proc/<pid>/fd` and exits 2. Under
`set -o pipefail` the pipeline's status is the rightmost *non-zero* status, so even when `grep`
matches (status 0) the pipeline reports 2, the `if` takes the false branch, and the gate passes.
Verified empirically on this host:
`bash -c 'ls -l /proc/[0-9]*/fd >/dev/null 2>&1; echo $?'` -> `2`, and
`bash -c 'set -euo pipefail; if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q socket; then echo T; else echo F; fi'`
-> `F` although 110 lines match.

**Why it matters.** `README.md:25-27` and `docs/results/stage1.md:132-134` both claim the gate
"refuses to start ... when one of our own processes holds `/dev/tenstorrent`". It does not; the gate
is the tt-smi telemetry check alone. Two of our own jobs can overlap if the first happens to be in a
low-clock phase, and the study's whole "measured on an idle board" argument rests on this gate.
(A secondary, smaller issue: `grep -q tenstorrent` matches any open path containing the string, not
just the device node, so it can also false-positive.)

**How to verify.** The two `bash -c` commands above, or `bash -x scripts/device_free.sh` with the
telemetry block commented out.

**Suggested fix.** `if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q '/dev/tenstorrent'; then` inside a
subshell with `set +o pipefail`, or better `pids=$(grep -l . /dev/null; ...)`; simplest robust form:
`if ls -l /proc/[0-9]*/fd 2>/dev/null | { grep -q "/dev/tenstorrent" || true; } ...` - or capture
`out=$(ls -l /proc/[0-9]*/fd 2>/dev/null || true)` first and grep the variable.

### [Important] The JIT kernel-cache guard hashes too little: shared headers and the stock op are never purged
**File:** `scripts/run_device.sh:33-45`.

**What.** The guard hashes `find "$KDIR" -type f` where
`KDIR=.../transformer/palu_sdpa_decode/device/kernels`, i.e. exactly five files
(`compute/palu_compute_common.hpp`, `compute/palu_sdpa_flash_decode.cpp`,
`dataflow/palu_dataflow_common.hpp`, `dataflow/palu_reader_decode.cpp`,
`dataflow/palu_writer_decode.cpp`). Those kernels `#include` sources **outside** `KDIR`:
`ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp`,
`.../sdpa_decode/device/kernels/dataflow/dataflow_common.hpp`,
`.../sdpa/device/kernels/compute/compute_common.hpp`,
`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, `ttnn/kernel_lib/tilize_helpers.hpp`,
`ttnn/kernel_lib/untilize_helpers.hpp`. Editing any of them changes the compiled kernel without
changing the hash, so the purge does not fire and (per the guard's own premise - tt-metal keys the
cache on path + defines + compile-time args, never source bytes) the **stale binary is measured**.
The purge pattern `-path '*/kernels/palu_*'` also only removes palu entries: the cache layout is
`~/.cache/tt-metal-cache/<hash>/kernels/{palu_sdpa_flash_decode,palu_reader_decode,palu_writer_decode}`
(verified), so the *stock* `sdpa_decode` entries that the baseline and naive rows run are never
purged even when a shared header changes under them.

**Why it matters.** This is precisely the failure the guard was added for (README:46-52, "stage 2e
finding"), and the stage 2e result - that `dst` + multicast take the step from 42.8/65.0/94.5 to
39.0/55.5/77.4 - is a kernel-source comparison. A stale-binary run would produce a wrong number
silently.

**How to verify.**
`grep -rh '#include "' third_party/tt-metal/ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/device/kernels | sort -u`
(shows the out-of-tree includes); `find ~/.cache/tt-metal-cache -maxdepth 6 -type d -name 'palu*'`
(shows the layout the purge pattern targets). The stamp currently matches the sources
(`~/.cache/palu_tt/palu_kernels.sha256` == the recomputed hash), so the last run was not stale.

**Suggested fix.** Hash the include closure (e.g. add the four shared headers and `kernel_lib/` to
the hashed set), and purge `*/kernels/{palu_*,*sdpa_decode*,*flash_decode*}` - or simply purge all
`*/kernels/*` entries when the hash changes; a full recompile costs one run.

### [Important] Stage-2 rows cannot be reproduced from the documented command line any more
**File:** `README.md:112-115` (the stage-2 recipe), `README.md:242-245`,
`palu_tt/bench.py:396-401` (`--rot-mode` default `dst`, `--table-mcast` default `on`),
`palu_tt/bench.py:461-466` (fields added in stage 2e).

**What.** 108 of 132 committed `bench_*.json` predate the `rot_mode`/`table_mcast` fields, including
`bench_decode_palu_full_50_fk2.json` - the file holding the 42.8 / 65.0 / 94.5 that the report quotes
as "Palu 0.5, fused kernel (stage 2)". Those runs used the then-default *legacy* rebuild with the
per-core angle table. Since 2026-09-14 the CLI defaults are `--rot-mode dst --table-mcast on`, but
the README's stage-2 recipe is still
`bench --row palu_full_50_fk2 ... --what decode --fused p2` with no rotation flags. Re-running it
today overwrites `bench_decode_palu_full_50_fk2.json` with the shipped kernel's ~39.0/55.5/77.4 -
under the same row name that the report and `report.py`'s VARIANTS2E table treat as "legacy".
(`docs/results/stage2d.md:9` does say "reproduce them with `--rot-mode legacy --table-mcast off`";
the README recipe does not, and neither do the 108 JSONs.)

**Why it matters.** The whole stage 2e improvement claim is a comparison between two files that
differ only in flags that one of them does not record.

**How to verify.**
`python3 -c "import json,glob;print(sum('rot_mode' not in json.load(open(f)) for f in glob.glob('docs/results/raw/bench_*.json')))"`
-> 108; then `git log -1 --format=%ad -- docs/results/raw/bench_decode_palu_full_50_fk2.json` against
the commit that changed the defaults.

**Suggested fix.** Put `--rot-mode legacy --table-mcast off` in the README's stage-2 recipe (as
stage2d.md does), and/or backfill the legacy switches into the pre-2e JSONs with a one-off script
that records `"rot_mode": "legacy", "table_mcast": "off"` plus a `"backfilled": true` marker.

### [Important] `_is_oom` is a substring match on a message truncated to 200 characters
**File:** `palu_tt/bench.py:164` / `:238` / `:257` / `:303` (`str(e)[:200]`), `:306-310` (`_is_oom`),
and the same rule in `report.py:241` / `:454`.

**What.** Every error cell stores only the first 200 characters of the exception, and OOM detection
is `"Out of Memory" in err["error"]`. In the committed JSONs the message is
`TT_FATAL @ <abs path to bank_manager.cpp>:462: false\ninfo:\nOut of Memory: ...`; measured over all
error cells, the string is exactly 200 chars and `"Out of Memory"` starts at index **147** and ends
at 160 - a 40-character margin that is entirely consumed by the absolute path of the tt-metal
checkout (this worktree path alone adds 33 characters over the main checkout). A deeper checkout
path silently flips every OOM cell to `limit: "model"` in `bench_maxctx_*` and to `"err"` instead of
`"OOM"` in the published decode tables.

Two related defects in the same code: (a) during bisection any non-OOM exception is treated exactly
like an OOM (`hi = mid`), so a transient/unrelated build failure silently lowers the reported
`max_context`; (b) `fail_err` is overwritten by the *last* failure, so an OOM-bounded search that
ends on a non-OOM probe is labelled `limit: "model"` - the same label stage2.md:124 documents as "the
model's own `max_position_embeddings` (131072)".

**How to verify.**
`python3 -c "import json,glob;e=[c['error'] for f in glob.glob('docs/results/raw/bench_*.json') for c in (json.load(open(f))['data'].values() if isinstance(json.load(open(f))['data'],dict) else []) if isinstance(c,dict) and c.get('error')];print({(len(x),x.find('Out of Memory')) for x in e})"`
-> `{(200, 147)}`.

**Suggested fix.** Match on the exception class / a normalised key (`"Out of Memory" in str(e)`
computed *before* truncation, stored as a boolean `"oom": True` field), keep the truncated message
only for humans, and record the probe's error kind per probe rather than only the last one.

### [Important] `maxctx` measures build-time allocation only, never a decode step
**File:** `palu_tt/bench.py:284-303` (`_maxctx_probe`: `build_model` -> `kv_bytes` -> `del`).

**What.** A probe builds the model and tears it down. It never runs `ttnn_decode_forward`, so the
per-step transients are not allocated. On the naive Palu path those transients are large -
`attention.py:660-703` allocates `[B, nkv, S, 128]` bf16 reconstruction tensors plus the rotated and
typecast copies per layer per step (the report itself says "268 + 268 + 143 MB at 8k"). Direct
evidence that build success != decode success: `bench_decode_glrd_70.json` at 8192 fails with
`"phase": "timing"` (not `"build"`), i.e. the model built and the *step* ran out of memory.

**Why it matters.** Report section 1 says "the largest `max_seq_len` at which the model **allocates**"
(accurate) but section 6's column is headed "max context at batch 32" and section 4 says "every
compressed row allocates 8192 tokens of context at batch 32", which readers will take as "can decode
at 8192". For the fused rows (no transients) the two coincide; for the naive rows they need not.

**How to verify.**
`python3 -c "import json;print(json.load(open('docs/results/raw/bench_decode_glrd_70.json'))['data']['8192'])"`
-> `{'error': ..., 'phase': 'timing', ...}`.

**Suggested fix.** Either run one decode step inside `_maxctx_probe` (turning it into "largest context
that decodes") or label the column "max context that allocates" and note that the naive rows'
decodable ceiling is lower.

### [Important] One row of the report's headline table quotes p50 while the protocol says minimum
**File:** `docs/report_palu_on_tt.md:75` ("Palu 0.7, shipped kernel | 761 | 8192 | 39.4 / 57.0 / 80.6")
vs `docs/report_palu_on_tt.md:17` ("the minimum is reported").

**What.** `bench_decode_palu_full_70_fk2dm.json` has `min_ms` 39.301 / 56.929 / 80.497 (-> 39.3 /
56.9 / 80.5, which is what `docs/results/stage2e.md` prints) and `p50_ms` 39.360 / 57.015 / 80.608
(-> 39.4 / 57.0 / 80.6). The report's row is the p50 triple. Every other row in that table is the
minimum (baseline 38.0/55.4, Palu 0.5 vanilla 88.5/265.3/488.5, fused 42.8/65.0/94.5, shipped
39.0/55.5/77.4, G-LRD 0.7 59.6/121.2/203.4 - all `min_ms`).

**Why it matters.** Small in magnitude (0.1 ms, ~0.2 %, and in the conservative direction) but it
makes one row of the headline table internally inconsistent with the rest, with the stated protocol
and with stage2e.md.

**How to verify.**
`python3 -c "import json;print({c:(round(r['p50_ms'],1),round(r['min_ms'],1)) for c,r in json.load(open('docs/results/raw/bench_decode_palu_full_70_fk2dm.json'))['data'].items()})"`.

**Suggested fix.** Use 39.3 / 56.9 / 80.5.

### [Important] The report's capacity protocol claims a device-memory cross-check that does not exist
**File:** `docs/report_palu_on_tt.md:18` ("KV bytes per token are computed from the allocated caches
and cross-checked against the device memory report"); `palu_tt/attention.py:352-358`;
`palu_tt/model.py:182-193`.

**What.** `kv_bytes_per_token()` is pure arithmetic on the configured ranks and a bytes-per-element
table: `n_local_groups * (r_k * bpe[k_latent_dtype] + r_v * bpe[v_latent_dtype])`. Nothing reads an
allocated buffer's size, and nothing in `palu_tt/`, `tests/` or `scripts/` mentions a device memory
report (`grep -rn "memory_report\|DumpDeviceMemoryState\|GetMemoryView\|device memory" palu_tt tests scripts`
returns nothing). The stage documents make no such claim either.

**Why it matters.** It is a stated validation of the byte counts (1088 / 574 / 761 / ...) that has no
implementation, in the protocol section of a report presented externally.

**How to verify.** The grep above; read `attention.py:352-358`.

**Suggested fix.** Drop the clause, or add a real check (`ttnn` exposes an allocator statistics call;
a device test asserting allocated-DRAM delta against the formula would make the claim true).

### [Important] A published decode cell carries the contention signature the docs say was discarded
**File:** `docs/results/raw/bench_decode_palu_full_70.json` ctx 1024; quoted at
`docs/results/stage1.md:43`; protocol claim at `docs/report_palu_on_tt.md:17` ("the median is within
1 % of it on an idle host, and every run was gated on the board being idle") and
`docs/results/stage1.md:137-141` ("those medians were inflated ... They were discarded").

**What.** Over all 220 decode cells, exactly one exceeds p50/min = 1.03: `palu_full_70` at 1024 with
p50 115.53 ms and min 90.20 ms (+28.1 %). The next worst is 1.009. The attention-only bench is worse
across the board: 69 cells exceed 3 %, up to p50/min = 2.18 (`bench_attention_baseline_b16.json` at
1024: p50 3.07, min 1.41) - unsurprising for 0.6-3 ms measurements dominated by host dispatch, but it
means the `p50` half of every published "p50 / min" attention cell is noise.

**Why it matters.** `min` (90.2) is the value the tables use downstream, and the same host contention
that inflated the median by 28 % can also inflate the minimum; the memory rule of thumb for this
project is that p50/min > 3 % flags a corrupted wall-clock cell. The report's "within 1 %" is
contradicted by a cell printed in stage1.md and stage2.md.

**How to verify.**
`python3 -c "import json,glob;print(sorted(((r['p50_ms']/r['min_ms'],f,c) for f in glob.glob('docs/results/raw/bench_decode_*.json') for c,r in json.load(open(f))['data'].items() if 'min_ms' in r),reverse=True)[:3])"`.

**Suggested fix.** Re-measure `bench --row palu_full_70 --what decode` (or at least the 1k cell) on a
quiet host, and soften the report's "within 1 %" to "within 1 % on all but one cell (listed in
stage1.md)".

### [Minor] `bench --what maxctx` never records the kernel layout its docstring promises
**File:** `palu_tt/bench.py:284-296` (docstring + `GROUP_HEADS[:] = fused.group_heads_used(cfg)`).

**What.** The docstring says "A successful probe also records the layout it built with in
`GROUP_HEADS` ... so a maxctx row ... says which kernel layout produced the capacity they report".
Every committed `bench_maxctx_*.json` has `group_heads_per_layer: []`, including the grouped fused
rows (`bench_maxctx_glrd_50_f256_l4_fk2gdm.json`, `bench_maxctx_glrd_70_f256_l4_fk2gdm.json`), where
the corresponding decode rows carry `[2, 2, 2, ...]`. The recording fix landed in c7b2904
(2026-09-14 16:21); the JSONs were committed in d3bbe25 (2026-09-14 15:32) and no maxctx run has
happened since, so the fix is untested against data.

**How to verify.**
`python3 -c "import json,glob;print({f.split('/')[-1]:json.load(open(f)).get('group_heads_per_layer') for f in glob.glob('docs/results/raw/bench_maxctx_*fk2*dm.json')})"`;
`git log --format='%h %ad' --date=format:'%m-%d %H:%M' -1 c7b2904 d3bbe25`.

**Suggested fix.** Re-run one grouped maxctx row, or drop the claim.

### [Minor] `profile_fit`'s "per item per chunk" is `per_chunk / 2`, a hard-coded batch-32 constant
**File:** `palu_tt/profile_fit.py:221` (`per_item_per_chunk_us=per_chunk / 2`).

**What.** `docs/results/stage2d.md` section 4 explains the 2 ("64 cores each owning 2 head-items",
i.e. batch 32 x 4 KV heads per chip / 64 cores). `profile_fit.py` has no comment, no batch input and
no way to know a profile's batch (the profile JSONs do not record it - see the profiling finding
above), so the column is only meaningful for batch-32, 4-KV-head-per-chip profiles. `stage2d.md` and
`stage2e.md` also disagree in the last digit for the same fit (61.6 vs 61.5 us for fused r_k 64 /
r_v 128), which is just rounding of 123.1/2.

**How to verify.** Read `profile_fit.py:200-222`; note `chunk` is a CLI default of 512 and `x` is
`ctx // chunk` derived from the *file name*, so a run with `--fused-k-chunk` != 512 would be fitted
against the wrong x.

**Suggested fix.** Derive the divisor from recorded batch / KV heads / core count (or take it as a
CLI argument), and name the column in the code the way stage2d names it ("per head-item").

### [Minor] Other items worth a line each
- `palu_tt/profile_report.py:57` - `classify(op, in0, in1, out, attrs)` never uses `out` or `attrs`.
- `palu_tt/profile_report.py:126-130` - `count` accumulates `1/steps`, giving values like
  `31.999999999999954` in every JSON; harmless but it makes exact comparisons awkward.
- `palu_tt/profile_report.py:113-115` - only the first `start`/`stop` pair is honoured; a CSV holding
  two profiled runs would be half-read without warning.
- `palu_tt/bench.py:253-254` - prefill hard-codes `latent_quant_in_prefill=False` (deliberate, to keep
  comparability with earlier files), but no JSON field records it, so a prefill row does not say which
  prefill path it timed.
- `palu_tt/bench.py:416-420` / `_out_name` - prefill ignores `--batch` yet still takes the `_b<N>`
  suffix, so `bench_prefill_x_b8.json` is a batch-1 measurement. Documented in `--help` only.
- `palu_tt/bench.py:101-109` - only p50/min/iters are kept; the raw samples are discarded, so a
  contaminated cell (see above) cannot be diagnosed after the fact.
- `palu_tt/attention.py:274-284` - the caches are zero-filled and never written by a prefill at the
  timed position; timing is unaffected (no data-dependent kernels on Tensix) but no timed decode ever
  attends over a cache a prefill produced.
- `palu_tt/decode_check.py:131` - only `pcc_mean`/`pcc_min` are stored, not the 63-value series, so the
  step that produced a low min cannot be located later.
- `palu_tt/decode_check.py:49` - fidelity is measured at batch 1, `max_seq_len` 2048, while every speed
  number is batch 32 at 1k/4k/8k. The fused op's work split (heads per core, sequence split, chunk
  count) differs between the two, so no fidelity measurement exists at the measured operating point.
  README:219-221 explains the batch-1 `--group-heads 2` default, but the gap itself is not stated.
- `palu_tt/bench.py:84-95` - `Mesh.reset()` deliberately bypasses `assert_device_free`; combined with
  the dead process check in `device_free.sh`, the only live gate is one telemetry sample at process
  start. `README.md:29` also notes that querying tt-smi wakes the board, i.e. the gate perturbs what
  it measures.

## Tests in this area

**`tests/test_bench_utils.py`** (imports `palu_tt.bench` -> imports `ttnn` and `models.tt_transformers`
at module level, so it is not device-free in the import sense, though it opens no device).
- `test_time_runs_warmup_plus_iters_and_reports_median`: real and good. Monkeypatches
  `time.perf_counter` with a scripted tick list and asserts call count (warmup+iters), the key set,
  `iters`, `p50_ms == 20.0` and `min_ms == 10.0`. Would catch a swapped median/min or a warm-up
  counted as timed.
- `test_time_defaults`: asserts 13 calls and `iters == 10` - the only test that pins the decode
  iteration count, and it pins it to 10, i.e. the tests agree with the code and disagree with the
  report. Nothing anywhere asserts which `iters` each `--what` uses.
- `test_parse_contexts`, `test_out_name`: real, low value, fine.
- The four `bench_maxctx` tests: the strongest tests in the file. They drive the real
  `bench_maxctx` with a threshold fake `build_model` and assert the bisection endpoint (4096), the
  first failing context (5120), `limit`, the mesh-reset count (3) and the probe list. I hand-traced
  the algorithm and the expected values are right, so these are genuine behavioural assertions, not
  restatements. Gaps: no test for a **non-OOM** failure during bisection (the case that mislabels
  `limit` as `"model"` and silently lowers `max_context`), and no test for the 200-char truncation
  interacting with `_is_oom`.
- `test_is_oom_...`: real but only exercises the untruncated happy path; it would pass with the
  truncation bug present because it passes a short string.
- `test_bench_decode_stop_after_oom_...`: real, but monkeypatches `_decode_one`, so nothing in the
  suite ever exercises a real decode cell's construction (the timed region, the sync, the
  `set_recon_len` call, the `iters`).
- `test_latent_dtype_flags_...`, `test_rot_switches_recorded_are_the_effective_ones_...`: real
  assertions against hand-written truth tables; would catch a regression in the recorded-switch rule.
- `test_kv_dtype_default_is_a_no_op_...`, `test_set_kv_cache_dtype_names_itself_...`,
  `test_apply_palu_flags_...`: good tests, but they cover `palu_tt/model.py`, not bench. The last one
  enumerates `PaluRuntime`'s fields from the dataclass rather than hand-listing them - the right
  pattern, and it would fail loudly if a switch were added without plumbing.

**`tests/test_profile_report.py`**
- `test_classify_covers_the_decode_step`: 12 assertions on hand-written shapes. Real, but the
  expectations are transcribed from the implementation's own rules, so it cannot discover a real op
  whose shape the rules misclassify (the risky cases - a Palu `wo` matmul with `k = 16*r_v`, a
  reconstructed tensor reported with fewer than 4 dims so `in0[2]` shifts, `r_v = 512` sitting exactly
  on the `k <= 512` boundary - are untested). I checked the actual profiles: nothing currently lands
  in `other` on any 4k run, and the Palu `wo` is `[2048, 4096]` because the narrow attention output is
  padded back to 128 per head first, so the classifier is in fact correct today. The test would not
  have told you that.
- `test_aggregate_windows_on_signposts_and_divides_by_steps`: genuinely good. It plants a 999 us op
  *before* the `start` signpost and asserts the totals exclude it, checks both chips, the /steps
  division, `ops_per_step`, two categories and the sort order. Would catch an off-by-one in the
  signpost window or a missing division.
- Untested: `latest_csv`'s "newest wins" ordering (it is `sorted(...)[-1]` on a glob, i.e. lexical,
  which works only because the timestamps are zero-padded), and `--steps` being unchecked against the
  run.

**`tests/test_profile_fit.py`**
- `test_fit_reproduces_the_stage2d_table`: a regression test against the committed
  `docs/results/raw/profile_*.json` with tolerances of 1.0-1.5 us. Real and valuable (it pins both the
  fit and the data), though it will fail for anyone who regenerates the raw dir, and it silently
  depends on CWD being the repo root.
- **`assert abs(f32["per_item_per_chunk_us"] - f32["per_chunk_us"] / 2) < 1e-9` is a test that cannot
  fail**: it restates `profile_fit.py:221` against itself. If the divisor should be 4 (batch 16) or
  anything else, this passes unchanged. It is the clearest instance in this area of asserting against
  the output of the code under test.
- `test_fit_skips_missing_contexts_and_needs_two_points`: real (asserts the x-values present are
  `[2, 8]` when the 8192 JSON is absent).
- Untested: `_lstsq` itself against a closed-form example, and the `< 2 distinct x` skip.

**Not covered anywhere (CPU-testable, currently unguarded):** which `iters`/`warmup` each `--what`
uses; that `dev.sync` is inside every timed region; `_out_name` for `attention`/`prefill`/`maxctx`;
the result-JSON field set (a field silently dropped from `r` in `bench.main` would break provenance
with nothing failing); `device_free.sh` and `run_device.sh` have no tests at all, which is how the
pipefail bug survived.

## Claims traced

| Documented claim | Code implementing it | Status |
|---|---|---|
| report §1: decode = "3 warm-up and 20 timed iterations" | `bench.py:136` -> `_time(step)` = 3+10 | **Contradicted** (JSONs record `iters: 10`) |
| report §1: "device sync inside the timed region" | `bench.py:132`, `:211`, `:266` -> `dev.sync` = `ttnn.synchronize_device` | Verified |
| report §1: "the minimum is reported" | `report.py:244/393/457/482/506` read `min_ms` | Verified for 5 of 6 section-6 rows; **contradicted** for "Palu 0.7, shipped kernel" (p50) |
| report §1: "the median is within 1 % of it on an idle host" | - | **Contradicted**: 1 decode cell at +28 %, 69 attention cells > 3 % |
| report §1: TTFT "at batch 1" | `bench.py:253` `max_batch_size=1` | Verified (2 warm-up + 5 timed, not stated) |
| report §1: profiling "3 warm-up and 3 profiled steps" | `profile_decode.py:180,179` defaults 3/3; `ReadDeviceProfiler` after warm-up | Verified (but `--steps` is re-typed into `profile_report`; see finding) |
| report §1: capacity "found by bisection" | `bench.py:346-358` | Verified for the newer JSONs; **contradicted** for baseline / `palu_full_50` / `palu_full_70` (power-of-two only) |
| report §1: KV bytes "cross-checked against the device memory report" | none found | **Not verified** (no such code anywhere) |
| report §1 / §5: decode fidelity = 512-token prompt, 63 teacher-forced steps, prefill logit excluded | `decode_check.py:28,50,54,118,130` | Verified (63 = `--steps 64` minus the prefill logit; all 19 JSONs record `steps: 63`) |
| report §1: fidelity compares "the per-step logits" of the two builds | `decode_check.py:109-119`, `generate.py:23/48` (`.float()` on host) | Verified: full-vocab fp32 host logits, PCC per step over 128256 values, identical teacher tokens from wikitext-2 fed to both builds (neither model's own argmax) |
| report §5: grouped rows "PCC 0.995 or better and 0.95 to 0.98 argmax agreement" | `decode_check_glrd_*.json` | **Contradicted** (0.9048 argmax; 0.99473 pcc_min) |
| report §4: "at 4k the reconstruction block costs 207 ms ... attention op itself is 8 ms" | `profile_palu_full_50_4096.json`: k_recon 206.74, attention **22.12** | k_recon verified; attention **contradicted** (8.307 is the layer-0 microbench) |
| report §2/§4: baseline attention op 24 ms at 4k; 1088 B/token/layer/chip | `profile_baseline_4096.json` 24.242; `model.py:192` | Verified |
| report §5: 38.0/55.4, 88.5/265.3/488.5, 42.8/65.0/94.5, 39.0/55.5/77.4, 59.6/121.2/203.4, 51/96/155 | the matching `bench_decode_*.json` `min_ms` | Verified |
| report §5: "0.89x and 0.85x of the baseline", "2.1 to 5.2x faster than the vanilla path", "339 tokens per second", "baseline tops out at batch 16 and 290" | arithmetic on the same `min_ms` (32*1000/94.48 = 338.7; 16*1000/55.14 = 290.2) | Verified |
| report §5: "the ratio-0.7 dump is within 3 % of the same numbers" | 43.13/66.36/97.72 vs 42.83/64.97/94.48 = +0.7 / +2.1 / **+3.4** % | Marginally contradicted at 8k |
| report §5: fused per-chunk 85.8 us < stock 88.5 us | `profile_fit` output in stage2e.md | Verified (against the committed fit JSONs) |
| README:26: gate "refuses to start ... when one of our own processes holds /dev/tenstorrent" | `device_free.sh:21-23` | **Contradicted** (pipefail makes the branch unreachable) |
| README:48-52: kernel-source hash purges stale JIT entries | `run_device.sh:33-45` | Partly verified: purge path matches the real cache layout, but the hash misses every included header outside `KDIR` and never purges the stock op |
| stage2d.md §0: "takes device 0, divides by 3" | `profile_report.py:118-140` | Verified (`min(per_chip)` = lowest device id) |
| stage2d.md §4: fit is `fixed + chunks x per_chunk`, x = ctx/512, y = us per layer | `profile_fit.py:200-222` | Verified; "per item per chunk" = per_chunk/2 is a hard-coded batch-32 constant |
| README:117-122: naive and fused fed the same tokens | `decode_check.py:50,109-113` (one `ids`, both builds) | Verified |
| `bench.py` docstring: "host-side input preparation is inside the timed region" | `bench.py:124` (outside the loop) | **Contradicted** |
| `_maxctx_probe` docstring: a successful probe records the layout it built with | `bench.py:296`; all committed maxctx JSONs | **Contradicted** in the data (fix postdates every run) |
| `profile_report` docstring: steps come from the run's `meta.json` | no `meta.json` is ever written | **Contradicted** |
| Every timing-affecting parameter is recorded in the JSON | `bench.py:462-468` | Verified for post-2e runs; 108/132 files predate `rot_mode`/`table_mcast`/latent formats, 44 predate `batch`, 28 predate `fused` |
| No JSON's recorded fields contradict its file name | scripted check of all 132 bench JSONs (what / row / batch suffix / `_fk1`-`_fk2` vs `fused`) and the stage-2e suffix -> (rot_mode, table_mcast) map | Verified - zero mismatches; the only gaps are *absent* fields, never wrong ones |

## Verdict

The harnesses themselves are honest and mostly careful: the device is synchronised inside every timed
region, the decode cell really is batch 32 with all 32 users at the last position and a K rebuild over
the full cached length, `decode_check` really does feed both builds the same wikitext tokens and
compare full-vocab fp32 logits over 63 decode steps, and the profile aggregator really does window on
the signposts and divide by the step count. What is unreliable is the layer of description on top:
the externally-facing `docs/report_palu_on_tt.md` misstates the decode iteration count, quotes a
layer-0 microbench figure (8 ms) as a profile figure that is really 22 ms, rounds a 0.905 argmax
agreement into a "0.95 to 0.98" range, prints a batch-32 capacity column that mixes power-of-two and
1k-bisected probes in a way that inverts the true ordering, and claims a device-memory cross-check
that does not exist - four of those five contradict the study's own stage documents, which got them
right. Behind that, three infrastructure defects could make a *future* number wrong without anyone
noticing: the device-idle gate's process check is unreachable under `set -o pipefail`, the JIT
kernel-cache guard hashes only five of the files that go into the kernel, and the profiler's
steps-divisor is a hand-typed constant the profile JSONs cannot be checked against. The tests are
better than average for a research repo - the maxctx bisection and the signpost-window tests are real
behavioural tests - but they leave every numeric path that actually produced a published number
(the timed regions, the iteration counts, the classifier against real op shapes) unguarded, and one
assertion in `test_profile_fit` cannot fail by construction.
