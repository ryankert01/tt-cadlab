# Palu on Tenstorrent — stages 1, 2 and 2b

Stage 1 ports [Palu](https://github.com/shadowpa0327/Palu)-style low-rank KV-cache compression to a
Tenstorrent Wormhole n300 (Llama-3.1-8B-Instruct, TP=2) and measures it against the stock model.
`palu_tt/attention.py` is a drop-in replacement for tt-metal v0.78.0's `tt_transformers`
`Attention`: the caches hold per-head K/V *latents* (`h = x @ VTᵀ`, rank `r ≤ head_dim`), K is
reconstructed with `latent @ U_k` plus a prefill-mode RoPE every decode step, and the V latent goes
straight into `flash_multi_latent_attention_decode` with its up-projection folded into `wo`.
Everything else stays stock, so a rank-128 ("identity") factorisation is bit-identical to stock and
isolates the plumbing cost. That is Palu's *naive* reconstruction path. Stage 2 replaces it at decode
with a fused op (`ttnn.transformer.palu_sdpa_decode`, in the tt-metal submodule branch) that
reconstructs and rotates K inside the flash-decode kernel, so the full K is never materialised —
see `docs/results/stage2.md`. Stage 2b adds Palu's **G-LRD** variant (group size 4: the four KV
heads of a chip share one K latent and one V latent, each head reconstructed with its own `U`
slice) and scores every row on five zero-shot lm-evaluation-harness tasks the way the paper does —
see `docs/results/stage2b.md`. On the G-LRD decode path V is *not* absorbed into `wo` (the group's V
latent is up to 512 wide); the fused kernel rebuilds V per head on core instead.

## Shared-server rules

The n300 and the host are shared with other users. No `sudo`, nothing written outside this
worktree, and `~/tt-data` and `third_party/` are off limits (`third_party/palu` shows as dirty by
design — it is patched at run time by `scripts/patch_palu_cpu.sh` and must never be staged).

* **Every device command goes through `scripts/run_device.sh <cmd>`**, which runs
  `scripts/device_free.sh` first and refuses to start when tt-smi telemetry shows the board busy or
  when one of our own processes holds `/dev/tenstorrent`. One device job at a time; if the gate
  reports busy, wait and retry.
* Do **not** poll `tt-smi` while measuring: it wakes the n300 out of idle and the gate then reports
  the board busy for minutes. CPU work runs `nice -n 10`, capped at 12 threads (`OMP_NUM_THREADS`).

## Setup

```bash
scripts/setup_env.sh          # submodules, tt-metal release build (clang-20, ~1.5-2 h at -j10), venv
source scripts/env.sh         # TT_METAL_HOME, PYTHONPATH, MESH_DEVICE=N300, HF_MODEL, .venv
```

The model weights are read from the user's read-only HF cache (`HF_HUB_OFFLINE=1`).

### Stage 2: fused kernel (tt-metal submodule branch)

The fused K-reconstruction decode op lives in the tt-metal submodule on branch `palu-sdpa`
(cut from `v0.78.0`); the superproject pins that branch's commit. Until the fork
`ryankert01/tt-metal` exists and the pointer is pushed there, a fresh clone cannot fetch the
submodule commit -- the working environment on the shared box has it. After editing host-side
C++ (program factory, op, bindings) run `scripts/build_ttnn.sh`; kernel-only edits are picked up
at the next op run (JIT).

## Producing the weight sets

Plain per-head truncated SVD of `k_proj`/`v_proj` (group size 1, no calibration), CPU, a few minutes
each:

```bash
python -m palu_tt.compress --rank-k 64 --rank-v 64 --out compressed/svd_r64     # likewise 96/96 ->
# svd_r96, 128/64 -> svd_rk128_rv64, 32/128 -> svd_rk32_rv128 (the mixed-path device-test fixture)
```

Palu's full recipe (Fisher-uniform rank search + whitened decomposition) runs on CPU in fp32 out of
the patched `third_party/palu` checkout — `scripts/run_calibration.sh` applies
`scripts/patch_palu_cpu.sh` at run time, so the submodule stays dirty and unstaged:

```bash
scripts/run_calibration.sh 0.5      # -> compressed/palu_full_0.5
scripts/run_calibration.sh 0.7      # -> compressed/palu_full_0.7
scripts/run_calibration.sh 0.5 4    # G-LRD, group size 4 -> compressed/glrd_0.5   (stage 2b)
scripts/run_calibration.sh 0.7 4    # -> compressed/glrd_0.7
```

The optional second argument is Palu's `--head_group_size`: 1 (the default) is M-LRD, one latent per
KV head; 4 is G-LRD with one group per n300 chip. A G-LRD dump stores one `U.<g>.weight` per group
and one rank per group in `head_wise_ranks`; `palu_tt.convert.load_palu_dir` reads either form.

Budget ~35 min for the Fisher stage and ~8 h for whitening + decomposition (~17 min/layer × 32); the
stats cache under `third_party/palu/cache` is reused, so a second ratio only redoes the decomposition.

## Running the measurements

```bash
scripts/run_device.sh python -m palu_tt.eval_ppl --row baseline --kind stock              # wikitext-2 ppl
scripts/run_device.sh python -m palu_tt.eval_ppl --row svd_r64 --kind palu --palu-dir compressed/svd_r64
scripts/run_ppl_cpu.sh baseline stock                                                     # fp32 CPU cross-check
scripts/run_device.sh python -m palu_tt.bench --row baseline --kind stock --what decode   # also: attention, prefill, maxctx
python -m palu_tt.report                                                                  # re-render the tables
```

### Stage 2 rows

`--fused p1|p2` switches `palu_tt.bench` onto the fused decode kernel (Phase 1 = streamed cos/sin,
Phase 2 = chunk-relative rotation); it is a Palu-path flag and is rejected with `--kind stock`.
`--fused-k-chunk N` overrides the op's K chunk. `--batch B` (default 32) sweeps the batch size for
decode, attention and maxctx (prefill is single-user by definition and ignores it); the output file
is `bench_<what>_<row>.json` at the default 32 and `bench_<what>_<row>_b<B>.json` otherwise, so the
stage-1/2 files never move. The stage-2 rows are named `<stage-1 row>_fk1` / `_fk2`:

```bash
scripts/run_device.sh python -m palu_tt.bench --row palu_full_50_fk2 --kind palu \
    --palu-dir compressed/palu_full_0.5 --what decode --fused p2      # also: attention
```

Perplexity is measured through prefill windows and therefore cannot see the decode kernel at all,
so decode-path fidelity has its own harness: `palu_tt.decode_check` runs the naive and the fused
model over the same 512-token prompt plus `--steps` (default 64) teacher-forced steps, then compares
per-step logits over the *decode* steps only (63 by default) -- the shared prefill logit is
bit-identical on both paths and is dropped rather than counted -- recording PCC and argmax
agreement.

```bash
scripts/run_device.sh python -m palu_tt.decode_check --row palu_full_50_fk2 \
    --palu-dir compressed/palu_full_0.5 --fused p2                    # -> decode_check_*.json
python -m palu_tt.report --stage 2                                    # -> docs/results/stage2.md
```

A fused row's perplexity and prefill time are its base row's by construction (the kernel changes
decode only), so they are not re-measured; KV bytes and max context are inherited unless the row has
its own probe (a `bench_maxctx_*` JSON), in which case that measurement wins. `report.py` reads the
base row's JSONs for whatever a row does not measure itself.

### Zero-shot task accuracy

`palu_tt/eval_tasks.py` scores five lm-evaluation-harness (0.4.2) tasks — openbookqa, piqa, arc_easy,
arc_challenge, winogrande — through device prefill: `palu_tt/lm_eval_tt.py`'s `TTLM` adapter tokenises
each (context, continuation) request, runs the whole sequence through
`generate.prefill_hidden_states`, and applies the perplexity harness's host RMSNorm + `lm_head` path
to get the continuation tokens' summed log-probability.

```bash
scripts/run_device.sh python -m palu_tt.eval_tasks --row baseline --kind stock              # all five tasks
scripts/run_device.sh python -m palu_tt.eval_tasks --row svd_r64 --kind palu --palu-dir compressed/svd_r64 --tasks arc_easy --limit 40
```

lm-eval stays at 0.4.2 (the Palu repo imports it); its built-in `openbookqa`, `piqa` and `winogrande`
task configs point at dataset ids the Hub no longer serves, so `palu_tt/lm_eval_tasks/*_hub.yaml`
`include:` the installed configs and override `dataset_path`/`dataset_name` to the ids that are
mirrored (`allenai/openbookqa`, `ybisk/piqa`, `allenai/winogrande`); `eval_tasks.TASK_NAME_MAP` maps
the override names back to the paper's task names in the output JSON. The datasets need downloading
once behind the proxy, with the system CA bundle and offline mode off for that one call only:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  python -c 'from palu_tt import eval_tasks; import lm_eval; lm_eval.tasks.get_task_dict(eval_tasks.lm_eval_names(eval_tasks.TASKS), eval_tasks.task_manager())'
```

After that, `HF_HUB_OFFLINE=1` (the default) is enough. Writes `docs/results/raw/tasks_<row>.json`.

### Stage 2b rows

The five stage-2b rows, in the order they were run (one gated device process at a time; ~55 min each
for the full five tasks, ~4.6 h in total), then the renderer:

```bash
scripts/run_device.sh python -m palu_tt.eval_tasks --row baseline --kind stock
scripts/run_device.sh python -m palu_tt.eval_tasks --row palu_full_50 --kind palu --palu-dir compressed/palu_full_0.5
scripts/run_device.sh python -m palu_tt.eval_tasks --row palu_full_70 --kind palu --palu-dir compressed/palu_full_0.7
scripts/run_device.sh python -m palu_tt.eval_tasks --row glrd_50 --kind palu --palu-dir compressed/glrd_0.5
scripts/run_device.sh python -m palu_tt.eval_tasks --row glrd_70 --kind palu --palu-dir compressed/glrd_0.7
python -m palu_tt.report --stage 2b                                   # -> docs/results/stage2b.md
```

The G-LRD quality/latency rows use the stage-1/2 harnesses unchanged, with the G-LRD dumps:

```bash
scripts/run_device.sh python -m palu_tt.eval_ppl --row glrd_50 --kind palu --palu-dir compressed/glrd_0.5
scripts/run_device.sh python -m palu_tt.bench --row glrd_50_fk2 --kind palu --palu-dir compressed/glrd_0.5 \
    --what decode --fused p2                                          # also: attention, maxctx --batch 8
scripts/run_device.sh python -m palu_tt.decode_check --row glrd_50_fk2 --palu-dir compressed/glrd_0.5 --fused p2
```

Task accuracy runs prefill only, so it never enters the fused decode kernel: the `--fused` flag is
accepted by `eval_tasks` but the five rows above were scored without it, and decode-path fidelity is
`decode_check`'s job.

Each harness writes one JSON to `docs/results/raw/`; `palu_tt.report` reads only those and refuses to
render when none are present. The samples (`docs/results/raw/samples.json`) come from
`palu_tt.generate.greedy` driven by a short ad-hoc runner, not a committed script.

`--what maxctx` probes powers of two from 1k up to Llama-3.1's 131072 `max_position_embeddings`, then
bisects on multiples of 1024 between the last success and the first failure to pin the boundary down
to 1k granularity (`limit: "oom"` for an allocation failure, `"model"` for the architectural ceiling
or a non-memory build rejection). `--stop-after-oom` on `--what decode` stops the context loop right
after the first error cell instead of trying every remaining (larger) context -- used by the
context-scaling scan (`docs/results/raw/bench_{decode,maxctx}_<row>_b{1,8}.json`). `python -m
palu_tt.plots --batch 1 --batch 8` renders `docs/results/ctx_scaling_b{1,8}.png` (decode step vs
context, one line per row) from those files.

## Tests

```bash
nice -n 10 pytest tests/test_compress.py tests/test_convert.py tests/test_reference.py \
                  tests/test_report.py tests/test_bench_utils.py tests/test_attention_tag.py \
                  tests/test_fused.py tests/test_lm_eval_tt.py -q
scripts/run_device.sh pytest tests/test_attention_device.py tests/test_model_device.py \
                             tests/test_ppl_harness.py tests/test_palu_sdpa_device.py -q
```

The first line is CPU-only (`test_fused.py` covers the stage-2 host-side table builders and the
fp32 torch reference for the op; `test_lm_eval_tt.py` drives the `lm_eval_tt.TTLM` adapter with a
toy tokenizer and a deterministic score function, no lm-eval task or device involved). The four
device files — `test_attention_device.py` (attention vs
the fp32 CPU reference, prefill and decode, cache-level K/V checks, and fused-vs-naive decode),
`test_model_device.py` (whole model, identity-vs-stock logits and greedy tokens),
`test_ppl_harness.py`, and `test_palu_sdpa_device.py` (the fused op against its reference, both
phases) — open the n300 and must go through the gate.

## Results

`docs/results/stage1.md` (naive reconstruction), `docs/results/stage2.md` (the fused kernel) and
`docs/results/stage2b.md` (G-LRD and zero-shot task accuracy) — the tables are generated by
`python -m palu_tt.report [--stage 2|2b]` from the raw JSONs in `docs/results/raw/` (tracked in git;
the run logs beside them are not), and everything below the marker in each file is hand-written prose
that a re-render preserves. The designs and the recorded deviations from them are in
`docs/superpowers/specs/2026-09-09-palu-tt-stage1-design.md`,
`docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md` and
`docs/superpowers/specs/2026-09-12-palu-tt-stage2b-design.md`.
