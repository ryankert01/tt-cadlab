# Palu on Tenstorrent — stages 1 and 2

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
see `docs/results/stage2.md`.

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
```

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
`--fused-k-chunk N` overrides the op's K chunk. The stage-2 rows are named `<stage-1 row>_fk1` /
`_fk2`:

```bash
scripts/run_device.sh python -m palu_tt.bench --row palu_full_50_fk2 --kind palu \
    --palu-dir compressed/palu_full_0.5 --what decode --fused p2      # also: attention
```

Perplexity is measured through prefill windows and therefore cannot see the decode kernel at all,
so decode-path fidelity has its own harness: `palu_tt.decode_check` runs the naive and the fused
model over the same 512-token prompt plus 64 teacher-forced decode steps and records the per-step
logit PCC and argmax agreement.

```bash
scripts/run_device.sh python -m palu_tt.decode_check --row palu_full_50_fk2 \
    --palu-dir compressed/palu_full_0.5 --fused p2                    # -> decode_check_*.json
python -m palu_tt.report --stage 2                                    # -> docs/results/stage2.md
```

A fused row's perplexity, prefill time, KV bytes and max context are its base row's by construction
(the kernel changes decode only), so they are not re-measured; `report.py` reads them from the base
row's JSONs.

Each harness writes one JSON to `docs/results/raw/`; `palu_tt.report` reads only those and refuses to
render when none are present. The samples (`docs/results/raw/samples.json`) come from
`palu_tt.generate.greedy` driven by a short ad-hoc runner, not a committed script.

## Tests

```bash
nice -n 10 pytest tests/test_compress.py tests/test_convert.py tests/test_reference.py \
                  tests/test_report.py tests/test_bench_utils.py tests/test_attention_tag.py \
                  tests/test_fused.py -q
scripts/run_device.sh pytest tests/test_attention_device.py tests/test_model_device.py \
                             tests/test_ppl_harness.py tests/test_palu_sdpa_device.py -q
```

The first line is CPU-only (`test_fused.py` covers the stage-2 host-side table builders and the
fp32 torch reference for the op). The four device files — `test_attention_device.py` (attention vs
the fp32 CPU reference, prefill and decode, cache-level K/V checks, and fused-vs-naive decode),
`test_model_device.py` (whole model, identity-vs-stock logits and greedy tokens),
`test_ppl_harness.py`, and `test_palu_sdpa_device.py` (the fused op against its reference, both
phases) — open the n300 and must go through the gate.

## Results

`docs/results/stage1.md` (naive reconstruction) and `docs/results/stage2.md` (the fused kernel) —
the tables are generated by `python -m palu_tt.report [--stage 2]` from the raw JSONs in
`docs/results/raw/` (tracked in git; the run logs beside them are not), and everything below the
marker in each file is hand-written prose that a re-render preserves. The designs and the recorded
deviations from them are in `docs/superpowers/specs/2026-09-09-palu-tt-stage1-design.md` and
`docs/superpowers/specs/2026-09-10-palu-tt-stage2-design.md`.
