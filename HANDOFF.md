# Handoff: Palu on Tenstorrent (tt-cadlab)

Written 2026-10-28 when the work moved off the original n300 box. Read this first; the README is
the operating manual, `docs/results/*.md` are the measurements, `docs/report_palu_on_tt.md` is the
write-up.

## What this project is

A paper-style study of [Palu](https://github.com/shadowpa0327/Palu) low-rank KV-cache compression on a
Wormhole n300 (two chips, TP=2) with Llama-3.1-8B-Instruct. Headline comparison: the stock model on
TT versus Palu(model) on TT, perplexity, five zero-shot tasks, decode latency and max context, all
measured on device through one harness (`palu_tt/`). Work ran 2026-09-09 to 2026-09-24.

Three planned stages:

1. Functional port with stock ttnn ops (done).
2. A fused tt-metal decode kernel that reconstructs and rotates K inside flash-decode so the full K is
   never materialised (done, with sub-stages 2b to 2g).
3. TT vLLM paged serving: capacity and throughput with the compressed cache (**not started**).

## Where everything lives

| What | Where |
|---|---|
| Superproject | `https://github.com/ryankert01/tt-cadlab`, `main` = `worktree-bright-elm-xum9` @ 90718bb (144 commits) |
| Fused kernel `ttnn.transformer.palu_sdpa_decode` | tt-metal fork `https://github.com/ryankert01/tt-metal`, branch `palu-sdpa` (cut from `v0.78.0`, 22 commits, tip 75d4765c054). The superproject pins it as the `third_party/tt-metal` submodule; `.gitmodules` already points at the fork. |
| Palu upstream | `third_party/palu` submodule, patched at run time by `scripts/patch_palu_cpu.sh`. It is **dirty by design**; never stage it. |
| Results per stage | `docs/results/stage1.md`, `stage2.md`, `stage2b.md` … `stage2g.md`, `stage2g_group_cap.md`; raw JSONs in `docs/results/raw/` (tracked) |
| Report and slides | `docs/report_palu_on_tt.md`, figures `docs/figures/` (regenerate with `scripts/report_figures.py`, `scripts/kernel_figures.py`) |
| Kernel explainer | `docs/kernel_guide.md` |
| Specs and plans | `docs/superpowers/specs/`, `docs/superpowers/plans/` (one per stage) |
| Independent audit | `docs/audit/2026-09-19-code-audit.md` |
| Issue drafts and reproducers | `report_issue/` |
| Session notes (gotchas, traps) | `docs/notes/claude-memory/` |
| Everything done on the old box besides Palu | `PROGRESS.md` in the `tt-data` repo (SARATHI, DistServe, the May benchmarks) |

## State at handoff

Stages 1 through 2g are complete, merged and documented. Key numbers (batch 32, decode step ms at
1k/4k/8k context): stock 38.0/55.4/OOM; fused Palu `palu_full_0.7` 39.0/55.5/77.4. The recommended
configuration is `palu_full_70_lv4` (decode parity with stock, 3.5x the max context, +2 ppl).
Honest framing for stage 3: stock with a bfp4 KV cache is *faster* than stock bfp8 (0.82x) and
3.25x longer context for +2.5 ppl; Palu wins on bytes and quality, never on decode latency.

The last session (2026-09-23/24) fixed two kernel bugs and measured group mode at eight cores per
item; per-head still wins below batch 32, so the selector did not change.

### Decisions left open

1. **Group-mode core cap**: still 4 in the factory. Recommendation was to raise it to 8 (only explicit
   `--group-heads` runs change, and they get faster). Debug override: `PALU_GROUP_MAX_CORES_PER_ITEM`.
2. **Audit corrections**: the 2026-09-19 audit found the measurements sound but the report prose
   wrong in 7 places plus 43 "Important" items. None applied yet. List in `docs/audit/`.
3. **Stage 3** (vLLM paged serving) has no spec yet. Stage 1 used a non-paged cache; the fused kernel
   works on the unpaged flash-decode path, so paged support is kernel work before any serving numbers.
4. Deferred from earlier stages: per-head Fisher allocation, HellaSwag, J-LRD, whitening-sample
   sensitivity, the WinoGrande model-revision check.

### Upstream status of the kernel bugs

- `correction_block` five-tile DST overflow under fp32 accumulate: fixed on `palu-sdpa` (a41a23f,
  four tiles in every mode). Upstream fixed it independently in tt-metal PR #56225 after v0.78.0.
  **When tt-metal is next upgraded, drop our fix in favour of upstream's.** Do not file the issue.
- Group-mode three-round reduction hang: our bug, fixed on `palu-sdpa` (75d4765). Root cause and
  reproducer in `report_issue/issue_002_group_reduction_three_round_hang.md`.
- The README's "Known issues" section predates both fixes and still describes them as open and
  unexplained; `docs/notes/claude-memory/correction-block-dst-fix.md` and
  `cb-two-sided-counter-trap.md` have the current story. Updating that section is part of the audit pass.

## Bringing it up on the lab machine

```bash
git clone --recurse-submodules https://github.com/ryankert01/tt-cadlab.git
cd tt-cadlab
scripts/setup_env.sh      # submodule checkout, tt-metal release build (clang-20, 1.5 to 2 h at -j10), venv
source scripts/env.sh     # TT_METAL_HOME, PYTHONPATH, MESH_DEVICE=N300, HF_MODEL, .venv
```

Weights come from the user's HF cache with `HF_HUB_OFFLINE=1`; Llama-3.1-8B-Instruct is gated, so
log in to Hugging Face once and download it (about 16 GB) before setting offline mode.

Things that were not in git on the old box:

| Item | Size | How to get it back |
|---|---|---|
| `compressed/` calibration dumps (the Palu weight sets every row reads) | 18 GB | rsync from the old box, or rebuild: `scripts/run_calibration_from_ranks.sh` rebuilds each dump from its committed ranks in 2.5 to 6 min (cached whitening); a full `scripts/run_calibration.sh` is hours of CPU |
| `model_cache/` (ttnn weight cache) | 50 GB | regenerated on first run of each row |
| tt-metal build tree | about 170 GB | `scripts/setup_env.sh` |
| `~/.cache/huggingface` | 27 GB | re-download |
| `~/migration-bundles/*.bundle` | 6.5 MB | git bundles of every branch, a no-GitHub fallback |

Old box paths, in case something is missing: the working checkout was the git worktree
`/home/user01/tt-cadlab/.claude/worktrees/bright-elm-xum9` (the main checkout at `/home/user01/tt-cadlab`
had uninitialised submodules). `compressed/`, `model_cache/`, `.venv` and the build all live under that
worktree.

## Operating rules that bit us

- **One device job at a time, through `scripts/run_device.sh <cmd>`.** It checks the board is idle,
  and it hashes the palu kernel sources and purges the matching entries of `~/.cache/tt-metal-cache`:
  tt-metal keys its JIT cache on kernel path, defines and compile-time args, never on source bytes,
  so an edited kernel otherwise keeps running from the stale binary.
- Do not poll `tt-smi` during a measurement; it wakes the board and the gate then reports busy for
  minutes. After a `TT_FATAL` or hang the board is wedged: `tt-smi -r 0`, then confirm with `tt-smi -s`.
- Wall-clock benches need a quiet CPU. If p50/min exceeds 3 percent the cell is contaminated. Attention-
  only cells at batch 16 and below are host-bound noise regardless; use the tracy device profiler
  (`python -m tracy ... -m palu_tt.profile_decode`) for kernel conclusions. Whole-step decode benches
  reproduce to 0.1 percent.
- Long CPU jobs (calibration, 30 GB dump writes) must be launched with `setsid -f`; the Claude harness
  kills background tasks on low memory.
- The tt-metal submodule's pre-commit clang-format hook reformats copied files; commit kernel work
  with `--no-verify`.
- ttnn's weight cache is keyed by name, shape and dtype, not content: bump cache names when contents change.
- The ttnn `fill_cache` grouped-prefill shard fails between 512 and 2047 tokens without the tile-
  alignment fallback already in `palu_tt/model.py`; `nlp_concat_heads_decode` pads users to 32.
- 40-window perplexity screens are about 0.5 ppl easier than the 141-window protocol. Never mix them
  in one table. Prefill-only perplexity is blind to the latent cache format unless
  `latent_quant_in_prefill` is on (default since stage 2f).

## Suggested next steps, in order

1. Apply the audit corrections to `docs/report_palu_on_tt.md` and regenerate the figures. This is
   the cheapest way to make the existing result publishable.
2. Decide the group-mode cap (raise to 8), re-run `tests/` on the lab n300 to confirm the environment
   reproduces `docs/results/stage2g.md` within noise.
3. Write the stage 3 spec: paged variant of `palu_sdpa_decode` (the stock paged flash-decode op is
   the template, as the unpaged one was for stage 2), then the vLLM plumbing on top of the TT vLLM
   plugin in `tenstorrent/tt-inference-server`. The SARATHI and DistServe branches from June (see
   `PROGRESS.md` in the `tt-data` repo) are the prior TT vLLM scheduler work and show where the
   scheduler and model runner hooks are.
