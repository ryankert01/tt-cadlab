# Group mode at eight cores per item: measure, then decide the cap and the selector

**Goal:** finish group-shared mode now that the three-round hang is fixed (submodule 75d4765): find
out whether splitting an item over 8 cores (three reduction rounds) beats the shipped four-core cap
and the per-head kernel at small batch, and re-tune the cap and selector from measurements.

**Constraints:** shared n300, every device run through `scripts/run_device.sh` after
`source scripts/env.sh`; no report or deck number changes in this plan (the results doc lists which
cells would move); nothing merged to main; commits on the branch only; logs never committed.

## Phase 1: attention-only sweep (layer 0, contexts 1k / 4k / 8k, 20 timed iterations)

`scripts/sweep_group_cap.sh` runs `python -m palu_tt.bench --what attention --fused p2` on the two
shipped G-LRD dumps (`compressed/glrd_0.5_f256`, `compressed/glrd_0.7_f256`) with `--group-heads`
forced and the cap set through `PALU_GROUP_MAX_CORES_PER_ITEM` (the factory and the Python L1 model
both read it). Cells, per dump, in this order (cap only changes the layout where marked):

| batch | H = 1 (per-head) | H = 2 | H = 4 |
|---|---|---|---|
| 8 | c4 | c4 (4 cores, 2 rounds either way) | c4 (4 cores, half grid) **and c8 (8 cores, 3 rounds, full grid)** |
| 4 | c4 | c4 **and c8** | c4 **and c8** |
| 1 | c4 | c4 **and c8** | c4 **and c8** |
| 16 | c4 | c4 (2 cores) | c4 (4 cores) |

Rows are named `g8_<dump>_h<H>c<cap>`, JSON `docs/results/raw/bench_attention_<row>_b<batch>.json`.
A cell that does not fit L1 records the factory's error; a hang is caught by
`TT_METAL_OPERATION_TIMEOUT_SECONDS=120`, the runner resets the board and moves on. Existing JSONs are
skipped, so the script is resumable.

## Phase 2: whole decode step for the layouts that won

For batch 8 and batch 4: `--what decode` for H = 1 and for the best cap-8 layout from phase 1 (best
`min_ms` at 4k), so the number the report quotes is measured, not inferred. Skipped for a batch where
no cap-8 layout beat H = 1 at 4k.

## Phase 3: summary and decisions (in the morning, with the user)

`scripts/sweep_group_cap_summary.py` writes `docs/results/stage2g_group_cap.md`: min and p50 per cell,
p50/min contamination flag, ratio against H = 1, L1 failures. Decisions it feeds:
1. the cap (`group_max_cores_per_head` in the factory and `_group_cap()` in `palu_tt/fused.py`);
2. the selector rule in `fused.group_heads_per_core` (occupancy first, then fewest rounds) if a
   three-round layout wins;
3. which report cells (G-LRD decode at batch 8 and below) would change, for the user to decide.

## Stop rules

Device busy (another user): the gate refuses, the runner waits 60 s and retries the cell. Two
consecutive timeouts on one cell: mark it and continue. Never raise the cap in code before phase 3.
