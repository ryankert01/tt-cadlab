---
name: bug-triage
description: When a wrong result or a hang shows up in tt-metal or in our kernels, decide whether it is our bug, an upstream bug, or already fixed upstream, with a runnable reproducer first; then fix it with the device tools that work on this server. Use before writing any issue or PR.
---

# Bug triage: reproduce, locate ownership, then fix

The order matters. Every step produces something a maintainer could run or read; none relies on
"we believe".

## 1. Reproduce with the smallest public surface

- Write a script, not a recipe. One file, no CLI flags beyond `--hang`, fixed shapes, prints a
  one-line verdict per case (PCC against a torch reference, or pass / timeout). Put it under
  `report_issue/repro/` and paste it verbatim into the issue draft (`report_issue/issue_NNN_*.md`).
- Prefer the **public op** (`ttnn.transformer.scaled_dot_product_attention_decode`, ...) over an
  internal harness. If the public op passes, the bug may be ours; if it fails, the issue is short and
  nobody has to trust our kernel.
- Include a control that passes next to the case that fails (other DEST config, other head count,
  other tree depth). The contrast is the evidence.
- A hang needs `TT_METAL_OPERATION_TIMEOUT_SECONDS=60` set **before `import ttnn`**, so it becomes an
  exception, and the script must `os._exit(2)` on timeout (closing a hung device blocks). On any other
  exception close the device normally, or the next open crashes.
- Run through the gate: `source scripts/env.sh` first, then `scripts/run_device.sh python ...`. In the
  same command line never write the literal `pytest tests/`; `pgrep` matches its own shell. Use
  `./tests/...`.

## 2. Decide who owns it before fixing anything

1. Is the code we blame ours or stock? `git -C third_party/tt-metal log --oneline -- <file>` and the
   branch (`palu-sdpa` carries our commits on top of the release tag).
2. Fetch upstream: `git -C third_party/tt-metal fetch origin main`, then
   `git show origin/main:<path> | grep <symptom>` and
   `git log --oneline <release-tag>..origin/main -- <files>`. Release tags sit on branches with
   cherry-picked SHAs, so `merge-base --is-ancestor` lies; compare **content**, not ancestry.
3. Search issues without `gh`: `curl -s https://api.github.com/repos/tenstorrent/tt-metal/issues/<n>`
   and the PR titles in the log. Read the fix PR's description: it states what layouts it covers.
4. Three outcomes:
   - **Fixed upstream after our tag**: no PR. Mark the draft superseded with the PR and issue numbers,
     say what our reproducer adds, keep our patch until the next upgrade, note it in memory.
   - **Upstream, unfixed**: file the issue from the draft (title with the category, e.g. `[Bug fix]`),
     PR from the fork `ryankert01/tt-metal`. This machine has no GitHub credentials; the user pushes.
   - **Ours** (stock op passes the same shapes, upstream code byte-identical): fix it here, step 3.

## 3. Find the cause on the device without guessing

Tools that worked on the shared n300, in the order to try them:

- **Program counters after a hang** (no build, no size cost): leave the device hung, then read every
  RISC's PC with tt-exalens (`scratchpad/inspect_hang2.py` pattern: halt, `read_gpr(32)`, locate the
  PC in the kernel ELF by byte search because kernels are position independent, `riscv-tt-elf-addr2line`
  the ELF address). Cores whose PC sits in the firmware are finished; the rest are stuck. NCRISC has no
  debug hardware; TRISC1/2 often refuse to halt; BRISC and TRISC0 are enough.
- **Saved return addresses**: `read_riscv_memory(loc, sp + k*4, risc)`; any word inside the kernel
  text range is a caller, which names the call site of a non-inlined helper such as `move_block`.
- **Circular-buffer counters**: received at `0xFFB40000 + cb*0x1000 + 4*4`, acked at `+3*4`, read
  over the NoC with `read_word_from_device`; the unpacker's private copy at
  `0xFFB00020 + cb*32 + 24` (low half acked) via `read_riscv_memory(..., "trisc0")`. A CB whose
  register `received - acked` wraps or disagrees with the private copy has two producers or two
  consumers; the LLK pack/unpack sides store private cumulative counts, the dataflow side increments.
- Debug prints and the watcher both add code to every kernel and pushed our program over the
  kernel-config buffer (70,656 bytes on Wormhole); check the program size headroom before using them.
  `TT_METAL_WATCHER_DISABLE_ETH=1` fixes the `idle_erisc.elf overflows` link error.
- After every hang: `tt-smi -r 0`, wait five seconds, `scripts/device_free.sh`, then a plain
  `ttnn.open_device` / `close_device` round trip before real work.
- Editing shared tt-metal headers (`compute_common.hpp`, `ckernel_sfpu_*.h`) does not change the
  palu kernel hash: purge `~/.cache/tt-metal-cache/*/kernels/{sdpa,sdpa_flash_decode,Kernel_Source_Code,palu_sdpa_flash_decode}` by hand.

## 4. Fix and prove it

- Change the smallest thing that removes the cause, keep numerics identical where the report depends
  on them, and say so in the commit message.
- Re-run the reproducer (the failing case must pass, the controls must not move), then the covering
  suite (`./tests/test_palu_sdpa_device.py`, 98 tests) and the upstream op's own tests when stock code
  changed (`tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py`).
- Commit in the submodule on `palu-sdpa` with `--no-verify` and explicit paths, bump the pointer in a
  separate superproject commit, both with the session trailers. Update the issue draft's status line,
  the ledger (`.superpowers/sdd/progress.md`) and memory. Report outcomes with the numbers, and say
  what was not run.
