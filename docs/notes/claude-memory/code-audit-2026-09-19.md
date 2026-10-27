---
name: code-audit-2026-09-19
description: The user asked for an audit of all agent-written code before presenting; the audit (docs/audit/) found the measurements sound but the external report's prose wrong in 7 places and 43 Important items; fixes await the user's decision
metadata:
  type: project
---

On 2026-09-19 the user asked: "We need to audit the code you write, all of them, eg. testing script, eval script, etc." Seven independent read-only auditors (one per area) plus coordinator verification produced `docs/audit/2026-09-19-code-audit.md` and `docs/audit/reports/{A_ppl,B_tasks,C_model,D_bench,E_report,F_calibration,G_kernel}.md` (commit 9cbc60f; artifact page palu-n300-code-audit.html).

**Why:** the numbers go to an external (ITRI) audience; the user's team must be able to trust and re-check them.

**How to apply:** the measurement code is sound, but `docs/report_palu_on_tt.md` (hand-written) is where errors live: 7 Critical sentences contradicted by the raw files (see the audit's section 3) and a list of Important protocol disclosures (BOS in task scoring differs from Palu's harness; blind vs faithful prefill rows must not be differenced; GPU dumps share ranks, not factors). Section 6 of the audit is the fix list; nothing was changed outside docs/audit/ because some fixes move reported numbers, and that is the user's call. Known latent infrastructure bugs to remember when running anything: `scripts/device_free.sh`'s own-process check never fires (pipefail), so gate on `pgrep -f "palu_tt\.|pytest tests/"` yourself; `run_device.sh`'s JIT-cache hash misses shared headers (purge `~/.cache/tt-metal-cache` palu+sdpa entries by hand after editing them); `eval_ppl` overwrites `ppl_<row>.json` with no guard and now defaults to the faithful prefill (pass `--no-latent-quant-in-prefill` to reproduce pre-2f rows). Related: [[palu-tt-project]], [[bench-needs-quiet-cpu]].
