---
name: bench-needs-quiet-cpu
description: "Wall-clock decode bench cells (palu_tt.bench) are corrupted by concurrent CPU/IO jobs (calibration dumps, test runs); tracy device profiles are not. Detector: p50/min spread > 3%."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4941661f-efa0-4f6f-865b-7b8ede31259c
  modified: 2026-09-13T21:22:50.538Z
---

Stage 2e matrix (2026-09-14): the decode bench rows measured while the Grid B calibration dump queue
(nice 10, 12 threads, 30 GB writes) ran on the same host came out with p50 far above min
(palu_full_70_fk2 at 1k: p50 88 ms vs min 48 ms; the quiet stage 2d value is 43.2 / 43.1). The
device-side tracy profiles taken in the same window were clean and consistent.

**Why:** `palu_tt.bench` times whole decode steps on the host (perf_counter around dispatch + sync), so
any host stall lands in the number; per-op device times from the profiler do not include host gaps.

**How to apply:** Never run calibration dumps, model conversions, or other CPU-heavy jobs of ours while
a bench cell runs (pause them with SIGSTOP/SIGCONT if a queue is in flight). Accept a bench cell only
when p50/min <= 1.03 at every context; otherwise re-run it. Two queues sharing the n300 must gate on
both `pgrep -f "palu_tt\.|pytest tests/"` and two idle telemetry readings, since the device_free check
alone races. See [[shared-server-no-tt-data]] and [[palu-tt-project]].

**Update 2026-09-24:** the attention-only bench cell (`--what attention`, layer 0, 20 iterations) is host-bound at batch <= 16: 1-3 ms per step, most cells p50/min > 1.03 and times non-monotonic in context even with the server idle (load 1.0). It cannot resolve 10-20 % kernel differences at small batch; use the device profiler (`palu_tt.profile_decode` under tracy, now with `--batch`) and read the attention category's device ms per step instead.
