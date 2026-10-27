---
name: cb-two-sided-counter-trap
description: tt-metal circular buffers must have one producer and one consumer; the dataflow side increments the stream-register counters while the LLK pack/unpack sides store private cumulative counts, so a CB shared in both directions between a dataflow kernel and compute silently desynchronises (the palu group-mode three-round hang, fixed 2026-09-24)
metadata:
  type: project
---

In tt-metal a CB's received/acked counts live in stream registers (`0xFFB40000 + cb*0x1000`, received at reg 4, acked at reg 3). `cb_push_back`/`cb_pop_front` on BRISC/NCRISC **increment** them; `llk_push_to_brisc` (packer) and `llk_pop_tiles` (unpacker) **store** their private cumulative counts. So a CB that a dataflow kernel pushes and pops while compute also pops and pushes it goes wrong after the first compute push: the writer's later pushes vanish from the register and the unpacker waits forever. The stock sdpa decode kernel aliases `cb_out_o == cb_out_worker` (c_16) this way and survives only because each core emits once per launch; our group-shared mode (H heads per core) hung on the second head of any node with two children and a parent. Fix on `palu-sdpa`: outbox = c_20 (`cb_out_final`, unused on non-root cores), inbox = c_16.

**Why:** the hazard is invisible in code review (the aliasing looks like a buffer-size optimisation) and only shows as a hang with no error.

**How to apply:** never let a dataflow kernel and compute both produce, or both consume, the same CB across iterations; when a hang appears, read the CB counters and the unpacker's private copy (`0xFFB00020 + cb*32 + 24`, low half) with tt-exalens as in `.claude/skills/bug-triage/SKILL.md`. See [[correction-block-dst-fix]], [[palu-tt-project]].
