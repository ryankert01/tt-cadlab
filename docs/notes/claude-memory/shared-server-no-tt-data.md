---
name: shared-server-no-tt-data
description: The n300 box is shared with other users; never depend on or modify /home/user01/tt-data; check device is idle before runs; cap CPU threads
metadata:
  type: feedback
---

The machine is a shared server (several users, one world-accessible n300, no sudo). The user said "this is a shared server, so be careful" and "we shouldn't base on tt-data".

**Why:** `/home/user01/tt-data` is a separate project checkout with uncommitted work; other users share the CPU and the single n300 device, and their processes are invisible without sudo.

**How to apply:** Keep all work self-contained in the tt-cadlab repo (own tt-metal submodule/venv). Never import from, read at runtime, or edit tt-data. Before any device run, check n300 telemetry is idle (AICLK 500 MHz, ~13 W) and fail fast. Nice and thread-cap (~10–12) builds, calibration and CPU eval. No sudo, no firmware/driver changes. See [[palu-tt-project]].
