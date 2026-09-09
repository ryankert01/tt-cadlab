#!/usr/bin/env bash
# Exit 0 only when the n300 looks idle. Telemetry words pack (max<<16 | current).
# Idle n300: AICLK current == 500 MHz, TDP current ~13 W. Busy: AICLK 1000 MHz and/or TDP >> 20 W.
set -euo pipefail
json="$(tt-smi -s 2>/dev/null)" || { echo "device_free: tt-smi failed"; exit 1; }
python3 - "$json" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
busy = []
for i, dev in enumerate(d["device_info"]):
    t = dev["smbus_telem"]
    aiclk = int(t["AICLK"], 16) & 0xFFFF
    tdp = int(t["TDP"], 16) & 0xFFFF
    if aiclk > 500 or tdp > 25:
        busy.append(f"chip{i}: AICLK={aiclk}MHz TDP={tdp}W")
if busy:
    print("device_free: n300 appears BUSY -> " + "; ".join(busy)); sys.exit(1)
print("device_free: n300 idle")
PY
# our own processes holding the device (other users' processes are invisible without sudo)
if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q tenstorrent; then
    echo "device_free: one of our processes already holds /dev/tenstorrent"; exit 1
fi
