#!/usr/bin/env bash
# Device-profiled decode steps for the group-mode cap sweep (plan 2026-09-24-group-mode-eight-cores.md,
# phase 1 redone with the device profiler because wall-clock attention cells at batch <= 16 are host-bound).
# One full decode step per cell under tracy, 3 timed steps; palu_tt.profile_report aggregates per-op device
# time into docs/results/raw/profile_<name>.json. Resumable: cells with a JSON are skipped.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source scripts/env.sh
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300
PROF=third_party/tt-metal/generated/profiler
CTX="${CTX:-4096}"
gate_wait() { until scripts/device_free.sh >/dev/null 2>&1; do echo "$(date +%T) device busy, waiting"; sleep 60; done; }
recover() { if ! scripts/device_free.sh >/dev/null 2>&1; then echo "$(date +%T) device not idle -> tt-smi -r 0"; tt-smi -r 0 >/dev/null 2>&1; sleep 10; fi; }
run_cell() {  # dump tag H cap batch
  local dump=$1 tag=$2 H=$3 cap=$4 B=$5
  local name="g8_${tag}_h${H}c${cap}_b${B}_${CTX}"
  if [[ -f docs/results/raw/profile_${name}.json ]]; then echo "skip $name (exists)"; return; fi
  gate_wait
  echo "$(date +%T) START $name"
  rm -rf "$PROF/$name"
  PALU_GROUP_MAX_CORES_PER_ITEM=$cap scripts/run_device.sh python -m tracy -r -p -o "$PROF/$name" --op-support-count 6000 -t 5000 \
      -m palu_tt.profile_decode --row "$name" --kind palu --palu-dir "compressed/$dump" --fused p2 --group-heads "$H" \
      --ctx "$CTX" --steps 3 --batch "$B" > "$PROF/$name.log" 2>&1
  local rc=$?
  python -m palu_tt.profile_report --name "$name" >> "$PROF/$name.log" 2>&1 || echo "$(date +%T) report failed for $name"
  echo "$(date +%T) END   $name exit=$rc $(grep -E "ms/step|attention " "$PROF/$name.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"
  recover
}
for pair in "glrd_0.5_f256 glrd50" "glrd_0.7_f256 glrd70"; do set -- $pair; dump=$1; tag=$2
  run_cell $dump $tag 1 4 8; run_cell $dump $tag 2 4 8; run_cell $dump $tag 4 8 8; run_cell $dump $tag 4 4 8
  run_cell $dump $tag 1 4 4; run_cell $dump $tag 2 4 4; run_cell $dump $tag 2 8 4; run_cell $dump $tag 4 8 4
  run_cell $dump $tag 1 4 1; run_cell $dump $tag 2 8 1; run_cell $dump $tag 4 8 1; run_cell $dump $tag 2 4 1
done
echo "$(date +%T) PROFILE SWEEP DONE"
