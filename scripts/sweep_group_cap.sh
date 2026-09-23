#!/usr/bin/env bash
# Group-mode cap sweep (plan docs/superpowers/plans/2026-09-24-group-mode-eight-cores.md).
# Attention-only bench cells at forced H and cap; resumable (skips cells whose JSON exists).
# Usage: scripts/sweep_group_cap.sh [attention|decode]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source scripts/env.sh
export TT_METAL_OPERATION_TIMEOUT_SECONDS=120
WHAT="${1:-attention}"
RAW=docs/results/raw
gate_wait() { until scripts/device_free.sh >/dev/null 2>&1; do echo "$(date +%T) device busy, waiting"; sleep 60; done; }
recover() { if ! scripts/device_free.sh >/dev/null 2>&1; then echo "$(date +%T) device not idle after cell -> tt-smi -r 0"; tt-smi -r 0 >/dev/null 2>&1; sleep 10; fi; }
run_cell() {  # dump tag H cap batch
  local dump=$1 tag=$2 H=$3 cap=$4 B=$5
  local row="g8_${tag}_h${H}c${cap}" suffix=""
  [[ $B != 32 ]] && suffix="_b${B}"
  local json="$RAW/bench_${WHAT}_${row}${suffix}.json"
  local log="$RAW/bench_${WHAT}_${row}${suffix}.log"
  if [[ -f $json ]]; then echo "skip $row b$B (exists)"; return; fi
  gate_wait
  echo "$(date +%T) START $row b$B ($WHAT)"
  PALU_GROUP_MAX_CORES_PER_ITEM=$cap scripts/run_device.sh python -m palu_tt.bench --row "$row" --kind palu \
      --palu-dir "compressed/$dump" --what "$WHAT" --fused p2 --group-heads "$H" --batch "$B" > "$log" 2>&1
  echo "$(date +%T) END   $row b$B exit=$? $(grep -oE '"(min_ms|error)": [^,}]*' "$log" | tr '\n' ' ' | cut -c1-200)"
  recover
}
if [[ $WHAT == attention ]]; then
  for pair in "glrd_0.5_f256 glrd50" "glrd_0.7_f256 glrd70"; do set -- $pair; dump=$1; tag=$2
    run_cell $dump $tag 1 4 8;  run_cell $dump $tag 2 4 8;  run_cell $dump $tag 4 4 8;  run_cell $dump $tag 4 8 8
    run_cell $dump $tag 1 4 4;  run_cell $dump $tag 2 4 4;  run_cell $dump $tag 2 8 4;  run_cell $dump $tag 4 4 4;  run_cell $dump $tag 4 8 4
    run_cell $dump $tag 1 4 1;  run_cell $dump $tag 2 4 1;  run_cell $dump $tag 2 8 1;  run_cell $dump $tag 4 4 1;  run_cell $dump $tag 4 8 1
    run_cell $dump $tag 1 4 16; run_cell $dump $tag 2 4 16; run_cell $dump $tag 4 4 16
  done
else
  # phase 2: decode for H=1 and the best cap-8 layout per batch, chosen from the attention JSONs
  for pair in "glrd_0.5_f256 glrd50" "glrd_0.7_f256 glrd70"; do set -- $pair; dump=$1; tag=$2
    for B in 8 4; do
      best=$(python scripts/sweep_group_cap_summary.py --best "$tag" "$B" 2>/dev/null || true)
      if [[ -z $best ]]; then echo "batch $B $tag: no cap-8 layout beat H=1 at 4k; decode skipped"; continue; fi
      run_cell $dump $tag 1 4 $B
      hb=${best#h}; run_cell $dump $tag ${hb%%c*} ${hb##*c} $B
    done
  done
fi
echo "$(date +%T) SWEEP DONE ($WHAT)"
