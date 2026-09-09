#!/usr/bin/env bash
# CPU sanity perplexity (fp32 8B, ~1 window/min at 12 threads). Usage: run_ppl_cpu.sh <row> <kind> [palu_dir] [windows]
set -euo pipefail
source "$(dirname "$0")/env.sh"
row="$1"; kind="$2"; palu_dir="${3:-}"; windows="${4:-20}"
extra=(); [ -n "$palu_dir" ] && extra=(--palu-dir "$palu_dir")
nice -n 10 python -m palu_tt.eval_ppl --cpu --row "$row" --kind "$kind" --windows "$windows" "${extra[@]}"
