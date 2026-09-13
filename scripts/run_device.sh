#!/usr/bin/env bash
# Gate + run: refuse to start while the n300 is busy, then exec the command.
#
# JIT kernel cache guard (stage 2e finding): tt-metal keys its compiled-kernel cache on the kernel's
# *path*, defines and compile-time args, never on the source bytes, so an edited palu kernel is silently
# run from the stale binary until the cache entry is removed. Hash the palu kernel sources on every
# device run and purge the palu entries whenever the hash changes (the first run only records it).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KDIR="$ROOT/third_party/tt-metal/ttnn/cpp/ttnn/operations/transformer/palu_sdpa_decode/device/kernels"
STAMP="${HOME}/.cache/palu_tt/palu_kernels.sha256"
if [[ -d "$KDIR" ]]; then
  cur="$(find "$KDIR" -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
  mkdir -p "$(dirname "$STAMP")"
  if [[ -f "$STAMP" ]] && [[ "$(cat "$STAMP")" != "$cur" ]]; then
    for c in "${TT_METAL_CACHE:-$HOME/.cache/tt-metal-cache}"; do
      [[ -d "$c" ]] && find "$c" -maxdepth 5 -type d -path '*/kernels/palu_*' -prune -exec rm -rf {} + 2>/dev/null || true
    done
    echo "run_device.sh: palu kernel sources changed -> purged the JIT kernel cache entries (next run recompiles)" >&2
  fi
  echo "$cur" > "$STAMP"
fi
"$(dirname "$0")/device_free.sh"
exec "$@"
