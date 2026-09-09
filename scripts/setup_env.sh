#!/usr/bin/env bash
# One-time environment setup. Shared server: niced, capped at JOBS cores.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="${JOBS:-10}"
cd "$ROOT"
git submodule update --init --recursive --depth 1 third_party/tt-metal
git submodule update --init third_party/palu

cd "$ROOT/third_party/tt-metal"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export UV_SYSTEM_CERTS=1   # this host's TLS-intercepting proxy CA is only in the system store, not uv's bundled roots
nice -n 10 ./build_metal.sh --release            # clang-20 toolchain is the default; ~1.5-2 h at -j10
nice -n 10 ./create_venv.sh --env-dir "$ROOT/.venv" --force   # uv venv, dev requirements, `uv pip install -e .`

source "$ROOT/.venv/bin/activate"
uv pip install -r "$ROOT/requirements.txt"
echo "setup done. Next: source scripts/env.sh"
