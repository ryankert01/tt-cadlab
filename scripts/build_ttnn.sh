#!/usr/bin/env bash
# Incremental tt-metal rebuild after host-side C++ edits (program factory, device op, nanobind,
# CMake lists). Kernels under device/kernels/ are JIT-compiled at first op run and need no rebuild.
# The `install` target is what build_metal.sh runs: it also copies the Python extension into the
# source tree (third_party/tt-metal/ttnn/ttnn/_ttnn.so), which is the file the venv imports.
# Shared server: niced, 10 parallel jobs.
set -euo pipefail
cd "$(dirname "$0")/../third_party/tt-metal"
nice -n 10 env CMAKE_BUILD_PARALLEL_LEVEL=10 cmake --build build --target install
