# source this file; never execute it
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TT_METAL_HOME="$ROOT/third_party/tt-metal"
export PYTHONPATH="$TT_METAL_HOME:$ROOT"
export MESH_DEVICE=N300
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=12
export TT_METAL_LOGGER_LEVEL=${TT_METAL_LOGGER_LEVEL:-ERROR}
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
