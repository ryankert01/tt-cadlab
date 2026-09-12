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

# Task datasets (one-time, needs the proxy's CA): HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
#   SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
#   python -c 'from palu_tt import eval_tasks; import lm_eval; lm_eval.tasks.get_task_dict(eval_tasks.lm_eval_names(eval_tasks.TASKS), eval_tasks.task_manager())'
