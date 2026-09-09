#!/usr/bin/env bash
# Full Palu recipe (Fisher-uniform rank search + whitening) on CPU fp32. Hours; niced. Usage: run_calibration.sh 0.5
set -euo pipefail
source "$(dirname "$0")/env.sh"
ratio="${1:-0.5}"
scripts/patch_palu_cpu.sh
uv pip install loguru click lm-eval==0.4.2 2>/dev/null || true
cd third_party/palu
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
nice -n 10 python compress.py --model_id "$HF_DIR" --device cpu --calib_dataset wikitext2 --calib_seqlen 1024 \
  --param_ratio_target "$ratio" --search_method fisher_uniform --decompose_method whiten --head_group_size 1 \
  --use_cache --dump_huggingface_model 2>&1 | tee "../../compressed/calibration_${ratio}.log"
# NOTE: model_id here is the HF snapshot dir (path ends in a content hash,
# not "Llama-3.1-8B-Instruct"), so dump_to_huggingface_repos's save_folder
# name is "<hash>_ratio-...-whiten"; match on the stable suffix instead of
# assuming the human-readable model name prefix from the brief's script.
out="$(ls -d *_ratio-${ratio}_gs-1-fisher_uniform-whiten 2>/dev/null | head -1)"
mv "$out" "../../compressed/palu_full_${ratio}"
echo "done -> compressed/palu_full_${ratio}"
