#!/usr/bin/env bash
# Calibrated (whitened) per-head factors at UNIFORM ranks (stage 2f): no Fisher search, r_k for every
# k_proj group and r_v for every v_proj group. Usage: scripts/run_calibration_uniform.sh <r_k> <r_v> [gs]
# Output: compressed/uni_rk<r_k>_rv<r_v> (gs 1) or compressed/uni_glrd_rk<r_k>_rv<r_v>; factors only.
# --search_method uniform is only used for the output folder name (compress.py's --ranks_json patch
# skips rank_search() entirely whenever --ranks_json is set, so the search method itself never runs).
set -euo pipefail
source "$(dirname "$0")/env.sh"
rk="$1"; rv="$2"; gs="${3:-1}"
scripts/patch_palu_cpu.sh
name="uni_rk${rk}_rv${rv}"; [[ "$gs" != "1" ]] && name="uni_glrd_rk${rk}_rv${rv}"
groups=$((8 / gs))
mkdir -p compressed
python -m palu_tt.uniform_ranks --out "compressed/${name}.ranks.json" --layers 32 --groups "$groups" --r-k "$rk" --r-v "$rv"
cd third_party/palu
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
nice -n 10 python compress.py --model_id "$HF_DIR" --device cpu --calib_dataset wikitext2 --calib_seqlen 1024 \
  --param_ratio_target 1.0 --search_method uniform --decompose_method whiten --head_group_size "$gs" \
  --ranks_json "../../compressed/${name}.ranks.json" --use_cache --dump_huggingface_model \
  2>&1 | tee "../../compressed/calibration_${name}.log"
out="$(ls -d *_ratio-1.0_gs-${gs}-uniform-whiten 2>/dev/null | head -1)"
python - "$out" <<'PY'
import sys, os
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
d = Path(sys.argv[1]); tensors = {}
for f in sorted(d.glob("*.safetensors")):
    with safe_open(str(f), framework="pt") as sf:
        for k in sf.keys():
            if ".k_proj." in k or ".v_proj." in k:
                tensors[k] = sf.get_tensor(k).contiguous()
save_file(tensors, str(d / "palu_lowrank.safetensors"), metadata={"format": "pt"})
for f in d.glob("model*.safetensors"):
    os.remove(f)
print("kept", len(tensors), "low-rank tensors, removed the full dump")
PY
mv "$out" "../../compressed/${name}"
echo "done -> compressed/${name}"
