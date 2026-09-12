#!/usr/bin/env bash
# Palu's own recipe on CPU: Fisher-uniform rank search + whitened decomposition (float64 whitening via
# scripts/patch_palu_cpu.sh). Usage: scripts/run_calibration.sh <ratio> [group_size]   (group_size 1 = M-LRD,
# 4 = G-LRD one group per chip). Output: compressed/palu_full_<ratio> (gs 1) or compressed/glrd_<ratio>.
# The full HF dump is 30 GB; only the k_proj/v_proj factors are kept (palu_lowrank.safetensors, ~0.6 GB).
set -euo pipefail
source "$(dirname "$0")/env.sh"
ratio="${1:-0.5}"
gs="${2:-1}"
scripts/patch_palu_cpu.sh
uv pip install loguru click lm-eval==0.4.2 2>/dev/null || true
cd third_party/palu
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
name="palu_full_${ratio}"; [[ "$gs" != "1" ]] && name="glrd_${ratio}"
nice -n 10 python compress.py --model_id "$HF_DIR" --device cpu --calib_dataset wikitext2 --calib_seqlen 1024 \
  --param_ratio_target "$ratio" --search_method fisher_uniform --decompose_method whiten --head_group_size "$gs" \
  --use_cache --dump_huggingface_model 2>&1 | tee "../../compressed/calibration_${name}.log"
out="$(ls -d *_ratio-${ratio}_gs-${gs}-fisher_uniform-whiten 2>/dev/null | head -1)"
python - "$out" <<'PY'
import sys, glob, os
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
