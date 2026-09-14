#!/usr/bin/env bash
# Split-budget Fisher-uniform search (stage 2f Task 6's conditional third grid, fired by Grid B's
# K-versus-V asymmetry): one rank search per matrix, k_proj at <k_ratio> and v_proj at <v_ratio>,
# instead of upstream's single pooled budget over both -- see the --k_ratio_target/--v_ratio_target
# patch in scripts/patch_palu_cpu.sh. The K/V byte split is then ours (a Grid B corner) and Fisher
# only decides how each matrix's own budget varies across layers, which is exactly the "does
# per-layer allocation add on top of the uniform ranks" comparison the spec asks for.
#
# Usage: scripts/run_calibration_split.sh <k_ratio> <v_ratio> [gs] [n_fisher]   (ratios as fractions,
# e.g. 0.25 0.75; gs 1 = M-LRD per head, 4 = G-LRD; n_fisher = Fisher calibration samples, default
# 32, matching the palu_full_* dumps these are compared against). Output: compressed/split_k<K>_v<V>
# with the ratios as whole percentages, factors only -- the full HF dump is 30 GB and only the
# k_proj/v_proj factors are kept (palu_lowrank.safetensors, ~0.6 GB), as the sibling scripts do.
set -euo pipefail
source "$(dirname "$0")/env.sh"
kr="$1"; vr="$2"; gs="${3:-1}"; nf="${4:-32}"
scripts/patch_palu_cpu.sh
# The dump name and the --param_ratio_target come out of one helper call so they cannot disagree:
# the ratio only reaches compress.py to name its own output folder (the split targets override it
# everywhere it would otherwise be used), and that same string is what the glob below looks for.
read -r name mean <<<"$(python -c '
import sys
from palu_tt.uniform_ranks import split_dump_name
k, v = float(sys.argv[1]), float(sys.argv[2])
print(split_dump_name(k, v), (k + v) / 2)' "$kr" "$vr")"
mkdir -p compressed
cd third_party/palu
folder_glob="*_ratio-${mean}_gs-${gs}-fisher_uniform-whiten"
# A leftover dump dir from an interrupted run would be picked up by the glob below and moved into
# compressed/ as if this run had produced it. Refuse instead of shipping someone else's factors.
if compgen -G "$folder_glob" >/dev/null; then
  echo "refusing to run: third_party/palu already holds $folder_glob from an earlier run" >&2
  exit 1
fi
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
nice -n 10 python compress.py --model_id "$HF_DIR" --device cpu --calib_dataset wikitext2 --calib_seqlen 1024 \
  --param_ratio_target "$mean" --k_ratio_target "$kr" --v_ratio_target "$vr" \
  --search_method fisher_uniform --decompose_method whiten --head_group_size "$gs" \
  --n_fisher_calib_samples "$nf" --use_cache --dump_huggingface_model \
  2>&1 | tee "../../compressed/calibration_${name}.log"
out="$(ls -d $folder_glob 2>/dev/null | head -1)"
[[ -n "$out" ]] || { echo "compress.py produced no $folder_glob" >&2; exit 1; }
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
