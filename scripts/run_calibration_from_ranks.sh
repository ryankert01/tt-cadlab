#!/usr/bin/env bash
# Rebuild Palu factors from a per-module rank list produced elsewhere (e.g. the paper-protocol 2048-sample
# Fisher search run on a GPU box, docs/remote/goal.md): no rank search here, the cached whitening
# statistics do the decomposition. Usage: scripts/run_calibration_from_ranks.sh <ranks.json> <name> [gs]
# Output: compressed/<name>/{palu_lowrank.safetensors,config.json}; the ranks file is copied beside it.
# <ranks.json> is the head_wise_ranks dict: {"model.layers.<i>.self_attn.k_proj": [r per group], ...}.
set -euo pipefail
source "$(dirname "$0")/env.sh"
ranks="$1"; name="$2"; gs="${3:-1}"
test -f "$ranks" || { echo "no such ranks file: $ranks"; exit 1; }
python - "$ranks" "$gs" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); gs = int(sys.argv[2]); groups = 8 // gs
ks = {k: v for k, v in d.items() if k.endswith("k_proj")}; vs = {k: v for k, v in d.items() if k.endswith("v_proj")}
assert len(ks) == 32 and len(vs) == 32, f"expected 32 k_proj and 32 v_proj entries, got {len(ks)} / {len(vs)}"
for k, v in d.items():
    assert len(v) == groups, f"{k}: {len(v)} groups, expected {groups} for group size {gs}"
    assert all(r % 32 == 0 and 32 <= r <= 128 * gs for r in v), f"{k}: ranks {v}"
print("ranks ok:", "r_k mean %.1f" % (sum(map(sum, ks.values())) / (32 * groups)), "r_v mean %.1f" % (sum(map(sum, vs.values())) / (32 * groups)))
PY
scripts/patch_palu_cpu.sh
mkdir -p compressed
cp "$ranks" "compressed/${name}.ranks.json"
cd third_party/palu
HF_DIR="$(python -c 'from palu_tt import hf; print(hf.snapshot_dir())')"
mkdir -p cache
# the output folder is keyed on ratio/gs/method; refuse to pick up a stale one from a crashed run
stale="$(ls -d *_ratio-1.0_gs-${gs}-uniform-whiten 2>/dev/null || true)"
[[ -n "$stale" ]] && { echo "refusing: stale output dir(s) present: $stale"; exit 1; }
[[ -e "../../compressed/${name}" ]] && { echo "refusing: compressed/${name} already exists"; exit 1; }
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
