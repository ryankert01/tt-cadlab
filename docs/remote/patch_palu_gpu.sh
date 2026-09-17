#!/usr/bin/env bash
# GPU-side patches for a fresh clone of Palu (https://github.com/shadowpa0327/palu.git, commit bb22666).
# Run from inside the clone: `bash patch_palu_gpu.sh`. Idempotent: it resets the touched files from git
# first. This is the subset of the CPU study's patch set that matters on a CUDA box; the CPU-only
# changes (fp32 model, no device_map) are deliberately NOT here.
set -euo pipefail
test -f compress.py && test -d palu || { echo "run this from the Palu clone root"; exit 1; }
git checkout -- palu/decomposition.py palu/rank_search.py palu/data_utils.py \
  palu/model/modules/hadamard_utils.py palu/model/modules/svd_linear.py

# 1. rank_search.py: the "fisher" and "fisher_uniform" branches hard-code 2048 Fisher samples and
#    ignore --n_fisher_calib_samples. Honour the flag (we pass 2048 explicitly, the paper's value).
sed -i 's/get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)/get_calib_data(args.calib_dataset, tokenizer, args.model_id, args.n_fisher_calib_samples, seqlen=args.calib_seqlen)/g' palu/rank_search.py

# 2. rank_search.py: key the Fisher cache by the sample count, so runs with different counts never
#    share a cache file.
sed -i "s|cache_file = f\"cache/{model_id.replace('/','_')}_calib_fisher_info.pt\"|cache_file = f\"cache/{model_id.replace('/','_')}_calib_fisher_info_n{len(calib_loader)}.pt\"|" palu/rank_search.py

# 3. data_utils.py: the bare "wikitext" dataset id no longer resolves on the Hub; the Salesforce mirror
#    is byte-identical.
sed -i 's/load_dataset("wikitext", "wikitext-2-raw-v1", split="train")/load_dataset("Salesforce\/wikitext", "wikitext-2-raw-v1", split="train")/' palu/data_utils.py

# 4. hadamard_utils.py: fast_hadamard_transform is only needed with hadamard=True, which the
#    fisher_uniform + whiten recipe never sets; make the import optional so the extension need not build.
python3 - <<'PY'
import pathlib
p = pathlib.Path("palu/model/modules/hadamard_utils.py"); s = p.read_text()
s = s.replace(
    "import torch, math\nimport fast_hadamard_transform\n",
    "import torch, math\ntry:\n    import fast_hadamard_transform\nexcept ImportError:\n    fast_hadamard_transform = None\n",
    1,
)
p.write_text(s)
PY

# 5. decomposition.py: newer transformers (>= 4.45) pass `position_embeddings` to every decoder layer
#    and, from 5.x, return the hidden states directly instead of a tuple. Palu's Catcher (which snapshots
#    the first-layer inputs for the whitening statistics) captured neither; forward the kwarg and accept
#    both return types. Anchors are upstream's own lines, so this applies on any transformers version.
python3 - <<'PY'
import pathlib
p = pathlib.Path("palu/decomposition.py"); s = p.read_text()
old = "    cache = {'i': 0, 'attention_mask': None, \"position_ids\": None}\n"
assert old in s, "anchor missing: Catcher cache dict init"
s = s.replace(old, "    cache = {'i': 0, 'attention_mask': None, \"position_ids\": None, \"position_embeddings\": None}\n", 1)
old = (
"        def forward(self, inp, **kwargs):\n"
"            inps[cache['i']] = inp\n"
"            cache['i'] += 1\n"
"            if cache['attention_mask'] is None:\n"
"                cache['attention_mask'] = kwargs['attention_mask']\n"
"                cache['position_ids'] = kwargs['position_ids']\n"
"            else:\n"
"                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)\n"
"                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)\n"
"            raise ValueError\n"
)
assert old in s, "anchor missing: Catcher.forward body"
new = (
"        def forward(self, inp, **kwargs):\n"
"            inps[cache['i']] = inp\n"
"            cache['i'] += 1\n"
"            if cache['i'] == 1:\n"
"                cache['attention_mask'] = kwargs.get('attention_mask')\n"
"                cache['position_ids'] = kwargs.get('position_ids')\n"
"                cache['position_embeddings'] = kwargs.get('position_embeddings')\n"
"            else:\n"
"                if kwargs.get('attention_mask') is not None and cache['attention_mask'] is not None:\n"
"                    cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)\n"
"                if kwargs.get('position_ids') is not None and cache['position_ids'] is not None:\n"
"                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)\n"
"            raise ValueError\n"
)
s = s.replace(old, new, 1)
old = (
"        for j in range(inps.shape[0]):\n"
"            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(0))[0]\n"
)
assert old in s, "anchor missing: layer-wise replay call"
new = (
"        for j in range(inps.shape[0]):\n"
"            _kw = dict(attention_mask=attention_masks,\n"
"                       position_ids=position_ids[0].unsqueeze(0) if position_ids is not None else None)\n"
"            if cache.get('position_embeddings') is not None:\n"
"                _kw['position_embeddings'] = cache['position_embeddings']\n"
"            _out = layer(inps[j].unsqueeze(0), **_kw)\n"
"            outs[j] = _out[0] if isinstance(_out, (tuple, list)) else _out\n"
)
s = s.replace(old, new, 1)
p.write_text(s)
PY

# 6. svd_linear.py: do the whitening inverse and product in float64. On Llama-3.1-8B the whitening
#    matrices of layers 0-1 have condition numbers around 1e16; in float32 the factorisation of those
#    layers is destroyed (full-rank reconstruction error 47 % / 235 %), in float64 it is exact to 1e-9.
python3 - <<'PY'
import pathlib
p = pathlib.Path("palu/model/modules/svd_linear.py"); s = p.read_text()
old = """    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
"""
new = """    # Get the inverse of scaling_diag_matrix (float64: the matrix can be ill-conditioned, ~1e16+)
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float64))

    # Multiply scaling_diag_matrix to weight matrix (float64 end to end; cast back at the end)
    W_scale = torch.matmul(weight.to(torch.float64), scaling_diag_matrix.to(torch.float64))
"""
assert old in s, "svd_linear.py anchor not found"
p.write_text(s.replace(old, new))
PY
grep -q "float64" palu/model/modules/svd_linear.py || { echo "float64 patch missing"; exit 1; }
grep -q "args.n_fisher_calib_samples, seqlen" palu/rank_search.py || { echo "n_fisher patch missing"; exit 1; }
echo "patch_palu_gpu.sh: applied cleanly"
