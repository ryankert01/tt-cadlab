#!/usr/bin/env bash
# Idempotent CPU/transformers-5.x patches on the third_party/palu submodule
# working tree. Never committed inside the submodule (the parent repo does
# not record a changed submodule pointer either).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # resolved before the cd below: $0 may be a relative path
P="$SCRIPT_DIR/../third_party/palu"
cd "$P"
git checkout -- palu/decomposition.py palu/rank_search.py palu/data_utils.py \
  palu/model/modules/hadamard_utils.py utils.py palu/model/modules/svd_linear.py

# decomposition.py: three .cuda() sites -> stay on the working device (`dev`
# is already in scope at each call site).
sed -i 's/subset\[name\].scaling_diag_matrix.double().cuda()/subset[name].scaling_diag_matrix.double().to(dev)/' palu/decomposition.py
sed -i 's/torch.eye(raw_scaling_diag_matrix.shape\[0\]).cuda()/torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)/' palu/decomposition.py
sed -i 's/torch.eye(scaling_diag_matrix.shape\[0\]).cuda()/torch.eye(scaling_diag_matrix.shape[0]).to(dev)/' palu/decomposition.py
# (torch.cuda.empty_cache() calls elsewhere are no-ops on CPU-only torch; left as-is.)

# rank_search.py: fp32 on CPU (bf16/fp16 matmul is ~1000x slower on this CPU),
# grads restricted to attention linears only (Fisher only ever reads
# module.weight.grad for "attn" Linear modules; this keeps backward's memory
# footprint down since embed/mlp/norm params no longer need grad buffers).
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("palu/rank_search.py"); s = p.read_text()
s = s.replace("    model.half()\n    model.to(device)\n",
              "    model.float() if device.type == 'cpu' else model.half()\n    model.to(device)\n"
              "    for n, p_ in model.named_parameters():\n        p_.requires_grad_('attn' in n)\n")
p.write_text(s)
p = pathlib.Path("utils.py"); s = p.read_text()
s = s.replace('torch_dtype=torch.float16,', 'torch_dtype=torch.float32,').replace('device_map="auto",', '')
p.write_text(s)
PY

# rank_search.py: both the "fisher" and "fisher_uniform" branches hardcode
# nsamples=2048 in their get_calib_data(...) call, silently ignoring
# --n_fisher_calib_samples (default 32). At fp32 on this CPU that is >24h of
# forward+backward per Fisher pass instead of the ~1-3h the recipe budgets
# for 32 samples; honor the CLI arg instead of the upstream literal.
sed -i 's/get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)/get_calib_data(args.calib_dataset, tokenizer, args.model_id, args.n_fisher_calib_samples, seqlen=args.calib_seqlen)/g' palu/rank_search.py

# rank_search.py: the Fisher cache file is keyed by model id only, so a run with a different
# --n_fisher_calib_samples would silently reuse another run's Fisher information. Key it by the
# calibration set size as well (the stage-1 32-sample cache was renamed to the _n32 name by hand).
sed -i "s|cache_file = f\"cache/{model_id.replace('/','_')}_calib_fisher_info.pt\"|cache_file = f\"cache/{model_id.replace('/','_')}_calib_fisher_info_n{len(calib_loader)}.pt\"|" palu/rank_search.py

# data_utils.py: the canonical no-namespace "wikitext" HF dataset repo id
# fails to resolve under current huggingface_hub (hf:// URI parsing requires
# "namespace/name"); the "Salesforce/wikitext" mirror is byte-identical and
# resolves fine both online and from the offline cache.
sed -i 's/load_dataset("wikitext", "wikitext-2-raw-v1", split="train")/load_dataset("Salesforce\/wikitext", "wikitext-2-raw-v1", split="train")/' palu/data_utils.py

# hadamard_utils.py: fast_hadamard_transform is a CUDA extension that will
# not build on this CPU-only box. It is only used when a caller explicitly
# sets hadamard=True (never true in the fisher_uniform/whiten recipe below,
# where HeadwiseLowRankModule hardcodes hadamard = False), so make the
# import optional instead of building/stubbing the extension.
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

# decomposition.py: transformers 5.x moved RoPE computation out of
# LlamaDecoderLayer.forward -- LlamaModel.forward now computes
# `position_embeddings = self.rotary_emb(hidden_states, position_ids)` once
# and passes it to every layer as a kwarg; the layer no longer derives cos/sin
# from position_ids itself. Palu's Catcher (wrapping layers[0] to snapshot the
# first-layer inputs during `get_whiten_scale_matrix`) only captured
# attention_mask/position_ids from kwargs, so the later layer-wise replay loop
# called `layer(...)` without position_embeddings and hit
# `TypeError: cannot unpack non-iterable NoneType object` inside the layer
# (which does `cos, sin = position_embeddings`). Also, under SDPA with
# is_causal, attention_mask is now None, and the installed
# LlamaDecoderLayer.forward returns the hidden_states tensor directly (not a
# tuple), so the pre-existing `[0]` indexing on the layer's return value would
# silently just take the first batch row instead of erroring.
python3 - <<'PY'
import pathlib
p = pathlib.Path("palu/decomposition.py"); s = p.read_text()

old_cache_init = "    cache = {'i': 0, 'attention_mask': None, \"position_ids\": None}\n"
assert old_cache_init in s, "anchor missing: Catcher cache dict init"
new_cache_init = "    cache = {'i': 0, 'attention_mask': None, \"position_ids\": None, \"position_embeddings\": None}\n"
s = s.replace(old_cache_init, new_cache_init, 1)

old_catcher_forward = (
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
assert old_catcher_forward in s, "anchor missing: Catcher.forward body"
new_catcher_forward = (
"        def forward(self, inp, **kwargs):\n"
"            inps[cache['i']] = inp\n"
"            cache['i'] += 1\n"
"            if cache['i'] == 1:\n"
"                cache['attention_mask'] = kwargs['attention_mask']\n"
"                cache['position_ids'] = kwargs['position_ids']\n"
"                cache['position_embeddings'] = kwargs.get('position_embeddings')\n"
"            else:\n"
"                if kwargs['attention_mask'] is not None and cache['attention_mask'] is not None:\n"
"                    cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)\n"
"                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)\n"
"            raise ValueError\n"
)
s = s.replace(old_catcher_forward, new_catcher_forward, 1)

old_layer_call = (
"        for j in range(inps.shape[0]):\n"
"            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(0))[0]\n"
)
assert old_layer_call in s, "anchor missing: layer-wise replay call"
new_layer_call = (
"        for j in range(inps.shape[0]):\n"
"            outs[j] = layer(\n"
"                inps[j].unsqueeze(0),\n"
"                attention_mask=attention_masks,\n"
"                position_ids=position_ids[0].unsqueeze(0) if position_ids is not None else None,\n"
"                position_embeddings=cache['position_embeddings'],\n"
"            )\n"
)
s = s.replace(old_layer_call, new_layer_call, 1)

p.write_text(s)
PY

grep -n "cuda()" palu/*.py palu/model/modules/*.py utils.py compress.py 2>/dev/null || echo "no .cuda() left"

# compress.py: --ranks_json replaces the rank search with a supplied per-module rank list
# (stage 2f uniform-rank dumps). Keys are module names, values the per-group rank lists.
python3 - <<'PY'
import pathlib
p = pathlib.Path("compress.py"); s = p.read_text()
old = "    search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)\n"
new = ("    if getattr(args, 'ranks_json', None):\n"
       "        import json\n"
       "        search_results = json.load(open(args.ranks_json))\n"
       "        rank_sum = sum(sum(v) for v in search_results.values())\n"
       "        total_rank = 0\n"
       "    else:\n"
       "        search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)\n")
if old in s and "getattr(args, 'ranks_json'" not in s:
    s = s.replace(old, new)
if '"--ranks_json"' not in s:
    s = s.replace('    parser.add_argument(\n        "--model_id",',
                  '    parser.add_argument("--ranks_json", type=str, default=None, help="per-module rank lists; skips the rank search")\n    parser.add_argument(\n        "--model_id",', 1)
p.write_text(s)
PY
grep -n "ranks_json" compress.py >/dev/null || { echo "ranks_json patch missing"; exit 1; }

# rank_search.py + compress.py: --k_ratio_target / --v_ratio_target, a *split budget* for the
# fisher_uniform search (stage 2f Task 6's conditional third grid). Upstream's fisher_uniform pools
# every k_proj and v_proj module into one Fisher-weighted budget, so the K/V byte split is whatever
# Fisher happens to pick (at ratio 0.5 on this model: r_k 33, r_v 102 -- roughly k26/v80). Grid B
# showed perplexity is far more sensitive to r_v than to r_k at equal bytes, and the spec's trigger
# asks the complementary question: at a K/V split *we* fix (a Grid B corner), does Fisher's per-layer
# allocation still add anything on top of the uniform ranks? So the allocation runs once per matrix
# family, each restricted to its own module set with its own total rank, its own Fisher sum and its
# own ratio. With neither flag set there is exactly one pass over both families at
# --param_ratio_target and the arithmetic is upstream's, unchanged. The transform itself lives in
# scripts/patch_rank_search_split.py (factored out so tests/test_patch_palu_cpu.py can import and
# exec it against a pristine copy of rank_search.py without touching this submodule tree).
python3 "$SCRIPT_DIR/patch_rank_search_split.py"
grep -n "k_ratio_target" palu/rank_search.py >/dev/null || { echo "split-budget rank_search patch missing"; exit 1; }

python3 - <<'PY'
import pathlib
p = pathlib.Path("compress.py"); s = p.read_text()
if '"--k_ratio_target"' not in s:
    s = s.replace('    parser.add_argument("--ranks_json",',
                  '    parser.add_argument("--k_ratio_target", type=float, default=None,\n'
                  '                        help="fisher_uniform split budget: k_proj target ratio (default: --param_ratio_target)")\n'
                  '    parser.add_argument("--v_ratio_target", type=float, default=None,\n'
                  '                        help="fisher_uniform split budget: v_proj target ratio (default: --param_ratio_target)")\n'
                  '    parser.add_argument("--ranks_json",', 1)
p.write_text(s)
PY
grep -n "k_ratio_target" compress.py >/dev/null || { echo "split-budget compress.py patch missing"; exit 1; }

echo "patch_palu_cpu.sh: applied cleanly"

# svd_linear.py: whitening decomposition in float64. The whitening matrices of Llama-3.1-8B layers 0-1
# have condition numbers ~1e16-1e18; inverting them in fp32 destroys the factorisation (full-rank
# reconstruction error 47 % / 235 %). In float64 the full-rank reconstruction is exact to 1e-9.
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
grep -n "float64" palu/model/modules/svd_linear.py || { echo "svd_linear float64 patch missing"; exit 1; }
