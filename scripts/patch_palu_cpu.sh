#!/usr/bin/env bash
# Idempotent CPU/transformers-5.x patches on the third_party/palu submodule
# working tree. Never committed inside the submodule (the parent repo does
# not record a changed submodule pointer either).
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)/third_party/palu"
cd "$P"
git checkout -- palu/decomposition.py palu/rank_search.py palu/data_utils.py \
  palu/model/modules/hadamard_utils.py utils.py

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
echo "patch_palu_cpu.sh: applied cleanly"
