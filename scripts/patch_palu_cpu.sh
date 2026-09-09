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

grep -n "cuda()" palu/*.py palu/model/modules/*.py utils.py compress.py 2>/dev/null || echo "no .cuda() left"
echo "patch_palu_cpu.sh: applied cleanly"
