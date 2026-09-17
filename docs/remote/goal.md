# Goal: Palu rank search with the paper's 2048 Fisher samples, on a GPU box

You are Claude Code on a machine with an NVIDIA GPU. This machine is not the machine where the main study lives (a Tenstorrent n300 host with no GPU). The two machines do not talk to each other; the only thing that travels between them is this text (pasted in) and the final report you print (copied out). Everything you need is in this message, including the patch script in section 2; there is no repository of ours to fetch. Work autonomously; when something is blocked, stop and write down exactly what is blocked rather than working around a restriction.

## Why this exists, in three lines

Palu (Chang et al. 2024, "Palu: Compressing KV-Cache with Low-Rank Projection") factorises a Llama layer's K and V projections into low-rank pairs and caches the small latent instead of K and V. Its recipe decides each layer's rank with a Fisher-information search over calibration text, then decomposes in a whitened basis. The main study ran this recipe on the n300 host's CPU, which is so slow that the Fisher search used 32 calibration samples instead of the paper's 2048; that is the one known gap to the paper's protocol. The Fisher search needs backward passes through the full model, which only a GPU can do in reasonable time.

## What to bring back (the whole deliverable is a few KB of text)

For each of the four configurations below:
1. `ranks_<name>_f2048.json`: the per-layer rank lists that the search chose (the `head_wise_ranks` block of the dumped model's `config.json`). This is the only thing the main study needs; it regenerates the factors locally from these ranks in minutes.
2. The calibration log (`calibration_<name>.log`).
3. Quality numbers measured here on the GPU (section 6): wikitext-2 perplexity for each dump and for the unmodified model, and, if time allows, the five zero-shot task accuracies.

Do NOT bring back the dumped models (30 GB each), the Fisher cache (about 5 GB) or the whitening cache. Print the final report (section 8) at the end so it can be copied out of the terminal.

## 1. Preflight: check before installing anything

Run these and record the answers in the report. If any of them fails, stop after the preflight and report.

- `nvidia-smi`: which GPU, how much memory. Below 24 GB the runs will not fit; 24 to 40 GB works in fp16; 80 GB is comfortable.
- `df -h .`: at least 80 GB free (one model download, one dump at a time, the caches).
- `python3 --version` (3.10 or 3.11 preferred), and whether `pip install` can reach PyPI (`pip download --no-deps --dest /tmp tqdm`).
- Whether the Hugging Face Hub is reachable and the gated model is accessible: `huggingface-cli whoami` (needs `HF_TOKEN` or a prior login) and `huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --include "config.json"`. If the model is already on disk somewhere (ask the person who carried this file; look for a directory containing `config.json` and `model-0000*-of-*.safetensors` with `"_name_or_path"` mentioning Llama-3.1-8B-Instruct), use that path as `--model_id` everywhere below instead of downloading.
- Whether `git clone https://github.com/shadowpa0327/palu.git` works.
- Whether `datasets` can load `Salesforce/wikitext` (`wikitext-2-raw-v1`). The calibration also uses it.

Do not try to tunnel, proxy, or otherwise route around a network restriction; report it.

## 2. Setup

```bash
git clone https://github.com/shadowpa0327/palu.git
cd palu && git checkout bb22666
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# torch with CUDA for this box (pick the wheel index that matches `nvidia-smi`'s CUDA version, e.g. cu121):
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Llama-3.1 needs transformers >= 4.43 (rope_scaling "llama3"); stay below 5.0 to match the pinned scripts:
pip install "transformers==4.46.3" "datasets==2.19.2" accelerate loguru safetensors omegaconf yacs einops tqdm sentencepiece "numpy<2"
```

Then create `patch_palu_gpu.sh` in the clone root with exactly this content (write it with a heredoc or your file tool; do not retype it) and run `bash patch_palu_gpu.sh`:

```bash
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
```

The patch script resets the touched files from git and re-applies, so it is safe to run again. It does five things: honours `--n_fisher_calib_samples`, keys the Fisher cache by sample count, points the dataset at `Salesforce/wikitext`, makes the CUDA Hadamard extension optional, and does the whitening inverse in float64 (on Llama-3.1-8B the whitening matrices of layers 0 and 1 are conditioned around 1e16 and float32 destroys those two layers; this was found the hard way).

Do not run `pip install -e 3rdparty/fast-hadamard-transform`; it is not needed. `3rdparty/lm-evaluation-harness` is needed only for section 6b.

## 3. The four runs

All runs share one Fisher pass (cached after the first run, keyed `_n2048`) and one whitening pass (cached). Expected on an 80 GB A100: the Fisher pass 1 to 3 hours (2048 sequences of 1024 tokens, forward and backward), the whitening pass about 10 minutes, then a few minutes per run for the decomposition plus the time to write a 30 GB dump. On a 24 GB GPU expect two to four times longer.

Set `MODEL` to `meta-llama/Llama-3.1-8B-Instruct` or to the local model directory. Run them in this order, one at a time:

| name | `--param_ratio_target` | `--head_group_size` | meaning |
|---|---|---|---|
| `mlrd_0.5` | 0.5 | 1 | one latent per KV head, half the cache |
| `mlrd_0.7` | 0.7 | 1 | one latent per KV head |
| `glrd_0.5` | 0.5 | 4 | four KV heads share one latent |
| `glrd_0.7` | 0.7 | 4 | four KV heads share one latent |

```bash
mkdir -p cache results
for cfg in "mlrd_0.5 0.5 1" "mlrd_0.7 0.7 1" "glrd_0.5 0.5 4" "glrd_0.7 0.7 4"; do
  set -- $cfg; name=$1; ratio=$2; gs=$3
  python compress.py --model_id "$MODEL" --calib_dataset wikitext2 --calib_seqlen 1024 \
    --param_ratio_target "$ratio" --search_method fisher_uniform --decompose_method whiten \
    --head_group_size "$gs" --n_fisher_calib_samples 2048 --use_cache --dump_huggingface_model \
    2>&1 | tee "results/calibration_${name}.log"
  dump="$(ls -d *_ratio-${ratio}_gs-${gs}-fisher_uniform-whiten | head -1)"
  python - "$dump" "results/ranks_${name}_f2048.json" <<'PY'
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/config.json"))
ranks = cfg["head_wise_ranks"]
json.dump(ranks, open(sys.argv[2], "w"), indent=1)
ks = [v for k, v in ranks.items() if "k_proj" in k]; vs = [v for k, v in ranks.items() if "v_proj" in k]
g = len(ks[0])
print(sys.argv[2], "layers", len(ks), "groups/layer", g,
      "r_k mean %.1f min %d max %d" % (sum(map(sum, ks)) / (len(ks) * g), min(map(min, ks)), max(map(max, ks))),
      "r_v mean %.1f min %d max %d" % (sum(map(sum, vs)) / (len(vs) * g), min(map(min, vs)), max(map(max, vs))))
assert all(r % 32 == 0 for row in ks + vs for r in row), "ranks must be multiples of 32"
PY
done
```

Keep each dump until its quality check (section 6) is done, then delete it (`rm -rf "$dump"`) before the next run if disk is tight. The Fisher and whitening caches in `cache/` stay.

Sanity expectations, from the 32-sample runs on the main study's host (the 2048-sample ranks should be in the same neighbourhood; large departures are worth a note, not a retry):
- `mlrd_0.5`: r_k 32 in most layers (mean 33), r_v 64 to 128 (mean 102).
- `mlrd_0.7`: r_k 32 to 128 (mean 58), r_v 96 to 128 (mean 121).
- `glrd_0.5` (per group of 4 heads): r_k 64 to 288 (mean about 110), r_v 192 to 512 (mean about 400).
- `glrd_0.7`: r_k 160 to 512 (mean about 230), r_v 352 to 512 (mean about 485).

## 4. If a run fails

- `CUDA out of memory` in the Fisher pass: the patch already restricts nothing on GPU; try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and if it still fails, halve `--calib_seqlen` to 512 and say so in the report (the main study used 1024).
- A dataset id that will not resolve: the patch already points wikitext at `Salesforce/wikitext`; do not substitute a different text corpus.
- An anchor assertion inside the patch script: the clone is not at commit bb22666; check out that commit and re-run the patch.
- Anything network-related: stop and report.

## 5. What NOT to do

- Do not change the recipe (`fisher_uniform`, `whiten`, `--calib_seqlen 1024`, 2048 Fisher samples) except as section 4 allows, and record every deviation.
- Do not push, upload or sync anything anywhere. Do not fetch the main study's repository; it is not online.
- Do not spend more than about 12 hours of wall clock on this; if the Fisher pass alone is heading past 6 hours, report the rate and stop.

## 6. Quality check on the GPU (fp16, the paper's own scripts)

This gives two things the main study cannot produce: a check that the calibration is sane before anyone spends device time on it, and a GPU reference for the same factorisation, so the n300 numbers can be split into "what the factorisation costs" and "what the port costs".

6a. Perplexity, for the unmodified model and for every dump (a few minutes each):

```bash
python run_ppl_eval.py --model_name_or_path "$MODEL" --datasets wikitext2 --seqlen 2048 2>&1 | tee results/ppl_baseline.log
python run_ppl_eval.py --model_name_or_path "$dump"  --datasets wikitext2 --seqlen 2048 2>&1 | tee "results/ppl_${name}.log"
```

Expected scale (main study, on the n300, bfp8 caches, 141 windows of 2048): baseline 7.25; M-LRD 0.7 9.03; M-LRD 0.5 12.93; G-LRD 0.7 (256 Fisher samples) 8.14; G-LRD 0.5 10.56. The Palu paper (Llama-3-8B-Instruct, fp16, 50 %) reports 8.28 baseline and 12.38 M-LRD. Your numbers should land between those two pictures; a dump whose perplexity is above 30 means the whitening went wrong (check that the float64 patch applied).

6b. Zero-shot tasks, if time allows (about 30 to 60 minutes per model on an A100; run the baseline and the two 0.7 dumps at least):

```bash
pip install -e 3rdparty/lm-evaluation-harness      # lm-eval 0.4.2, as pinned by Palu
python run_lm_eval.py --model_name_or_path "$MODEL" --tasks "openbookqa,piqa,arc_easy,arc_challenge,winogrande" 2>&1 | tee results/tasks_baseline.log
python run_lm_eval.py --model_name_or_path "$dump"  --tasks "openbookqa,piqa,arc_easy,arc_challenge,winogrande" 2>&1 | tee "results/tasks_${name}.log"
```

Known snag: lm-eval 0.4.2's task configs name dataset ids that the Hub no longer serves. If `openbookqa`, `piqa` or `winogrande` fail to load, override their `dataset_path` to `allenai/openbookqa`, `ybisk/piqa`, `allenai/winogrande` (copy the task YAML, edit the path, pass the directory with `--include_path` if the wrapper exposes it, or edit the installed YAML in place). The main study did the same. Report `acc` per task and the mean of the five; HellaSwag is deliberately omitted.

## 7. Verify before reporting

- Each `ranks_*.json` has 64 keys (32 layers x k_proj/v_proj), lists of length 8 (gs 1) or 2 (gs 4), every entry a multiple of 32, and `sha256sum results/ranks_*.json` recorded.
- The calibration logs contain the Fisher and whitening phases and end without a traceback.
- The versions actually used: `python -c "import torch, transformers, datasets; print(torch.__version__, torch.version.cuda, transformers.__version__, datasets.__version__)"`, plus the GPU model.

## 8. Final report (print this, then stop)

```
== Palu 2048-sample calibration on <hostname / GPU> ==
preflight: <each check and its answer>
versions: <torch / cuda / transformers / datasets>, palu commit bb22666 + patch_palu_gpu.sh
deviations: <none, or the list>
wall clock: fisher <h:mm>, whitening <mm>, per run <mm>

ranks (paste each file verbatim):
--- results/ranks_mlrd_0.5_f2048.json  sha256 <...>
<file contents>
--- results/ranks_mlrd_0.7_f2048.json  sha256 <...>
<file contents>
--- results/ranks_glrd_0.5_f2048.json  sha256 <...>
<file contents>
--- results/ranks_glrd_0.7_f2048.json  sha256 <...>
<file contents>

quality on this GPU (fp16, wikitext-2 seqlen 2048; lm-eval zero-shot acc):
| model | ppl | OBQA | PIQA | ARC-e | ARC-c | Wino | mean |
| baseline | ... |
| mlrd_0.5 | ... |
| mlrd_0.7 | ... |
| glrd_0.5 | ... |
| glrd_0.7 | ... |

blocked / not done: <anything from sections 1, 4, 6>
```

## 9. What happens back on the main machine (for the reader, not for you)

The printed report is carried back as text. The four rank lists are pasted into files there and `scripts/run_calibration_from_ranks.sh` rebuilds the factors from the cached whitening statistics in about ten minutes per dump; the usual screens and evaluations follow, and the write-up compares 32-sample against 2048-sample ranks, and the n300's numbers against this GPU's for the same factorisation. Because the report is copied by hand, print the JSON files compactly (one line each is fine: `python -c "import json;print(json.dumps(json.load(open('results/ranks_mlrd_0.5_f2048.json'))))"`) and include the sha256 so a paste error can be caught.
