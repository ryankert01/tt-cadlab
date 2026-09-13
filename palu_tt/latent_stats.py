"""Per-component standard deviation of the K and V latents on wikitext windows, through the CPU reference
model (stage 2f, spec section 3): shows whether Palu's balanced split leaves a decaying variance profile
and whether the unit rescale flattens it on the real factors.

    python -m palu_tt.latent_stats --palu-dir compressed/palu_full_0.5 --layers 0 15 --windows 2 [--latent-scale unit]
"""
import argparse
import json
from pathlib import Path

import torch

from palu_tt import convert, eval_ppl, hf, reference

RAW = Path("docs/results/raw")


def latent_std(model, ids, layers, windows, seq_len=2048):
    stats = {}
    hooks = []
    captured = {}

    def hook(layer, proj):
        def fn(module, inp, out):
            x = inp[0]
            lat = (x @ module.VT.T).reshape(-1, module.n_kv, module.r)
            captured.setdefault((layer, proj), []).append(lat.float())
        return fn

    for i in layers:
        attn = model.model.layers[i].self_attn
        hooks.append(attn.k_proj.register_forward_hook(hook(i, "k")))
        hooks.append(attn.v_proj.register_forward_hook(hook(i, "v")))
    with torch.no_grad():
        for w in range(windows):
            model(ids[:, w * seq_len:(w + 1) * seq_len])
    for h in hooks:
        h.remove()
    for (i, proj), chunks in captured.items():
        lat = torch.cat(chunks, 0)                       # [tokens, n_kv, r]
        std = lat.std(0)                                 # [n_kv, r]
        stats[f"layer{i}_{proj}"] = {"std_per_component_head0": std[0].tolist(),
                                     "max_over_min": float(std.max(1).values.div(std.min(1).values).mean())}
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--palu-dir", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 15])
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--latent-scale", choices=["balanced", "unit"], default="balanced")
    a = p.parse_args()
    from transformers import AutoTokenizer
    hf_dir = hf.snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    lm = reference.build_cpu_lowrank_llama(hf_dir, a.palu_dir, latent_scale=a.latent_scale)
    ids = eval_ppl.wikitext2_ids(tok)
    r = {"palu_dir": a.palu_dir, "latent_scale": a.latent_scale, "layers": a.layers, "windows": a.windows,
         "stats": latent_std(lm, ids, a.layers, a.windows)}
    name = Path(a.palu_dir).name + ("_u" if a.latent_scale == "unit" else "")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"latent_stats_{name}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v["max_over_min"] for k, v in r["stats"].items()}))


if __name__ == "__main__":
    main()
