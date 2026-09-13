"""Wikitext-2 perplexity. Device rows: prefill 2048-token windows, all-position hidden states, final norm +
lm_head + NLL on host in fp32. CPU rows use the HF/low-rank model directly (sanity check only)."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset

from palu_tt import generate, hf

RAW = Path("docs/results/raw")
# "wikitext" is a legacy alias that this huggingface_hub version can only resolve online;
# "Salesforce/wikitext" is the same dataset under its canonical repo id and works offline.
DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def wikitext2_ids(tokenizer) -> torch.Tensor:
    ds = load_dataset(*DATASET, split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids


def _head(hf_dir):
    t = hf.load_tensors(hf_dir, lambda k: k in ("model.norm.weight", "lm_head.weight"))
    eps = hf.load_config(hf_dir)["rms_norm_eps"]
    return t["model.norm.weight"], t["lm_head.weight"], eps


def _nll_from_hidden(h, norm_w, lm_w, eps, targets):
    """h [S, dim] pre-norm hidden states; targets [S-1] next tokens. Returns summed NLL over S-1 positions."""
    h = h.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
    nll = 0.0
    for s in range(0, h.shape[0] - 1, 512):            # chunk the [S, vocab] logits
        logits = h[s:min(s + 512, h.shape[0] - 1)] @ lm_w.T
        nll += torch.nn.functional.cross_entropy(logits, targets[s:s + logits.shape[0]], reduction="sum").item()
    return nll


def device_ppl(args, model, hf_dir, ids, seq_len=2048, n_windows=None):
    torch.set_num_threads(12)
    norm_w, lm_w, eps = _head(hf_dir)
    N = ids.shape[1]
    windows = N // seq_len if n_windows is None else min(n_windows, N // seq_len)
    nll, tokens, t0 = 0.0, 0, time.time()
    for w in range(windows):
        chunk = ids[:, w * seq_len:(w + 1) * seq_len]
        h = generate.prefill_hidden_states(args, model, chunk)
        nll += _nll_from_hidden(h, norm_w, lm_w, eps, chunk[0, 1:])
        tokens += seq_len - 1
        if w % 10 == 0:
            print(f"window {w}/{windows} running ppl {math.exp(nll / tokens):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {"ppl": math.exp(nll / tokens), "nll": nll, "tokens": tokens, "windows": windows, "seq_len": seq_len}


@torch.no_grad()
def cpu_ppl(lm, ids, seq_len=2048, n_windows=None):
    torch.set_num_threads(12)
    N = ids.shape[1]
    windows = N // seq_len if n_windows is None else min(n_windows, N // seq_len)
    nll, tokens = 0.0, 0
    for w in range(windows):
        chunk = ids[:, w * seq_len:(w + 1) * seq_len]
        logits = lm(chunk).logits[0, :-1]
        nll += torch.nn.functional.cross_entropy(logits, chunk[0, 1:], reduction="sum").item()
        tokens += seq_len - 1
    return {"ppl": math.exp(nll / tokens), "nll": nll, "tokens": tokens, "windows": windows, "seq_len": seq_len}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True, help="results row name, e.g. baseline, svd_r64")
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--windows", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--cpu", action="store_true", help="CPU HF/low-rank model instead of the device")
    p.add_argument("--latent-dtype", choices=["bfp8", "bfp4"], default="bfp8",
                   help="tile format of the Palu latent K/V caches (default bfp8, the model's KV cache format).")
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.cpu:
        from palu_tt import reference
        from transformers import AutoTokenizer
        hf_dir = hf.snapshot_dir()
        tok = AutoTokenizer.from_pretrained(str(hf_dir))
        lm = reference.build_cpu_lowrank_llama(hf_dir, a.palu_dir if a.kind == "palu" else None)
        r = cpu_ppl(lm, wikitext2_ids(tok), a.seq_len, a.windows)
        out = RAW / f"ppl_cpu_{a.row}.json"
    else:
        import ttnn
        from palu_tt import device as dev, model as pm
        LATENT_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        md = dev.open_n300()
        try:
            args, model, _ = pm.build_model(md, a.kind, a.palu_dir, max_batch_size=1, max_seq_len=a.seq_len,
                                            latent_dtype=LATENT_DTYPES[a.latent_dtype])
            r = device_ppl(args, model, hf.snapshot_dir(), wikitext2_ids(args.tokenizer), a.seq_len, a.windows)
        finally:
            dev.close(md)
        out = RAW / f"ppl_{a.row}.json"
    r.update({"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir, "latent_dtype": a.latent_dtype})
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
