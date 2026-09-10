"""Decode-path fidelity of the fused kernel: naive vs fused Palu model, teacher-forced.

Perplexity in this study is measured through prefill windows, so it cannot see the decode kernel
at all. This harness feeds both models the same prompt (the first 512 wikitext-2 tokens), then the
same next ``steps`` tokens one decode step at a time, and compares the per-step logits: PCC per step
and how often the argmax agrees. The two models are built one after the other (two 8B models do not
both fit next to their caches).

Usage: python -m palu_tt.decode_check --row palu_full_50_fk2 --palu-dir compressed/palu_full_0.5 --fused p2
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from models.common.utility_functions import comp_pcc

from palu_tt import device as dev, eval_ppl, generate, model as pm

RAW = Path("docs/results/raw")
PROMPT = 512


def _pcc(a, b):
    """PCC of two logit vectors.

    ``comp_pcc`` returns ``(passed, value)``; on this tt-metal (v0.78.0) ``value`` is already a
    float, but older revisions return a ``"PCC: 0.9987"`` message instead, so accept both rather
    than depending on which one is installed.
    """
    v = comp_pcc(a, b, 0.0)[1]
    if isinstance(v, str):
        return float(v.split("PCC:")[-1].strip())
    return float(v)


def _logits(md, palu_dir, ids, steps, **flags):
    """[steps, vocab] fp32 logits: one prefill of the prompt, then ``steps - 1`` teacher-forced decode steps."""
    args, model, cfg = pm.build_model(md, "palu", palu_dir, max_batch_size=1, max_seq_len=2048, **flags)
    prompt, teacher = ids[0, :PROMPT], ids[0, PROMPT:PROMPT + steps]
    out = [generate.prefill_last_logits(args, model, prompt[None]).float()]
    pos = PROMPT
    for t in teacher[:-1].tolist():
        out.append(generate.decode_step(args, model, cfg, t, pos).float())
        pos += 1
    del model
    gc.collect()
    return torch.stack(out)  # [steps, vocab]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--row", required=True)
    p.add_argument("--palu-dir", required=True)
    p.add_argument("--fused", choices=["p1", "p2"], required=True)
    p.add_argument("--fused-k-chunk", type=int, default=None)
    p.add_argument("--steps", type=int, default=64)
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    md = dev.open_n300()
    try:
        args_tok = pm.ModelArgs(md, instruct=True, max_batch_size=1, max_seq_len=2048, cache_hf=False)
        ids = eval_ppl.wikitext2_ids(args_tok.tokenizer)
        naive = _logits(md, a.palu_dir, ids, a.steps)
        fused = _logits(md, a.palu_dir, ids, a.steps, use_fused_k=True, cs_recurrence=a.fused == "p2",
                        fused_k_chunk=a.fused_k_chunk)
    finally:
        dev.close(md)
    pccs = [_pcc(naive[i], fused[i]) for i in range(a.steps)]
    r = {"row": a.row, "palu_dir": a.palu_dir, "fused": a.fused, "fused_k_chunk": a.fused_k_chunk, "steps": a.steps,
         "pcc_mean": sum(pccs) / len(pccs), "pcc_min": min(pccs),
         "argmax_agree": float((naive.argmax(-1) == fused.argmax(-1)).float().mean())}
    (RAW / f"decode_check_{a.row}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
