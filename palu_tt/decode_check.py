"""Decode-path fidelity of the fused kernel: naive vs fused Palu model, teacher-forced.

Perplexity in this study is measured through prefill windows, so it cannot see the decode kernel
at all. This harness feeds both models the same prompt (the first 512 wikitext-2 tokens), then the
same next ``steps`` tokens one decode step at a time, and compares the per-step logits: PCC per step
and how often the argmax agrees. The two models are built one after the other (two 8B models do not
both fit next to their caches).

The prefill logit (index 0 of ``_logits``' output) is produced by the same prefill path on both the
naive and fused model -- the fused kernel only replaces *decode* -- so it is bit-identical between
runs and is excluded from the comparison; including it would silently inflate agreement by 1/steps.
``steps`` in the JSON is the number of *decode* steps compared, i.e. ``--steps`` minus the one
prefill step that seeds them.

Usage: python -m palu_tt.decode_check --row palu_full_50_fk2 --palu-dir compressed/palu_full_0.5 --fused p2
"""
import argparse
import gc
import json
from pathlib import Path

import torch
import ttnn
from models.common.utility_functions import comp_pcc

from palu_tt import device as dev, eval_ppl, fused, generate, model as pm

RAW = Path("docs/results/raw")
PROMPT = 512
LATENT_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


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


GROUP_HEADS = []    # per-layer heads_per_core the fused model actually ran with (stage 2c)


def _logits(md, palu_dir, ids, steps, **flags):
    """[steps, vocab] fp32 logits: one prefill of the prompt, then ``steps - 1`` teacher-forced decode steps."""
    args, model, cfg = pm.build_model(md, "palu", palu_dir, max_batch_size=1, max_seq_len=2048, **flags)
    prompt, teacher = ids[0, :PROMPT], ids[0, PROMPT:PROMPT + steps]
    out = [generate.prefill_last_logits(args, model, prompt[None]).float()]
    pos = PROMPT
    for t in teacher[:-1].tolist():
        out.append(generate.decode_step(args, model, cfg, t, pos).float())
        pos += 1
    GROUP_HEADS[:] = fused.group_heads_used(cfg)
    del model
    gc.collect()
    return torch.stack(out)  # [steps, vocab]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--row", required=True)
    p.add_argument("--palu-dir", required=True)
    p.add_argument("--fused", choices=["p1", "p2"], required=True)
    p.add_argument("--fused-k-chunk", type=int, default=None)
    p.add_argument("--group-heads", choices=["auto", "1", "2", "4"], default="2",
                   help="stage-2c group-shared mode: KV heads per core in the fused op. Default 2 "
                        "here, not auto: this harness runs at batch 1, and auto's occupancy rule "
                        "returns 1 for every layer below batch 8 (a four-head group would use 4 of "
                        "the 64 cores there), so auto at batch 1 measures the stage 2b per-head "
                        "kernel. 2 exercises the group kernel and its sequence-split reduction on "
                        "every layer without the allocator fallback that 4 hits on most layers at "
                        "batch 1 (4 is a ceiling the run may fall back from). Pass 1 for the "
                        "per-head kernel on purpose, or auto to mirror what bench picks.")
    p.add_argument("--rot-mode", choices=["legacy", "pack", "dst"], default="legacy",
                   help="stage-2e K rebuild mode in the fused op (see docs/superpowers/specs/2026-09-14-palu-tt-stage2e-design.md).")
    p.add_argument("--table-mcast", choices=["off", "on"], default="off",
                   help="stage-2e: multicast the Phase 2 angle table down each grid column (Phase 2 only).")
    p.add_argument("--latent-dtype", choices=["bfp8", "bfp4"], default="bfp8",
                   help="tile format of the Palu latent K/V caches (default bfp8, the model's KV cache format).")
    p.add_argument("--steps", type=int, default=64)
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    md = dev.open_n300()
    try:
        args_tok = pm.ModelArgs(md, instruct=True, max_batch_size=1, max_seq_len=2048, cache_hf=False)
        ids = eval_ppl.wikitext2_ids(args_tok.tokenizer)
        naive = _logits(md, a.palu_dir, ids, a.steps, latent_dtype=LATENT_DTYPES[a.latent_dtype])
        fused_logits = _logits(md, a.palu_dir, ids, a.steps, use_fused_k=True, cs_recurrence=a.fused == "p2",
                        fused_k_chunk=a.fused_k_chunk,
                        group_heads=None if a.group_heads == "auto" else int(a.group_heads),
                        rot_mode=a.rot_mode, table_mcast=a.table_mcast == "on",
                        latent_dtype=LATENT_DTYPES[a.latent_dtype])
    finally:
        dev.close(md)
    # Index 0 of both stacks is the shared prefill logit (bit-identical on both paths, since the
    # fused kernel only replaces decode) -- drop it so the comparison is over decode steps only.
    naive_decode, fused_decode = naive[1:], fused_logits[1:]
    pccs = [_pcc(naive_decode[i], fused_decode[i]) for i in range(naive_decode.shape[0])]
    r = {"row": a.row, "palu_dir": a.palu_dir, "fused": a.fused, "fused_k_chunk": a.fused_k_chunk,
         "group_heads": a.group_heads, "group_heads_per_layer": list(GROUP_HEADS),
         "rot_mode": a.rot_mode, "table_mcast": a.table_mcast, "latent_dtype": a.latent_dtype,
         "steps": naive_decode.shape[0],
         "pcc_mean": sum(pccs) / len(pccs), "pcc_min": min(pccs),
         "argmax_agree": float((naive_decode.argmax(-1) == fused_decode.argmax(-1)).float().mean())}
    (RAW / f"decode_check_{a.row}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
