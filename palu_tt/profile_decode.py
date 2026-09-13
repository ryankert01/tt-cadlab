"""Per-op device profile of one decode step (stage 2d).

Run *under the tracy runner*, which turns the device profiler on and writes the ops report:

    python -m tracy -r -p -o <artifacts dir> --op-support-count 6000 -t 5000 \
        -m palu_tt.profile_decode --row baseline --kind stock --ctx 4096 --steps 3

Builds the model exactly as ``palu_tt.bench`` does (batch 32, the given context, decode at the last
position), runs warm-up steps (JIT + program cache), then brackets ``--steps`` decode steps between the
``start`` / ``stop`` signposts the tt-metal post-processing understands, flushing the device profiler
after every step so the on-device buffers never overflow. The ops CSV lands under
``<artifacts dir>/reports/<date>/ops_perf_results_<date>.csv``; ``palu_tt.profile_report`` reads it.
"""
import argparse

import torch
import ttnn
from tracy import signpost

from models.tt_transformers.tt.generator import copy_host_to_device
from palu_tt import device as dev, model as pm

BATCH = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--fused", choices=["off", "p1", "p2"], default="off")
    p.add_argument("--fused-k-chunk", type=int, default=None)
    p.add_argument("--group-heads", choices=["auto", "1", "2", "4"], default="auto")
    p.add_argument("--rot-mode", choices=["legacy", "pack", "dst"], default="dst",
                   help="stage-2e K rebuild mode in the fused op (default: dst, the shipped variant; see "
                        "docs/superpowers/specs/2026-09-14-palu-tt-stage2e-design.md).")
    p.add_argument("--table-mcast", choices=["off", "on"], default="on",
                   help="stage-2e: multicast the Phase 2 angle table down each grid column, instead of "
                        "reading it per core (default: on, the shipped variant; Phase 2 only).")
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    a = p.parse_args()
    flags = {}
    if a.fused != "off":
        flags = dict(use_fused_k=True, cs_recurrence=a.fused == "p2", fused_k_chunk=a.fused_k_chunk,
                     group_heads=None if a.group_heads == "auto" else int(a.group_heads),
                     rot_mode=a.rot_mode, table_mcast=a.table_mcast == "on")
    md = dev.open_n300()
    try:
        args, model, cfg = pm.build_model(md, a.kind, a.palu_dir, max_batch_size=BATCH, max_seq_len=a.ctx, **flags)
        tokens = torch.randint(0, args.vocab_size, (BATCH,))
        current_pos = torch.full((BATCH,), a.ctx - 1)
        pm.set_recon_len(cfg, current_pos)
        host = model.prepare_decode_inputs_host(tokens, current_pos)

        def step():
            tok_d, pos_d, rope_d, _ = copy_host_to_device(host, mesh_device=md)
            logits, _ = model.ttnn_decode_forward(tok_d, pos_d, rot_mat_idxs=rope_d)
            dev.sync(md)
            ttnn.deallocate(logits)

        for _ in range(a.warmup):
            step()
        ttnn.ReadDeviceProfiler(md)          # drop the warm-up from the device buffers
        signpost("start")
        for _ in range(a.steps):
            step()
            ttnn.ReadDeviceProfiler(md)
        signpost("stop")
        print(f"profiled {a.steps} decode steps of {a.row} at ctx {a.ctx}, batch {BATCH}")
        del model
    finally:
        dev.close(md)


if __name__ == "__main__":
    main()
