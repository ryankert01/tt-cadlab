"""Timing and memory numbers for the results table. All device work is untraced (stage 1).

Methodology (identical for both kinds, so the stock/Palu comparison is apples to apples):
  * warm-up iterations first (JIT compile + program cache), then the median of N timed iterations;
  * ``dev.sync`` inside the timed region, so device work is included in every sample;
  * host-side input preparation is inside the timed region too (it is part of a decode step);
  * for Palu, ``set_recon_len`` is set once per context and never changes inside a timed loop, so
    no op-shape change (and hence no recompile) can happen between samples;
  * ``palu.runtime.check`` stays False -- it reads ``current_pos`` back to host per layer.
An allocation failure (the n300 has 12 GB per chip; a non-paged batch-32 cache at 8k does not
always fit) is recorded as an ``error`` entry for that cell, not raised.
"""
import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup

from palu_tt import convert, device as dev, hf, model as pm
from palu_tt.attention import PaluAttention

RAW = Path("docs/results/raw")
# PALU_BENCH_CONTEXTS=1024,4096 shortens a smoke run; the default is what the results table uses.
CONTEXTS = tuple(int(c) for c in os.environ.get("PALU_BENCH_CONTEXTS", "1024,4096,8192").split(","))
BATCH = 32


class Mesh:
    """Holds the open mesh so a bench can reopen it after an allocation failure.

    ttnn frees a failed build's tensors on garbage collection, but a device whose allocator was
    driven into a failure mid-build can still be fragmented enough to refuse the next (smaller)
    build; reopening gives the next context a clean allocator.
    """

    def __init__(self):
        self.md = dev.open_n300()

    def reset(self):
        """Reopen the mesh so the next build sees a fresh allocator.

        Reopens through ttnn rather than ``dev.open_n300``: that helper asserts the n300 looks
        idle (an AICLK/TDP check meant to keep two jobs off the shared board), which is never
        true of a device this very process has just been driving. We already hold the device.
        """
        dev.close(self.md)
        gc.collect()
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        self.md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))
        return self.md

    def close(self):
        dev.close(self.md)


def _time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"p50_ms": statistics.median(ts) * 1e3, "min_ms": min(ts) * 1e3, "iters": iters}


def _kv_bytes(attn, args):
    """Per-chip KV bytes per token per layer, from the attention module of a built model."""
    return pm.kv_bytes_per_token_per_layer(SimpleNamespace(layers=[SimpleNamespace(attention=attn)]), args)


def _decode_one(md, kind, palu_dir, ctx):
    """One (kind, context) decode cell. A function so every device tensor it holds dies on return."""
    args, model, cfg = pm.build_model(md, kind, palu_dir, max_batch_size=BATCH, max_seq_len=ctx)
    tokens = torch.randint(0, args.vocab_size, (BATCH,))
    current_pos = torch.full((BATCH,), ctx - 1)
    pm.set_recon_len(cfg, current_pos)  # fixed for the whole timed loop -> no recompiles inside it
    host = model.prepare_decode_inputs_host(tokens, current_pos)

    def step():
        tok_d, pos_d, rope_d, _ = copy_host_to_device(host, mesh_device=md)
        logits, _ = model.ttnn_decode_forward(tok_d, pos_d, rot_mat_idxs=rope_d)
        dev.sync(md)
        ttnn.deallocate(logits)

    row = _time(step)
    row["kv_bytes_per_token_per_layer_per_chip"] = pm.kv_bytes_per_token_per_layer(model, args)
    del model
    return row


def bench_decode(mesh, kind, palu_dir):
    """Whole decode step at batch 32 for each context; the model is rebuilt per context (max_seq_len == context)."""
    rows = {}
    failed = False
    for ctx in CONTEXTS:
        reopened = False
        if failed:  # the previous context exhausted/fragmented the allocator -> start clean
            mesh.reset()
            failed, reopened = False, True
        try:
            rows[ctx] = _decode_one(mesh.md, kind, palu_dir, ctx)
        except Exception as e:  # allocation failure -> report and continue
            rows[ctx] = {"error": str(e)[:200]}
            failed = True
        rows[ctx]["mesh_reopened_first"] = reopened
        gc.collect()  # ttnn deallocates device tensors on collection
    return rows


def _attention_one(md, kind, palu_dir, ctx):
    """One attention layer, decode, batch 32. A function so the 16 GB state dict and the layer's
    device tensors are freed before the next context is built."""
    args = ModelArgs(md, instruct=True, max_batch_size=BATCH, max_seq_len=ctx, cache_hf=False)
    args.use_qk_fused = False
    args.n_layers = 1
    sd = args.load_state_dict()
    cls = Attention
    if kind == "palu":
        args.palu = convert.add_palu_to_state_dict(sd, hf.snapshot_dir(), palu_dir, 1, args.n_heads,
                                                   args.n_kv_heads, args.head_dim)
        cls = PaluAttention
    rope = RotarySetup(md, BATCH, args.head_dim, ctx, args.rope_theta, args.rope_scaling, False)
    attn = cls(md, TT_CCL(md), args, sd, weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b), layer_num=0,
               dtype=ttnn.bfloat8_b, transformation_mats=rope.get_both_trans_mats(), configuration=args)
    del sd
    gc.collect()
    current_pos = torch.full((BATCH,), ctx - 1)
    if kind == "palu":
        pm.set_recon_len(args.palu, current_pos)
    pos_tt = ttnn.from_torch(current_pos, device=md, dtype=ttnn.int32,
                             mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, None), mesh_shape=args.cluster_shape))
    rot = rope.get_rot_mats(current_pos)
    x = torch.randn(BATCH, 1, args.dim) * 0.05

    def step():
        x_tt = args.prepare_residual_tensor_decode(x, args.get_attn_input_mem_config(Mode.DECODE, None), force_replicated=True)
        y = attn.forward(x_tt, pos_tt, rot_mats=rot, mode=Mode.DECODE)
        dev.sync(md)
        ttnn.deallocate(y)

    row = _time(step, iters=20)
    row["kv_bytes_per_token_per_layer_per_chip"] = _kv_bytes(attn, args)
    del attn
    return row


def bench_attention(mesh, kind, palu_dir):
    """One attention layer, decode, batch 32, per context. Isolates the K-reconstruction cost."""
    rows = {}
    failed = False
    for ctx in CONTEXTS:
        reopened = False
        if failed:
            mesh.reset()
            failed, reopened = False, True
        try:
            rows[ctx] = _attention_one(mesh.md, kind, palu_dir, ctx)
        except Exception as e:
            rows[ctx] = {"error": str(e)[:200]}
            failed = True
        rows[ctx]["mesh_reopened_first"] = reopened
        gc.collect()
    return rows


def bench_prefill(mesh, kind, palu_dir):
    """Time to first token: a single-user prefill of L tokens (batch 1)."""
    md = mesh.md
    rows = {}
    args, model, _ = pm.build_model(md, kind, palu_dir, max_batch_size=1, max_seq_len=8192)
    for L in (2048, 8192):
        ids = torch.randint(0, args.vocab_size, (1, L))

        def step():
            x, rot_g, rot_l, *_ = model.prepare_inputs_prefill(ids, start_pos=0)
            out = model.ttnn_prefill_forward(x, rot_mats_global=rot_g, rot_mats_local=rot_l, user_id=0,
                                             get_last_token=((L - 1) // 32) * 32)
            dev.sync(md)
            ttnn.deallocate(out)

        try:
            rows[L] = _time(step, iters=5, warmup=2)
        except Exception as e:
            rows[L] = {"error": str(e)[:200]}
    del model
    gc.collect()
    return rows


def bench_maxctx(mesh, kind, palu_dir):
    """Largest max_seq_len (power of two from 1k) at batch 32 that allocates."""
    ok, kv = None, None
    for ctx in (1024, 2048, 4096, 8192, 16384, 32768):
        try:
            args, model, _ = pm.build_model(mesh.md, kind, palu_dir, max_batch_size=BATCH, max_seq_len=ctx)
            ok = ctx
            kv = pm.kv_bytes_per_token_per_layer(model, args)
            del model, args
            gc.collect()
        except Exception as e:
            gc.collect()
            return {"max_context_batch32": ok, "failed_at": ctx, "error": str(e)[:200],
                    "kv_bytes_per_token_per_layer_per_chip": kv if ok else None}
    return {"max_context_batch32": ok, "kv_bytes_per_token_per_layer_per_chip": kv}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--what", choices=["decode", "attention", "prefill", "maxctx"], required=True)
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    mesh = Mesh()
    try:
        r = {"decode": bench_decode, "attention": bench_attention, "prefill": bench_prefill,
             "maxctx": bench_maxctx}[a.what](mesh, a.kind, a.palu_dir)
    finally:
        mesh.close()
    r = {"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir, "what": a.what, "contexts": list(CONTEXTS), "data": r}
    (RAW / f"bench_{a.what}_{a.row}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
