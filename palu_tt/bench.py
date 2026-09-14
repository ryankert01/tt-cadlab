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
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup

from palu_tt import convert, device as dev, fused, hf, model as pm
from palu_tt.attention import PaluAttention
# LATENT_DTYPES / resolve_latent_dtypes(_names) / resolve_rot_switches live in palu_tt.model
# (imported by all four CLIs already); re-exported here so this module's own tests keep working.
from palu_tt.model import (LATENT_DTYPES, resolve_latent_dtype_names, resolve_latent_dtypes,
                           resolve_rot_switches)

RAW = Path("docs/results/raw")


def _parse_contexts(s):
    """Parse PALU_BENCH_CONTEXTS ("1024,4096,8192" -> (1024, 4096, 8192)).

    Empty items are ignored, so a trailing/doubled comma (e.g. "1024,4096,") does not crash;
    ``None`` or an empty string falls back to the default matrix used by the results table.
    """
    if not s:
        s = "1024,4096,8192"
    return tuple(int(x) for x in s.split(",") if x.strip())


# PALU_BENCH_CONTEXTS=1024,4096 shortens a smoke run; the default is what the results table uses.
CONTEXTS = _parse_contexts(os.environ.get("PALU_BENCH_CONTEXTS"))

# Stage 2 batch sweep: the batch size for this process, set once by ``main`` from ``--batch`` (same
# pattern as ``FUSED``). Defaults to 32, which reproduces every stage-1/stage-2 row exactly.
BATCH = 32

# Stage 2: the fused-kernel runtime flags for this process, set once by ``main`` from ``--fused``
# and passed to every model/attention build below. ``--fused off`` (the default) reproduces the
# stage-1 rows exactly -- ``build_model``'s own defaults are these same values, including the stage 2e
# shipped rebuild mode ("dst") and angle-table multicast (on), which are inert without use_fused_k.
FUSED = {"use_fused_k": False, "cs_recurrence": False, "fused_k_chunk": None, "group_heads": None,
         "rot_mode": "dst", "table_mcast": True, "latent_dtype": None, "k_latent_dtype": None,
         "v_latent_dtype": None, "kv_cache_dtype": None, "latent_scale": "balanced"}


# stage 2c: per-layer heads_per_core the last model/layer built actually ran with (after any allocator
# fallback), so a row's timings can be read against the H that produced them. Empty for non-grouped rows.
GROUP_HEADS = []


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
    try:
        args, model, cfg = pm.build_model(md, kind, palu_dir, max_batch_size=BATCH, max_seq_len=ctx, **FUSED)
        tokens = torch.randint(0, args.vocab_size, (BATCH,))
        current_pos = torch.full((BATCH,), ctx - 1)
        pm.set_recon_len(cfg, current_pos)  # fixed for the whole timed loop -> no recompiles inside it
        host = model.prepare_decode_inputs_host(tokens, current_pos)
    except Exception as e:
        e.phase = "build"
        raise

    def step():
        tok_d, pos_d, rope_d, _ = copy_host_to_device(host, mesh_device=md)
        logits, _ = model.ttnn_decode_forward(tok_d, pos_d, rot_mat_idxs=rope_d)
        dev.sync(md)
        ttnn.deallocate(logits)

    try:
        row = _time(step)
    except Exception as e:
        e.phase = "timing"
        raise
    row["kv_bytes_per_token_per_layer_per_chip"] = pm.kv_bytes_per_token_per_layer(model, args)
    GROUP_HEADS[:] = fused.group_heads_used(cfg)
    del model
    return row


def bench_decode(mesh, kind, palu_dir, stop_after_oom=False):
    """Whole decode step at batch 32 for each context; the model is rebuilt per context (max_seq_len == context).

    ``stop_after_oom``: break out of the context loop right after the first error cell instead of
    trying every remaining (larger) context -- used by the context-scaling scan, where a failure at
    one power-of-two context means every later (larger) one will fail too.
    """
    rows = {}  # keys are ints here; json.dumps() below turns them into strings ("1024") -- Task 10's reader must expect that
    failed = False
    for ctx in CONTEXTS:
        reopened = False
        if failed:  # the previous context exhausted/fragmented the allocator -> start clean
            mesh.reset()
            failed, reopened = False, True
        try:
            rows[ctx] = _decode_one(mesh.md, kind, palu_dir, ctx)
        except Exception as e:  # allocation failure -> report and continue
            traceback.print_exc()  # keep the details in the log; the JSON cell only gets 200 chars
            rows[ctx] = {"error": str(e)[:200], "phase": getattr(e, "phase", "unknown"), "exc": type(e).__name__}
            failed = True
        rows[ctx]["mesh_reopened_first"] = reopened
        gc.collect()  # ttnn deallocates device tensors on collection
        if failed and stop_after_oom:
            break
    return rows


def _attention_one(md, kind, palu_dir, ctx):
    """One attention layer, decode, batch 32. A function so the 16 GB state dict and the layer's
    device tensors are freed before the next context is built."""
    try:
        args = ModelArgs(md, instruct=True, max_batch_size=BATCH, max_seq_len=ctx, cache_hf=False)
        args.use_qk_fused = False
        args.n_layers = 1
        pm.set_kv_cache_dtype(args, FUSED["kv_cache_dtype"])     # --kv-dtype, exactly as build_model applies it
        sd = args.load_state_dict()
        cls = Attention
        if kind == "palu":
            args.palu = convert.add_palu_to_state_dict(sd, hf.snapshot_dir(), palu_dir, 1, args.n_heads,
                                                       args.n_kv_heads, args.head_dim, latent_scale=FUSED["latent_scale"])
            # This path builds the layer by hand rather than through ``build_model``, so it applies the
            # run's switches through the same helper ``build_model`` uses, splatting the very dict it
            # would have passed. Hand-copying them here is how --what attention silently lost
            # --latent-dtype (and would have lost every switch added since).
            pm.apply_palu_flags(args.palu, **FUSED)
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
    except Exception as e:
        e.phase = "build"
        raise

    def step():
        x_tt = args.prepare_residual_tensor_decode(x, args.get_attn_input_mem_config(Mode.DECODE, None), force_replicated=True)
        y = attn.forward(x_tt, pos_tt, rot_mats=rot, mode=Mode.DECODE)
        dev.sync(md)
        ttnn.deallocate(y)

    try:
        row = _time(step, iters=20)
    except Exception as e:
        e.phase = "timing"
        raise
    row["kv_bytes_per_token_per_layer_per_chip"] = _kv_bytes(attn, args)
    GROUP_HEADS[:] = fused.group_heads_used(getattr(args, "palu", None))
    del attn
    return row


def bench_attention(mesh, kind, palu_dir):
    """One attention layer, decode, batch 32, per context. Isolates the K-reconstruction cost."""
    rows = {}  # keys are ints here; json.dumps() below turns them into strings ("1024") -- Task 10's reader must expect that
    failed = False
    for ctx in CONTEXTS:
        reopened = False
        if failed:
            mesh.reset()
            failed, reopened = False, True
        try:
            rows[ctx] = _attention_one(mesh.md, kind, palu_dir, ctx)
        except Exception as e:
            traceback.print_exc()  # keep the details in the log; the JSON cell only gets 200 chars
            rows[ctx] = {"error": str(e)[:200], "phase": getattr(e, "phase", "unknown"), "exc": type(e).__name__}
            failed = True
        rows[ctx]["mesh_reopened_first"] = reopened
        gc.collect()
    return rows


def bench_prefill(mesh, kind, palu_dir):
    """Time to first token: a single-user prefill of L tokens (batch 1)."""
    md = mesh.md
    try:
        # stage 2f: time-to-first-token stays the stage 1..2e measurement. The prefill round trip is
        # instrumentation for the *scores* (it makes perplexity see the cache format), not something a
        # deployment has to run, and leaving it on would silently break comparability with every
        # committed bench_prefill_*.json.
        args, model, _ = pm.build_model(md, kind, palu_dir, max_batch_size=1, max_seq_len=8192,
                                        latent_quant_in_prefill=False, **FUSED)
    except Exception as e:  # allocation failure -> report both lengths and stop, like the other modes
        traceback.print_exc()
        return {L: {"error": str(e)[:200], "phase": "build", "exc": type(e).__name__} for L in (2048, 8192)}
    rows = {}  # keys are ints here; json.dumps() below turns them into strings ("2048") -- Task 10's reader must expect that
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
            traceback.print_exc()
            rows[L] = {"error": str(e)[:200], "phase": "timing", "exc": type(e).__name__}
    del model
    gc.collect()
    return rows


# Llama-3.1-8B's max_position_embeddings. bench_maxctx probes powers of two up to here, then
# bisects (multiples of 1024) between the last success and the first failure.
MODEL_MAX_CTX = 131072


def _maxctx_probe(mesh, kind, palu_dir, ctx):
    """One build-and-teardown at ``max_seq_len=ctx``. ``(ok, kv_bytes, err_dict_or_None)``."""
    try:
        args, model, _ = pm.build_model(mesh.md, kind, palu_dir, max_batch_size=BATCH, max_seq_len=ctx, **FUSED)
        kv = pm.kv_bytes_per_token_per_layer(model, args)
        del model, args
        gc.collect()
        return True, kv, None
    except Exception as e:
        traceback.print_exc()  # keep the details in the log; the JSON cell only gets 200 chars
        gc.collect()
        return False, None, {"error": str(e)[:200], "phase": "build", "exc": type(e).__name__}


def _is_oom(err):
    """``err["error"]`` names an allocation failure vs. tt_transformers rejecting the value itself
    (e.g. a hard-coded cap below the model's ``max_position_embeddings``) -- same heuristic
    ``report._timing`` already uses to tell OOM cells apart from other build errors."""
    return bool(err) and "Out of Memory" in str(err.get("error", ""))


def bench_maxctx(mesh, kind, palu_dir):
    """Largest ``max_seq_len`` that allocates, at ``BATCH``.

    Probes powers of two from 1k up to ``MODEL_MAX_CTX`` (Llama-3.1's ``max_position_embeddings``);
    on the first failure, bisects on multiples of 1024 between the last success and the first
    failure to pin the boundary down to 1k granularity. A failed build can leave the allocator
    fragmented (see the ``Mesh`` class docstring), so every failure -- in the power-of-two scan or
    during bisection -- is followed by ``mesh.reset()`` before the next probe.

    ``limit`` is ``"model"`` when every power of two up to ``MODEL_MAX_CTX`` allocated (the true
    architectural ceiling, not a resource limit) *or* when the failure that bounded the search was
    not an out-of-memory error (tt_transformers refusing the value outright); otherwise ``"oom"``.
    """
    probes = []  # [ctx, ok] in probe order, power-of-two scan then bisection
    ok_ctx, ok_kv = None, None
    fail_ctx, fail_err = None, None
    for ctx in (1024, 2048, 4096, 8192, 16384, 32768, 65536, MODEL_MAX_CTX):
        success, kv, err = _maxctx_probe(mesh, kind, palu_dir, ctx)
        probes.append([ctx, success])
        if success:
            ok_ctx, ok_kv = ctx, kv
        else:
            fail_ctx, fail_err = ctx, err
            mesh.reset()
            break
    else:
        result = {"max_context": ok_ctx, "batch": BATCH, "limit": "model", "failed_at": None,
                  "error": None, "phase": None, "exc": None,
                  "kv_bytes_per_token_per_layer_per_chip": ok_kv, "probes": probes}
        if BATCH == 32:
            result["max_context_batch32"] = ok_ctx
        return result

    if ok_ctx is not None:  # bisect (ok_ctx, fail_ctx) on multiples of 1024; skip if 1024 itself failed
        lo, hi = ok_ctx, fail_ctx
        while hi - lo > 1024:
            mid = lo + round((hi - lo) / 2 / 1024) * 1024
            mid = min(max(mid, lo + 1024), hi - 1024)
            success, kv, err = _maxctx_probe(mesh, kind, palu_dir, mid)
            probes.append([mid, success])
            if success:
                lo, ok_kv = mid, kv
            else:
                hi, fail_ctx, fail_err = mid, mid, err
                mesh.reset()
        ok_ctx = lo

    result = {"max_context": ok_ctx, "batch": BATCH,
              "limit": "model" if not _is_oom(fail_err) else "oom",
              "failed_at": fail_ctx,
              "error": fail_err["error"] if fail_err else None,
              "phase": fail_err["phase"] if fail_err else None,
              "exc": fail_err["exc"] if fail_err else None,
              "kv_bytes_per_token_per_layer_per_chip": ok_kv, "probes": probes}
    if BATCH == 32:
        result["max_context_batch32"] = ok_ctx
    return result


def _out_name(what, row, batch):
    """``bench_<what>_<row>.json`` at the default batch (32, unchanged stage-1/2 naming), else
    ``bench_<what>_<row>_b<batch>.json``."""
    suffix = "" if batch == 32 else f"_b{batch}"
    return f"bench_{what}_{row}{suffix}.json"


def main():
    global BATCH
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--kind", choices=["stock", "palu"], required=True)
    p.add_argument("--palu-dir", default=None)
    p.add_argument("--what", choices=["decode", "attention", "prefill", "maxctx"], required=True)
    p.add_argument("--fused", choices=["off", "p1", "p2"], default="off",
                   help="stage-2 fused K-reconstruction decode kernel: p1 = Phase 1 (streamed "
                        "cos/sin), p2 = Phase 2 (chunk-relative rotation). Palu rows only.")
    p.add_argument("--fused-k-chunk", type=int, default=None,
                   help="override the fused op's K chunk (default: the op's own choice; Phase 1 is "
                        "pinned to 128 by PaluAttention).")
    p.add_argument("--group-heads", choices=["auto", "1", "2", "4"], default="auto",
                   help="stage-2c group-shared mode: KV heads per core in the fused op. auto = the "
                        "largest that fits L1 per layer; 1 reproduces the stage 2b per-head kernel. "
                        "Grouped (G-LRD) Palu rows only.")
    p.add_argument("--rot-mode", choices=["legacy", "pack", "dst"], default="dst",
                   help="stage-2e K rebuild mode in the fused op (default: dst, the shipped variant; see "
                        "docs/superpowers/specs/2026-09-14-palu-tt-stage2e-design.md).")
    p.add_argument("--table-mcast", choices=["off", "on"], default="on",
                   help="stage-2e: multicast the Phase 2 angle table down each grid column, instead of "
                        "reading it per core (default: on, the shipped variant; Phase 2 only).")
    p.add_argument("--latent-dtype", choices=["bfp8", "bfp4"], default="bfp8",
                   help="tile format of the Palu latent K/V caches (default bfp8, the model's KV cache format).")
    p.add_argument("--k-latent-dtype", choices=["bfp8", "bfp4"], default=None,
                   help="stage 2f: tile format of the Palu *K* latent cache alone (default: follow --latent-dtype).")
    p.add_argument("--v-latent-dtype", choices=["bfp8", "bfp4"], default=None,
                   help="stage 2f: tile format of the Palu *V* latent cache alone (default: follow --latent-dtype).")
    p.add_argument("--kv-dtype", choices=["bfp8", "bfp4"], default="bfp8",
                   help="stage 2f: the model's KV_CACHE precision. On --kind stock this is the whole KV "
                        "cache (the format-only competitor); on --kind palu it is what the naive decode's "
                        "reconstructed K/V temporaries are cast to (and the prefill SDPA's precision on "
                        "reconstructed sides), not the latent caches themselves.")
    p.add_argument("--latent-scale", choices=["balanced", "unit"], default="balanced",
                   help="stage 2f: 'unit' moves the singular values into the up-projection so every latent "
                        "component has unit variance (what bfp4 caches want); 'balanced' is Palu's split.")
    p.add_argument("--batch", type=int, default=32,
                   help="batch size for decode, attention and maxctx (default: 32). "
                        "tt_transformers accepts any value <= 32. prefill is single-user by "
                        "definition and ignores this flag (always batch 1); the flag still "
                        "selects the output file name in every case.")
    p.add_argument("--stop-after-oom", action="store_true", default=False,
                   help="decode only: stop the context loop right after the first error cell "
                        "instead of trying every remaining (larger) context.")
    a = p.parse_args()
    if a.kind == "stock" and a.fused != "off":
        p.error("--fused is a Palu-path flag; --kind stock has no K reconstruction to fuse")
    if a.kind == "stock" and (a.latent_dtype != "bfp8" or a.k_latent_dtype is not None or a.v_latent_dtype is not None):
        p.error("--latent-dtype/--k-latent-dtype/--v-latent-dtype are Palu-path flags; --kind stock has "
                "no latent cache (use --kv-dtype for the stock KV cache format)")
    if not 1 <= a.batch <= 32:
        p.error("--batch must be between 1 and 32 (tt_transformers's cap)")
    k_latent_dtype, v_latent_dtype = resolve_latent_dtypes(a.latent_dtype, a.k_latent_dtype, a.v_latent_dtype)
    FUSED.update(use_fused_k=a.fused != "off", cs_recurrence=a.fused == "p2", fused_k_chunk=a.fused_k_chunk,
                 group_heads=None if a.group_heads == "auto" else int(a.group_heads),
                 rot_mode=a.rot_mode, table_mcast=a.table_mcast == "on", latent_dtype=LATENT_DTYPES[a.latent_dtype],
                 k_latent_dtype=k_latent_dtype, v_latent_dtype=v_latent_dtype,
                 kv_cache_dtype=LATENT_DTYPES[a.kv_dtype], latent_scale=a.latent_scale)
    BATCH = a.batch
    RAW.mkdir(parents=True, exist_ok=True)
    mesh = Mesh()
    try:
        if a.what == "decode":
            data = bench_decode(mesh, a.kind, a.palu_dir, stop_after_oom=a.stop_after_oom)
        else:
            data = {"attention": bench_attention, "prefill": bench_prefill,
                    "maxctx": bench_maxctx}[a.what](mesh, a.kind, a.palu_dir)
    finally:
        mesh.close()
    # stage 2f (Minor 5): --kind stock has no latent cache, so the three Palu-only format fields are
    # null on a stock row rather than echoing flags that (by the check above) can only be at their
    # defaults -- the same "null, not absent" rule R1 already applies to eval_ppl's --cpu rows.
    if a.kind == "stock":
        latent_fields = {"latent_dtype": None, "k_latent_dtype": None, "v_latent_dtype": None}
    else:
        # the *resolved* per-cache formats (--k/--v-latent-dtype falling back to --latent-dtype), so a
        # row says what each cache actually held rather than which flags were typed
        k_name, v_name = resolve_latent_dtype_names(a.latent_dtype, a.k_latent_dtype, a.v_latent_dtype)
        latent_fields = {"latent_dtype": a.latent_dtype, "k_latent_dtype": k_name, "v_latent_dtype": v_name}
    # the *effective* stage 2e switches (not the flags): --fused off has no K rebuild mode at all and
    # the multicast is folded away outside Phase 2, so a row says what the op ran with
    rot_mode, table_mcast = resolve_rot_switches(a.fused, a.rot_mode, a.table_mcast)
    r = {"row": a.row, "kind": a.kind, "palu_dir": a.palu_dir, "what": a.what,
         "fused": a.fused, "fused_k_chunk": a.fused_k_chunk, "group_heads": a.group_heads,
         "group_heads_per_layer": list(GROUP_HEADS), "batch": a.batch,
         "rot_mode": rot_mode, "table_mcast": table_mcast, **latent_fields, "kv_dtype": a.kv_dtype,
         "latent_scale": a.latent_scale}
    if a.what in ("decode", "attention"):  # prefill/maxctx use their own fixed lengths, not CONTEXTS
        r["contexts"] = list(CONTEXTS)
    r["data"] = data
    (RAW / _out_name(a.what, a.row, a.batch)).write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
