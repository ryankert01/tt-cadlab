"""Aggregate a tracy ops report (one profiled decode run) into per-op-category device time (stage 2d).

    python -m palu_tt.profile_report --name baseline_1024 [--name ...] [--profiler-dir third_party/tt-metal/generated/profiler]

For every named run it reads the newest ``ops_perf_results_*.csv`` under ``<profiler-dir>/<name>/reports``,
keeps the ops between the ``start`` / ``stop`` signposts, takes device 0 (the two n300 chips run the same
program; chip 1's numbers are reported as a check), divides by the number of profiled steps (recorded
by ``palu_tt.profile_decode`` as ``steps`` in the run's ``meta.json``, default 3), classifies every op
into a category, and writes ``docs/results/raw/profile_<name>.json``:

    {"name", "steps", "device_ms_per_step", "device_ms_per_step_chip1", "ops_per_step",
     "categories": {cat: {"ms": ..., "count": ...}}, "ops": [{"op", "category", "ms", "count", "cores",
     "in0", "in1", "out"} ...]}   # ops = distinct (op code, shapes) signatures, summed over the step

Categories (what a decode step is made of):
  attention      the decode attention op itself: stock SDPA decode, the MLA op used by the naive Palu path,
                 or the fused palu_sdpa_decode
  k_recon        the naive Palu path's host-side K/V reconstruction: latent @ U matmuls, the prefill-mode
                 RoPE over the rebuilt K, typecasts and slices of the reconstructed tensors
  qkv            the fused QKV projection matmul
  wo             the output projection matmul
  mlp            w1/w3/w2 matmuls
  lm_head        the vocabulary projection
  ccl            reduce-scatter / all-gather / all-reduce across the two chips
  norm           RMSNorm (LayerNorm op)
  cache          KV cache update (paged_update_cache)
  heads          create/concat heads, rotary embedding of q/k in decode mode
  elementwise    binary/unary ops (residual adds, SiLU, mul), embeddings
  layout         sharded<->interleaved, reshard, transpose, typecast/slice outside k_recon
  other          anything else
"""
import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

RAW = Path("docs/results/raw")
DUR = "DEVICE KERNEL DURATION [ns]"


def _shape(row, pfx):
    dims = []
    for d in ("W", "Z", "Y", "X"):
        v = row.get(f"{pfx}_{d}_PAD[LOGICAL]", "-")
        if v == "-" or (not isinstance(v, str) and pd.isna(v)):
            continue
        v = str(v)                              # "4096[4096]" = padded[logical]; keep the logical extent
        logical = v[v.index("[") + 1:v.index("]")] if "[" in v else v
        dims.append(int(float(logical)))
    return tuple(dims)


def classify(op, in0, in1, out, attrs):
    """Category for one op given its code, input/output shapes and the ATTRIBUTES string."""
    if op in ("SdpaDecodeDeviceOperation", "PaluSdpaDecodeDeviceOperation", "FlashMultiLatentAttentionDecodeDeviceOperation"):
        return "attention"
    if op == "MatmulDeviceOperation":
        k = in1[-2] if len(in1) >= 2 else 0
        n = in1[-1] if in1 else 0
        if n >= 20000:
            return "lm_head"
        if k == 4096 and n == 3072:
            return "qkv"                      # 16 q + 4 k + 4 v heads x 128 per chip (stock and Palu alike)
        if k == 4096 and n > 3072 and n <= 8192:
            return "mlp"                      # w1 / w3 (14336 / 2 = 7168)
        if k == 7168 and n == 4096:
            return "mlp"                      # w2
        if k == 2048 and n == 4096:
            return "wo"
        if n == 128 and k <= 512 and len(in0) == 4 and in0[2] >= 32:
            return "k_recon"                  # [B, nkv, S, r] @ [r, 128]: the naive path's K / V rebuild
        return "other"
    if op in ("ReduceScatterMinimalAsyncDeviceOperation", "AllGatherAsyncDeviceOperation", "AllReduceAsyncDeviceOperation",
              "ReduceScatterAsyncDeviceOperation", "AllGatherDeviceOperation"):
        return "ccl"
    if op in ("LayerNormDeviceOperation", "RMSNormDeviceOperation", "RMSAllGatherDeviceOperation"):
        return "norm"
    if op in ("PagedUpdateCacheDeviceOperation", "UpdateCacheDeviceOperation", "FillCacheDeviceOperation"):
        return "cache"
    if op == "RotaryEmbeddingLlamaDeviceOperation":
        # decode-mode RoPE on q/k is 'heads'; the naive Palu path rotates the rebuilt K over S rows in prefill mode
        return "k_recon" if (len(in0) == 4 and in0[2] >= 64) else "heads"
    if op in ("NLPCreateQKVHeadsDecodeDeviceOperation", "NLPConcatHeadsDecodeDeviceOperation", "NLPCreateQKVHeadsDeviceOperation",
              "NLPConcatHeadsDeviceOperation"):
        return "heads"
    if op in ("BinaryNgDeviceOperation", "BinaryDeviceOperation", "UnaryDeviceOperation", "EmbeddingsDeviceOperation",
              "EltwiseBinaryDeviceOperation"):
        return "elementwise"
    if op in ("TypecastDeviceOperation", "SliceDeviceOperation", "CopyDeviceOperation", "RepeatDeviceOperation", "PadDeviceOperation"):
        # inside the naive path these run over the S-row reconstructed tensors
        return "k_recon" if (len(in0) == 4 and in0[2] >= 64) else "layout"
    if op in ("ShardedToInterleavedDeviceOperation", "InterleavedToShardedDeviceOperation", "ReshardDeviceOperation",
              "TransposeDeviceOperation", "UntilizeCodegenDeviceOperation", "TilizeDeviceOperation", "ConcatDeviceOperation",
              "ReshapeDeviceOperation", "UntilizeDeviceOperation", "FillPadDeviceOperation"):
        return "layout"
    return "other"


def latest_csv(profiler_dir, name):
    files = sorted(glob.glob(os.path.join(profiler_dir, name, "reports", "*", "ops_perf_results_*.csv")))
    if not files:
        raise FileNotFoundError(f"no ops report under {profiler_dir}/{name}")
    return files[-1]


def aggregate(csv_path, steps):
    df = pd.read_csv(csv_path)
    marks = df[df["OP TYPE"] == "signpost"]["OP CODE"]
    start = marks[marks == "start"].index[0]
    stop = marks[marks == "stop"].index[0]
    win = df.iloc[start + 1:stop]
    win = win[win[DUR] != "-"].copy()
    win[DUR] = win[DUR].astype(float)
    win["DEVICE ID"] = pd.to_numeric(win["DEVICE ID"], errors="coerce").astype(int)   # "-" on signposts makes it an object column
    per_chip = {int(d): g[DUR].sum() / 1e6 / steps for d, g in win.groupby("DEVICE ID")}
    d0 = win[win["DEVICE ID"] == min(per_chip)]
    sigs = {}
    for _, r in d0.iterrows():
        in0, in1, out = _shape(r, "INPUT_0"), _shape(r, "INPUT_1"), _shape(r, "OUTPUT_0")
        cat = classify(r["OP CODE"], in0, in1, out, str(r.get("ATTRIBUTES", "")))
        key = (r["OP CODE"], in0, in1, out, cat)
        s = sigs.setdefault(key, {"op": r["OP CODE"], "category": cat, "ms": 0.0, "count": 0,
                                  "cores": float(r["CORE COUNT"]) if r["CORE COUNT"] != "-" else None,
                                  "in0": list(in0), "in1": list(in1), "out": list(out)})
        s["ms"] += r[DUR] / 1e6 / steps
        s["count"] += 1 / steps
    ops = sorted(sigs.values(), key=lambda s: -s["ms"])
    cats = {}
    for s in ops:
        c = cats.setdefault(s["category"], {"ms": 0.0, "count": 0.0})
        c["ms"] += s["ms"]
        c["count"] += s["count"]
    chips = sorted(per_chip)
    return {"steps": steps, "device_ms_per_step": per_chip[chips[0]],
            "device_ms_per_step_chip1": per_chip[chips[1]] if len(chips) > 1 else None,
            "ops_per_step": len(d0) / steps, "categories": cats, "ops": ops, "csv": csv_path}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", action="append", required=True)
    p.add_argument("--profiler-dir", default="third_party/tt-metal/generated/profiler")
    p.add_argument("--steps", type=int, default=3)
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in a.name:
        res = aggregate(latest_csv(a.profiler_dir, name), a.steps)
        res["name"] = name
        (RAW / f"profile_{name}.json").write_text(json.dumps(res, indent=1))
        cats = sorted(res["categories"].items(), key=lambda kv: -kv[1]["ms"])
        print(f"{name}: {res['device_ms_per_step']:.2f} ms/step on chip 0 ({res['device_ms_per_step_chip1']:.2f} chip 1), "
              f"{res['ops_per_step']:.0f} ops/step")
        for c, v in cats:
            print(f"   {c:12s} {v['ms']:8.3f} ms  ({v['count']:.0f} ops)")
        unk = [s for s in res["ops"] if s["category"] == "other"]
        for s in unk[:6]:
            print(f"   other: {s['op']} in0={s['in0']} in1={s['in1']} out={s['out']} {s['ms']:.3f} ms")


if __name__ == "__main__":
    main()
