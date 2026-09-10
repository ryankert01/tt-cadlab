"""Render the stage 1 results tables from ``docs/results/raw/*.json``.

Every number in ``docs/results/stage1.md`` above the hand-written sections comes from this
module reading the JSON the harnesses wrote -- nothing is transcribed by hand. The renderer is
deliberately tolerant of the real schema:

* ``json.dumps`` stringifies the int context keys, so cells are looked up as ``"1024"``;
* a context whose model did not allocate is an ``{"error", "phase", "exc"}`` cell, not a timing;
* a row whose files do not exist yet renders as dashes -- never as invented ranks.

Ranks come from the compressed dump's ``config.json``: our own ``compress.py`` dumps carry a
``palu_tt.rank_k/rank_v`` block, while a dump produced by Palu's own recipe carries only
``head_wise_ranks``, so for those the per-layer ranks are derived with ``convert.load_palu_dir``
(the same reader the device path uses) and rendered as a range plus a mean.

Usage: ``python -m palu_tt.report`` (writes the file and prints it). ``main(raw=..., out=...)``
takes both paths as parameters so it can be rendered from a fixture directory in tests.

Everything from ``MARKER`` onwards in an existing output file is hand-written prose and is
carried over verbatim, so re-rendering after a re-measured row does not destroy the notes.
"""
import argparse
import json
import subprocess
from pathlib import Path

RAW = Path("docs/results/raw")
OUT = Path("docs/results/stage1.md")

# row -> human label. Order is the render order of every table.
ROWS = {
    "baseline": "stock `tt_transformers` attention",
    "identity": "Palu path, identity factorisation (r = 128)",
    "svd_r96": "plain per-head SVD, r_k = r_v = 96",
    "svd_r64": "plain per-head SVD, r_k = r_v = 64",
    "svd_rk128_rv64": "plain per-head SVD, V only (K left full rank)",
    "palu_full_50": "Palu full recipe (Fisher + whitening), ratio 0.5",
    "palu_full_70": "Palu full recipe (Fisher + whitening), ratio 0.7",
}
CONTEXTS = ("1024", "4096", "8192")
PREFILLS = ("2048", "8192")
HEAD_DIM = 128
LAYERS = 32
CHIPS = 2
MARKER = "<!-- everything below this line is hand-written and preserved by palu_tt.report -->"


def _load(raw, name):
    p = Path(raw) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def _cell(bench, key):
    """The per-context/per-length dict, or None when the file or the cell is missing."""
    if not bench:
        return None
    return (bench.get("data") or {}).get(key)


def _timing(cell):
    """``"p50 / min"`` in ms, ``"OOM"``/``"err"`` for an error cell, ``"-"`` when absent."""
    if cell is None:
        return "-"
    if "error" in cell:
        return "OOM" if "Out of Memory" in str(cell["error"]) else "err"
    if "p50_ms" not in cell:
        return "-"
    return f"{_fmt(cell.get('p50_ms'), 1)} / {_fmt(cell.get('min_ms'), 1)}"


def _kv_bytes(row_files):
    """KV bytes/token/layer/chip, from whichever bench file for this row recorded it."""
    for b in row_files:
        if not b:
            continue
        data = b.get("data") or {}
        cells = list(data.values()) if isinstance(data, dict) else []
        for c in [data] + cells:
            if isinstance(c, dict) and c.get("kv_bytes_per_token_per_layer_per_chip") is not None:
                return c["kv_bytes_per_token_per_layer_per_chip"]
    return None


def _palu_dir(files):
    return next((f["palu_dir"] for f in files if f and f.get("palu_dir")), None)


def _per_layer_ranks(palu_dir, cfg):
    """(r_k list, r_v list) for a dump that has no ``palu_tt`` block, via the device-side reader.

    ``convert.load_palu_dir`` is the same function ``build_palu_model`` uses, so the numbers
    reported here are exactly the per-layer ranks the device allocates its caches for (per-head
    ranks zero-padded up to the layer max rounded to a multiple of 32).
    """
    from palu_tt.convert import load_palu_dir  # imports tt-metal; keep the module itself light

    ranks = cfg["head_wise_ranks"]
    n_layers = sum(1 for k in ranks if k.endswith(".k_proj"))
    n_kv_heads = len(next(iter(ranks.values())))
    head_dim = int(cfg.get("head_dim", HEAD_DIM))
    lr = load_palu_dir(Path(palu_dir), n_layers, n_kv_heads, head_dim)
    return [lr[i].r_k for i in range(n_layers)], [lr[i].r_v for i in range(n_layers)]


def _fmt_rank(vals):
    """``"64"`` when every layer agrees, else ``"32-64 (mean 33)"``."""
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return str(lo)
    return f"{lo}–{hi} (mean {round(sum(vals) / len(vals))})"


def _ranks(files):
    """(r_k, r_v) as rendered strings.

    ``"-"`` for a row with no files at all (nothing is known about it) and for a row that names a
    compressed dump whose config cannot be read -- an unreadable dump must never be reported as
    full rank. Stock/identity rows (no ``palu_dir``) are full rank by construction.
    """
    if not any(files):
        return "-", "-"
    palu_dir = _palu_dir(files)
    if not palu_dir:
        return HEAD_DIM, HEAD_DIM
    cfg_path = Path(palu_dir) / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return "-", "-"
    p = cfg.get("palu_tt") or {}
    if p.get("rank_k") is not None and p.get("rank_v") is not None:
        return p["rank_k"], p["rank_v"]
    if "head_wise_ranks" not in cfg:
        return "-", "-"
    rk, rv = _per_layer_ranks(palu_dir, cfg)
    return _fmt_rank(rk), _fmt_rank(rv)


def _kv_model_mean(kv_layer0, files):
    """KV bytes/token/layer/chip averaged over the 32 layers.

    ``bench`` reports the figure for **layer 0** (``model.layers[0].attention``). That is the whole
    model for every uniform-rank row, but the calibrated dumps have per-layer ranks, so the layer-0
    cell understates or overstates the model. Rescale the measured figure by the ratio of the mean
    per-layer rank sum to layer 0's -- this keeps the number anchored to the measurement (dtype and
    head count included) instead of re-deriving bytes from scratch.
    """
    if kv_layer0 is None:
        return None
    palu_dir = _palu_dir(files)
    if not palu_dir:
        return kv_layer0
    cfg_path = Path(palu_dir) / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return None
    if (cfg.get("palu_tt") or {}).get("rank_k") is not None or "head_wise_ranks" not in cfg:
        return kv_layer0  # uniform ranks: layer 0 is the model
    rk, rv = _per_layer_ranks(palu_dir, cfg)
    per_layer = [a + b for a, b in zip(rk, rv)]
    return kv_layer0 * (sum(per_layer) / len(per_layer)) / per_layer[0]


def _windows(files):
    """The window counts present in a set of perplexity JSONs, as a header string."""
    ws = sorted({f["windows"] for f in files if f and f.get("windows") is not None})
    return "/".join(str(w) for w in ws) if ws else "?"


def _commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _row_data(raw, row):
    ppl = _load(raw, f"ppl_{row}")
    cpu = _load(raw, f"ppl_cpu_{row}")
    dec = _load(raw, f"bench_decode_{row}")
    att = _load(raw, f"bench_attention_{row}")
    pre = _load(raw, f"bench_prefill_{row}")
    mx = _load(raw, f"bench_maxctx_{row}")
    return ppl, cpu, dec, att, pre, mx


def main(raw=RAW, out=OUT):
    raw = Path(raw)
    loaded = {row: _row_data(raw, row) for row in ROWS}
    base = loaded.get("baseline")
    kv_base = _kv_bytes(base[2:]) if base else None
    dev_windows = _windows([v[0] for v in loaded.values()])
    cpu_windows = _windows([v[1] for v in loaded.values()])

    lines = [
        "# Stage 1 results: Llama-3.1-8B-Instruct vs Palu(Llama-3.1-8B-Instruct) on a Wormhole n300",
        "",
        f"Generated by `python -m palu_tt.report` from `docs/results/raw/*.json` at commit "
        f"`{_commit()}`; tt-metal v0.78.0, TP=2 (one n300, two chips), bfp8 weights and KV cache, "
        "non-paged, untraced decode, `use_qk_fused=False` for both models.",
        "",
        "## Quality and memory",
        "",
        f"| row | recipe | r_k | r_v | ppl (device, {dev_windows} windows) | "
        f"ppl (CPU fp32, {cpu_windows} windows) | KV B/token/layer/chip | "
        "KV B/token (32 layers, both chips) | KV reduction | max ctx @ batch 32 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row, label in ROWS.items():
        ppl, cpu, dec, att, pre, mx = loaded[row]
        rk, rv = _ranks([ppl, dec, att, pre, mx])
        kv = _kv_model_mean(_kv_bytes([dec, att, mx]), [ppl, dec, att, pre, mx])
        kv_total = "-" if kv is None else f"{round(kv * LAYERS * CHIPS)}"
        ratio = "-" if (kv is None or not kv_base) else f"{kv_base / kv:.2f}x"
        maxctx = (mx.get("data") or {}).get("max_context_batch32") if mx else None
        lines.append(
            f"| {row} | {label} | {rk} | {rv} | {_fmt(ppl and ppl.get('ppl'), 3)} | "
            f"{_fmt(cpu and cpu.get('ppl'), 3)} | {'-' if kv is None else round(kv)} | "
            f"{kv_total} | {ratio} | {maxctx or '-'} |"
        )

    lines += [
        "",
        "KV bytes are per token, per layer, per chip, **averaged over the 32 layers**. `bench` reports "
        "the figure for layer 0; for the uniform-rank rows that is the whole model, and for the two "
        "calibrated rows (per-layer ranks) it is rescaled by the mean rank sum from `head_wise_ranks`. "
        "`KV reduction` is stock bytes / row bytes.",
        "",
        "## Decode and attention latency, batch 32",
        "",
        "Measured on an idle host (device-idle gate passed, no other job of ours running); the "
        "contended numbers from the first matrix run were discarded. Decode is untraced and therefore "
        "partly host-dispatch-bound, so `p50` and `min` are both reported; `min` is the cleaner "
        "signal and every ratio quoted below the marker uses it.",
        "",
        "| row | context | decode step p50 / min ms | attention-only p50 / min ms |",
        "|---|---|---|---|",
    ]
    for row in ROWS:
        _, _, dec, att, _, _ = loaded[row]
        for ctx in CONTEXTS:
            lines.append(f"| {row} | {ctx} | {_timing(_cell(dec, ctx))} | {_timing(_cell(att, ctx))} |")

    lines += [
        "",
        "## Prefill (TTFT), batch 1",
        "",
        "| row | " + " | ".join(f"{L} tokens p50 / min ms" for L in PREFILLS) + " |",
        "|---|" + "---|" * len(PREFILLS),
    ]
    for row in ROWS:
        _, _, _, _, pre, _ = loaded[row]
        lines.append(f"| {row} | " + " | ".join(_timing(_cell(pre, L)) for L in PREFILLS) + " |")

    s = _load(raw, "samples")
    if s:
        lines += ["", "## Generation samples (greedy, 48 tokens)", ""]
        for prompt in next(iter(s.values())):
            lines.append(f"**{prompt}**")
            lines.append("")
            for row, d in s.items():
                text = " ".join(str(d.get(prompt, "")).split())
                lines.append(f"- `{row}`: {text}")
            lines.append("")

    lines += ["", MARKER, ""]
    text = "\n".join(lines).rstrip("\n") + "\n"

    if out is not None:
        out = Path(out)
        if out.exists() and MARKER in out.read_text():
            kept = out.read_text().split(MARKER, 1)[1].lstrip("\n")
            if kept.strip():
                text = text + "\n" + kept.rstrip("\n") + "\n"
        out.write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render docs/results/stage1.md from the raw JSONs.")
    ap.add_argument("--raw", default=RAW, type=Path)
    ap.add_argument("--out", default=OUT, type=Path)
    a = ap.parse_args()
    print(main(raw=a.raw, out=a.out))
