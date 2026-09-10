"""Render the stage 1 results tables from ``docs/results/raw/*.json``.

Every number in ``docs/results/stage1.md`` above the hand-written sections comes from this
module reading the JSON the harnesses wrote -- nothing is transcribed by hand. The renderer is
deliberately tolerant of the real schema:

* ``json.dumps`` stringifies the int context keys, so cells are looked up as ``"1024"``;
* a context whose model did not allocate is an ``{"error", "phase", "exc"}`` cell, not a timing;
* a row whose files do not exist yet renders as dashes -- never as invented ranks, and a raw
  directory with *no* row files at all is refused outright rather than blanking the tables.

Ranks come from the compressed dump's ``config.json``: our own ``compress.py`` dumps carry a
``palu_tt.rank_k/rank_v`` block, while a dump produced by Palu's own recipe carries only
``head_wise_ranks``, so for those the per-layer ranks are derived with ``convert.load_palu_dir``
(the same reader the device path uses) and rendered as a range plus a mean.

Stage 2 (``--stage 2``) renders the same tables for the fused-kernel rows into
``docs/results/stage2.md``: the ``_fk1``/``_fk2`` rows carry their own decode/attention timings and a
``decode_check_*`` fidelity JSON, and inherit perplexity, prefill, KV bytes and max context from
their base row -- those are prefill/allocation properties and are identical by construction, so they
are not re-measured. Both stages go through the same ``main``; ``rows`` selects which.

Usage: ``python -m palu_tt.report [--stage 2]`` (writes the file and prints it).
``main(raw=..., out=..., rows=...)`` takes the paths and the row set as parameters so either stage
can be rendered from a fixture directory in tests.

Everything from ``MARKER`` onwards in an existing output file is hand-written prose and is
carried over verbatim, so re-rendering after a re-measured row does not destroy the notes.
"""
import argparse
import json
import subprocess
from pathlib import Path

RAW = Path("docs/results/raw")
OUT = Path("docs/results/stage1.md")
OUT2 = Path("docs/results/stage2.md")

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
# Stage 2. A ``_fk1``/``_fk2`` row is the row above it re-run with the fused decode kernel; the two
# leading spaces in its label indent it under its base row in the raw markdown.
ROWS2 = {
    "baseline": "stock `tt_transformers` attention",
    "svd_r64": "plain per-head SVD, r_k = r_v = 64 (naive K reconstruction)",
    "svd_r64_fk1": "  same, fused kernel Phase 1 (cos/sin streamed)",
    "svd_r64_fk2": "  same, fused kernel Phase 2 (chunk-relative rotation)",
    "svd_r96": "plain per-head SVD, r_k = r_v = 96 (naive)",
    "svd_r96_fk2": "  same, fused Phase 2",
    "palu_full_50": "Palu full recipe, ratio 0.5 (naive)",
    "palu_full_50_fk1": "  same, fused Phase 1",
    "palu_full_50_fk2": "  same, fused Phase 2",
    "palu_full_70": "Palu full recipe, ratio 0.7 (naive)",
    "palu_full_70_fk1": "  same, fused Phase 1",
    "palu_full_70_fk2": "  same, fused Phase 2",
}
CONTEXTS = ("1024", "4096", "8192")
PREFILLS = ("2048", "8192")
# Stage 2 batch sweep (Task 7): decode step time at these batch sizes, for these rows. 32 is the
# stage-1/2 default batch and therefore has no ``_b<B>`` suffix on its JSON file.
SWEEP_BATCHES = (1, 2, 4, 8, 16, 32)
SWEEP_ROWS = ("baseline", "palu_full_50", "palu_full_50_fk2", "palu_full_70_fk2")
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


def _base_row(row):
    """``'palu_full_50_fk2'`` -> ``'palu_full_50'``; a stage-1 row maps to itself."""
    return row.rsplit("_fk", 1)[0] if "_fk" in row else row


def _row_data(raw, row):
    """``(ppl, cpu, dec, att, pre, mx, chk)`` for one row.

    A fused row (``_fk1``/``_fk2``) has its own decode/attention timings and its own decode-fidelity
    JSON; perplexity, prefill, KV bytes and max context are prefill/allocation properties that the
    fused decode kernel cannot change, so they are not re-measured and are read from the base row.
    The row's own file still wins where it exists, so a deliberately re-measured fused perplexity
    would be picked up.
    """
    base = _base_row(row)
    ppl = _load(raw, f"ppl_{row}") or _load(raw, f"ppl_{base}")   # prefill measurement: same for fused rows
    cpu = _load(raw, f"ppl_cpu_{row}") or _load(raw, f"ppl_cpu_{base}")
    dec = _load(raw, f"bench_decode_{row}")
    att = _load(raw, f"bench_attention_{row}")
    pre = _load(raw, f"bench_prefill_{row}") or _load(raw, f"bench_prefill_{base}")
    mx = _load(raw, f"bench_maxctx_{row}") or _load(raw, f"bench_maxctx_{base}")
    chk = _load(raw, f"decode_check_{row}")
    return ppl, cpu, dec, att, pre, mx, chk


def _vs_stock(stock_dec, dec, ctx):
    """stock ``min_ms`` / row ``min_ms`` at one context, or ``"-"`` when either cell is missing or errored."""
    a, b = _cell(stock_dec, ctx), _cell(dec, ctx)
    if not a or not b or "min_ms" not in a or "min_ms" not in b:
        return "-"
    return f"{a['min_ms'] / b['min_ms']:.2f}x"


def _fidelity(chk):
    """``("min / mean", "agree")`` from a ``decode_check_*`` JSON; dashes when the row has none."""
    if not chk:
        return "-", "-"
    return f"{chk['pcc_min']:.4f} / {chk['pcc_mean']:.4f}", f"{chk['argmax_agree']:.3f}"


def _sweep_filename(row, batch):
    """``bench_decode_<row>.json`` at batch 32 (the default, unsuffixed file), else ``_b<batch>``."""
    return f"bench_decode_{row}" if batch == 32 else f"bench_decode_{row}_b{batch}"


def _sweep_cell(raw, row, batch, ctx):
    """Decode ``min_ms`` for one (row, batch, ctx) sweep cell.

    ``None`` when the file does not exist, the context is missing, or the cell errored (an errored
    cell has no ``min_ms``) -- the caller distinguishes "missing" from "errored" itself, by looking
    at the same raw cell for an ``"error"`` key.
    """
    cell = _cell(_load(raw, _sweep_filename(row, batch)), str(ctx))
    return cell.get("min_ms") if cell and "min_ms" in cell else None


def _sweep_display(raw, row, batch, ctx):
    """``"min ms (tok/s)"``, ``"OOM"`` for an error cell, ``"-"`` when the cell is simply missing."""
    cell = _cell(_load(raw, _sweep_filename(row, batch)), str(ctx))
    if cell and "error" in cell:
        return "OOM"
    min_ms = _sweep_cell(raw, row, batch, ctx)
    if min_ms is None:
        return "-"
    return f"{_fmt(min_ms, 1)} ({round(batch * 1000 / min_ms)})"


def _steps(files):
    """The decode-check step counts present, as a header string (mirrors ``_windows``)."""
    ss = sorted({f["steps"] for f in files if f and f.get("steps") is not None})
    return "/".join(str(x) for x in ss) if ss else "?"


def main(raw=RAW, out=OUT, rows=ROWS):
    """Render one stage's tables. ``rows`` is ``ROWS`` (stage 1) or ``ROWS2`` (stage 2).

    Stage 2 adds the two decode-fidelity columns to the quality table and a ``vs stock`` column to
    the latency table; everything else -- the loaders, the KV arithmetic, the error cells and the
    hand-written tail -- is shared, so the two documents cannot drift apart.
    """
    raw = Path(raw)
    stage2 = rows is ROWS2
    loaded = {row: _row_data(raw, row) for row in rows}
    # A raw directory with nothing in it renders every row as dashes -- and would then overwrite the
    # committed tables with that. That is exactly what a fresh clone or a mistyped --raw looks like,
    # so refuse instead of writing. (One loaded file anywhere is enough: a partial re-measurement
    # legitimately renders the rest as dashes.)
    if not any(f is not None for files in loaded.values() for f in files):
        raise RuntimeError(
            f"no results found in {raw}: not one of the {len(rows)} rows has a ppl_*/bench_* JSON, so "
            f"rendering would replace the tables in {out} with dashes. Run the harnesses "
            "(python -m palu_tt.eval_ppl, python -m palu_tt.bench) or restore docs/results/raw/*.json first."
        )
    base = loaded.get("baseline")
    kv_base = _kv_bytes(base[2:6]) if base else None
    stock_dec = base[2] if base else None
    dev_windows = _windows([v[0] for v in loaded.values()])
    cpu_windows = _windows([v[1] for v in loaded.values()])
    steps = _steps([v[6] for v in loaded.values()])

    if stage2:
        lines = [
            "# Stage 2 results: fused K-reconstruction decode kernel on a Wormhole n300",
            "",
            f"Generated by `python -m palu_tt.report --stage 2` from `docs/results/raw/*.json` at commit "
            f"`{_commit()}`; tt-metal v0.78.0 plus the `palu-sdpa` submodule branch, TP=2 (one n300, two "
            "chips), bfp8 weights and latent caches, non-paged, untraced decode, `use_qk_fused=False` "
            "for every row.",
            "",
            "A `_fk1`/`_fk2` row is the row above it re-run with `--fused p1` / `--fused p2`. Perplexity "
            "here is measured **through prefill windows**, which reconstruct K and V in flight and never "
            "touch the decode kernel, so a fused row's perplexity and prefill time are its base row's by "
            "construction and are not re-measured; KV bytes and max context are inherited too unless the "
            "row has its own `bench_maxctx_*` file, in which case that measurement wins. The decode path is "
            "checked instead by `palu_tt.decode_check` (naive vs fused, teacher-forced), whose two "
            "columns are in the quality table.",
            "",
            "## Quality and memory",
            "",
            f"| row | recipe | r_k | r_v | ppl (device, {dev_windows} windows) | "
            f"ppl (CPU fp32, {cpu_windows} windows) | decode PCC (min / mean, {steps} steps) | "
            "argmax agree | KV B/token/layer/chip | KV B/token (32 layers, both chips) | "
            "KV reduction | max ctx @ batch 32 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
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
    for row, label in rows.items():
        ppl, cpu, dec, att, pre, mx, chk = loaded[row]
        rk, rv = _ranks([ppl, dec, att, pre, mx])
        kv = _kv_model_mean(_kv_bytes([dec, att, mx]), [ppl, dec, att, pre, mx])
        kv_total = "-" if kv is None else f"{round(kv * LAYERS * CHIPS)}"
        ratio = "-" if (kv is None or not kv_base) else f"{kv_base / kv:.2f}x"
        maxctx = (mx.get("data") or {}).get("max_context_batch32") if mx else None
        fid = ""
        if stage2:
            pcc, agree = _fidelity(chk)
            fid = f"{pcc} | {agree} | "
        lines.append(
            f"| {row} | {label} | {rk} | {rv} | {_fmt(ppl and ppl.get('ppl'), 3)} | "
            f"{_fmt(cpu and cpu.get('ppl'), 3)} | {fid}{'-' if kv is None else round(kv)} | "
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
    ]
    lines += [
        ("Measured on an idle host (device-idle gate passed, no other job of ours running). Decode is "
         "untraced and therefore partly host-dispatch-bound, so `p50` and `min` are both reported; "
         "`min` is the cleaner signal and every ratio quoted below the marker uses it. The stage-1 "
         "rows here are the stage-1 measurements, unchanged.") if stage2 else
        ("Measured on an idle host (device-idle gate passed, no other job of ours running); the "
         "contended numbers from the first matrix run were discarded. Decode is untraced and therefore "
         "partly host-dispatch-bound, so `p50` and `min` are both reported; `min` is the cleaner "
         "signal and every ratio quoted below the marker uses it."),
        "",
    ]
    if stage2:
        lines += [
            "`vs stock` is the stock row's decode `min_ms` divided by this row's at the same context "
            "(> 1.00x is faster than stock); it is a dash at 8192, where stock does not allocate.",
            "",
            "| row | context | decode step p50 / min ms | attention-only p50 / min ms | vs stock |",
            "|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| row | context | decode step p50 / min ms | attention-only p50 / min ms |",
            "|---|---|---|---|",
        ]
    for row in rows:
        dec, att = loaded[row][2], loaded[row][3]
        for ctx in CONTEXTS:
            cells = f"| {row} | {ctx} | {_timing(_cell(dec, ctx))} | {_timing(_cell(att, ctx))} |"
            if stage2:
                cells += f" {_vs_stock(stock_dec, dec, ctx)} |"
            lines.append(cells)

    if stage2:
        lines += ["", "## Decode step by batch size", ""]
        for ctx in CONTEXTS:
            lines += [
                f"### Context {ctx}",
                "",
                "| row | " + " | ".join(f"b={b}" for b in SWEEP_BATCHES) + " |",
                "|---|" + "---|" * len(SWEEP_BATCHES),
            ]
            for row in SWEEP_ROWS:
                cells = " | ".join(_sweep_display(raw, row, b, ctx) for b in SWEEP_BATCHES)
                lines.append(f"| {row} | {cells} |")
            lines.append("")
        lines.append(
            "Tokens per second is the whole batch's throughput at that step time; per-user "
            "latency is the step time itself."
        )

    lines += [
        "",
        "## Prefill (TTFT), batch 1",
        "",
        "| row | " + " | ".join(f"{L} tokens p50 / min ms" for L in PREFILLS) + " |",
        "|---|" + "---|" * len(PREFILLS),
    ]
    for row in rows:
        pre = loaded[row][4]
        lines.append(f"| {row} | " + " | ".join(_timing(_cell(pre, L)) for L in PREFILLS) + " |")

    smp = _load(raw, "samples") if not stage2 else None   # the samples are a stage-1 artefact
    if smp:
        lines += ["", "## Generation samples (greedy, 48 tokens)", ""]
        for prompt in next(iter(smp.values())):
            lines.append(f"**{prompt}**")
            lines.append("")
            for row, d in smp.items():
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
    ap = argparse.ArgumentParser(description="Render docs/results/stage{1,2}.md from the raw JSONs.")
    ap.add_argument("--raw", default=RAW, type=Path)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--out", default=None, type=Path, help="default: stage1.md / stage2.md")
    a = ap.parse_args()
    print(main(raw=a.raw, out=a.out or (OUT2 if a.stage == 2 else OUT),
               rows=ROWS2 if a.stage == 2 else ROWS))
