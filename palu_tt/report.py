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

Stage 2c (``--stage 2c``) adds two more tables into ``docs/results/stage2c.md``: calibration
sensitivity for G-LRD at 256 Fisher samples (against the stage 2b 32-sample rows), and the
shared-latent ("group-shared") fused kernel's decode/attention timings against the stage 2b
per-head ``_fk2`` rows, plus decode fidelity and batch-8 max context for the new rows.

Stage 2e (``--stage 2e``) renders ``docs/results/stage2e.md``: a per-variant ablation fit
(``profile_fit.fit_signatures``) for each of the ``VARIANTS2E`` rotation-kernel modes, the whole
decode step for every ``ROWS2E`` row, decode fidelity (including the bfp4-latent ``_l4`` rows,
loaded on top of ``rows`` specifically for this table), and the bfp4-latent (``L4``) perplexity/KV/
fidelity comparison. The measurement matrix can still be running when this renders, so a variant or
row with no JSON yet renders as an explicit note rather than a crash or an invented number; the
ablation table distinguishes a variant with *no* profile JSON at all (``(not measured)``) from one
with fewer than the two distinct contexts a fit needs (``(N profile JSON(s) at M distinct
context(s): ...)``, counting distinct contexts rather than files -- two rows profiled at the same
one context is still one context), since the
two mean different things and collapsing them would silently misreport a partially-measured variant
as unmeasured.

Stage 2f (``--stage 2f``) renders ``docs/results/stage2f.md``: the rank/format frontier -- wikitext
perplexity against KV bytes per token for every cell of the stage 2f grid (rank x cache format x
latent scale). Unlike every earlier stage, the row set is not a fixed dict: ``frontier_rows`` *
discovers* it by globbing ``docs/results/raw/ppl_*.json`` for names matching the row-naming scheme
(``<dump>[_u][_lk4|_lv4|_l4][_w40]``, stock rows ``baseline``/``baseline_kv4``; ``FRONTIER_RE``) and
reading each row's ppl JSON directly, so a
grid that is still mid-measurement (most of it, at the time this was written) renders whatever
exists instead of refusing or inventing dashes for a whole table of rows that have simply not run
yet. A row's K/V cache format and latent scale come from parsing its own name, which is the only
thing that works uniformly across every row (some pre-date the JSON carrying its own format fields
at all); where a row's JSON *does* carry ``k_latent_dtype``/``v_latent_dtype``/``kv_dtype``/
``latent_scale``, a disagreement with the name's parse is a warning, not something silently
ignored. Two committed rows (``palu_full_{50,70}_l4``) were measured before the cache-faithful
prefill existed and are annotated ``"latent_quant_in_prefill": false``: their perplexity is
identical to the bfp8 row by construction and is never presented as bfp4 evidence -- the table
renders their ppl cell as ``blind`` instead of a number.

Usage: ``python -m palu_tt.report [--stage 2]`` (writes the file and prints it).
``main(raw=..., out=..., rows=...)`` takes the paths and the row set as parameters so either stage
can be rendered from a fixture directory in tests.

Everything from ``MARKER`` onwards in an existing output file is hand-written prose and is
carried over verbatim, so re-rendering after a re-measured row does not destroy the notes.
"""
import argparse
import json
import re as _re
import subprocess
import warnings
from pathlib import Path

RAW = Path("docs/results/raw")
OUT = Path("docs/results/stage1.md")
OUT2 = Path("docs/results/stage2.md")
OUT2B = Path("docs/results/stage2b.md")
OUT2C = Path("docs/results/stage2c.md")
OUT2E = Path("docs/results/stage2e.md")
OUT2F = Path("docs/results/stage2f.md")

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
# Stage 2b (Task 5/6): Palu G-LRD rows plus zero-shot task accuracy (``palu_tt.eval_tasks``).
ROWS2B = {
    "baseline": "stock `tt_transformers` attention",
    "palu_full_50": "Palu full recipe (Fisher + whitening), ratio 0.5",
    "palu_full_70": "Palu full recipe (Fisher + whitening), ratio 0.7",
    "glrd_50": "Palu G-LRD (group size 4), ratio 0.5",
    "glrd_70": "Palu G-LRD (group size 4), ratio 0.7",
}
# Stage 2c (Task 3/4): 256-sample calibration sensitivity for G-LRD, plus the shared-latent
# ("group-shared") fused kernel. A single dict, in render order: the four calibration rows first,
# then the seven kernel-table rows (``main``'s "any data" guard and ``_row_data`` loading run over
# all of them together; ``_stage2c_lines`` splits them back out per table).
ROWS2C = {
    "glrd_50": "Palu G-LRD (group size 4), ratio 0.5",
    "glrd_50_f256": "  same, 256-sample Fisher calibration",
    "glrd_70": "Palu G-LRD (group size 4), ratio 0.7",
    "glrd_70_f256": "  same, 256-sample Fisher calibration",
    "baseline": "stock `tt_transformers` attention",
    "palu_full_50_fk2": "Palu full recipe (M-LRD), ratio 0.5, fused Phase 2",
    "glrd_50_fk2": "Palu G-LRD, ratio 0.5, fused Phase 2 (per-head, stage 2b)",
    "glrd_50_fk2g": "  same, group-shared kernel",
    "palu_full_70_fk2": "Palu full recipe (M-LRD), ratio 0.7, fused Phase 2",
    "glrd_70_fk2": "Palu G-LRD, ratio 0.7, fused Phase 2 (per-head, stage 2b)",
    "glrd_70_fk2g": "  same, group-shared kernel",
}
CALIB_ROWS2C = ("glrd_50", "glrd_50_f256", "glrd_70", "glrd_70_f256")
KERNEL_ROWS2C = ("baseline", "palu_full_50_fk2", "glrd_50_fk2", "glrd_50_fk2g",
                 "palu_full_70_fk2", "glrd_70_fk2", "glrd_70_fk2g")
# Decode fidelity (Task 4) is reported for the group-shared rows and for the fused kernel run on
# the f256-calibrated dumps -- neither the calibration table nor the kernel table's rows cover the
# latter, so they are loaded separately in ``_stage2c_lines``.
DECODE_FIDELITY_ROWS2C = ("glrd_50_fk2g", "glrd_70_fk2g", "glrd_50_f256_fk2", "glrd_70_f256_fk2")
# Stage 2e (Task 6/7/8): three rotation-kernel modes for the fused Phase 2 decode kernel's per-chunk
# cos/sin angle table (legacy: table per core: pack: packed into the K tile; dst: read from a
# separate DST buffer), each with and without multicasting that table instead of recomputing it per
# core, plus bfp4 latent caches (L4). ``"r"`` is not a new kernel mode: it is ``legacy`` (empty
# suffix) re-measured the same day as the five new variants, so it is a same-day noise reference
# alongside stage 2d's own ``_fk2`` file (kept as-is, not overwritten). The suffix is appended
# straight onto the stage 2c/2d ``_fk2`` row name; ``ROWS2E`` and the per-variant ablation table in
# ``_stage2e_lines`` are both driven off this one list so they cannot drift apart.
VARIANTS2E = [("", "legacy, table per core (stage 2 = 2d baseline)"),
              ("r", "legacy, re-measured the same day as the variants"),
              ("p", "pack"), ("d", "dst"),
              ("m", "legacy + table multicast"), ("pm", "pack + multicast"), ("dm", "dst + multicast")]
ROWS2E = {"baseline": "stock `tt_transformers` attention"}
for _sfx, _label in VARIANTS2E:
    ROWS2E[f"palu_full_50_fk2{_sfx}"] = f"Palu M-LRD ratio 0.5, fused Phase 2: {_label}"
    ROWS2E[f"palu_full_70_fk2{_sfx}"] = f"Palu M-LRD ratio 0.7, fused Phase 2: {_label}"
# bfp4 latents ("L4"): the M-LRD 0.5/0.7 dumps with the latent cache itself stored in bfp4 instead of
# bfp8. Perplexity is a prefill-only measurement that reconstructs K/V from the bf16 latent and never
# touches the stored cache dtype (see the caveat printed in ``_stage2e_lines``), so ``ppl_<base>`` /
# ``ppl_<base>_l4`` are read directly rather than through ``_row_data``; ``decode_check``/
# ``bench_decode`` are each variant's own files, bfp8 vs bfp4.
L4_ROWS2E = (("palu_full_50", "palu_full_50_l4", "palu_full_50_fk2", "palu_full_50_fk2_l4"),
             ("palu_full_70", "palu_full_70_l4", "palu_full_70_fk2", "palu_full_70_fk2_l4"))
# The ablation table's stock-signature reference fit (Task 8 fix): baseline and the naive Palu row
# only, never a variant's own ``_fk2*`` row -- a variant row can carry a leftover count=1 stock-kernel
# op (an uncompressed layer), and mixing that in shifts the published stock fit away from stage 2d's
# own 47.7/88.4 fixed/per-chunk us.
REF_ROWS2E = ("baseline", "palu_full_50")
# Stage 2f (Task 4): the rank/format frontier. There is no fixed row dict to visit here -- unlike
# every earlier stage, ``_stage2f_lines`` gets its rows from ``frontier_rows`` discovering whichever
# ``ppl_*.json`` files exist and match the row-naming scheme (``FRONTIER_RE``), because most of the
# grid (Grid A/B) has not run yet at the time this table is first generated. ``ROWS2F`` exists only
# so ``--stage 2f`` fits the same rows/out dispatch table as ``main``'s other stages; it is never
# iterated, and is deliberately empty rather than a guess at what the grid will contain.
ROWS2F = {}
TASK_COLS = ("openbookqa", "piqa", "arc_easy", "arc_challenge", "winogrande")
TASK_LABELS = {"openbookqa": "OBQA", "piqa": "PIQA", "arc_easy": "ARC-e",
               "arc_challenge": "ARC-c", "winogrande": "WinoGrande"}
# The paper's reference table (spec Sec 0): Llama-3-8B-Instruct, 50 % compression, lm-eval
# zero-shot, acc in %. Transcribed from the spec's table (tests/test_report.py checks the two agree);
# ``avg5`` is our own addition -- the mean of the five
# task columns we also evaluate (HellaSwag dropped), recomputed here rather than transcribed.
PAPER = {
    "baseline": {"openbookqa": 43.20, "hellaswag": 75.80, "piqa": 78.62, "arc_easy": 81.61,
                 "arc_challenge": 56.83, "winogrande": 71.90, "avg6": 67.99},
    "J-LRD": {"openbookqa": 43.40, "hellaswag": 73.20, "piqa": 76.50, "arc_easy": 79.63,
              "arc_challenge": 51.96, "winogrande": 72.45, "avg6": 66.19},
    "G-LRD": {"openbookqa": 42.60, "hellaswag": 70.36, "piqa": 76.06, "arc_easy": 76.30,
              "arc_challenge": 48.99, "winogrande": 72.38, "avg6": 64.45},
    "M-LRD": {"openbookqa": 38.80, "hellaswag": 63.04, "piqa": 73.67, "arc_easy": 69.78,
              "arc_challenge": 42.58, "winogrande": 62.51, "avg6": 58.40},
}
PAPER = {
    method: {**entry, "avg5": round(sum(entry[t] for t in TASK_COLS) / len(TASK_COLS), 2)}
    for method, entry in PAPER.items()
}
CONTEXTS = ("1024", "4096", "8192")
PREFILLS = ("2048", "8192")
# Stage 2 batch sweep (Task 7): decode step time at these batch sizes, for these rows. 32 is the
# stage-1/2 default batch and therefore has no ``_b<B>`` suffix on its JSON file. Reduced from
# (1, 2, 4, 8, 16, 32) to (1, 4, 16, 32) mid-run by the user; batches 2 and 8 were never measured.
SWEEP_BATCHES = (1, 4, 16, 32)
SWEEP_ROWS = ("baseline", "palu_full_50", "palu_full_50_fk2", "palu_full_70_fk2")
# Stage 2 context scaling (Task 8): one request (batch 1) and a small serving batch (batch 8),
# scaled out to the model's max_position_embeddings. Same four rows as the batch sweep.
SCALING_BATCHES = (1, 8)
SCALING_ROWS = SWEEP_ROWS
SCALING_CONTEXTS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
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
    # A group-wise (G-LRD) dump's rank lists are one entry per *group*, not per head, and carries
    # the full HF config, so ``num_key_value_heads`` is read from there when present; a head-wise
    # dump (group size 1, including the minimal ``head_wise_ranks``-only configs in tests) has no
    # such key and falls back to the rank-list length, which is the head count in that case.
    n_kv_heads = cfg.get("num_key_value_heads")
    if n_kv_heads is None:
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
    """``(ppl, cpu, dec, att, pre, mx, chk, tsk)`` for one row.

    A fused row (``_fk1``/``_fk2``) has its own decode/attention timings and its own decode-fidelity
    JSON; perplexity, prefill, KV bytes and max context are prefill/allocation properties that the
    fused decode kernel cannot change, so they are not re-measured and are read from the base row.
    The row's own file still wins where it exists, so a deliberately re-measured fused perplexity
    would be picked up. ``tsk`` is the zero-shot task-accuracy JSON (Task 5/stage 2b); stage 1/2
    ignore it.
    """
    base = _base_row(row)
    ppl = _load(raw, f"ppl_{row}") or _load(raw, f"ppl_{base}")   # prefill measurement: same for fused rows
    cpu = _load(raw, f"ppl_cpu_{row}") or _load(raw, f"ppl_cpu_{base}")
    dec = _load(raw, f"bench_decode_{row}")
    att = _load(raw, f"bench_attention_{row}")
    pre = _load(raw, f"bench_prefill_{row}") or _load(raw, f"bench_prefill_{base}")
    mx = _load(raw, f"bench_maxctx_{row}") or _load(raw, f"bench_maxctx_{base}")
    chk = _load(raw, f"decode_check_{row}")
    tsk = _load(raw, f"tasks_{row}")
    return ppl, cpu, dec, att, pre, mx, chk, tsk


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


def _task_display(tsk, task):
    """``"43.0"``, ``"43.0 (45.0)"`` when ``acc_norm`` is present, ``"-"`` when missing."""
    t = (tsk or {}).get("tasks", {}).get(task)
    if not t or t.get("acc") is None:
        return "-"
    s = f"{t['acc'] * 100:.1f}"
    if t.get("acc_norm") is not None:
        s += f" ({t['acc_norm'] * 100:.1f})"
    return s


def _avg5(tsk):
    """Mean ``acc`` over ``TASK_COLS`` as a percentage, or ``"-"`` unless every task is present."""
    tasks = (tsk or {}).get("tasks", {})
    accs = [tasks.get(t, {}).get("acc") for t in TASK_COLS]
    if any(a is None for a in accs):
        return "-"
    return f"{sum(accs) / len(accs) * 100:.2f}"


def _avg5_paper(tsk):
    """Mean of the paper's protocol over ``TASK_COLS`` as a percentage, or ``"-"`` unless every
    required field is present: ``acc_norm`` for openbookqa/arc_challenge, plain ``acc`` for the
    other three (spec Sec 0; matches the hand-computed ``avg (5) paper protocol`` column in
    ``stage2b.md``'s hand-written analysis, now generated instead of transcribed)."""
    tasks = (tsk or {}).get("tasks", {})
    norm_tasks = {"openbookqa", "arc_challenge"}
    vals = []
    for t in TASK_COLS:
        d = tasks.get(t, {})
        v = d.get("acc_norm") if t in norm_tasks else d.get("acc")
        if v is None:
            return "-"
        vals.append(v)
    return f"{sum(vals) / len(vals) * 100:.2f}"


def _calib_delta(f256_avg, base_avg):
    """``f256_avg - base_avg`` (both ``_avg5_paper`` strings) with an explicit sign, or ``"-"``
    when either side is missing."""
    if f256_avg == "-" or base_avg == "-":
        return "-"
    return f"{float(f256_avg) - float(base_avg):+.2f}"


def _min_only(cell):
    """``min_ms`` formatted to 1 decimal, ``"OOM"``/``"err"`` for an error cell, ``"-"`` when the
    cell is missing -- like ``_timing`` but the stage 2c kernel table only needs ``min``."""
    if cell is None:
        return "-"
    if "error" in cell:
        return "OOM" if "Out of Memory" in str(cell["error"]) else "err"
    if "min_ms" not in cell:
        return "-"
    return _fmt(cell["min_ms"], 1)


def _slope(att):
    """``(8k - 1k) / 7`` ms per 1k of context from an attention-only bench JSON, 3 decimals, or
    ``"-"`` when either context is missing or errored."""
    c1, c8 = _cell(att, "1024"), _cell(att, "8192")
    if not c1 or not c8 or "min_ms" not in c1 or "min_ms" not in c8:
        return "-"
    return f"{(c8['min_ms'] - c1['min_ms']) / 7:.3f}"


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


def _scaling_filename(what, row, batch):
    """``bench_<what>_<row>_b<batch>.json``, except batch 32's unsuffixed name (unused by the
    context-scaling scan itself, but kept so the helper matches ``_out_name``/``_sweep_filename``)."""
    return f"bench_{what}_{row}" if batch == 32 else f"bench_{what}_{row}_b{batch}"


def _scaling_decode_cell(raw, row, batch, ctx):
    """Decode ``min_ms`` for one context-scaling cell, or ``None`` (missing file/context, or an
    errored cell -- mirrors ``_sweep_cell``)."""
    cell = _cell(_load(raw, _scaling_filename("decode", row, batch)), str(ctx))
    return cell.get("min_ms") if cell and "min_ms" in cell else None


def _scaling_decode_display(raw, row, batch, ctx):
    """``"min ms"``, ``"OOM"`` for an error cell, ``"-"`` when the cell is simply missing."""
    cell = _cell(_load(raw, _scaling_filename("decode", row, batch)), str(ctx))
    if cell and "error" in cell:
        return "OOM"
    min_ms = _scaling_decode_cell(raw, row, batch, ctx)
    return "-" if min_ms is None else _fmt(min_ms, 1)


def _scaling_maxctx(raw, row, batch):
    """``(max_context, limit)`` for one (row, batch) from its ``bench_maxctx_*`` JSON, or
    ``("-", "-")`` when that file does not exist."""
    data = (_load(raw, _scaling_filename("maxctx", row, batch)) or {}).get("data") or {}
    mc = data.get("max_context")
    return (mc if mc is not None else "-"), (data.get("limit") or "-")


def _steps(files):
    """The decode-check step counts present, as a header string (mirrors ``_windows``)."""
    ss = sorted({f["steps"] for f in files if f and f.get("steps") is not None})
    return "/".join(str(x) for x in ss) if ss else "?"


def _quality_row(entry, kv_base):
    """Common quality-table cells for one row: ranks, KV bytes/total/reduction, max ctx.

    Shared by stage 1/2 (``main``) and stage 2b (``_stage2b_lines``); the two callers differ only
    in which extra columns (CPU ppl, decode fidelity) they append around this.
    """
    ppl, cpu, dec, att, pre, mx, chk, tsk = entry
    rk, rv = _ranks([ppl, dec, att, pre, mx])
    kv = _kv_model_mean(_kv_bytes([dec, att, mx]), [ppl, dec, att, pre, mx])
    kv_total = "-" if kv is None else f"{round(kv * LAYERS * CHIPS)}"
    ratio = "-" if (kv is None or not kv_base) else f"{kv_base / kv:.2f}x"
    maxctx = (mx.get("data") or {}).get("max_context_batch32") if mx else None
    return rk, rv, kv, kv_total, ratio, maxctx


def _latency_rows(rows, loaded, stock_dec, vs_stock):
    """``| row | context | decode | attention |`` lines for each row across ``CONTEXTS``, with an
    optional trailing ``vs stock`` column -- shared by stage 1/2 (``main``) and stage 2b so the two
    tables cannot drift apart."""
    for row in rows:
        dec, att = loaded[row][2], loaded[row][3]
        for ctx in CONTEXTS:
            cells = f"| {row} | {ctx} | {_timing(_cell(dec, ctx))} | {_timing(_cell(att, ctx))} |"
            if vs_stock:
                cells += f" {_vs_stock(stock_dec, dec, ctx)} |"
            yield cells


def main(raw=RAW, out=OUT, rows=ROWS, stage=None):
    """Render one stage's tables. ``rows`` is ``ROWS`` (stage 1), ``ROWS2`` (stage 2), ``ROWS2B``,
    ``ROWS2C`` or ``ROWS2E``. ``stage`` resolves to one explicit id up front: an explicit
    ``stage=...`` always wins (so ``stage="2e"`` renders 2e even for a ``rows`` dict that is not
    literally ``ROWS2E`` itself, e.g. a subset built for a test); omitted, it falls back to the
    older stages' own convention of inferring from ``rows`` by identity. Resolving one id rather
    than checking ``rows is ROWS2X`` and an explicit ``stage`` string independently means the two
    can never disagree (``main(rows=ROWS2C, stage="2e")`` used to silently render 2c).

    Stage 2 adds the two decode-fidelity columns to the quality table and a ``vs stock`` column to
    the latency table; everything else -- the loaders, the KV arithmetic, the error cells and the
    hand-written tail -- is shared, so the two documents cannot drift apart.
    """
    raw = Path(raw)
    stage = stage or ("2" if rows is ROWS2 else "2b" if rows is ROWS2B else "2c" if rows is ROWS2C
                       else "2e" if rows is ROWS2E else "2f" if rows is ROWS2F else "1")
    if stage == "2f":
        # The frontier's row set is discovered from the raw dir itself (frontier_rows), not a fixed
        # dict, so it skips the generic per-row loading and the "no data at all" refusal below --
        # those protect the fixed-row stages from silently blanking a committed table over a bad
        # --raw, but an almost-empty raw dir is the normal, expected state for most of stage 2f's
        # own run and must render as a (mostly empty) table, not a crash.
        return _finish(_stage2f_lines(raw), out)
    stage2 = stage == "2"
    stage2b = stage == "2b"
    stage2c = stage == "2c"
    stage2e = stage == "2e"
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

    if stage2b:
        return _finish(_stage2b_lines(raw, rows, loaded, base, kv_base, dev_windows), out)

    if stage2c:
        return _finish(_stage2c_lines(raw, rows, loaded), out)

    if stage2e:
        # The fidelity table also reports the bfp4-latent (L4) rows, which are not part of ROWS2E
        # itself (perplexity/KV/decode-fidelity for those is read directly in _stage2e_lines) -- load
        # them here so they reach the generic "any row with 'fk2' in its name and a decode_check"
        # fidelity loop without duplicating that loop's logic.
        loaded = {**loaded, **{r: _row_data(raw, r) for t in L4_ROWS2E for r in (t[2], t[3])}}
        return _finish(_stage2e_lines(raw, rows, loaded, steps), out)

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
        ppl, cpu, dec, att, pre, mx, chk, tsk = loaded[row]
        rk, rv, kv, kv_total, ratio, maxctx = _quality_row(loaded[row], kv_base)
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
    lines += _latency_rows(rows, loaded, stock_dec, stage2)

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

        lines += ["", "## Context scaling: one request (batch 1) and a small serving batch (batch 8)", ""]
        ctx_header = " | ".join(f"{c // 1024}k" for c in SCALING_CONTEXTS)
        for b in SCALING_BATCHES:
            lines += [
                f"### Batch {b}",
                "",
                f"| row | max context | limit | {ctx_header} |",
                "|---|---|---|" + "---|" * len(SCALING_CONTEXTS),
            ]
            for row in SCALING_ROWS:
                mc, limit = _scaling_maxctx(raw, row, b)
                cells = " | ".join(_scaling_decode_display(raw, row, b, c) for c in SCALING_CONTEXTS)
                lines.append(f"| {row} | {mc} | {limit} | {cells} |")
            lines.append("")
        lines += [
            "`max context` is the largest `max_seq_len` that allocated (bisected to 1k granularity); "
            "`limit` is `oom` when a build failed there, `model` when it is the model's own "
            f"`max_position_embeddings` ({SCALING_CONTEXTS[-1]}) or a non-memory build rejection. Decode "
            "step cells are `min_ms`, `OOM` for a context whose build failed, `-` for a context not run "
            "(the scan stops after the first failure).",
            "",
            "![Decode step vs context, batch 1](ctx_scaling_b1.png)",
            "",
            "![Decode step vs context, batch 8](ctx_scaling_b8.png)",
            "",
        ]

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

    return _finish(lines, out)


def _finish(lines, out):
    """Append the marker, freeze the text, and (when ``out`` is given) write it back with any
    hand-written tail preserved -- shared by every stage so the bootstrap/preserve behaviour
    cannot drift between them."""
    lines = lines + ["", MARKER, ""]
    text = "\n".join(lines).rstrip("\n") + "\n"

    if out is not None:
        out = Path(out)
        if out.exists() and MARKER in out.read_text():
            kept = out.read_text().split(MARKER, 1)[1].lstrip("\n")
            if kept.strip():
                text = text + "\n" + kept.rstrip("\n") + "\n"
        out.write_text(text)
    return text


def _stage2b_fk2_rows(raw, rows):
    """Base rows in render order, each immediately followed by its ``_fk2`` variant when that
    variant has its own decode-timing JSON (the fused kernel is optional per row)."""
    out = []
    for row, label in rows.items():
        out.append((row, label))
        fk2 = f"{row}_fk2"
        if _load(raw, f"bench_decode_{fk2}") is not None:
            out.append((fk2, "  same, fused Phase 2"))
    return out


def _stage2b_lines(raw, rows, loaded, base, kv_base, dev_windows):
    """Stage 2b: zero-shot task accuracy (Task 5) plus the G-LRD rows, on top of the same
    quality/latency tables as stage 1/2. ``_fk2`` variants are picked up when present rather than
    listed in ``ROWS2B`` itself, since not every row has been re-run with the fused kernel yet."""
    ext = _stage2b_fk2_rows(raw, rows)
    ext_loaded = dict(loaded)
    for row, _ in ext:
        ext_loaded.setdefault(row, _row_data(raw, row))
    stock_dec = base[2] if base else None

    lines = [
        "# Stage 2b results: G-LRD and zero-shot task accuracy on a Wormhole n300",
        "",
        f"Generated by `python -m palu_tt.report --stage 2b` from `docs/results/raw/*.json` at commit "
        f"`{_commit()}`; tt-metal v0.78.0 plus the `palu-sdpa` submodule branch, TP=2 (one n300, two "
        "chips), bfp8 weights and latent caches, non-paged, untraced decode. Zero-shot task accuracy "
        "is measured by `palu_tt.eval_tasks` (lm-eval, 0-shot, greedy) on the device.",
        "",
        "## Zero-shot accuracy",
        "",
        "| row | recipe | " + " | ".join(TASK_LABELS[t] for t in TASK_COLS) + " | avg (5) |",
        "|---|---|" + "---|" * (len(TASK_COLS) + 1),
    ]
    for row, label in rows.items():
        tsk = loaded[row][7]
        cells = " | ".join(_task_display(tsk, t) for t in TASK_COLS)
        lines.append(f"| {row} | {label} | {cells} | {_avg5(tsk)} |")

    lines += [
        "",
        "### Paper (Llama-3-8B-Instruct, 50 %)",
        "",
        "| method | OBQA | HellaSwag | PIQA | ARC-e | ARC-c | WinoGrande | avg (6) | avg (5) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for method in ("baseline", "J-LRD", "G-LRD", "M-LRD"):
        p = PAPER[method]
        lines.append(
            f"| {method} | {p['openbookqa']:.1f} | {p['hellaswag']:.1f} | {p['piqa']:.1f} | "
            f"{p['arc_easy']:.1f} | {p['arc_challenge']:.1f} | {p['winogrande']:.1f} | "
            f"{p['avg6']:.2f} | {p['avg5']:.2f} |"
        )
    lines += [
        "",
        "The paper's numbers are for Llama-3-8B-Instruct (ours is Llama-3.1-8B-Instruct), calibrated "
        "with 2048 Fisher samples (ours: 32), and report six tasks; we evaluate five, dropping "
        "HellaSwag for time -- so `avg (5)` above, not `avg (6)`, is the comparable column.",
        "",
        "## Quality and memory",
        "",
        f"| row | recipe | r_k | r_v | ppl (device, {dev_windows} windows) | "
        "KV B/token/layer/chip | KV B/token (32 layers, both chips) | KV reduction | "
        "max ctx @ batch 32 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row, label in rows.items():
        ppl = loaded[row][0]
        rk, rv, kv, kv_total, ratio, maxctx = _quality_row(loaded[row], kv_base)
        lines.append(
            f"| {row} | {label} | {rk} | {rv} | {_fmt(ppl and ppl.get('ppl'), 3)} | "
            f"{'-' if kv is None else round(kv)} | {kv_total} | {ratio} | {maxctx or '-'} |"
        )

    lines += [
        "",
        "KV bytes are per token, per layer, per chip, **averaged over the 32 layers**. `bench` reports "
        "the figure for layer 0; for the calibrated rows (per-layer ranks) it is rescaled by the mean "
        "rank sum from `head_wise_ranks`. `KV reduction` is stock bytes / row bytes.",
        "",
        "## Decode and attention latency, batch 32",
        "",
        "Measured on an idle host (device-idle gate passed, no other job of ours running). Decode is "
        "untraced and therefore partly host-dispatch-bound, so `p50` and `min` are both reported; "
        "`min` is the cleaner signal and every ratio quoted below the marker uses it. `vs stock` is "
        "the stock row's decode `min_ms` divided by this row's at the same context (> 1.00x is "
        "faster than stock); it is a dash at 8192, where stock does not allocate.",
        "",
        "| row | context | decode step p50 / min ms | attention-only p50 / min ms | vs stock |",
        "|---|---|---|---|---|",
    ]
    lines += _latency_rows([row for row, _ in ext], ext_loaded, stock_dec, True)

    fk2_rows = [(row, label) for row, label in ext if row.endswith("_fk2")]
    lines += ["", "## Decode fidelity", ""]
    if fk2_rows:
        fk2_steps = _steps([ext_loaded[row][6] for row, _ in fk2_rows])
        lines += [
            f"Naive vs fused (`palu_tt.decode_check`, teacher-forced, {fk2_steps} steps).",
            "",
            "| row | decode PCC (min / mean) | argmax agree |",
            "|---|---|---|",
        ]
        for row, _ in fk2_rows:
            pcc, agree = _fidelity(ext_loaded[row][6])
            lines.append(f"| {row} | {pcc} | {agree} |")
    else:
        lines.append("No fused-kernel row has a `decode_check_*` JSON yet.")

    lines += ["", "## Batch-8 max context", ""]
    lines += ["| row | max context | limit |", "|---|---|---|"]
    for row, _ in ext:
        mc, limit = _scaling_maxctx(raw, row, 8)
        lines.append(f"| {row} | {mc} | {limit} |")

    return lines


def _stage2c_lines(raw, rows, loaded):
    """Stage 2c (Task 3): calibration sensitivity for G-LRD at 256 Fisher samples, and the
    shared-latent ("group-shared") fused kernel against the stage 2b per-head ``_fk2`` rows.

    ``loaded`` already carries every ``ROWS2C`` row (``main``'s generic ``_row_data`` loop); the
    two extra decode-fidelity rows (the fused kernel run on an f256-calibrated dump) are not part
    of either table above and are loaded here on demand.
    """
    lines = [
        "# Stage 2c results: 256-sample calibration and the shared-latent decode kernel",
        "",
        f"Generated by `python -m palu_tt.report --stage 2c` from `docs/results/raw/*.json` at "
        f"commit `{_commit()}`; tt-metal v0.78.0 plus the `palu-sdpa` submodule branch, TP=2 (one "
        "n300, two chips), bfp8 weights and latent caches, non-paged, untraced decode. `_f256` rows "
        "are the G-LRD dumps re-calibrated with 256 Fisher samples in place of stage 2b's 32; "
        "`_fk2g` rows are the fused Phase 2 kernel in group-shared mode (`heads_per_core` chosen "
        "automatically per layer), compared against the stage 2b `_fk2` (per-head) rows.",
        "",
        "## Calibration sensitivity",
        "",
        "Zero-shot task accuracy (`palu_tt.eval_tasks`) and perplexity for the two G-LRD ratios, "
        "32- vs 256-sample Fisher calibration. `avg (5)` is plain `acc`; `avg (5) paper protocol` "
        "uses `acc_norm` for OpenBookQA/ARC-Challenge and `acc` for the rest (the paper's "
        "protocol, spec Sec 0); `delta` is the f256 row's paper-protocol average minus its "
        "32-sample counterpart's.",
        "",
        "| row | recipe | " + " | ".join(TASK_LABELS[t] for t in TASK_COLS) +
        " | avg (5) | avg (5) paper protocol | delta | ppl (device) | r_k | r_v |",
        "|---|---|" + "---|" * (len(TASK_COLS) + 6),
    ]
    for row in CALIB_ROWS2C:
        entry = loaded[row]
        ppl, tsk = entry[0], entry[7]
        cells = " | ".join(_task_display(tsk, t) for t in TASK_COLS)
        avg5p = _avg5_paper(tsk)
        rk, rv, *_rest = _quality_row(entry, None)
        if row.endswith("_f256"):
            delta = _calib_delta(avg5p, _avg5_paper(loaded[row[:-len("_f256")]][7]))
        else:
            delta = "-"
        lines.append(
            f"| {row} | {rows[row]} | {cells} | {_avg5(tsk)} | {avg5p} | {delta} | "
            f"{_fmt(ppl and ppl.get('ppl'), 3)} | {rk} | {rv} |"
        )

    lines += [
        "",
        "## Shared-latent kernel",
        "",
        "Decode step and attention-only `min_ms` (untraced, idle host, same convention as the "
        "earlier stages -- `min` is the cleaner signal). Batch 32 is the default `bench_decode_*` "
        "file, batch 8 the `_b8` sweep file. `slope` is `(attention 8k - attention 1k) / 7`, ms per "
        "1k of context. `vs _fk2` is the `_fk2` sibling's decode step at 4k divided by this row's "
        "at 4k (> 1.00x is faster than the per-head kernel); it is only rendered for the `_fk2g` "
        "rows.",
        "",
        "| row | recipe | decode 1k b32 | decode 4k b32 | decode 8k b32 | decode 1k b8 | "
        "decode 4k b8 | decode 8k b8 | attn 1k | attn 8k | slope (ms/1k) | vs `_fk2` |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in KERNEL_ROWS2C:
        dec32, att = loaded[row][2], loaded[row][3]
        dec8 = _load(raw, f"bench_decode_{row}_b8")
        d32 = " | ".join(_min_only(_cell(dec32, c)) for c in CONTEXTS)
        d8 = " | ".join(_min_only(_cell(dec8, c)) for c in CONTEXTS)
        if row.endswith("_fk2g"):
            vs = _vs_stock(loaded[row[:-1]][2], dec32, "4096")   # "..._fk2g" -> its "..._fk2" sibling at 4k
        else:
            vs = "-"
        lines.append(
            f"| {row} | {rows[row]} | {d32} | {d8} | {_min_only(_cell(att, '1024'))} | "
            f"{_min_only(_cell(att, '8192'))} | {_slope(att)} | {vs} |"
        )

    ext_loaded = dict(loaded)
    for row in DECODE_FIDELITY_ROWS2C:
        ext_loaded.setdefault(row, _row_data(raw, row))
    lines += [
        "",
        "## Decode fidelity",
        "",
        "Naive vs fused (`palu_tt.decode_check`, teacher-forced), for the group-shared rows and for "
        "the fused kernel run on the f256-calibrated dumps.",
        "",
        "| row | decode PCC (min / mean) | argmax agree |",
        "|---|---|---|",
    ]
    for row in DECODE_FIDELITY_ROWS2C:
        pcc, agree = _fidelity(ext_loaded[row][6])
        lines.append(f"| {row} | {pcc} | {agree} |")

    lines += ["", "## Batch-8 max context", "", "| row | max context | limit |", "|---|---|---|"]
    for base in ("glrd_50", "glrd_70"):
        for row in (f"{base}_fk2", f"{base}_fk2g"):
            mc, limit = _scaling_maxctx(raw, row, 8)
            lines.append(f"| {row} | {mc} | {limit} |")

    return lines


def _stage2e_lines(raw, rows, loaded, steps):
    """Stage 2e (Task 6/7): the three rotation-kernel modes (legacy/pack/dst), each with and without
    angle-table multicast, plus bfp4 latent caches (L4) -- all on top of the stage 2d fused Phase 2
    M-LRD kernel.

    The measurement matrix (``VARIANTS2E`` x ratio, plus the L4 rows) can still be running when this
    renders: a variant or row with no JSON yet is skipped with a one-line ``(not measured)`` note in
    whichever table it would have appeared in, never a crash -- every lookup below goes through
    ``_load``/``loaded``, both of which return ``None``/absent for a file that does not exist yet.
    """
    from palu_tt import profile_fit

    lines = [
        "# Stage 2e results: idle-resource kernel variants and bfp4 latents",
        "",
        f"Generated by `python -m palu_tt.report --stage 2e` from `docs/results/raw/*.json` at commit "
        f"`{_commit()}`; tt-metal v0.78.0 plus the `palu-sdpa` submodule branch, TP=2 (one n300, two "
        "chips), bfp8 weights, non-paged, untraced decode, on top of the stage 2d fused Phase 2 "
        "M-LRD ratio-0.5/0.7 kernel. `legacy`/`pack`/`dst` are the three ways the rotation kernel "
        "gets its per-chunk cos/sin angle table onto the compute core (table per core, packed into "
        "the K tile, or read from a separate DST buffer); `multicast` broadcasts that table instead "
        "of recomputing it per core. `r` is not a new kernel mode -- it is `legacy` re-measured the "
        "same day as the five new variants, a same-day noise reference kept alongside stage 2d's own "
        "`_fk2` file rather than overwriting it. `l4` (section 4) stores the latent cache itself, not "
        "K/V, in bfp4 instead of bfp8.",
        "",
    ]

    # 1. ablation: fixed + per-chunk cost per layer, per kernel variant (the fit from stage 2d
    # section 4, repeated for each variant). ``sfx == ""`` (legacy) also shows the stock reference
    # signature(s) -- fit *only* from REF_ROWS2E (baseline, palu_full_50), never from a variant row,
    # so a count=1 noise op left over in e.g. palu_full_70_fk2's own JSON (an uncompressed layer that
    # still runs the stock kernel) cannot shift the published stock fit away from stage 2d's own
    # 47.7/88.4 -- alongside its own fused one; every other variant shows only its own fused
    # (``PaluSdpaDecode*``) signature, fit from its own two rows only, so the six later blocks do not
    # just repeat the same two reference rows six times over.
    lines += [
        "## Kernel variants: fixed and per-chunk cost per layer (batch 32)",
        "",
        "| variant | signature | rows | fixed us | per chunk us | per item per chunk us | "
        "points (chunks: us/layer) |",
        "|---|---|---|---|---|---|---|",
    ]
    for sfx, label in VARIANTS2E:
        variant_rows = [f"palu_full_50_fk2{sfx}", f"palu_full_70_fk2{sfx}"]
        shown = [s for s in profile_fit.fit_signatures(raw, variant_rows, [1024, 4096, 8192])
                 if s["signature"].startswith("PaluSdpaDecode")]
        if sfx == "":
            shown = profile_fit.fit_signatures(raw, list(REF_ROWS2E), [1024, 4096, 8192]) + shown
        if not shown:
            # No fitted signature: either no profile JSON exists for this variant's rows at all
            # ("not measured"), or some exist but fewer than the two distinct contexts a fit needs
            # (fit_signatures silently drops those) -- these mean different things and collapsing
            # them would misreport a partially-measured variant as untouched.
            existing = [(row, ctx) for row in variant_rows for ctx in (1024, 4096, 8192)
                        if (Path(raw) / f"profile_{row}_{ctx}.json").exists()]
            if existing:
                # Distinct contexts, not profile JSON files: two rows profiled at the same one
                # context is still only one context, and a fit needs two distinct ones -- counting
                # files there would over-report how measured a variant actually is.
                n_ctx = len({ctx for _, ctx in existing})
                desc = ", ".join(f"{row} @ {ctx}" for row, ctx in existing)
                lines.append(f"| {label} | ({len(existing)} profile JSON(s) at {n_ctx} distinct "
                             f"context(s): {desc} -- a fit needs two distinct contexts) |")
            else:
                lines.append(f"| {label} | (not measured) |")
            continue
        for s in shown:
            pts = ", ".join(f"{x}: {y:.0f}" for x, y in s["points"])
            lines.append(f"| {label} | {s['signature']} | {', '.join(s['rows'])} | {s['fixed_us']:.1f} | "
                         f"{s['per_chunk_us']:.1f} | {s['per_item_per_chunk_us']:.1f} | {pts} |")
    lines.append("")

    # 2. whole step: decode min_ms at each context, vs stock, for every row in ROWS2E (in order).
    stock_dec = loaded["baseline"][2] if "baseline" in loaded else None
    lines += [
        "## Decode step, batch 32 (min ms; `vs stock` = stock / row at the same context)",
        "",
        "| row | 1024 | 4096 | 8192 | vs stock 1k | vs stock 4k |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        dec = loaded[row][2] if row in loaded else None
        if dec is None:
            lines.append(f"| {row} | (not measured) |")
            continue
        cells = " | ".join(_min_only(_cell(dec, ctx)) for ctx in CONTEXTS)
        vs = " | ".join(_vs_stock(stock_dec, dec, ctx) for ctx in CONTEXTS[:2])
        lines.append(f"| {row} | {cells} | {vs} |")
    lines.append("")

    # 3. fidelity of every measured fk2* row (naive vs fused, teacher-forced decode_check), including
    # the bfp4-latent (_l4) rows main() loads on top of ROWS2E for this table.
    lines += [
        f"## Decode fidelity (`decode_check`, {steps} teacher-forced steps)",
        "",
        "| row | PCC min / mean | argmax agree |",
        "|---|---|---|",
    ]
    fid_rows = sorted(r for r in loaded if "fk2" in r and loaded[r][6] is not None)
    if fid_rows:
        for row in fid_rows:
            pcc, agree = _fidelity(loaded[row][6])
            lines.append(f"| {row} | {pcc} | {agree} |")
    else:
        lines.append("No row has a `decode_check_*` JSON yet.")
    lines.append("")

    # 4. bfp4 latents (L4): the same M-LRD dumps with the latent cache stored in bfp4 instead of
    # bfp8. Perplexity is read directly from ``ppl_<base>``/``ppl_<base>_l4`` (not through ``loaded``
    # or ``_row_data``) since it is a prefill-only measurement, not one of ``ROWS2E``'s own rows.
    lines += [
        "## bfp4 latents (L4): the same dumps with the latent cache stored in bfp4",
        "",
        "| dump | ppl (prefill-blind, see stage 2f) bfp8 | ppl (prefill-blind, see stage 2f) bfp4 | "
        "KV B/token/layer/chip bfp8 -> bfp4 | decode PCC min bfp8 -> bfp4 | decode 4k ms bfp8 -> bfp4 |",
        "|---|---|---|---|---|---|",
    ]
    for base, l4, fk2, fk2_l4 in L4_ROWS2E:
        ppl8, ppl4 = _load(raw, f"ppl_{base}"), _load(raw, f"ppl_{l4}")
        chk8, chk4 = _load(raw, f"decode_check_{fk2}"), _load(raw, f"decode_check_{fk2_l4}")
        b8, b4 = _load(raw, f"bench_decode_{fk2}"), _load(raw, f"bench_decode_{fk2_l4}")
        if not any([ppl8, ppl4, chk8, chk4, b8, b4]):
            lines.append(f"| {base} | (not measured) |")
            continue
        kv8 = _kv_bytes([b8]) if b8 else None
        kv4 = _kv_bytes([b4]) if b4 else None
        pcc8 = chk8["pcc_min"] if chk8 else None
        pcc4 = chk4["pcc_min"] if chk4 else None
        lines.append(
            f"| {base} | {_fmt(ppl8 and ppl8.get('ppl'), 3)} | {_fmt(ppl4 and ppl4.get('ppl'), 3)} | "
            f"{kv8 if kv8 is not None else '-'} -> {kv4 if kv4 is not None else '-'} | "
            f"{_fmt(pcc8, 4)} -> {_fmt(pcc4, 4)} | "
            f"{_min_only(_cell(b8, '4096')) if b8 else '-'} -> {_min_only(_cell(b4, '4096')) if b4 else '-'} |"
        )
    lines += [
        "",
        "Perplexity here comes from a prefill-only instrument that reconstructs K/V from the bf16 "
        "latent before the forward pass, so it is blind to the latent cache's on-device storage "
        "format -- bfp4 and bfp8 perplexity are identical to 15 digits by construction, and this "
        "column is not evidence about bfp4 quality. Stage 2f's cache-faithful prefill (which reads "
        "K/V back out of the stored-format cache) supersedes it; the `decode_check` PCC / "
        "argmax-agreement figures above are measured against the real on-device cache and are the "
        "meaningful bfp4 evidence.",
        "",
    ]
    return lines


# Stage 2f (Task 4): the rank/format frontier.
#
# Row naming (the 2f spec): ``<dump>[_kv4][_u][_lk4|_lv4|_l4][_w40]`` -- ``_kv4`` only on ``baseline``
# (stock's whole KV cache in bfp4, the format-only competitor); ``_u`` is unit-variance latent scale;
# ``_lk4``/``_lv4``/``_l4`` put the K/V/both latent caches in bfp4 (default bfp8); ``_w40`` is a
# 40-window perplexity screen (default 141 windows). ``FRONTIER_RE`` is anchored on both ends so a
# row that does not fit the scheme (an unrelated ``ppl_*.json``, e.g. a stage 1/2 row) is quietly
# skipped rather than mis-parsed.
FRONTIER_RE = _re.compile(r"^(?P<dump>baseline|palu_full_50|palu_full_70|glrd_50_f256|glrd_70_f256|uni_rk\d+_rv\d+)"
                          r"(?P<kv4>_kv4)?(?P<u>_u)?(?P<fmt>_lk4|_lv4|_l4)?(?P<w40>_w40)?$")
BPE = {"bfp8": 1.0625, "bfp4": 0.5625}  # bytes per tile element (spec sec 2)
# Stock is 4 KV heads per chip (8 heads / 2 chips), group size 1 -- one latent group per head, so
# this is unchanged by the groups-per-chip fix below (see ``_dump_ranks``).
STOCK_GROUPS_PER_CHIP = 4
STOCK_BYTES_LAYER_CHIP = STOCK_GROUPS_PER_CHIP * (HEAD_DIM * BPE["bfp8"] + HEAD_DIM * BPE["bfp8"])  # 1088 B bfp8


def _config_groups_per_layer(cfg):
    """Latent groups per layer for a Palu-recipe dump, from the length of one of ``head_wise_ranks``'s
    own per-layer rank lists.

    ``head_wise_ranks`` carries no explicit group-size field, so this length is the only place the
    group count is recoverable: a per-head dump (group size 1, e.g. the M-LRD ``palu_full_*``
    recipes) has one entry per KV head (8, for this model), while a group-shared dump (G-LRD, group
    size 4) has one entry per *group* (2) -- ``convert.load_palu_dir``/``_per_layer_ranks`` already
    zero-pad/broadcast these back out to per-head ranks for the device allocation, but the raw list
    length here is the group count before that broadcast.
    """
    ranks = cfg["head_wise_ranks"]
    return len(next(iter(ranks.values())))


def _dump_ranks(dump, ranks):
    """(mean r_k, mean r_v, groups per chip) for a dump: the caller's override, else derived from
    the dump itself.

    ``baseline`` is full rank by construction, 4 KV heads per chip; a ``uni_rk<K>_rv<V>`` dump's
    ranks are in its own name, also 4 (per-head, group size 1) heads per chip; every other dump is a
    Palu/G-LRD recipe whose per-layer ranks are read the same way ``_ranks`` reads them for stage
    1/2/2b/2c -- via ``convert.load_palu_dir`` against its ``config.json``, since a calibrated dump's
    per-head ranks are not uniform and only that reader gives the exact per-layer allocation the
    device builds caches for. Its groups-per-chip is derived from that same ``config.json`` (see
    ``_config_groups_per_layer``): 4 for a per-head (M-LRD) dump, 1 for a group-shared (G-LRD, group
    size 4) one, since the 8 KV heads are split evenly across the n300's two chips either way. Every
    one of these rank sources is either the caller's override, the row name, or a fixed path under
    ``compressed/`` -- never anything read from a raw results dir -- so, unlike ``frontier_rows``'
    other per-row helpers, this one takes no ``raw`` parameter.

    ``ranks``' override values may be a 2-tuple ``(r_k, r_v)``, which defaults groups-per-chip to 4
    (a per-head dump, the common case in existing tests), or a 3-tuple ``(r_k, r_v, groups_per_chip)``
    to also override the group count -- needed to exercise a group-shared (G-LRD-like) dump without
    building a real ``compressed/`` recipe directory.
    """
    if dump in ranks:
        v = ranks[dump]
        return (v[0], v[1], v[2]) if len(v) > 2 else (v[0], v[1], 4)
    if dump == "baseline":
        return HEAD_DIM, HEAD_DIM, STOCK_GROUPS_PER_CHIP
    if dump.startswith("uni_"):
        m = _re.match(r"uni_rk(\d+)_rv(\d+)", dump)
        return int(m.group(1)), int(m.group(2)), 4
    palu_dir = {"palu_full_50": "compressed/palu_full_0.5", "palu_full_70": "compressed/palu_full_0.7",
                "glrd_50_f256": "compressed/glrd_0.5_f256", "glrd_70_f256": "compressed/glrd_0.7_f256"}[dump]
    cfg = json.loads((Path(palu_dir) / "config.json").read_text())
    rk, rv = _per_layer_ranks(Path(palu_dir), cfg)
    groups_per_chip = _config_groups_per_layer(cfg) // CHIPS
    return sum(rk) / len(rk), sum(rv) / len(rv), groups_per_chip


def _row_is_blind(d, has_bfp4_latent_suffix):
    """Whether this row's perplexity is prefill-blind to the stored latent cache format (measured
    before stage 2f Task 1b's cache-faithful prefill) and must not be reported as bfp4 evidence.

    ``latent_quant_in_prefill`` is the authoritative field when a row's JSON carries it -- ``False``
    on the two committed ``_l4`` rows measured before Task 1b landed. A row with no such field at
    all is a pre-2f measurement (older than the field itself): bfp8 rows were never blind, so those
    are trusted as faithful, but a bfp4-latent-suffixed row with no field cannot be -- it predates
    both the fix and the record of whether the fix was in effect.
    """
    lqip = d.get("latent_quant_in_prefill")
    if lqip is not None:
        return not lqip
    return has_bfp4_latent_suffix


def _check_row_fields(row, d, stock, k_fmt, v_fmt, scale):
    """Warn (never raise, never silently pass) when a row's own JSON carries the 2f format/scale
    fields (``k_latent_dtype``/``v_latent_dtype``/``kv_dtype``/``latent_scale``, written by the CLIs
    since 2f Tasks 1b/2) and they disagree with what its file name parses to. The name is what every
    row -- including the ones measured before these fields existed -- can be read from, so it stays
    the source of truth here, but a real disagreement on a row that *does* carry the fields would
    mean a mis-named file or a CLI bug, and that must not pass unnoticed.
    """
    mismatches = []
    if stock:
        if d.get("kv_dtype") is not None and d["kv_dtype"] != k_fmt:
            mismatches.append(f"kv_dtype={d['kv_dtype']!r} vs name-parsed {k_fmt!r}")
    else:
        if d.get("k_latent_dtype") is not None and d["k_latent_dtype"] != k_fmt:
            mismatches.append(f"k_latent_dtype={d['k_latent_dtype']!r} vs name-parsed {k_fmt!r}")
        if d.get("v_latent_dtype") is not None and d["v_latent_dtype"] != v_fmt:
            mismatches.append(f"v_latent_dtype={d['v_latent_dtype']!r} vs name-parsed {v_fmt!r}")
    if d.get("latent_scale") is not None and d["latent_scale"] != scale:
        mismatches.append(f"latent_scale={d['latent_scale']!r} vs name-parsed {scale!r}")
    if mismatches:
        warnings.warn(f"frontier_rows: ppl_{row}.json disagrees with its own row name: "
                      + "; ".join(mismatches) + " -- trusting the name.")


# A frontier row's decode_4k/maxctx columns must only ever show the fused Phase 2 decode kernel's
# own numbers, never the naive (reconstruct-then-attend) row's -- see ``_frontier_fused_bench``.
# Preference order: the shipped-default multicast variant (stage 2e) first, then plain fused Phase 2.
FRONTIER_FUSED_SUFFIXES = ("_fk2dm", "_fk2")


def _frontier_fused_bench(raw, what, row):
    """``(data, suffix)`` for ``row``'s fused Phase 2 decode-kernel bench file, preferring
    ``_fk2dm`` (the shipped-default rotation-kernel variant, stage 2e) over plain ``_fk2``; ``(None,
    None)`` when neither exists.

    ``bench_<what>_<row>.json`` (no suffix) is the naive, reconstruct-then-attend kernel for a
    ``palu_full_*``/``glrd_*`` dump -- multiple times slower than the fused kernel it stands in for
    on a decode-latency frontier -- and is never read here, on purpose, even as a fallback.
    """
    if row == "baseline" or row.startswith("baseline_"):
        # stock rows have no fused/naive split: the plain file IS the stock tt_transformers kernel
        data = _load(raw, f"bench_{what}_{row}")
        return (data, "") if data is not None else (None, None)
    for suffix in FRONTIER_FUSED_SUFFIXES:
        data = _load(raw, f"bench_{what}_{row}{suffix}")
        if data is not None:
            return data, suffix
    return None, None


def frontier_rows(raw, ranks=None):
    """One dict per stage 2f row found in ``raw`` (see module docstring and ``FRONTIER_RE``).

    ``ranks`` overrides ``_dump_ranks``' own per-dump lookup -- tests pass synthetic ranks so they
    never have to build a real ``compressed/*`` dump just to exercise the byte/format parsing.
    Sorted by KB/token then perplexity, i.e. roughly the Pareto-front order ``_stage2f_lines`` and
    ``plots.frontier_figure`` both want.
    """
    raw, ranks = Path(raw), ranks or {}
    out = []
    for p in sorted(raw.glob("ppl_*.json")):
        row = p.stem[len("ppl_"):]
        m = FRONTIER_RE.match(row)
        if not m:
            continue
        dump = m.group("dump")
        stock = dump == "baseline"
        fmt = m.group("fmt") or ""
        k_fmt = "bfp4" if (fmt in ("_lk4", "_l4") or (stock and m.group("kv4"))) else "bfp8"
        v_fmt = "bfp4" if (fmt in ("_lv4", "_l4") or (stock and m.group("kv4"))) else "bfp8"
        scale = "unit" if m.group("u") else "balanced"
        r_k, r_v, groups_per_chip = _dump_ranks(dump, ranks)
        bytes_lc = groups_per_chip * (r_k * BPE[k_fmt] + r_v * BPE[v_fmt])
        d = json.loads(p.read_text())
        _check_row_fields(row, d, stock, k_fmt, v_fmt, scale)
        tsk = _load(raw, f"tasks_{row}")
        dec, dec_sfx = _frontier_fused_bench(raw, "decode", row)
        mx, mx_sfx = _frontier_fused_bench(raw, "maxctx", row)
        out.append({"row": row, "dump": dump, "r_k_mean": r_k, "r_v_mean": r_v, "k_fmt": k_fmt, "v_fmt": v_fmt,
                    "scale": scale, "bytes_layer_chip": bytes_lc,
                    "kb_token": bytes_lc * LAYERS * CHIPS / 1024, "reduction": STOCK_BYTES_LAYER_CHIP / bytes_lc,
                    "ppl": d["ppl"], "windows": d["windows"], "blind": _row_is_blind(d, fmt != ""),
                    "avg5": _avg5(tsk) if tsk else None,
                    "decode_4k": _min_only(_cell(dec, "4096")) if dec else None, "decode_4k_suffix": dec_sfx,
                    "maxctx": (mx.get("data") or {}).get("max_context_batch32") if mx else None,
                    "maxctx_suffix": mx_sfx})
    return sorted(out, key=lambda r: (r["kb_token"], r["ppl"]))


def _fmt_mean_rank(x):
    """``"33"`` for a whole-number mean rank, ``"33.5"`` otherwise."""
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _stage2f_lines(raw):
    """Stage 2f (Task 4): the rank/format frontier -- ``frontier_rows`` rendered as the spec section
    3 table (row, ranks, formats, scale, bytes, perplexity, tasks, decode, max context).

    The row set is whatever ``frontier_rows`` finds, so an empty or almost-empty raw dir (most of
    stage 2f's own grid, at the time this table was first added) renders a table with only its
    header, or a header plus a one-line note, rather than refusing like the fixed-row stages do.
    """
    rows = frontier_rows(raw)
    lines = [
        "# Stage 2f results: the rank/format frontier",
        "",
        f"Generated by `python -m palu_tt.report --stage 2f` from `docs/results/raw/*.json` at "
        f"commit `{_commit()}`; tt-metal v0.78.0 plus the `palu-sdpa` submodule branch, TP=2 (one "
        "n300, two chips), non-paged, untraced decode. Bytes are per token, per layer, per chip: "
        "`groups_per_chip x (r_k x bpe_k + r_v x bpe_v)`, `bpe` 1.0625 for bfp8 and 0.5625 for bfp4. "
        "`groups_per_chip` is the dump's own latent-group count per chip -- 4 for a per-head dump "
        "(group size 1: `baseline`, `uni_*`, `palu_full_*`/M-LRD, the 8 KV heads split across the "
        "n300's two chips) or 1 for a group-shared dump (G-LRD, group size 4: `glrd_*`, 2 groups per "
        "layer split across the two chips), read from the dump's own `head_wise_ranks` (see "
        f"`_config_groups_per_layer`); stock is {STOCK_BYTES_LAYER_CHIP:.0f} B bfp8 / "
        f"{STOCK_BYTES_LAYER_CHIP * BPE['bfp4'] / BPE['bfp8']:.0f} B bfp4 either way, at "
        f"{STOCK_GROUPS_PER_CHIP} groups/chip. `reduction` is stock bfp8 bytes / this row's bytes. "
        "`scale` is the latent scale (`balanced`, Palu's default split, or `unit`, stage 2f's "
        "rescaled latents). `avg5` is the same five-task zero-shot average as the earlier stages; "
        "`decode 4k (fused)`/`max ctx (fused)` are the fused Phase 2 decode kernel's own decode-step "
        "`min_ms` at 4096 and `bench --what maxctx` at batch 32 -- read from the row's "
        "`bench_<what>_<row>_fk2dm.json` (the shipped-default rotation-kernel variant, stage 2e) when "
        "present, else `bench_<what>_<row>_fk2.json`, and a dash otherwise; the row's own (naive, "
        "reconstruct-then-attend) `bench_<what>_<row>.json` is never shown in these two columns, since "
        "it is multiple times slower than the fused kernel it stands in for on a decode-latency "
        "frontier. All three are rendered only where measured -- most rows have none of them yet.",
        "",
        "| row | dump | r_k | r_v | K fmt | V fmt | scale | B/token/layer/chip | KB/token (model) | "
        "reduction | ppl (windows) | avg5 | decode 4k ms (fused) | max ctx (fused) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| (no stage 2f rows measured yet) |")
    any_blind = False
    for r in rows:
        if r["blind"]:
            any_blind = True
            ppl_cell = f"blind ({r['windows']})"
        else:
            ppl_cell = f"{_fmt(r['ppl'], 3)} ({r['windows']})"
        lines.append(
            f"| {r['row']} | {r['dump']} | {_fmt_mean_rank(r['r_k_mean'])} | {_fmt_mean_rank(r['r_v_mean'])} | "
            f"{r['k_fmt']} | {r['v_fmt']} | {r['scale']} | {r['bytes_layer_chip']:.1f} | "
            f"{r['kb_token']:.2f} | {r['reduction']:.2f}x | {ppl_cell} | "
            f"{r['avg5'] if r['avg5'] is not None else '-'} | "
            f"{r['decode_4k'] if r['decode_4k'] is not None else '-'} | "
            f"{r['maxctx'] if r['maxctx'] is not None else '-'} |"
        )
    lines.append("")
    if any_blind:
        lines += [
            "A `ppl` cell of `blind` is a row measured before the cache-faithful prefill (stage 2f "
            "Task 1b): its perplexity comes from a prefill-only instrument that reconstructs K/V "
            "from the bf16 latent before the forward pass and never touches the stored cache dtype, "
            "so it is identical to the row's bfp8 sibling by construction and is not evidence about "
            "that format -- see stage 2e's L4 section and the `note` field on the row's own "
            "`ppl_*.json`.",
            "",
        ]
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render docs/results/stage{1,2,2b,2c,2e,2f}.md from the raw JSONs.")
    ap.add_argument("--raw", default=RAW, type=Path)
    ap.add_argument("--stage", choices=("1", "2", "2b", "2c", "2e", "2f"), default="1")
    ap.add_argument("--out", default=None, type=Path,
                     help="default: stage1.md / stage2.md / stage2b.md / stage2c.md / stage2e.md / stage2f.md")
    a = ap.parse_args()
    _rows = {"1": ROWS, "2": ROWS2, "2b": ROWS2B, "2c": ROWS2C, "2e": ROWS2E, "2f": ROWS2F}[a.stage]
    _default_out = {"1": OUT, "2": OUT2, "2b": OUT2B, "2c": OUT2C, "2e": OUT2E, "2f": OUT2F}[a.stage]
    print(main(raw=a.raw, out=a.out or _default_out, rows=_rows, stage=a.stage))
