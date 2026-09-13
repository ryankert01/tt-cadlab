"""Context-scaling figures for the stage 2 report (Task 8): decode step time vs context, one line
per row, at a fixed batch size, from the ``bench_decode_<row>_b<B>.json`` / ``bench_maxctx_<row>_
b<B>.json`` files ``palu_tt.bench`` writes.

Colours are fixed per row, in the categorical order the project's data-viz palette validates for
an adjacent-pairlist chart (a line chart with a legend, not scatter/all-pairs): stock = slot 1
(blue), naive = slot 2 (orange), fused ratio 0.5 = slot 3 (aqua), fused ratio 0.7 = slot 4
(yellow) -- the same four hues used unmodified from the palette's documented default order.

Usage: ``python -m palu_tt.plots --batch 1 --batch 8`` writes both PNGs into ``docs/results/``.

``--frontier`` renders ``docs/results/frontier_2f.png`` instead: the stage 2f rank/format frontier
(``frontier_figure``), wikitext perplexity against KV bytes per token for every cell
``report.frontier_rows`` finds in the raw dir, marker/fill encoding K/V cache format and colour the
dump family, with the Pareto front drawn through the points. A row whose perplexity predates the
cache-faithful prefill (``frontier_rows``' ``blind`` flag) is left off the figure entirely -- see
``frontier_figure``'s own docstring.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on this host
import matplotlib.pyplot as plt

from palu_tt.report import SCALING_BATCHES, SCALING_CONTEXTS, SCALING_ROWS

RAW = Path("docs/results/raw")
OUT_DIR = Path("docs/results")

ROW_COLOR = {
    "baseline": "#2a78d6",           # slot 1: blue
    "palu_full_50": "#eb6834",       # slot 2: orange
    "palu_full_50_fk2": "#1baf7a",   # slot 3: aqua
    "palu_full_70_fk2": "#eda100",   # slot 4: yellow
}
ROW_LABEL = {
    "baseline": "stock",
    "palu_full_50": "naive (ratio 0.5)",
    "palu_full_50_fk2": "fused ratio 0.5",
    "palu_full_70_fk2": "fused ratio 0.7",
}


def _load(raw, name):
    p = Path(raw) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fname(what, row, batch):
    """Mirrors ``bench._out_name`` / ``report._scaling_filename``: unsuffixed at batch 32."""
    return f"bench_{what}_{row}" if batch == 32 else f"bench_{what}_{row}_b{batch}"


def ctx_scaling_figure(raw_dir, batch, out_png, rows=SCALING_ROWS):
    """Decode step (min ms) vs context (log2 x-axis) for ``rows`` at ``batch``, one line each.

    The last plotted point of each line is annotated with its ``bench_maxctx`` max context and
    whether that run ended in OOM or at the model limit. A row with no decode JSON (or none of
    its cells timed) is skipped rather than drawn as an empty line.
    """
    raw_dir = Path(raw_dir)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    annotations = []  # (y_last, log_x_last, note, color) -- stacked after the loop to avoid overlap
    for row in rows:
        dec = _load(raw_dir, _fname("decode", row, batch))
        data = (dec or {}).get("data") or {}
        points = sorted(
            ((int(ctx), cell["min_ms"]) for ctx, cell in data.items() if "min_ms" in cell),
            key=lambda p: p[0],
        )
        if not points:
            continue
        xs, ys = zip(*points)
        log_xs = [math.log2(x) for x in xs]
        color = ROW_COLOR.get(row, "#4a3aa7")
        label = ROW_LABEL.get(row, row)
        ax.plot(log_xs, ys, marker="o", markersize=6, linewidth=2, color=color, label=label)

        mx = (_load(raw_dir, _fname("maxctx", row, batch)) or {}).get("data") or {}
        max_ctx, limit = mx.get("max_context"), mx.get("limit")
        note = f"max {max_ctx // 1024}k" if max_ctx else "max ?"
        if limit == "oom":
            note += " (OOM)"
        elif limit == "model":
            note += " (model limit)"
        annotations.append((ys[-1], log_xs[-1], note, color))

    # Two lines can end almost on top of each other (e.g. the two fused rows hitting the same
    # model limit); stack their labels by rank rather than at an identical offset so they never
    # overlap, regardless of how close the underlying values are.
    for i, (y_last, log_x_last, note, color) in enumerate(sorted(annotations, reverse=True)):
        ax.annotate(note, (log_x_last, y_last), textcoords="offset points", xytext=(8, 4 + 14 * i),
                    fontsize=9, color=color)

    ax.set_xlabel("Context (tokens)")
    ax.set_ylabel("Decode step, min ms")
    ax.set_title(f"Decode step vs context, batch {batch}")
    ticks = [math.log2(c) for c in SCALING_CONTEXTS]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{c // 1024}k" for c in SCALING_CONTEXTS])
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", frameon=False)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


FRONTIER_COLOR = {"baseline": "#333333", "palu_full_50": "#2f6fdb", "palu_full_70": "#7fa8f2",
                   "glrd_50_f256": "#e07b39", "glrd_70_f256": "#f0a06a"}


def frontier_figure(raw_dir, out_png, ranks=None):
    """Perplexity against KB per token for every stage 2f cell (``report.frontier_rows``): marker
    shape is K format, fill is V format, colour is dump family, with the Pareto front (lowest
    perplexity at each byte budget) drawn through the points.

    A row measured before the cache-faithful prefill (``report.frontier_rows``' ``blind`` flag --
    see its docstring) is excluded: its perplexity is its bfp8 sibling's by construction, so it would
    otherwise plot as a bfp4 point at bfp8-level quality, which is exactly the kind of misleading
    reading the frontier table works to avoid.
    """
    from palu_tt import report
    rows = [r for r in report.frontier_rows(raw_dir, ranks=ranks) if not r["blind"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        c = FRONTIER_COLOR.get(r["dump"], "#3c9d6b")
        marker = "s" if r["k_fmt"] == "bfp4" else "o"
        face = c if r["v_fmt"] == "bfp8" else "white"
        ax.scatter(r["kb_token"], r["ppl"], marker=marker, s=48, facecolors=face, edgecolors=c, linewidths=1.5)
        ax.annotate(r["row"], (r["kb_token"], r["ppl"]), textcoords="offset points", xytext=(4, 3), fontsize=6)
    front, best = [], float("inf")
    for r in sorted(rows, key=lambda r: r["kb_token"]):
        if r["ppl"] < best:
            best = r["ppl"]
            front.append(r)
    if len(front) > 1:
        ax.plot([r["kb_token"] for r in front], [r["ppl"] for r in front], color="#999999", linewidth=1,
                 linestyle="--")
    ax.set_xlabel("KV cache, KB per token (whole model, both chips)")
    ax.set_ylabel("wikitext-2 perplexity")
    ax.set_title("quality per byte: rank x format (filled = bfp8 V, hollow = bfp4 V; square = bfp4 K)")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Render docs/results/ctx_scaling_b<B>.png / frontier_2f.png "
                                              "from the raw JSONs.")
    ap.add_argument("--batch", type=int, action="append", default=None,
                     help="repeatable; default: both %s" % (SCALING_BATCHES,))
    ap.add_argument("--frontier", action="store_true",
                     help="render docs/results/frontier_2f.png (stage 2f) instead of the context-scaling figures")
    ap.add_argument("--raw", default=RAW, type=Path)
    ap.add_argument("--out-dir", default=OUT_DIR, type=Path)
    a = ap.parse_args()
    if a.frontier:
        print(frontier_figure(a.raw, a.out_dir / "frontier_2f.png"))
        return
    for b in (a.batch or list(SCALING_BATCHES)):
        out = ctx_scaling_figure(a.raw, b, a.out_dir / f"ctx_scaling_b{b}.png")
        print(out)


if __name__ == "__main__":
    main()
