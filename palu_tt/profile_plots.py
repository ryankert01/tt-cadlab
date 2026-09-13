"""Figures for the decode profiling study (stage 2d), from docs/results/raw/profile_<row>_<ctx>.json.

    python -m palu_tt.profile_plots            # writes docs/results/profile_*.png

Figures:
  profile_breakdown.png      one panel per context (1k / 4k / 8k): device time per decode step, stacked by
                             op category, one bar per row
  profile_attention_ctx.png  attention + K-reconstruction device time per step against context, per row
  profile_bridge_4k.png      how the ratio-0.5 rows differ from stock at 4k: stock -> naive (what the
                             reconstruction adds as separate device ops) -> fused (what the kernel removes)
  profile_top_ops_4k.png     the largest op signatures at 4k for stock and the two fused rows
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("docs/results/raw")
OUT_DIR = Path("docs/results")
CONTEXTS = (1024, 4096, 8192)
ROWS = ("baseline", "palu_full_50", "palu_full_50_fk2", "palu_full_70", "palu_full_70_fk2")
ROW_LABEL = {"baseline": "stock", "palu_full_50": "naive 0.5", "palu_full_50_fk2": "fused 0.5",
             "palu_full_70": "naive 0.7", "palu_full_70_fk2": "fused 0.7"}
ROW_COLOR = {"baseline": "#2a78d6", "palu_full_50": "#eb6834", "palu_full_50_fk2": "#1baf7a",
             "palu_full_70": "#b04a1f", "palu_full_70_fk2": "#eda100"}
# stacked-bar order: the interesting parts at the bottom
CATS = ("attention", "k_recon", "mlp", "ccl", "qkv", "wo", "lm_head", "norm", "heads", "cache", "elementwise", "layout", "other")
CAT_COLOR = {"attention": "#5347c9", "k_recon": "#eb6834", "mlp": "#8f99a7", "ccl": "#b8bfca", "qkv": "#6b7684",
             "wo": "#7f8896", "lm_head": "#a1a9b5", "norm": "#c9cfd8", "heads": "#d3d8df", "cache": "#dde1e7",
             "elementwise": "#e6e9ee", "layout": "#eef0f3", "other": "#f4f5f8"}
CAT_LABEL = {"attention": "attention op", "k_recon": "K/V reconstruction (separate device ops)", "mlp": "MLP matmuls",
             "ccl": "cross-chip CCL", "qkv": "QKV matmul", "wo": "wo matmul", "lm_head": "lm_head", "norm": "RMSNorm",
             "heads": "heads / RoPE", "cache": "KV cache update", "elementwise": "elementwise", "layout": "layout", "other": "other"}


def _load(row, ctx):
    p = RAW / f"profile_{row}_{ctx}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cat_ms(prof, cat):
    return prof["categories"].get(cat, {}).get("ms", 0.0) if prof else 0.0


def breakdown_figure(out_png):
    fig, axes = plt.subplots(1, len(CONTEXTS), figsize=(13, 4.6), sharey=False)
    for ax, ctx in zip(axes, CONTEXTS):
        rows = [r for r in ROWS if _load(r, ctx)]
        bottoms = [0.0] * len(rows)
        for cat in CATS:
            vals = [_cat_ms(_load(r, ctx), cat) for r in rows]
            if not any(vals):
                continue
            ax.bar(range(len(rows)), vals, bottom=bottoms, color=CAT_COLOR[cat], edgecolor="white", linewidth=0.5,
                   label=CAT_LABEL[cat] if ctx == CONTEXTS[0] else None)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        for i, (r, tot) in enumerate(zip(rows, bottoms)):
            ax.text(i, tot + 1, f"{tot:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([ROW_LABEL[r] for r in rows], rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{ctx // 1024}k context", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e7eaef", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ms per decode step (device kernel time, chip 0)")
    fig.suptitle("What a decode step is made of, batch 32", fontsize=12)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.legend(loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stock_op_at(ctx, layers=32):
    """32 x the per-call cost of the full-width (out 128) stock SdpaDecode signature, which the naive
    Palu rows still run at 8192 even though the stock row itself cannot allocate there."""
    p = _load("palu_full_50", ctx)
    for o in p["ops"] if p else []:
        if o["op"] == "SdpaDecodeDeviceOperation" and o["out"][-1] == 128:
            return layers * o["ms"] / o["count"]
    return None


def attention_ctx_figure(out_png):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in ROWS:
        xs, ys = [], []
        for ctx in CONTEXTS:
            p = _load(r, ctx)
            if p:
                xs.append(ctx // 1024)
                ys.append(_cat_ms(p, "attention") + _cat_ms(p, "k_recon"))
        if xs:
            ax.plot(xs, ys, marker="o", color=ROW_COLOR[r], label=ROW_LABEL[r], linewidth=2)
            # within each near-coincident pair the upper line labels above and the lower below, so no two
            # labels (nor stock's, nor the dotted stock-op point) land on the same spot
            dy = {"baseline": -15, "palu_full_50": 8, "palu_full_70": -15,
                  "palu_full_50_fk2": -15, "palu_full_70_fk2": 8}[r]
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, dy), fontsize=8,
                            color=ROW_COLOR[r], ha="center")
    stock_8k = _stock_op_at(8192)
    if stock_8k:
        st4 = _cat_ms(_load("baseline", 4096), "attention")
        ax.plot([4, 8], [st4, stock_8k], color=ROW_COLOR["baseline"], linewidth=1.5, linestyle=":")
        ax.plot([8], [stock_8k], marker="o", markerfacecolor="white", color=ROW_COLOR["baseline"])
        ax.annotate(f"{stock_8k:.1f}", (8, stock_8k), textcoords="offset points", xytext=(0, -15), fontsize=8,
                    color=ROW_COLOR["baseline"], ha="center")
        ax.plot([], [], color=ROW_COLOR["baseline"], linestyle=":", label="stock op at 8k (measured\ninside the naive rows; stock\nitself does not allocate)")
    ax.set_xticks([c // 1024 for c in CONTEXTS])
    ax.set_xticklabels([f"{c // 1024}k" for c in CONTEXTS])
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50, 100, 200, 500])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("context")
    ax.set_ylabel("attention + K/V reconstruction, ms per step (log)")
    ax.set_title("The part of the step that depends on context", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e7eaef", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def bridge_figure(out_png, ctx=4096, ratio="50"):
    """Waterfall: stock -> naive (adds) -> fused (removes), by category, at one context."""
    st, nv, fu = _load("baseline", ctx), _load(f"palu_full_{ratio}", ctx), _load(f"palu_full_{ratio}_fk2", ctx)
    if not (st and nv and fu):
        return
    steps = [("stock", st["device_ms_per_step"], None)]
    cats_show = ("k_recon", "attention", "cache", "layout", "heads", "ccl")
    run = st["device_ms_per_step"]
    for cat in cats_show:
        d = _cat_ms(nv, cat) - _cat_ms(st, cat)
        if abs(d) > 0.05:
            steps.append((f"+ {CAT_LABEL[cat]}", d, run)); run += d
    rest = nv["device_ms_per_step"] - run
    if abs(rest) > 0.05:
        steps.append(("+ everything else", rest, run)); run += rest
    steps.append((f"naive {ratio}", nv["device_ms_per_step"], None))
    run = nv["device_ms_per_step"]
    for cat in cats_show:
        d = _cat_ms(fu, cat) - _cat_ms(nv, cat)
        if abs(d) > 0.05:
            steps.append((f"fused: {CAT_LABEL[cat]}", d, run)); run += d
    rest = fu["device_ms_per_step"] - run
    if abs(rest) > 0.05:
        steps.append(("fused: everything else", rest, run)); run += rest
    steps.append((f"fused {ratio}", fu["device_ms_per_step"], None))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (label, val, base) in enumerate(steps):
        if base is None:
            ax.bar(i, val, color="#5347c9" if "fused" in label else ("#2a78d6" if label == "stock" else "#eb6834"), width=0.7)
            ax.text(i, val + 1, f"{val:.1f}", ha="center", fontsize=9)
        else:
            color = "#c9433a" if val > 0 else "#1e7443"
            ax.bar(i, val, bottom=base, color=color, width=0.7)
            ax.text(i, base + max(val, 0) + 1, f"{val:+.1f}", ha="center", fontsize=8, color=color)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ms per decode step")
    ax.set_ylim(0, max(v if b is None else b + max(v, 0) for _, v, b in steps) * 1.14)
    ax.set_title(f"From stock to naive to fused, ratio 0.{ratio}, {ctx // 1024}k context, batch 32", fontsize=11, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e7eaef", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _op_note(o):
    """Second label line: category, plus the ranks for an attention op (that is what varies between layers)."""
    if o["category"] != "attention":
        return o["category"]
    r_k = o["in1"][-1] if o["op"].startswith("Palu") else 128
    return f"r_k {r_k} / r_v {o['out'][-1]}"


def top_ops_figure(out_png, ctx=4096, rows=("baseline", "palu_full_50_fk2", "palu_full_70_fk2"), n=10):
    profs = [(r, _load(r, ctx)) for r in rows]
    profs = [(r, p) for r, p in profs if p]
    if not profs:
        return
    fig, axes = plt.subplots(1, len(profs), figsize=(4.6 * len(profs), 4.8), sharex=True)
    if len(profs) == 1:
        axes = [axes]
    for ax, (r, p) in zip(axes, profs):
        ax.set_xlim(0, max(o["ms"] for _, q in profs for o in q["ops"][:n]) * 1.30)
        ops = p["ops"][:n]
        labels = [f"{o['op'].replace('DeviceOperation', '')} x{o['count']:.0f}\n{_op_note(o)}" for o in ops]
        ax.barh(range(len(ops))[::-1], [o["ms"] for o in ops], color=[CAT_COLOR.get(o["category"], "#999") for o in ops],
                edgecolor="#d8dce3", linewidth=0.5)
        for y, o in zip(range(len(ops))[::-1], ops):
            ax.text(o["ms"] + 0.3, y, f"{o['ms'] / max(o['count'], 1e-9):.2f} ms/call", va="center", fontsize=6.5,
                    color="#55606e")
        ax.set_yticks(range(len(ops))[::-1])
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{ROW_LABEL[r]}: {p['device_ms_per_step']:.1f} ms/step", fontsize=10)
        ax.set_xlabel("ms per step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e7eaef", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"Largest op signatures per decode step, {ctx // 1024}k context (count = calls per step)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    breakdown_figure(OUT_DIR / "profile_breakdown.png")
    attention_ctx_figure(OUT_DIR / "profile_attention_ctx.png")
    bridge_figure(OUT_DIR / "profile_bridge_4k.png")
    bridge_figure(OUT_DIR / "profile_bridge_4k_70.png", ratio="70")
    top_ops_figure(OUT_DIR / "profile_top_ops_4k.png")
    print("wrote", sorted(str(p.name) for p in OUT_DIR.glob("profile_*.png")))


if __name__ == "__main__":
    main()
