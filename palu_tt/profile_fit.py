"""Least-squares fit of per-layer attention-op time as fixed + chunks x per_chunk (stage 2d section 4).

Reads the profile JSONs ``docs/results/raw/profile_<row>_<ctx>.json`` written by ``palu_tt.profile_report``
and fits every attention-category op *signature* (op name, K width = in1's last dim, output width) over
the contexts given. A signature can be fed by several rows: the stock op at 8k only exists inside the
naive Palu rows (a whole stock step does not allocate there), and the fused rows carry both r_k 32 and
r_k 64 layers of the same model.

    python -m palu_tt.profile_fit --row baseline --row palu_full_50 --row palu_full_50_fk2 --contexts 1024 4096 8192
"""
import argparse
import json
from pathlib import Path

RAW = Path("docs/results/raw")


def _signature(op):
    return f"{op['op']} r_k={op['in1'][-1]} out={op['out'][-1]}"


def _lstsq(points):
    """points: [(x, y), ...] -> (intercept, slope) by ordinary least squares."""
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return intercept, slope


def fit_signatures(raw, rows, contexts, chunk=512):
    per_sig = {}
    for row in rows:
        for ctx in contexts:
            p = Path(raw) / f"profile_{row}_{ctx}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for op in d["ops"]:
                if op["category"] != "attention" or op["count"] <= 0:
                    continue
                sig = per_sig.setdefault(_signature(op), {"signature": _signature(op), "rows": [], "points": []})
                if row not in sig["rows"]:
                    sig["rows"].append(row)
                sig["points"].append((ctx // chunk, 1000.0 * op["ms"] / op["count"]))   # us per layer
    out = []
    for sig in per_sig.values():
        sig["points"].sort()
        if len({x for x, _ in sig["points"]}) < 2:
            continue
        fixed, per_chunk = _lstsq(sig["points"])
        sig.update(fixed_us=fixed, per_chunk_us=per_chunk, per_item_per_chunk_us=per_chunk / 2)
        out.append(sig)
    return sorted(out, key=lambda s: s["signature"])


def table(rows):
    lines = ["| signature | rows | points (chunks: us/layer) | fixed us | per chunk us | per item per chunk us |",
             "|---|---|---|---|---|---|"]
    for s in rows:
        pts = ", ".join(f"{x}: {y:.0f}" for x, y in s["points"])
        lines.append(f"| {s['signature']} | {', '.join(s['rows'])} | {pts} | {s['fixed_us']:.1f} | "
                     f"{s['per_chunk_us']:.1f} | {s['per_item_per_chunk_us']:.1f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--row", action="append", required=True)
    p.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 8192])
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--json", type=Path, default=None, help="also write the fitted rows to this JSON file")
    a = p.parse_args()
    rows = fit_signatures(RAW, a.row, a.contexts, a.chunk)
    print(table(rows))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
