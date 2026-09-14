"""Stage 2f dump plumbing for Palu's compress.py: uniform per-group rank lists in the format it
consumes through --ranks_json (Grid B's ``uni_rk*_rv*`` dumps), and the naming of the
split-budget Fisher dumps (``split_k*_v*``) that ``scripts/run_calibration_split.sh`` produces.
"""
import argparse
import json
from pathlib import Path


def write(path, n_layers, groups_per_matrix, r_k, r_v):
    d = {}
    for i in range(n_layers):
        d[f"model.layers.{i}.self_attn.k_proj"] = [int(r_k)] * groups_per_matrix
        d[f"model.layers.{i}.self_attn.v_proj"] = [int(r_v)] * groups_per_matrix
    Path(path).write_text(json.dumps(d, indent=1))
    return d


def split_dump_name(k_ratio, v_ratio):
    """``split_k<K>_v<V>`` for a stage 2f split-budget dump: the two per-matrix Fisher targets as
    whole percentages (0.25, 0.75 -> ``"split_k25_v75"``), which is what ``report.FRONTIER_RE``
    parses and what the row names are built from.

    A ratio that is not a whole percentage is rejected rather than rounded: two dumps that rounded
    to the same name would share both a ``compressed/`` directory and -- through
    ``convert.palu_dir_cache_id`` -- a device weight-cache key, and the second one would silently
    read the first one's factors.
    """
    pcts = []
    for label, ratio in (("k", k_ratio), ("v", v_ratio)):
        pct = float(ratio) * 100
        if not 0 < float(ratio) <= 1 or abs(pct - round(pct)) > 1e-9:
            raise ValueError(f"{label} ratio must be a whole percentage in (0, 1], got {ratio!r}")
        pcts.append(round(pct))
    return f"split_k{pcts[0]}_v{pcts[1]}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--groups", type=int, default=8, help="latent groups per matrix: n_kv_heads / group size")
    p.add_argument("--r-k", type=int, required=True)
    p.add_argument("--r-v", type=int, required=True)
    a = p.parse_args()
    write(a.out, a.layers, a.groups, a.r_k, a.r_v)


if __name__ == "__main__":
    main()
