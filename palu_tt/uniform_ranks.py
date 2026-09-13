"""Uniform per-group rank lists in the format Palu's compress.py consumes through --ranks_json (stage 2f)."""
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
