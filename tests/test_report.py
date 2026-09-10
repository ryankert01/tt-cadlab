"""CPU test for the results-table renderer: it must read the real JSON schema.

The schema exercised here is the one ``palu_tt.bench``/``palu_tt.eval_ppl`` actually write:
string context keys (``json.dumps`` stringifies the int keys) and error cells for the
contexts whose model does not allocate. The rank column has two producers -- our own
``compress.py`` dumps (a ``palu_tt`` block) and Palu-recipe dumps (``head_wise_ranks`` only) --
and must never fall back to inventing full rank for a dump it cannot read.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

from palu_tt import report

NKV, HD, DIM = 2, 128, 64


def _write(raw, name, obj):
    (raw / f"{name}.json").write_text(json.dumps(obj))


def _palu_recipe_dir(path, per_layer):
    """A dump in Palu's own layout: ``head_wise_ranks`` and no ``palu_tt`` block.

    ``per_layer`` is a list of ``(r_k, r_v)``, one per layer, so the fixture can be non-uniform
    the way the calibrated dumps are.
    """
    path.mkdir(parents=True, exist_ok=True)
    tensors, ranks = {}, {}
    for i, (r_k, r_v) in enumerate(per_layer):
        key = f"model.layers.{i}.self_attn."
        for proj, r in (("k_proj", r_k), ("v_proj", r_v)):
            tensors[key + proj + ".VT.weight"] = torch.randn(NKV * r, DIM)
            for h in range(NKV):
                tensors[key + proj + f".U.{h}.weight"] = torch.randn(HD, r)
            ranks[key + proj] = [r] * NKV
    save_file(tensors, str(path / "palu_attn.safetensors"))
    (path / "config.json").write_text(json.dumps({"head_wise_ranks": ranks, "head_dim": HD}))
    return path


def test_main_renders_cells_errors_and_missing_rows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "ppl_baseline", {"ppl": 7.25, "windows": 141, "seq_len": 2048,
                                 "row": "baseline", "kind": "stock", "palu_dir": None})
    _write(raw, "ppl_cpu_baseline", {"ppl": 7.36, "windows": 20, "seq_len": 2048,
                                     "row": "baseline", "kind": "stock", "palu_dir": None})
    _write(raw, "bench_decode_baseline", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode",
        "contexts": [1024, 4096, 8192],
        "data": {
            "1024": {"p50_ms": 12.34, "min_ms": 10.0, "iters": 10,
                     "kv_bytes_per_token_per_layer_per_chip": 1088.0},
            "4096": {"p50_ms": 45.0, "min_ms": 44.0, "iters": 10,
                     "kv_bytes_per_token_per_layer_per_chip": 1088.0},
            "8192": {"error": "TT_FATAL ... Out of Memory: Not enough space to allocate 142606336",
                     "phase": "build", "exc": "RuntimeError"},
        },
    })
    out = tmp_path / "stage1.md"

    md = report.main(raw=raw, out=out)

    assert out.read_text() == md
    # quality/memory table: label, ranks default to head_dim, ppl and both KV columns rendered
    assert "| baseline | stock `tt_transformers` attention | 128 | 128 | 7.250 | 7.360 |" in md
    assert "| 1088 |" in md
    assert f"| {1088 * report.LAYERS * report.CHIPS} |" in md  # 69632 B/token for the whole model
    # window counts are read from the JSON, not hard-coded
    assert "ppl (device, 141 windows)" in md and "ppl (CPU fp32, 20 windows)" in md
    # the ratio column is named for what it is
    assert "| KV reduction |" in md and "| 1.00x |" in md
    # timing table: p50 and min side by side, and the OOM cell rendered, not crashed on
    assert "| baseline | 1024 | 12.3 / 10.0 |" in md
    assert "| baseline | 8192 | OOM |" in md
    # both calibrated rows are in the table
    for row in ("palu_full_50", "palu_full_70"):
        assert any(line.startswith(f"| {row} |") for line in md.splitlines())
    # rows with no files at all still appear, as dashes
    assert any(line.startswith("| palu_full_50 |") and "| - |" in line for line in md.splitlines())
    # missing files never raise and never invent numbers (not even the default ranks)
    assert "| identity | Palu path, identity factorisation (r = 128) | - | - | - | - |" in md


def test_window_counts_come_from_the_json(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "ppl_baseline", {"ppl": 7.25, "windows": 7, "row": "baseline", "palu_dir": None})
    md = report.main(raw=raw, out=None)
    assert "ppl (device, 7 windows)" in md
    assert "141" not in md
    # no perplexity JSON at all for the CPU column -> no number is asserted
    assert "ppl (CPU fp32, ? windows)" in md


def test_ranks_from_head_wise_ranks_when_the_dump_has_no_palu_tt_block(tmp_path):
    """A Palu-recipe dump carries per-layer ranks only; report them as a range plus a mean."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pd = _palu_recipe_dir(tmp_path / "palu_full_0.5", [(32, 128), (32, 64), (64, 64)])
    _write(raw, "ppl_palu_full_50", {"ppl": 12.93, "windows": 141, "row": "palu_full_50",
                                     "kind": "palu", "palu_dir": str(pd)})
    _write(raw, "bench_maxctx_palu_full_50", {
        "row": "palu_full_50", "kind": "palu", "palu_dir": str(pd), "what": "maxctx",
        "data": {"max_context_batch32": 8192, "kv_bytes_per_token_per_layer_per_chip": 680.0},
    })

    md = report.main(raw=raw, out=None)

    line = next(l for l in md.splitlines() if l.startswith("| palu_full_50 |"))
    assert "| 32–64 (mean 43) | 64–128 (mean 85) |" in line, line
    # the KV cell is the model mean, not layer 0's 680 (layer 0 sums to 160, the mean to 128)
    assert "| 544 |" in line, line


def test_an_unreadable_palu_dump_renders_dashes_not_full_rank(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "ppl_svd_r64", {"ppl": 703.6, "windows": 141, "row": "svd_r64",
                                "kind": "palu", "palu_dir": str(tmp_path / "gone")})

    md = report.main(raw=raw, out=None)

    line = next(l for l in md.splitlines() if l.startswith("| svd_r64 |"))
    assert "| - | - | 703.600 |" in line, line
    assert "128" not in line, line


def test_hand_written_tail_survives_a_re_render(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "ppl_baseline", {"ppl": 7.25, "row": "baseline", "kind": "stock", "palu_dir": None})
    out = tmp_path / "stage1.md"

    report.main(raw=raw, out=out)
    out.write_text(out.read_text() + "\n## Notes\n\nHand-written paragraph.\n")
    _write(raw, "ppl_baseline", {"ppl": 7.30, "row": "baseline", "kind": "stock", "palu_dir": None})
    md = report.main(raw=raw, out=out)

    assert "Hand-written paragraph." in md
    assert "7.300" in md and "7.250" not in md
    assert md.count(report.MARKER) == 1


def test_main_refuses_to_render_when_no_row_has_any_data(tmp_path):
    """A fresh clone (or a mistyped ``--raw``) must not blank the committed tables."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "stage1.md"
    out.write_text("| baseline | 7.254 |\n")

    with pytest.raises(RuntimeError, match="no results found"):
        report.main(raw=raw, out=out)

    assert out.read_text() == "| baseline | 7.254 |\n"  # untouched
