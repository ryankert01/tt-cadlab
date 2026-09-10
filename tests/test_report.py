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


def test_stage2_renders_fused_rows_with_fidelity_and_vs_stock(tmp_path):
    """A fused row: own timings + decode-check, inherited perplexity, and the stock ratio.

    The fused kernel changes the decode path only, so ``ppl_palu_full_50_fk2.json`` deliberately
    does not exist -- the row must show the base row's perplexity rather than a dash, and the
    ``vs stock`` cell must be the stock decode minimum over this row's.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    pd = _palu_recipe_dir(tmp_path / "palu_full_0.5", [(32, 128), (32, 64), (64, 64)])
    _write(raw, "ppl_baseline", {"ppl": 7.254, "windows": 141, "row": "baseline",
                                 "kind": "stock", "palu_dir": None})
    _write(raw, "bench_decode_baseline", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode",
        "data": {"1024": {"p50_ms": 38.1, "min_ms": 38.0, "iters": 10,
                          "kv_bytes_per_token_per_layer_per_chip": 1088.0},
                 "8192": {"error": "TT_FATAL ... Out of Memory: Not enough space", "phase": "build",
                          "exc": "RuntimeError"}},
    })
    # the base Palu row: perplexity and max context live here and are never re-measured for _fk*
    _write(raw, "ppl_palu_full_50", {"ppl": 12.931, "windows": 141, "row": "palu_full_50",
                                     "kind": "palu", "palu_dir": str(pd)})
    _write(raw, "bench_maxctx_palu_full_50", {
        "row": "palu_full_50", "kind": "palu", "palu_dir": str(pd), "what": "maxctx",
        "data": {"max_context_batch32": 8192, "kv_bytes_per_token_per_layer_per_chip": 680.0},
    })
    # the fused row: its own decode/attention timings and its own fidelity check
    _write(raw, "bench_decode_palu_full_50_fk2", {
        "row": "palu_full_50_fk2", "kind": "palu", "palu_dir": str(pd), "what": "decode",
        "fused": "p2", "fused_k_chunk": None,
        "data": {"1024": {"p50_ms": 15.2, "min_ms": 15.0, "iters": 10,
                          "kv_bytes_per_token_per_layer_per_chip": 680.0},
                 "8192": {"p50_ms": 40.4, "min_ms": 40.0, "iters": 10,
                          "kv_bytes_per_token_per_layer_per_chip": 680.0}},
    })
    _write(raw, "bench_attention_palu_full_50_fk2", {
        "row": "palu_full_50_fk2", "kind": "palu", "palu_dir": str(pd), "what": "attention",
        "fused": "p2", "fused_k_chunk": None,
        "data": {"1024": {"p50_ms": 0.9, "min_ms": 0.8, "iters": 20,
                          "kv_bytes_per_token_per_layer_per_chip": 680.0}},
    })
    _write(raw, "decode_check_palu_full_50_fk2", {
        "row": "palu_full_50_fk2", "palu_dir": str(pd), "fused": "p2", "fused_k_chunk": None,
        "steps": 64, "pcc_mean": 0.99884, "pcc_min": 0.99712, "argmax_agree": 0.984,
    })
    out = tmp_path / "stage2.md"

    md = report.main(raw=raw, out=out, rows=report.ROWS2)

    assert out.read_text() == md
    assert md.startswith("# Stage 2 results: fused K-reconstruction decode kernel on a Wormhole n300")
    assert "decode PCC (min / mean, 64 steps)" in md and "| argmax agree |" in md
    quality = next(l for l in md.splitlines() if l.startswith("| palu_full_50_fk2 | ") and "0.9971" in l)
    # inherited from the base row (no ppl_palu_full_50_fk2.json exists), plus the fidelity columns
    assert "| 12.931 |" in quality, quality
    assert "| 0.9971 / 0.9988 | 0.984 |" in quality, quality
    assert "| 8192 |" in quality, quality                       # max context inherited too
    # the naive base row has no decode-check JSON -> dashes, never invented numbers
    base = next(l for l in md.splitlines() if l.startswith("| palu_full_50 | "))
    assert "| - | - |" in base, base
    # latency table: stock min 38.0 / row min 15.0
    assert "| palu_full_50_fk2 | 1024 | 15.2 / 15.0 | 0.9 / 0.8 | 2.53x |" in md
    # 8192: stock is OOM, so there is no ratio to quote
    assert "| palu_full_50_fk2 | 8192 | 40.4 / 40.0 | - | - |" in md
    # the stage-1 samples block is not repeated in stage 2
    assert "## Generation samples" not in md


def test_stage2_falls_back_to_the_base_row_for_prefill_and_leaves_stage1_alone(tmp_path):
    """``_base_row`` maps an ``_fk*`` row to its base and a stage-1 row to itself."""
    assert report._base_row("palu_full_50_fk2") == "palu_full_50"
    assert report._base_row("svd_r64_fk1") == "svd_r64"
    assert report._base_row("baseline") == "baseline"

    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "bench_prefill_svd_r64", {
        "row": "svd_r64", "kind": "palu", "palu_dir": None, "what": "prefill",
        "data": {"2048": {"p50_ms": 443.3, "min_ms": 442.3, "iters": 5}},
    })
    md = report.main(raw=raw, out=None, rows=report.ROWS2)
    assert "| svd_r64_fk1 | 443.3 / 442.3 | - |" in md


def test_batch_sweep_table_renders_present_and_missing_cells(tmp_path):
    """Task 7: the sweep table reads ``bench_decode_<row>_b<B>.json``; batch 32 has no suffix."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "ppl_baseline", {"ppl": 7.25, "windows": 141, "row": "baseline",
                                 "kind": "stock", "palu_dir": None})
    _write(raw, "bench_decode_baseline", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 32,
        "data": {"1024": {"p50_ms": 12.0, "min_ms": 10.0, "iters": 10}},
    })
    _write(raw, "bench_decode_baseline_b4", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 4,
        "data": {"1024": {"p50_ms": 41.0, "min_ms": 40.0, "iters": 10}},
    })
    out = tmp_path / "stage2.md"

    md = report.main(raw=raw, out=out, rows=report.ROWS2)

    assert out.read_text() == md
    assert "## Decode step by batch size" in md
    # b=4 -> 40.0 ms, 4 * 1000 / 40.0 = 100 tok/s; b=1/16 have no file -> "-"; b=32 is read
    # from the unsuffixed file (the same one the latency table uses)
    assert "| baseline | - | 40.0 (100) | - | 10.0 (3200) |" in md


def test_sweep_cell_and_display_distinguish_missing_from_errored(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "bench_decode_baseline_b1", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 1,
        "data": {"8192": {"error": "TT_FATAL ... Out of Memory: Not enough space",
                          "phase": "build", "exc": "RuntimeError"}},
    })

    assert report._sweep_cell(raw, "baseline", 1, "8192") is None  # errored -> None
    assert report._sweep_cell(raw, "baseline", 1, "1024") is None  # missing context -> None
    assert report._sweep_cell(raw, "baseline", 32, "1024") is None  # missing file entirely -> None

    assert report._sweep_display(raw, "baseline", 1, "8192") == "OOM"
    assert report._sweep_display(raw, "baseline", 1, "1024") == "-"


def test_context_scaling_table_renders_maxctx_and_an_oom_cell(tmp_path):
    """Task 8: the batch-1/batch-8 tables read ``bench_decode_<row>_b<B>.json`` for the per-context
    cells and ``bench_maxctx_<row>_b<B>.json`` for the max-context/limit columns."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "bench_decode_baseline", {   # batch 32, unsuffixed: satisfies main()'s any-data guard
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 32,
        "data": {"1024": {"p50_ms": 38.0, "min_ms": 38.0, "iters": 10}},
    })
    _write(raw, "bench_decode_baseline_b1", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 1,
        "data": {"1024": {"p50_ms": 31.3, "min_ms": 31.0, "iters": 10},
                 "2048": {"p50_ms": 31.8, "min_ms": 31.5, "iters": 10},
                 "4096": {"error": "TT_FATAL ... Out of Memory: Not enough space",
                          "phase": "build", "exc": "RuntimeError"}},
    })
    _write(raw, "bench_maxctx_baseline_b1", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "maxctx", "batch": 1,
        "data": {"max_context": 3072, "batch": 1, "limit": "oom", "failed_at": 4096,
                 "error": "TT_FATAL ... Out of Memory: Not enough space", "phase": "build",
                 "exc": "RuntimeError", "kv_bytes_per_token_per_layer_per_chip": 1088.0,
                 "probes": [[1024, True], [2048, True], [4096, False], [3072, True]]},
    })
    out = tmp_path / "stage2.md"

    md = report.main(raw=raw, out=out, rows=report.ROWS2)

    assert out.read_text() == md
    assert "## Context scaling: one request (batch 1) and a small serving batch (batch 8)" in md
    assert "### Batch 1" in md and "### Batch 8" in md
    assert "| baseline | 3072 | oom | 31.0 | 31.5 | OOM | - | - | - | - | - |" in md
    # batch 8 was never run in this fixture: every cell dashes, no max context
    assert "| baseline | - | - | - | - | - | - | - | - | - | - |" in md
    assert "![Decode step vs context, batch 1](ctx_scaling_b1.png)" in md
    assert "![Decode step vs context, batch 8](ctx_scaling_b8.png)" in md


def test_scaling_helpers_distinguish_missing_from_errored_and_read_maxctx(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "bench_decode_baseline_b1", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 1,
        "data": {"4096": {"error": "TT_FATAL ... Out of Memory: Not enough space",
                          "phase": "build", "exc": "RuntimeError"}},
    })
    _write(raw, "bench_maxctx_baseline_b1", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "maxctx", "batch": 1,
        "data": {"max_context": 131072, "batch": 1, "limit": "model", "failed_at": None,
                 "error": None, "phase": None, "exc": None,
                 "kv_bytes_per_token_per_layer_per_chip": 42.0, "probes": []},
    })

    assert report._scaling_decode_cell(raw, "baseline", 1, "4096") is None       # errored
    assert report._scaling_decode_cell(raw, "baseline", 1, "1024") is None       # missing context
    assert report._scaling_decode_cell(raw, "baseline", 8, "4096") is None       # missing file
    assert report._scaling_decode_display(raw, "baseline", 1, "4096") == "OOM"
    assert report._scaling_decode_display(raw, "baseline", 1, "1024") == "-"
    assert report._scaling_maxctx(raw, "baseline", 1) == (131072, "model")
    assert report._scaling_maxctx(raw, "baseline", 8) == ("-", "-")
