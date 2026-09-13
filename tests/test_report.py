"""CPU test for the results-table renderer: it must read the real JSON schema.

The schema exercised here is the one ``palu_tt.bench``/``palu_tt.eval_ppl`` actually write:
string context keys (``json.dumps`` stringifies the int keys) and error cells for the
contexts whose model does not allocate. The rank column has two producers -- our own
``compress.py`` dumps (a ``palu_tt`` block) and Palu-recipe dumps (``head_wise_ranks`` only) --
and must never fall back to inventing full rank for a dump it cannot read.
"""
import json
from pathlib import Path

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


def test_stage2b_accuracy_table_and_paper_block(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    _write(raw, "tasks_baseline", {"row": "baseline", "tasks": {"openbookqa": {"acc": 0.43, "acc_norm": 0.45, "n": 500},
                                                                "piqa": {"acc": 0.78, "acc_norm": 0.79, "n": 1838}}})
    _write(raw, "tasks_glrd_50", {"row": "glrd_50", "tasks": {"openbookqa": {"acc": 0.41, "acc_norm": 0.42, "n": 500}}})
    _write(raw, "ppl_glrd_50", {"row": "glrd_50", "ppl": 10.5, "windows": 141, "palu_dir": "compressed/glrd_0.5"})
    out = tmp_path / "stage2b.md"
    from palu_tt import report
    report.main(raw=raw, out=out, rows=report.ROWS2B)
    text = out.read_text()
    assert "| baseline |" in text and "43.0" in text and "78.0" in text
    assert "| glrd_50 |" in text and "41.0" in text and " - |" in text          # missing task cells render as dashes
    assert "Paper (Llama-3-8B-Instruct, 50 %)" in text and "64.45" in text      # the reference block
    assert "10.500" in text


def test_paper_reference_matches_the_spec_table():
    """report.PAPER is a transcription of the stage 2b spec's paper table (arXiv 2407.21118 Table 1);
    the spec is the single source, so the two must agree cell for cell."""
    from palu_tt import report
    spec = Path("docs/superpowers/specs/2026-09-12-palu-tt-stage2b-design.md").read_text()
    labels = {"baseline": "baseline", "J-LRD": "J-LRD", "G-LRD": "G-LRD (gs 4)", "M-LRD": "M-LRD"}
    cols = ("openbookqa", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "avg6")
    for method, label in labels.items():
        row = next(line.strip() for line in spec.splitlines() if line.strip().startswith(f"| {label} |"))
        cells = [c.strip() for c in row.strip("|").split("|")][1:]
        want = dict(zip(cols, (float(c) for c in cells[:7])))
        got = {k: report.PAPER[method][k] for k in cols}
        assert got == want, (method, got, want)
        assert abs(sum(want[c] for c in cols[:6]) / 6 - want["avg6"]) < 0.01, method


def _tasks_json(row, obqa, piqa, arc_e, arc_c, wino):
    """A ``tasks_<row>.json`` with all five task cols, ``acc`` plus ``acc_norm`` for OBQA/ARC-c."""
    return {
        "row": row,
        "tasks": {
            "openbookqa": {"acc": obqa[0], "acc_norm": obqa[1], "n": 500},
            "piqa": {"acc": piqa, "n": 1838},
            "arc_easy": {"acc": arc_e, "n": 2376},
            "arc_challenge": {"acc": arc_c[0], "acc_norm": arc_c[1], "n": 1172},
            "winogrande": {"acc": wino, "n": 1267},
        },
    }


def test_avg5_paper_matches_the_papers_protocol():
    """acc_norm for OBQA/ARC-c, plain acc for the other three (spec Sec 0 / stage 2b analysis)."""
    from palu_tt import report
    tsk = _tasks_json("glrd_50", (0.320, 0.418), 0.762, 0.747, (0.439, 0.476), 0.706)
    assert report._avg5_paper(tsk) == "62.18"
    assert report._avg5_paper(None) == "-"
    assert report._avg5_paper({"tasks": {"openbookqa": {"acc": 0.3, "acc_norm": 0.4}}}) == "-"


def test_stage2c_calibration_table_paper_protocol_and_delta(tmp_path):
    """Task 3, step 1 (calibration sensitivity): ``acc (acc_norm)`` cells, the paper-protocol
    average side by side for the 32- and 256-sample rows, and a delta column on the f256 row."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pd = _palu_recipe_dir(tmp_path / "glrd_0.5", [(64, 192), (288, 512)])
    pd_f256 = _palu_recipe_dir(tmp_path / "glrd_0.5_f256", [(96, 224), (288, 512)])
    _write(raw, "ppl_glrd_50", {"ppl": 10.716, "windows": 141, "row": "glrd_50",
                                "kind": "palu", "palu_dir": str(pd)})
    _write(raw, "ppl_glrd_50_f256", {"ppl": 10.234, "windows": 141, "row": "glrd_50_f256",
                                     "kind": "palu", "palu_dir": str(pd_f256)})
    _write(raw, "tasks_glrd_50", _tasks_json("glrd_50", (0.320, 0.418), 0.762, 0.747, (0.439, 0.476), 0.706))
    _write(raw, "tasks_glrd_50_f256", _tasks_json("glrd_50_f256", (0.350, 0.450), 0.770, 0.760, (0.460, 0.490), 0.710))
    out = tmp_path / "stage2c.md"

    md = report.main(raw=raw, out=out, rows=report.ROWS2C)

    assert out.read_text() == md
    assert md.startswith("# Stage 2c results: 256-sample calibration and the shared-latent decode kernel")
    assert "## Calibration sensitivity" in md
    base_line = next(l for l in md.splitlines() if l.startswith("| glrd_50 |"))
    f256_line = next(l for l in md.splitlines() if l.startswith("| glrd_50_f256 |"))
    assert "32.0 (41.8)" in base_line and "10.716" in base_line and "62.18" in base_line
    assert base_line.split("|")[-5].strip() == "-"    # no delta on the 32-sample row
    assert "35.0 (45.0)" in f256_line and "10.234" in f256_line and "63.60" in f256_line
    assert "+1.42" in f256_line                        # 63.60 - 62.18
    # rank means, via the existing rank helpers, from the dump configs
    assert "64–288 (mean 176)" in base_line
    assert "96–288 (mean 192)" in f256_line
    # glrd_70 / glrd_70_f256 have no files at all -> dashes throughout, never invented numbers
    row70 = next(l for l in md.splitlines() if l.startswith("| glrd_70 |"))
    assert "| - |" in row70 and "128" not in row70


def test_stage2c_kernel_table_decode_batch8_and_attention_slope(tmp_path):
    """Task 3, step 1 (kernel table): decode ``min_ms`` at 1k/4k/8k for batch 32 and batch 8, the
    attention slope, and ``vs _fk2`` for the group-shared row; missing files render dashes."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "bench_decode_baseline", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode",
        "data": {"1024": {"p50_ms": 38.1, "min_ms": 38.0, "iters": 10},
                 "4096": {"p50_ms": 55.5, "min_ms": 55.4, "iters": 10},
                 "8192": {"error": "TT_FATAL ... Out of Memory: Not enough space",
                          "phase": "build", "exc": "RuntimeError"}},
    })
    _write(raw, "bench_decode_baseline_b8", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 8,
        "data": {"1024": {"p50_ms": 20.1, "min_ms": 20.0, "iters": 10}},
    })
    _write(raw, "bench_attention_baseline", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "attention",
        "data": {"1024": {"p50_ms": 0.9, "min_ms": 0.8, "iters": 20},
                 "8192": {"p50_ms": 2.3, "min_ms": 2.1, "iters": 20}},
    })
    _write(raw, "bench_decode_glrd_50_fk2", {
        "row": "glrd_50_fk2", "kind": "palu", "palu_dir": None, "what": "decode",
        "data": {"4096": {"p50_ms": 160.2, "min_ms": 160.0, "iters": 10}},
    })
    _write(raw, "bench_decode_glrd_50_fk2g", {
        "row": "glrd_50_fk2g", "kind": "palu", "palu_dir": None, "what": "decode",
        "data": {"4096": {"p50_ms": 80.1, "min_ms": 80.0, "iters": 10}},
    })
    out = tmp_path / "stage2c.md"

    md = report.main(raw=raw, out=out, rows=report.ROWS2C)

    assert out.read_text() == md
    assert "## Shared-latent kernel" in md
    line = next(l for l in md.splitlines() if l.startswith("| baseline |"))
    assert "| 38.0 | 55.4 | OOM |" in line          # decode batch 32, 1k/4k/8k
    assert "| 20.0 | - | - |" in line               # decode batch 8: only 1k measured
    assert "| 0.8 | 2.1 |" in line                  # attention-only 1k/8k
    assert "0.186" in line                          # slope: (2.1 - 0.8) / 7
    assert line.rstrip("|").rsplit("|", 1)[-1].strip() == "-"   # no `_fk2` sibling for baseline
    fk2g_line = next(l for l in md.splitlines() if l.startswith("| glrd_50_fk2g |"))
    assert "2.00x" in fk2g_line                     # 160.0 / 80.0, vs its `_fk2` sibling
    # a row with no files at all: every cell dashes
    fk2_70_line = next(l for l in md.splitlines() if l.startswith("| glrd_70_fk2 |"))
    assert "| - | - | - | - | - | - | - | - | - | - |" in fk2_70_line


def test_stage2c_decode_fidelity_and_batch8_maxctx(tmp_path):
    """Task 3, steps 2/3: decode fidelity for the f256/group-shared rows, and batch-8 max context
    for the group-shared rows next to their stage 2b ``_fk2`` values."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "decode_check_glrd_50_fk2g", {
        "row": "glrd_50_fk2g", "palu_dir": None, "steps": 64,
        "pcc_mean": 0.9999, "pcc_min": 0.9998, "argmax_agree": 0.99,
    })
    _write(raw, "decode_check_glrd_50_f256_fk2", {
        "row": "glrd_50_f256_fk2", "palu_dir": None, "steps": 64,
        "pcc_mean": 0.9990, "pcc_min": 0.9970, "argmax_agree": 0.98,
    })
    _write(raw, "bench_maxctx_glrd_50_fk2_b8", {
        "row": "glrd_50_fk2", "kind": "palu", "palu_dir": None, "what": "maxctx", "batch": 8,
        "data": {"max_context": 57344, "batch": 8, "limit": "oom"},
    })
    _write(raw, "bench_maxctx_glrd_50_fk2g_b8", {
        "row": "glrd_50_fk2g", "kind": "palu", "palu_dir": None, "what": "maxctx", "batch": 8,
        "data": {"max_context": 57344, "batch": 8, "limit": "oom"},
    })
    out = tmp_path / "stage2c.md"

    md = report.main(raw=raw, out=out, rows=report.ROWS2C)

    assert out.read_text() == md
    assert "## Decode fidelity" in md
    assert "| glrd_50_fk2g | 0.9998 / 0.9999 | 0.990 |" in md
    assert "| glrd_50_f256_fk2 | 0.9970 / 0.9990 | 0.980 |" in md
    # not measured in this fixture -> dashes, never invented
    assert "| glrd_70_fk2g | - | - |" in md
    assert "## Batch-8 max context" in md
    assert "| glrd_50_fk2 | 57344 | oom |" in md
    assert "| glrd_50_fk2g | 57344 | oom |" in md
    assert "| glrd_70_fk2 | - | - |" in md
    assert "| glrd_70_fk2g | - | - |" in md


def test_stage2c_tables_have_matching_header_and_separator_widths(tmp_path):
    """GFM only recognises a table when the separator row has as many cells as the header."""
    raw = tmp_path / "raw"; raw.mkdir()
    _write(raw, "ppl_glrd_50", {"row": "glrd_50", "ppl": 10.7, "windows": 141, "palu_dir": "compressed/glrd_0.5"})
    out = tmp_path / "stage2c.md"
    from palu_tt import report
    report.main(raw=raw, out=out, rows=report.ROWS2C)
    lines = out.read_text().splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.startswith("| ") and lines[i + 1].startswith("|---"):
            assert line.count("|") == lines[i + 1].count("|"), (line, lines[i + 1])


def _prof(entries):
    """A minimal ``profile_<row>_<ctx>.json`` -- ``profile_fit.fit_signatures`` only reads ``ops``.
    ``entries`` is ``(op, in1_last, out_last, ms, count)``."""
    return {"ops": [{"op": op, "category": "attention", "ms": ms, "count": count,
                     "in1": [0, 0, 0, in1_last], "out": [0, 0, 0, out_last]}
                    for op, in1_last, out_last, ms, count in entries]}


def _decode_check(pcc_min, pcc_mean, argmax, steps=63):
    return {"steps": steps, "pcc_mean": pcc_mean, "pcc_min": pcc_min, "argmax_agree": argmax}


def test_stage2e_tables_render_from_a_fixture_raw_dir(tmp_path):
    """A synthetic fixture raw dir, not the live ``docs/results/raw`` -- a measurement queue writes
    there concurrently with this test suite running, and a half-written JSON must never be able to
    fail (or flake) this test. The fixture models the same "matrix partially complete" shape the
    generator has to tolerate: full data for one variant ("" / legacy), a single measured context for
    another ("p" / pack, not yet fittable), and none at all for a third ("d" / dst)."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # Reference rows (baseline, the naive Palu row) -- the ablation table's stock-signature fit must
    # come from these two alone.
    for ctx, ms in ((1024, 7.0), (4096, 15.0), (8192, 28.0)):
        _write(raw, f"profile_baseline_{ctx}", _prof([("SdpaDecodeDeviceOperation", 128, 128, ms, 32.0)]))
        _write(raw, f"profile_palu_full_50_{ctx}", _prof([("SdpaDecodeDeviceOperation", 128, 128, ms + 1, 32.0)]))
    # The "" (legacy) variant's own two rows: fully measured at all three contexts.
    for ctx, ms in ((1024, 6.0), (4096, 11.0), (8192, 20.0)):
        _write(raw, f"profile_palu_full_50_fk2_{ctx}", _prof([("PaluSdpaDecodeDeviceOperation", 32, 128, ms, 18.0)]))
        _write(raw, f"profile_palu_full_70_fk2_{ctx}", _prof([("PaluSdpaDecodeDeviceOperation", 64, 128, ms + 2, 18.0)]))
    # The "p" (pack) variant: exactly one context measured for one of its two rows -- not enough for a
    # fit (needs two distinct contexts), but not "no data" either.
    _write(raw, "profile_palu_full_50_fk2p_8192", _prof([("PaluSdpaDecodeDeviceOperation", 32, 128, 19.0, 18.0)]))
    # The "d" (dst) variant: no profile JSON at all.

    _write(raw, "bench_decode_baseline", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode",
        "data": {"1024": {"p50_ms": 38.0, "min_ms": 38.0}, "4096": {"p50_ms": 55.4, "min_ms": 55.4},
                 "8192": {"error": "TT_FATAL ... Out of Memory", "phase": "build", "exc": "RuntimeError"}},
    })
    _write(raw, "bench_decode_palu_full_50_fk2", {
        "row": "palu_full_50_fk2", "kind": "palu", "palu_dir": "/x", "what": "decode",
        "data": {"1024": {"p50_ms": 42.9, "min_ms": 42.9, "kv_bytes_per_token_per_layer_per_chip": 680.0},
                 "4096": {"p50_ms": 65.0, "min_ms": 65.0}, "8192": {"p50_ms": 94.5, "min_ms": 94.5}},
    })
    _write(raw, "bench_decode_palu_full_50_fk2_l4", {
        "row": "palu_full_50_fk2_l4", "kind": "palu", "palu_dir": "/x", "what": "decode",
        "data": {"1024": {"p50_ms": 42.7, "min_ms": 42.7, "kv_bytes_per_token_per_layer_per_chip": 360.0},
                 "4096": {"p50_ms": 64.7, "min_ms": 64.7}},
    })
    _write(raw, "bench_decode_palu_full_70_fk2", {
        "row": "palu_full_70_fk2", "kind": "palu", "palu_dir": "/y", "what": "decode",
        "data": {"1024": {"p50_ms": 43.1, "min_ms": 43.1}, "4096": {"p50_ms": 66.4, "min_ms": 66.4},
                 "8192": {"p50_ms": 97.7, "min_ms": 97.7}},
    })

    _write(raw, "decode_check_palu_full_50_fk2", _decode_check(0.9980, 0.9994, 0.921))
    _write(raw, "decode_check_palu_full_70_fk2", _decode_check(0.9988, 0.9993, 0.984))
    _write(raw, "decode_check_palu_full_50_fk2_l4", _decode_check(0.9935, 0.9987, 0.905))

    _write(raw, "ppl_palu_full_50", {"ppl": 12.930530567327075, "windows": 141, "row": "palu_full_50",
                                     "kind": "palu", "palu_dir": "compressed/palu_full_0.5"})
    _write(raw, "ppl_palu_full_50_l4", {"ppl": 12.930530567327075, "windows": 141, "row": "palu_full_50_l4",
                                        "kind": "palu", "palu_dir": "compressed/palu_full_0.5"})

    out = tmp_path / "stage2e.md"

    text = report.main(raw=raw, out=out, rows=report.ROWS2E, stage="2e")

    assert out.read_text() == text
    ablation = text.split("## Kernel variants", 1)[1].split("## Decode step", 1)[0]
    assert "| variant | signature | rows |" in ablation and "fixed us" in ablation

    # Important 1: "no data" ("d") vs "not fittable" ("p") are distinguished, not both "(not measured)".
    dst_label = dict(report.VARIANTS2E)["d"]
    pack_label = dict(report.VARIANTS2E)["p"]
    assert f"| {dst_label} | (not measured) |" in ablation
    assert (f"| {pack_label} | (1 context(s) measured: palu_full_50_fk2p @ 8192 -- a fit needs two) |"
            in ablation)

    # Minor 5: the stock signature's "rows" column is fed by baseline/palu_full_50 only -- restricting
    # the reference fit means no "_fk2" row can appear in it (that was the fixture pollution finding 5
    # described: the stock fit picking up a variant row's own leftover noise).
    stock_line = next(l for l in ablation.splitlines() if "SdpaDecodeDeviceOperation r_k=128 out=128" in l)
    assert "| baseline, palu_full_50 |" in stock_line
    assert "fk2" not in stock_line.split("|")[3]   # the "rows" cell specifically, not the whole line

    # whole-step table: both real rows render as numbers, not "(not measured)".
    assert "| palu_full_50_fk2 | 42.9 | 65.0 | 94.5 |" in text
    assert "| baseline | " in text and "(not measured)" not in next(
        l for l in text.splitlines() if l.startswith("| baseline | "))

    # Important 2: the bfp4-latent (L4) row reaches the fidelity table (main() loads it on top of
    # ROWS2E specifically for this table). Minor 6: the step count is back in the header.
    assert "decode_check`, 63 teacher-forced steps)" in text
    fidelity = text.split("## Decode fidelity", 1)[1].split("## bfp4", 1)[0]
    assert "| palu_full_50_fk2 | 0.9980 / 0.9994 | 0.921 |" in fidelity
    assert "| palu_full_50_fk2_l4 | 0.9935 / 0.9987 | 0.905 |" in fidelity

    # Minor 9: the bfp8/bfp4 perplexity cells are checked for equality, not against a hard-coded number.
    bfp4 = text.split("## bfp4 latents", 1)[1]
    ppl_row = next(l for l in bfp4.splitlines() if l.startswith("| palu_full_50 | "))
    cells = [c.strip() for c in ppl_row.split("|")]
    ppl8, ppl4 = cells[2], cells[3]
    assert ppl8 == ppl4 and ppl8 != "-"
    assert "ppl (prefill-blind, see stage 2f)" in text
    assert "stage 2f" in bfp4   # the caveat sentence under the table
