"""CPU test for the context-scaling figures: renders from a fixture raw dir (matplotlib, Agg
backend) and asserts the PNG exists and is non-empty. Does not touch a device."""
import json

from palu_tt import plots


def _write(raw, name, obj):
    (raw / f"{name}.json").write_text(json.dumps(obj))


def test_ctx_scaling_figure_renders_a_nonempty_png(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "bench_decode_baseline_b1", {
        "row": "baseline", "kind": "stock", "palu_dir": None, "what": "decode", "batch": 1,
        "contexts": [1024, 2048, 4096],
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
    _write(raw, "bench_decode_palu_full_50_fk2_b1", {
        "row": "palu_full_50_fk2", "kind": "palu", "palu_dir": "/x", "what": "decode", "batch": 1,
        "contexts": [1024, 2048, 4096],
        "data": {"1024": {"p50_ms": 34.5, "min_ms": 34.0, "iters": 10},
                 "2048": {"p50_ms": 35.0, "min_ms": 34.6, "iters": 10},
                 "4096": {"p50_ms": 36.4, "min_ms": 36.0, "iters": 10}},
    })
    _write(raw, "bench_maxctx_palu_full_50_fk2_b1", {
        "row": "palu_full_50_fk2", "kind": "palu", "palu_dir": "/x", "what": "maxctx", "batch": 1,
        "data": {"max_context": 131072, "batch": 1, "limit": "model", "failed_at": None,
                 "error": None, "phase": None, "exc": None,
                 "kv_bytes_per_token_per_layer_per_chip": 680.0,
                 "probes": [[c, True] for c in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)]},
    })
    out = tmp_path / "ctx_scaling_b1.png"

    result = plots.ctx_scaling_figure(raw, 1, out, rows=("baseline", "palu_full_50_fk2"))

    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_ctx_scaling_figure_skips_a_row_with_no_decode_json(tmp_path):
    """A row that was never run (e.g. batch 8 blocked entirely) must not crash the render."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "ctx_scaling_b8.png"

    result = plots.ctx_scaling_figure(raw, 8, out, rows=("baseline",))

    assert result == out
    assert out.exists() and out.stat().st_size > 0
