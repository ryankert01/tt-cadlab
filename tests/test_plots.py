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


def _profile(ms):
    """Minimal profile JSON: only the ``attention`` category and ``device_ms_per_step`` are read by
    ``attention_ctx_variants_figure``/``_cat_ms``; ``ops`` is unused there but present so the schema
    matches the real ``profile_report`` output."""
    return {"categories": {"attention": {"ms": ms, "count": 32.0}}, "ops": [], "device_ms_per_step": ms + 20.0}


def test_attention_ctx_variants_figure_plots_exactly_the_measured_variants(tmp_path, monkeypatch):
    """Stage 2e's rotation-kernel-variant figure, against a synthetic fixture raw dir rather than the
    live ``docs/results/raw`` -- a measurement queue can be writing there concurrently, and reading a
    half-written JSON must never be able to fail this test (module docstring's own stated intent:
    render "from a fixture raw dir"). Only two rows exist here (stock, and the "d"/dst variant); the
    figure must plot exactly those two lines, under exactly their ``report.VARIANTS2E`` labels, and
    silently skip every other variant rather than inventing a point or crashing on the missing file.
    """
    from palu_tt import profile_plots
    from palu_tt.report import VARIANTS2E

    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "profile_baseline_1024", _profile(7.1))
    _write(raw, "profile_baseline_4096", _profile(15.0))
    _write(raw, "profile_palu_full_50_fk2d_4096", _profile(9.5))
    _write(raw, "profile_palu_full_50_fk2d_8192", _profile(17.0))
    monkeypatch.setattr(profile_plots, "RAW", raw)

    captured = {}
    real_close = profile_plots.plt.close
    monkeypatch.setattr(profile_plots.plt, "close",
                         lambda fig: (captured.setdefault("fig", fig), real_close(fig)))
    out = tmp_path / "profile_attention_ctx_2e.png"

    profile_plots.attention_ctx_variants_figure(out)

    assert out.exists() and out.stat().st_size > 0
    ax = captured["fig"].axes[0]
    expected = ["stock"] + [label for sfx, label in VARIANTS2E if sfx == "d"]
    assert [line.get_label() for line in ax.lines] == expected
