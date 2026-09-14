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
    # "m" (legacy + multicast): its VARIANTS2E label is long prose, unlike "d"'s ("dst"), which is
    # already short -- this is what actually exercises the short-label carried finding below.
    _write(raw, "profile_palu_full_50_fk2m_4096", _profile(8.0))
    monkeypatch.setattr(profile_plots, "RAW", raw)

    captured = {}
    real_close = profile_plots.plt.close
    monkeypatch.setattr(profile_plots.plt, "close",
                         lambda fig: (captured.setdefault("fig", fig), real_close(fig)))
    out = tmp_path / "profile_attention_ctx_2e.png"

    profile_plots.attention_ctx_variants_figure(out)

    assert out.exists() and out.stat().st_size > 0
    ax = captured["fig"].axes[0]
    long_labels = dict(VARIANTS2E)
    expected = ["stock"] + [profile_plots.PLOT_LABEL2E[sfx] for sfx, _ in VARIANTS2E if sfx in ("d", "m")]
    got = [line.get_label() for line in ax.lines]
    assert got == expected

    # Carried finding (stage 2e review): the legend uses PLOT_LABEL2E's short labels, not
    # VARIANTS2E's own long table prose -- "m"'s table label is long enough that this actually
    # distinguishes the two ("d"'s table label, "dst", already happens to be short).
    assert "legacy+mcast" in got
    assert long_labels["m"] not in got


def test_frontier_figure_renders(tmp_path):
    from palu_tt import plots
    (tmp_path / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    (tmp_path / "ppl_baseline_kv4.json").write_text('{"ppl": 7.9, "windows": 141}')
    (tmp_path / "ppl_uni_rk64_rv32_l4_w40.json").write_text('{"ppl": 9.9, "windows": 40}')
    out = tmp_path / "frontier.png"
    plots.frontier_figure(tmp_path, out, ranks={"uni_rk64_rv32": (64, 32)})
    assert out.exists() and out.stat().st_size > 1000


def test_frontier_figure_skips_prefill_blind_rows(tmp_path, monkeypatch):
    """A row measured before the cache-faithful prefill (``latent_quant_in_prefill: false``) must not
    be drawn as a bfp4 point -- its perplexity is its bfp8 sibling's by construction, so plotting it
    would show a fake, too-good bfp4 point on the frontier. ``frontier_rows`` already flags it
    (``blind``); the figure must actually act on that flag, not just the table."""
    from palu_tt import plots

    (tmp_path / "ppl_palu_full_50.json").write_text(
        '{"ppl": 12.93, "windows": 141, "palu_dir": "compressed/palu_full_0.5"}')
    (tmp_path / "ppl_palu_full_50_l4.json").write_text(
        '{"ppl": 12.93, "windows": 141, "palu_dir": "compressed/palu_full_0.5", '
        '"latent_quant_in_prefill": false, "note": "prefill-blind"}')
    out = tmp_path / "frontier.png"

    captured = {}
    real_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda fig: (captured.setdefault("fig", fig), real_close(fig)))

    plots.frontier_figure(tmp_path, out, ranks={"palu_full_50": (33, 102)})

    assert out.exists()
    ax = captured["fig"].axes[0]
    plotted_x = {round(pt[0], 3) for coll in ax.collections for pt in coll.get_offsets()}
    from palu_tt import report
    rows = {r["row"]: r for r in report.frontier_rows(tmp_path, ranks={"palu_full_50": (33, 102)})}
    assert round(rows["palu_full_50"]["kb_token"], 3) in plotted_x
    assert round(rows["palu_full_50_l4"]["kb_token"], 3) not in plotted_x


def test_frontier_figure_skips_a_blind_row_with_no_field_at_all(tmp_path, monkeypatch):
    """The pre-2f fallback (no ``latent_quant_in_prefill`` field at all): a suffix-less bfp8 row is
    plotted as faithful, but a ``_l4``-suffixed row with the field simply missing is excluded from
    the figure exactly like one explicitly marked ``false`` -- ``uni_rk64_rv32`` needs no rank
    override lookup against a real dump, so this stays a synthetic fixture."""
    from palu_tt import plots, report

    (tmp_path / "ppl_uni_rk64_rv32.json").write_text('{"ppl": 8.5, "windows": 141}')
    (tmp_path / "ppl_uni_rk64_rv32_l4.json").write_text('{"ppl": 8.5, "windows": 141}')
    out = tmp_path / "frontier.png"

    captured = {}
    real_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda fig: (captured.setdefault("fig", fig), real_close(fig)))

    plots.frontier_figure(tmp_path, out, ranks={"uni_rk64_rv32": (64, 32)})

    assert out.exists()
    ax = captured["fig"].axes[0]
    plotted_x = {round(pt[0], 3) for coll in ax.collections for pt in coll.get_offsets()}
    rows = {r["row"]: r for r in report.frontier_rows(tmp_path, ranks={"uni_rk64_rv32": (64, 32)})}
    assert rows["uni_rk64_rv32"]["blind"] is False and rows["uni_rk64_rv32_l4"]["blind"] is True
    assert round(rows["uni_rk64_rv32"]["kb_token"], 3) in plotted_x
    assert round(rows["uni_rk64_rv32_l4"]["kb_token"], 3) not in plotted_x


def test_frontier_figure_labels_only_the_front_and_the_stock_rows(tmp_path, monkeypatch):
    """Carried review finding (stage 2f): every point used to carry its row name, which was
    unreadable where a dozen cells share a byte budget. Only Pareto-front points and the stock rows
    are labelled now -- here ``uni_rk64_rv96`` (dominated: more bytes *and* worse perplexity than
    ``uni_rk32_rv96``) must have no label, while the front point and ``baseline`` must."""
    from palu_tt import plots

    (tmp_path / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    (tmp_path / "ppl_uni_rk32_rv96_u_w40.json").write_text('{"ppl": 8.5, "windows": 40}')
    (tmp_path / "ppl_uni_rk64_rv96_u_w40.json").write_text('{"ppl": 9.9, "windows": 40}')
    out = tmp_path / "frontier.png"

    captured = {}
    real_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda fig: (captured.setdefault("fig", fig), real_close(fig)))

    plots.frontier_figure(tmp_path, out, ranks={"uni_rk32_rv96": (32, 96), "uni_rk64_rv96": (64, 96)})

    labels = {t.get_text() for t in captured["fig"].axes[0].texts}
    # 40-window rows are labelled with their window count appended (see the window-count tests
    # below): the leading edge of the screened cloud, never the dominated cell behind it.
    assert "uni_rk32_rv96_u_w40 (40w)" in labels
    assert "baseline" in labels
    assert not any(t.startswith("uni_rk64_rv96_u_w40") for t in labels)


def test_pareto_front_ignores_rows_screened_at_fewer_windows(tmp_path):
    """I-7 (stage 2e final review): 40-window perplexity is ~0.5 easier than the 141-window
    references on the same row and format, so a front through both would rank points by how much
    text they were scored over. ``_pareto_front`` drops them before the scan -- here the 40-window
    row is both cheaper *and* lower-perplexity than the only 141-window row, i.e. it would dominate
    the front outright if window count were ignored, and it must still not appear on it."""
    from palu_tt import plots

    rows = [{"row": "cheap_w40", "kb_token": 10.0, "ppl": 8.0, "windows": 40},
            {"row": "full", "kb_token": 20.0, "ppl": 9.0, "windows": 141}]
    assert [r["row"] for r in plots._pareto_front(rows)] == ["full"]
    # the unfiltered scan is what the screened cloud's own leading edge is drawn from
    assert [r["row"] for r in plots._scan_front(rows)] == ["cheap_w40"]


def test_frontier_figure_draws_the_front_through_full_length_rows_only(tmp_path, monkeypatch):
    """The dashed line itself: with a 40-window screen sitting below and left of every 141-window
    row, the polyline must still connect only the 141-window points, and the screen must be drawn
    faint and behind them rather than dropped."""
    from palu_tt import plots

    (tmp_path / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    (tmp_path / "ppl_palu_full_50.json").write_text(
        '{"ppl": 12.93, "windows": 141, "palu_dir": "compressed/palu_full_0.5"}')
    (tmp_path / "ppl_uni_rk32_rv32_u_w40.json").write_text('{"ppl": 6.0, "windows": 40}')
    out = tmp_path / "frontier.png"

    captured = {}
    real_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda fig: (captured.setdefault("fig", fig), real_close(fig)))

    plots.frontier_figure(tmp_path, out, ranks={"palu_full_50": (33, 102), "uni_rk32_rv32": (32, 32)})

    from palu_tt import report
    rows = {r["row"]: r for r in report.frontier_rows(
        tmp_path, ranks={"palu_full_50": (33, 102), "uni_rk32_rv32": (32, 32)})}
    ax = captured["fig"].axes[0]
    line = next(ln for ln in ax.lines if ln.get_linestyle() in ("--", (0, (6.4, 1.6))))
    on_front = {round(x, 3) for x in line.get_xdata()}
    assert round(rows["baseline"]["kb_token"], 3) in on_front
    assert round(rows["palu_full_50"]["kb_token"], 3) in on_front
    assert round(rows["uni_rk32_rv32_u_w40"]["kb_token"], 3) not in on_front

    # the screen is still plotted, faint and behind the full-length points
    by_x = {round(c.get_offsets()[0][0], 3): c for c in ax.collections}
    screen = by_x[round(rows["uni_rk32_rv32_u_w40"]["kb_token"], 3)]
    full = by_x[round(rows["baseline"]["kb_token"], 3)]
    assert screen.get_alpha() < 1.0 and (full.get_alpha() or 1.0) == 1.0
    assert screen.get_zorder() < full.get_zorder()


def test_frontier_figure_legend_names_the_screened_series(tmp_path, monkeypatch):
    """The faint cloud gets its own legend entry naming the window count, so a reader is told the
    screens are a different series rather than left to infer it from the shading."""
    from palu_tt import plots

    (tmp_path / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    (tmp_path / "ppl_palu_full_50.json").write_text(
        '{"ppl": 12.93, "windows": 141, "palu_dir": "compressed/palu_full_0.5"}')
    (tmp_path / "ppl_uni_rk32_rv32_u_w40.json").write_text('{"ppl": 9.0, "windows": 40}')
    out = tmp_path / "frontier.png"

    captured = {}
    real_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda fig: (captured.setdefault("fig", fig), real_close(fig)))

    plots.frontier_figure(tmp_path, out, ranks={"palu_full_50": (33, 102), "uni_rk32_rv32": (32, 32)})

    legend = captured["fig"].axes[0].get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert "141-window measurement" in labels
    assert "40-window screen (not on the front)" in labels
    assert "Pareto front (141-window rows)" in labels
