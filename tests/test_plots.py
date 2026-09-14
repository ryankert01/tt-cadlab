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

    # Important 3 (2f Task 6 review): a linear y-axis put the whole decision-relevant 7-16 ppl band
    # into the bottom ~5% of the plot once Grid B's ~270-ppl bfp4/unit cells were added. Log scale
    # separates them again.
    assert captured["fig"].axes[0].get_yscale() == "log"


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


def test_frontier_figure_labels_the_finalists_even_when_dominated(tmp_path, monkeypatch):
    """2f final review, Important 2: the study's recommendation, ``palu_full_70_lv4``, is dominated
    on the (bytes, perplexity) plane by ``glrd_70_f256_l4``, so it fell off the Pareto front and was
    the one finalist with no label -- a reader could not find the recommendation on the figure.
    ``plots.FINALISTS`` is unioned into the labelled set, so a dominated finalist keeps its name
    while a dominated non-finalist (here ``palu_full_70_lk4_w40``) still has none."""
    from palu_tt import plots

    (tmp_path / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    # bfp4 K and V: fewer bytes *and* lower perplexity, so it dominates the _lv4 row below.
    (tmp_path / "ppl_palu_full_70_l4.json").write_text(
        '{"ppl": 9.0, "windows": 141, "latent_quant_in_prefill": true}')
    (tmp_path / "ppl_palu_full_70_lv4.json").write_text(
        '{"ppl": 9.3, "windows": 141, "latent_quant_in_prefill": true}')
    # two screens, so the leading edge of the screened cloud is the cheaper one and ``_lk4_w40``
    # is dominated there too -- otherwise it would be labelled as a scan-front point, not as a
    # finalist, and the negative half of this test would prove nothing.
    (tmp_path / "ppl_palu_full_70_l4_w40.json").write_text(
        '{"ppl": 9.4, "windows": 40, "latent_quant_in_prefill": true}')
    (tmp_path / "ppl_palu_full_70_lk4_w40.json").write_text(
        '{"ppl": 9.9, "windows": 40, "latent_quant_in_prefill": true}')
    out = tmp_path / "frontier.png"

    captured = {}
    real_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda fig: (captured.setdefault("fig", fig), real_close(fig)))

    plots.frontier_figure(tmp_path, out, ranks={"palu_full_70": (58, 121)})

    from palu_tt import report
    rows = {r["row"]: r for r in report.frontier_rows(tmp_path, ranks={"palu_full_70": (58, 121)})}
    assert rows["palu_full_70_lv4"]["kb_token"] > rows["palu_full_70_l4"]["kb_token"]
    assert rows["palu_full_70_lv4"]["ppl"] > rows["palu_full_70_l4"]["ppl"]  # dominated: off the front
    ax = captured["fig"].axes[0]
    front = next(ln for ln in ax.lines if ln.get_linestyle() in ("--", (0, (6.4, 1.6))))
    assert round(rows["palu_full_70_lv4"]["kb_token"], 3) not in {round(x, 3) for x in front.get_xdata()}

    labels = {t.get_text() for t in ax.texts}
    assert "palu_full_70_lv4" in labels
    assert "palu_full_70_l4" in labels
    assert not any(t.startswith("palu_full_70_lk4") for t in labels)


def test_frontier_figure_keeps_every_label_inside_the_axes(tmp_path, monkeypatch):
    """2f final review, Important 3 (carried Minor A): ``_place_labels`` clamped on the *anchor*
    pixel, not on the text extent, so a long row name anchored well inside a spine still ran off the
    axes -- four labels overflowed the shipped figure, the worst by 63 px into the y-tick column.
    The two long-named rows below sit at the extreme low- and high-byte ends, which is exactly where
    the anchor test cannot help."""
    from palu_tt import plots

    (tmp_path / "ppl_baseline.json").write_text('{"ppl": 7.254, "windows": 141}')
    # a cheaper, short-named row holds the left spine, so the long-named one below is anchored ~110
    # px *inside* the axes -- the case the anchor-only clamp cannot see.
    (tmp_path / "ppl_palu_full_50_l4.json").write_text(
        '{"ppl": 200.0, "windows": 141, "latent_quant_in_prefill": true}')
    (tmp_path / "ppl_uni_rk32_rv64_u_l4_w40.json").write_text(
        '{"ppl": 94.4, "windows": 40, "latent_quant_in_prefill": true}')
    # two dominated (so unlabelled) cells just to its right at the same perplexity, which is what
    # makes the side-choice send that label leftward -- the dense low-byte cluster's effect.
    (tmp_path / "ppl_uni_rk32_rv96_u_l4_w40.json").write_text(
        '{"ppl": 96.0, "windows": 40, "latent_quant_in_prefill": true}')
    (tmp_path / "ppl_uni_rk64_rv64_u_l4_w40.json").write_text(
        '{"ppl": 98.0, "windows": 40, "latent_quant_in_prefill": true}')
    (tmp_path / "ppl_uni_rk96_rv128_u_w40.json").write_text('{"ppl": 7.7, "windows": 40}')
    (tmp_path / "ppl_palu_full_70_l4.json").write_text(
        '{"ppl": 9.0, "windows": 141, "latent_quant_in_prefill": true}')
    out = tmp_path / "frontier.png"

    captured = {}
    real_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda fig: (captured.setdefault("fig", fig), real_close(fig)))

    ranks = {"uni_rk32_rv64": (32, 64), "uni_rk96_rv128": (96, 128), "palu_full_70": (58, 121),
             "palu_full_50": (10, 10), "uni_rk32_rv96": (32, 96), "uni_rk64_rv64": (64, 64)}
    plots.frontier_figure(tmp_path, out, ranks=ranks)

    fig = captured["fig"]
    ax = fig.axes[0]
    rend = fig.canvas.get_renderer()
    box = ax.get_window_extent(rend)
    assert len(ax.texts) >= 5  # the long-named screens really are labelled, or this proves nothing
    outside = [(t.get_text(), round(box.x0 - t.get_window_extent(rend).x0, 1),
                round(t.get_window_extent(rend).x1 - box.x1, 1))
               for t in ax.texts
               if not box.contains(*t.get_window_extent(rend).p0)
               or not box.contains(*t.get_window_extent(rend).p1)]
    assert outside == [], f"labels outside the axes (label, left overflow px, right overflow px): {outside}"

    # The same review finding measured three labels drawn on top of markers, one of them a
    # finalist's; the placement scores candidates by the markers the text would actually cover.
    from palu_tt import report
    pts = [ax.transData.transform((r["kb_token"], r["ppl"]))
           for r in report.frontier_rows(tmp_path, ranks=ranks) if not r["blind"]]
    covered = [t.get_text() for t in ax.texts
               if any(t.get_window_extent(rend).contains(qx, qy) for qx, qy in pts)]
    assert covered == [], f"labels drawn over a marker: {covered}"
