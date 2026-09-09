"""CPU-only checks for the benchmark timing helper (no device involved)."""
import time

import pytest

from palu_tt import bench


def test_time_runs_warmup_plus_iters_and_reports_median(monkeypatch):
    """Warm-ups are called but not timed; p50/min come from the timed iterations only."""
    # fake clock: each perf_counter() call advances by the next step. _time calls it twice per
    # timed iteration (before/after), so the durations below are 0.030 / 0.010 / 0.020 s.
    ticks = iter([0.0, 0.030, 0.100, 0.110, 0.200, 0.220])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    calls = []
    r = bench._time(lambda: calls.append(1), iters=3, warmup=2)

    assert len(calls) == 5, f"expected warmup(2) + iters(3) calls, got {len(calls)}"
    assert set(r) == {"p50_ms", "min_ms", "iters"}
    assert r["iters"] == 3
    assert r["p50_ms"] == pytest.approx(20.0)
    assert r["min_ms"] == pytest.approx(10.0)


def test_time_defaults_and_contexts():
    calls = []
    r = bench._time(lambda: calls.append(1))
    assert len(calls) == 13 and r["iters"] == 10  # defaults: warmup=3, iters=10
    assert bench.CONTEXTS == (1024, 4096, 8192)
