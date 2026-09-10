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


def test_time_defaults():
    calls = []
    r = bench._time(lambda: calls.append(1))
    assert len(calls) == 13 and r["iters"] == 10  # defaults: warmup=3, iters=10


@pytest.mark.parametrize("env_value, expected", [
    (None, (1024, 4096, 8192)),  # unset -> default matrix used by the results table
    ("", (1024, 4096, 8192)),  # empty string -> same default
    ("1024,4096", (1024, 4096)),
    ("1024", (1024,)),
    ("1024,,4096", (1024, 4096)),  # doubled comma -> empty item ignored, not a crash
    ("1024,4096,", (1024, 4096)),  # trailing comma -> empty item ignored
    (" 1024 , 4096 ", (1024, 4096)),  # incidental whitespace
])
def test_parse_contexts(env_value, expected):
    assert bench._parse_contexts(env_value) == expected


@pytest.mark.parametrize("batch, expected", [
    (32, "bench_decode_baseline.json"),  # default batch: unsuffixed, so stage-1/2 names don't move
    (1, "bench_decode_baseline_b1.json"),
    (4, "bench_decode_baseline_b4.json"),
    (16, "bench_decode_baseline_b16.json"),
])
def test_out_name(batch, expected):
    assert bench._out_name("decode", "baseline", batch) == expected
