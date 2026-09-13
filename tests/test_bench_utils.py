"""CPU-only checks for the benchmark timing helper (no device involved)."""
import time
from types import SimpleNamespace

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


class _FakeMesh:
    """Stands in for ``bench.Mesh`` in CPU tests: no real device, just a reset counter."""

    def __init__(self):
        self.md = "fake-md"
        self.resets = 0

    def reset(self):
        self.resets += 1
        return self.md


def _threshold_build_model(threshold, kv=42.0):
    """A fake ``pm.build_model`` that succeeds for ``max_seq_len <= threshold`` and otherwise
    raises like a real allocation failure (message contains "Out of Memory")."""

    def build(md, kind, palu_dir, max_batch_size, max_seq_len, **fused):
        if max_seq_len <= threshold:
            return SimpleNamespace(), SimpleNamespace(), None
        raise RuntimeError("TT_FATAL ... Out of Memory: Not enough space")

    return build


def test_bench_maxctx_bisects_to_1k_granularity_between_last_success_and_first_failure(monkeypatch):
    """threshold=5000: 1k/2k/4k succeed, 8k is the first failure -> bisect (4096, 8192) down to a
    1024-wide bracket around the true (unknown-to-the-algorithm) boundary."""
    monkeypatch.setattr(bench.pm, "build_model", _threshold_build_model(5000))
    monkeypatch.setattr(bench.pm, "kv_bytes_per_token_per_layer", lambda model, args: 42.0)
    mesh = _FakeMesh()

    r = bench.bench_maxctx(mesh, "stock", None)

    assert r["max_context"] == 4096          # largest known-good context
    assert r["failed_at"] == 5120            # smallest known-bad context, exactly 1024 above
    assert r["limit"] == "oom"
    assert r["kv_bytes_per_token_per_layer_per_chip"] == 42.0
    assert r["batch"] == bench.BATCH
    # every failed probe (8192, 6144, 5120) resets the mesh before the next attempt
    assert mesh.resets == 3
    assert [1024, True] in r["probes"] and [8192, False] in r["probes"]


def test_bench_maxctx_full_success_hits_the_model_limit(monkeypatch):
    monkeypatch.setattr(bench.pm, "build_model", _threshold_build_model(bench.MODEL_MAX_CTX))
    monkeypatch.setattr(bench.pm, "kv_bytes_per_token_per_layer", lambda model, args: 7.0)
    mesh = _FakeMesh()

    r = bench.bench_maxctx(mesh, "stock", None)

    assert r["max_context"] == bench.MODEL_MAX_CTX
    assert r["limit"] == "model"
    assert r["failed_at"] is None and r["error"] is None
    assert mesh.resets == 0
    assert r["probes"][-1] == [bench.MODEL_MAX_CTX, True]


def test_bench_maxctx_writes_max_context_batch32_only_at_batch_32(monkeypatch):
    monkeypatch.setattr(bench.pm, "build_model", _threshold_build_model(bench.MODEL_MAX_CTX))
    monkeypatch.setattr(bench.pm, "kv_bytes_per_token_per_layer", lambda model, args: 7.0)
    mesh = _FakeMesh()
    monkeypatch.setattr(bench, "BATCH", 1)

    r = bench.bench_maxctx(mesh, "stock", None)
    assert "max_context_batch32" not in r

    monkeypatch.setattr(bench, "BATCH", 32)
    r = bench.bench_maxctx(mesh, "stock", None)
    assert r["max_context_batch32"] == bench.MODEL_MAX_CTX


def test_bench_maxctx_first_probe_failing_leaves_no_known_good_context(monkeypatch):
    """threshold < 1024: even the smallest probe fails -> nothing to bisect against."""
    monkeypatch.setattr(bench.pm, "build_model", _threshold_build_model(0))
    mesh = _FakeMesh()

    r = bench.bench_maxctx(mesh, "stock", None)

    assert r["max_context"] is None
    assert r["failed_at"] == 1024
    assert r["limit"] == "oom"
    assert mesh.resets == 1


def test_is_oom_distinguishes_allocation_failure_from_a_model_side_rejection():
    assert bench._is_oom({"error": "TT_FATAL ... Out of Memory: Not enough space"})
    assert not bench._is_oom({"error": "ValueError: max_seq_len exceeds rope scaling window"})
    assert not bench._is_oom(None)


def test_bench_decode_stop_after_oom_breaks_right_after_the_first_error_cell(monkeypatch):
    monkeypatch.setattr(bench, "CONTEXTS", (1024, 2048, 4096))

    def fake_decode_one(md, kind, palu_dir, ctx):
        if ctx >= 2048:
            raise RuntimeError("Out of Memory: Not enough space")
        return {"p50_ms": 1.0, "min_ms": 1.0, "iters": 10}

    monkeypatch.setattr(bench, "_decode_one", fake_decode_one)
    mesh = _FakeMesh()

    r = bench.bench_decode(mesh, "stock", None, stop_after_oom=True)

    assert set(r) == {1024, 2048}  # 4096 never attempted
    assert "error" in r[2048]
    assert mesh.resets == 0  # never reopened for a context that will not run


def test_latent_dtype_flags_resolve_to_ttnn_formats():
    from palu_tt.bench import resolve_latent_dtypes
    import ttnn
    assert resolve_latent_dtypes("bfp8", None, None) == (ttnn.bfloat8_b, ttnn.bfloat8_b)
    assert resolve_latent_dtypes("bfp8", "bfp4", None) == (ttnn.bfloat4_b, ttnn.bfloat8_b)
    assert resolve_latent_dtypes("bfp4", None, "bfp8") == (ttnn.bfloat4_b, ttnn.bfloat8_b)


def test_kv_dtype_default_is_a_no_op_and_bfp4_only_moves_the_kv_cache():
    """Review finding (Important 1): the no-op guard in ``set_kv_cache_dtype`` is the single thing
    keeping every stage 2/2c/2d/2e row byte-identical under the new ``--kv-dtype`` default (bfp8) --
    it never rebuilds tt-transformers' own precision object when every decoder already reports that
    format. Delete the guard, or let a tt-metal bump ship
    ``model_params/Llama-3.1-8B-Instruct/accuracy_decoder_config.json``, and every committed row's
    precision object would silently change with nothing failing. Host-only: a ``SimpleNamespace``
    stands in for ``ModelArgs``, and ``DecodersPrecision.accuracy`` needs no device."""
    import ttnn
    from types import SimpleNamespace
    from models.tt_transformers.tt.model_config import DecodersPrecision, OpGroup, TensorGroup
    from palu_tt import model as pm

    def mk(n=32):
        opt = DecodersPrecision.accuracy(num_decoders=n, model_name="Llama-3.1-8B-Instruct")
        return SimpleNamespace(n_layers=n, model_name="Llama-3.1-8B-Instruct", optimizations=opt,
                               model_config={"DECODERS_OPTIMIZATIONS": opt})

    a, before = mk(), None
    before = a.optimizations
    pm.set_kv_cache_dtype(a, None)
    pm.set_kv_cache_dtype(a, ttnn.bfloat8_b)
    assert a.optimizations is before                      # the default never rebuilds
    pm.set_kv_cache_dtype(a, ttnn.bfloat4_b)
    assert a.optimizations is not before
    assert a.model_config["DECODERS_OPTIMIZATIONS"] is a.optimizations
    assert len(a.optimizations.decoder_optimizations) == 32
    for g in TensorGroup:                                 # only KV_CACHE moves
        want = ttnn.bfloat4_b if g is TensorGroup.KV_CACHE else \
            before.get_tensor_dtype(decoder_id=0, tensor=g)
        assert a.optimizations.get_tensor_dtype(decoder_id=0, tensor=g) == want
    for o in OpGroup:                                     # fidelity untouched
        assert (a.optimizations.decoder_optimizations[0].op_fidelity_settings[o]
                == before.decoder_optimizations[0].op_fidelity_settings[o])


def test_set_kv_cache_dtype_names_itself_correctly_after_a_rebuild():
    """Review finding (Minor 2): the rebuilt ``DecodersPrecision`` must not misreport itself. The old
    code mutated ``tensor_dtype_settings`` *after* ``ModelOptimizations.__init__`` had already computed
    ``_full_name`` from the pre-override dict, so a bfp4 rebuild's ``_full_name`` still claimed bfp8 for
    every decoder; and ``DecodersPrecision`` itself carries no ``__name__`` unless something sets one,
    even though the demo flows' docstring says they print it."""
    import ttnn
    from types import SimpleNamespace
    from models.tt_transformers.tt.model_config import DecodersPrecision
    from palu_tt import model as pm

    args = SimpleNamespace(n_layers=32, model_name="Llama-3.1-8B-Instruct",
                           optimizations=DecodersPrecision.accuracy(num_decoders=32, model_name="Llama-3.1-8B-Instruct"),
                           model_config={})
    args.model_config["DECODERS_OPTIMIZATIONS"] = args.optimizations
    pm.set_kv_cache_dtype(args, ttnn.bfloat4_b)

    conf = args.optimizations.decoder_optimizations[0]
    assert "kv_cache: bfp4" in conf._full_name
    assert args.optimizations.__name__ == "accuracy(kv_cache=bfp4)"


def test_apply_palu_flags_writes_every_paluruntime_switch_and_the_three_format_fields():
    """Review finding (Important 1, second test): R4 made ``apply_palu_flags`` the single place a
    run's switches become ``PaluConfig`` state, so ``bench.FUSED`` -- the dict every ``build_model`` /
    ``_attention_one`` caller splats -- must set every ``PaluRuntime`` switch, or a switch added to the
    dataclass without a matching ``apply_palu_flags`` parameter would silently keep its class default
    (exactly the bug R4 found: ``_attention_one``'s hand-copied block never set ``cfg.latent_dtype``).
    The switch set is enumerated from the dataclass rather than hand-listed, so a future field cannot
    be missed here."""
    import dataclasses

    from palu_tt import bench, model as pm
    from palu_tt.convert import PaluConfig, PaluRuntime

    # PaluRuntime fields apply_palu_flags does not own: recon_len is written per decode step by
    # set_recon_len, and check is a manual debug switch with no CLI flag.
    not_a_switch = {"recon_len", "check"}
    switch_fields = {f.name for f in dataclasses.fields(PaluRuntime)} - not_a_switch

    want = {"use_fused_k": True, "cs_recurrence": True, "fused_k_chunk": 256,
            "group_heads_per_core": 2, "rot_mode": "dst", "table_mcast": True,
            "latent_quant_in_prefill": False}
    assert switch_fields == set(want), \
        "a PaluRuntime field was added or renamed without updating this test (and, most likely, apply_palu_flags)"

    flags = dict(bench.FUSED, use_fused_k=True, cs_recurrence=True, fused_k_chunk=256, group_heads=2,
                rot_mode="dst", table_mcast=True, latent_quant_in_prefill=False,
                latent_dtype="bfp4", k_latent_dtype="bfp8", v_latent_dtype="bfp8")
    cfg = PaluConfig(rank_k=[64], rank_v=[64], cache_id="t")
    pm.apply_palu_flags(cfg, **flags)

    rt = cfg.runtime
    for name, expected in want.items():
        assert getattr(rt, name) == expected, name
    # the three format fields live on PaluConfig, not PaluRuntime, but are the other half of the
    # contract apply_palu_flags exists to guarantee
    assert (cfg.latent_dtype, cfg.k_latent_dtype, cfg.v_latent_dtype) == ("bfp4", "bfp8", "bfp8")
