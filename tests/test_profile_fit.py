from pathlib import Path

from palu_tt import profile_fit

RAW = Path("docs/results/raw")


def _by_sig(rows):
    return {r["signature"]: r for r in rows}


def test_fit_reproduces_the_stage2d_table():
    """docs/results/stage2d.md section 4: stock 47.7 + 88.4 x chunks; fused r_k 32 / r_v 128
    118.1 + 114.3 x chunks; fused r_k 64 / r_v 128 129.5 + 123.1 x chunks (two significant figures).

    The stock signature is fitted from baseline + palu_full_50 only (naive palu_full_50 runs the stock
    SdpaDecode op and is the only source of a stock point at x=16, since baseline has no 8192 JSON):
    47.8 / 88.5. The fused rows are left out of the stock fit on purpose: palu_full_70_fk2's
    uncompressed layers run the stock op as a count=1 signature (one noisy layer per context) and the
    unweighted fit counts those points like any other, which is what shifted the intercept when all
    four rows were fitted together."""
    stock = _by_sig(profile_fit.fit_signatures(RAW, ["baseline", "palu_full_50"], [1024, 4096, 8192]))[
        "SdpaDecodeDeviceOperation r_k=128 out=128"]
    assert abs(stock["fixed_us"] - 47.7) < 1.0 and abs(stock["per_chunk_us"] - 88.4) < 1.0
    rows = profile_fit.fit_signatures(RAW, ["baseline", "palu_full_50", "palu_full_50_fk2", "palu_full_70_fk2"],
                                      [1024, 4096, 8192])
    s = _by_sig(rows)
    f32 = s["PaluSdpaDecodeDeviceOperation r_k=32 out=128"]
    assert abs(f32["fixed_us"] - 118.1) < 1.5 and abs(f32["per_chunk_us"] - 114.3) < 1.0
    f64 = s["PaluSdpaDecodeDeviceOperation r_k=64 out=128"]
    assert abs(f64["fixed_us"] - 129.5) < 1.5 and abs(f64["per_chunk_us"] - 123.1) < 1.0
    assert abs(f32["per_item_per_chunk_us"] - f32["per_chunk_us"] / 2) < 1e-9


def test_fit_skips_missing_contexts_and_needs_two_points():
    rows = profile_fit.fit_signatures(RAW, ["baseline"], [1024, 4096, 8192])   # no stock JSON at 8192
    stock = _by_sig(rows)["SdpaDecodeDeviceOperation r_k=128 out=128"]
    assert [p[0] for p in stock["points"]] == [2, 8]
