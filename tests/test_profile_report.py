import csv
from pathlib import Path

from palu_tt import profile_report


def _write_csv(path, rows):
    cols = ["OP CODE", "OP TYPE", "DEVICE ID", "CORE COUNT", "ATTRIBUTES", "DEVICE KERNEL DURATION [ns]",
            "INPUT_0_W_PAD[LOGICAL]", "INPUT_0_Z_PAD[LOGICAL]", "INPUT_0_Y_PAD[LOGICAL]", "INPUT_0_X_PAD[LOGICAL]",
            "INPUT_1_W_PAD[LOGICAL]", "INPUT_1_Z_PAD[LOGICAL]", "INPUT_1_Y_PAD[LOGICAL]", "INPUT_1_X_PAD[LOGICAL]",
            "OUTPUT_0_W_PAD[LOGICAL]", "OUTPUT_0_Z_PAD[LOGICAL]", "OUTPUT_0_Y_PAD[LOGICAL]", "OUTPUT_0_X_PAD[LOGICAL]"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "-") for c in cols})


def _op(code, dev, ns, in0=(), in1=(), out=(), typ="tt_dnn_device"):
    r = {"OP CODE": code, "OP TYPE": typ, "DEVICE ID": dev, "CORE COUNT": 64, "DEVICE KERNEL DURATION [ns]": ns}
    for pfx, shp in (("INPUT_0", in0), ("INPUT_1", in1), ("OUTPUT_0", out)):
        for d, v in zip(("W", "Z", "Y", "X")[-len(shp):] if shp else (), shp):
            r[f"{pfx}_{d}_PAD[LOGICAL]"] = f"{v}[{v}]"
    return r


def test_classify_covers_the_decode_step():
    c = profile_report.classify
    assert c("MatmulDeviceOperation", (1, 1, 32, 4096), (1, 1, 4096, 3072), (), "") == "qkv"
    assert c("MatmulDeviceOperation", (1, 1, 32, 4096), (1, 1, 4096, 7168), (), "") == "mlp"
    assert c("MatmulDeviceOperation", (1, 1, 32, 7168), (1, 1, 7168, 4096), (), "") == "mlp"
    assert c("MatmulDeviceOperation", (1, 1, 32, 2048), (1, 1, 2048, 4096), (), "") == "wo"
    assert c("MatmulDeviceOperation", (1, 1, 32, 4096), (1, 1, 4096, 64128), (), "") == "lm_head"
    assert c("MatmulDeviceOperation", (32, 4, 1024, 32), (32, 4, 32, 128), (32, 4, 1024, 128), "") == "k_recon"
    assert c("RotaryEmbeddingLlamaDeviceOperation", (1, 128, 1024, 128), (1, 1, 1024, 128), (), "") == "k_recon"
    assert c("RotaryEmbeddingLlamaDeviceOperation", (1, 32, 16, 128), (1, 1, 32, 128), (), "") == "heads"
    assert c("TypecastDeviceOperation", (32, 4, 1024, 128), (), (), "") == "k_recon"
    assert c("PaluSdpaDecodeDeviceOperation", (), (), (), "") == "attention"
    assert c("ReduceScatterMinimalAsyncDeviceOperation", (), (), (), "") == "ccl"
    assert c("SomethingNewDeviceOperation", (), (), (), "") == "other"


def test_aggregate_windows_on_signposts_and_divides_by_steps(tmp_path):
    csv_path = tmp_path / "ops_perf_results_x.csv"
    rows = [_op("MatmulDeviceOperation", 0, 999_000, (1, 1, 32, 4096), (1, 1, 4096, 3072))]   # warm-up: outside the window
    rows.append({"OP CODE": "start", "OP TYPE": "signpost", "DEVICE ID": "-", "DEVICE KERNEL DURATION [ns]": "-"})
    for _ in range(2):                                                                       # two steps, both chips
        for dev in (0, 1):
            rows.append(_op("MatmulDeviceOperation", dev, 2_000_000, (1, 1, 32, 4096), (1, 1, 4096, 3072)))
            rows.append(_op("SdpaDecodeDeviceOperation", dev, 3_000_000, (1, 32, 16, 128), (32, 4, 4096, 128)))
    rows.append({"OP CODE": "stop", "OP TYPE": "signpost", "DEVICE ID": "-", "DEVICE KERNEL DURATION [ns]": "-"})
    _write_csv(csv_path, rows)
    res = profile_report.aggregate(str(csv_path), steps=2)
    assert abs(res["device_ms_per_step"] - 5.0) < 1e-9 and abs(res["device_ms_per_step_chip1"] - 5.0) < 1e-9
    assert res["ops_per_step"] == 2
    assert abs(res["categories"]["qkv"]["ms"] - 2.0) < 1e-9 and abs(res["categories"]["attention"]["ms"] - 3.0) < 1e-9
    assert res["ops"][0]["op"] == "SdpaDecodeDeviceOperation" and res["ops"][0]["in1"] == [32, 4, 4096, 128]
