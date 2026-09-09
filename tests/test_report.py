"""CPU test for the results-table renderer: it must read the real JSON schema.

The schema exercised here is the one ``palu_tt.bench``/``palu_tt.eval_ppl`` actually write:
string context keys (``json.dumps`` stringifies the int keys) and error cells for the
contexts whose model does not allocate.
"""
import json

from palu_tt import report


def _write(raw, name, obj):
    (raw / f"{name}.json").write_text(json.dumps(obj))


def test_main_renders_cells_errors_and_missing_rows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw, "ppl_baseline", {"ppl": 7.25, "windows": 141, "seq_len": 2048,
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
    # quality/memory table: ranks default to head_dim, ppl and both KV columns rendered
    assert "| baseline | 128 | 128 | 7.250 |" in md
    assert "| 1088 |" in md
    assert f"| {1088 * report.LAYERS * report.CHIPS} |" in md  # 69632 B/token for the whole model
    # timing table: p50 and min side by side, and the OOM cell rendered, not crashed on
    assert "| baseline | 1024 | 12.3 / 10.0 |" in md
    assert "| baseline | 8192 | OOM |" in md
    # rows with no files at all still appear, as dashes
    assert any(line.startswith("| palu_full_50 |") and "| - |" in line for line in md.splitlines())
    # missing files never raise and never invent numbers (not even the default ranks)
    assert "| identity | - | - | - | - |" in md


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
