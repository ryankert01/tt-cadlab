import json

from palu_tt import uniform_ranks


def test_writer_emits_one_entry_per_projection_with_the_uniform_rank(tmp_path):
    p = tmp_path / "ranks.json"
    uniform_ranks.write(p, n_layers=3, groups_per_matrix=8, r_k=64, r_v=32)
    d = json.loads(p.read_text())
    assert set(d) == {f"model.layers.{i}.self_attn.{m}" for i in range(3) for m in ("k_proj", "v_proj")}
    assert d["model.layers.1.self_attn.k_proj"] == [64] * 8
    assert d["model.layers.2.self_attn.v_proj"] == [32] * 8


def test_split_dump_name_uses_whole_percentages():
    assert uniform_ranks.split_dump_name(0.25, 0.75) == "split_k25_v75"
    assert uniform_ranks.split_dump_name(0.5, 0.75) == "split_k50_v75"
    assert uniform_ranks.split_dump_name(0.5, 1.0) == "split_k50_v100"


def test_split_dump_name_rejects_a_ratio_that_is_not_a_whole_percentage():
    """Rounding 0.255 and 0.25 to the same name would give two different dumps one
    ``compressed/`` directory and one device weight-cache key."""
    import pytest
    for bad in (0.255, 0.0, 1.5, -0.25):
        with pytest.raises(ValueError):
            uniform_ranks.split_dump_name(bad, 0.75)
    with pytest.raises(ValueError):
        uniform_ranks.split_dump_name(0.25, 0.755)
