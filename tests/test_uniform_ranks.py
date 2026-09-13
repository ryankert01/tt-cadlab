import json

from palu_tt import uniform_ranks


def test_writer_emits_one_entry_per_projection_with_the_uniform_rank(tmp_path):
    p = tmp_path / "ranks.json"
    uniform_ranks.write(p, n_layers=3, groups_per_matrix=8, r_k=64, r_v=32)
    d = json.loads(p.read_text())
    assert set(d) == {f"model.layers.{i}.self_attn.{m}" for i in range(3) for m in ("k_proj", "v_proj")}
    assert d["model.layers.1.self_attn.k_proj"] == [64] * 8
    assert d["model.layers.2.self_attn.v_proj"] == [32] * 8
