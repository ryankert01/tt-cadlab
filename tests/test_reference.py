import torch
from transformers import LlamaConfig, LlamaForCausalLM

from palu_tt import convert, reference

CFG = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
           intermediate_size=128, vocab_size=256, max_position_embeddings=512, rope_theta=10000.0,
           rms_norm_eps=1e-5, tie_word_embeddings=False)


def _tiny_model(tmp_path):
    m = LlamaForCausalLM(LlamaConfig(**CFG)).float().eval()
    m.save_pretrained(tmp_path / "hf", safe_serialization=True)
    return m, tmp_path / "hf"


def test_ref_attention_identity_matches_hf_layer(tmp_path):
    m, hf_dir = _tiny_model(tmp_path)
    attn = m.model.layers[0].self_attn
    w = {"q": attn.q_proj.weight.data, "k": attn.k_proj.weight.data, "v": attn.v_proj.weight.data, "o": attn.o_proj.weight.data}
    lr = convert.identity_lowrank(w, 2, 16)
    ref = reference.PaluRefAttention(CFG, w, lr)
    x = torch.randn(1, 8, 64)
    pos = torch.arange(8)[None]
    cos, sin = m.model.rotary_emb(x, pos)
    y_hf, _ = attn(x, position_embeddings=(cos, sin), attention_mask=None)
    y = ref.forward(x, start_pos=0)
    assert torch.allclose(y, y_hf, atol=1e-4)
    # decode continues from the cache: prefill 8 then one token must equal a fresh 9-token prefill
    x9 = torch.cat([x, torch.randn(1, 1, 64)], dim=1)
    y_last = ref.forward(x9[:, 8:], start_pos=8)
    ref2 = reference.PaluRefAttention(CFG, w, lr)
    y_full = ref2.forward(x9, start_pos=0)
    assert torch.allclose(y_last, y_full[:, 8:], atol=1e-4)
    assert ref.cache_k_lat.shape == (1, 2, 9, 16)


def test_cpu_lowrank_llama_identity_equals_hf(tmp_path):
    m, hf_dir = _tiny_model(tmp_path)
    lm = reference.build_cpu_lowrank_llama(hf_dir, None)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        a = m(ids).logits
        b = lm(ids).logits
    assert torch.allclose(a, b, atol=1e-4)


def test_cpu_lowrank_llama_uses_palu_factors(tmp_path):
    from palu_tt import compress
    m, hf_dir = _tiny_model(tmp_path)
    palu_dir = compress.compress_attention(hf_dir, tmp_path / "palu", rank_k=8, rank_v=8)
    lm = reference.build_cpu_lowrank_llama(hf_dir, palu_dir)
    kp = lm.model.layers[0].self_attn.k_proj
    assert isinstance(kp, reference.LowRankProj) and kp.VT.shape == (2 * 8, 64)
    ids = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        assert torch.isfinite(lm(ids).logits).all()


def test_identity_lowrank_reports_group_size_one():
    from palu_tt import convert
    w = {"k": torch.randn(2 * 16, 64), "v": torch.randn(2 * 16, 64)}
    assert convert.identity_lowrank(w, 2, 16).group_size == 1
