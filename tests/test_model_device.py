import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from palu_tt import generate, hf, model as pm, reference

PROMPT = "The Tenstorrent Wormhole chip is"
NEW = 32


def _ids(model_args, text):
    return model_args.tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def _cpu_greedy(lm, ids, max_new):
    ids = torch.tensor([ids])
    with torch.no_grad():
        out = lm.generate(ids, max_new_tokens=max_new, do_sample=False)
    return out[0, ids.shape[1]:].tolist()


def test_identity_path_matches_stock_model(mesh_device):
    args_s, stock, _ = pm.build_model(mesh_device, "stock", max_batch_size=1, max_seq_len=2048)
    args_p, palu, cfg = pm.build_model(mesh_device, "palu", palu_dir=None, max_batch_size=1, max_seq_len=2048)
    ids = _ids(args_s, PROMPT)
    ls = generate.prefill_last_logits(args_s, stock, torch.tensor([ids]))
    lp = generate.prefill_last_logits(args_p, palu, torch.tensor([ids]))
    ok, msg = comp_pcc(ls, lp, 0.999)
    assert ok, msg
    ts = generate.greedy(args_s, stock, None, ids, NEW)
    tp = generate.greedy(args_p, palu, cfg, ids, NEW)
    logger.info(f"stock: {args_s.tokenizer.decode(ts)}\npalu-identity: {args_p.tokenizer.decode(tp)}")
    assert ts == tp


def _teacher_forced_match(args, model, cfg, ids, ref_tokens):
    """Fraction of steps where the TT model's argmax equals the reference's next token, when fed the
    reference's own tokens. Free-running greedy cannot be compared token-by-token: one differing token
    sends the two models down different continuations (measured below, and for the stock model too)."""
    hits = int(int(generate.prefill_last_logits(args, model, torch.tensor([ids])).argmax()) == ref_tokens[0])
    pos = len(ids)
    for i in range(1, len(ref_tokens)):
        hits += int(int(generate.decode_step(args, model, cfg, ref_tokens[i - 1], pos).argmax()) == ref_tokens[i])
        pos += 1
    return hits / len(ref_tokens)


@pytest.mark.slow
def test_r64_model_tracks_cpu_lowrank_model(mesh_device):
    args, palu, cfg = pm.build_model(mesh_device, "palu", palu_dir="compressed/svd_r64", max_batch_size=1, max_seq_len=2048)
    ids = _ids(args, PROMPT)
    lm = reference.build_cpu_lowrank_llama(hf.snapshot_dir(), "compressed/svd_r64")
    with torch.no_grad():
        cpu_logits = lm(torch.tensor([ids])).logits[0, -1]
    tt_logits = generate.prefill_last_logits(args, palu, torch.tensor([ids]))
    # 0.95: tt_transformers' own full-model test accepts 0.954 (perf) / 0.965 (accuracy) for bfp8
    # Llama-3.1-8B against its fp32 reference (tests/test_model.py:final_model_pcc).
    ok, msg = comp_pcc(cpu_logits, tt_logits, 0.95)
    logger.info(msg)
    assert ok, msg
    cpu_tokens = _cpu_greedy(lm, ids, NEW)
    tt_tokens = generate.greedy(args, palu, cfg, ids, NEW)
    free = sum(a == b for a, b in zip(cpu_tokens, tt_tokens)) / NEW
    match = _teacher_forced_match(args, palu, cfg, ids, cpu_tokens)
    logger.info(f"cpu: {args.tokenizer.decode(cpu_tokens)}\ntt : {args.tokenizer.decode(tt_tokens)}\n"
                f"teacher-forced match={match:.2f} free-running match={free:.2f}")
    assert match >= 0.75  # same bar tt_transformers' own tests accept for bfp8 models
