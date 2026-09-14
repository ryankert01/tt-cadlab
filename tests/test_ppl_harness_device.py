import torch

from palu_tt import eval_ppl, hf, model as pm


def test_device_ppl_two_windows_is_sane(mesh_device):
    args, model, _ = pm.build_model(mesh_device, "stock", max_batch_size=1, max_seq_len=2048)
    ids = eval_ppl.wikitext2_ids(args.tokenizer)
    assert ids.shape[1] > 200_000
    r = eval_ppl.device_ppl(args, model, hf.snapshot_dir(), ids, seq_len=2048, n_windows=2)
    assert r["windows"] == 2 and r["tokens"] == 2 * 2047
    assert 3.0 < r["ppl"] < 20.0, r   # Llama-3.1-8B-Instruct wikitext2 ppl is ~7-9 at 2k context
