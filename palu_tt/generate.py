"""Greedy generation and single-step helpers on top of tt_transformers' Transformer (batch-1 models)."""
import torch
import ttnn
from models.tt_transformers.tt.common import copy_host_to_device

from palu_tt import model as pm


def _pad_to(ids: torch.Tensor, multiple: int) -> torch.Tensor:
    L = ids.shape[1]
    P = ((L + multiple - 1) // multiple) * multiple
    return torch.nn.functional.pad(ids, (0, P - L), value=0)


def prefill_last_logits(args, model, ids: torch.Tensor) -> torch.Tensor:
    """ids [1, L] -> logits of position L-1 (fills user 0's cache with positions < L)."""
    L = ids.shape[1]
    padded = _pad_to(ids, 128)
    x, rot_g, rot_l, *_ = model.prepare_inputs_prefill(padded, start_pos=0)
    tt_logits = model.ttnn_prefill_forward(x, rot_mats_global=rot_g, rot_mats_local=rot_l, user_id=0,
                                           get_last_token=((L - 1) // 32) * 32)
    # prefill logits stay sharded over the vocab across chips -> concat, unlike the decode path
    out = model.process_output_prefill(tt_logits.cpu(), (L - 1) % 32).float()
    ttnn.deallocate(tt_logits)
    return out


def prefill_hidden_states(args, model, ids: torch.Tensor) -> torch.Tensor:
    """ids [1, L] (L multiple of 128) -> pre-final-norm hidden states [L, dim]."""
    x, rot_g, rot_l, *_ = model.prepare_inputs_prefill(ids, start_pos=0)
    h = model.ttnn_prefill_forward(x, rot_mats_global=rot_g, rot_mats_local=rot_l, user_id=0, get_last_token=-1)
    out = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float()[0, 0, : ids.shape[1], : args.dim]
    ttnn.deallocate(h)
    return out


def decode_step(args, model, cfg, token: int, pos: int) -> torch.Tensor:
    B = args.max_batch_size
    tokens = torch.full((B,), token, dtype=torch.long)
    current_pos = torch.full((B,), pos, dtype=torch.long)
    pm.set_recon_len(cfg, current_pos)
    host = model.prepare_decode_inputs_host(tokens, current_pos)
    tok_d, pos_d, rope_d, _ = copy_host_to_device(host, mesh_device=model.mesh_device)
    logits, _ = model.ttnn_decode_forward(tok_d, pos_d, rot_mat_idxs=rope_d)
    out = model.process_output_decode(logits, B=B)[0, 0]
    ttnn.deallocate(logits)
    return out


def greedy(args, model, cfg, prompt_ids, max_new: int):
    ids = torch.tensor([prompt_ids])
    nxt = int(prefill_last_logits(args, model, ids).argmax())
    out = [nxt]
    pos = ids.shape[1]
    for _ in range(max_new - 1):
        nxt = int(decode_step(args, model, cfg, nxt, pos).argmax())
        out.append(nxt)
        pos += 1
    return out
