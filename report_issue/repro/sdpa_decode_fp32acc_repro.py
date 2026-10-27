# ttnn.transformer.scaled_dot_product_attention_decode returns wrong output with fp32_dest_acc_en=True
# when there are more than 32 query heads per KV head and the sequence is split across cores.
import torch, ttnn

def run(dev, nqh, fp32_acc, full_sync, B=1, S=4096, D=128):
    torch.manual_seed(0)
    q, k, v = [t.to(torch.bfloat16) for t in (torch.randn(1, B, nqh, D), torch.randn(B, 1, S, D), torch.randn(B, 1, S, D))]
    tq, tk, tv = [ttnn.from_torch(t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for t in (q, k, v)]
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq, tk, tv, cur_pos=[S - 1] * B, scale=D ** -0.5,
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc, packer_l1_acc=False, dst_full_sync_en=full_sync))
    got = ttnn.to_torch(out).float()[0, :, :nqh]
    p = torch.softmax(torch.einsum("bhd,bsd->bhs", q[0].float(), k[:, 0].float()) * D ** -0.5, dim=-1)
    ref = torch.einsum("bhs,bsd->bhd", p, v[:, 0].float())            # fp32 reference over the whole sequence
    pcc = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
    print(f"q_heads={nqh} fp32_dest_acc_en={str(fp32_acc):5s} dst_full_sync_en={str(full_sync):5s} | PCC {pcc:.4f}", flush=True)

dev = ttnn.open_device(device_id=0)
run(dev, 64, True, False)   # 2 query-head tiles, 4-tile fp32 DST window
run(dev, 64, True, True)    # 8-tile window
run(dev, 64, False, False)  # bf16 DST, 8-tile window
run(dev, 32, True, False)   # 1 query-head tile
ttnn.close_device(dev)
