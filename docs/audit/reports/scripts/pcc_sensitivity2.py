import torch, sys
sys.path.insert(0, sys.argv[1])
from pcc_sensitivity import *   # reuses helpers; module-level run() calls are guarded below

def run2(batch, S, chunk, pos_mode):
    torch.manual_seed(213919)
    q, k_lat, u_k, v, pos, cos, sin = inputs(batch, S, 64, 64)
    if pos_mode == 'all_full': pos[:] = S - 1
    Tm = T_interleaved(); scale = HD ** -0.5
    K0 = torch.einsum('bhsr,hrd->bhsd', k_lat, u_k); s = torch.arange(S)
    ref = attend(q, K0 * cos[s][None, None] + (K0 @ Tm) * sin[s][None, None], v, pos, scale)
    nostep = attend(q, K0 * cos[s % chunk][None, None] + (K0 @ Tm) * sin[s % chunk][None, None], v, pos, scale)
    sg = torch.ones(HD); sg[64:] = -1
    lowflip = attend(q, K0 * cos[s][None, None] + (K0 @ Tm) * sin[s][None, None] * sg, v, pos, scale)
    # per-row output std: the mechanism behind the flattened PCC
    stds = [(int(pos[b]), float(ref[0, b].std())) for b in range(batch)]; stds.sort()
    print(f"batch={batch} S={S} chunk={chunk} pos={pos_mode}: no-Phase2-stepping PCC={pcc(ref, nostep):.5f}  low-freq-sign-flip PCC={pcc(ref, lowflip):.5f}  "
          f"row std: {stds[0]} ... {stds[-1]}")

for (b, S, c) in [(32, 1024, 128), (32, 1024, 512), (8, 8192, 128), (32, 2048, 128), (16, 2048, 128)]:
    for pm in ('test', 'all_full'):
        run2(b, S, c, pm)
