"""Pure-torch model of tests/test_palu_sdpa_device.py::_check inputs; measures the PCC that
deliberately-wrong kernels would score against the fp32 reference, to see whether the 0.99 / 0.995
bars can tell them apart. No ttnn, no palu_tt imports."""
import torch, math
torch.manual_seed(213919)
HD, NH, NKV, THETA = 128, 16, 4, 500000.0

def tables(S):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
    t = torch.arange(S + 1024).float()
    fr = torch.outer(t, inv)
    cos, sin = fr.cos(), fr.sin()
    cos = torch.stack([cos, cos], -1).flatten(-2); sin = torch.stack([sin, sin], -1).flatten(-2)
    return cos, sin

def T_interleaved():
    T = torch.zeros(HD, HD); T[torch.arange(0, HD, 2), torch.arange(1, HD, 2)] = 1; T[torch.arange(1, HD, 2), torch.arange(0, HD, 2)] = -1
    return T

def T_halfsplit():
    T = torch.zeros(HD, HD); h = HD // 2
    T[torch.arange(0, h), torch.arange(h, HD)] = 1; T[torch.arange(h, HD), torch.arange(0, h)] = -1
    return T

def bf(x): return x.to(torch.bfloat16).float()

def inputs(batch, S, r_k, r_v):
    q = bf(torch.randn(1, batch, NH, HD)); k_lat = bf(torch.randn(batch, NKV, S, r_k))
    u_k = bf(torch.randn(NKV, r_k, HD) / r_k ** 0.5); v = bf(torch.randn(batch, NKV, S, r_v))
    pos = torch.randint(0, S, (batch,)); pos[0] = S - 1
    if batch >= 4: pos[1], pos[2], pos[3] = 0, 127, 128
    cos, sin = tables(S)
    return q, k_lat, u_k, v, pos, bf(cos), bf(sin)

def attend(q, K, V, pos, scale):
    B, nkv, S, _ = K.shape; rep = NH // nkv
    K = K.repeat_interleave(rep, 1); V = V.repeat_interleave(rep, 1)
    Q = q.permute(1, 2, 0, 3)
    mask = torch.zeros(B, 1, 1, S)
    for b in range(B): mask[b, :, :, int(pos[b]) + 1:] = float('-inf')
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask, scale=scale).permute(2, 0, 1, 3)

def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

def run(batch, S, r_k, r_v, chunk):
    q, k_lat, u_k, v, pos, cos, sin = inputs(batch, S, r_k, r_v)
    Tm = T_interleaved(); scale = HD ** -0.5
    K0 = torch.einsum('bhsr,hrd->bhsd', k_lat, u_k)
    s = torch.arange(S)
    def rot(K, T, sgn=1.0, shift=0, mask_dims=None):
        c = cos[(s + shift).clamp(max=cos.shape[0]-1)][None, None]; sn = sin[(s + shift).clamp(max=sin.shape[0]-1)][None, None]
        sgn_v = torch.full((HD,), sgn)
        if mask_dims is not None: sgn_v = torch.ones(HD); sgn_v[mask_dims] = sgn
        return K * c + (K @ T) * sn * sgn_v
    ref = attend(q, rot(K0, Tm), v, pos, scale)
    cases = {}
    cases['bf16-rounded K (what the device-vs-stock comparison sees)'] = attend(q, bf(rot(K0, Tm)), v, pos, scale)
    cases['sin sign flipped on all of K (R_-s instead of R_s)'] = attend(q, rot(K0, Tm, sgn=-1.0), v, pos, scale)
    cases['sin sign flipped on the 32 highest-frequency pairs only'] = attend(q, rot(K0, Tm, sgn=-1.0, mask_dims=slice(0, 64)), v, pos, scale)
    cases['sin sign flipped on the 32 lowest-frequency pairs only'] = attend(q, rot(K0, Tm, sgn=-1.0, mask_dims=slice(64, 128)), v, pos, scale)
    cases['HF half-split pairing instead of interleaved'] = attend(q, rot(K0, T_halfsplit()), v, pos, scale)
    cases['off-by-one chunk in the query state (K angle s + chunk)'] = attend(q, rot(K0, Tm, shift=chunk), v, pos, scale)
    cases['no k_chunk_start seeding (K angle s mod chunk)'] = attend(q, K0 * cos[s % chunk][None, None] + (K0 @ Tm) * sin[s % chunk][None, None], v, pos, scale)
    # wrong GQA map: q head j -> kv head j % NKV instead of j // rep
    Kr = rot(K0, Tm); idx = torch.arange(NH) % NKV
    cases['GQA map j % nkv instead of j // rep'] = attend(q, Kr[:, idx], v[:, idx] if False else v.repeat_interleave(NH // NKV, 1)[:, torch.arange(NH)], pos, scale) if False else None
    Q = q.permute(1, 2, 0, 3); Kg = Kr[:, idx]; Vg = v[:, idx]
    mask = torch.zeros(batch, 1, 1, S)
    for b in range(batch): mask[b, :, :, int(pos[b]) + 1:] = float('-inf')
    cases['GQA map j % nkv instead of j // rep'] = torch.nn.functional.scaled_dot_product_attention(Q, Kg, Vg, attn_mask=mask, scale=scale).permute(2, 0, 1, 3)
    # heads swapped inside a group (w_rot block of head h+1 used for head h) with per-head latents
    perm = torch.tensor([1, 0, 3, 2]); Ksw = rot(torch.einsum('bhsr,hrd->bhsd', k_lat, u_k[perm]), Tm)
    cases['w_rot block of the neighbouring head used (head swap)'] = attend(q, Ksw, v, pos, scale)
    cases['K not rotated at all'] = attend(q, K0, v, pos, scale)
    print(f"\n== batch={batch} S={S} r_k={r_k} r_v={r_v} chunk={chunk}  pos={pos.tolist()[:8]}...")
    for name, out in cases.items():
        p = pcc(ref, out)
        # per-row PCC for the rows with the longest and shortest context
        rows = [(int(pos[b]), pcc(ref[0, b], out[0, b])) for b in range(batch)]
        rows.sort()
        lo = ' '.join(f"{p_:.3f}@{ps}" for ps, p_ in rows[:3]); hi = ' '.join(f"{p_:.3f}@{ps}" for ps, p_ in rows[-2:])
        print(f"  {p:.5f}  {'PASS@0.99' if p >= 0.99 else 'fail     '}  {name}   [rows: {lo} ... {hi}]")

if __name__ == "__main__":
    run(32, 1024, 64, 64, 512)
    run(1, 8192, 64, 64, 512)
    run(32, 8192, 32, 64, 512)
