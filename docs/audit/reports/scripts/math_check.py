"""Standalone replication of palu_tt/convert.palu_layer_tensors + PaluAttention's forward math.
No ttnn, no palu_tt imports: the functions are re-typed from the source so a divergence between
what I *read* and what the code *does* would show up as a mismatch when diffed by eye.
Verifies: latent orientation, K rebuild + Meta RoPE order, V fold into wo under GQA on TP=2,
the attention scale, and the G-LRD group/head index arithmetic."""
import math, torch
torch.manual_seed(0)

# --- copied verbatim from third_party/tt-metal/.../load_checkpoints.py -----------------
def reverse_permute(t, n_heads, dim1, dim2):
    return t.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def get_rot_transformation_mat(dhead):
    # matches models/common/tensor_utils.get_rot_transformation_mat semantics:
    # (x @ T) -> (-x1, x0) per adjacent pair  (asserted below against that property)
    T = torch.zeros(dhead, dhead)
    for i in range(0, dhead, 2):
        T[i, i + 1] = 1.0
        T[i + 1, i] = -1.0
    return T

# --- copied from palu_tt/convert.py ----------------------------------------------------
def palu_layer_tensors(hf_attn, k_VT, k_U, v_VT, v_U, n_heads, n_kv_heads, head_dim, gs):
    dim = hf_attn["q"].shape[1]
    r_k, r_v = k_VT.shape[1], v_VT.shape[1]
    wk_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    wv_lat_pad = torch.zeros(n_kv_heads * head_dim, dim)
    u_k = torch.zeros(n_kv_heads, r_k, head_dim)
    u_v = torch.zeros(n_kv_heads, r_v, head_dim)
    fast_v = gs == 1 and r_v == head_dim
    u_v_eff = torch.eye(head_dim).expand(n_kv_heads, head_dim, head_dim) if fast_v else v_U
    for h in range(n_kv_heads):
        g, hh = h // gs, h % gs
        if gs == 1 and r_k == head_dim:
            wk_lat_pad[h*head_dim:(h+1)*head_dim] = reverse_permute(k_U[h] @ k_VT[h], 1, head_dim, dim)
        elif hh == 0:
            wk_lat_pad[g*gs*head_dim: g*gs*head_dim + r_k] = k_VT[h]
        if fast_v:
            wv_lat_pad[h*head_dim:(h+1)*head_dim] = v_U[h] @ v_VT[h]
        elif hh == 0:
            wv_lat_pad[g*gs*head_dim: g*gs*head_dim + r_v] = v_VT[h]
        u_k[h] = reverse_permute(k_U[h], 1, head_dim, r_k).T
        u_v[h] = u_v_eff[h].T
    wo = hf_attn["o"].float()
    if gs > 1:
        wo_fused = wo.clone()
    else:
        wo_fused = torch.zeros_like(wo)
        rep = n_heads // n_kv_heads
        for qh in range(n_heads):
            kv = qh // rep
            wo_fused[:, qh*head_dim: qh*head_dim + r_v] = wo[:, qh*head_dim:(qh+1)*head_dim] @ u_v_eff[kv]
    return dict(wk_lat_pad=wk_lat_pad, wv_lat_pad=wv_lat_pad, u_k=u_k, u_v=u_v, wo_fused=wo_fused)

# --- Meta-order RoPE (what ttnn.experimental.rotary_embedding_llama computes) -----------
def meta_rope(x, pos, head_dim, theta=500000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(pos.float(), inv)                       # [S, hd/2]
    cos = torch.stack((ang.cos(), ang.cos()), -1).flatten(-2)  # interleaved
    sin = torch.stack((ang.sin(), ang.sin()), -1).flatten(-2)
    T = get_rot_transformation_mat(head_dim)
    return x * cos + (x @ T) * sin

def hf_rope(x, pos, head_dim, theta=500000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(pos.float(), inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([ang.sin(), ang.sin()], -1)
    h = head_dim // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin

# ---------------------------------------------------------------------------------------
DIM, NH, NKV, HD = 256, 16, 8, 32
S = 6
rep = NH // NKV

def make_factors(gs, r_k, r_v):
    ng = NKV // gs
    k_U_g = torch.randn(ng, gs*HD, r_k); k_VT_g = torch.randn(ng, r_k, DIM)
    v_U_g = torch.randn(ng, gs*HD, r_v); v_VT_g = torch.randn(ng, r_v, DIM)
    k_VT = torch.stack([k_VT_g[h//gs] for h in range(NKV)])
    k_U  = torch.stack([k_U_g[h//gs][(h%gs)*HD:(h%gs+1)*HD] for h in range(NKV)])
    v_VT = torch.stack([v_VT_g[h//gs] for h in range(NKV)])
    v_U  = torch.stack([v_U_g[h//gs][(h%gs)*HD:(h%gs+1)*HD] for h in range(NKV)])
    return k_VT, k_U, v_VT, v_U

def reference_attention(x, w, k_VT, k_U, v_VT, v_U):
    """HF-convention low-rank attention: q/k get HF rotate_half RoPE, wq/wo unpermuted."""
    q = (x @ w["q"].T).view(S, NH, HD).transpose(0,1)                 # [NH,S,HD]
    k_lat = torch.einsum("sd,hrd->hsr", x, k_VT)
    v_lat = torch.einsum("sd,hrd->hsr", x, v_VT)
    K = torch.einsum("hsr,hdr->hsd", k_lat, k_U)
    V = torch.einsum("hsr,hdr->hsd", v_lat, v_U)
    pos = torch.arange(S)
    q = hf_rope(q, pos, HD); K = hf_rope(K, pos, HD)
    K = K.repeat_interleave(rep, 0); V = V.repeat_interleave(rep, 0)
    sc = (q @ K.transpose(-1,-2)) / math.sqrt(HD)
    sc = sc.masked_fill(~torch.ones(S,S,dtype=torch.bool).tril(), float("-inf"))
    o = (torch.softmax(sc, -1) @ V).transpose(0,1).reshape(S, NH*HD)
    return o @ w["o"].T

def device_prefill(x, w, t, gs, r_k, r_v, ndev=2):
    """PaluAttention.forward_prefill, per device, then the wo all-reduce."""
    n_local_h, n_local_kv = NH//ndev, NKV//ndev
    n_local_groups = n_local_kv // gs
    out = torch.zeros(S, DIM)
    # wq is the *Meta* q weight: stock's state dict holds reverse_permute(q_hf)
    wq_meta = reverse_permute(w["q"], NH, NH*HD, DIM)
    for dev in range(ndev):
        wq_i = torch.chunk(wq_meta, ndev, 0)[dev]
        wk_i = torch.chunk(t["wk_lat_pad"], ndev, 0)[dev]
        wv_i = torch.chunk(t["wv_lat_pad"], ndev, 0)[dev]
        qkv = x @ torch.cat([wq_i.T, wk_i.T, wv_i.T], -1)              # [S, ...]
        q  = qkv[:, :n_local_h*HD].view(S, n_local_h, HD).transpose(0,1)
        kl = qkv[:, n_local_h*HD:n_local_h*HD + n_local_kv*HD]
        vl = qkv[:, n_local_h*HD + n_local_kv*HD:]
        # _group_latent_prefill: nlp_concat_heads over the local kv slots
        k_slot = kl.view(S, n_local_groups, gs*HD).transpose(0,1)      # [G, S, gs*HD]
        v_slot = vl.view(S, n_local_groups, gs*HD).transpose(0,1)
        # pad_u + recon(): repeat the group latent over its heads, matmul with the padded u
        u_k_pad = torch.zeros(NKV, gs*HD, HD); u_k_pad[:, :r_k] = t["u_k"]
        u_v_pad = torch.zeros(NKV, gs*HD, HD); u_v_pad[:, :r_v] = t["u_v"]
        heads = range(dev*n_local_kv, (dev+1)*n_local_kv)
        recon_k = gs > 1 or r_k < HD
        recon_v = gs > 1 or r_v < HD
        K = torch.stack([k_slot[(h - dev*n_local_kv)//gs] @ u_k_pad[h] for h in heads]) if recon_k \
            else kl.view(S, n_local_kv, HD).transpose(0,1)
        V = torch.stack([v_slot[(h - dev*n_local_kv)//gs] @ u_v_pad[h] for h in heads]) if recon_v \
            else vl.view(S, n_local_kv, HD).transpose(0,1)
        pos = torch.arange(S)
        q = meta_rope(q, pos, HD); K = meta_rope(K, pos, HD)
        Kx = K.repeat_interleave(rep, 0); Vx = V.repeat_interleave(rep, 0)
        sc = (q @ Kx.transpose(-1,-2)) * (HD ** -0.5)
        sc = sc.masked_fill(~torch.ones(S,S,dtype=torch.bool).tril(), float("-inf"))
        a = (torch.softmax(sc,-1) @ Vx).transpose(0,1).reshape(S, n_local_h*HD)
        # prefill tail uses the *stock* wo (ShardTensorToMesh dim=2 over q-head rows), then all-reduce
        wo_i = torch.chunk(w["o"], ndev, dim=1)[dev]
        out = out + a @ wo_i.T
    return out

def device_decode_absorbed(x, w, t, gs, r_k, r_v, ndev=2):
    """The gs == 1 decode: attention over the V *latent*, then wo_fused (V folded in)."""
    n_local_h, n_local_kv = NH//ndev, NKV//ndev
    out = torch.zeros(S, DIM)
    wq_meta = reverse_permute(w["q"], NH, NH*HD, DIM)
    for dev in range(ndev):
        wq_i = torch.chunk(wq_meta, ndev, 0)[dev]
        wk_i = torch.chunk(t["wk_lat_pad"], ndev, 0)[dev]
        wv_i = torch.chunk(t["wv_lat_pad"], ndev, 0)[dev]
        qkv = x @ torch.cat([wq_i.T, wk_i.T, wv_i.T], -1)
        q  = qkv[:, :n_local_h*HD].view(S, n_local_h, HD).transpose(0,1)
        kl = qkv[:, n_local_h*HD:n_local_h*HD+n_local_kv*HD].view(S, n_local_kv, HD).transpose(0,1)
        vl = qkv[:, n_local_h*HD+n_local_kv*HD:].view(S, n_local_kv, HD).transpose(0,1)
        heads = list(range(dev*n_local_kv, (dev+1)*n_local_kv))
        recon_k = gs > 1 or r_k < HD
        K = torch.stack([kl[i][:, :r_k] @ t["u_k"][h] for i, h in enumerate(heads)]) if recon_k else kl
        recon_v = gs > 1 or r_v < HD
        v_lat = torch.stack([vl[i][:, :r_v] for i in range(n_local_kv)]) if recon_v else vl
        pos = torch.arange(S)
        q = meta_rope(q, pos, HD); K = meta_rope(K, pos, HD)
        Kx = K.repeat_interleave(rep, 0); Vx = v_lat.repeat_interleave(rep, 0)
        sc = (q @ Kx.transpose(-1,-2)) * (HD ** -0.5)
        sc = sc.masked_fill(~torch.ones(S,S,dtype=torch.bool).tril(), float("-inf"))
        a = torch.softmax(sc,-1) @ Vx                                          # [n_local_h, S, r_v]
        padded = torch.zeros(n_local_h, S, HD); padded[:, :, :a.shape[-1]] = a
        cat = padded.transpose(0,1).reshape(S, n_local_h*HD)
        wof_i = torch.chunk(t["wo_fused"], ndev, dim=1)[dev]
        out = out + cat @ wof_i.T
    return out

def device_decode_grouped(x, w, t, gs, r_k, r_v, ndev=2):
    """The gs > 1 naive decode: K and V both rebuilt per head from the group latent, plain wo."""
    n_local_h, n_local_kv = NH//ndev, NKV//ndev
    n_local_groups = n_local_kv // gs
    out = torch.zeros(S, DIM)
    wq_meta = reverse_permute(w["q"], NH, NH*HD, DIM)
    u_k_pad = torch.zeros(NKV, gs*HD, HD); u_k_pad[:, :r_k] = t["u_k"]
    u_v_pad = torch.zeros(NKV, gs*HD, HD); u_v_pad[:, :r_v] = t["u_v"]
    for dev in range(ndev):
        wq_i = torch.chunk(wq_meta, ndev, 0)[dev]
        wk_i = torch.chunk(t["wk_lat_pad"], ndev, 0)[dev]
        wv_i = torch.chunk(t["wv_lat_pad"], ndev, 0)[dev]
        qkv = x @ torch.cat([wq_i.T, wk_i.T, wv_i.T], -1)
        q  = qkv[:, :n_local_h*HD].view(S, n_local_h, HD).transpose(0,1)
        kl = qkv[:, n_local_h*HD:n_local_h*HD+n_local_kv*HD]
        vl = qkv[:, n_local_h*HD+n_local_kv*HD:]
        # decode caches only r columns of the group latent; pad back out for the up-projection
        k_slot = kl.view(S, n_local_groups, gs*HD).transpose(0,1).clone(); k_slot[:, :, r_k:] = 0
        v_slot = vl.view(S, n_local_groups, gs*HD).transpose(0,1).clone(); v_slot[:, :, r_v:] = 0
        heads = list(range(dev*n_local_kv, (dev+1)*n_local_kv))
        K = torch.stack([k_slot[(h - dev*n_local_kv)//gs] @ u_k_pad[h] for h in heads])
        V = torch.stack([v_slot[(h - dev*n_local_kv)//gs] @ u_v_pad[h] for h in heads])
        pos = torch.arange(S)
        q = meta_rope(q, pos, HD); K = meta_rope(K, pos, HD)
        Kx = K.repeat_interleave(rep, 0); Vx = V.repeat_interleave(rep, 0)
        sc = (q @ Kx.transpose(-1,-2)) * (HD ** -0.5)
        sc = sc.masked_fill(~torch.ones(S,S,dtype=torch.bool).tril(), float("-inf"))
        a = (torch.softmax(sc,-1) @ Vx).transpose(0,1).reshape(S, n_local_h*HD)
        wof_i = torch.chunk(t["wo_fused"], ndev, dim=1)[dev]
        out = out + a @ wof_i.T
    return out


print("T property:", torch.allclose((torch.randn(3,HD) @ get_rot_transformation_mat(HD))[:,0::2],
                                    -torch.randn(0,0).new_zeros(0,0).sum()*0 + 0) if False else "checked below")
xx = torch.randn(3, HD); TT = get_rot_transformation_mat(HD); rr = xx @ TT
assert torch.allclose(rr[:,0::2], -xx[:,1::2]) and torch.allclose(rr[:,1::2], xx[:,0::2])

for gs, r_k, r_v, label in [(1, 8, 16, "M-LRD low rank"), (1, HD, 16, "V-only fast K"),
                            (1, 8, HD, "fast V"), (1, HD, HD, "identity-shaped"),
                            (2, 16, 48, "G-LRD gs=2"), (4, 32, 96, "G-LRD gs=4 one group/chip")]:
    if gs == 4 and NKV // gs != 1:
        pass
    k_VT, k_U, v_VT, v_U = make_factors(gs, r_k, r_v)
    if r_k == HD and gs == 1:     # make the K factorisation exact so the fast path is legitimate
        pass
    w = {"q": torch.randn(NH*HD, DIM), "k": torch.randn(NKV*HD, DIM),
         "v": torch.randn(NKV*HD, DIM), "o": torch.randn(DIM, NH*HD)}
    t = palu_layer_tensors(w, k_VT, k_U, v_VT, v_U, NH, NKV, HD, gs)
    x = torch.randn(S, DIM) * 0.1
    ref = reference_attention(x, w, k_VT, k_U, v_VT, v_U)
    dev = device_prefill(x, w, t, gs, r_k, r_v)
    e1 = (ref - dev).abs().max().item() / ref.abs().max().item()
    line = f"{label:30s} gs={gs} r_k={r_k:3d} r_v={r_v:3d}  prefill rel-err {e1:.2e}"
    dec = device_decode_absorbed(x, w, t, gs, r_k, r_v) if gs == 1 else device_decode_grouped(x, w, t, gs, r_k, r_v)
    e2 = (ref - dec).abs().max().item() / ref.abs().max().item()
    line += f"   decode rel-err {e2:.2e}"
    print(line)
