#!/usr/bin/env python3
"""Draw the figures for docs/report_palu_on_tt.md (what Palu changes in the model, and how the model
runs on the n300: the vanilla ttnn path of stage 1 and the fused kernel of stage 2) as standalone SVG
files under docs/figures/report_*.svg, with the same primitives and palette as the kernel guide.

    python scripts/report_figures.py            # writes docs/figures/report_*.svg
    python scripts/report_figures.py --png DIR  # also rasterises each figure into DIR (needs cairosvg)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.kernel_figures import (C_CB, C_DST, C_FPU, C_IDLE, C_K, C_L1, C_NOC, C_PALU, C_SFPU, INK, LINE,
                                    MUTED, PLATE, Fig)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"


# =============================================================================================
# R1. What Palu changes in one attention layer
# =============================================================================================
def fig_palu_architecture() -> Fig:
    f = Fig(940, 560, "One attention layer, one KV head: what the baseline caches and what Palu caches instead")
    f.text(14, 24, "What Palu changes: the K and V projections become low-rank, and the cache stores the latent", size=15, weight="bold")

    # ---- left: stock -------------------------------------------------------------------------
    lx = 30
    f.text(lx, 52, "Baseline attention: the unmodified model (per KV head)", size=12.5, weight="bold")
    f.box(lx, 64, 70, 34, "x", "1 x 4096", role=C_NOC, size=11)
    f.arrow(lx + 70, 81, lx + 100, 81)
    f.box(lx + 100, 64, 92, 34, "W_k", "4096 x 128", role=C_K, size=11, mono_sub=True)
    f.arrow(lx + 192, 81, lx + 222, 81)
    f.box(lx + 222, 64, 76, 34, "RoPE_p", "position p", role=C_SFPU, size=11)
    f.arrow(lx + 298, 81, lx + 328, 81)
    f.box(lx + 328, 60, 96, 42, "K cache", "S x 128 bfp8", role=C_K, size=11, mono_sub=True)

    f.box(lx, 124, 70, 34, "x", "1 x 4096", role=C_NOC, size=11)
    f.arrow(lx + 70, 141, lx + 100, 141)
    f.box(lx + 100, 124, 92, 34, "W_v", "4096 x 128", role=C_K, size=11, mono_sub=True)
    f.arrow(lx + 192, 141, lx + 328, 141)
    f.box(lx + 328, 120, 96, 42, "V cache", "S x 128 bfp8", role=C_K, size=11, mono_sub=True)

    f.lines(lx, 190, [
        "decode, per step:  scores = q_p . K^T,  out = softmax(scores) . V,  y = out . W_o",
        "cache per token, per layer, per chip (4 KV heads): 4 x (136 + 136) B = 1088 B",
        "baseline on the n300, batch 32: 68 KB/token model-wide -> max context 4096",
    ], size=10.5, lh=15, color=MUTED)

    # ---- right: Palu -------------------------------------------------------------------------
    rx = 500
    f.text(rx, 52, "Palu (M-LRD, per KV head)", size=12.5, weight="bold")
    f.box(rx, 64, 70, 34, "x", "1 x 4096", role=C_NOC, size=11)
    f.arrow(rx + 70, 81, rx + 100, 81)
    f.box(rx + 100, 64, 92, 34, "VT_k^T", "4096 x r_k", role=C_PALU, size=11, mono_sub=True)
    f.arrow(rx + 192, 81, rx + 222, 81)
    f.box(rx + 222, 60, 96, 42, "h_k cache", "S x r_k bfp8", role=C_PALU, size=11, mono_sub=True)
    f.arrow(rx + 318, 81, rx + 346, 81)
    f.rect(rx + 340, 54, 96, 72, (PLATE, C_DST[1]), r=6, sw=1.4, dash="4 3")
    f.text(rx + 388, 46, "the expensive part", size=9.5, anchor="middle", color=C_DST[1])
    f.box(rx + 346, 64, 84, 34, "h_k . U_k^T", "r_k x 128", role=C_FPU, size=10.5, mono_sub=True)
    f.text(rx + 388, 110, "rebuild K, then RoPE_p", size=9, anchor="middle", color=C_DST[1])
    f.text(rx + 388, 121, "before every q . K^T", size=9, anchor="middle", color=C_DST[1])

    f.box(rx, 134, 70, 34, "x", "1 x 4096", role=C_NOC, size=11)
    f.arrow(rx + 70, 151, rx + 100, 151)
    f.box(rx + 100, 134, 92, 34, "VT_v^T", "4096 x r_v", role=C_PALU, size=11, mono_sub=True)
    f.arrow(rx + 192, 151, rx + 222, 151)
    f.box(rx + 222, 130, 96, 42, "h_v cache", "S x r_v bfp8", role=C_PALU, size=11, mono_sub=True)
    f.arrow(rx + 318, 151, rx + 346, 151)
    f.box(rx + 346, 134, 84, 34, "absorbed", "W_o' = U_v W_o", role=C_FPU, size=10.5, mono_sub=True)

    f.lines(rx, 196, [
        "W ~ U . VT (rank r) from a whitened SVD; the cache keeps h = x . VT^T",
        "V never comes back: softmax . h_v . U_v^T . W_o = softmax . h_v . W_o'",
        "K must come back: RoPE sits between the projection and q . K^T",
    ], size=10.5, lh=15, color=MUTED)

    # ---- bottom: the offline recipe and the sizes --------------------------------------------
    y0 = 262
    f.line(30, y0 - 12, 910, y0 - 12, color=LINE, sw=0.8, dash="2 3")
    f.text(30, y0 + 8, "How the factors are made (offline, on the CPU, once per model)", size=12.5, weight="bold")
    steps = [
        ("1. calibrate", "wikitext-2 sequences (1024 tokens) through the fp32 model; per-layer activation covariance; 32 or 256 Fisher samples", C_NOC),
        ("2. whiten", "SVD of W in the whitened basis (Sigma^1/2 W): the truncation error is measured on real activations, not on the weights", C_CB),
        ("3. rank search", "Fisher information per matrix; ranks are multiples of 32, chosen per layer for a target ratio", C_CB),
        ("4. export", "U (up-projection) and VT (down-projection) per head or per group; nothing else in the model changes", C_PALU),
    ]
    for i, (t, s, role) in enumerate(steps):
        x = 30 + i * 222
        f.rect(x, y0 + 20, 208, 92, role, r=6)
        f.text(x + 10, y0 + 40, t, size=11.5, weight="bold")
        words = s.split()
        rows, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 34:
                rows.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        rows.append(cur)
        f.lines(x + 10, y0 + 56, rows[:4], size=9.5, lh=12, color=MUTED)
        if i < 3:
            f.arrow(x + 208, y0 + 59, x + 222, y0 + 59)

    # ---- sizes table --------------------------------------------------------------------------
    ty = 400
    f.text(30, ty, "Two granularities, two calibrated dumps used throughout the study (Llama-3.1-8B-Instruct)", size=12.5, weight="bold")
    cols = [("", 30), ("latent shared by", 230), ("ratio", 410), ("r_k per head (mean)", 470), ("r_v per head (mean)", 620), ("B / token / layer / chip", 775)]
    rows = [
        ("M-LRD  palu_full_0.5", "one KV head", "0.5", "32-64 (33)", "64-128 (102)", "574  (1.90x smaller)"),
        ("M-LRD  palu_full_0.7", "one KV head", "0.7", "32-128 (58)", "96-128 (121)", "761  (1.43x)"),
        ("G-LRD  glrd_0.7 (gs 4)", "four KV heads (one per chip)", "0.7", "160-512 per group", "352-512 per group", "one group latent per chip"),
        ("baseline (unmodified)", "-", "-", "128", "128", "1088"),
    ]
    for j, (h, x) in enumerate(cols):
        f.text(x, ty + 20, h, size=10, weight="bold", color=MUTED)
    f.line(30, ty + 26, 910, ty + 26, color=LINE, sw=0.8)
    for i, r in enumerate(rows):
        y = ty + 44 + i * 18
        for j, (h, x) in enumerate(cols):
            f.text(x, y, r[j], size=10.5, mono=(j == 0), color=INK if i < 3 else MUTED)
    f.caption("Fig. R1. The model keeps its shapes everywhere except inside the two caches and the attention call; the price of the smaller K cache is a rebuild before every use.")
    return f


# =============================================================================================
# R2. Stage 1: the vanilla ttnn path, one decode step in one layer
# =============================================================================================
def fig_vanilla_path() -> Fig:
    f = Fig(940, 600, "Stage 1 on the n300: the ttnn ops of one decode step in one layer, and the reconstruction block that dominates it")
    f.text(14, 24, "Stage 1, the vanilla path: existing ttnn ops only, K rebuilt over the whole window every step", size=15, weight="bold")
    f.text(14, 42, "per chip (tensor-parallel 2): 16 query heads, 4 KV heads, head_dim 128, batch 32; caches [32, 4, max_seq, r] in bfp8", size=10.5, color=MUTED)

    # front of the step
    y = 70
    boxes = [
        ("ttnn.linear", "x . [W_q | VT_k^T | VT_v^T]", C_K, 150),
        ("nlp_create_qkv_heads_decode", "q [1,32,16,128]  h_k [32,4,1,r_k]  h_v [32,4,1,r_v]", C_K, 236),
        ("rotary_embedding_llama", "q only, decode mode", C_SFPU, 150),
        ("paged_update_cache x2", "h_k -> K-latent cache, h_v -> V-latent cache", C_PALU, 236),
    ]
    x = 30
    for k, (t, s, role, w) in enumerate(boxes):
        f.box(x, y, w, 44, t, s, role=role, size=10.5, mono_sub=True)
        x += w + 14
        if k < len(boxes) - 1:
            f.arrow(x - 14, y + 22, x, y + 22)
    # reconstruction block
    ry = 150
    f.rect(30, ry, 880, 150, (PLATE, C_DST[1]), r=8, sw=1.6, dash="6 4")
    f.text(44, ry + 20, "K reconstruction, every step, over the live window S = max(current_pos) + 1 (rounded up to 256)", size=12, weight="bold", color=C_DST[1])
    rboxes = [
        ("slice", "h_k[:, :, :S]  [32,4,S,r_k]", C_PALU, 150),
        ("ttnn.matmul", "h_k . U_k^T  (bf16, HiFi2)\n[32,4,S,128]  268 MB at 8k", C_FPU, 200),
        ("rotary_embedding_llama", "prefill mode over positions 0..S-1\n[32,4,S,128]  268 MB", C_SFPU, 220),
        ("ttnn.typecast", "bf16 -> bfp8 (like the baseline K cache)\nK_att  143 MB", C_K, 200),
    ]
    x = 44
    for t, s, role, w in rboxes:
        f.rect(x, ry + 36, w, 84, role, r=5)
        f.text(x + w / 2, ry + 56, t, size=11, anchor="middle", weight="bold")
        for k, line in enumerate(s.split("\n")):
            f.text(x + w / 2, ry + 74 + k * 14, line, size=9.5, anchor="middle", color=MUTED, mono=True)
        x += w + 20
        if x < 850:
            f.arrow(x - 20, ry + 78, x, ry + 78)
    f.text(44, ry + 138, "three fresh tensors per layer per step, freed as soon as the next consumes them; nothing is reused across steps (ttnn ops return new tensors)", size=9.5, color=MUTED)

    # attention + tail
    y2 = 330
    f.box(30, y2, 330, 44, "flash_multi_latent_attention_decode", "q, K_att, V = h_v cache, head_dim_v = r_v  ->  [1,32,16,r_v]", role=C_K, size=10.5, mono_sub=True)
    f.arrow(360, y2 + 22, 380, y2 + 22)
    f.box(380, y2, 180, 44, "nlp_concat_heads_decode", "pad to 128 if r_v < 128", role=C_K, size=10.5, mono_sub=True)
    f.arrow(560, y2 + 22, 580, y2 + 22)
    f.box(580, y2, 170, 44, "ttnn.linear", "wo_fused = U_v W_o (absorbed V)", role=C_FPU, size=10.5, mono_sub=True)
    f.arrow(750, y2 + 22, 770, y2 + 22)
    f.box(770, y2, 140, 44, "all-reduce", "tensor-parallel 2", role=C_NOC, size=10.5)
    f.arrow(880, y + 44, 880, ry - 2, color=INK)                      # caches feed the reconstruction block
    f.text(872, y + 70, "latent caches", size=9.5, color=MUTED, mono=True, anchor="end")
    f.arrow(330, ry + 150, 330, y2 - 2, color=INK)                    # K_att into the attention op
    f.text(338, ry + 168, "K_att", size=9.5, color=MUTED, mono=True)

    # cost panel
    cy = 400
    f.rect(30, cy, 430, 150, (PLATE, C_DST[1]), r=6)
    f.text(42, cy + 20, "what it costs (batch 32, context 4k, per decode step; stage 2d profile)", size=11.5, weight="bold")
    f.lines(42, cy + 42, [
        "baseline decode step                   55 ms",
        "Palu step, ratio 0.5, this path       265 ms   (4.8x the baseline)",
        "  of which reconstruction             207 ms   matmul 91 + RoPE 80 + typecast 33",
        "  the attention op itself               8 ms",
        "cost is linear in S and set by the 128-wide output, not by r_k",
    ], size=10.5, lh=16, mono=True)
    f.text(42, cy + 128, "V-only rows (r_k = 128) skip the block entirely and decode at baseline speed:", size=10, color=MUTED)
    f.text(42, cy + 141, "38.8 vs 38.1 ms at 1k, and they run at 8k where the baseline does not allocate.", size=10, color=MUTED)

    px = 480
    f.rect(px, cy, 430, 150, (PLATE, C_FPU[1]), r=6)
    f.text(px + 12, cy + 20, "prefill (once per request) and the fast paths", size=11.5, weight="bold")
    f.lines(px + 12, cy + 42, [
        "prefill: rebuild K and V in full (h . U^T), RoPE on q and K,",
        "  tt-metal scaled_dot_product_attention, then fill_cache(latents only)",
        "  -> compute-bound; TTFT unchanged (440-448 ms at 2k tokens)",
        "r_k = 128: baseline K path (RoPE'd K straight into a full-width cache)",
        "r_v = 128: V block folded into the QKV weight, wo unchanged",
        "cache tag carries a converter version: ttnn's weight cache keys on",
        "  name/shape/dtype only, never on content",
    ], size=10, lh=14, color=MUTED)
    f.caption("Fig. R2. Every box is an existing ttnn op; the dashed block is Palu's non-fused path and the whole of stage 2's target.")
    return f


# =============================================================================================
# R3. Stage 2: the fused op replaces the reconstruction block
# =============================================================================================
def fig_fused_path() -> Fig:
    f = Fig(940, 470, "Stage 2: what the fused op replaces and what happens inside it per chunk of 512 positions")
    f.text(14, 24, "Stage 2, the fused kernel: K is rebuilt chunk by chunk inside the attention op and never leaves L1", size=15, weight="bold")

    # stage 1 chain (compressed)
    y1 = 56
    f.text(30, y1 + 6, "stage 1", size=11, weight="bold", color=MUTED)
    chain1 = [("h_k cache", "S x r_k", C_PALU), ("matmul", "S x 128 bf16", C_FPU), ("RoPE", "S x 128", C_SFPU), ("typecast", "S x 128 bfp8", C_K), ("SDPA decode", "MLA op", C_K)]
    x = 100
    for t, s, role in chain1:
        f.box(x, y1 - 8, 130, 40, t, s, role=role, size=10.5, mono_sub=True)
        x += 150
        if x < 850:
            f.arrow(x - 20, y1 + 12, x, y1 + 12)
    f.text(100, y1 + 50, "three full-size K tensors (S x 128) in DRAM per layer per step; 207 ms of a 265 ms step at 4k", size=9.5, color=C_DST[1])

    y2 = 122
    f.text(30, y2 + 6, "stage 2", size=11, weight="bold", color=C_PALU[1])
    f.box(100, y2 - 8, 130, 40, "h_k cache", "S x r_k", role=C_PALU, size=10.5, mono_sub=True)
    f.arrow(230, y2 + 12, 250, y2 + 12)
    f.box(250, y2 - 8, 330, 40, "ttnn.transformer.palu_sdpa_decode", "q, h_k, w_rot, cos/sin table, h_v, r_v, cur_pos", role=C_PALU, size=10.5, mono_sub=True)
    f.arrow(580, y2 + 12, 600, y2 + 12)
    f.box(600, y2 - 8, 130, 40, "[1,32,16,r_v]", "same tail as stage 1", role=C_K, size=10.5, mono_sub=True)
    f.text(100, y2 + 50, "no reconstruction tensors: K exists only as one 512-row chunk in the core's L1; 65 ms per step at 4k (55 with the stage 2e kernel)", size=9.5, color=C_FPU[1])

    # inside one core, one chunk
    iy = 176
    f.rect(30, iy, 880, 250, (PLATE, C_PALU[1]), r=8, sw=1.4)
    f.text(44, iy + 20, "inside one Tensix core, per chunk of 512 positions (16 K row tiles; the op is a copy of flash decode with six borrowed CB slots)", size=11.5, weight="bold")
    # data in
    f.box(44, iy + 40, 150, 46, "latent chunk", "512 x r_k bfp8, 17 KB", role=C_PALU, size=10.5, mono_sub=True)
    f.box(44, iy + 96, 150, 46, "w_rot resident", "[u_k | u_k T]  r_k x 256", role=C_PALU, size=10.5, mono_sub=True)
    f.box(44, iy + 152, 150, 46, "angle rows 0..511", "cos | sin, read once per column", role=C_CB, size=10.5, mono_sub=True)
    f.arrow(194, iy + 63, 230, iy + 63)
    f.arrow(194, iy + 119, 230, iy + 90)
    f.arrow(194, iy + 175, 230, iy + 118)
    # compute
    f.rect(230, iy + 40, 250, 92, C_FPU, r=6)
    f.text(355, iy + 60, "rebuild one K row tile", size=11, anchor="middle", weight="bold")
    f.lines(240, iy + 78, [
        "AB = lat_i . [u_k | u_k T]      (matmul)",
        "K_rot = A . cos + B . sin        (rotation)",
        "packed into c_1 in flash decode's K order",
    ], size=9.8, lh=13, mono=True)
    f.arrow(480, iy + 86, 516, iy + 86)
    f.rect(516, iy + 40, 200, 92, C_K, r=6)
    f.text(616, iy + 60, "flash-decode step, unchanged", size=11, anchor="middle", weight="bold")
    f.lines(526, iy + 78, [
        "scores = q~ . K_rot^T",
        "online softmax (m, l, O)",
        "O += P . V   (V = h_v latent)",
    ], size=9.8, lh=13, mono=True)
    f.rect(740, iy + 40, 160, 92, C_IDLE, r=6)
    f.text(820, iy + 62, "next chunk", size=11, anchor="middle", weight="bold")
    f.lines(748, iy + 80, ["16 chunks at 8k, 2 at 1k;", "the root core owns the last", "chunks (the causal mask)"], size=9.5, lh=13, color=MUTED)
    # phase 2 note
    f.rect(230, iy + 144, 486, 66, C_SFPU, r=6)
    f.text(240, iy + 162, "Phase 2 (the shipped variant): relative rotation", size=10.5, weight="bold")
    f.lines(240, iy + 178, ["q R_p . (k R_{s0+t})^T = (q R_{p-s0}) . (k R_t)^T",
                            "K only ever needs angles 0..511 (resident); the query state q~ is advanced",
                            "by R_{-512} once per chunk (8 fp32 tiles) instead of streaming a 256 KB table"], size=9.5, lh=13, mono=True)
    f.lines(740, iy + 150, ["DRAM bytes per chunk:", "baseline 136 KB (K + V)", "fused 17 KB latent", "  + 68 KB V (r_v 128)"], size=9.8, lh=13, mono=True, color=MUTED)
    f.text(44, iy + 226, "stage 2e made the rebuild cheaper (matmul straight into DST, dest-reuse multiplies; the angle table multicast down each grid column) and it is the default kernel today.", size=9.8, color=MUTED)
    f.caption("Fig. R3. Same caches, same front and tail of the layer as stage 1; only the dashed block of Fig. R2 changes, into a per-chunk rebuild inside the op.")
    return f


# =============================================================================================
# R4. What it bought: decode step and cache size
# =============================================================================================
def fig_results() -> Fig:
    f = Fig(940, 400, "Decode step time per context and KV cache bytes: the baseline, the vanilla Palu path, the fused kernel, and the shipped kernel")
    f.text(14, 24, "What the two paths cost: whole decode step at batch 32 (min ms), ratio-0.5 dump", size=15, weight="bold")
    # bars: groups per context
    ctx = ["1k", "4k", "8k"]
    series = [
        ("baseline: unmodified model (bfp8 KV)", [38.0, 55.4, None], C_K),
        ("Palu, vanilla path (stage 1)", [88.5, 265.3, 488.5], C_DST),
        ("Palu, fused kernel (stage 2)", [42.8, 65.0, 94.5], C_PALU),
        ("Palu, shipped kernel (stage 2e)", [39.0, 55.5, 77.4], C_FPU),
    ]
    bx, by, bw, bh = 70, 60, 34, 220
    ymax = 500.0
    for i in range(0, 6):
        v = i * 100
        yy = by + bh - v / ymax * bh
        f.line(bx - 6, yy, bx + 3 * 180, yy, color="#e4e7ec", sw=1)
        f.text(bx - 10, yy + 4, f"{v}", size=9.5, anchor="end", color=MUTED)
    f.text(bx - 8, by - 12, "ms per decode step", size=9.5, color=MUTED)
    for gi, c in enumerate(ctx):
        gx = bx + gi * 180
        for si, (name, vals, role) in enumerate(series):
            v = vals[gi]
            x = gx + si * (bw + 4)
            if v is None:
                f.rect(x, by + bh - 14, bw, 14, C_IDLE, r=2)
                f.text(x + bw / 2, by + bh - 20, "OOM", size=9, anchor="middle", color=MUTED)
                continue
            h = v / ymax * bh
            f.rect(x, by + bh - h, bw, h, role, r=2)
            f.text(x + bw / 2, by + bh - h - 5, f"{v:.0f}", size=9.5, anchor="middle")
        f.text(gx + 2 * (bw + 4), by + bh + 16, f"context {c}", size=10.5, anchor="middle", color=MUTED)
    f.legend(bx, by + bh + 40, [(role, name) for name, _, role in series], step=15)

    # right: bytes and context
    rx = 640
    f.text(rx, 62, "KV bytes per token per layer per chip; max context", size=11.5, weight="bold")
    rows = [
        ("baseline", 1088, "4096", C_K),
        ("Palu 0.7 (bfp8 latents)", 761, "8192", C_PALU),
        ("Palu 0.5 (bfp8 latents)", 574, "8192", C_PALU),
        ("Palu 0.7, bfp4 latents (stage 2f)", 403, "19456", C_FPU),
    ]
    scale = 0.135
    for i, (n, b, mc, role) in enumerate(rows):
        y = 84 + i * 46
        f.text(rx, y + 2, n, size=10, color=MUTED)
        f.rect(rx, y + 8, b * scale, 16, role, r=2)
        f.text(rx + b * scale + 6, y + 20, f"{b} B", size=9.5, mono=True)
        f.text(rx + 270, y + 2, f"max ctx {mc}", size=10, mono=True, anchor="end")
    f.lines(rx, 276, [
        "The vanilla path pays for the smaller cache with a",
        "4.8x slower step at 4k; the fused kernel closes 95 %",
        "of that gap (0.85x the baseline); the stage 2e kernel reaches",
        "parity at 4k and runs at 8k where the baseline cannot allocate.",
    ], size=10, lh=14, color=MUTED)
    f.caption("Fig. R4. Same dump, same quality (ppl 12.93 at ratio 0.5); only the way K is rebuilt changes between the three Palu rows.")
    return f


FIGURES = {
    "report_r1_palu_architecture": fig_palu_architecture,
    "report_r2_vanilla_path": fig_vanilla_path,
    "report_r3_fused_path": fig_fused_path,
    "report_r4_results": fig_results,
}


def write_all(out: Path = OUT, png_dir: Path | None = None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in FIGURES.items():
        fig = fn()
        p = out / f"{name}.svg"
        p.write_text(fig.render(), encoding="utf-8")
        written.append(p)
        if png_dir is not None:
            import cairosvg  # optional, for a visual check

            png_dir.mkdir(parents=True, exist_ok=True)
            cairosvg.svg2png(url=str(p), write_to=str(png_dir / f"{name}.png"), output_width=fig.w * 2)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--png", type=Path, default=None, help="also rasterise into this directory (cairosvg)")
    args = ap.parse_args(argv)
    for p in write_all(args.out, args.png):
        print(p)


if __name__ == "__main__":
    main()
