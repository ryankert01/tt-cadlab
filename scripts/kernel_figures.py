#!/usr/bin/env python3
"""Draw the figures for docs/kernel_guide.md (how flash decode is written on a Tensix core, and how
the Palu fused kernel changes it) as standalone SVG files under docs/figures/.

Every figure is built from a handful of primitives so the geometry is computed, not hand-typed.
Colours are fixed (light plate) on purpose: the same files are embedded on a white plate in the
results page and rendered by GitHub, neither of which can restyle an SVG.

    python scripts/kernel_figures.py            # writes docs/figures/kernel_*.svg
    python scripts/kernel_figures.py --png DIR  # also rasterises each figure into DIR (needs cairosvg)
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"

INK = "#1f2430"
MUTED = "#5f6673"
LINE = "#8e94a0"
PLATE = "#ffffff"

# role palette: (fill, stroke)
C_L1 = ("#eef2f7", "#6b7a99")       # memory / L1
C_CB = ("#f7f2e4", "#b08a3e")       # circular buffers
C_K = ("#dbe7fb", "#2f6fdb")        # stock data (K, V, Q)
C_PALU = ("#fde7d6", "#e07b39")     # Palu data (latents, weights, tables)
C_FPU = ("#dff3e6", "#3c9d6b")      # FPU / matmul
C_SFPU = ("#ece4fb", "#8b5cf6")     # SFPU / vector
C_DST = ("#fbe1e1", "#d64545")      # DST register / sync
C_NOC = ("#e8e8ee", "#6b6f8a")      # NoC / RISC-V
C_IDLE = ("#f4f4f4", "#c2c6cc")     # unused


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


class Fig:
    """Minimal SVG accumulator."""

    def __init__(self, w: int, h: int, title: str):
        self.w, self.h = w, h
        self.parts: list[str] = []
        self.title = title

    # ---- primitives -------------------------------------------------------------------------
    def rect(self, x, y, w, h, role=C_L1, r=4, sw=1.2, dash=None, opacity=1.0):
        fill, stroke = role
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{extra}/>'
        )

    def text(self, x, y, s, size=12, anchor="start", weight="normal", color=INK, mono=False, italic=False):
        fam = MONO if mono else FONT
        style = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{color}"{style}>{esc(s)}</text>'
        )

    def lines(self, x, y, rows, size=11.5, lh=15, anchor="start", color=INK, mono=False):
        for i, s in enumerate(rows):
            self.text(x, y + i * lh, s, size=size, anchor=anchor, color=color, mono=mono)

    def line(self, x1, y1, x2, y2, color=LINE, sw=1.2, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{extra}/>'
        )

    def arrow(self, x1, y1, x2, y2, color=INK, sw=1.4, label=None, lsize=11, dash=None, loff=(0, -5)):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
            f'marker-end="url(#ah)"{extra}/>'
        )
        if label:
            mx, my = (x1 + x2) / 2 + loff[0], (y1 + y2) / 2 + loff[1]
            self.text(mx, my, label, size=lsize, anchor="middle", color=MUTED)

    def path_arrow(self, pts, color=INK, sw=1.4, dash=None):
        d = "M " + " L ".join(f"{x},{y}" for x, y in pts)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#ah)"{extra}/>'
        )

    def box(self, x, y, w, h, title, sub=None, role=C_L1, size=12, r=5, mono_sub=False, sw=1.2, tsize=None):
        self.rect(x, y, w, h, role, r=r, sw=sw)
        cx = x + w / 2
        if sub:
            self.text(cx, y + h / 2 - 3, title, size=tsize or size, anchor="middle", weight="bold")
            self.text(cx, y + h / 2 + 12, sub, size=size - 1.5, anchor="middle", color=MUTED, mono=mono_sub)
        else:
            self.text(cx, y + h / 2 + 4, title, size=tsize or size, anchor="middle", weight="bold")

    def tiles(self, x, y, cols, rows, s, role=C_K, gap=2, labels=None, size=9, stroke_w=0.8):
        """A grid of tiles. labels: function (r, c) -> str or None."""
        fill, stroke = role
        for rr in range(rows):
            for cc in range(cols):
                tx, ty = x + cc * (s + gap), y + rr * (s + gap)
                self.parts.append(
                    f'<rect x="{tx}" y="{ty}" width="{s}" height="{s}" fill="{fill}" stroke="{stroke}" '
                    f'stroke-width="{stroke_w}"/>'
                )
                if labels:
                    lab = labels(rr, cc)
                    if lab:
                        self.text(tx + s / 2, ty + s / 2 + size * 0.36, lab, size=size, anchor="middle", mono=True)
        return x + cols * (s + gap) - gap, y + rows * (s + gap) - gap

    def brace_label(self, x1, x2, y, label, size=10.5, above=True):
        d = -1 if above else 1
        self.line(x1, y, x2, y, color=MUTED, sw=1)
        self.line(x1, y, x1, y + 4 * d * -1, color=MUTED, sw=1)
        self.line(x2, y, x2, y + 4 * d * -1, color=MUTED, sw=1)
        self.text((x1 + x2) / 2, y + (-6 if above else 13), label, size=size, anchor="middle", color=MUTED)

    def vbrace_label(self, x, y1, y2, label, size=10.5):
        self.line(x, y1, x, y2, color=MUTED, sw=1)
        self.line(x, y1, x + 4, y1, color=MUTED, sw=1)
        self.line(x, y2, x + 4, y2, color=MUTED, sw=1)
        self.parts.append(
            f'<text x="{x - 6}" y="{(y1 + y2) / 2}" font-family="{FONT}" font-size="{size}" text-anchor="middle" '
            f'fill="{MUTED}" transform="rotate(-90 {x - 6} {(y1 + y2) / 2})">{esc(label)}</text>'
        )

    def caption(self, s, size=11):
        self.text(14, self.h - 10, s, size=size, color=MUTED, italic=True)

    def legend(self, x, y, items, size=10.5, step=16):
        for i, (role, label) in enumerate(items):
            self.rect(x, y + i * step - 9, 14, 11, role, r=2)
            self.text(x + 20, y + i * step, label, size=size, color=MUTED)

    # ---- output ---------------------------------------------------------------------------
    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(self.title)}">\n'
            f"<title>{esc(self.title)}</title>\n"
            "<defs><marker id=\"ah\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" "
            "orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"" + INK + "\"/></marker></defs>\n"
            f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="{PLATE}"/>\n'
        )
        return head + "\n".join(self.parts) + "\n</svg>\n"


# Arrow heads are solid ink (not `context-stroke`, which older renderers and cairosvg do not honour),
# so every arrow is drawn in ink.


# =============================================================================================
# A. Tensix basics
# =============================================================================================
def fig_tensix_core() -> Fig:
    f = Fig(940, 560, "One Tensix core: five RISC-V dispatchers, L1, the compute engine and two NoCs")
    f.text(14, 24, "A Wormhole chip and one of its Tensix cores", size=15, weight="bold")

    # chip mini-grid (left)
    gx, gy, s, gap = 20, 50, 14, 3
    f.text(gx, gy - 6, "one chip: 8 x 8 Tensix cores used by this op", size=10.5, color=MUTED)
    for r in range(8):
        for c in range(9):
            role = C_L1 if c == 0 else C_NOC
            lab = "D" if c == 0 else "T"
            tx, ty = gx + c * (s + gap), gy + r * (s + gap)
            f.parts.append(
                f'<rect x="{tx}" y="{ty}" width="{s}" height="{s}" fill="{role[0]}" stroke="{role[1]}" stroke-width="0.6"/>'
            )
            f.text(tx + s / 2, ty + s / 2 + 3, lab, size=7, anchor="middle", color=MUTED)
    f.lines(gx, gy + 8 * (s + gap) + 14, [
        "D = DRAM controller tile",
        "    (6 per chip, 12 GB GDDR6)",
        "T = Tensix core; Ethernet tiles",
        "    link the two chips of an n300",
        "two NoCs ring the mesh in",
        "    opposite directions",
        "every core's L1 has a chip-wide",
        "    address (x, y, offset)",
    ], size=10, lh=13, color=MUTED)

    # the core (right)
    cx, cy, cw, ch = 226, 44, 700, 480
    f.rect(cx, cy, cw, ch, (PLATE, INK), r=8, sw=1.4)
    f.text(cx + 12, cy + 20, "Tensix core", size=13, weight="bold")
    f.text(cx + 110, cy + 20, "five \"baby\" RISC-V cores issue commands; the engines below do the work", size=10.5, color=MUTED)

    # five RISC-V boxes
    bx, by, bw, bh, bg = cx + 16, cy + 34, 128, 46, 8
    names = [
        ("BRISC", "NoC 0: reader kernel", C_NOC),
        ("NCRISC", "NoC 1: writer kernel", C_NOC),
        ("TRISC 0", "unpack", C_FPU),
        ("TRISC 1", "math", C_FPU),
        ("TRISC 2", "pack", C_FPU),
    ]
    for i, (n, sub, role) in enumerate(names):
        f.box(bx + i * (bw + bg), by, bw, bh, n, sub, role=role, size=11.5)
    f.brace_label(bx + 2 * (bw + bg), bx + 5 * (bw + bg) - bg, by + bh + 12, "compute kernel: one C++ source, compiled three times", above=False)

    # NoC ports
    nx, ny = cx + 16, cy + 140
    f.box(nx, ny, 70, 40, "NoC 0", role=C_NOC, size=11)
    f.box(nx, ny + 56, 70, 40, "NoC 1", role=C_NOC, size=11)
    f.arrow(nx + 70, ny + 76, nx + 110, ny + 76)
    f.text(nx, ny + 114, "to / from DRAM", size=9.5, color=MUTED)
    f.text(nx, ny + 126, "and other cores", size=9.5, color=MUTED)

    # L1
    lx, ly, lw, lh = cx + 110, cy + 120, 574, 96
    f.rect(lx, ly, lw, lh, C_L1, r=6)
    f.text(lx + 10, ly + 18, "L1 SRAM, 1.5 MB (about 1.41 MB usable by this op's buffers)", size=12, weight="bold")
    f.text(lx + 10, ly + 34, "no cache: bytes get here only by NoC transfer or by the packer", size=10, color=MUTED)
    f.text(lx + lw - 10, ly + 34, "32 circular-buffer slots c_0 .. c_31", size=10, anchor="end", color=MUTED)
    # CB slots strip
    sx, sy, ss = lx + 10, ly + 48, 16
    for i in range(32):
        f.parts.append(
            f'<rect x="{sx + i * (ss + 1.5)}" y="{sy}" width="{ss}" height="{22}" fill="{C_CB[0]}" stroke="{C_CB[1]}" stroke-width="0.8"/>'
        )
        if i in (0, 15, 16, 31):
            f.text(sx + i * (ss + 1.5) + ss / 2, sy + 15, f"{i}", size=7.5, anchor="middle", color=MUTED, mono=True)
    f.arrow(nx + 70, ny + 20, lx, ny + 20)

    # compute engine
    ex, ey, ew, eh = cx + 110, cy + 240, 574, 220
    f.rect(ex, ey, ew, eh, (PLATE, LINE), r=6, dash="4 3")
    f.text(ex + 130, ey + 18, "compute engine (the three TRISCs drive it)", size=12, weight="bold")

    f.box(ex + 14, ey + 44, 110, 44, "Unpacker", "L1 -> registers", role=C_FPU, size=11)
    f.box(ex + 148, ey + 34, 110, 30, "SrcA", "one tile", role=C_FPU, size=11)
    f.box(ex + 148, ey + 70, 110, 30, "SrcB", "one tile", role=C_FPU, size=11)
    f.box(ex + 284, ey + 40, 116, 56, "FPU", "matmul, reduce, mul, add", role=C_FPU, size=11)
    f.box(ex + 428, ey + 34, 126, 68, "DST", "16 x 16-bit / 8 x fp32 tiles", role=C_DST, size=11)
    f.box(ex + 284, ey + 130, 116, 56, "SFPU", "vector unit: exp, recip, adds", role=C_SFPU, size=11)
    f.box(ex + 14, ey + 130, 110, 44, "Packer", "DST -> L1", role=C_FPU, size=11)

    f.arrow(lx + 60, ly + lh, ex + 60, ey + 44)
    f.text(lx + 66, ly + lh + 16, "wait_front", size=10, color=MUTED)
    f.arrow(ex + 124, ey + 60, ex + 148, ey + 49)
    f.arrow(ex + 124, ey + 74, ex + 148, ey + 85)
    f.arrow(ex + 258, ey + 49, ex + 284, ey + 60)
    f.arrow(ex + 258, ey + 85, ex + 284, ey + 76)
    f.arrow(ex + 400, ey + 68, ex + 428, ey + 68)
    f.path_arrow([(ex + 490, ey + 102), (ex + 490, ey + 158), (ex + 400, ey + 158)])
    f.path_arrow([(ex + 342, ey + 130), (ex + 342, ey + 110), (ex + 460, ey + 110), (ex + 460, ey + 102)])
    f.text(ex + 430, ey + 175, "SFPU reads and writes DST in place", size=9.5, color=MUTED)
    f.path_arrow([(ex + 428, ey + 90), (ex + 270, ey + 152), (ex + 124, ey + 152)])
    f.text(ex + 150, ey + 200, "pack_tile: DST -> a CB in L1 (push_back)", size=9.5, color=MUTED)
    f.path_arrow([(ex + 60, ey + 174), (ex + 60, ey + 205), (lx + 20, ey + 205), (lx + 20, ly + lh + 2)], dash="3 3")
    f.caption("Fig. A1. The reader and writer kernels move bytes between DRAM, other cores and L1 over the NoCs; the compute kernel moves tiles L1 -> Src -> FPU/SFPU -> DST -> L1.")
    return f


def fig_tile_formats() -> Fig:
    f = Fig(940, 440, "The 32 x 32 tile, its four faces, the data formats, and tile order inside a matrix")
    f.text(14, 24, "Tiles: the unit of everything on device", size=15, weight="bold")

    # a tile with 4 faces
    tx, ty, fs = 24, 50, 60
    for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        x, y = tx + c * (fs + 3), ty + r * (fs + 3)
        f.rect(x, y, fs, fs, C_K, r=2)
        f.text(x + fs / 2, y + fs / 2 + 4, f"face {i}", size=11, anchor="middle", mono=True)
    # half tile
    hx, hy = tx + 2 * fs + 40, ty
    f.rect(hx, hy, fs, fs, C_K, r=2)
    f.rect(hx + fs + 3, hy, fs, fs, C_K, r=2)
    f.rect(hx, hy + fs + 3, fs, fs, C_IDLE, r=2, dash="3 3")
    f.rect(hx + fs + 3, hy + fs + 3, fs, fs, C_IDLE, r=2, dash="3 3")
    f.text(hx + fs + 1, hy + fs / 2 + 4, "16 x 32 half tile", size=10.5, anchor="middle")
    f.lines(tx, ty + 2 * fs + 24, [
        "full tile: 32 x 32 elements = 4 faces of 16 x 16,",
        "stored face by face (f0, f1, f2, f3), each row-major;",
        "rows 0-15 live in faces 0 and 1.",
        "",
        "half tile: 16 x 32 = faces 0 and 1 only. Stock decode",
        "uses it for Q, the scores, the mask and the statistics",
        "when there are <= 16 query heads; the Palu op forces",
        "full tiles everywhere (one of the untried levers).",
    ], size=10.5, lh=14, color=MUTED)

    # formats table
    fx, fy = 330, 52
    rows = [
        ("format", "bytes / element", "bytes / tile", "used here for"),
        ("bfloat16 (bf16)", "2", "2048", "Q, K_rot, weights, cos/sin, intermediates"),
        ("bfp8_b (block fp8)", "1 + 8-bit exponent per 16", "1088", "stock K and V caches, Palu latent caches"),
        ("float32 (fp32)", "4", "4096", "Phase-2 query state and rotation step"),
        ("bfp4_b", "0.5 + exponent per 16", "576", "(supported, not used)"),
    ]
    colx = [fx, fx + 128, fx + 300, fx + 400]
    for i, row in enumerate(rows):
        y = fy + i * 22
        if i == 0:
            f.line(fx - 4, y + 6, fx + 600, y + 6, color=LINE)
        for j, cell in enumerate(row):
            f.text(colx[j], y, cell, size=10.5, weight="bold" if i == 0 else "normal", color=INK if i else MUTED,
                   mono=(i > 0 and j == 2))
    f.text(fx, fy + 5 * 22 + 6, "bfp8: 16 elements share one exponent, so a tile is 1024 mantissa bytes + 64 exponent bytes.", size=10, color=MUTED)
    f.text(fx, fy + 5 * 22 + 20, "The FPU multiplies at 5 x 7 mantissa bits per pass; HiFi2 runs two passes, HiFi4 four (this study: HiFi2).", size=10, color=MUTED)

    # tile order in a matrix
    mx, my, s = 330, 230, 22
    f.text(mx, my - 8, "A K chunk of 512 positions x 128 dims is 16 row tiles x 4 column tiles", size=11, weight="bold")
    f.tiles(mx, my, 4, 6, s, role=C_K, labels=lambda r, c: f"{r * 4 + c}" if r < 5 else ("..." if c == 0 else ""), size=8)
    f.text(mx + 4 * (s + 2) + 8, my + 14, "row-major tile order: index = i * 4 + j", size=10.5, color=MUTED)
    f.text(mx + 4 * (s + 2) + 8, my + 28, "(how the tensor sits in DRAM)", size=10.5, color=MUTED)
    mx2 = mx + 300
    f.tiles(mx2, my, 4, 6, s, role=C_K, labels=lambda r, c: f"{c * 16 + r}" if r < 5 else ("..." if c == 0 else ""), size=8)
    f.text(mx2 + 4 * (s + 2) + 8, my + 14, "column-major tile order: index = j * 16 + i", size=10.5, color=MUTED)
    f.text(mx2 + 4 * (s + 2) + 8, my + 28, "(the K chunk's layout in L1 for Q K^T, and", size=10.5, color=MUTED)
    f.text(mx2 + 4 * (s + 2) + 8, my + 42, "the order the Palu kernel packs K_rot in)", size=10.5, color=MUTED)
    f.text(mx, my + 6 * (s + 2) + 16, "row tile i = positions 32i .. 32i+31;  column tile j = head dims 32j .. 32j+31", size=10.5, color=MUTED)
    f.caption("Fig. A2. Tiles are 32 x 32; a tensor is padded to whole tiles and addressed by tile index, so every shape in the kernels is written in tiles (St, DHt, Sk_chunk_t, ...).")
    return f


def fig_circular_buffers() -> Fig:
    f = Fig(940, 440, "Circular buffers: the producer/consumer queues in L1 that connect the three kernels")
    f.text(14, 24, "Circular buffers (CBs): how the three kernels hand tiles to each other", size=15, weight="bold")

    # ring
    rx, ry, n, pw, ph = 60, 70, 8, 52, 40
    f.text(rx, ry - 12, "one CB = a fixed L1 region of N pages (here 8 tiles), used as a FIFO", size=11, color=MUTED)
    states = ["consumed", "consumed", "ready", "ready", "ready", "free", "free", "free"]
    for i, st in enumerate(states):
        role = C_CB if st == "ready" else (C_IDLE if st == "free" else (C_L1))
        f.rect(rx + i * (pw + 2), ry, pw, ph, role, r=2)
        f.text(rx + i * (pw + 2) + pw / 2, ry + ph / 2 + 4, f"tile {i}", size=9.5, anchor="middle", mono=True)
    f.arrow(rx + 2 * (pw + 2) + pw / 2, ry + ph + 34, rx + 2 * (pw + 2) + pw / 2, ry + ph + 4)
    f.text(rx + 2 * (pw + 2) + pw / 2, ry + ph + 48, "read pointer (front)", size=10, anchor="middle", color=MUTED)
    f.arrow(rx + 5 * (pw + 2) + pw / 2, ry - 30, rx + 5 * (pw + 2) + pw / 2, ry - 2)
    f.text(rx + 5 * (pw + 2) + pw / 2, ry - 36, "write pointer (back)", size=10, anchor="middle", color=MUTED)
    f.brace_label(rx + 2 * (pw + 2), rx + 5 * (pw + 2) - 2, ry + ph + 62, "3 tiles produced and not yet consumed", above=False)
    f.text(rx + 8 * (pw + 2) + 8, ry + 16, "pointers wrap", size=10, color=MUTED)
    f.text(rx + 8 * (pw + 2) + 8, ry + 30, "around the end", size=10, color=MUTED)

    # producer / consumer API
    px, py = 60, 200
    f.rect(px, py, 380, 88, (PLATE, C_CB[1]), r=6)
    f.text(px + 10, py + 18, "producer side (reader kernel, or the packer inside compute)", size=11, weight="bold")
    f.lines(px + 10, py + 38, [
        "cb_reserve_back(cb, n)   block until n pages are free",
        "write n pages at get_write_ptr()   (NoC read into L1, or pack_tile)",
        "cb_push_back(cb, n)      publish them: the consumer may now see them",
    ], size=10.5, lh=15, mono=True)
    qx = px + 420
    f.rect(qx, py, 400, 88, (PLATE, C_CB[1]), r=6)
    f.text(qx + 10, py + 18, "consumer side (compute's unpacker, or the writer kernel)", size=11, weight="bold")
    f.lines(qx + 10, py + 38, [
        "cb_wait_front(cb, n)     block until n pages are ready",
        "read them: tile index i = i pages past the front (no wrap!)",
        "cb_pop_front(cb, n)      free them for the producer",
    ], size=10.5, lh=15, mono=True)

    # chain of kernels
    ky = 320
    f.box(60, ky, 140, 40, "reader", "BRISC, NoC 0", role=C_NOC, size=11)
    f.box(380, ky, 160, 40, "compute", "unpack / math / pack", role=C_FPU, size=11)
    f.box(740, ky, 140, 40, "writer", "NCRISC, NoC 1", role=C_NOC, size=11)
    f.arrow(200, ky + 20, 380, ky + 20, label="inputs, c_0 .. c_15 by convention", loff=(0, -7))
    f.arrow(540, ky + 20, 740, ky + 20, label="outputs, c_16 .. c_20", loff=(0, -7))
    f.path_arrow([(470, ky + 40), (470, ky + 62), (455, ky + 62)])
    f.text(480, ky + 66, "intermediates c_21 .. c_31: compute produces and consumes them itself", size=10, color=MUTED)
    f.path_arrow([(810, ky), (810, ky - 18), (600, ky - 18), (600, ky + 4)], dash="3 3")
    f.text(690, ky - 24, "the writer also produces inputs (mask, scalers)", size=9.5, anchor="middle", color=MUTED)
    f.caption("Fig. A3. A CB is a fixed L1 region with a read and a write pointer; every kernel blocks on the other side's progress, which is the only synchronisation inside a core.")
    return f


def fig_dst_pipeline() -> Fig:
    f = Fig(940, 560, "The DST register and one compute round trip: acquire, math, commit, wait, pack, release")
    f.text(14, 24, "DST and the compute round trip", size=15, weight="bold")

    # timeline lanes
    lx, ly, lw = 110, 60, 800
    lanes = [("TRISC 0 unpack", C_FPU), ("TRISC 1 math", C_FPU), ("TRISC 2 pack", C_FPU)]
    lh = 44
    for i, (n, role) in enumerate(lanes):
        y = ly + i * (lh + 10)
        f.text(lx - 8, y + lh / 2 + 4, n, size=11, anchor="end", weight="bold")
        f.line(lx, y + lh / 2, lx + lw, y + lh / 2, color=LINE, dash="2 3")

    def seg(lane, x0, x1, label, role, size=10):
        y = ly + lane * (lh + 10) + 6
        f.rect(x0, y, x1 - x0, lh - 12, role, r=3)
        f.text((x0 + x1) / 2, y + (lh - 12) / 2 + 4, label, size=size, anchor="middle")

    # round trip 1 (uses DST half A)
    seg(0, lx + 10, lx + 150, "unpack tiles -> SrcA / SrcB", C_FPU)
    seg(1, lx + 30, lx + 90, "acquire", C_DST)
    seg(1, lx + 92, lx + 250, "FPU / SFPU ops into DST half A", C_FPU)
    seg(1, lx + 252, lx + 300, "commit", C_DST)
    seg(2, lx + 252, lx + 330, "wait", C_DST)
    seg(2, lx + 332, lx + 470, "pack_tile x k  (DST -> L1)", C_FPU)
    seg(2, lx + 472, lx + 520, "release", C_DST)
    # round trip 2 (half B) overlaps
    seg(0, lx + 260, lx + 400, "unpack next tiles", C_FPU)
    seg(1, lx + 320, lx + 380, "acquire", C_DST)
    seg(1, lx + 382, lx + 540, "ops into DST half B", C_FPU)
    seg(1, lx + 542, lx + 590, "commit", C_DST)
    seg(2, lx + 542, lx + 620, "wait", C_DST)
    seg(2, lx + 622, lx + 760, "pack_tile x k", C_FPU)
    seg(2, lx + 762, lx + 800, "rel.", C_DST)
    f.brace_label(lx + 30, lx + 520, ly + 3 * (lh + 10) + 6, "one DST round trip: tile_regs_acquire .. tile_regs_release (a full three-thread handshake)", above=False)
    f.text(lx + 10, ly + 3 * (lh + 10) + 42, "Half-sync: math fills one half of DST while pack drains the other, so consecutive round trips overlap.", size=10.5, color=MUTED)
    f.text(lx + 10, ly + 3 * (lh + 10) + 56, "But the work inside one acquisition is bounded by the half's capacity, and every acquisition costs the handshake.", size=10.5, color=MUTED)

    # DST capacity panel
    dx, dy = 40, 290
    f.rect(dx, dy, 400, 120, (PLATE, C_DST[1]), r=6)
    f.text(dx + 10, dy + 18, "DST capacity per acquisition (dest_helpers.hpp)", size=11, weight="bold")
    rows = [("", "16-bit dest", "fp32 dest"), ("full sync", "16 tiles", "8 tiles"), ("half sync (default)", "8 tiles", "4 tiles")]
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            f.text(dx + 12 + j * 120, dy + 42 + i * 18, c, size=10.5, weight="bold" if (i == 0 or j == 0) else "normal",
                   color=INK if (i and j) else MUTED)
    f.rect(dx + 246, dy + 66, 64, 18, C_DST, r=3, opacity=0.6)
    f.text(dx + 12, dy + 112, "This study runs fp32 dest accumulate, half sync: dst_size = 4 tiles in both kernels.", size=10, color=MUTED)

    # FPU vs SFPU
    ex, ey = 470, 290
    f.rect(ex, ey, 440, 120, (PLATE, LINE), r=6)
    f.text(ex + 10, ey + 18, "Two engines write DST", size=11, weight="bold")
    f.rect(ex + 10, ey + 28, 14, 11, C_FPU, r=2)
    f.lines(ex + 30, ey + 38, [
        "FPU (matrix): SrcA x SrcB -> DST. 8x16 by 16x16 per cycle; matmul_block,",
        "mul_tiles, add_tiles, reduce. Operands come straight from CBs via the unpacker.",
    ], size=10, lh=13, color=MUTED)
    f.rect(ex + 10, ey + 66, 14, 11, C_SFPU, r=2)
    f.lines(ex + 30, ey + 76, [
        "SFPU (vector): 32 lanes, fp32, reads and writes DST in place; 32 passes per",
        "full tile. exp_tile, recip_tile, add_binary_tile. Input must already be in DST.",
    ], size=10, lh=13, color=MUTED)
    f.text(ex + 10, ey + 108, "Both are issued by TRISC 1; results leave DST only through the packer.", size=10, color=MUTED)

    # data format reconfig note
    ny = 430
    f.rect(40, ny, 870, 96, (PLATE, LINE), r=6, dash="4 3")
    f.text(52, ny + 18, "Three more things every op in the kernels does, and that show up as cost", size=11, weight="bold")
    f.lines(52, ny + 38, [
        "reconfig_data_format(cbA, cbB) / pack_reconfig_data_format(cb): tell the unpacker and packer the tile format of the CBs about to be touched (bf16, bfp8, fp32).",
        "<op>_init(...): program the engine for the next kind of op (matmul_block_init, mul_init, add_binary_tile_init, exp_tile_init). Switching op types costs an init.",
        "fp32 dest accumulate: DST holds fp32 so a matmul's inner sum does not round to bf16; the price is half the DST tiles per acquisition (8 -> 4).",
    ], size=10, lh=15, color=MUTED)
    f.caption("Fig. A4. A compute kernel is a sequence of DST acquisitions; each one does at most dst_size tiles of output and costs a three-thread handshake. Fewer, fuller acquisitions win.")
    return f


# =============================================================================================
# B. Flash decode
# =============================================================================================
def fig_decode_math() -> Fig:
    f = Fig(940, 470, "Decode attention for one (batch row, KV head) work item, processed in K chunks with an online softmax")
    f.text(14, 24, "What one work item computes: decode attention over K/V chunks", size=15, weight="bold")

    s = 16
    # Q
    qx, qy = 40, 80
    f.text(qx, qy - 26, "Q: 1 x 4 tiles", size=11, weight="bold")
    f.text(qx, qy - 12, "16 query heads (rows 0-15) + 16 pad rows", size=9.5, color=MUTED)
    f.tiles(qx, qy, 4, 1, s, role=C_K)
    f.text(qx + 4 * (s + 2) + 6, qy + 12, "x 128 dims", size=9.5, color=MUTED)
    # K^T chunk
    kx, ky = 40, 150
    f.text(kx, ky - 12, "K chunk k, transposed: 4 x 16 tiles (128 dims x 512 positions)", size=11, weight="bold")
    f.tiles(kx, ky, 16, 4, s, role=C_K)
    # scores
    sx, sy = 40, 250
    f.text(sx, sy - 12, "S = Q K^T: 1 x 16 tiles (32 rows x 512 positions), bf16 in c_24", size=11, weight="bold")
    f.tiles(sx, sy, 16, 1, s, role=C_SFPU)
    # V chunk
    vx, vy = 40, 300
    f.text(vx, vy - 12, "V chunk k: 16 x 4 tiles (512 positions x 128 dims)", size=11, weight="bold")
    f.tiles(vx, vy, 4, 6, s, role=C_K, labels=lambda r, c: "..." if (r == 5 and c == 0) else None, size=8)
    f.text(vx + 4 * (s + 2) + 8, vy + 40, "P V: 1 x 4 tiles into c_25", size=10.5, color=MUTED)
    f.text(vx + 4 * (s + 2) + 8, vy + 54, "accumulated into O in c_26", size=10.5, color=MUTED)

    # formulas
    fx, fy = 470, 70
    f.rect(fx - 12, fy - 24, 460, 380, (PLATE, LINE), r=6)
    f.text(fx, fy, "Online softmax, per chunk (flash decode)", size=12, weight="bold")
    f.lines(fx, fy + 24, [
        "state per item:  m (running row max), l (running row sum), O (1 x 4 tiles)",
        "",
        "S      = Q K_k^T                                  (QK matmul, transpose=true)",
        "S     += mask         only on the last chunk, positions > cur_pos -> -inf",
        "m_new  = max(m, rowmax(S))                        (reduce_c MAX)",
        "P      = exp((S - m_new) * scale)                 (sub_exp, scale folded in)",
        "r      = rowsum(P)                                (reduce_c SUM over the 16 tiles)",
        "a      = exp((m - m_new) * scale)                 (the correction factor)",
        "l      = a * l + r",
        "O      = a * O + P V_k                            (PV matmul, then rescale + add)",
        "m      = m_new",
        "",
        "after the last chunk (or after merging other cores):  out = O / l",
    ], size=10.5, lh=17, mono=True)
    f.lines(fx, fy + 262, [
        "Why chunks: the whole K/V for one item at 8k is 2 x 8192 x 128 bfp8 = 2.2 MB,",
        "more than L1. A 512-position chunk of K and V is 2 x 68 KB and is double-",
        "buffered, so the reader can fetch chunk k+1 while compute works on chunk k.",
        "",
        "Why 'online': m and l let each chunk's softmax be normalised later, so the",
        "chunks (and cores) can be combined in any order with the same a-factors.",
    ], size=10.5, lh=15, color=MUTED)
    f.caption("Fig. B1. One item = one batch row x one KV head; its 16 query heads share the tile's rows, so a single 1 x 4-tile Q does all of them at once.")
    return f


def fig_work_layout() -> Fig:
    f = Fig(940, 360, "How the 128, 32 or 4 work items of a decode step are laid out on the 8 x 8 core grid")
    f.text(14, 24, "Work layout on one chip (Llama-3.1-8B, TP = 2: 4 KV heads and 16 query heads per chip)", size=15, weight="bold")

    def grid(x, y, title, sub, colour_fn, s=22, gap=3):
        f.text(x, y - 22, title, size=11.5, weight="bold")
        f.text(x, y - 8, sub, size=10, color=MUTED)
        for r in range(8):
            for c in range(8):
                fill, stroke, lab = colour_fn(r, c)
                tx, ty = x + c * (s + gap), y + r * (s + gap)
                f.parts.append(
                    f'<rect x="{tx}" y="{ty}" width="{s}" height="{s}" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
                )
                if lab:
                    f.text(tx + s / 2, ty + s / 2 + 3, lab, size=7.5, anchor="middle", mono=True)

    pal = ["#dbe7fb", "#fde7d6", "#dff3e6", "#ece4fb", "#fbe1e1", "#f7f2e4", "#e8e8ee", "#d9f0f5"]

    def b32(r, c):
        i = r * 8 + c           # core index
        b = i // 2              # batch row: 2 cores per batch (4 heads / 2 heads per core)
        return pal[b % 8], LINE, f"b{b}"

    def b8(r, c):
        i = r * 8 + c
        item = i // 2           # 32 items x 2 cores
        b = item // 4
        return pal[b % 8], LINE, f"{item % 4}{'r' if i % 2 == 0 else ''}"

    def b1(r, c):
        i = r * 8 + c
        item = i // 16          # 4 items x 16 cores
        return pal[item], LINE, "r" if i % 16 == 0 else ""

    grid(30, 70, "batch 32: 128 items, 2 heads per core", "no sequence split, no reduction", b32)
    grid(330, 70, "batch 8: 32 items, 2 cores per item", "each core does half the chunks; 1 merge round", b8)
    grid(630, 70, "batch 1: 4 items, 16 cores per item", "each core does 1/16 of the chunks; 4 merge rounds", b1)
    f.lines(30, 290, [
        "cores per item = min(64, max_cores_per_head x B x n_kv) / B / n_kv;  heads per core = ceil(n_kv / (64 / B)).",
        "Label: b<batch> for the batch row at 32; at 8 the digit is the KV head and 'r' marks the reducer (root) core of each pair; at 1 the shade is the KV head.",
        "Chunks are dealt backwards: the root core (core_num 0) owns the LAST chunks, where the causal mask lands; core N-1 owns the first.",
    ], size=10, lh=14, color=MUTED)
    f.caption("Fig. B2. The same kernel binary runs on every core; per-core runtime arguments say which (batch, head) item, which slice of chunks, and which tree role it has.")
    return f


def fig_core_dataflow() -> Fig:
    f = Fig(940, 720, "Inside one core for one K chunk: reader, compute chain with CB indices, writer")
    f.text(14, 24, "One core, one chunk: who does what, through which circular buffer", size=15, weight="bold")
    f.text(14, 42, "numbers in [ ] are DST round trips for a 512-position chunk with dst_size = 4 (batch 32 layout: 1 x 16 score tiles)", size=10.5, color=MUTED)

    # columns
    rx, cx, wx = 30, 300, 700
    colw = [250, 380, 210]
    f.rect(rx - 8, 56, colw[0], 640, (PLATE, C_NOC[1]), r=6)
    f.rect(cx - 8, 56, colw[1], 640, (PLATE, C_FPU[1]), r=6)
    f.rect(wx - 8, 56, colw[2], 640, (PLATE, C_NOC[1]), r=6)
    f.text(rx, 74, "reader (BRISC, NoC 0)", size=12, weight="bold")
    f.text(cx, 74, "compute (TRISC 0/1/2)", size=12, weight="bold")
    f.text(wx, 74, "writer (NCRISC, NoC 1)", size=12, weight="bold")

    # reader items
    y = 96
    f.box(rx, y, 226, 40, "Q for this batch row", "DRAM -> c_0, once (4 tiles)", role=C_K, size=10.5)
    y += 56
    f.text(rx, y, "per chunk (double buffered, so up to", size=10, color=MUTED)
    f.text(rx, y + 13, "one chunk ahead of compute):", size=10, color=MUTED)
    y += 26
    f.box(rx, y, 226, 54, "K chunk, 64 tiles bfp8", "read tile by tile in column-major order into c_1", role=C_K, size=10.5)
    y += 68
    f.box(rx, y, 226, 54, "V chunk, 64 tiles bfp8", "row-major into c_2", role=C_K, size=10.5)
    y += 74
    f.lines(rx, y, [
        "each tile is one NoC read of 1088 B;",
        "a barrier every few tiles keeps the",
        "number of reads in flight bounded.",
        "",
        "Q is read once per item, K and V",
        "once per chunk; nothing is reused",
        "across decode steps (no cache).",
    ], size=10, lh=13, color=MUTED)

    # compute chain
    steps = [
        ("S = Q K^T", "c_0 x c_1 -> c_24 (16 tiles bf16)  [4: 1x4 tiles each, inner 4]", C_FPU),
        ("S += mask (last chunk only)", "c_24 += c_3, tile by tile  [16]", C_FPU),
        ("m_new = max(m_prev, rowmax S)", "c_24, c_28 -> c_27  [1]", C_FPU),
        ("P = exp((S - m_new) * scale)", "in place in c_24, 4 tiles per trip (FPU subtract, SFPU exp)  [4]", C_SFPU),
        ("r = rowsum P", "c_24 -> c_29  [1]", C_FPU),
        ("PV = P V", "c_24 x c_2 -> c_25 (4 tiles)  [1: 1x4 tiles, inner 16]", C_FPU),
        ("a = exp((m_prev - m_new) * scale)", "c_28, c_27 -> c_31  [1]", C_SFPU),
        ("l = a * l_prev + r", "c_30 *= c_31 [1];  c_29 += c_30 [1]", C_FPU),
        ("O = a * O + PV", "c_26 *= c_31 [4];  c_26 += c_25 [4]", C_FPU),
        ("m_prev <- m_new, l_prev <- l", "c_27 -> c_28, c_29 -> c_30 (ping-pong)  [2]", C_FPU),
    ]
    y = 96
    bh = 46
    for i, (t, sub, role) in enumerate(steps):
        f.box(cx, y, 356, bh, t, sub, role=role, size=10.5)
        if i < len(steps) - 1:
            f.arrow(cx + 178, y + bh, cx + 178, y + bh + 10)
        y += bh + 10
    f.text(cx, y + 8, "about 24 round trips per chunk (13 on the first, +16 on the masked last one)", size=10, color=MUTED)
    f.text(cx, y + 22, "after the last chunk: l = 1/l, O *= l  ->  c_20 (or c_16/17/18 for the tree)", size=10, color=MUTED)

    # writer items
    y = 96
    f.box(wx, y, 190, 54, "scalers, before the loop", "c_5 reduce scaler, c_11 column", role=C_CB, size=10.5)
    f.text(wx + 95, y + 48, "identity, c_12 zero tile", size=9, anchor="middle", color=MUTED)
    y += 68
    f.box(wx, y, 190, 54, "causal mask", "0 / -inf tiles into c_3, from", role=C_CB, size=10.5)
    f.text(wx + 95, y + 48, "cur_pos, in L1 (no DRAM read)", size=9, anchor="middle", color=MUTED)
    y += 68
    f.box(wx, y, 190, 66, "tree reduction", "children's (l, m, O) -> c_7, c_6, c_16;", role=C_NOC, size=10.5)
    f.text(wx + 95, y + 52, "own (l, m, O) -> parent's c_19", size=9, anchor="middle", color=MUTED)
    y += 80
    f.box(wx, y, 190, 54, "output", "c_20 -> DRAM (per-head rows as", role=C_K, size=10.5)
    f.text(wx + 95, y + 48, "partial tiles when GQA)", size=9, anchor="middle", color=MUTED)
    y += 74
    f.lines(wx, y, [
        "The writer is idle during the",
        "chunk loop; all its work is",
        "before (mask, scalers) or after",
        "(reduce, write) the loop.",
        "",
        "cur_pos arrives through c_8",
        "(writer) and c_15 (compute).",
    ], size=10, lh=13, color=MUTED)

    # arrows across columns
    f.arrow(rx + 226, 116, cx, 119, dash="3 3")
    f.arrow(rx + 226, 205, cx, 119, dash="3 3")
    f.arrow(rx + 226, 273, cx, 399, dash="3 3")
    f.arrow(wx, 191, cx + 356, 175, dash="3 3")
    f.arrow(cx + 356, 620, wx, 320, dash="3 3")
    f.caption("Fig. B3. The compute kernel is straight-line code over CBs; the reader stays one chunk ahead; the writer only frames the loop. Every box in the middle column is one or more DST round trips.")
    return f


def fig_chunk_timeline() -> Fig:
    f = Fig(940, 300, "Pipelining across chunks: the reader stays one chunk ahead of compute; the writer frames the loop")
    f.text(14, 24, "Timeline of one work item over its chunks", size=15, weight="bold")
    lx, ly, lw = 120, 70, 780
    lanes = ["reader", "compute", "writer"]
    lh = 40
    for i, n in enumerate(lanes):
        y = ly + i * (lh + 14)
        f.text(lx - 8, y + lh / 2 + 4, n, size=11, anchor="end", weight="bold")
        f.line(lx, y + lh / 2, lx + lw, y + lh / 2, color=LINE, dash="2 3")

    def seg(lane, x0, x1, label, role, size=10):
        y = ly + lane * (lh + 14) + 5
        f.rect(x0, y, x1 - x0, lh - 10, role, r=3)
        f.text((x0 + x1) / 2, y + (lh - 10) / 2 + 4, label, size=size, anchor="middle")

    cw = 128
    seg(0, lx + 10, lx + 60, "Q", C_K)
    for k in range(4):
        seg(0, lx + 62 + k * cw, lx + 62 + k * cw + 100, f"K{k} V{k}", C_K)
    seg(0, lx + 62 + 4 * cw, lx + 62 + 4 * cw + 40, "...", C_IDLE)
    for k in range(4):
        seg(1, lx + 100 + k * cw, lx + 100 + k * cw + cw - 6, f"chunk {k}", C_FPU)
    seg(1, lx + 100 + 4 * cw, lx + 100 + 4 * cw + 40, "...", C_IDLE)
    seg(2, lx + 10, lx + 96, "mask, scalers", C_CB)
    seg(2, lx + 100 + 4 * cw + 50, lx + 100 + 4 * cw + 160, "reduce, write", C_NOC)
    f.arrow(lx + 62 + cw + 50, ly + lh + 2, lx + 100 + cw + 10, ly + lh + 12, dash="3 3")
    f.text(lx + 62 + cw + 110, ly + lh + 9, "K1/V1 must be in c_1/c_2 before chunk 1 starts; the CBs hold two chunks, so the reader is at most one ahead", size=9.5, color=MUTED)
    f.lines(lx, ly + 3 * (lh + 14) + 8, [
        "Stock at 4k, batch 32: about 44 us of compute per chunk per item against 136 KB of K+V per chunk. The stock op reaches 77 % of the",
        "bandwidth the chip can deliver in this position, i.e. the reader lane and the compute lane are close to the same length: it is near DRAM-bound.",
        "The fused Palu kernel reads fewer bytes per chunk but its compute lane is longer (Fig. C6), so the reader waits on compute instead.",
    ], size=10.5, lh=14, color=MUTED)
    f.caption("Fig. B4. Whichever lane is longer sets the pace; double buffering hides the other lane only up to one chunk.")
    return f


def fig_tree_reduction() -> Fig:
    f = Fig(940, 470, "Tree reduction across the cores of one item: payloads over the NoC, per-round semaphore nibbles, softmax-correct merge")
    f.text(14, 24, "When one item spans several cores: the reduction tree", size=15, weight="bold")

    # 4 cores
    cx0, cy = 60, 80
    cores = [(0, "core 0", "vid 3, root"), (1, "core 1", "vid 2"), (2, "core 2", "vid 1"), (3, "core 3", "vid 0")]
    pos = {}
    for i, (n, name, vid) in enumerate(cores):
        x = cx0 + i * 190
        role = C_DST if i == 0 else C_NOC
        f.box(x, cy, 150, 50, name, vid, role=role, size=11.5)
        pos[i] = (x + 75, cy)
    f.text(cx0, cy + 186, "chunks are dealt backwards: core 0 (the root) owns the last ones, where the causal mask lands; core 3 the first. vid = N - 1 - core_num; a core with no chunk is skipped", size=10, color=MUTED)
    # round 0 arrows
    f.path_arrow([(pos[3][0], cy + 50), (pos[3][0], cy + 110), (pos[2][0] + 20, cy + 110), (pos[2][0] + 20, cy + 52)])
    f.text((pos[3][0] + pos[2][0]) / 2, cy + 124, "round 0: (l, m, O) -> parent", size=10, anchor="middle", color=MUTED)
    f.path_arrow([(pos[1][0], cy + 50), (pos[1][0], cy + 110), (pos[0][0] + 20, cy + 110), (pos[0][0] + 20, cy + 52)])
    f.text((pos[1][0] + pos[0][0]) / 2, cy + 124, "round 0", size=10, anchor="middle", color=MUTED)
    # round 1
    f.path_arrow([(pos[2][0] - 20, cy + 50), (pos[2][0] - 20, cy + 150), (pos[0][0] - 20, cy + 150), (pos[0][0] - 20, cy + 52)])
    f.text((pos[2][0] + pos[0][0]) / 2, cy + 164, "round 1: core 2's merged (l, m, O) -> root", size=10, anchor="middle", color=MUTED)

    # protocol panel
    px, py = 60, 270
    f.rect(px, py, 400, 150, (PLATE, C_NOC[1]), r=6)
    f.text(px + 10, py + 18, "Sender (child), in the writer kernel", size=11, weight="bold")
    f.lines(px + 10, py + 36, [
        "wait until compute has moved O, m, l into c_16, c_17, c_18",
        "NoC-write l, m, O into the parent's c_19 at slot [round]",
        "(same L1 offset on every core, so the address is known)",
        "noc write barrier, then semaphore.up(parent, 1 << (4 * round))",
        "pop c_16/17/18 and finish: a sender has nothing more to do",
    ], size=10, lh=15, mono=True)
    f.text(px + 10, py + 128, "semaphore value: nibble r (bits 4r..4r+3) counts round r's arrivals", size=10, color=MUTED)

    qx = 490
    f.rect(qx, py, 420, 150, (PLATE, C_DST[1]), r=6)
    f.text(qx + 10, py + 18, "Receiver (parent), writer + compute", size=11, weight="bold")
    f.lines(qx + 10, py + 36, [
        "writer: poll nibble[round] >= 1, then L1->L1 copy the slot",
        "        into c_7 (l), c_6 (m), c_16 (O) and push them",
        "compute: correction_block merges child into local state:",
        "   m' = max(m, m_c);  a = exp((m - m')s);  b = exp((m_c - m')s)",
        "   l' = a l + b l_c;   O' = a O + b O_c",
        "root only: out = O / l -> c_20 -> DRAM",
    ], size=10, lh=15, mono=True)
    f.text(qx + 10, py + 128, "one merge = the same correction the chunk loop applies, applied to another core's state", size=10, color=MUTED)
    f.caption("Fig. B5. ceil(log2 N) rounds for N cores per item; c_19 holds one (l, m, O) slot per round. The Palu group kernel gives each head its own slots and counts heads in the nibble.")
    return f


# =============================================================================================
# C. The fused Palu kernel
# =============================================================================================
def fig_palu_data() -> Fig:
    f = Fig(940, 400, "What Palu changes in the data the decode kernel reads: a narrow latent plus per-head weights instead of the full K cache")
    f.text(14, 24, "Stock K cache vs Palu K latent, per (batch row, KV head)", size=15, weight="bold")
    s = 14
    # stock
    sx, sy = 40, 70
    f.text(sx, sy - 10, "stock: K cache S x 128, bfp8", size=11.5, weight="bold")
    f.tiles(sx, sy, 4, 8, s, role=C_K, labels=lambda r, c: "..." if (r == 7 and c == 0) else None, size=7)
    f.text(sx + 4 * (s + 2) + 8, sy + 30, "136 B per position", size=10.5, color=MUTED)
    f.text(sx + 4 * (s + 2) + 8, sy + 44, "already rotated (RoPE applied", size=10.5, color=MUTED)
    f.text(sx + 4 * (s + 2) + 8, sy + 58, "when the token was written)", size=10.5, color=MUTED)
    f.text(sx, sy + 8 * (s + 2) + 16, "read per decode step per item: S x 136 B", size=10.5, color=MUTED)

    # palu
    px, py = 330, 70
    f.text(px, py - 10, "Palu: K latent S x r_k, bfp8", size=11.5, weight="bold")
    f.tiles(px, py, 1, 8, s, role=C_PALU, labels=lambda r, c: "..." if r == 7 else None, size=7)
    f.text(px + 1 * (s + 2) + 8, py + 16, "r_k = 32 / 64 / 96 = 1 / 2 / 3 tiles wide", size=10.5, color=MUTED)
    f.text(px + 1 * (s + 2) + 8, py + 30, "34 / 68 / 102 B per position", size=10.5, color=MUTED)
    f.text(px + 1 * (s + 2) + 8, py + 44, "unrotated: RoPE must be applied", size=10.5, color=MUTED)
    f.text(px + 1 * (s + 2) + 8, py + 58, "after the up-projection", size=10.5, color=MUTED)
    # weights
    wx, wy = 560, 70
    f.text(wx, wy - 10, "per KV head, once per call: w_rot = [u_k | u_k T],  r_k x 256 bf16", size=11.5, weight="bold")
    f.tiles(wx, wy, 8, 1, s, role=C_PALU, labels=lambda r, c: ("A" if c < 4 else "B"), size=7)
    f.text(wx, wy + 32, "A columns: u_k (r_k x 128).  B columns: u_k T, where T is RoPE's pair-swap matrix", size=10, color=MUTED)
    f.text(wx, wy + 46, "so that (lat u_k) T = lat (u_k T): the rotated half comes out of the same matmul.", size=10, color=MUTED)
    f.text(wx, wy + 76, "plus [cos | sin] rows (S x 256 bf16): Phase 1 streams the rows of each chunk,", size=10, color=MUTED)
    f.text(wx, wy + 90, "Phase 2 keeps only rows 0 .. c-1 resident (Fig. C4).", size=10, color=MUTED)
    f.text(wx, wy + 116, "V: the V latent S x r_v is used directly as V (its u_v is folded into", size=10, color=MUTED)
    f.text(wx, wy + 130, "the output projection) for per-head Palu; for G-LRD it is rebuilt on core.", size=10, color=MUTED)

    # identity
    ix, iy = 40, 250
    f.rect(ix, iy, 870, 110, (PLATE, C_PALU[1]), r=6)
    f.text(ix + 12, iy + 20, "The identity the kernel implements, per K row tile i (32 positions) and column tile j (32 dims)", size=11.5, weight="bold")
    f.lines(ix + 12, iy + 42, [
        "K_rot[i, j] = A[i, j] * cos[i, j] + B[i, j] * sin[i, j]       with  [A | B] = lat_i @ [u_k | u_k T]",
        "= RoPE_p(lat_p u_k) for every position p in the tile, which is exactly what stage 1 computed as matmul + rotary_embedding_llama on the host side.",
    ], size=10.5, lh=17, mono=True)
    f.text(ix + 12, iy + 92, "The stock QK matmul then reads K_rot from c_1 exactly as it reads a stock K chunk; nothing downstream of that point knows Palu exists.", size=10, color=MUTED)
    f.caption("Fig. C1. Bytes per position drop from 136 to 34-102, but every position now costs a small matmul and a rotation on the core that reads it.")
    return f


def fig_palu_cb_map() -> Fig:
    f = Fig(940, 430, "Circular-buffer slot map: which stock slots the Palu op reuses, and what they hold (chunk 512, r_k 32, r_v 128, Phase 2)")
    f.text(14, 24, "The 32 CB slots: stock flash decode vs the Palu op", size=15, weight="bold")
    stock = {
        0: ("Q", "4 tiles"), 1: ("K chunk x2", "128 bfp8"), 2: ("V chunk x2", "128 bfp8"), 3: ("causal mask", "16"),
        4: ("sink", "unused here"), 5: ("reduce scaler", "1"), 6: ("m from child", "1"), 7: ("l from child", "1"),
        8: ("cur_pos (writer)", "1 page"), 9: ("page table", "unused here"), 10: ("row-major Q", "unused here"),
        11: ("col identity", "1"), 12: ("zero tile", "1"), 13: ("sliding window", "unused here"),
        14: ("block pad", "unused here"), 15: ("cur_pos (compute)", "1 page"),
        16: ("O out / child O", "4"), 17: ("m out", "1"), 18: ("l out", "1"), 19: ("tree slots", "(N-1) x 6"), 20: ("final out", "4"),
        21: ("child l tmp", "1"), 22: ("exp diff 2", "1"), 23: ("O acc 2", "4"), 24: ("scores S", "16"), 25: ("PV", "4"),
        26: ("O acc", "4"), 27: ("max 1", "1"), 28: ("max 2", "1"), 29: ("sum 1", "1"), 30: ("sum 2", "1"), 31: ("exp diff", "1"),
    }
    palu = {
        1: ("K_rot x2 (built)", "128 bf16 = 256 KB"),
        4: ("w_rot block(s)", "[u_k|u_k T] (+u_v) + T"),
        9: ("latent chunk x2", "32 bfp8 = 35 KB"),
        10: ("AB row block x2", "16 bf16 = 32 KB"),
        12: ("q state [q~|q~T]", "16 fp32 = 64 KB"),
        13: ("cos/sin rows 0..c-1", "128 bf16 = 256 KB"),
        14: ("rot_step", "8 fp32 = 32 KB"),
        26: ("O acc, H blocks", "group mode"), 28: ("max 2, H blocks", "group mode"), 30: ("sum 2, H blocks", "group mode"),
    }
    x0, y0, w, h, gx, gy = 30, 60, 106, 62, 6, 8
    for i in range(32):
        r, c = divmod(i, 8)
        x, y = x0 + c * (w + gx), y0 + r * (h + gy)
        st = stock[i]
        if i in palu and i in (26, 28, 30):
            role, tag = C_L1, "sized H x in group mode"
        elif i in palu:
            role, tag = C_PALU, "Palu"
        elif "unused" in st[1]:
            role, tag = C_IDLE, "unused"
        else:
            role, tag = C_L1, ""
        f.rect(x, y, w, h, role, r=4)
        f.text(x + 6, y + 14, f"c_{i}", size=10.5, weight="bold", mono=True)
        if i in palu and i not in (26, 28, 30):
            f.text(x + 6, y + 28, st[0], size=9, color=MUTED)
            f.text(x + 6, y + 42, palu[i][0], size=9.5, weight="bold")
            f.text(x + 6, y + 55, palu[i][1], size=8.5, color=MUTED, mono=True)
        else:
            f.text(x + 6, y + 30, st[0], size=9.5)
            f.text(x + 6, y + 44, st[1], size=8.5, color=MUTED, mono=True)
            if i in (26, 28, 30):
                f.text(x + 6, y + 56, "x H in group mode", size=8.5, color=C_PALU[1])
    f.legend(30, 360, [(C_L1, "stock slot, unchanged"), (C_IDLE, "stock slot idle in the Palu envelope"), (C_PALU, "slot re-used by the Palu op (small grey line: its stock use)")], step=15)
    f.text(560, 356, "Extra L1 at chunk 512, r_k 32, Phase 2: roughly 0.6 MB on top of the stock", size=10, color=MUTED)
    f.text(560, 370, "footprint; c_1 alone grows 120 KB (K_rot is bf16, stock K is bfp8).", size=10, color=MUTED)
    f.text(560, 384, "At r_k 96 the total reaches ~990 KB of the ~1.41 MB available.", size=10, color=MUTED)
    f.caption("Fig. C2. The op keeps the stock slot numbering so the flash-decode code paths are untouched; the six idle stock features are asserted off so their slots can be borrowed.")
    return f


def fig_palu_build_k() -> Fig:
    f = Fig(940, 580, "palu_build_k_chunk: one K row tile is one small matmul plus four multiply-multiply-add trips, packed column-major into c_1")
    f.text(14, 24, "Rebuilding one K chunk on the core: per row tile i (32 positions)", size=15, weight="bold")
    s = 18
    # latent row
    lx, ly = 40, 70
    f.text(lx, ly - 12, "latent row tile i", size=11, weight="bold")
    f.tiles(lx, ly, 1, 1, s, role=C_PALU)
    f.text(lx, ly + 34, "1 x Rt tiles", size=9.5, color=MUTED)
    f.text(lx, ly + 46, "from c_9", size=9.5, color=MUTED)
    f.text(lx + 30, ly + 13, "x", size=14, weight="bold")
    # weight block
    wx, wy = 150, 70
    f.text(wx, wy - 12, "w_rot block (resident in c_4)", size=11, weight="bold")
    f.tiles(wx, wy, 8, 1, s, role=C_PALU, labels=lambda r, c: ("A" if c < 4 else "B"), size=8)
    f.text(wx, wy + 34, "Rt x 8 tiles: [u_k | u_k T]", size=9.5, color=MUTED)
    f.arrow(wx + 8 * (s + 2) + 6, wy + 9, wx + 8 * (s + 2) + 40, wy + 9)
    # AB
    ax, ay = 370, 70
    f.text(ax, ay - 12, "AB_i in c_10 (8 tiles bf16)", size=11, weight="bold")
    f.tiles(ax, ay, 8, 1, s, role=C_SFPU, labels=lambda r, c: (f"A{c}" if c < 4 else f"B{c - 4}"), size=8)
    f.text(ax, ay + 34, "2 DST round trips: 4 output tiles each (dst_size = 4), inner dim Rt", size=9.5, color=MUTED)
    f.text(ax, ay + 46, "matmul_block_init + reconfig before, pack_tile x 4 after, per trip", size=9.5, color=MUTED)

    # per j
    jx, jy = 40, 140
    f.rect(jx, jy, 870, 150, (PLATE, C_DST[1]), r=6)
    f.text(jx + 12, jy + 20, "then for each column tile j = 0..3: one DST round trip", size=11.5, weight="bold")
    f.lines(jx + 12, jy + 42, [
        "tile_regs_acquire()",
        "mul_tiles(A_j,  cos[i, j])  -> dst 0        (FPU)",
        "mul_tiles(B_j,  sin[i, j])  -> dst 1        (FPU)",
        "add_binary_tile(dst 0, dst 1 -> dst 0)     (SFPU: a full-tile vector pass)",
        "tile_regs_commit(); tile_regs_wait(); pack_tile(dst 0 -> c_1 at index j * Sk_chunk_t + i); tile_regs_release()",
    ], size=10.5, lh=16, mono=True)
    f.text(jx + 12, jy + 138, "cos/sin tiles come from c_13 (row i of the table for Phase 1; row i mod c, i.e. always rows 0..15, for Phase 2). mul_init + add_binary_tile_init run once per row tile.", size=9.5, color=MUTED)

    # K_rot layout
    kx, ky = 40, 320
    f.text(kx, ky - 12, "K_rot chunk in c_1, column-major tile order = what the stock Q K^T matmul reads", size=11, weight="bold")
    def lab(r, c):
        return f"{c * 16 + r}" if r < 4 else ("..." if c == 0 else "")
    f.tiles(kx, ky, 4, 5, s, role=C_K, labels=lab, size=7.5)
    f.text(kx + 4 * (s + 2) + 10, ky + 14, "16 row tiles x 4 column tiles;", size=10, color=MUTED)
    f.text(kx + 4 * (s + 2) + 10, ky + 28, "row tile i lands at j * 16 + i for j = 0..3", size=10, color=MUTED)
    f.text(kx + 4 * (s + 2) + 10, ky + 42, "(the reader wrote stock K in this order too)", size=10, color=MUTED)

    # cost box
    cx2, cy2 = 360, 310
    f.rect(cx2, cy2, 550, 190, (PLATE, LINE), r=6)
    f.text(cx2 + 12, cy2 + 20, "What this costs per chunk of 512 positions (16 row tiles), r_k = 32", size=11.5, weight="bold")
    rows = [
        ("", "stock", "fused (K rebuild only)"),
        ("DST round trips", "~24 for the whole flash step", "6 x 16 = 96, before the flash step"),
        ("engine inits", "a handful", "3 x 16 = 48 (matmul, mul, add)"),
        ("format reconfigs", "~10", "4 x 16 = 64"),
        ("SFPU passes", "exp on 16 score tiles", "+ 64 full-tile adds"),
        ("FPU work", "QK (64 tile-mults), PV (64)", "+ AB matmul 128 tile-mults + 128 eltwise muls"),
        ("DRAM bytes", "136 KB (K + V)", "17 KB latent + 68 KB V (r_v 128)"),
    ]
    for i, row in enumerate(rows):
        y = cy2 + 42 + i * 19
        for j, cell in enumerate(row):
            f.text(cx2 + 12 + [0, 130, 300][j], y, cell, size=10, weight="bold" if i == 0 else "normal",
                   color=INK if i else MUTED)
        if i == 0:
            f.line(cx2 + 8, y + 5, cx2 + 540, y + 5, color=LINE)
    f.text(cx2 + 12, cy2 + 180, "measured: the rebuild adds ~0.8 us per row tile of which ~0.28 us is the matmul (stage 2d)", size=9.5, color=MUTED)
    f.caption("Fig. C3. The rebuild is cheap in bytes and FLOPs and expensive in round trips: it adds ~100 small DST acquisitions per chunk to a flash step that needs ~24.")
    return f


def fig_palu_phase2() -> Fig:
    f = Fig(940, 520, "Phase 1 streams cos/sin per chunk; Phase 2 keeps a chunk-relative table resident and rotates the query once per chunk instead")
    f.text(14, 24, "Where the rotation angles come from: Phase 1 vs Phase 2", size=15, weight="bold")

    # position axis
    ax, ay, aw = 60, 90, 820
    f.line(ax, ay, ax + aw, ay, color=INK, sw=1.4)
    for k in range(5):
        x = ax + k * (aw / 4)
        f.line(x, ay - 6, x, ay + 6, color=INK)
        f.text(x, ay + 22, f"{k * 512}", size=10, anchor="middle", mono=True)
    f.text(ax + aw + 10, ay + 4, "position", size=10, color=MUTED)
    for k in range(4):
        x = ax + k * (aw / 4)
        f.rect(x + 3, ay - 40, aw / 4 - 6, 26, C_K if k % 2 == 0 else C_PALU, r=3)
        f.text(x + aw / 8, ay - 23, f"chunk {k}: s0 = {k * 512}", size=10, anchor="middle")
    f.text(ax, ay - 52, "K side needs R_{s0 + t} for the positions of chunk k, t = 0 .. 511", size=10.5, color=MUTED)

    # phase 1
    p1x, p1y = 40, 140
    f.rect(p1x, p1y, 410, 150, (PLATE, C_K[1]), r=6)
    f.text(p1x + 12, p1y + 20, "Phase 1 (checkpoint): absolute angles", size=11.5, weight="bold")
    f.lines(p1x + 12, p1y + 42, [
        "reader streams [cos | sin] rows s0 .. s0+511 into c_13 per chunk",
        "= 512 x 256 bf16 = 256 KB per chunk per core: 2x the bytes of the",
        "K it replaces, so it can only ever lose on traffic. Also forces",
        "chunk 128 so the streamed table fits L1.",
        "",
        "kernel: K_rot[i, j] = A * cos[s0+i, j] + B * sin[s0+i, j]",
    ], size=10.5, lh=15, mono=False, color=MUTED)

    # phase 2
    p2x, p2y = 470, 140
    f.rect(p2x, p2y, 440, 150, (PLATE, C_PALU[1]), r=6)
    f.text(p2x + 12, p2y + 20, "Phase 2 (the one measured): chunk-relative angles", size=11.5, weight="bold")
    f.lines(p2x + 12, p2y + 42, [
        "2-D rotations commute, so for query position p and key s0 + t:",
        "   q R_p . (k R_{s0+t})^T  =  (q R_{p-s0}) . (k R_t)^T",
        "K side uses rows t = 0..511 only: read ONCE per core, resident in c_13.",
        "The chunk base s0 moves to the query: a per-chunk state",
        "   q~_k = q R_{-k c},  advanced by the constant R_{-c} each chunk.",
        "Q K^T reads q~ from c_12 instead of Q from c_0; nothing else changes.",
    ], size=10.5, lh=15, mono=False, color=MUTED)

    # state update
    ux, uy = 40, 310
    f.rect(ux, uy, 870, 150, (PLATE, C_DST[1]), r=6)
    f.text(ux + 12, uy + 20, "The query step (palu_step_q), fp32 tiles in c_12, once per chunk", size=11.5, weight="bold")
    f.lines(ux + 12, uy + 42, [
        "state block = [ q~ | q~ T ]",
        "q~'     = q~ * cos_c  -  (q~ T) * sin_c",
        "(q~' T) = (q~ T) * cos_c  +  q~ * sin_c",
    ], size=10.5, lh=17, mono=True)
    f.lines(ux + 330, uy + 42, [
        "2n tiles, n = 4; T = the pair-swap tile carried in c_4, so q~ T is 4 single-tile matmuls at seed time",
        "cos_c, sin_c = row c of the table, broadcast to all 32 rows (rot_step, c_14)",
        "a complex multiply, no matmul: 2 x 4 DST trips of (mul, mul, sub or add)",
    ], size=10, lh=17, color=MUTED)
    f.lines(ux + 12, uy + 100, [
        "The state is kept in fp32 tiles so 16 steps at 8k do not accumulate bf16 rounding; the multiplies themselves run at ~11 mantissa bits (Wormhole unpacks fp32 into 19-bit Src).",
        "Cores that own a later slice of the sequence seed q~ and step it k_chunk_start times before their first chunk. Cost per chunk: 8 round trips on 8 tiles, independent of r_k.",
    ], size=10, lh=14, color=MUTED)
    f.caption("Fig. C4. Phase 2 removes the cos/sin traffic entirely: the table is 256 KB read once per core per call, and the per-chunk price is one small in-register rotation of the query.")
    return f


def fig_palu_group_mode() -> Fig:
    f = Fig(940, 560, "G-LRD: four heads share one latent; group-shared mode reads each latent chunk once per core and keeps per-head state as rings")
    f.text(14, 24, "In-kernel V and the group-shared loop (G-LRD, stage 2b/2c)", size=15, weight="bold")

    # left: per-head kernel
    lx, ly = 40, 60
    f.rect(lx, ly, 420, 220, (PLATE, LINE), r=6)
    f.text(lx + 12, ly + 20, "per-head kernel: an item is (batch, head)", size=11.5, weight="bold")
    f.box(lx + 20, ly + 40, 180, 70, "core A: heads 0, 1", "head 0 loop, then head 1 loop", role=C_NOC, size=10.5)
    f.box(lx + 220, ly + 40, 180, 70, "core B: heads 2, 3", "head 2 loop, then head 3 loop", role=C_NOC, size=10.5)
    f.box(lx + 120, ly + 150, 180, 44, "group latent chunk", "S x r_k, one per 4 heads", role=C_PALU, size=10.5)
    for x in (lx + 60, lx + 150, lx + 270, lx + 360):
        f.arrow(lx + 210, ly + 150, x, ly + 112, dash="3 3")
    f.text(lx + 20, ly + 210, "each latent chunk is read 4 times per step (2 per core); each head has its own", size=9.5, color=MUTED)
    f.text(lx + 20, ly + 222, "weight block and flash state, sequentially.", size=9.5, color=MUTED)

    # right: group mode
    rx, ry = 480, 60
    f.rect(rx, ry, 430, 220, (PLATE, C_PALU[1]), r=6)
    f.text(rx + 12, ry + 20, "group-shared mode: an item is (batch, H heads), chunk outer, head inner", size=11.5, weight="bold")
    f.lines(rx + 12, ry + 40, [
        "for chunk k:",
        "    (Phase 2) q~ <- q~ R_{-c}           once for the core: Q is per core, not per head",
        "    for h in 0 .. H-1:",
        "        K_h  = build_k(latent chunk, w block h)   -> c_1",
        "        S, softmax update of state h (m_h, l_h, O_h)",
        "        V_h  = build_v(V latent chunk, u_v block h) -> c_1 (K_h is spent)",
        "        O_h += P V_h",
        "    pop the shared latent chunks once",
        "for h: tree-reduce head h (sequential), emit head h",
    ], size=10, lh=14, mono=True)
    f.text(rx + 12, ry + 180, "latent chunk read once per core; H weight blocks resident in c_4 (this bounds H).", size=9.5, color=MUTED)
    f.text(rx + 12, ry + 192, "Auto selection: H = 2 at batch 32 (1 core per item), per-head kernel below (Fig. B2).", size=9.5, color=MUTED)

    # ring
    gx, gy = 40, 310
    f.rect(gx, gy, 420, 200, (PLATE, C_CB[1]), r=6)
    f.text(gx + 12, gy + 20, "per-head state as a FIFO ring (c_26 O, c_28 m, c_30 l, sized H blocks)", size=11.5, weight="bold")
    bw, bh = 120, 30
    for i, lab in enumerate(["head 0 block", "head 1 block"]):
        f.rect(gx + 20 + i * (bw + 8), gy + 40, bw, bh, C_CB, r=3)
        f.text(gx + 20 + i * (bw + 8) + bw / 2, gy + 60, lab, size=10, anchor="middle")
    f.text(gx + 20, gy + 92, "front = the head whose turn it is.  A head's step does wait_front, pop_front,", size=10, color=MUTED)
    f.text(gx + 20, gy + 106, "reserve_back, push_back on each ring: its block goes to the back, the next", size=10, color=MUTED)
    f.text(gx + 20, gy + 120, "head's comes to the front. Because every head runs the identical sequence,", size=10, color=MUTED)
    f.text(gx + 20, gy + 134, "the ring hands each head its own block back on the next chunk.", size=10, color=MUTED)
    f.text(gx + 20, gy + 156, "palu_rotate_blocks: pop + reserve + push with no data movement (the CB is", size=10, color=MUTED)
    f.text(gx + 20, gy + 170, "exactly full, so the bytes re-published at the back are the ones just popped).", size=10, color=MUTED)

    # V rebuild
    vx, vy = 480, 310
    f.rect(vx, vy, 430, 200, (PLATE, C_PALU[1]), r=6)
    f.text(vx + 12, vy + 20, "in-kernel V (palu_build_v_chunk)", size=11.5, weight="bold")
    s = 16
    f.tiles(vx + 20, vy + 40, 2, 4, s, role=C_PALU, labels=lambda r, c: "..." if (r == 3 and c == 0) else None, size=7)
    f.text(vx + 20, vy + 40 + 4 * (s + 2) + 12, "V latent chunk", size=9.5, color=MUTED)
    f.text(vx + 20, vy + 40 + 4 * (s + 2) + 24, "512 x r_v (c_2)", size=9.5, color=MUTED)
    f.text(vx + 70, vy + 76, "x", size=14, weight="bold")
    f.tiles(vx + 90, vy + 40, 4, 2, s, role=C_PALU)
    f.text(vx + 90, vy + 40 + 2 * (s + 2) + 12, "u_v: r_v x 128", size=9.5, color=MUTED)
    f.text(vx + 90, vy + 40 + 2 * (s + 2) + 24, "(in the c_4 block)", size=9.5, color=MUTED)
    f.arrow(vx + 170, vy + 58, vx + 200, vy + 58)
    f.tiles(vx + 210, vy + 40, 4, 4, s, role=C_K, labels=lambda r, c: "..." if (r == 3 and c == 0) else None, size=7)
    f.text(vx + 210, vy + 40 + 4 * (s + 2) + 12, "V_h: 512 x 128 into c_1,", size=9.5, color=MUTED)
    f.text(vx + 210, vy + 40 + 4 * (s + 2) + 24, "row-major, for the PV matmul", size=9.5, color=MUTED)
    f.text(vx + 12, vy + 160, "one matmul per row tile (inner dim Rvt, 4 output tiles = 1 trip); needed because a", size=9.5, color=MUTED)
    f.text(vx + 12, vy + 172, "group's u_v cannot be folded into wo without making wo 4x wider.", size=9.5, color=MUTED)
    f.text(vx + 12, vy + 186, "Group mode's merge uses unfused primitives (Fig. B5 note); split capped at 4 cores.", size=9.5, color=MUTED)
    f.caption("Fig. C5. Group mode changes the loop nesting and the CB bookkeeping, not the per-head math: head h's chunk and accumulation order are unchanged (bit-identical when no child is merged).")
    return f


def fig_stock_vs_fused() -> Fig:
    f = Fig(940, 480, "Per head-item per chunk at 4k (from the stage 2d fit): where the fused kernel's extra time goes, and the levers that address it")
    f.text(14, 24, "Why the fused op is 1.36x the stock op at 4k, in one chunk", size=15, weight="bold")
    # bars
    bx, by, scale = 200, 70, 9.0  # px per us
    bars = [
        ("stock (r_v 128)", 44.2, [(44.2, C_K, "flash step, near DRAM-bound")]),
        ("fused r_k 32 / r_v 128", 57.2, [(44.2, C_K, "flash step"), (4.4, C_FPU, "AB matmul"), (8.6, C_DST, "round trips, inits, eltwise, SFPU adds")]),
        ("fused r_k 64 / r_v 128", 61.6, [(44.2, C_K, "flash step"), (8.8, C_FPU, "AB matmul (2 inner tiles)"), (8.6, C_DST, "same per-row-tile overhead")]),
    ]
    for i, (name, total, segs) in enumerate(bars):
        y = by + i * 56
        f.text(bx - 10, y + 20, name, size=11, anchor="end", weight="bold")
        x = bx
        for val, role, lab in segs:
            w = val * scale
            f.rect(x, y, w, 30, role, r=2)
            if w > 200:
                f.text(x + w / 2, y + 19, lab, size=9, anchor="middle")
            x += w
        f.text(x + 8, y + 20, f"{total:.1f} us", size=11, mono=True)
    f.legend(bx, by + 3 * 56 - 6, [(C_K, "flash step (same code as stock)"), (C_FPU, "AB matmul: the rank-dependent part"), (C_DST, "per-row-tile overhead: round trips, inits, reconfigs, eltwise, SFPU adds")], step=14)
    by += 40
    f.text(bx, by + 3 * 56 + 4, "us per head-item per chunk of 512 positions, 4k context, batch 32 (stage 2d least-squares fit over 1k / 4k / 8k)", size=10, color=MUTED)
    f.text(bx, by + 3 * 56 + 18, "the AB-matmul share is inferred from the r_k 32 -> 64 difference (doubling the matmul costs 4.4 us); the rest is rank-independent", size=10, color=MUTED)
    f.text(bx, by + 3 * 56 + 32, "on top of this, the fused op carries 60-80 us more fixed cost per layer per call (full-tile Q math, seeding, table read).", size=10, color=MUTED)

    # per row tile
    px, py = 40, 330
    f.rect(px, py, 420, 120, (PLATE, C_DST[1]), r=6)
    f.text(px + 12, py + 20, "per K row tile (16 per chunk)", size=11.5, weight="bold")
    f.lines(px + 12, py + 42, [
        "fused excess over stock:            ~810 ns  (~800-1100 cycles)",
        "of which the AB matmul (rank part):  ~275 ns",
        "the remaining ~535 ns is the 6 round trips, 3 inits,",
        "4 reconfigs, 8 FPU multiplies and 4 SFPU adds",
    ], size=10.5, lh=16, mono=True)

    lx, ly = 480, 330
    f.rect(lx, ly, 430, 120, (PLATE, C_FPU[1]), r=6)
    f.text(lx + 12, ly + 20, "levers (untried; parity needs -27 % at 4k)", size=11.5, weight="bold")
    f.lines(lx + 12, ly + 40, [
        "1. two column tiles per acquisition: 6 -> 4 round trips per row tile",
        "2. hoist matmul/mul/add inits out of the row-tile loop",
        "3. accumulate A*cos + B*sin on the FPU (no SFPU pass, no dst 1)",
        "4. restore 16-row (half-tile) Q math: attacks the fixed part, worst at 1k",
        "5. bigger: rebuild two row tiles per AB matmul so one matmul feeds 8 packs",
    ], size=10, lh=14, color=MUTED)
    f.caption("Fig. C6. The bytes are already gone; what is left is the number of times the three compute threads have to synchronise per chunk.")
    return f


FIGURES = {
    "kernel_a1_tensix_core": fig_tensix_core,
    "kernel_a2_tiles": fig_tile_formats,
    "kernel_a3_circular_buffers": fig_circular_buffers,
    "kernel_a4_dst_pipeline": fig_dst_pipeline,
    "kernel_b1_decode_math": fig_decode_math,
    "kernel_b2_work_layout": fig_work_layout,
    "kernel_b3_core_dataflow": fig_core_dataflow,
    "kernel_b4_chunk_timeline": fig_chunk_timeline,
    "kernel_b5_tree_reduction": fig_tree_reduction,
    "kernel_c1_palu_data": fig_palu_data,
    "kernel_c2_cb_map": fig_palu_cb_map,
    "kernel_c3_build_k": fig_palu_build_k,
    "kernel_c4_phase2": fig_palu_phase2,
    "kernel_c5_group_mode": fig_palu_group_mode,
    "kernel_c6_stock_vs_fused": fig_stock_vs_fused,
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
