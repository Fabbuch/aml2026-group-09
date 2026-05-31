"""
visualize_pope_forward.py
Light-background, large-text forward-pass diagram for PoPEViT.
Suitable for presentations and papers.

Usage:  python visualize_pope_forward.py
Output: pope_forward_pass.png
Requires: matplotlib, numpy, Pillow
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, FancyArrowPatch
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
MRI_PATH   = "Brain-Tumor-Classification-DataSet/test/glioma_tumor/083_gg (32).jpg"
IMG_SIZE   = 224
PATCH_SIZE = 16
N_GRID     = IMG_SIZE // PATCH_SIZE   # 14
PR, PC     = 4, 7                     # highlighted patch (row, col)

# ── Light palette ──────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
PANEL   = "#F6F8FA"
BORDER  = "#D0D7DE"
TEXT    = "#1F2328"
MUTED   = "#57606A"
BLUE    = "#0969DA"
GREEN   = "#1A7F37"
RED     = "#CF222E"
YELLOW  = "#9A6700"
PURPLE  = "#7B45E7"
ORANGE  = "#BC4C00"
TEAL    = "#1B7C83"
LGRAY   = "#D0D7DE"

# panel fill colours (tinted)
FC_BLUE   = "#DFF0FF"
FC_GREEN  = "#DAFBE1"
FC_RED    = "#FFEBE9"
FC_YELLOW = "#FFF8C5"
FC_PURPLE = "#FBEFFF"
FC_ORANGE = "#FFF1E5"
FC_GRAY   = "#F6F8FA"
FC_MRI    = "#0D1117"   # dark only for the MRI image panel

plt.rcParams.update({"font.family": "DejaVu Sans"})


# ── Utilities ──────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, title, sub="", fc=PANEL, ec=BORDER, elw=1.4,
         tsz=19, ssz=15, tc=TEXT, sc=MUTED, bold=True, pad=0.012):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad}",
        linewidth=elw, edgecolor=ec, facecolor=fc, zorder=3))
    if sub:
        ax.text(x + w/2, y + h*0.63, title, ha="center", va="center",
                fontsize=tsz, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)
        ax.text(x + w/2, y + h*0.24, sub, ha="center", va="center",
                fontsize=ssz, color=sc, zorder=4, linespacing=1.35)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=tsz, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)


def varrow(ax, x, y_top, y_bot, color=MUTED, lw=1.8):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=5)


def section_chip(fig, xc, text, color=BLUE):
    fig.text(xc, 0.052, text, ha="center", va="center", fontsize=18,
             fontweight="bold", color=color,
             bbox=dict(boxstyle="round,pad=0.40", facecolor=BG,
                       edgecolor=color, linewidth=1.5, alpha=1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(32, 14), facecolor=BG)


AX_Y0, AX_H = 0.09, 0.83


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MRI INPUT
# ═════════════════════════════════════════════════════════════════════════════
ax_mri = fig.add_axes((0.025, AX_Y0, 0.145, AX_H))
ax_mri.set_facecolor(FC_MRI)

try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = Image.LANCZOS

mri_raw = Image.open(MRI_PATH).convert("L").resize((IMG_SIZE, IMG_SIZE), resample)
mri_np  = np.array(mri_raw, dtype=float)

ax_mri.imshow(mri_np, cmap="gray", aspect="equal",
              extent=[0, IMG_SIZE, 0, IMG_SIZE], vmin=0, vmax=255)

for i in range(N_GRID + 1):
    v = i * PATCH_SIZE
    ax_mri.axhline(v, color="#58A6FF", lw=0.45, alpha=0.40)
    ax_mri.axvline(v, color="#58A6FF", lw=0.45, alpha=0.40)

px0 = PC * PATCH_SIZE
py0 = (N_GRID - 1 - PR) * PATCH_SIZE
ax_mri.add_patch(mpatches.Rectangle(
    (px0, py0), PATCH_SIZE, PATCH_SIZE,
    lw=2.5, edgecolor="#E3B341", facecolor="#E3B341", alpha=0.30, zorder=4))
ax_mri.add_patch(mpatches.Rectangle(
    (px0, py0), PATCH_SIZE, PATCH_SIZE,
    lw=2.5, edgecolor="#E3B341", facecolor="none", zorder=5))

ax_mri.set_xlim(0, IMG_SIZE); ax_mri.set_ylim(0, IMG_SIZE)
ax_mri.set_xticks([]); ax_mri.set_yticks([])
for sp in ax_mri.spines.values():
    sp.set_edgecolor(BORDER); sp.set_linewidth(1.5)
ax_mri.set_title(f"Input MRI  {IMG_SIZE}×{IMG_SIZE}",
                 color=TEXT, fontsize=19, fontweight="bold", pad=8)

section_chip(fig, 0.097, "1. Input")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PATCH ZOOM
# ═════════════════════════════════════════════════════════════════════════════
ax_zoom = fig.add_axes((0.183, AX_Y0 + 0.20, 0.112, AX_H - 0.20))
ax_zoom.set_facecolor(FC_MRI)

patch_arr = mri_np[PR * PATCH_SIZE:(PR + 1) * PATCH_SIZE,
                   PC * PATCH_SIZE:(PC + 1) * PATCH_SIZE]

ax_zoom.imshow(patch_arr, cmap="gray", aspect="equal",
               extent=[0, PATCH_SIZE, 0, PATCH_SIZE], vmin=0, vmax=255)

for i in range(PATCH_SIZE + 1):
    ax_zoom.axhline(i, color="#2ACFCF", lw=0.65, alpha=0.70)
    ax_zoom.axvline(i, color="#2ACFCF", lw=0.65, alpha=0.70)

for ri in range(PATCH_SIZE):
    for ci in range(PATCH_SIZE):
        val = int(patch_arr[PATCH_SIZE - 1 - ri, ci])
        col = FC_MRI if val > 160 else "#E6EDF3"
        ax_zoom.text(ci + 0.5, ri + 0.5, str(val),
                     ha="center", va="center", fontsize=7.0,
                     color=col, fontweight="bold")

ax_zoom.set_xlim(0, PATCH_SIZE); ax_zoom.set_ylim(0, PATCH_SIZE)
ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
for sp in ax_zoom.spines.values():
    sp.set_edgecolor("#E3B341"); sp.set_linewidth(2.5)
ax_zoom.set_title(f"Patch  [{PATCH_SIZE}×{PATCH_SIZE} px]",
                  color=YELLOW, fontsize=18, fontweight="bold", pad=7)

for frac_y, az_y in [(1.0, 1.0), (0.0, 0.0)]:
    fig.add_artist(ConnectionPatch(
        xyA=(px0 + PATCH_SIZE, py0 + frac_y * PATCH_SIZE),
        coordsA="data", axesA=ax_mri,
        xyB=(0, az_y), coordsB="axes fraction", axesB=ax_zoom,
        color="#E3B341", lw=1.3, linestyle="--", alpha=0.75, zorder=10))

fig.text(0.239, AX_Y0 + 0.375, "pixel values  (0–255)",
         ha="center", va="top", fontsize=16, color=MUTED)

section_chip(fig, 0.239, "2. Patch")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FLATTEN + LINEAR PROJECTION
# ═════════════════════════════════════════════════════════════════════════════
ax_flat = fig.add_axes((0.307, AX_Y0, 0.095, AX_H))
ax_flat.set_xlim(0, 1); ax_flat.set_ylim(0, 1)
ax_flat.axis("off")

# horizontal pixel vector strip
N_SHOW  = 80
VBAR_W  = 0.76 / N_SHOW
VBAR_H  = 0.058
VY      = 0.895
VX0     = 0.12

ax_flat.text(0.50, VY + 0.050, "flatten  16×16×3",
             ha="center", va="bottom", fontsize=17, color=MUTED, fontstyle="italic")

flat_vals = patch_arr.flatten()
idxs      = np.linspace(0, len(flat_vals) - 1, N_SHOW, dtype=int)
gray_cmap = plt.colormaps["gray"]
for k, idx in enumerate(idxs):
    c = gray_cmap(flat_vals[idx] / 255.0)
    ax_flat.add_patch(FancyBboxPatch(
        (VX0 + k * VBAR_W, VY - VBAR_H), VBAR_W * 0.91, VBAR_H,
        boxstyle="square,pad=0", facecolor=c, edgecolor="none", zorder=3))

ax_flat.annotate("",
    xy=(VX0 + N_SHOW * VBAR_W, VY - VBAR_H - 0.016),
    xytext=(VX0, VY - VBAR_H - 0.016),
    arrowprops=dict(arrowstyle="|-|,widthA=0.12,widthB=0.12", color=TEAL, lw=1.3))
ax_flat.text(0.50, VY - VBAR_H - 0.034, "768 values",
             ha="center", va="top", fontsize=17, color=TEAL, fontweight="bold")

varrow(ax_flat, 0.50, VY - VBAR_H - 0.068, 0.690, color=BLUE, lw=1.8)

# weight matrix mosaic
np.random.seed(7)
W_ROWS, W_COLS = 16, 28
W_H  = 0.195
W_Y0 = 0.480
cw   = 0.76 / W_COLS
ch   = W_H / W_ROWS
W    = np.random.randn(W_ROWS, W_COLS)
Wn   = (W - W.min()) / (W.max() - W.min())
wcm  = plt.colormaps["RdBu_r"]
for ri in range(W_ROWS):
    for ci in range(W_COLS):
        ax_flat.add_patch(FancyBboxPatch(
            (VX0 + ci * cw, W_Y0 + ri * ch), cw * 0.88, ch * 0.88,
            boxstyle="square,pad=0", facecolor=wcm(Wn[ri, ci]),
            edgecolor="none", zorder=3))
ax_flat.add_patch(FancyBboxPatch(
    (VX0 - 0.012, W_Y0 - 0.006), 0.76 + 0.024, W_H + 0.012,
    boxstyle="round,pad=0.006", facecolor="none",
    edgecolor=BLUE, lw=2.0, zorder=4))

ax_flat.text(0.50, W_Y0 - 0.028, "Linear  768 → D=512",
             ha="center", va="top", fontsize=17,
             fontweight="bold", color=BLUE, zorder=5)

varrow(ax_flat, 0.50, W_Y0 - 0.068, 0.315, color=BLUE, lw=1.8)

# output token strip
OUT_H  = 0.058
OUT_Y0 = 0.255
D_SHOW = 80
dw     = 0.76 / D_SHOW
ecm    = plt.colormaps["plasma"]
for k in range(D_SHOW):
    c = ecm(k / D_SHOW * 0.7 + 0.1)
    ax_flat.add_patch(FancyBboxPatch(
        (VX0 + k * dw, OUT_Y0), dw * 0.91, OUT_H,
        boxstyle="square,pad=0", facecolor=c, edgecolor="none", zorder=3))
ax_flat.annotate("",
    xy=(VX0 + D_SHOW * dw, OUT_Y0 - 0.016),
    xytext=(VX0, OUT_Y0 - 0.016),
    arrowprops=dict(arrowstyle="|-|,widthA=0.12,widthB=0.12", color=PURPLE, lw=1.3))
ax_flat.text(0.50, OUT_Y0 - 0.034, "D=512 patch token",
             ha="center", va="top", fontsize=17, color=PURPLE, fontweight="bold")

ax_flat.text(0.50, 0.975, "Flatten + Embed",
             ha="center", va="center", fontsize=19, fontweight="bold", color=TEXT)

section_chip(fig, 0.354, "3. Embed")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TOKEN SEQUENCE
# ═════════════════════════════════════════════════════════════════════════════
ax_tok = fig.add_axes((0.415, AX_Y0, 0.135, AX_H))
ax_tok.set_xlim(0, 1); ax_tok.set_ylim(0, 1)
ax_tok.axis("off")

TOK_TOP = 0.920
TOK_BOT = 0.235
CLS_H   = 0.062
N_TOK   = 196
tok_area = TOK_TOP - CLS_H - 0.012 - TOK_BOT
tok_h    = tok_area / N_TOK
tok_x0, tok_w = 0.22, 0.44

ax_tok.add_patch(FancyBboxPatch(
    (tok_x0, TOK_TOP - CLS_H), tok_w, CLS_H,
    boxstyle="round,pad=0.010", facecolor=FC_PURPLE,
    edgecolor=PURPLE, lw=2.2, zorder=3))
ax_tok.text(tok_x0 + tok_w/2, TOK_TOP - CLS_H/2,
            "[CLS]", ha="center", va="center",
            fontsize=18, fontweight="bold", color=PURPLE, zorder=4)

np.random.seed(0)
tcm = plt.colormaps["plasma"]
for i in range(N_TOK):
    c = tcm(0.15 + 0.7 * (i / N_TOK))
    y_i = TOK_TOP - CLS_H - 0.010 - (i + 1) * tok_h
    ax_tok.add_patch(FancyBboxPatch(
        (tok_x0, y_i), tok_w, tok_h * 0.87,
        boxstyle="square,pad=0", facecolor=c, edgecolor="none", zorder=3))

hi_i = PR * N_GRID + PC
hi_y = TOK_TOP - CLS_H - 0.010 - (hi_i + 1) * tok_h
ax_tok.add_patch(FancyBboxPatch(
    (tok_x0 - 0.03, hi_y), tok_w + 0.06, tok_h * 1.6,
    boxstyle="square,pad=0", facecolor="none",
    edgecolor=YELLOW, lw=2.2, zorder=5))
ax_tok.text(tok_x0 - 0.05, hi_y + tok_h * 0.80,
            "p", ha="right", va="center",
            fontsize=17, color=YELLOW, fontweight="bold")

bx = tok_x0 + tok_w + 0.04
ax_tok.annotate("",
    xy=(bx, TOK_BOT + 0.010),
    xytext=(bx, TOK_TOP - CLS_H - 0.010),
    arrowprops=dict(arrowstyle="|-|,widthA=0.12,widthB=0.12",
                    color=ORANGE, lw=1.3))
ax_tok.text(bx + 0.07, (TOK_TOP + TOK_BOT)/2 - 0.03,
            "196\npatch\ntokens", ha="left", va="center",
            fontsize=16, color=ORANGE)

POS_Y0 = 0.127
rbox(ax_tok, 0.03, POS_Y0, 0.90, 0.082,
     "+  Positional Embedding",
     "learned  ·  197 × D",
     fc=FC_GREEN, ec=GREEN, elw=2.0, tsz=18, ssz=15, tc=GREEN)
varrow(ax_tok, 0.50, TOK_BOT, POS_Y0 + 0.082 + 0.005, color=GREEN, lw=1.8)

rbox(ax_tok, 0.03, POS_Y0 - 0.095, 0.90, 0.070,
     "Dropout", "training only  ·  p=0.1",
     fc=FC_GRAY, ec=BORDER, tsz=17, ssz=14)

ax_tok.text(0.50, 0.970, "Token Sequence",
            ha="center", va="center", fontsize=19, fontweight="bold", color=TEXT)
ax_tok.text(0.50, 0.943, "B × 197 × D",
            ha="center", va="center", fontsize=16, color=MUTED)

section_chip(fig, 0.482, "4. Tokens")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PoPE TRANSFORMER BLOCK  ×6
# ═════════════════════════════════════════════════════════════════════════════
ax_blk = fig.add_axes((0.562, AX_Y0, 0.215, AX_H))
ax_blk.set_xlim(0, 1); ax_blk.set_ylim(0, 1)
ax_blk.axis("off")

CX, BW = 0.54, 0.80

ITEMS = [
    # (cy,   h,    title,                  sub,                              fc,       ec,    tc)
    # cy values computed so every inter-box gap = 0.040 exactly
    (0.930, 0.040, "x  (B × 197 × D)",     "",                               FC_GRAY,  BORDER, TEXT),
    (0.847, 0.046, "Layer Norm",            "normalise feature dim D",        FC_BLUE,  BLUE,   TEXT),
    (0.755, 0.058, "PoPEAttention",         "polar pos. encoding on Q and K", FC_RED,   RED,    RED),
    (0.663, 0.046, "Residual",          "x  =  x  +  attn( LN(x) )",     FC_GREEN, GREEN,  GREEN),
    (0.577, 0.046, "Layer Norm",            "normalise feature dim D",        FC_BLUE,  BLUE,   TEXT),
    (0.484, 0.060, "Feed-Forward",          "Linear→GELU→Drop→Linear→Drop",   FC_YELLOW,YELLOW, TEXT),
    (0.391, 0.046, "Residual",          "x  =  x  +  FFN( LN(x) )",      FC_GREEN, GREEN,  GREEN),
    (0.308, 0.040, "x  (B × 197 × D)",     "",                               FC_GRAY,  BORDER, TEXT),
]

for (cy, h, title, sub, fc, ec, tc) in ITEMS:
    rbox(ax_blk, CX - BW/2, cy - h/2, BW, h, title, sub,
         fc=fc, ec=ec, elw=2.0 if ec != BORDER else 1.3,
         tsz=19, ssz=15, tc=tc)

for i in range(len(ITEMS) - 1):
    cy_a, h_a = ITEMS[i][0],   ITEMS[i][1]
    cy_b, h_b = ITEMS[i+1][0], ITEMS[i+1][1]
    varrow(ax_blk, CX, cy_a - h_a/2 - 0.007, cy_b + h_b/2 + 0.007, color=MUTED, lw=1.8)

SX = CX - BW/2 - 0.085
for (src, dst) in [(0.930, 0.663), (0.663, 0.391)]:
    ax_blk.plot([SX, SX], [src, dst], color=MUTED, lw=1.5, alpha=0.7, zorder=2)
    ax_blk.plot([SX, CX - BW/2], [src, src], color=MUTED, lw=1.5, alpha=0.7, zorder=2)
    ax_blk.plot([SX, CX - BW/2], [dst, dst], color=MUTED, lw=1.5, alpha=0.7, zorder=2)
    ax_blk.annotate("", xy=(CX - BW/2, dst), xytext=(SX - 0.005, dst),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                    lw=1.5, mutation_scale=12), zorder=3)
    ax_blk.text(SX - 0.040, (src + dst)/2, "skip",
                ha="center", va="center", fontsize=15,
                color=MUTED, rotation=90)

ax_blk.add_patch(FancyBboxPatch(
    (CX - BW/2 - 0.095, ITEMS[-1][0] - ITEMS[-1][1]/2 - 0.020),
    BW + 0.120, ITEMS[0][0] - ITEMS[-1][0] + (ITEMS[0][1] + ITEMS[-1][1])/2 + 0.050,
    boxstyle="round,pad=0.01", linewidth=2.0,
    edgecolor=PURPLE, facecolor="none", linestyle="--", alpha=0.60, zorder=1))
ax_blk.text(CX + BW/2 + 0.095, (ITEMS[0][0] + ITEMS[-1][0])/2,
            "×6", ha="center", va="center",
            fontsize=34, color=PURPLE, fontweight="bold", fontstyle="italic")

# ── PoPE polar diagram — standalone axes below the purple repeat box ──────────
ax_polar = fig.add_axes((0.589, 0.075, 0.161, 0.207))
ax_polar.set_facecolor("#FFF5F5")
ax_polar.set_aspect("equal")
ax_polar.set_xlim(-1.65, 1.65); ax_polar.set_ylim(-1.80, 1.80)
ax_polar.set_xticks([]); ax_polar.set_yticks([])
for sp in ax_polar.spines.values():
    sp.set_edgecolor(RED); sp.set_linewidth(2.0)

theta_c = np.linspace(0, 2*np.pi, 300)
ax_polar.plot(np.cos(theta_c), np.sin(theta_c), color=LGRAY, lw=1.2)
ax_polar.axhline(0, color=LGRAY, lw=0.7); ax_polar.axvline(0, color=LGRAY, lw=0.7)

for lbl, (theta, col) in [("pos₁", (np.pi/5, GREEN)), ("pos₂", (1.9, ORANGE))]:
    ax_polar.annotate("", xy=(np.cos(theta), np.sin(theta)), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=col,
                                      lw=2.2, mutation_scale=12))
    ax_polar.text(1.28*np.cos(theta), 1.28*np.sin(theta), lbl,
                  ha="center", va="center", fontsize=16, color=col, fontweight="bold")

arc = np.linspace(np.pi/5, 1.9, 60)
ax_polar.plot(0.55*np.cos(arc), 0.55*np.sin(arc), color=RED, lw=2.0)
ax_polar.text(0.72*np.cos(1.05), 0.72*np.sin(1.05), "Δθ",
              ha="center", va="center", fontsize=17, color=RED, fontweight="bold")
ax_polar.text(0, -1.48, r"$\tilde{q}=|q|\cdot e^{i\theta_{pos}}$",
              ha="center", va="center", fontsize=17, color=RED)
ax_polar.text(0, 1.62, "PoPE Polar Encoding",
              ha="center", va="center", fontsize=16,
              color=RED, fontweight="bold")

ax_blk.text(CX, 0.975, "PoPE Transformer Block",
            ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT)

section_chip(fig, 0.669, "5. Transformer  ×6")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  CLASSIFICATION HEAD + OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
ax_out = fig.add_axes((0.789, AX_Y0, 0.195, AX_H))
ax_out.set_xlim(0, 1); ax_out.set_ylim(0, 1)
ax_out.axis("off")

HEAD = [
    (0.895, 0.065, "Layer Norm",           "B × 197 × D",          FC_BLUE,   BLUE),
    (0.797, 0.065, "Mean Pool over tokens", "B × 197×D  →  B × D", FC_PURPLE, PURPLE),
    (0.694, 0.065, "MLP Head",             "Linear  D → 4",        FC_ORANGE, ORANGE),
]
for (cy, h, title, sub, fc, ec) in HEAD:
    rbox(ax_out, 0.04, cy - h/2, 0.92, h, title, sub,
         fc=fc, ec=ec, elw=2.0, tsz=19, ssz=15)

varrow(ax_out, 0.50, HEAD[0][0] - HEAD[0][1]/2 - 0.007,
       HEAD[1][0] + HEAD[1][1]/2 + 0.005, color=MUTED)
varrow(ax_out, 0.50, HEAD[1][0] - HEAD[1][1]/2 - 0.007,
       HEAD[2][0] + HEAD[2][1]/2 + 0.005, color=MUTED)

varrow(ax_out, 0.50, HEAD[2][0] - HEAD[2][1]/2 - 0.007, 0.598, color=MUTED)
ax_out.text(0.50, 0.592, "softmax", ha="center", va="top",
            fontsize=17, color=MUTED, fontstyle="italic")

# bar chart
classes = ["No\nTumour", "Menin-\ngioma", "Glioma", "Pitui-\ntary"]
probs   = [0.04, 0.07, 0.84, 0.05]
cols    = [LGRAY, LGRAY, RED, LGRAY]
fcs     = [FC_GRAY, FC_GRAY, FC_RED, FC_GRAY]

BAR_BOT = 0.095
MAX_H   = 0.46
bw      = 0.165
gap     = 0.050
x0      = 0.04 + (0.92 - 4*(bw + gap) + gap) / 2

for i, (cls, prob, col, fc) in enumerate(zip(classes, probs, cols, fcs)):
    bx  = x0 + i * (bw + gap)
    bhi = prob * MAX_H
    ax_out.add_patch(FancyBboxPatch(
        (bx, BAR_BOT), bw, bhi,
        boxstyle="round,pad=0.007", facecolor=fc,
        edgecolor=col, linewidth=1.8, zorder=3))
    ax_out.text(bx + bw/2, BAR_BOT + bhi + 0.020,
                f"{prob:.0%}", ha="center", va="bottom",
                fontsize=17, color=col,
                fontweight="bold" if col == RED else "normal")
    ax_out.text(bx + bw/2, BAR_BOT - 0.018,
                cls, ha="center", va="top", fontsize=15, color=MUTED)

ax_out.axhline(BAR_BOT, xmin=0.03, xmax=0.97, color=BORDER, lw=1.0, zorder=2)
ax_out.text(0.50, 0.560, "Predicted Class Probabilities",
            ha="center", va="top", fontsize=17, color=MUTED)

ax_out.text(0.50, 0.975, "Classification Head",
            ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT)

section_chip(fig, 0.886, "6. Output")


# ═════════════════════════════════════════════════════════════════════════════
# INTER-SECTION ARROWS
# ═════════════════════════════════════════════════════════════════════════════
CY = 0.505
for x1, x2 in [(0.172, 0.182), (0.297, 0.306),
               (0.404, 0.414), (0.552, 0.561), (0.779, 0.788)]:
    fig.add_artist(FancyArrowPatch(
        (x1, CY), (x2, CY), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=18,
        color=BLUE, lw=2.5, zorder=10))



plt.savefig("figures/pope_forward_pass.png", dpi=170, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Saved: figures/pope_forward_pass.png")
plt.show()
