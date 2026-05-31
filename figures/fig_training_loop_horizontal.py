#!/usr/bin/env python3
"""
fig_training_loop_horizontal.py
Landscape hybrid layout for PowerPoint.
  - Width >> Height  (≈ 2 : 1 aspect ratio)
  - General flow reads left → right
  - Batch ops and validation stack vertically to save x-space
  - Short, compact boxes throughout

Output: figures/fig_training_loop_horizontal.{png,pdf}
"""

import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ── Palette ────────────────────────────────────────────────────────────────
BG="#FFFFFF"; TEXT="#1F2328"; MUTED="#57606A"; BORDER="#D0D7DE"
BLUE="#0969DA"; GREEN="#1A7F37"; RED="#CF222E"; YELLOW="#9A6700"
PURPLE="#7B45E7"; ORANGE="#BC4C00"
FC_BLUE="#DFF0FF"; FC_GREEN="#DAFBE1"; FC_RED="#FFEBE9"
FC_YELLOW="#FFF8C5"; FC_PURPLE="#FBEFFF"; FC_ORANGE="#FFF1E5"

def rbox(ax, cx, cy, w, h, title, sub="", fc="#F6F8FA", ec=BORDER, elw=1.6,
         tsz=10.5, ssz=8.5, tc=TEXT, sc=MUTED, bold=True):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.014", facecolor=fc, edgecolor=ec,
        linewidth=elw, zorder=3))
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy+h*0.22, title, ha="center", va="center",
            fontsize=tsz, fontweight=fw, color=tc, zorder=4)
        ax.text(cx, cy-h*0.26, sub, ha="center", va="center",
            fontsize=ssz, color=sc, style="italic", zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
            fontsize=tsz, fontweight=fw, color=tc, zorder=4)

def va(ax, x, y0, y1, color=MUTED, lw=1.6):
    ax.annotate("", xy=(x,y1), xytext=(x,y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11), zorder=5)

def ha(ax, x0, y, x1, color=MUTED, lw=1.6):
    ax.annotate("", xy=(x1,y), xytext=(x0,y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11), zorder=5)

def diamond(ax, cx, cy, w, h, title, sub="", ec=ORANGE, fc=FC_ORANGE, tsz=10):
    pts = np.array([[cx,cy+h/2],[cx+w/2,cy],[cx,cy-h/2],[cx-w/2,cy]])
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
        linewidth=2.0, zorder=3))
    if sub:
        ax.text(cx, cy+h*0.15, title, ha="center", va="center",
            fontsize=tsz, color=ec, fontweight="bold", zorder=4)
        ax.text(cx, cy-h*0.18, sub, ha="center", va="center",
            fontsize=tsz, color=ec, fontweight="bold", zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
            fontsize=tsz, color=ec, fontweight="bold", zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# Layout — all measurements in axis-coordinate units
# ══════════════════════════════════════════════════════════════════════════

# Box heights  (keep compact)
BH    = 0.66    # standard  (title only or very short sub)
BHT   = 0.78    # with subtitle
BHV   = 0.90    # validation boxes (need two sub-lines)
OPT_H = 1.00    # optimizer (two-line subtitle)
GAP   = 0.19    # vertical gap between batch boxes

# Column X centres and widths
BATCH_CX, BW   = 4.60, 3.40
VAL_CX,   VW   = 8.60, 3.20

# Row Y positions
CY_ES   = 9.10   # Epoch Start row
CY_DIA  = 4.90   # Decision / Checkpoint / Done row

# Batch column Y (top → bottom)
LOAD_CY  = 7.65
FWD_CY   = LOAD_CY  - BHT/2 - GAP - BHT/2
LOSS_CY  = FWD_CY   - BHT/2 - GAP - BHT/2
BWD_CY   = LOSS_CY  - BHT/2 - GAP - BHT/2
OPT_CY   = BWD_CY   - BHT/2 - GAP - OPT_H/2

OPT_BOT  = OPT_CY   - OPT_H/2
LOAD_TOP = LOAD_CY  + BHT/2

# Validation column Y
VAL_CY   = 6.60
CKP_CY   = CY_DIA

# Frame edges
BATCH_L  = BATCH_CX - BW/2 - 0.28
BATCH_R  = BATCH_CX + BW/2 + 0.28
BATCH_T  = LOAD_TOP + 0.36
BATCH_B  = OPT_BOT  - 0.30

EP_X0    = BATCH_L  - 0.36
EP_X1    = VAL_CX   + VW/2 + 0.28
EP_Y0    = BATCH_B  - 0.44
EP_Y1    = CY_ES    + BH/2  + 0.36

# Routing tracks
LB_X     = BATCH_L  - 0.28   # batch loop-back, left of batch frame
EP_LB_X  = EP_X0    - 0.18   # epoch loop-back, left of epoch frame
ROUTE_X  = BATCH_R  + 0.26   # vertical run connecting batch → validation

# Diamond and Done
DIA_W, DIA_H = 1.80, 1.25
DIA_CX   = EP_X1 + 0.20 + DIA_W/2
DIA_BOT  = CY_DIA - DIA_H/2
DONE_W   = 1.70
DONE_H   = BHV
DONE_CX  = DIA_CX
DONE_CY  = VAL_CY

# Init
INIT_W   = 1.65
INIT_CX  = EP_X0 - 0.52 - INIT_W/2

XL  = DONE_CX + DONE_W/2 + 0.55
EL_Y = EP_Y0 - 0.26

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9), facecolor=BG)

ax = fig.add_axes([0.03, 0.03, 0.95, 0.95])
ax.set_xlim(0, XL)
ax.set_ylim(EL_Y - 0.45, EP_Y1 + 0.28)
ax.axis("off"); ax.set_facecolor(BG)

# ── Epoch frame ───────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((EP_X0, EP_Y0), EP_X1-EP_X0, EP_Y1-EP_Y0,
    boxstyle="round,pad=0.06", facecolor="#F2FCF2", edgecolor=GREEN,
    linewidth=1.8, zorder=1, alpha=0.65))
ax.text(EP_X0+0.10, EP_Y1+0.05,
    "for epoch in 1 … max_epochs = 30:",
    fontsize=9, color=GREEN, va="bottom", style="italic", fontweight="bold")

# ── Batch frame ───────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((BATCH_L, BATCH_B), BATCH_R-BATCH_L, BATCH_T-BATCH_B,
    boxstyle="round,pad=0.04", facecolor="#F0F4FF", edgecolor=BLUE,
    linewidth=1.5, zorder=1, alpha=0.65))
ax.text(BATCH_L+0.08, BATCH_T+0.14,
    "for batch in DataLoader (shuffle=True):",
    fontsize=8, color=BLUE, va="bottom", style="italic", fontweight="bold")
ax.text(BATCH_R-0.08, BATCH_T+0.14,
    "train  |  B = 32",
    fontsize=8, color=BLUE, va="bottom", style="italic", fontweight="bold", ha="right")

# ══════════════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, INIT_CX, CY_ES, INIT_W, 1.20, "Initialise",
    sub="model, AdamW\nLR sched, wᵢ∝1/freqᵢ",
    fc=FC_BLUE, ec=BLUE, elw=2.0, tsz=11.5, ssz=9)
ha(ax, INIT_CX+INIT_W/2, CY_ES, EP_X0+0.10, BLUE, 1.7)

# ══════════════════════════════════════════════════════════════════════════
# EPOCH START  (wide banner across both sub-columns)
# ══════════════════════════════════════════════════════════════════════════
ES_W = EP_X1 - EP_X0 - 0.22
ES_CX = (EP_X0 + EP_X1) / 2
rbox(ax, ES_CX, CY_ES, ES_W, BH,
    "model.train()  |  LR warm-up: linear ramp for first 5 epochs",
    fc=FC_GREEN, ec=GREEN, elw=1.8, tsz=11, ssz=9.5)
va(ax, BATCH_CX, CY_ES-BH/2, BATCH_T-0.04, GREEN, 1.5)

# ══════════════════════════════════════════════════════════════════════════
# BATCH COLUMN  — five compact stacked boxes
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, BATCH_CX, LOAD_CY, BW, BHT, "Load Mini-Batch",
    sub="(B=32, 3, 224, 224)  |  augmented  |  to GPU",
    fc=FC_BLUE, ec=BLUE, elw=1.6, tsz=10.5, ssz=8.5)
va(ax, BATCH_CX, LOAD_CY-BHT/2, FWD_CY+BHT/2, BLUE, 1.5)

rbox(ax, BATCH_CX, FWD_CY, BW, BHT, "Forward Pass",
    sub="logits = model( x )     (B, 4)  raw logits",
    fc=FC_GREEN, ec=GREEN, elw=1.9, tsz=11, ssz=9, tc=GREEN)
va(ax, BATCH_CX, FWD_CY-BHT/2, LOSS_CY+BHT/2, GREEN, 1.5)

rbox(ax, BATCH_CX, LOSS_CY, BW, BHT, "Class-Weighted CE Loss",
    sub="ℒ = −∑ᵢ wᵢ · yᵢ · log p̂ᵢ     wᵢ ∝ 1/freqᵢ",
    fc=FC_RED, ec=RED, elw=1.6, tsz=10.5, ssz=8.5)
va(ax, BATCH_CX, LOSS_CY-BHT/2, BWD_CY+BHT/2, RED, 1.5)

rbox(ax, BATCH_CX, BWD_CY, BW, BHT, "Backward Pass  (autograd)",
    sub="zero_grad()  |  ℒ.backward()  →  ∂ℒ/∂θ",
    fc=FC_RED, ec=RED, elw=1.6, tsz=10.5, ssz=8.5)
va(ax, BATCH_CX, BWD_CY-BHT/2, OPT_CY+OPT_H/2, RED, 1.5)

rbox(ax, BATCH_CX, OPT_CY, BW, OPT_H, "Clip  |  AdamW  |  LR Scheduler",
    sub="max=1.0  |  θ←θ−lr·m̂/(√v̂+ε)−λθ\nCosineAnnealingLR",
    fc=FC_ORANGE, ec=ORANGE, elw=1.6, tsz=10.5, ssz=8.5)

# ── Batch loop-back (U-shape: down from Clip+Adam → under batch column → up → into Load Batch top)
UB_LOOP_X  = BATCH_L - 0.14          # left track: outside blue frame, inside green frame
UB_UNDER_Y = BATCH_B - 0.18          # bottom horizontal: below blue frame, inside green frame
UB_TOP_Y   = LOAD_TOP + 0.15         # top horizontal: above Load Batch, inside blue frame
UB_ENTRY_X = BATCH_CX - BW * 0.25   # x where arrow descends into Load Batch top
ax.plot([BATCH_CX, BATCH_CX], [OPT_BOT, UB_UNDER_Y],        # down from Clip+Adam
        color=BLUE, lw=1.3, alpha=0.85, zorder=2)
ax.plot([BATCH_CX, UB_LOOP_X], [UB_UNDER_Y, UB_UNDER_Y],    # left under batch column
        color=BLUE, lw=1.3, alpha=0.85, zorder=2)
ax.plot([UB_LOOP_X, UB_LOOP_X], [UB_UNDER_Y, UB_TOP_Y],     # up alongside batch column
        color=BLUE, lw=1.3, alpha=0.85, zorder=2)
ax.plot([UB_LOOP_X, UB_ENTRY_X], [UB_TOP_Y, UB_TOP_Y],      # right above Load Batch
        color=BLUE, lw=1.3, alpha=0.85, zorder=2)
ax.annotate("", xy=(UB_ENTRY_X, LOAD_TOP), xytext=(UB_ENTRY_X, UB_TOP_Y),
    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3, mutation_scale=10), zorder=5)
ax.text((BATCH_CX + UB_LOOP_X) / 2, UB_UNDER_Y - 0.13,
    "next batch", ha="center", va="top",
    fontsize=8.5, color=BLUE, style="italic")

# ── Route: batch bottom → validation top ───────────────────────────────────
ax.plot([BATCH_CX+BW/2, ROUTE_X], [OPT_CY, OPT_CY],
        color=MUTED, lw=1.2, alpha=0.85, zorder=2)
ax.plot([ROUTE_X, ROUTE_X], [OPT_CY, VAL_CY],
        color=MUTED, lw=1.2, alpha=0.85, zorder=2)
ha(ax, ROUTE_X, VAL_CY, VAL_CX-VW/2-0.04, MUTED, 1.3)

# ══════════════════════════════════════════════════════════════════════════
# VALIDATION COLUMN  — two compact boxes
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, VAL_CX, VAL_CY, VW, BHV, "Validate",
    sub="model.eval() + no_grad\nforward val set  |  per-class AUROC",
    fc=FC_PURPLE, ec=PURPLE, elw=1.6, tsz=10.5, ssz=8.5)
va(ax, VAL_CX, VAL_CY-BHV/2, CKP_CY+BHT/2, PURPLE, 1.5)

rbox(ax, VAL_CX, CKP_CY, VW, BHT, "Checkpoint  |  Patience",
    sub="save if best  |  else patience += 1",
    fc=FC_YELLOW, ec=YELLOW, elw=1.6, tsz=10.5, ssz=8.5)
ha(ax, VAL_CX+VW/2, CKP_CY, DIA_CX-DIA_W/2, YELLOW, 1.5)

# ══════════════════════════════════════════════════════════════════════════
# DECISION DIAMOND
# ══════════════════════════════════════════════════════════════════════════
diamond(ax, DIA_CX, CY_DIA, DIA_W, DIA_H,
    "patience > 8?",
    sub="or epoch ≥ 30",
    ec=ORANGE, fc=FC_ORANGE, tsz=10)

# Yes → Training Complete (upward arrow from diamond top)
va(ax, DIA_CX, CY_DIA+DIA_H/2, DONE_CY-DONE_H/2, GREEN, 1.7)
ax.text(DIA_CX + 0.12, (CY_DIA + DIA_H/2 + DONE_CY - DONE_H/2) / 2,
    "Yes",
    fontsize=9, color=GREEN, fontweight="bold", ha="left", va="center")

# No → epoch loop-back
ax.plot([DIA_CX, DIA_CX], [DIA_BOT, EL_Y],
        color=GREEN, lw=1.3, alpha=0.85, zorder=2)
ax.plot([DIA_CX, EP_LB_X], [EL_Y, EL_Y],
        color=GREEN, lw=1.3, alpha=0.85, zorder=2)
ax.plot([EP_LB_X, EP_LB_X], [EL_Y, CY_ES],
        color=GREEN, lw=1.3, alpha=0.85, zorder=2)
ax.annotate("", xy=(EP_X0+0.12, CY_ES), xytext=(EP_LB_X, CY_ES),
    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3, mutation_scale=10), zorder=5)
ax.text(DIA_CX + 0.12, DIA_BOT - 0.12,
    "No  →  next epoch",
    ha="left", va="top", fontsize=9, color=GREEN, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════
# TRAINING COMPLETE
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, DONE_CX, DONE_CY, DONE_W, DONE_H, "Training\nComplete",
    sub="model.eval()\nready for test",
    fc=FC_PURPLE, ec=PURPLE, elw=2.0, tsz=12, ssz=9.5, tc=PURPLE)


fig.savefig("figures/fig_training_loop_horizontal.png", dpi=200,
    bbox_inches="tight", facecolor=BG)
fig.savefig("figures/fig_training_loop_horizontal.pdf",
    bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved → figures/fig_training_loop_horizontal.{png,pdf}")
