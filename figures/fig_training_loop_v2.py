#!/usr/bin/env python3
"""
fig_training_loop_v2.py
Clean academic training-loop flowchart for PoPEViT.

Layout:   portrait, single vertical column
Visual:   nested coloured frames for epoch / batch loops
          decision diamond for early stopping
          loop-back arrows on the sides
Style:    matches visualize_pope_forward.py

Output:   figures/fig_training_loop_v2.{png,pdf}
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ── Palette (mirrors visualize_pope_forward.py) ───────────────────────────
BG     = "#FFFFFF"; TEXT   = "#1F2328"; MUTED  = "#57606A"; BORDER = "#D0D7DE"
BLUE   = "#0969DA"; GREEN  = "#1A7F37"; RED    = "#CF222E"; YELLOW = "#9A6700"
PURPLE = "#7B45E7"; ORANGE = "#BC4C00"
FC_BLUE   = "#DFF0FF"; FC_GREEN  = "#DAFBE1"; FC_RED    = "#FFEBE9"
FC_YELLOW = "#FFF8C5"; FC_PURPLE = "#FBEFFF"; FC_ORANGE = "#FFF1E5"
FC_GRAY   = "#F6F8FA"

# ── Primitives ────────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, title, sub="",
         fc=FC_GRAY, ec=BORDER, elw=1.8,
         tsz=13, ssz=10.5, tc=TEXT, sc=MUTED, bold=True):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.018",
        facecolor=fc, edgecolor=ec, linewidth=elw, zorder=3))
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy + h * 0.20, title,
                ha="center", va="center", fontsize=tsz,
                fontweight=fw, color=tc, zorder=4)
        ax.text(cx, cy - h * 0.22, sub,
                ha="center", va="center", fontsize=ssz,
                color=sc, style="italic", zorder=4)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center", fontsize=tsz,
                fontweight=fw, color=tc, zorder=4)


def va(ax, x, y0, y1, color=MUTED, lw=1.8):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=5)


def diamond(ax, cx, cy, w, h, title, sub="",
            ec=ORANGE, fc=FC_ORANGE, tsz=12):
    pts = np.array([
        [cx,       cy + h/2],
        [cx + w/2, cy      ],
        [cx,       cy - h/2],
        [cx - w/2, cy      ],
    ])
    ax.add_patch(Polygon(pts, closed=True,
                         facecolor=fc, edgecolor=ec,
                         linewidth=2.2, zorder=3))
    if sub:
        ax.text(cx, cy + h * 0.14, title,
                ha="center", va="center", fontsize=tsz,
                color=ec, fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.24, sub,
                ha="center", va="center", fontsize=tsz - 1.5,
                color=MUTED, style="italic", zorder=4)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center", fontsize=tsz,
                color=ec, fontweight="bold", zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# Pre-calculate ALL Y positions top-down before drawing anything
# ══════════════════════════════════════════════════════════════════════════
CX  = 5.0    # centre x of figure  (xlim = 0–10)
BW  = 8.0    # standard box width
BH  = 0.80   # standard box height
GAP = 0.28   # gap between consecutive box edges

# Top of content
IY = 19.30                          # Init centre

EP_TOP = 18.50                      # epoch frame top

ES_Y = 18.00                        # Epoch-start centre

BA_TOP = 17.35                      # batch frame top

# ── Batch-loop boxes (top-down) ───────────────────────────────────────────
LB_Y  = 16.60                                           # Load Batch
LB_H  = BH

FW_Y  = LB_Y  - LB_H/2 - GAP - (BH + 0.12)/2          # Forward Pass
FW_H  = BH + 0.12

LS_Y  = FW_Y  - FW_H/2 - GAP - (BH + 0.20)/2          # CE Loss
LS_H  = BH + 0.20

BK_Y  = LS_Y  - LS_H/2 - GAP - BH/2                   # Backward
BK_H  = BH

OP_H  = BH + 0.24
OP_Y  = BK_Y  - BK_H/2 - GAP - OP_H/2                 # Clip+Adam+Sched
OP_BOT = OP_Y - OP_H/2                                  # bottom edge

BA_BOT = OP_BOT - 0.42                                  # batch frame hugs boxes

# ── Epoch-level boxes (below batch frame) ────────────────────────────────
VL_H  = BH + 0.10
VL_Y  = BA_BOT - 0.45 - VL_H/2                         # Validate

CK_Y  = VL_Y  - VL_H/2 - GAP - BH/2 - 0.05            # Checkpoint/Patience

DW, DH = 4.8, 1.10
DY    = CK_Y  - BH/2 - GAP - DH/2                      # Decision diamond

# Epoch frame bottom gives room for loop-back turn and "No" label
EP_BOT = max(1.60, DY - DH/2 - 0.55)

# Done box
DONE_Y  = EP_BOT - 0.45 - 0.72/2

# ══════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════
fig_h = IY + 1.0   # a little above Init
fig = plt.figure(figsize=(11, fig_h * 11/20), facecolor=BG)
fig.text(0.5, 0.988, "Training Loop — PoPEViT",
         ha="center", va="top", fontsize=26,
         fontweight="bold", color=TEXT)
fig.text(0.5, 0.972,
         "AdamW  ·  Cosine-LR Warm-up  ·  Class-Weighted CE Loss"
         "  ·  Early Stopping on val AUROC",
         ha="center", va="top", fontsize=11.5, color=MUTED, style="italic")

ax = fig.add_axes([0.06, 0.025, 0.88, 0.905])
ax.set_xlim(0, 10)
ax.set_ylim(DONE_Y - 0.50, IY + 0.85)
ax.axis("off")
ax.set_facecolor(BG)

YLO = DONE_Y - 0.60

# ══════════════════════════════════════════════════════════════════════════
# DRAW — epoch frame (outermost, drawn first so it sits behind everything)
# ══════════════════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch(
    (0.22, EP_BOT), 9.56, EP_TOP - EP_BOT,
    boxstyle="round,pad=0.06",
    facecolor="#F2FCF2", edgecolor=GREEN,
    linewidth=2.2, zorder=1, alpha=0.60))
ax.text(0.40, EP_TOP + 0.05,
        "for epoch in 1 … max_epochs = 30:",
        fontsize=10.5, color=GREEN, va="bottom",
        style="italic", fontweight="bold")

# ── Batch frame ───────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (0.68, BA_BOT), 8.64, BA_TOP - BA_BOT,
    boxstyle="round,pad=0.05",
    facecolor="#F0F4FF", edgecolor=BLUE,
    linewidth=1.8, zorder=1, alpha=0.60))
ax.text(0.86, BA_TOP + 0.05,
        "for batch in DataLoader                (train,  batch_size=32,  shuffle=True):",
        fontsize=9.5, color=BLUE, va="bottom",
        style="italic", fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, CX, IY, BW, 1.10,
     "Initialise",
     sub="model  ·  AdamW (lr=3×10⁻⁴, wd=10⁻²)  ·  CosineAnnealingLR (T=30, 5-ep warm-up)"
         "  ·  class weights wᵢ∝1/freqᵢ",
     fc=FC_BLUE, ec=BLUE, elw=2.4, tsz=15, ssz=10.5)
va(ax, CX, IY - 0.50, EP_TOP + 0.04, BLUE, 1.9)

# ══════════════════════════════════════════════════════════════════════════
# EPOCH-START
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, CX, ES_Y, BW - 0.3, 0.65,
     "model.train()  ·  LR warm-up: linear ramp for first 5 epochs",
     fc=FC_GREEN, ec=GREEN, elw=1.8, tsz=12, ssz=10.5)
va(ax, CX, ES_Y - 0.325, BA_TOP + 0.04, GREEN, 1.6)

# ══════════════════════════════════════════════════════════════════════════
# BATCH-LOOP STEPS
# ══════════════════════════════════════════════════════════════════════════

# 1. Load batch
rbox(ax, CX, LB_Y, BW - 0.8, LB_H,
     "Load Mini-Batch",
     sub="(B=32, 3, 224, 224)  ·  augmented on-the-fly  ·  pin to GPU",
     fc=FC_BLUE, ec=BLUE, elw=1.8, tsz=13, ssz=10.5)
va(ax, CX, LB_Y - LB_H/2, FW_Y + FW_H/2, BLUE, 1.6)

# 2. Forward pass
rbox(ax, CX, FW_Y, BW - 0.8, FW_H,
     "Forward Pass",
     sub="logits  =  model( x )          output shape:  (B, 4)  raw logits",
     fc=FC_GREEN, ec=GREEN, elw=2.2, tsz=14, ssz=11.5, tc=GREEN)
va(ax, CX, FW_Y - FW_H/2, LS_Y + LS_H/2, GREEN, 1.6)

# 3. CE loss
rbox(ax, CX, LS_Y, BW - 0.8, LS_H,
     "Class-Weighted Cross-Entropy Loss",
     sub="ℒ  =  −  ∑ᵢ  wᵢ · yᵢ · log p̂ᵢ          wᵢ  ∝  1 / freq( classᵢ )",
     fc=FC_RED, ec=RED, elw=1.8, tsz=13, ssz=11.5)
va(ax, CX, LS_Y - LS_H/2, BK_Y + BK_H/2, RED, 1.6)

# 4. Backward
rbox(ax, CX, BK_Y, BW - 0.8, BK_H,
     "Backward Pass  (autograd)",
     sub="optimizer.zero_grad()  ·  ℒ.backward()  →  ∂ℒ/∂θ  for all parameters",
     fc=FC_RED, ec=RED, elw=1.8, tsz=13, ssz=10.5)
va(ax, CX, BK_Y - BK_H/2, OP_Y + OP_H/2, RED, 1.6)

# 5. Clip + AdamW + Sched  (combined)
rbox(ax, CX, OP_Y, BW - 0.8, OP_H,
     "Clip Gradients  ·  AdamW Step  ·  LR Scheduler Step",
     sub="max_norm=1.0   ·   θ ← θ − lr·m̂/(√v̂+ε) − λθ   ·   CosineAnnealingLR.step()",
     fc=FC_ORANGE, ec=ORANGE, elw=1.8, tsz=12.5, ssz=10.5)

# ── Batch loop-back arrow (left side) ─────────────────────────────────────
BOX_L  = CX - (BW - 0.8)/2     # left edge of inner boxes = 1.4
LB_TOP = LB_Y + LB_H/2          # top of Load Batch = 17.0
LB_X   = BOX_L - 0.58           # left column for return arrow = 0.82

ax.plot([BOX_L, LB_X], [OP_BOT, OP_BOT],
        color=BLUE, lw=1.5, alpha=0.80, zorder=2)
ax.plot([LB_X, LB_X], [OP_BOT, LB_TOP],
        color=BLUE, lw=1.5, alpha=0.80, zorder=2)
ax.annotate("", xy=(BOX_L, LB_TOP), xytext=(LB_X, LB_TOP),
            arrowprops=dict(arrowstyle="-|>", color=BLUE,
                            lw=1.5, mutation_scale=12), zorder=5)
ax.text(LB_X - 0.28, (OP_BOT + LB_TOP) / 2,
        "next  batch", ha="center", va="center",
        fontsize=9, color=BLUE, style="italic", rotation=90)

# Exit batch loop → Validation
va(ax, CX, OP_BOT, VL_Y + VL_H/2 + 0.04, MUTED, 1.5)

# ══════════════════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, CX, VL_Y, BW - 0.3, VL_H,
     "Validate",
     sub="model.eval() + no_grad  ·  forward on val set  ·  per-class AUROC (one-vs-rest, macro-avg)",
     fc=FC_PURPLE, ec=PURPLE, elw=1.8, tsz=13, ssz=10.5)
va(ax, CX, VL_Y - VL_H/2, CK_Y + BH/2, PURPLE, 1.6)

# ══════════════════════════════════════════════════════════════════════════
# CHECKPOINT + PATIENCE UPDATE
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, CX, CK_Y, BW - 0.3, BH,
     "Update Checkpoint  ·  Patience Counter",
     sub="if val_AUROC > best: save weights, patience=0     else: patience += 1",
     fc=FC_YELLOW, ec=YELLOW, elw=1.8, tsz=12, ssz=10.5)
va(ax, CX, CK_Y - BH/2, DY + DH/2, YELLOW, 1.5)

# ══════════════════════════════════════════════════════════════════════════
# DECISION DIAMOND  —  patience > 8?
# ══════════════════════════════════════════════════════════════════════════
diamond(ax, CX, DY, DW, DH,
        "patience > 8?",
        sub="early stopping criterion",
        ec=ORANGE, fc=FC_ORANGE, tsz=12)

# ── "Yes" → Stop Training  (exits RIGHT) ─────────────────────────────────
YES_X = CX + DW/2     # right vertex x
STOP_W, STOP_H = 1.55, 0.68
STOP_CX = YES_X + 0.20 + STOP_W/2

ax.annotate("", xy=(STOP_CX - STOP_W/2, DY),
            xytext=(YES_X, DY),
            arrowprops=dict(arrowstyle="-|>", color=RED,
                            lw=1.5, mutation_scale=11), zorder=5)
rbox(ax, STOP_CX, DY, STOP_W, STOP_H,
     "Stop\nTraining",
     fc=FC_RED, ec=RED, elw=2.0, tsz=11, tc=RED)
ax.text(YES_X + 0.12, DY + 0.28,
        "Yes", fontsize=10, color=RED, fontweight="bold")

# ── "No" → next epoch  (exits BOTTOM, curves right, up right side) ────────
NO_BOT = DY - DH/2
TURN_Y = NO_BOT - 0.30   # where the line turns horizontal
ER_X   = 9.52             # right-side loop-back column

va(ax, CX, NO_BOT, TURN_Y, ORANGE, 1.4)
ax.text(CX + 0.16, TURN_Y - 0.22,
        "No", fontsize=10, color=GREEN, fontweight="bold")

ax.plot([CX,   ER_X], [TURN_Y, TURN_Y],
        color=GREEN, lw=1.5, alpha=0.80, zorder=2)
ax.plot([ER_X, ER_X], [TURN_Y, ES_Y],
        color=GREEN, lw=1.5, alpha=0.80, zorder=2)
ax.annotate("", xy=(CX + (BW - 0.3)/2, ES_Y),
            xytext=(ER_X, ES_Y),
            arrowprops=dict(arrowstyle="-|>", color=GREEN,
                            lw=1.5, mutation_scale=12), zorder=5)
ax.text(ER_X + 0.22, (TURN_Y + ES_Y) / 2,
        "next  epoch", ha="left", va="center",
        fontsize=9, color=GREEN, style="italic")

# ══════════════════════════════════════════════════════════════════════════
# TRAINING COMPLETE  (below epoch frame)
# ══════════════════════════════════════════════════════════════════════════
va(ax, CX, EP_BOT, DONE_Y + 0.72/2 + 0.04, MUTED, 1.9)
rbox(ax, CX, DONE_Y, BW, 0.72,
     "Training Complete  —  Load Best Checkpoint  →  model.eval()",
     fc=FC_PURPLE, ec=PURPLE, elw=2.4, tsz=13.5)

# ── Section chip ──────────────────────────────────────────────────────────
fig.text(0.5, 0.014,
         "PoPEViT  ·  Brain Tumour MRI  ·  4-class classification",
         ha="center", va="center", fontsize=11, color=PURPLE,
         bbox=dict(boxstyle="round,pad=0.38", facecolor=BG,
                   edgecolor=PURPLE, linewidth=1.5))

fig.savefig("figures/fig_training_loop_v2.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
fig.savefig("figures/fig_training_loop_v2.pdf",
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved → figures/fig_training_loop_v2.{png,pdf}")
