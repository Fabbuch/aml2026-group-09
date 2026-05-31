#!/usr/bin/env python3
"""
academic_figures.py
Academic visualizations for Brain Tumour MRI Classification with PoPEViT.

Figures produced (saved to ./figures/):
  fig1_research_pipeline.{png,pdf}  — end-to-end research pipeline overview
  fig2_training_loop.{png,pdf}      — detailed training loop flowchart
  fig3_architecture.{png,pdf}       — PoPEViT architecture + PoPE mechanism

Style matches visualize_pope_forward.py — light panels, colored borders, DejaVu Sans.

Usage:
    python academic_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans"})

# ── Palette (mirrors visualize_pope_forward.py) ───────────────────────────────
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

FC_BLUE   = "#DFF0FF"
FC_GREEN  = "#DAFBE1"
FC_RED    = "#FFEBE9"
FC_YELLOW = "#FFF8C5"
FC_PURPLE = "#FBEFFF"
FC_ORANGE = "#FFF1E5"
FC_GRAY   = "#F6F8FA"
FC_TEAL   = "#DDF4FF"


# ═════════════════════════════════════════════════════════════════════════════
# Drawing utilities
# ═════════════════════════════════════════════════════════════════════════════

def rbox(ax, x, y, w, h, title, sub="", fc=PANEL, ec=BORDER, elw=1.4,
         tsz=14, ssz=11, tc=TEXT, sc=MUTED, bold=True, pad=0.012):
    """Rounded box. (x, y) = bottom-left corner."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad}",
        linewidth=elw, edgecolor=ec, facecolor=fc, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=tsz, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
                fontsize=ssz, color=sc, zorder=4, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tsz, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)


def varrow(ax, x, y_top, y_bot, color=MUTED, lw=1.8):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=5)


def harrow(ax, x_left, y, x_right, color=MUTED, lw=2.0):
    ax.annotate("", xy=(x_right, y), xytext=(x_left, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14), zorder=5)


def section_chip(fig, xc, text, color=BLUE):
    fig.text(xc, 0.048, text, ha="center", va="center", fontsize=13,
             fontweight="bold", color=color,
             bbox=dict(boxstyle="round,pad=0.40", facecolor=BG,
                       edgecolor=color, linewidth=1.5, alpha=1.0))


def dashed_outline(ax, x, y, w, h, ec=LGRAY, lw=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor="none", edgecolor=ec, linewidth=lw,
        linestyle="--", zorder=1))


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Research Pipeline Overview
# ═════════════════════════════════════════════════════════════════════════════

def fig_pipeline():
    fig = plt.figure(figsize=(32, 13), facecolor=BG)
    fig.text(0.5, 0.975, "Brain Tumour MRI Classification — Research Pipeline",
             ha="center", va="top", fontsize=30, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.945, "Comparing Polar (PoPE) vs Rotary (RoPE) positional embeddings "
             "in Vision Transformers on 4-class MRI classification",
             ha="center", va="top", fontsize=15, color=MUTED)

    AX_Y0, AX_H = 0.10, 0.82

    # ── Section widths and x-positions ───────────────────────────────────
    secs = [
        (0.020, 0.155),   # ① Dataset
        (0.185, 0.125),   # ② Split
        (0.320, 0.155),   # ③ Preprocessing
        (0.485, 0.175),   # ④ Models
        (0.670, 0.155),   # ⑤ Optimisation
        (0.835, 0.148),   # ⑥ Evaluation
    ]

    axes = []
    for x0, w in secs:
        ax = fig.add_axes((x0, AX_Y0, w, AX_H))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(BG)
        axes.append(ax)

    # Inter-section arrows
    CY = 0.50
    for i in range(len(secs) - 1):
        x1 = secs[i][0] + secs[i][1]
        x2 = secs[i + 1][0]
        fig.add_artist(FancyArrowPatch(
            (x1 + 0.002, CY), (x2 - 0.002, CY),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=18,
            color=BLUE, lw=2.5, zorder=10))

    # ── ① Dataset ──────────────────────────────────────────────────────────
    a = axes[0]
    a.set_title("Brain MRI Dataset", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)
    rbox(a, 0.06, 0.78, 0.88, 0.14,
         "3,264 images  ·  4 classes",
         sub="Kaggle Brain Tumour Classification",
         fc=FC_BLUE, ec=BLUE, elw=2.0, tsz=13, ssz=11, tc=BLUE)

    cls_data = [
        ("No Tumour",   "~900 imgs", FC_GRAY,   BORDER),
        ("Meningioma",  "~704 imgs", FC_ORANGE,  ORANGE),
        ("Glioma",      "~826 imgs", FC_RED,     RED),
        ("Pituitary",   "~827 imgs", FC_PURPLE,  PURPLE),
    ]
    bh = 0.128
    for i, (cn, cnt, fc, ec) in enumerate(cls_data):
        y = 0.56 - i * (bh + 0.025)
        rbox(a, 0.06, y, 0.88, bh, cn, sub=cnt,
             fc=fc, ec=ec, elw=2.0, tsz=13, ssz=11, tc=TEXT)

    section_chip(fig, secs[0][0] + secs[0][1] / 2, "① Dataset", BLUE)

    # ── ② Split ────────────────────────────────────────────────────────────
    a = axes[1]
    a.set_title("Stratified Split", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)
    splits = [
        ("Train",  "76%  ·  2,476 imgs", FC_GREEN,  GREEN,  0.82),
        ("Val",    "12%  ·  394 imgs",   FC_BLUE,   BLUE,   0.52),
        ("Test",   "12%  ·  394 imgs",   FC_YELLOW, YELLOW, 0.22),
    ]
    for lbl, sub_, fc, ec, y in splits:
        rbox(a, 0.06, y, 0.88, 0.24, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=14, ssz=11, tc=TEXT)

    section_chip(fig, secs[1][0] + secs[1][1] / 2, "② Split", GREEN)

    # ── ③ Preprocessing ────────────────────────────────────────────────────
    a = axes[2]
    a.set_title("Preprocessing / Augmentation", fontsize=16,
                fontweight="bold", color=TEXT, pad=6)

    rbox(a, 0.05, 0.80, 0.90, 0.12,
         "Train  (augmented)", fc=FC_GREEN, ec=GREEN,
         elw=1.5, tsz=12, tc=GREEN, bold=True)
    augs = [
        ("Resize  224 × 224",          FC_GRAY, BORDER),
        ("RandomHorizontalFlip  p=0.5", FC_GRAY, BORDER),
        ("RandomAffine  ±15°",          FC_GRAY, BORDER),
        ("Normalize  μ/σ",              FC_GRAY, BORDER),
    ]
    for i, (lbl, fc, ec) in enumerate(augs):
        y = 0.68 - i * 0.095
        rbox(a, 0.05, y, 0.90, 0.082, lbl,
             fc=fc, ec=ec, elw=1.3, tsz=12, tc=TEXT, bold=False)
        varrow(a, 0.50, y + 0.082, y + 0.082 + 0.010, LGRAY, 1.0)

    rbox(a, 0.05, 0.26, 0.90, 0.12,
         "Val / Test  (deterministic)", fc=FC_BLUE, ec=BLUE,
         elw=1.5, tsz=12, tc=BLUE, bold=True)
    det_augs = [
        ("Resize  224 × 224",  FC_GRAY, BORDER),
        ("Normalize  μ/σ",     FC_GRAY, BORDER),
    ]
    for i, (lbl, fc, ec) in enumerate(det_augs):
        y = 0.145 - i * 0.095
        rbox(a, 0.05, y, 0.90, 0.082, lbl,
             fc=fc, ec=ec, elw=1.3, tsz=12, tc=TEXT, bold=False)

    section_chip(fig, secs[2][0] + secs[2][1] / 2, "③ Preprocessing", TEAL)

    # ── ④ Models ───────────────────────────────────────────────────────────
    a = axes[3]
    a.set_title("Model Zoo  (4 architectures)", fontsize=16,
                fontweight="bold", color=TEXT, pad=6)
    models = [
        ("PoPEViT  (ours)",
         "Polar positional encoding on Q, K\n"
         "ViT-base  ·  d=512  ·  L=6  ·  h=8",
         FC_RED, RED),
        ("RoPEViT  (baseline)",
         "1-D rotary positional embedding\n"
         "ViT-base  ·  d=512  ·  L=6  ·  h=8",
         FC_ORANGE, ORANGE),
        ("DeiT-Small  +  PoPE/RoPE",
         "Pretrained ViT, attn heads swapped\n"
         "ImageNet weights  ·  22M params",
         FC_PURPLE, PURPLE),
        ("ResNet-18  (CNN baseline)",
         "Standard CNN, trained from scratch\n"
         "11M params  ·  batch-norm",
         FC_GREEN, GREEN),
    ]
    bh = 0.175
    for i, (lbl, sub_, fc, ec) in enumerate(models):
        y = 0.78 - i * (bh + 0.028)
        rbox(a, 0.04, y, 0.92, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=13, ssz=10.5, tc=TEXT)

    section_chip(fig, secs[3][0] + secs[3][1] / 2, "④ Models", RED)

    # ── ⑤ Optimisation ─────────────────────────────────────────────────────
    a = axes[4]
    a.set_title("Optimisation", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)
    opt_items = [
        ("AdamW",
         "lr = 3×10⁻⁴\nweight decay = 1×10⁻²",
         FC_BLUE, BLUE),
        ("Cosine Annealing LR",
         "5-epoch linear warm-up\nmax epochs = 30",
         FC_TEAL, TEAL),
        ("Class-Weighted CE Loss",
         "wᵢ ∝ 1 / freq(classᵢ)\ncorrects class imbalance",
         FC_RED, RED),
        ("Gradient Clipping",
         "max_norm = 1.0\nprevents exploding gradients",
         FC_YELLOW, YELLOW),
        ("Early Stopping",
         "patience = 8 epochs\nmonitor val AUROC",
         FC_ORANGE, ORANGE),
    ]
    bh = 0.14
    for i, (lbl, sub_, fc, ec) in enumerate(opt_items):
        y = 0.79 - i * (bh + 0.018)
        rbox(a, 0.04, y, 0.92, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=13, ssz=10.5, tc=TEXT)

    section_chip(fig, secs[4][0] + secs[4][1] / 2, "⑤ Optimisation", ORANGE)

    # ── ⑥ Evaluation ───────────────────────────────────────────────────────
    a = axes[5]
    a.set_title("Evaluation", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)
    metrics = [
        ("Mean AUROC  (macro)",
         "primary metric\none-vs-rest per class",
         FC_PURPLE, PURPLE),
        ("Per-class AUROC",
         "no_tumor · meningioma\nglioma · pituitary",
         FC_BLUE, BLUE),
        ("Accuracy",
         "secondary metric\n4-class top-1",
         FC_GREEN, GREEN),
        ("Confusion Matrix",
         "normalised 4×4\nheatmap visualisation",
         FC_ORANGE, ORANGE),
    ]
    bh = 0.16
    for i, (lbl, sub_, fc, ec) in enumerate(metrics):
        y = 0.78 - i * (bh + 0.025)
        rbox(a, 0.04, y, 0.92, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=13, ssz=10.5, tc=TEXT)

    # Results summary box
    rbox(a, 0.04, 0.02, 0.92, 0.12,
         "Best Results",
         sub="ResNet-18: AUROC 0.999  ·  PoPEViT: AUROC 0.964",
         fc=FC_GRAY, ec=BORDER, elw=1.5, tsz=12, ssz=10, tc=TEXT)

    section_chip(fig, secs[5][0] + secs[5][1] / 2, "⑥ Evaluation", PURPLE)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Training Loop
# ═════════════════════════════════════════════════════════════════════════════

def fig_training_loop():
    fig = plt.figure(figsize=(32, 14), facecolor=BG)
    fig.text(0.5, 0.975, "Training Loop — PoPEViT / Vision Transformer",
             ha="center", va="top", fontsize=30, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.945,
             "AdamW + Cosine-LR + Class-Weighted CE  ·  "
             "4-class Brain Tumour MRI  ·  early stopping on val AUROC",
             ha="center", va="top", fontsize=15, color=MUTED)

    AX_Y0, AX_H = 0.10, 0.82

    secs = [
        (0.020, 0.185),   # ① Setup
        (0.215, 0.230),   # ② Batch loop
        (0.455, 0.215),   # ③ Validation
        (0.680, 0.195),   # ④ Final evaluation
        (0.885, 0.098),   # ⑤ Results
    ]

    axes = []
    for x0, w in secs:
        ax = fig.add_axes((x0, AX_Y0, w, AX_H))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(BG)
        axes.append(ax)

    # Inter-section arrows
    CY = 0.50
    for i in range(len(secs) - 1):
        x1 = secs[i][0] + secs[i][1]
        x2 = secs[i + 1][0]
        fig.add_artist(FancyArrowPatch(
            (x1 + 0.002, CY), (x2 - 0.002, CY),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=18,
            color=BLUE, lw=2.5, zorder=10))

    # ── ① Setup ────────────────────────────────────────────────────────────
    a = axes[0]
    a.set_title("Initialisation", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)

    setup_items = [
        ("Model",
         "PoPEViT / RoPEViT / DeiT-S\nResNet-18  ·  random or pretrained init",
         FC_BLUE, BLUE),
        ("Class Weights",
         "wᵢ = 1 / freq(classᵢ)\nnormalised to sum to C=4",
         FC_RED, RED),
        ("AdamW Optimiser",
         "lr=3×10⁻⁴  ·  wd=1×10⁻²\nbetas=(0.9, 0.999)  ·  eps=1×10⁻⁸",
         FC_ORANGE, ORANGE),
        ("LR Scheduler",
         "CosineAnnealingLR  T=30\n+ linear warm-up for 5 epochs",
         FC_TEAL, TEAL),
        ("Early-Stop Tracker",
         "best_auroc = 0\npatience_counter = 0  ·  patience=8",
         FC_YELLOW, YELLOW),
    ]
    bh = 0.148
    for i, (lbl, sub_, fc, ec) in enumerate(setup_items):
        y = 0.80 - i * (bh + 0.020)
        rbox(a, 0.04, y, 0.92, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=13, ssz=10.5, tc=TEXT)
        if i < len(setup_items) - 1:
            varrow(a, 0.50, y, y - 0.015, LGRAY, 1.2)

    section_chip(fig, secs[0][0] + secs[0][1] / 2, "① Setup", BLUE)

    # ── ② Batch Loop ──────────────────────────────────────────────────────
    a = axes[1]
    a.set_title("Batch Training Loop\n(for each epoch → for each batch)",
                fontsize=15, fontweight="bold", color=TEXT, pad=6)

    # Outer dashed frame (epoch loop)
    a.add_patch(FancyBboxPatch(
        (0.02, 0.01), 0.96, 0.95,
        boxstyle="round,pad=0.01", facecolor="none",
        edgecolor=GREEN, linewidth=1.5, linestyle="--", zorder=1))
    a.text(0.08, 0.97, "for epoch in 1…30:", fontsize=10,
           color=GREEN, style="italic", va="top")

    # Inner dashed frame (batch loop)
    a.add_patch(FancyBboxPatch(
        (0.06, 0.03), 0.88, 0.80,
        boxstyle="round,pad=0.01", facecolor="none",
        edgecolor=BLUE, linewidth=1.2, linestyle=":", zorder=1))
    a.text(0.12, 0.84, "for batch in DataLoader(batch_size=32):",
           fontsize=9, color=BLUE, style="italic", va="top")

    batch_items = [
        ("Load Mini-Batch",
         "(B=32, 3, 224, 224) · augmented",
         FC_BLUE, BLUE),
        ("Forward Pass",
         "logits = model(x)\nshape: (B, 4)",
         FC_GREEN, GREEN),
        ("Class-Weighted CE Loss",
         "ℒ = −∑ᵢ wᵢ · yᵢ · log p̂ᵢ",
         FC_RED, RED),
        ("Backward  (autograd)",
         "optimizer.zero_grad()\nℒ.backward() → ∂ℒ/∂θ",
         FC_RED, RED),
        ("Gradient Clipping",
         "clip_grad_norm_(θ, max_norm=1.0)",
         FC_ORANGE, ORANGE),
        ("AdamW Step",
         "θ ← θ − lr·m̂/(√v̂+ε) − lr·λ·θ",
         FC_ORANGE, ORANGE),
        ("LR Scheduler Step",
         "lrₜ = ½η(1+cos(πt/T))+η_min",
         FC_TEAL, TEAL),
    ]
    bh = 0.096
    for i, (lbl, sub_, fc, ec) in enumerate(batch_items):
        y = 0.73 - i * (bh + 0.014)
        rbox(a, 0.10, y, 0.80, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=1.8, tsz=12, ssz=10, tc=TEXT)
        if i < len(batch_items) - 1:
            varrow(a, 0.50, y, y - 0.010, LGRAY, 1.2)

    section_chip(fig, secs[1][0] + secs[1][1] / 2, "② Batch Loop", GREEN)

    # ── ③ Validation ──────────────────────────────────────────────────────
    a = axes[2]
    a.set_title("End-of-Epoch Validation", fontsize=16,
                fontweight="bold", color=TEXT, pad=6)

    val_items = [
        ("model.eval()  +  torch.no_grad()",
         "disable dropout · stop gradient tracking",
         FC_GRAY, BORDER),
        ("Forward on Val Set",
         "all 394 validation images\nbatch_size=32, shuffle=False",
         FC_BLUE, BLUE),
        ("Compute Per-Class AUROC",
         "one-vs-rest  ·  sklearn roc_auc_score\nmacro-averaged mean AUROC",
         FC_PURPLE, PURPLE),
        ("Checkpoint Save",
         "if val_AUROC > best_AUROC:\n  save model weights to .pt",
         FC_GREEN, GREEN),
        ("Early Stopping",
         "patience_counter += 1 if no improvement\nstop if patience_counter > 8",
         FC_RED, RED),
        ("LR Warmup / Scheduler",
         "step scheduler after validation\n(CosineAnnealingLR.step())",
         FC_TEAL, TEAL),
    ]
    bh = 0.128
    for i, (lbl, sub_, fc, ec) in enumerate(val_items):
        y = 0.80 - i * (bh + 0.020)
        rbox(a, 0.04, y, 0.92, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=12, ssz=10, tc=TEXT)
        if i < len(val_items) - 1:
            varrow(a, 0.50, y, y - 0.015, LGRAY, 1.2)

    section_chip(fig, secs[2][0] + secs[2][1] / 2, "③ Validation", PURPLE)

    # ── ④ Final Evaluation ────────────────────────────────────────────────
    a = axes[3]
    a.set_title("Final Test Evaluation", fontsize=16,
                fontweight="bold", color=TEXT, pad=6)

    final_items = [
        ("Load Best Checkpoint",
         "restore weights with\nhighest val AUROC",
         FC_GRAY, BORDER),
        ("model.eval()  +  torch.no_grad()",
         "held-out test set  ·  394 images\nnever seen during training",
         FC_BLUE, BLUE),
        ("Per-Class AUROC",
         "no_tumor · meningioma\nglioma · pituitary  ·  one-vs-rest",
         FC_PURPLE, PURPLE),
        ("Confusion Matrix",
         "4×4 normalised heatmap\nrow = true, col = predicted",
         FC_ORANGE, ORANGE),
        ("Model Comparison",
         "rank all models by\nmean test AUROC",
         FC_GREEN, GREEN),
    ]
    bh = 0.143
    for i, (lbl, sub_, fc, ec) in enumerate(final_items):
        y = 0.80 - i * (bh + 0.020)
        rbox(a, 0.04, y, 0.92, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=12.5, ssz=10.5, tc=TEXT)
        if i < len(final_items) - 1:
            varrow(a, 0.50, y, y - 0.015, LGRAY, 1.2)

    section_chip(fig, secs[3][0] + secs[3][1] / 2, "④ Final Eval", TEAL)

    # ── ⑤ Results ─────────────────────────────────────────────────────────
    a = axes[4]
    a.set_title("Results", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)

    results = [
        ("ResNet-18",   "AUROC\n0.999", FC_GREEN,  GREEN),
        ("DeiT-S",      "AUROC\n0.986", FC_BLUE,   BLUE),
        ("DeiT+PoPE",   "AUROC\n0.979", FC_RED,    RED),
        ("RoPEViT",     "AUROC\n0.965", FC_ORANGE, ORANGE),
        ("PoPEViT",     "AUROC\n0.964", FC_PURPLE, PURPLE),
    ]
    bh = 0.140
    for i, (lbl, sub_, fc, ec) in enumerate(results):
        y = 0.80 - i * (bh + 0.018)
        rbox(a, 0.04, y, 0.92, bh, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=12, ssz=12, tc=TEXT)

    # Grid search note
    rbox(a, 0.04, 0.02, 0.92, 0.085,
         "Grid Search",
         sub="lr × dropout × patch (27 combos)",
         fc=FC_YELLOW, ec=YELLOW, elw=1.5, tsz=11, ssz=9.5, tc=TEXT)

    section_chip(fig, secs[4][0] + secs[4][1] / 2, "⑤ Results", ORANGE)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PoPEViT Architecture + PoPE Mechanism
# ═════════════════════════════════════════════════════════════════════════════

def fig_architecture():
    fig = plt.figure(figsize=(32, 14), facecolor=BG)
    fig.text(0.5, 0.975, "PoPEViT — Model Architecture",
             ha="center", va="top", fontsize=30, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.945,
             "Vision Transformer with Polar Positional Embeddings  ·  "
             "4-class Brain Tumour MRI  ·  patch_size=16  ·  d=512  ·  L=6  ·  h=8",
             ha="center", va="top", fontsize=15, color=MUTED)

    AX_Y0, AX_H = 0.10, 0.82

    secs = [
        (0.020, 0.145),   # ① Input
        (0.175, 0.145),   # ② Embedding
        (0.330, 0.155),   # ③ Tokens
        (0.495, 0.250),   # ④ Transformer block (wide)
        (0.755, 0.125),   # ⑤ PoPE attention detail
        (0.890, 0.095),   # ⑥ Head + output
    ]

    axes = []
    for x0, w in secs:
        ax = fig.add_axes((x0, AX_Y0, w, AX_H))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(BG)
        axes.append(ax)

    CY = 0.50
    for i in range(len(secs) - 1):
        x1 = secs[i][0] + secs[i][1]
        x2 = secs[i + 1][0]
        fig.add_artist(FancyArrowPatch(
            (x1 + 0.002, CY), (x2 - 0.002, CY),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=18,
            color=BLUE, lw=2.5, zorder=10))

    # ── ① Input ────────────────────────────────────────────────────────────
    a = axes[0]
    a.set_title("Input", fontsize=16, fontweight="bold", color=TEXT, pad=6)

    rbox(a, 0.05, 0.72, 0.90, 0.18,
         "Brain MRI Image",
         sub="(B, 3, 224, 224)\nfloat32 · normalised [0,1]",
         fc=FC_BLUE, ec=BLUE, elw=2.0, tsz=14, ssz=11)
    varrow(a, 0.50, 0.72, 0.57, BLUE, 1.6)
    rbox(a, 0.05, 0.42, 0.90, 0.14,
         "EnsureRGB",
         sub="1ch/4ch → 3ch\nchannel normalisation",
         fc=FC_GRAY, ec=BORDER, elw=1.5, tsz=13, ssz=11)
    varrow(a, 0.50, 0.42, 0.28, MUTED, 1.4)
    rbox(a, 0.05, 0.12, 0.90, 0.14,
         "ImageNet Norm",
         sub="μ=[.485,.456,.406]\nσ=[.229,.224,.225]",
         fc=FC_GRAY, ec=BORDER, elw=1.5, tsz=13, ssz=11)

    section_chip(fig, secs[0][0] + secs[0][1] / 2, "① Input", BLUE)

    # ── ② Embedding ────────────────────────────────────────────────────────
    a = axes[1]
    a.set_title("Patch Embedding", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)

    rbox(a, 0.05, 0.75, 0.90, 0.15,
         "Split into Patches",
         sub="16×16 px patches\n14×14 = 196 patches",
         fc=FC_GREEN, ec=GREEN, elw=2.0, tsz=13, ssz=11)
    varrow(a, 0.50, 0.75, 0.60, GREEN, 1.6)
    rbox(a, 0.05, 0.44, 0.90, 0.15,
         "Linear Projection",
         sub="16²×3 = 768  →  d=512\none weight matrix W_E",
         fc=FC_GREEN, ec=GREEN, elw=2.0, tsz=13, ssz=11)
    varrow(a, 0.50, 0.44, 0.29, GREEN, 1.6)
    rbox(a, 0.05, 0.13, 0.90, 0.15,
         "Patch Tokens",
         sub="(B, 196, 512)\nsequence of patch embeddings",
         fc=FC_GREEN, ec=GREEN, elw=2.0, tsz=13, ssz=11)

    section_chip(fig, secs[1][0] + secs[1][1] / 2, "② Embedding", GREEN)

    # ── ③ Tokens ───────────────────────────────────────────────────────────
    a = axes[2]
    a.set_title("Token Sequence", fontsize=16, fontweight="bold",
                color=TEXT, pad=6)

    rbox(a, 0.05, 0.78, 0.90, 0.14,
         "Prepend [CLS] Token",
         sub="learnable vector (1, 512)\naggregate sequence info",
         fc=FC_PURPLE, ec=PURPLE, elw=2.0, tsz=13, ssz=11)
    varrow(a, 0.50, 0.78, 0.63, PURPLE, 1.6)
    rbox(a, 0.05, 0.48, 0.90, 0.14,
         "Add Position Embedding",
         sub="learned absolute  (197, 512)\ncoarse spatial information",
         fc=FC_TEAL, ec=TEAL, elw=2.0, tsz=13, ssz=11)
    varrow(a, 0.50, 0.48, 0.33, TEAL, 1.6)
    rbox(a, 0.05, 0.20, 0.90, 0.12,
         "Dropout  (p=0.1)",
         fc=FC_GRAY, ec=BORDER, elw=1.5, tsz=13, ssz=11)
    varrow(a, 0.50, 0.20, 0.08, MUTED, 1.4)
    rbox(a, 0.05, 0.01, 0.90, 0.07,
         "(B, 197, 512)  →  Transformer",
         fc=FC_GRAY, ec=BORDER, elw=1.2, tsz=11, bold=False)

    section_chip(fig, secs[2][0] + secs[2][1] / 2, "③ Tokens", TEAL)

    # ── ④ Transformer Block ────────────────────────────────────────────────
    a = axes[3]
    a.set_title("Transformer Block  ×6  (with residual connections)",
                fontsize=15, fontweight="bold", color=TEXT, pad=6)

    # Dashed outer repeat box
    a.add_patch(FancyBboxPatch(
        (0.02, 0.01), 0.96, 0.95,
        boxstyle="round,pad=0.01", facecolor="none",
        edgecolor=GREEN, linewidth=2.0, linestyle="--", zorder=1))
    a.text(0.85, 0.96, "× 6", fontsize=22, color=GREEN,
           fontweight="bold", ha="center", va="top")

    block = [
        (0.79, 0.10, "LayerNorm₁",
         "normalise over d=512",        FC_BLUE, BLUE),
        (0.63, 0.12, "PoPE Multi-Head Attention",
         "8 heads  ·  d_head=64\npolar encoding on Q and K",  FC_RED, RED),
        (0.48, 0.08, "Residual Add",
         "x  ←  x  +  Attn(LN₁(x))",   FC_GREEN, GREEN),
        (0.34, 0.10, "LayerNorm₂",
         "normalise over d=512",        FC_BLUE, BLUE),
        (0.18, 0.12, "FeedForward  MLP",
         "d→1024→d  ·  GELU  ·  Dropout", FC_YELLOW, YELLOW),
        (0.03, 0.08, "Residual Add",
         "x  ←  x  +  FFN(LN₂(x))",    FC_GREEN, GREEN),
    ]
    bw = 0.72
    for y_frac, h, lbl, sub_, fc, ec in block:
        rbox(a, 0.14, y_frac, bw, h, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=13, ssz=10.5, tc=TEXT)

    # Arrows between block items
    for i in range(len(block) - 1):
        y_cur  = block[i][0]
        h_cur  = block[i][1]
        y_next = block[i + 1][0]
        h_next = block[i + 1][1]
        varrow(a, 0.50, y_cur, y_next + h_next, MUTED, 1.4)

    # Skip connection left-side lines
    skip_pairs = [
        (block[0][0] + block[0][1], block[2][0] + block[2][1] / 2),
        (block[3][0] + block[3][1], block[5][0] + block[5][1] / 2),
    ]
    sx = 0.07
    for y_top, y_bot in skip_pairs:
        a.plot([sx, sx], [y_top, y_bot], color=MUTED, lw=1.3, alpha=0.6, zorder=2)
        a.plot([sx, 0.14], [y_top, y_top], color=MUTED, lw=1.3, alpha=0.6, zorder=2)
        a.annotate("", xy=(0.14, y_bot), xytext=(sx, y_bot),
                   arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                   lw=1.3, mutation_scale=10), zorder=4)
        a.text(sx - 0.025, (y_top + y_bot) / 2, "skip",
               ha="center", va="center", fontsize=9,
               color=MUTED, rotation=90, alpha=0.7)

    section_chip(fig, secs[3][0] + secs[3][1] / 2, "④ Transformer  ×6", RED)

    # ── ⑤ PoPE Attention Detail ────────────────────────────────────────────
    a = axes[4]
    a.set_title("PoPE Attention\n(per head)", fontsize=15,
                fontweight="bold", color=TEXT, pad=6)

    pope_items = [
        (0.840, 0.10, "Q, K, V Projections",
         "W_Q · W_K · W_V\nlinear, d→d_h=64", FC_GRAY, BORDER),
        (0.695, 0.10, "Polar  Q̃, K̃",
         "|q|=softplus(·)\nθ=arctan2-angle", FC_RED, RED),
        (0.550, 0.10, "Phase Shift on K",
         "K̃  ←  K̃ · e^{iφ}\nφ: per-head learned bias", FC_RED, RED),
        (0.410, 0.10, "Attention Scores",
         "A = Q̃K̃ᵀ / √d_h\nnormalised dot-product", FC_ORANGE, ORANGE),
        (0.270, 0.10, "Softmax  +  V sum",
         "α=softmax(A)\nO=α·V  shape(N,d_h)", FC_PURPLE, PURPLE),
        (0.120, 0.10, "Output Projection",
         "concat heads\nW_O · O  →  (N,d)", FC_GREEN, GREEN),
    ]
    bw = 0.88
    for y_frac, h, lbl, sub_, fc, ec in pope_items:
        rbox(a, 0.06, y_frac, bw, h, lbl, sub=sub_,
             fc=fc, ec=ec, elw=2.0, tsz=12, ssz=10, tc=TEXT)
    for i in range(len(pope_items) - 1):
        y_cur  = pope_items[i][0]
        y_next = pope_items[i + 1][0]
        h_next = pope_items[i + 1][1]
        varrow(a, 0.50, y_cur, y_next + h_next, MUTED, 1.4)

    # Polar diagram inset
    ax_p = fig.add_axes([secs[4][0] + 0.005, AX_Y0 + 0.005, 0.050, 0.10])
    ax_p.set_facecolor("#FFF0F0")
    ax_p.set_aspect("equal")
    ax_p.set_xlim(-1.5, 1.5)
    ax_p.set_ylim(-1.5, 1.5)
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    for sp in ax_p.spines.values():
        sp.set_edgecolor(RED)
        sp.set_linewidth(1.5)
    th = np.linspace(0, 2 * np.pi, 300)
    ax_p.plot(np.cos(th), np.sin(th), color=LGRAY, lw=1.0)
    ax_p.axhline(0, color=LGRAY, lw=0.6)
    ax_p.axvline(0, color=LGRAY, lw=0.6)
    for lbl_, (theta_, col_) in [("q", (0.65, GREEN)), ("k", (1.85, ORANGE))]:
        ax_p.annotate("", xy=(np.cos(theta_), np.sin(theta_)), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=col_,
                                      lw=2.0, mutation_scale=10))
        ax_p.text(1.35 * np.cos(theta_), 1.35 * np.sin(theta_), lbl_,
                  ha="center", va="center", fontsize=9, color=col_, fontweight="bold")
    arc = np.linspace(0.65, 1.85, 50)
    ax_p.plot(0.55 * np.cos(arc), 0.55 * np.sin(arc), color=RED, lw=1.8)
    ax_p.text(0, -1.42, r"$\tilde{q}=|q|e^{i\theta}$",
              ha="center", fontsize=8, color=RED)

    section_chip(fig, secs[4][0] + secs[4][1] / 2, "⑤ PoPE Attn", RED)

    # ── ⑥ Head + Output ────────────────────────────────────────────────────
    a = axes[5]
    a.set_title("Head\n+ Output", fontsize=15, fontweight="bold",
                color=TEXT, pad=6)

    head_items = [
        (0.82, 0.10, "LayerNorm", "", FC_BLUE, BLUE),
        (0.67, 0.10, "Mean Pool", "197 tokens → 1", FC_PURPLE, PURPLE),
        (0.52, 0.10, "Linear Head", "512 → 4 logits", FC_ORANGE, ORANGE),
        (0.37, 0.10, "Softmax", "class probabilities", FC_GRAY, BORDER),
    ]
    for y_frac, h, lbl, sub_, fc, ec in head_items:
        rbox(a, 0.06, y_frac, 0.88, h, lbl, sub=sub_ if sub_ else "",
             fc=fc, ec=ec, elw=2.0, tsz=12, ssz=10.5, tc=TEXT)
    for i in range(len(head_items) - 1):
        y_cur  = head_items[i][0]
        y_next = head_items[i + 1][0]
        h_next = head_items[i + 1][1]
        varrow(a, 0.50, y_cur, y_next + h_next, MUTED, 1.4)

    # Class chips
    cls4 = [("No\nTumour", FC_BLUE, BLUE), ("Menin-\ngioma", FC_ORANGE, ORANGE),
            ("Glioma", FC_RED, RED), ("Pituitary", FC_PURPLE, PURPLE)]
    bw4 = 0.88 / 4
    for i, (cn, fc, ec) in enumerate(cls4):
        x_ = 0.06 + i * bw4
        rbox(a, x_, 0.16, bw4 - 0.02, 0.18, cn,
             fc=fc, ec=ec, elw=1.8, tsz=9.5, tc=TEXT, bold=False)
    varrow(a, 0.50, 0.37, 0.34, MUTED, 1.2)

    section_chip(fig, secs[5][0] + secs[5][1] / 2, "⑥ Output", PURPLE)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dpi = 180

    print("Generating fig1_research_pipeline …")
    f1 = fig_pipeline()
    f1.savefig("figures/fig1_research_pipeline.png", dpi=dpi,
               bbox_inches="tight", facecolor=BG)
    f1.savefig("figures/fig1_research_pipeline.pdf",
               bbox_inches="tight", facecolor=BG)
    plt.close(f1)
    print("  → figures/fig1_research_pipeline.{png,pdf}")

    print("Generating fig2_training_loop …")
    f2 = fig_training_loop()
    f2.savefig("figures/fig2_training_loop.png", dpi=dpi,
               bbox_inches="tight", facecolor=BG)
    f2.savefig("figures/fig2_training_loop.pdf",
               bbox_inches="tight", facecolor=BG)
    plt.close(f2)
    print("  → figures/fig2_training_loop.{png,pdf}")

    print("Generating fig3_architecture …")
    f3 = fig_architecture()
    f3.savefig("figures/fig3_architecture.png", dpi=dpi,
               bbox_inches="tight", facecolor=BG)
    f3.savefig("figures/fig3_architecture.pdf",
               bbox_inches="tight", facecolor=BG)
    plt.close(f3)
    print("  → figures/fig3_architecture.{png,pdf}")

    print("\nDone — 3 figures (PNG + PDF) saved to ./figures/")
