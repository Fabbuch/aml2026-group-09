"""
Generate the two figures needed for the README:
  1. figures/fig_dataset_classes.png   — 4-class MRI sample panel
  2. figures/fig_data_pipeline.png     — split overview + augmentation variants
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as v2
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "Brain-Tumor-Classification-DataSet" / "train"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

CLASSES = {
    "glioma_tumor":      "Glioma",
    "meningioma_tumor":  "Meningioma",
    "pituitary_tumor":   "Pituitary Tumor",
    "no_tumor":          "No Tumor",
}

COLORS = {
    "Glioma":          "#e07070",
    "Meningioma":      "#70a0d4",
    "Pituitary Tumor": "#80c080",
    "No Tumor":        "#c0a0d0",
}

random.seed(42)


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def pick_image(class_dir: Path, seed_offset: int = 0) -> Path:
    imgs = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
    random.seed(42 + seed_offset)
    return random.choice(imgs)


# ── Figure 1: 4-class MRI sample panel ────────────────────────────────────────

def make_class_panel():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.patch.set_facecolor("#111111")

    for ax, (folder, label) in zip(axes, CLASSES.items()):
        path = pick_image(DATA_DIR / folder)
        img  = load_rgb(path)
        ax.imshow(img, cmap="gray")
        ax.set_title(label, color="white", fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")
        # coloured border
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS[label])
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.tight_layout(pad=0.5)
    out = FIG_DIR / "fig_dataset_classes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: Data-pipeline overview  ─────────────────────────────────────────

def make_pipeline_figure():
    # ---- augmentation transforms (same as training pipeline) ----
    aug_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.RandomAffine(degrees=15, translate=(0.05, 0.05),
                        scale=(0.95, 1.05), shear=5),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # pick one glioma image for the augmentation strip
    src_path = pick_image(DATA_DIR / "glioma_tumor", seed_offset=10)
    src_pil  = Image.open(src_path).convert("RGB").resize((224, 224))
    src_t    = v2.functional.to_image(src_pil)  # uint8 tensor

    random.seed(0)
    aug_variants = []
    torch.manual_seed(0)
    for _ in range(4):
        aug_variants.append(np.array(v2.functional.to_pil_image(aug_transform(src_t))))

    # ---- counts for split bar ----
    N_TOTAL = 3264
    N_TRAIN = int(N_TOTAL * 0.80)
    N_VAL   = int(N_TOTAL * 0.10)
    N_TEST  = N_TOTAL - N_TRAIN - N_VAL

    class_counts = {"Glioma": 926, "Meningioma": 937, "Pituitary Tumor": 932, "No Tumor": 469}

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#f8f8f8")

    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                              left=0.06, right=0.97, top=0.92, bottom=0.08)

    # ── panel A: split bar ──────────────────────────────────────
    ax_split = fig.add_subplot(outer[0, 0])
    ax_split.set_facecolor("white")

    bar_colors = ["#4a90d9", "#f5a623", "#e74c3c"]
    fracs      = [N_TRAIN / N_TOTAL, N_VAL / N_TOTAL, N_TEST / N_TOTAL]
    labels     = [f"Train  {N_TRAIN:,}  (80%)", f"Val  {N_VAL:,}  (10%)", f"Test  {N_TEST:,}  (10%)"]
    left = 0.0
    for frac, col, lbl in zip(fracs, bar_colors, labels):
        ax_split.barh(0, frac, left=left, color=col, height=0.4, label=lbl)
        cx = left + frac / 2
        ax_split.text(cx, 0, f"{frac*100:.0f}%", ha="center", va="center",
                      color="white", fontsize=11, fontweight="bold")
        left += frac

    ax_split.set_xlim(0, 1)
    ax_split.set_ylim(-0.5, 0.9)
    ax_split.axis("off")
    ax_split.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                    ncol=3, fontsize=9, frameon=False)
    ax_split.set_title("Stratified Train / Val / Test Split\n(label distribution identical across splits)",
                        fontsize=10, color="#333333")

    # ── panel B: class distribution bar chart ──────────────────
    ax_cls = fig.add_subplot(outer[0, 1])
    ax_cls.set_facecolor("white")
    cls_names = list(class_counts.keys())
    cls_vals  = list(class_counts.values())
    cls_cols  = [COLORS[n] for n in cls_names]
    bars = ax_cls.bar(cls_names, cls_vals, color=cls_cols, width=0.55,
                      edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, cls_vals):
        ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                    f"N={val}", ha="center", va="bottom", fontsize=9, color="#444444")
    ax_cls.set_ylim(0, 1100)
    ax_cls.set_ylabel("Image count", fontsize=9)
    ax_cls.set_title("Class Distribution (imbalanced → inverse-freq weights)",
                     fontsize=10, color="#333333")
    ax_cls.tick_params(axis="x", labelsize=9)
    ax_cls.spines[["top", "right"]].set_visible(False)

    # ── panel C: augmentation strip ────────────────────────────
    aug_outer = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1, :],
                                                 wspace=0.05)
    aug_titles = ["Original", "Aug variant 1\n(rotate + scale)",
                  "Aug variant 2\n(flip + translate)",
                  "Aug variant 3\n(rotate + shear)",
                  "Aug variant 4\n(flip + scale)"]
    aug_images = [np.array(src_pil)] + aug_variants

    for i, (img, title) in enumerate(zip(aug_images, aug_titles)):
        ax = fig.add_subplot(aug_outer[i])
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=8.5, color="#222222", pad=4)
        ax.axis("off")
        col = "#4a90d9" if i == 0 else "#f5a623"
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2 if i == 0 else 1.2)
            spine.set_visible(True)

    fig.text(0.5, 0.48, "Glioma — training-time augmentation (stochastic, VARIANTS_PER_IMAGE=4)",
             ha="center", fontsize=10, color="#555555", style="italic")

    out = FIG_DIR / "fig_data_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    make_class_panel()
    make_pipeline_figure()
    print("Done.")
