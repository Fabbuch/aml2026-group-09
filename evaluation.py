"""
Standalone evaluation script for brain tumour classification models.

Usage examples:
  python evaluation.py --model resnet18_scratch --checkpoint checkpoints/cnn_resnet18_scratch_best.pt --split test
  python evaluation.py --model rope_vit --checkpoint checkpoints/rope_vit_best.pt --split test
  python evaluation.py --model popevit --checkpoint checkpoints/popevit_best.pt --split dev
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2

from BrainTumorDatasetClass import BrainTumorDataset
from model import PoPEViT, RoPEViT
from model import _build_deit_small_pope, _build_deit_small_pretrained


CLASS_NAMES = [
    "no_tumor",
    "meningioma_tumor",
    "glioma_tumor",
    "pituitary_tumor",
]


class EnsureRGB:
    """Convert any image tensor to exactly 3 channels."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        channels = img.shape[0]
        if channels == 1:
            return img.repeat(3, 1, 1)
        if channels == 4:
            return img[:3]
        return img

    def __repr__(self) -> str:
        return "EnsureRGB()"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on brain tumour dataset splits.")
    parser.add_argument(
        "--model",
        choices=[
            "resnet18_scratch",
            "rope_vit",
            "pope_vit",
            "popevit",
            "deit_small_pretrained",
            "deit_small_pope",
        ],
        required=True,
        help="Model architecture to instantiate.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.pt). Supports raw state_dict or dict with 'model_state_dict'.",
    )
    parser.add_argument(
        "--data-dir",
        default="Brain-Tumor-Classification-DataSet",
        help="Dataset root containing train/dev|val/test folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "val", "test"],
        default="test",
        help="Split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--norm-from-split", default="train", help="Split used to compute normalization stats.")
    parser.add_argument("--unweighted-loss", action="store_true", help="Use unweighted CE loss.")
    parser.add_argument("--output-dir", default="results", help="Directory where evaluation outputs are saved.")
    parser.add_argument("--no-save", action="store_true", help="Disable writing JSON/CSV result files.")
    return parser.parse_args()


def resolve_split(data_dir: str, requested_split: str) -> str:
    if requested_split in ("dev", "val"):
        if os.path.isdir(os.path.join(data_dir, requested_split)):
            return requested_split
        alternative = "val" if requested_split == "dev" else "dev"
        if os.path.isdir(os.path.join(data_dir, alternative)):
            return alternative
        raise FileNotFoundError(f"Neither '{requested_split}' nor '{alternative}' exists under {data_dir}")

    split_path = os.path.join(data_dir, requested_split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")
    return requested_split


def build_eval_transforms(image_size: int, img_mean: List[float], img_std: List[float]) -> v2.Compose:
    steps = [
        v2.Resize((image_size, image_size), antialias=True),
        EnsureRGB(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std),
    ]
    return v2.Compose(steps)


def build_model(model_name: str, image_size: int, num_classes: int = 4) -> nn.Module:
    if model_name == "popevit":
        model_name = "pope_vit"
    if model_name == "resnet18_scratch":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, num_classes))
        return model
    if model_name == "rope_vit":
        return RoPEViT(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
        )
    if model_name == "pope_vit":
        return PoPEViT(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
        )
    if model_name == "deit_small_pretrained":
        class _Cfg:
            dropout = 0.1

        return _build_deit_small_pretrained(_Cfg(), num_classes=num_classes)
    if model_name == "deit_small_pope":
        class _Cfg:
            dropout = 0.1

        return _build_deit_small_pope(_Cfg(), num_classes=num_classes)
    raise ValueError(f"Unsupported model '{model_name}'")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    y_true = []
    y_prob = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (preds == labels).sum().item()

        y_true.append(labels.cpu().numpy())
        y_prob.append(probs.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_prob_np = np.concatenate(y_prob)
    y_pred_np = np.concatenate(y_pred)

    y_true_one_hot = np.eye(len(CLASS_NAMES))[y_true_np]
    auroc_per_class = {}
    for idx, cls_name in enumerate(CLASS_NAMES):
        try:
            auroc_per_class[cls_name] = float(roc_auc_score(y_true_one_hot[:, idx], y_prob_np[:, idx]))
        except ValueError:
            auroc_per_class[cls_name] = float("nan")

    auroc_mean = float(np.nanmean(list(auroc_per_class.values())))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(CLASS_NAMES))))

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "auroc": {**auroc_per_class, "mean": auroc_mean},
        "confusion_matrix": cm,
    }


def compute_norm_constants(data_dir: str, split: str, image_size: int) -> Tuple[List[float], List[float]]:
    split = resolve_split(data_dir, split)
    ds = BrainTumorDataset(
        root=os.path.join(data_dir, split),
        transform=v2.Compose(
            [
                v2.Resize((image_size, image_size), antialias=True),
                EnsureRGB(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        variants_per_image=1,
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for image, _ in ds:
        for channel in range(3):
            mean[channel] += image[channel].mean()
            std[channel] += image[channel].std()
    mean /= len(ds)
    std /= len(ds)
    return mean.tolist(), std.tolist()


def compute_class_weights(data_dir: str) -> torch.Tensor:
    ds = BrainTumorDataset(root=os.path.join(data_dir, "train"), transform=None, variants_per_image=1)
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for _, label_idx, _ in ds.samples:
        label_map = {
            "no_tumor": 0,
            "meningioma_tumor": 1,
            "glioma_tumor": 2,
            "pituitary_tumor": 3,
        }
        counts[label_map[label_idx]] += 1
    class_w = counts.sum() / (len(CLASS_NAMES) * np.clip(counts, a_min=1.0, a_max=None))
    class_w = class_w / class_w.mean()
    return torch.tensor(class_w, dtype=torch.float32)


def print_metrics(metrics: Dict[str, object]) -> None:
    print("\n=== Evaluation Results ===")
    print(f"Loss     : {metrics['loss']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print("AUROC:")
    for cls_name in CLASS_NAMES:
        value = metrics["auroc"][cls_name]
        if np.isnan(value):
            print(f"  - {cls_name:18s}: nan")
        else:
            print(f"  - {cls_name:18s}: {value:.4f}")
    print(f"  - {'mean':18s}: {metrics['auroc']['mean']:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(metrics["confusion_matrix"])


def save_results(
    args: argparse.Namespace,
    split: str,
    dataset_size: int,
    device: torch.device,
    img_mean: List[float],
    img_std: List[float],
    metrics: Dict[str, object],
) -> Tuple[str, str]:
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_")
    split_slug = split.replace("/", "_")

    json_path = os.path.join(args.output_dir, f"eval_{model_slug}_{split_slug}_{timestamp}.json")
    csv_path = os.path.join(args.output_dir, "eval_summary.csv")

    payload = {
        "timestamp": timestamp,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": split,
        "samples": dataset_size,
        "image_size": args.image_size,
        "norm_from_split": args.norm_from_split,
        "norm_mean": img_mean,
        "norm_std": img_std,
        "loss_type": "unweighted_ce" if args.unweighted_loss else "weighted_ce_train_class_weights",
        "device": str(device),
        "metrics": {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "auroc": metrics["auroc"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    row = {
        "timestamp": timestamp,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": split,
        "samples": dataset_size,
        "image_size": args.image_size,
        "norm_from_split": args.norm_from_split,
        "loss_type": "unweighted_ce" if args.unweighted_loss else "weighted_ce_train_class_weights",
        "loss": round(float(metrics["loss"]), 6),
        "accuracy": round(float(metrics["accuracy"]), 6),
        "auroc_no_tumor": round(float(metrics["auroc"]["no_tumor"]), 6),
        "auroc_meningioma_tumor": round(float(metrics["auroc"]["meningioma_tumor"]), 6),
        "auroc_glioma_tumor": round(float(metrics["auroc"]["glioma_tumor"]), 6),
        "auroc_pituitary_tumor": round(float(metrics["auroc"]["pituitary_tumor"]), 6),
        "auroc_mean": round(float(metrics["auroc"]["mean"]), 6),
        "json_path": json_path,
    }

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return json_path, csv_path


def main() -> None:
    args = parse_args()
    split = resolve_split(args.data_dir, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_mean, img_std = compute_norm_constants(args.data_dir, args.norm_from_split, args.image_size)
    transform = build_eval_transforms(image_size=args.image_size, img_mean=img_mean, img_std=img_std)
    dataset = BrainTumorDataset(
        root=os.path.join(args.data_dir, split),
        transform=transform,
        variants_per_image=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, image_size=args.image_size, num_classes=len(CLASS_NAMES)).to(device)
    load_checkpoint(model, args.checkpoint, device)
    if args.unweighted_loss:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(args.data_dir).to(device))
    metrics = evaluate(model, loader, device, criterion)

    print(f"Model      : {args.model}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {split}")
    print(f"Image size : {args.image_size}")
    print(f"Samples    : {len(dataset)}")
    print(f"Device     : {device}")
    print(f"Norm split : {args.norm_from_split}")
    print(f"Norm mean  : {[round(v, 6) for v in img_mean]}")
    print(f"Norm std   : {[round(v, 6) for v in img_std]}")
    print(f"Loss type  : {'unweighted CE' if args.unweighted_loss else 'weighted CE (train class weights)'}")
    print_metrics(metrics)

    if not args.no_save:
        json_path, csv_path = save_results(
            args=args,
            split=split,
            dataset_size=len(dataset),
            device=device,
            img_mean=img_mean,
            img_std=img_std,
            metrics=metrics,
        )
        print(f"\nSaved JSON : {json_path}")
        print(f"Saved CSV  : {csv_path}")


if __name__ == "__main__":
    main()
