"""
Standalone evaluation entrypoint reusing training_pipeline_func utilities.

Usage examples:
  python3 evaluation.py --model rope_vit --checkpoint checkpoints/rope_vit_best.pt --split test
  python3 evaluation.py --model pope_vit --checkpoint checkpoints/pope_vit_best.pt --split dev
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from BrainTumorDatasetClass import BrainTumorDataset
from training_pipeline_func import *  # noqa: F401,F403
import training_pipeline_func as tpf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument(
        "--model",
        choices=[
            "resnet18_scratch",
            "rope_vit",
            "pope_vit",
            "popevit",
            "deit_small_pretrained",
            "deit_small_pope",
            "deit_small_rope",
        ],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--data-dir", default="Brain-Tumor-Classification-DataSet")
    parser.add_argument("--split", choices=["train", "dev", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--norm-from-split", default="train")
    parser.add_argument("--unweighted-loss", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-save", action="store_true")
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


def compute_class_weights(data_dir: str) -> torch.Tensor:
    raw_train = BrainTumorDataset(root=os.path.join(data_dir, "train"), transform=None, variants_per_image=1)
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, label_str, _ in raw_train.samples:
        counts[tpf._LABEL_MAP[label_str]] += 1
    class_w = counts.sum() / (NUM_CLASSES * counts.clip(min=1))
    class_w = class_w / class_w.mean()
    return torch.tensor(class_w, dtype=torch.float32).to(DEVICE)


def print_metrics(metrics: Dict[str, object], cm: np.ndarray) -> None:
    print("\n=== Evaluation Results ===")
    print(f"Loss     : {metrics['loss']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print("AUROC:")
    for cls in CLASS_NAMES:
        print(f"  - {cls:18s}: {metrics['auroc'][cls]:.4f}")
    print(f"  - {'mean':18s}: {metrics['auroc']['mean']:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)


def save_results(
    args: argparse.Namespace,
    split: str,
    dataset_size: int,
    img_mean: List[float],
    img_std: List[float],
    metrics: Dict[str, object],
    cm: np.ndarray,
) -> Tuple[str, str]:
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_")
    json_path = os.path.join(args.output_dir, f"eval_{model_slug}_{split}_{timestamp}.json")
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
        "device": str(DEVICE),
        "metrics": {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "auroc": metrics["auroc"],
            "confusion_matrix": cm.tolist(),
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
    set_seed(42)
    split = resolve_split(args.data_dir, args.split)
    model_name = "pope_vit" if args.model == "popevit" else args.model

    cfg = TrainingConfig(
        model_name=model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout=0.1,
        patch_size=16,
        vit_dim=512,
        vit_depth=6,
        vit_heads=8,
        vit_mlp_dim=1024,
    )
    cfg.data_dir = args.data_dir

    img_mean, img_std = tpf._get_normalization_constants(cfg, args.norm_from_split)
    eval_transform = get_val_transforms(cfg, img_mean, img_std)
    dataset = BrainTumorDataset(
        root=os.path.join(args.data_dir, split),
        transform=eval_transform,
        variants_per_image=1,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint, DEVICE)
    criterion = nn.CrossEntropyLoss() if args.unweighted_loss else nn.CrossEntropyLoss(weight=compute_class_weights(args.data_dir))
    metrics = evaluate(model, loader, criterion, DEVICE)

    labels = metrics["labels"]
    probs = metrics["probs"]
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    print(f"Model      : {model_name}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {split}")
    print(f"Image size : {cfg.image_size}")
    print(f"Samples    : {len(dataset)}")
    print(f"Device     : {DEVICE}")
    print(f"Norm split : {args.norm_from_split}")
    print(f"Norm mean  : {[round(v, 6) for v in img_mean]}")
    print(f"Norm std   : {[round(v, 6) for v in img_std]}")
    print(f"Loss type  : {'unweighted CE' if args.unweighted_loss else 'weighted CE (train class weights)'}")
    print_metrics(metrics, cm)

    if not args.no_save:
        json_path, csv_path = save_results(args, split, len(dataset), img_mean, img_std, metrics, cm)
        print(f"\nSaved JSON : {json_path}")
        print(f"Saved CSV  : {csv_path}")


if __name__ == "__main__":
    main()
