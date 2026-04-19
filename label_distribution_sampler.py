
import argparse
import json
import os
import random
from datetime import datetime

from sklearn.metrics import roc_auc_score


CLASSES = ["no_tumor", "meningioma_tumor", "glioma_tumor", "pituitary_tumor"]


def count_files(folder):
    return sum(1 for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))

# Compute class counts and class priors from the train split
def get_train_priors(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    counts = {cls: count_files(os.path.join(train_root, cls)) for cls in CLASSES}
    total = sum(counts.values())
    priors = {cls: counts[cls] / total for cls in CLASSES}
    return counts, priors

# Load true labels for a split using the fixed class-to-index order
def load_true_labels(dataset_root, split):
    split_root = os.path.join(dataset_root, split)
    y_true = []
    for i, cls in enumerate(CLASSES):
        n = count_files(os.path.join(split_root, cls))
        y_true.extend([i] * n)
    return y_true

# Build baseline scores either from fixed priors or weighted random sampling
def build_scores(n_samples, priors, mode, seed):
    probs = [priors[c] for c in CLASSES]
    if mode == "prior":
        return [probs[:] for _ in range(n_samples)]

    random.seed(seed)
    sampled = random.choices(range(len(CLASSES)), weights=probs, k=n_samples)
    scores = [[0.0] * len(CLASSES) for _ in range(n_samples)]
    for i, cls_idx in enumerate(sampled):
        scores[i][cls_idx] = 1.0
    return scores

# Compute one-vs-rest AUROC per class and macro average AUROC
def compute_multiclass_auroc(y_true, scores):
    per_class = {}
    for i, cls in enumerate(CLASSES):
        y_bin = [1 if y == i else 0 for y in y_true]
        s_bin = [row[i] for row in scores]
        per_class[cls] = roc_auc_score(y_bin, s_bin)
    avg = sum(per_class.values()) / len(CLASSES)
    return per_class, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="Brain-Tumor-Classification-DataSet")
    parser.add_argument("--eval-splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--prediction-mode", choices=["sample", "prior"], default="sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    counts, priors = get_train_priors(args.dataset_root)
    print("Train counts and priors:")
    for cls in CLASSES:
        print(f"  {cls}: count={counts[cls]}, prior={priors[cls]:.4f}")

    run_results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": args.dataset_root,
        "prediction_mode": args.prediction_mode,
        "seed": args.seed,
        "classes": CLASSES,
        "train_counts": counts,
        "train_priors": priors,
        "splits": {},
    }

    for split in args.eval_splits:
        y_true = load_true_labels(args.dataset_root, split)
        scores = build_scores(len(y_true), priors, args.prediction_mode, args.seed)
        per_class, avg = compute_multiclass_auroc(y_true, scores)

        print(f"\n[{split}] mode={args.prediction_mode}, samples={len(y_true)}")
        for cls in CLASSES:
            print(f"  auroc_{cls}={per_class[cls]:.4f}")
        print(f"  auroc_average={avg:.4f}")

        run_results["splits"][split] = {
            "samples": len(y_true),
            "auroc_per_class": per_class,
            "auroc_average": avg,
        }

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(run_results, f, indent=2)
        print(f"\nSaved results to: {args.output_file}")


if __name__ == "__main__":
    main()
