# Performance of PoPE in Vision Transformers for Brain Tumor Classification

## Hyperparameter Tuning

We ran grid search over the following hyperparameter Grid
- Learning Rate: [1e-4, 3e-4, 1e-3]
- Dropout:       [0.1, 0.2, 0.5]
- Patch Size:    [8, 16, 32]

And found the best setting to be:
- Learning Rate: 3e-4
- Dropout: 0.1
- Patch Size: 8

## Results 

Final model comparison on the **test split** uses **mean AUROC** as the primary metric (with accuracy as secondary support).

### Test ranking (mean AUROC)

1. **ResNet18 (scratch)** — AUROC mean: **0.998989**, Accuracy: **0.987692**
2. **DeiT-Small pretrained** — AUROC mean: **0.986074**, Accuracy: **0.876923**
3. **DeiT-Small + PoPE** — AUROC mean: **0.978867**, Accuracy: **0.796923**
4. **RoPE-ViT** — AUROC mean: **0.964530**, Accuracy: **0.852308**
5. **PoPE-ViT** — AUROC mean: **0.964402**, Accuracy: **0.867692**

### Interpretation

- In the current setup, **ResNet18 (scratch)** is the strongest overall model.
- **RoPE-ViT** and **PoPE-ViT** are very close on the primary metric (mean AUROC).
- **DeiT-Small pretrained** outperforms custom ViT variants, and **DeiT-Small + PoPE** is competitive but below the pretrained DeiT baseline.

### Artifacts

- Per-run detailed outputs: `results/eval_*.json`
- Presentation figures: `results/figures/`
