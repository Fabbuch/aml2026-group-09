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