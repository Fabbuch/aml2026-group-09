# Project Proposal Group 09

## Problem Setting

The problem is a multi-class brain-tumor MRI classification task. This means that, for an MRI image $I$ the goal is to predict a label $y \in$ {glioma, meningioma, pituitary, no tumor}.

## Dataset

The dataset we use for training is available [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data) on Kaggle. It consists of brain MRI images of four types glioma, meningioma, pituitary tumor and no tumor.

## Proposed Approach

Our approach is a vision transformer encoder, that uses [PoPE](https://github.com/lucidrains/PoPE-pytorch?tab=readme-ov-file) (polar coordinate positional embeddings) in place of the usual RoPE in a vision transformer. PoPE is intended to disentangle position and content information in the embeddings, so therefore we expect improvements on a tumor classification task, where both position and content may both carry important information independently.

The vision transformer's encoder states will be fed into a MLP in order to map it down to the output dimension (4, as we are predicting 4 classes).

## Evaluation Protocol

### Baselines

Our machine learning baseline is a vanilla vision transformer with RoPE as positional emeddings. In addition to this, we use sampling from the label distribution as a simple statistical baseline.

### Data Split

The dataset already has a train-test split, where ca. 12% of the data is used for testing and 86% for training. We will use the same split in our evaluation for both the basline and the proposed model.

### Metrics

We will calculate AUROC for each cancer type and use the average as our main metric, since AUROC balances the true positive rate with the false positive rate. We specifically want a model that reaches a high true positive rate (sensitivity).

### Hyperparameter Tuning

There are two hyperparameters that we will tune for 1) the patch size of the transformer and 2) the dropout percentage as well as 3) the learning rate of the transformer. We will find optimal values for these hyperparameters through grid search.
