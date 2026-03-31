# Project Proposal Group 09

## Problem Setting

The problem is a multi-class brain-tumor MRI classification task. This means that, for an MRI image $I$ the goal is to predict a label $y \in$ {glioma, meningioma, pituitary, no tumor}.

## Dataset

The dataset we use for training is available [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data) on Kaggle. It consists of brain MRI images of four types glioma, meningioma, pituitary tumor and no tumor. The dataset has a predefined train-test split, where ca. 12% of the data is used for testing and 86% for training. We will use the same split in our evaluation.

## Proposed Approach

Our approach uses [PoPE](https://github.com/lucidrains/PoPE-pytorch?tab=readme-ov-file) (polar coordinate positional embeddings) in place of the usual RoPE in a vision transformer. PoPE is intended to disentangle position and content information in the embeddings, so therefore we expect improvements on a tumor classification task, where both position and content may both carry important information independently.

## Evaluation Protocol

We will use a vanilla vision transformer as our baseline. We will also sample from the label distribution as a statistical baseline. We will calculate AUROC for each cancer type and use the average as our main metric, since AUROC balances the true positive rate with the false positive rate. We specifically want a model that reaches a high true positive rate (sensitivity).

We will tune hyperparameters like patch size and dropout percentage using grid search to find the best performing values.
