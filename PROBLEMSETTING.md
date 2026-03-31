# Project Proposal Group 09

## Problem Setting

The problem is a multi-class english hate speech detection task. This means that, for an english text $t_i$ the goal is to predict a label $y \in {normal, hate, offensive}$. 

## Dataset

We use the dataset from [Toraman et al. 2022](https://github.com/metunlp/hate-speech) available on Github. The dataset consists of 68,597 english and 60,310 turkish tweets with human annotations of hate/offensive/normal labels.

## Proposed Approach

We use a data augmentation approach for this task. We use machine translation to translate the turkish tweets to english in order to enhance the minority classes ("hate" and "offensive"). This will yield a final label distribution of 56% normal, 29% offensive and 15% hate.

## Evaluation Protocol

We evaluate our proposed model using hold-out testing with a 80-10-10 split of training-test-validation data. Over the test set, we calculate precision, recall and F1-score for every class and average them to obtain overall evaluation metrics.

We use the results from the best scoring model as reported in the paper (a MegatronLM model trained on the english part of the dataset) as a baseline. We also compare our results to sampling from the label distribution as a statistical baseline.

## Model Architecture

Our architecture consists of a [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large) encoder model with a linear classifier head and fine-tune it on our augmented english dataset.