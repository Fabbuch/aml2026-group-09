#!/bin/env bash
# This script is used to unzip a compressed copy of the dataset included in this repository and write it to the
# local session of Renku. This ensures that the dataset can be loaded with lower latency during training.

# Check if the dataset is already copied to the local session
if [ -d "/home/renku/work/brain-tumor-classification-dataset_copy" ]; then
    echo "Dataset already copied to the local session."
else
# Unzip the dataset to ./brain-tumor-classification-dataset_copy
    echo "Unzipping dataset to the local session..."
    unzip /home/renku/work/aml2026-group-09/Brain-Tumor-Classification-DataSet_copy.zip -d /home/renku/work/brain-tumor-classification-dataset_copy
    echo "Dataset unzipped to the local session."
fi