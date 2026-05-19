#!/bin/env bash
# This script is used to copy the dataset to the local Renku session to avoid latency issues.

# Check if the dataset is already copied to the local session
if [ -d "/home/renku/work/brain-tumor-classification-dataset_copy" ]; then
    echo "Dataset already copied to the local session."
else
# Copy the dataset to ./brain-tumor-classification-dataset_copy
    echo "Copying dataset to the local session..."
    cp -r /home/renku/work/brain-tumor-classification-dataset /home/renku/work/brain-tumor-classification-dataset_copy
    echo "Dataset copied to the local session."
fi