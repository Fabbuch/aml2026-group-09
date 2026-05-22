"""Runs Grid Search using the pipeline from training_pipeline_func.py. This script serves as an entrypoint to be run from Renku."""

from training_pipeline_func import *

import zipfile
import os

def unzip_dataset(zip_file: str, output_dir: str):
    """Unzips the dataset to the local session storage of Renku.
    This avoids large latency when reading from the connected switchdrive storage. 
    
    Args:
        zip_file: Path to the zip file containing the dataset.
        output_dir: Path to the directory where the dataset should be unzipped.
    """
    if os.path.exists(output_dir):
        print(f"Dataset already at {output_dir}, skipping unzipping.")
    else:
        print("Unzipping dataset to the local session...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted dataset to {output_dir}")

if __name__ == '__main__':
    # Unzip the dataset to the local session storage of Renku:
    unzip_dataset(
        zip_file = '/home/renku/work/aml2026-group-09/Brain-Tumor-Classification-DataSet_copy.zip',
        output_dir = '/home/renku/work/brain-tumor-classification-dataset_copy',
    )
    
    # Load a default configuration for a pope_vit model. This uses only 5 epochs to speed up the grid search
    base_cfg = TrainingConfig(
        model_name    = 'pope_vit',
        vit_dim       = 512,
        vit_depth     = 6,
        vit_heads     = 8,
        vit_mlp_dim   = 1024,
        epochs        = 5,
        batch_size    = 64,
        warmup_epochs = 1,
        early_stopping_patience = 2,
        seed          = 42,
        # Renku specific paths for the data and checkpoints:
        data_dir = './../brain-tumor-classification-dataset_copy/Brain-Tumor-Classification-DataSet_copy',
        checkpoint_dir = './../persistent-storage/checkpoints',
    )

    # Run the grid search over all combinations of learning_rate, dropout and patch_size.
    # This saves intermediate results after each run, and the final sorted results at the end.
    gs_results = run_grid_search(
        param_grid = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'dropout':       [0.1, 0.2, 0.5],
            'patch_size':    [8, 16, 32],
        },
        base_cfg = base_cfg,
    )
    # Save final results:
    torch.save(gs_results, os.path.join(base_cfg.checkpoint_dir, f'{base_cfg.model_name}_grid_search_results.pt'))