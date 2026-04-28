import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import PoPEViT
from tqdm import tqdm

from training import TRANSFORMS
from BrainTumorDatasetClass import BrainTumorDataset

# set torch random seed
torch.manual_seed(0)

def load_model(patch_size: int, dropout: float) -> PoPEViT:
    """Utility function to load a PoPeViT model with the given hyperparameters for grid search."""
    return PoPEViT(
        # Default parameters
        image_size=256,
        num_classes=4,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        # Hyperparameters for grid search
        patch_size=patch_size,
        dropout=dropout
        )
    
def train(model: torch.nn.Module, lr: float, epochs: int = 10):
    # TODO: Add logging of training loss, validation loss, eval metrics etc.
    # TODO: Add early stopping based on validation metrics
    # TODO: Add cuda support
    # TODO: Save the model at reasonable intervals
    """General train function for a ViT model.
    
    Args:
        model: The model to train (vit_pytorch.Vit or PoPeViT).
        lr: Learning rate for the optimizer (Adam).
        epochs: Number of training epochs.
    """
    train_loader = get_split_dataloader("train", 10)
    optimizer = Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = batch
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

def get_split_dataloader(split: str, augmentation_factor: int) -> DataLoader:
    """Utility function that returns a dataloader for the given split with the given augmentation factor.

    Args:
        split (str): "train", "dev" or "test"
        augmentation_factor (int): Number of variants to generate per image for data augmentation.
    """
    dataset_split = BrainTumorDataset(
        f"Brain-Tumor-Classification-DataSet/{split}",
        transform=TRANSFORMS,
        variants_per_image=augmentation_factor
    )

    # Create dataloader
    dl = DataLoader(dataset_split, batch_size=512, shuffle=True)
    return dl

if __name__ == "__main__":
    patch_sizes = [8, 16, 32]
    lrs = [1e-3, 5e-3, 1e-2, 5e-2]
    dropouts = [0.0, 0.1]

    for patch_size in patch_sizes:
        for lr in lrs:
            for dropout in dropouts:
                # TODO: Add logging of the current hyperparameter combination
                model = load_model(patch_size, dropout)
                model = train(model, lr)