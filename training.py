import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
# Custom class
from BrainTumorDatasetClass import BrainTumorDataset

# set torch random seed
torch.manual_seed(42)

# define transforms for data augmentation and preprocessing
transforms = v2.Compose([
    ## Preprocessing ###
    # resize image to 256x256
    v2.Resize((256, 256), antialias=True),
    
    ## Data Augmentation ##
    v2.RandomAffine(
        # random rotation
        degrees=15,
        # random translation
        translate=(0.05, 0.05),
        # random rescaling
        scale=(0.95, 1.05),
        # random shear
        shear=0.05
        ),
    # random horizontal or vertical flip
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    
    ## Further Preprocessing ##
    # Convert to float tensor
    v2.ToDtype(torch.float32, scale=True)
])

# Construct dataset from train data with data augmentation using 10 variants per image
train_dataset = BrainTumorDataset(
    "Brain-Tumor-Classification-DataSet/train",
    transform=transforms,
    variants_per_image=10
)
print(f"Number of samples in the training dataset: {len(train_dataset)}")

# Create dataloader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

# Simulate training loop by showing a single batch
for images, labels in train_loader:
    print(f"Batch of images, shape: {images.shape}")
    print(f"Batch of labels: {labels}")
    # Show the first three images in the batch (with transforms applied)
    v2.ToPILImage()(images[0]).show()
    v2.ToPILImage()(images[1]).show()
    v2.ToPILImage()(images[2]).show()
    break