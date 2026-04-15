import os
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import decode_image
    
class BrainTumorDataset(VisionDataset):
    """Custom VisionDataset subclass for a Brain Tumor Classification Dataset. The specified directory (root)
    should contain four subdirectories, each corresponding to one of these labels:
    - no_tumor
    - meningioma_tumor
    - glioma_tumor
    - pituitary_tumor
    Each subdirectory should contain the corresponding images for that label.
    
    This class also applies data augmentation to increase the effective size of the dataset. This is achieved
    by applying random transforms to each image to produce multiple variants. The resulting dataset size is 
    defined as the number of images in the original dataset multiplied by the number of variants per image
    (variants_per_image). E.g. a dataset with 1000 images and variants_per_image=10 will produce a dataset
    of 10000 samples.
    """
    def __init__(self, root, transform=None, variants_per_image=10):
        super().__init__(root, transform=transform)
        # Each sample will be represented as a tuple (image_path, label, seed)
        self.samples = []
        # There will be <variants_per_image> samples for each image in the original dataset
        self.variants_per_image = variants_per_image
        for label in ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']:
            for image in os.listdir(os.path.join(root, label)):
                # This generates a unique seed for each sample
                # The seed will later be used when applying the transforms
                for _ in range(self.variants_per_image):
                    seed = torch.randint(0, 2147483647, (1,)).item()
                    self.samples.append((os.path.join(root, label, image), label, seed))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, seed = self.samples[idx]
        img = decode_image(path)
        
        # The random transforms use the sample's seed. This makes sure each sample is the same, each time
        # the dataset is iterated over
        torch.manual_seed(seed)
        if self.transform:
            img = self.transform(img)
        label_map = {
            "no_tumor": 0,
            "meningioma_tumor": 1,
            "glioma_tumor": 2,
            "pituitary_tumor": 3
        }
        label = label_map[label]
        return img, label
