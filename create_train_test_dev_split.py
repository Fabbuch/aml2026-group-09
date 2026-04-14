#!/usr/bin/env python
import os
import sys
from random import shuffle

def clear_train_test_split(root):
    root = os.path.abspath(root)
    
    split_dirs = ["Training", "Testing"]
    labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

    # Create a directory for each label on the top-level of the dataset directory
    for label in labels:
        os.makedirs(os.path.join(root, label), exist_ok=True)

    # Move files from each split into the right label directory
    # They are also renamed with indices for each label to avoid name collisions
    indices = {label: 0 for label in labels}
    for split in split_dirs:
        for label in labels:
            # Move all files from the current split and label to the top-level label directory
            src_dir = os.path.join(root, split, label)
            dst_dir = os.path.join(root, label)
            for filename in os.listdir(src_dir):
                i = indices[label]
                indices[label] += 1
                os.rename(os.path.join(src_dir, filename), os.path.join(dst_dir, f"{i:03d}_{filename}"))

    # Remove empty directories
    for split in split_dirs:
        for label in labels:
            # Remove the directory itself after moving all its contents
            rmdir_path = os.path.join(root, split, label)
            if not os.listdir(rmdir_path):
                os.rmdir(rmdir_path)
        rmdir_path = os.path.join(root, split)
        if not os.listdir(rmdir_path):
            os.rmdir(rmdir_path)


def split_train_test_dev(root, train_split, test_split, dev_split):
    root = os.path.abspath(root)

    # Subdirectories for each label
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    train_path = os.path.join(root, 'train')
    test_path = os.path.join(root, 'test')
    dev_path = os.path.join(root, 'dev')
    
    # Create the train, test and dev directories if they don't exist
    for path in [train_path, test_path, dev_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        for label in labels:
            label_path = os.path.join(path, label)
            if not os.path.exists(label_path):
                os.makedirs(label_path)
    
    # Collect the image paths for each label
    for label in labels:
        label_path = os.path.join(root, label)
        image_paths = [os.path.join(label_path, f) for f in os.listdir(label_path)]

        # Distribute the images for each label according to the specified splits
        # This ensures that the label distribution is the same across all splits
        shuffle(image_paths)
        n_images = len(image_paths)
        n_train = int(n_images * int(train_split) / 100)
        n_test = int(n_images * int(test_split) / 100)
    
        # Move the images to the corresponding directories
        for i, path in enumerate(image_paths):
            if i < n_train:
                os.rename(path, os.path.join(train_path, label, os.path.basename(path)))
            elif i < n_train + n_test:
                os.rename(path, os.path.join(test_path, label, os.path.basename(path)))
            else:
                os.rename(path, os.path.join(dev_path, label, os.path.basename(path)))
            
    # Remove the top-level label directories if they are empty
    for label in labels:
        label_path = os.path.join(root, label)
        if not os.listdir(label_path):
            os.rmdir(label_path)

if __name__ == '__main__':
    ## Handle command line arguments
    assert len(sys.argv) == 4, 'Usage: ./apply_train_test_dev_split.py train_split test_split dev_split (e.g. 80 10 10)'
    train_split = sys.argv[1]
    test_split = sys.argv[2]
    dev_split = sys.argv[3]
    assert train_split.isalnum() and test_split.isalnum() and dev_split.isalnum(),  'Usage: ./apply_train_test_dev_split.py train_split test_split dev_split (e.g. 80 10 10)'
    assert int(train_split) + int(test_split) + int(dev_split) == 100, 'Splits have to add up to 100'
    
    # clear existing train/test split
    clear_train_test_split('Brain-Tumor-Classification-DataSet')
    # apply new train/test/dev split
    split_train_test_dev('Brain-Tumor-Classification-DataSet', train_split, test_split, dev_split)
    
