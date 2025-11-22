# dataset.py
print("Starting dataset.py...")
import os
print("Imported os in dataset")
from glob import glob
print("Imported glob in dataset")
from PIL import Image
print("Imported PIL in dataset")
import numpy as np
print("Imported numpy in dataset")
import torch
print("Imported torch in dataset")
from torch.utils.data import Dataset
print("Imported Dataset in dataset")
from torchvision import transforms
print("Imported torchvision.transforms in dataset")

IMG_SIZE = 224

def get_transforms(img_size=IMG_SIZE):
    """
    Returns (train_transform, val_transform).
    Uses torchvision transforms.
    """
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

class BrainMRIDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        # Keep as PIL Image for torchvision
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def make_file_lists(train_dir, test_dir):
    """
    Expects train_dir to have class subfolders and test_dir similarly.
    Returns: classes, train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    from sklearn.model_selection import train_test_split
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class2idx = {c:i for i,c in enumerate(classes)}

    all_paths = []
    all_labels = []
    for c in classes:
        folder = os.path.join(train_dir, c)
        imgs = glob(os.path.join(folder, "*"))
        imgs = [p for p in imgs if p.lower().endswith(('.png','.jpg','.jpeg'))]
        all_paths += imgs
        all_labels += [class2idx[c]] * len(imgs)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )

    test_paths = []
    test_labels = []
    for c in classes:
        folder = os.path.join(test_dir, c)
        imgs = glob(os.path.join(folder, "*"))
        imgs = [p for p in imgs if p.lower().endswith(('.png','.jpg','.jpeg'))]
        test_paths += imgs
        test_labels += [class2idx[c]] * len(imgs)

    return classes, train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
