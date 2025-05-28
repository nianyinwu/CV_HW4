""" Dataloader utilities for training, validation, and testing. """

import os
from argparse import Namespace
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

def data_transform(mode):
    '''
    Transform for training and validation datasets.
    '''

    if mode == 'train':
        return v2.Compose([
            v2.ToImage(),
            v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.4),
            v2.RandomHorizontalFlip(p=0.4),
            v2.RandomVerticalFlip(p=0.4),
            v2.RandomApply([
                v2.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                )
            ], p=0.4),
            v2.ToDtype(torch.float32, scale=True)
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

class RestorationDataset(Dataset):
    """
    Restoration Dataset
    """
    def __init__(self, root, mode, transform=None):
        self.mode = mode
        self.transform = transform

        if self.mode == 'test':
            self.image_dir = os.path.join(root, 'degraded')
            self.images = sorted(os.listdir(self.image_dir))
        else:
            degraded_dir = os.path.join(root, 'degraded')
            clean_dir = os.path.join(root, 'clean')

            self.pairs = []
            for filename in sorted(os.listdir(degraded_dir)):
                if filename.startswith('rain-'):
                    clean_name = filename.replace('rain-', 'rain_clean-')
                elif filename.startswith('snow-'):
                    clean_name = filename.replace('snow-', 'snow_clean-')
                else:
                    continue
                self.pairs.append((
                    os.path.join(degraded_dir, filename),
                    os.path.join(clean_dir, clean_name)
                ))

    def __len__(self):
        if self.mode == 'test':
            return len(self.images)
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.mode == 'test':
            fname = self.images[idx]
            path = os.path.join(self.image_dir, fname)
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return fname, image

        degraded_path, clean_path = self.pairs[idx]
        degraded = Image.open(degraded_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        if self.transform:
            degraded, clean = self.transform(degraded, clean)

        return degraded, clean


def dataloader(args: Namespace, mode: str) -> DataLoader:
    """
    Create dataloader based on the mode: train, val, or test.

    Args:
        args (Namespace): Command-line arguments containing data_path and batch_size.
        mode (str): Mode of the data loader ('train', 'val', 'test').

    Returns:
        DataLoader: PyTorch DataLoader for the corresponding dataset.
    """

    dataset = None
    shuffle = False
    transform = data_transform(mode)

    if mode in ['train', 'valid']:
        data_path = os.path.join(args.data_path, mode)
        dataset = RestorationDataset(data_path, mode, transform=transform)
        if mode == 'train':
            shuffle = True
    elif mode == 'test':
        data_path = os.path.join(args.data_path, mode)
        dataset = RestorationDataset(data_path, mode, transform=transform)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=6,
        pin_memory=True,
    )
