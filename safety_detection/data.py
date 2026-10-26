"""Dataset loading, preprocessing and train/validation splitting."""

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 224
# ImageNet statistics, matching the pretrained ViT-B/16 weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform).

    Augmentation is applied only at training time; evaluation and inference
    use a deterministic resize + normalize so results are reproducible.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def stratified_split(targets: List[int], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """Split sample indices so each class keeps the same proportion in both sets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for label in sorted(set(targets)):
        idx = [i for i, t in enumerate(targets) if t == label]
        idx = [idx[i] for i in torch.randperm(len(idx), generator=generator).tolist()]
        n_val = max(1, round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return sorted(train_idx), sorted(val_idx)


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    val_fraction: float = 0.3,
    seed: int = 42,
    num_workers: int = 2,
    image_size: int = IMAGE_SIZE,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build train/validation loaders from an ImageFolder-style directory.

    Expected layout::

        data_dir/
            Not Wearing Safety Gear/*.jpg
            Wearing Safety Gear/*.jpg
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_transform, eval_transform = build_transforms(image_size)
    # Two views over the same files so the validation split never sees augmentation.
    train_view = datasets.ImageFolder(data_dir, transform=train_transform)
    eval_view = datasets.ImageFolder(data_dir, transform=eval_transform)

    train_idx, val_idx = stratified_split(train_view.targets, val_fraction, seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(Subset(train_view, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(Subset(eval_view, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, train_view.classes
