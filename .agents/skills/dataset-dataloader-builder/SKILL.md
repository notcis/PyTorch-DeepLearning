---
name: dataset-dataloader-builder
description: Generates PyTorch train_dataloader, val_dataloader, test_dataloader, and class_names from a user-specified dataset path using torchvision, torch, and torch.utils.data with safe Dataset wrappers.
---

# PyTorch DataLoader & Dataset Builder Guidelines

When asked to construct DataLoaders from a dataset path, follow these standards:

1. **Path Handling**: Accept root path (either standard `train/val/test` directory structure or a unified `ImageFolder` path).
2. **Transform Decoupling**: If using `random_split`, always use a custom Dataset Wrapper (`DatasetFromSubset` / `CustomSubsetWrapper`) so training augmentations do NOT leak into validation/testing sets[cite: 3].
3. **DataLoaders Configuration**: 
   - `shuffle=True` for train; `shuffle=False` for val/test[cite: 3].
   - Configure `num_workers`, `pin_memory=True`, and `persistent_workers=True` when `num_workers > 0`[cite: 1].
4. **Metadata Extraction**: Return `class_names` via `dataset.classes` or folder names.

---

## Reference Implementation

```python
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


class DatasetFromSubset(Dataset):
    """
    Wrapper to apply distinct transforms on subsets derived from random_split.
    Prevents shared transform references across splits.
    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


def create_dataloaders(
    dataset_path: Union[str, Path],
    batch_size: int = 32,
    img_size: int = 224,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Builds train, val, and test DataLoaders along with class_names.
    Handles both unified folders and pre-split (train/val/test) folder structures.
    """
    data_path = Path(dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    # Standard transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    # Case 1: Pre-split directory structure
    if train_dir.exists() and (val_dir.exists() or test_dir.exists()):
        train_data = datasets.ImageFolder(train_dir, transform=train_transform)
        class_names = train_data.classes

        val_source = val_dir if val_dir.exists() else test_dir
        test_source = test_dir if test_dir.exists() else val_dir

        val_data = datasets.ImageFolder(val_source, transform=eval_transform)
        test_data = datasets.ImageFolder(test_source, transform=eval_transform)

    # Case 2: Unified dataset folder (split via random_split)
    else:
        raw_dataset = datasets.ImageFolder(data_path, transform=None)
        class_names = raw_dataset.classes

        total_size = len(raw_dataset)
        val_size = int(total_size * val_split)
        test_size = int(total_size * test_split)
        train_size = total_size - val_size - test_size

        generator = torch.Generator().manual_seed(seed)
        train_sub, val_sub, test_sub = random_split(
            raw_dataset, [train_size, val_size, test_size], generator=generator
        )

        train_data = DatasetFromSubset(train_sub, transform=train_transform)
        val_data = DatasetFromSubset(val_sub, transform=eval_transform)
        test_data = DatasetFromSubset(test_sub, transform=eval_transform)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names
```