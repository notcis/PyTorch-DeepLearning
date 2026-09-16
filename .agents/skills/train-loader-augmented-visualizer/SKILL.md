---
name: train-loader-augmented-visualizer
description: Samples and displays 12 augmented training images from a PyTorch train_loader (or train_dataloader) alongside their class labels, handling ImageNet tensor un-normalization and clipping automatically to observe transform/augmentation effects.
---

# Train Loader Augmented Sample Visualizer Guidelines

When visualizing transformed/augmented samples from a training DataLoader:

1. **Batch Extraction**: Fetch batches using `next(iter(train_loader))` until at least 12 sample image tensors and target labels are collected.
2. **Channel Format & Un-normalization**:
   - PyTorch images arrive as normalized tensors $(C, H, W)$[cite: 3, 12].
   - Reverse the standard normalization: $\text{img} = \text{img} \times \text{std} + \text{mean}$ (using ImageNet stats by default: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)[cite: 1, 8].
   - Clip tensor values strictly to the range $[0.0, 1.0]$ via `torch.clamp` to avoid distortion artifacts during rendering.
   - Permute tensor dimensions to $(H, W, C)$ for Matplotlib display[cite: 3].
3. **Label Resolution**: Map integer label targets to human-readable strings if `class_names` is provided; otherwise, display raw class indices.
4. **Grid Layout**: Arrange images into a balanced $3 \times 4$ grid (12 samples total) with clear annotations highlighting the visual changes introduced by transformations (e.g., flips, rotations, cropping, jittering)[cite: 1, 8].

---

## Reference Implementation

```python
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def visualize_train_loader_samples(
    train_loader: DataLoader,
    class_names: Optional[List[str]] = None,
    num_samples: int = 12,
    cols: int = 4,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Visualizes transformed/augmented images directly from train_loader.

    Args:
        train_loader (DataLoader): The training DataLoader with active augmentations/transforms.
        class_names (List[str], optional): List of human-readable class names matching dataset index order.
        num_samples (int): Number of images to render. Default: 12.
        cols (int): Number of grid columns. Default: 4.
        mean (Tuple[float, float, float]): Channel-wise means used in transforms.Normalize.
        std (Tuple[float, float, float]): Channel-wise stds used in transforms.Normalize.
        figsize (Tuple[int, int], optional): Figure dimension override.
    """
    collected_images = []
    collected_labels = []

    # 1. Fetch images from the DataLoader iterator
    for images, labels in train_loader:
        collected_images.append(images)
        collected_labels.append(labels)
        if sum(x.size(0) for x in collected_images) >= num_samples:
            break

    if not collected_images:
        print("⚠️ Warning: No batches retrieved from train_loader.")
        return

    # Concatenate and slice to exact sample size
    batch_images = torch.cat(collected_images, dim=0)[:num_samples]
    batch_labels = torch.cat(collected_labels, dim=0)[:num_samples]

    rows = (num_samples + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 3.5, rows * 3.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]

    # Pre-calculate un-normalization vectors
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    for idx in range(num_samples):
        img = batch_images[idx].detach().cpu()
        label_idx = batch_labels[idx].item()

        # 2. Revert normalization (if 3-channel RGB image)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img * std_tensor + mean_tensor
            img = torch.clamp(img, 0.0, 1.0)
            img_np = img.permute(1, 2, 0).numpy()
            cmap = None
        elif img.ndim == 3 and img.shape[0] == 1:
            img_np = img.squeeze(0).numpy()
            cmap = "gray"
        else:
            img_np = img.numpy()
            cmap = None

        # 3. Resolve class title
        if class_names and 0 <= label_idx < len(class_names):
            title = f"{class_names[label_idx]} (idx: {label_idx})"
        else:
            title = f"Class: {label_idx}"

        axes[idx].imshow(img_np, cmap=cmap)
        axes[idx].set_title(title, fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    # Turn off leftover unused axes
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Transformed & Augmented Samples from train_loader", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
```