---
name: dataset-image-visualizer
description: Visualizes raw image datasets from a directory path prior to dataset/dataloader preparation, including random image grid previews with class labels, image dimensions/aspect ratio inspection, and class balance bar plots.
---

# Dataset Image Visualizer Guidelines

When inspecting raw image datasets from a folder path, follow these steps:

1. **Path & Extension Validation**: Scan directories recursively for common image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) and detect class labels from parent folder names[cite: 13].
2. **Class Distribution Check**: Count samples per class and generate a bar plot to check for potential data imbalances[cite: 1, 13].
3. **Dimension & Aspect Ratio Audit**: Sample image files to analyze width, height, and aspect ratio distributions to help choose optimal resize transformations[cite: 13].
4. **Random Image Inspection Grid**: Sample and render a grid of random images with class names and resolution metadata[cite: 13].

---

## Reference Implementation

```python
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns


def scan_image_dataset(dataset_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Scans a directory path structured by class subfolders or flat image folders.
    Returns a list of (image_file_path, class_name) tuples.
    """
    data_path = Path(dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_label_pairs = []

    # Detect subdirectories as classes
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if subdirs:
        for subdir in subdirs:
            cls_name = subdir.name
            for file in subdir.rglob("*.*"):
                if file.suffix.lower() in valid_extensions:
                    image_label_pairs.append((str(file), cls_name))
    else:
        for file in data_path.glob("*.*"):
            if file.suffix.lower() in valid_extensions:
                image_label_pairs.append((str(file), "Uncategorized"))

    return image_label_pairs


def plot_random_dataset_images(
    dataset_path: Union[str, Path],
    num_samples: int = 12,
    cols: int = 4,
    seed: Optional[int] = None
) -> None:
    """
    Samples random images from the dataset path and displays them in a grid
    with class labels and resolution (Width x Height).
    """
    if seed is not None:
        random.seed(seed)

    image_label_pairs = scan_image_dataset(dataset_path)
    if not image_label_pairs:
        print(f"⚠️ No valid image files found in '{dataset_path}'.")
        return

    selected = random.sample(image_label_pairs, min(num_samples, len(image_label_pairs)))
    rows = (len(selected) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if len(selected) > 1 else [axes]

    for idx, (img_path, label) in enumerate(selected):
        try:
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                axes[idx].imshow(rgb_img)
                axes[idx].set_title(f"Class: {label}\n({img.width}x{img.height})", fontsize=10)
        except Exception as e:
            axes[idx].set_title(f"Error loading:\n{label}", color="red", fontsize=9)
        axes[idx].axis("off")

    for j in range(len(selected), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Random Dataset Images Preview", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_dataset_statistics(
    dataset_path: Union[str, Path],
    max_dim_samples: int = 300
) -> None:
    """
    Plots class balance bar plot and image dimension/aspect ratio distributions.
    """
    image_label_pairs = scan_image_dataset(dataset_path)
    if not image_label_pairs:
        print(f"⚠️ No valid images found in '{dataset_path}'.")
        return

    sns.set_theme(style="whitegrid")

    # 1. Class Distribution Barplot
    labels = [pair[1] for pair in image_label_pairs]
    label_counts = pd.Series(labels).value_counts()

    plt.figure(figsize=(max(8, len(label_counts) * 0.5), 4))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title(f"Dataset Class Distribution (Total Images: {len(image_label_pairs):,})", fontsize=13, fontweight="bold")
    plt.xlabel("Class Name", fontsize=11)
    plt.ylabel("Image Count", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 2. Dimensions and Aspect Ratio Check
    sampled_for_dim = random.sample(image_label_pairs, min(max_dim_samples, len(image_label_pairs)))
    widths, heights, aspect_ratios = [], [], []

    for img_path, _ in sampled_for_dim:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(round(w / h, 2))
        except Exception:
            continue

    if widths:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.scatterplot(x=widths, y=heights, alpha=0.6, color="royalblue", ax=axes[0])
        axes[0].set_title("Image Dimensions (Width vs Height)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Height (px)")

        sns.histplot(aspect_ratios, kde=True, color="teal", ax=axes[1])
        axes[1].set_title("Aspect Ratio Distribution (Width / Height)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Aspect Ratio")
        plt.tight_layout()
        plt.show()


def visualize_dataset(dataset_path: Union[str, Path], num_samples: int = 12) -> None:
    """
    Main visualization entry point executing random grid inspection and dataset metrics.
    """
    print(f"🔍 Inspecting dataset at: {dataset_path}")
    plot_dataset_statistics(dataset_path)
    plot_random_dataset_images(dataset_path, num_samples=num_samples)
```