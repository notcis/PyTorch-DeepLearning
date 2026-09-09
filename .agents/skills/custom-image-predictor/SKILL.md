---
name: custom-image-predictor
description: Predicts custom images from a single URL, local image file path, or an entire folder of images using a trained PyTorch model. Visualizes predictions with confidence scores and top-k bar plots, handling transforms and ImageNet un-normalization cleanly.
---

# Custom Image Predictor & Inference Visualizer Guidelines

When running predictions on user-provided images (URLs, local files, or folder paths):

1. **Input Flexibility**: Support image URLs (fetched via `urllib` or `requests`), single file paths, or directories with recursive image search (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)[cite: 13].
2. **Image Preprocessing**:
   - Convert images to RGB via `PIL.Image.open().convert("RGB")`[cite: 13, 26].
   - Apply the exact evaluation `transforms` required by the trained model (Resize, CenterCrop, ToTensor, Normalize)[cite: 1, 6, 8].
   - Add a batch dimension using `.unsqueeze(0)`[cite: 6].
3. **Inference Safety**:
   - Always run within `model.eval()` and `torch.inference_mode()` (or `torch.no_grad()`)[cite: 10, 11].
   - Ensure the image tensor is moved to the same `device` as the model (`next(model.parameters()).device`)[cite: 12].
   - Obtain prediction logits, apply `torch.softmax(logits, dim=-1)` for multi-class or `torch.sigmoid` for binary classification to derive probabilities[cite: 1, 7, 10].
4. **Visual Analytics**:
   - Un-normalize normalized tensors before plotting using `mean` and `std` to avoid distorted colors[cite: 1, 11].
   - Plot predicted class name and confidence score (%) on titles[cite: 10, 11].
   - For single-image inspection, display the input image alongside a horizontal bar chart showing the Top-K predicted probabilities.

---

## Reference Implementation

```python
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import torch
import torch.nn as nn
from torchvision import transforms


def load_image_from_source(source: Union[str, Path]) -> Image.Image:
    """
    Loads an image from either a web URL or a local file path.
    """
    src_str = str(source).strip()
    if src_str.startswith("http://") or src_str.startswith("https://"):
        response = requests.get(src_str, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
    else:
        file_path = Path(src_str)
        if not file_path.exists():
            raise FileNotFoundError(f"Image not found at: {file_path}")
        img = Image.open(file_path)

    return img.convert("RGB")


def predict_single_image(
    model: nn.Module,
    image_source: Union[str, Path],
    class_names: List[str],
    transform: Optional[transforms.Compose] = None,
    device: Optional[Union[str, torch.device]] = None,
    top_k: int = 5
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Predicts a single image and returns (pred_class, confidence, top_k_predictions).
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    img = load_image_from_source(image_source)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_tensor)
        if logits.ndim > 1 and logits.shape[-1] > 1:
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        else:
            prob = torch.sigmoid(logits.squeeze(-1)).item()
            probs = torch.tensor([1.0 - prob, prob], device=device)

    k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)

    top_predictions = [
        (class_names[idx.item()], prob.item() * 100.0)
        for prob, idx in zip(top_probs, top_indices)
    ]

    best_pred_class = top_predictions[0][0]
    best_confidence = top_predictions[0][1]

    return best_pred_class, best_confidence, top_predictions


def plot_single_prediction(
    model: nn.Module,
    image_source: Union[str, Path],
    class_names: List[str],
    transform: Optional[transforms.Compose] = None,
    top_k: int = 5
) -> None:
    """
    Visualizes single image prediction with an adjacent Top-K confidence bar chart.
    """
    img = load_image_from_source(image_source)
    best_class, best_conf, top_preds = predict_single_image(
        model=model,
        image_source=image_source,
        class_names=class_names,
        transform=transform,
        top_k=top_k
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1. Display original image
    axes[0].imshow(img)
    axes[0].set_title(f"Pred: {best_class} ({best_conf:.1f}%)", fontsize=12, fontweight="bold", color="darkgreen")
    axes[0].axis("off")

    # 2. Top-K probabilities bar plot
    labels = [item[0] for item in reversed(top_preds)]
    scores = [item[1] for item in reversed(top_preds)]

    axes[1].barh(labels, scores, color="royalblue")
    axes[1].set_xlabel("Confidence (%)", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_title(f"Top-{len(top_preds)} Predictions", fontsize=12, fontweight="bold")
    for idx, score in enumerate(scores):
        axes[1].text(score + 1.5, idx, f"{score:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_folder_predictions(
    model: nn.Module,
    folder_path: Union[str, Path],
    class_names: List[str],
    transform: Optional[transforms.Compose] = None,
    num_images: int = 12,
    cols: int = 4
) -> None:
    """
    Finds images in a local folder and plots a grid of predicted outputs.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in folder.rglob("*.*") if p.suffix.lower() in valid_extensions]

    if not image_paths:
        print(f"⚠️ No image files found in '{folder_path}'.")
        return

    selected_paths = image_paths[:num_images]
    rows = (len(selected_paths) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if len(selected_paths) > 1 else [axes]

    for idx, img_p in enumerate(selected_paths):
        pred_cls, conf, _ = predict_single_image(
            model=model,
            image_source=img_p,
            class_names=class_names,
            transform=transform
        )
        img = Image.open(img_p).convert("RGB")
        axes[idx].imshow(img)
        axes[idx].set_title(f"Pred: {pred_cls}\n({conf:.1f}%)", fontsize=10, fontweight="bold", color="darkgreen")
        axes[idx].axis("off")

    for j in range(len(selected_paths), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Predictions from: {folder.name}", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.show()


def predict_and_plot(
    model: nn.Module,
    target: Union[str, Path],
    class_names: List[str],
    transform: Optional[transforms.Compose] = None,
    top_k: int = 5,
    num_images: int = 12
) -> None:
    """
    Entry point that automatically distinguishes between URL/file and folder path.
    """
    target_path = Path(str(target))
    if target_path.exists() and target_path.is_dir():
        plot_folder_predictions(
            model=model,
            folder_path=target_path,
            class_names=class_names,
            transform=transform,
            num_images=num_images
        )
    else:
        plot_single_prediction(
            model=model,
            image_source=target,
            class_names=class_names,
            transform=transform,
            top_k=top_k
        )