---
name: model-test-evaluator
description: Evaluates a trained PyTorch model on a test_dataloader to compute total correct predictions, test accuracy, plot a multi-class Confusion Matrix with class names, and visualize a grid of random image predictions with ground truth vs predicted labels.
---

# PyTorch Test Evaluator & Prediction Visualizer Guidelines

When building test inference and evaluation routines in PyTorch, apply the following standards:

1. **Inference Mode & Device Handling**:
   - Set the model to evaluation mode using `model.eval()` and execute inference within `torch.inference_mode()` to optimize performance and prevent memory accumulation[cite: 1, 6].
   - Match tensor device with the model (`device = next(model.parameters()).device`)[cite: 1, 4].
2. **Metric Aggregation**:
   - Accumulate total samples, correct counts, loss, and append all `y_true` and `y_pred` tensors onto CPU for global metric calculations[cite: 1, 2].
   - Compute final accuracy as a percentage: $\text{Accuracy} = \frac{\text{Correct}}{\text{Total}} \times 100\%$[cite: 1, 4].
3. **Confusion Matrix Visualization**:
   - Use `sklearn.metrics.confusion_matrix` and `seaborn.heatmap` (or `ConfusionMatrixDisplay`) labeled with `class_names` on both axes[cite: 1, 3, 4].
4. **Random Prediction Grid**:
   - Sample $N$ random predictions, un-normalize images (reversing ImageNet mean/std if normalized), permute tensor shape to `(H, W, C)` for Matplotlib, and annotate titles with green (correct) or red (incorrect) text[cite: 2].

---

## Reference Implementation

```python
import random
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    class_names: List[str],
    loss_fn: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[float, int, int, np.ndarray, np.ndarray]:
    """
    Evaluates model performance over the test DataLoader.
    
    Returns:
        test_acc (float): Overall accuracy percentage.
        total_correct (int): Total number of correct predictions.
        total_samples (int): Total number of evaluated samples.
        all_preds (np.ndarray): Array of predicted class indices.
        all_targets (np.ndarray): Array of ground truth class indices.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)

    model.eval()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0

    all_preds_list = []
    all_targets_list = []

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            # Compute loss if criterion is provided
            if loss_fn is not None:
                loss = loss_fn(outputs, y)
                running_loss += loss.item() * X.size(0)

            # Prediction resolution for Multi-class / Binary classification
            if outputs.ndim > 1 and outputs.shape[-1] > 1:
                preds = torch.argmax(outputs, dim=1)
            elif outputs.ndim > 1 and outputs.shape[-1] == 1:
                preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).long()
            else:
                preds = torch.argmax(outputs, dim=1) if outputs.ndim > 1 else (outputs > 0.5).long()

            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)

            all_preds_list.extend(preds.cpu().numpy())
            all_targets_list.extend(y.cpu().numpy())

    all_preds = np.array(all_preds_list)
    all_targets = np.array(all_targets_list)
    test_acc = (total_correct / total_samples) * 100.0

    print("=" * 60)
    print("📊 TEST EVALUATION SUMMARY")
    print("=" * 60)
    if loss_fn is not None:
        print(f"Test Loss        : {running_loss / total_samples:.4f}")
    print(f"Total Samples    : {total_samples:,}")
    print(f"Correct Count    : {total_correct:,} / {total_samples:,}")
    print(f"Test Accuracy    : {test_acc:.2f}%")
    print("=" * 60)

    return test_acc, total_correct, total_samples, all_preds, all_targets


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues"
) -> None:
    """
    Plots a labeled confusion matrix heatmap using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fmt = ".2f" if normalize else "d"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title(f"Confusion Matrix {'(Normalized)' if normalize else ''}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_random_predictions(
    model: nn.Module,
    test_dataloader: DataLoader,
    class_names: List[str],
    num_images: int = 12,
    cols: int = 4,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    seed: Optional[int] = None
) -> None:
    """
    Samples and visualizes random images with predicted vs ground truth labels.
    Automatically un-normalizes standard ImageNet tensors for clean image display.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    model.eval()

    # Extract all samples or a subset from DataLoader
    collected_images = []
    collected_targets = []

    for X, y in test_dataloader:
        collected_images.append(X)
        collected_targets.append(y)
        if sum(len(x) for x in collected_images) >= num_images * 3:
            break

    all_X = torch.cat(collected_images, dim=0)
    all_y = torch.cat(collected_targets, dim=0)

    indices = random.sample(range(len(all_X)), min(num_images, len(all_X)))
    sample_images = all_X[indices]
    sample_targets = all_y[indices]

    with torch.inference_mode():
        logits = model(sample_images.to(device))
        if logits.ndim > 1 and logits.shape[-1] > 1:
            preds = torch.argmax(logits, dim=1).cpu()
            probs = torch.softmax(logits, dim=1).cpu()
        else:
            probs = torch.sigmoid(logits.squeeze(-1)).cpu()
            preds = (probs >= 0.5).long()

    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if len(indices) > 1 else [axes]

    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    for i, idx in enumerate(range(len(indices))):
        img = sample_images[idx].cpu()
        
        # Un-normalize if image has 3 color channels
        if img.shape[0] == 3:
            img = img * std_tensor + mean_tensor
            img = torch.clamp(img, 0.0, 1.0)
            img = img.permute(1, 2, 0).numpy()
        elif img.shape[0] == 1:
            img = img.squeeze(0).numpy()

        true_label = class_names[sample_targets[idx].item()]
        pred_label = class_names[preds[idx].item()]
        
        # Confidence score
        conf = probs[idx][preds[idx]].item() * 100.0 if probs.ndim > 1 else (probs[idx].item() if preds[idx] == 1 else 1 - probs[idx].item()) * 100.0

        is_correct = preds[idx].item() == sample_targets[idx].item()
        color = "green" if is_correct else "red"

        axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)",
            color=color,
            fontsize=10,
            fontweight="bold"
        )
        axes[i].axis("off")

    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Random Test Predictions (Green: Correct, Red: Incorrect)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
```