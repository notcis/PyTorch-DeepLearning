---
name: ann-train-loader-eda
description: Performs deep post-preprocessing Exploratory Data Analysis (EDA) on a PyTorch tabular train_dataloader. Audits batch shapes, feature normalization statistics (mean/std/ranges), target distributions, NaN/Inf anomalies, and plots feature correlation heatmaps and histograms.
---

# Post-Preprocessing Train DataLoader EDA Guidelines

When asked to inspect or perform EDA on a PyTorch `train_dataloader` (or `train_loader`) after data preparation for ANN training, follow these inspection protocols:

1. **Batch & Structural Verification**:
   - Inspect batch count, total sample volume, and input/output tensor shapes: `(batch_size, num_features)` and `(batch_size,)` or `(batch_size, 1)`.
   - Verify tensor dtypes: Features must be `torch.float32`[cite: 5]; Targets must be `torch.long` (multiclass/classification) or `torch.float32` (binary logits or regression)[cite: 3, 5, 8].
2. **Data Integrity & Numerical Health**:
   - Check for `NaN`, `Inf`, or extreme outliers across all batches.
3. **Post-Normalization Feature Statistics**:
   - For standardized data, features should have overall $\mu \approx 0$ and $\sigma \approx 1$[cite: 6].
   - Detect zero-variance features (dead constant columns after encoding/imputation).
4. **Target Distribution Audit**:
   - Audit target frequencies/classes in the training set to check for label imbalance.
5. **Visual Inspection**:
   - Plot histograms/KDE of post-processed feature distributions[cite: 3].
   - Plot post-processed feature correlation heatmap to verify multi-collinearity after encoding[cite: 3].

---

## Reference Implementation

```python
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader


def inspect_train_dataloader(
    train_dataloader: DataLoader,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    task_type: str = "classification",
    max_features_to_plot: int = 12
) -> None:
    """
    Performs full post-preprocessing EDA on PyTorch tabular train_dataloader.
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Inspect Single Batch Structure
    try:
        first_batch = next(iter(train_dataloader))
    except StopIteration:
        print("❌ Error: train_dataloader is empty.")
        return

    X_sample, y_sample = first_batch[0], first_batch[1]

    print("=" * 65)
    print("🔍 POST-PREPROCESSING TRAIN_DATALOADER INSPECTION")
    print("=" * 65)
    print(f"• Total Batches:      {len(train_dataloader):,}")
    print(f"• Batch Size:         {X_sample.shape[0]}")
    print(f"• Feature Shape (X):  {tuple(X_sample.shape)} | Dtype: {X_sample.dtype}")
    print(f"• Target Shape (y):   {tuple(y_sample.shape)} | Dtype: {y_sample.dtype}")

    # 2. Iterate and Collect Dataset Tensors
    all_x = []
    all_y = []
    total_samples = 0
    has_nan_or_inf = False

    for X_batch, y_batch in train_dataloader:
        if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
            has_nan_or_inf = True
        all_x.append(X_batch.detach().cpu())
        all_y.append(y_batch.detach().cpu())
        total_samples += X_batch.size(0)

    X_all = torch.cat(all_x, dim=0).numpy()
    y_all = torch.cat(all_y, dim=0).numpy().ravel()
    num_features = X_all.shape[1]

    print(f"• Total Train Volume: {total_samples:,} samples | Features Count: {num_features}")

    # Sanity Checks
    if has_nan_or_inf:
        print("⚠️ [WARNING] Detected NaN or Inf values inside train_dataloader! Check preprocessing/imputation.")
    else:
        print("✅ Data Integrity: Zero NaN/Inf values detected.")

    # 3. Normalization Summary Statistics (Mean, Std, Min, Max)
    means = np.mean(X_all, axis=0)
    stds = np.std(X_all, axis=0)
    mins = np.min(X_all, axis=0)
    maxs = np.max(X_all, axis=0)

    zero_std_cols = np.where(stds == 0)[0]
    if len(zero_std_cols) > 0:
        print(f"⚠️ [WARNING] Found constant features (std = 0) at indices: {zero_std_cols.tolist()}")
    else:
        print("✅ Feature Variance: All input features have non-zero variance.")

    print(f"\n📊 Global Scaled Feature Stats Across All Train Batches:")
    print(f"  - Average Mean across features : {np.mean(means):.4f} (Ideal ~ 0.0)")
    print(f"  - Average Std across features  : {np.mean(stds):.4f} (Ideal ~ 1.0)")
    print(f"  - Overall Minimum value        : {np.min(mins):.4f}")
    print(f"  - Overall Maximum value        : {np.max(maxs):.4f}")

    # 4. Target Distribution Inspection
    print(f"\n🎯 TARGET AUDIT ({task_type.upper()}):")
    if task_type.lower() == "classification":
        unique_classes, counts = np.unique(y_all, return_counts=True)
        plt.figure(figsize=(7, 4))
        
        labels = [class_names[int(c)] if class_names and int(c) < len(class_names) else f"Class {int(c)}" for c in unique_classes]
        sns.barplot(x=labels, y=counts, palette="mako")
        plt.title("Train DataLoader Target Class Balance", fontsize=12, fontweight="bold")
        plt.ylabel("Sample Count")
        plt.xlabel("Target Classes")
        
        for idx, count in enumerate(counts):
            pct = (count / total_samples) * 100.0
            plt.text(idx, count, f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.show()
    else:
        # Regression target
        plt.figure(figsize=(7, 4))
        sns.histplot(y_all, kde=True, color="teal")
        plt.title(f"Train DataLoader Target Distribution (Regression)", fontsize=12, fontweight="bold")
        plt.xlabel("Target Values")
        plt.tight_layout()
        plt.show()

    # 5. Feature Distributions Plot
    n_plot = min(num_features, max_features_to_plot)
    n_cols = 3
    n_rows = (n_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten() if n_plot > 1 else [axes]

    for i in range(n_plot):
        col_name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature [{i}]"
        sns.histplot(X_all[:, i], kde=True, ax=axes[i], color="royalblue")
        axes[i].set_title(f"{col_name}\nμ={means[i]:.2f}, σ={stds[i]:.2f}", fontsize=9)
        axes[i].set_xlabel("")

    for j in range(n_plot, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Sample Processed Feature Distributions (Showing First {n_plot}/{num_features})", 
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    # 6. Post-Processing Feature Correlation Matrix
    if num_features > 1:
        corr_subset = min(num_features, 20)
        corr = np.corrcoef(X_all[:, :corr_subset], rowvar=False)
        plt.figure(figsize=(min(12, corr_subset + 2), min(9, int(corr_subset * 0.7) + 2)))
        sns.heatmap(corr, cmap="coolwarm", cbar=True, annot=corr_subset <= 12, fmt=".2f", square=True)
        plt.title(f"Processed Features Correlation Heatmap (Top {corr_subset} Features)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    print("=" * 65 + "\n")
```