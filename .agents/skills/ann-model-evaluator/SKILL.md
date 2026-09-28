---
name: ann-model-evaluator
description: Evaluates a trained PyTorch ANN model on a test_dataloader for tabular datasets. Generates evaluation metrics and visualizations for Classification (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC) and Regression (MSE, RMSE, MAE, R2, Actual vs Predicted, Residuals).
---

# PyTorch Tabular ANN Test Evaluator Guidelines

When evaluating an ANN model on tabular test datasets, apply the following evaluation workflows:

1. **Inference Setup**:
   - Set the model to evaluation mode using `model.eval()`[cite: 1].
   - Wrap loop in `torch.inference_mode()` to prevent gradient computation and save memory[cite: 1].
   - Match tensor device dynamically (`device = next(model.parameters()).device`)[cite: 1].
2. **Classification Logic**:
   - Resolve raw logits: Apply `torch.sigmoid()` for Binary classification[cite: 1, 5] or `torch.softmax(..., dim=1)` for Multiclass classification[cite: 1, 3].
   - Gather predictions, probabilities, and labels onto CPU[cite: 1, 3, 8].
   - Compute metrics: Accuracy, Precision, Recall / Sensitivity, F1-Score (support `binary` and `weighted` modes)[cite: 1, 3].
   - Compute ROC-AUC score using prediction probabilities[cite: 1].
   - Plot Confusion Matrix heatmap with Matplotlib and Seaborn[cite: 1, 2].
3. **Regression Logic**:
   - Align output dimensions using `.squeeze()` or `.view(-1)`[cite: 1, 8].
   - Compute metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$)[cite: 3, 4].
   - Plot two diagnostic graphs: Actual vs. Predicted scatter plot and Residuals distribution histogram.

---

## Reference Implementation

```python
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_ann_testset(
    model: nn.Module,
    test_dataloader: DataLoader,
    task_type: str = "classification",
    class_names: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluates PyTorch ANN model on test_dataloader.
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()

    task = task_type.lower().strip()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            if task == "classification":
                if outputs.ndim > 1 and outputs.shape[-1] > 1:
                    # Multiclass classification
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.append(probs.cpu().numpy())
                else:
                    # Binary classification (Single Logit output)
                    probs = torch.sigmoid(outputs.squeeze(-1))
                    preds = (probs >= 0.5).long()
                    all_probs.append(probs.cpu().numpy())

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

            elif task == "regression":
                preds = outputs.view(-1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.view(-1).cpu().numpy())
            else:
                raise ValueError("task_type must be either 'classification' or 'regression'")

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    results: Dict[str, Union[float, np.ndarray]] = {}

    print("\n" + "=" * 65)
    print(f"📊 ANN TEST SET EVALUATION REPORT ({task.upper()})")
    print("=" * 65)

    # 1. Classification Metrics & Confusion Matrix
    if task == "classification":
        y_prob = np.concatenate(all_probs, axis=0)
        unique_classes = np.unique(y_true)
        avg_mode = "binary" if len(unique_classes) <= 2 else "weighted"

        acc = accuracy_score(y_true, y_pred) * 100.0
        prec = precision_score(y_true, y_pred, average=avg_mode, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg_mode, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg_mode, zero_division=0)

        results["accuracy"] = acc
        results["precision"] = prec
        results["recall"] = rec
        results["f1_score"] = f1

        print(f"• Accuracy               : {acc:.2f}%")
        print(f"• Precision ({avg_mode:<8}) : {prec:.4f}")
        print(f"• Recall / Sensitivity   : {rec:.4f}")
        print(f"• F1-Score               : {f1:.4f}")

        # Compute ROC-AUC
        try:
            if len(unique_classes) <= 2:
                roc_auc = roc_auc_score(y_true, y_prob)
            else:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            results["roc_auc"] = roc_auc
            print(f"• ROC-AUC Score          : {roc_auc:.4f}")
        except Exception:
            print("• ROC-AUC Score          : N/A (Requires at least two classes present in y_true)")

        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm
        labels = class_names if class_names and len(class_names) == len(cm) else [str(i) for i in range(len(cm))]

        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("ANN Test Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("Actual Label", fontsize=11)
        plt.tight_layout()
        plt.show()

    # 2. Regression Metrics & Diagnostic Plots
    elif task == "regression":
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results["mse"] = mse
        results["rmse"] = rmse
        results["mae"] = mae
        results["r2_score"] = r2

        print(f"• Mean Squared Error (MSE)       : {mse:.4f}")
        print(f"• Root Mean Squared Error (RMSE) : {rmse:.4f}")
        print(f"• Mean Absolute Error (MAE)     : {mae:.4f}")
        print(f"• R-squared (R²) Score           : {r2:.4f}")

        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.5, color="teal", edgecolors="none")
        min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
        max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
        axes[0].set_title("Actual vs. Predicted Values", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Actual Values")
        axes[0].set_ylabel("Predicted Values")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Residuals Distribution
        sns.histplot(residuals, kde=True, ax=axes[1], color="royalblue")
        axes[1].axvline(0, color="red", linestyle="--", lw=2)
        axes[1].set_title("Residuals Error (Actual - Predicted)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Error")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()

    print("=" * 65 + "\n")
    return results
```