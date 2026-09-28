---
name: ann-train-visualizer
description: Trains an Artificial Neural Network (ANN) model using PyTorch, calculates Loss and Accuracy for Train and Validation sets per epoch, supports Early Stopping, and plots comparative Loss & Accuracy curves side-by-side using matplotlib.
---

# PyTorch ANN Model Trainer & Learning Curves Visualizer Guidelines

When tasked with training an ANN model on tabular datasets and evaluating learning curves, follow these standards:

1. **Optimization & Criterion Setup**:
   - For Binary Classification: Use `nn.BCEWithLogitsLoss()`[cite: 2].
   - For Multi-class Classification: Use `nn.CrossEntropyLoss()`[cite: 2, 3].
   - For Optimizer: Default to `torch.optim.Adam` or `torch.optim.AdamW`[cite: 2, 3, 6].
2. **Loop Discipline**:
   - Training phase: Set `model.train()`, call `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`[cite: 1, 2].
   - Validation phase: Set `model.eval()`, compute metrics under `torch.inference_mode()` (or `torch.no_grad()`)[cite: 1, 2].
   - Use `loss.item() * X.size(0)` for proper weighted epoch loss aggregation without memory leaks[cite: 1, 2, 9].
3. **Metric Tracking**:
   - Record `train_loss`, `train_acc`, `val_loss`, and `val_acc` per epoch[cite: 1, 2].
4. **Learning Curves Plotting**:
   - Generate two side-by-side subplots: **Loss Curve** (Train vs Val) and **Accuracy Curve** (Train vs Val)[cite: 1, 2].

---

## Reference Implementation

```python
import copy
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Early Stopping mechanism to save the best model weights based on validation loss."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_ann_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    problem_type: str = "binary",
    patience: int = 7,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains an ANN model, tracks train/val metrics, and returns the trained model with history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Setup Criterion & Optimizer
    if problem_type.lower() == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif problem_type.lower() == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("problem_type must be 'binary' or 'multiclass'")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"🚀 Training ANN model on device: {device} | Total Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        # -----------------------------
        # Train Step
        # -----------------------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            if problem_type == "binary":
                loss = criterion(outputs.squeeze(-1), y_batch.float())
                preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).long()
            else:
                loss = criterion(outputs, y_batch.long())
                preds = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (preds == y_batch).sum().item()
            train_total += X_batch.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # -----------------------------
        # Validation Step
        # -----------------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.inference_mode():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)

                if problem_type == "binary":
                    v_loss = criterion(val_outputs.squeeze(-1), y_val.float())
                    v_preds = (torch.sigmoid(val_outputs.squeeze(-1)) >= 0.5).long()
                else:
                    v_loss = criterion(val_outputs, y_val.long())
                    v_preds = torch.argmax(val_outputs, dim=1)

                val_loss += v_loss.item() * X_val.size(0)
                val_correct += (v_preds == y_val).sum().item()
                val_total += X_val.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        # Record History
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # Early Stopping Check
        if early_stopper(epoch_val_loss, model):
            print(f"⏹️ Early stopping triggered at epoch {epoch}. Restoring best model weights.")
            early_stopper.restore_best_weights(model)
            break

    return model, history


def plot_ann_learning_curves(history: Dict[str, List[float]], figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Plots Train vs Validation Loss and Accuracy side-by-side.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Loss Plot
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    axes[0].set_title("ANN Loss Curve (Train vs Validation)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Accuracy Plot
    axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy", color="royalblue", lw=2)
    axes[1].plot(epochs_range, history["val_acc"], label="Val Accuracy", color="crimson", lw=2, linestyle="--")
    axes[1].set_title("ANN Accuracy Curve (Train vs Validation)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
```