---
name: model-trainer-visualizer
description: Configures task-appropriate optimizer and loss_fn, trains a PyTorch classification model with early stopping, best weight checkpointing, and dynamically plots Train vs Validation Loss & Accuracy curves.
---

# PyTorch Model Trainer & Performance Visualizer Guidelines

When setting up optimization and running training loops in PyTorch, follow these standards:

1. **Criterion & Optimizer Selection**:
   - **Multi-class Classification**: Use `nn.CrossEntropyLoss()` (paired with raw output logits, do NOT add `nn.Softmax` to the model)[cite: 1, 10, 16, 23].
   - **Binary Classification**: Use `nn.BCEWithLogitsLoss()` (more numerically stable than `nn.Sigmoid` + `nn.BCELoss`)[cite: 2, 10, 19].
   - **Optimizer**: Default to `torch.optim.Adam` or `torch.optim.AdamW` with configurable learning rate (`lr`) and `weight_decay`[cite: 1, 2, 8]. Only pass parameters where `requires_grad=True`[cite: 1, 4].
2. **Training & Evaluation Separation**:
   - Training: Use `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`[cite: 1, 2, 11, 15].
   - Evaluation: Use `model.eval()` and `torch.inference_mode()` (or `torch.no_grad()`) to lock batch normalization running stats and disable dropout[cite: 1, 2, 4, 18].
3. **Loss & Metric Aggregation**:
   - Accumulate loss via `loss.item() * batch_size` to compute the exact epoch loss without memory accumulation[cite: 1, 3, 4, 10].
   - Track accuracy on a percentage scale (`0-100%`)[cite: 1, 3, 10].
4. **Early Stopping Mechanism**:
   - Track `patience` counter on validation loss[cite: 1, 2, 6, 18].
   - Save the best state using `copy.deepcopy(model.state_dict())` and restore weights upon triggering or completion (`restore_best_weights=True`)[cite: 1, 2, 18].
5. **Visual Analytics**:
   - Plot side-by-side curves for **Loss** (Train vs Val) and **Accuracy** (Train vs Val) bounded to completed epochs[cite: 1, 2, 4].

---

## Reference Implementation

```python
import copy
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def setup_loss_and_optimizer(
    model: nn.Module,
    problem_type: str = "multiclass",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Configures task-appropriate criterion and optimizer.
    Filters trainable parameters to support frozen backbones.
    """
    # 1. Loss function selection
    if problem_type.lower() == "multiclass":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif problem_type.lower() == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("problem_type must be either 'multiclass' or 'binary'")

    # 2. Only optimize parameters that require gradients (freeze-aware)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 3. Optimizer instantiation
    optimizer = torch.optim.Adam(
        params=trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return loss_fn, optimizer


class EarlyStopping:
    """
    Monitors a metric to stop training early and restores the best model weights.
    """
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        monitor: str = "val_loss",
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, current_metric: float, model: nn.Module) -> bool:
        score = -current_metric if self.mode == "min" else current_metric

        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

        return self.early_stop

    def restore_model(self, model: nn.Module) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int = 50,
    device: Optional[Union[str, torch.device]] = None,
    patience: int = 5,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Standard PyTorch training loop with Early Stopping and Metric History tracking.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    early_stopper = EarlyStopping(patience=patience, monitor="val_loss", mode="min", restore_best_weights=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        # -----------------------------
        # 1. Training Phase
        # -----------------------------
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)

            # Binary vs Multi-class prediction handling
            if outputs.ndim > 1 and outputs.shape[-1] == 1 and y.dtype == torch.float32:
                loss = loss_fn(outputs.squeeze(-1), y)
                preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).long()
            elif outputs.ndim > 1 and outputs.shape[-1] > 1:
                loss = loss_fn(outputs, y)
                preds = torch.argmax(outputs, dim=1)
            else:
                loss = loss_fn(outputs, y)
                preds = torch.argmax(outputs, dim=1) if outputs.ndim > 1 else (outputs > 0.5).long()

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X.size(0)
            train_correct += (preds == y).sum().item()
            train_total += X.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # -----------------------------
        # 2. Validation Phase
        # -----------------------------
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for X_val, y_val in val_dataloader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)

                if val_outputs.ndim > 1 and val_outputs.shape[-1] == 1 and y_val.dtype == torch.float32:
                    v_loss = loss_fn(val_outputs.squeeze(-1), y_val)
                    v_preds = (torch.sigmoid(val_outputs.squeeze(-1)) >= 0.5).long()
                elif val_outputs.ndim > 1 and val_outputs.shape[-1] > 1:
                    v_loss = loss_fn(val_outputs, y_val)
                    v_preds = torch.argmax(val_outputs, dim=1)
                else:
                    v_loss = loss_fn(val_outputs, y_val)
                    v_preds = torch.argmax(val_outputs, dim=1) if val_outputs.ndim > 1 else (val_outputs > 0.5).long()

                running_val_loss += v_loss.item() * X_val.size(0)
                val_correct += (v_preds == y_val).sum().item()
                val_total += X_val.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        # Step learning rate scheduler at epoch boundary
        if scheduler is not None:
            scheduler.step()

        # Record metrics
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # -----------------------------
        # 3. Early Stopping Check
        # -----------------------------
        if early_stopper(epoch_val_loss, model):
            print(f"⏹️ Early stopping triggered at Epoch {epoch}. Restoring best weights...")
            early_stopper.restore_model(model)
            break

    if not early_stopper.early_stop:
        early_stopper.restore_model(model)

    return model, history


def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Plots Train vs Validation Loss and Accuracy side-by-side.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # 1. Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    plt.title("Model Loss (Train vs Validation)", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    # 2. Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy", color="royalblue", lw=2)
    plt.plot(epochs_range, history["val_acc"], label="Val Accuracy", color="crimson", lw=2, linestyle="--")
    plt.title("Model Accuracy (Train vs Validation)", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()