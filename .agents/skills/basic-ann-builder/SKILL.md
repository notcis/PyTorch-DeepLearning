---
name: basic-ann-builder
description: Builds and returns a modular baseline Artificial Neural Network (ANN / MLP) in PyTorch tailored for tabular classification or regression tasks, featuring customizable hidden layers, Batch Normalization, Dropout, and task-appropriate output dimensions.
---

# PyTorch Basic ANN Model Builder Guidelines

When asked to construct a basic ANN (Feedforward Neural Network / Multi-Layer Perceptron) for tabular datasets, apply the following engineering standards:

1. **Architecture Hierarchy**:
   - Subclass `nn.Module` or stack modules cleanly using `nn.Sequential`.
   - Hidden blocks: Use linear transformations (`nn.Linear`) followed by non-linear activations (`nn.ReLU`), stabilization (`nn.BatchNorm1d`), and regularization (`nn.Dropout`).
   - Hidden units funnel pattern: e.g., $64 \rightarrow 32 \rightarrow 16$ to compress dense tabular representations.
2. **Task-Specific Head Configuration**:
   - **Binary Classification**: 1 output node (`nn.Linear(last_hidden, 1)`). Do NOT add `nn.Sigmoid` at the end to keep compatibility with `nn.BCEWithLogitsLoss`.
   - **Multi-class Classification**: Output nodes equal to `num_classes` (`nn.Linear(last_hidden, num_classes)`). Do NOT add `nn.Softmax` at the end to keep compatibility with `nn.CrossEntropyLoss`.
   - **Regression**: 1 output node (`nn.Linear(last_hidden, 1)`) with raw continuous values (no activation) for `nn.MSELoss`.
3. **Weight Initialization**:
   - Apply He/Kaiming Normal initialization to weights followed by `nn.ReLU`.
4. **Interface**:
   - Provide a factory function returning the instantiated `ann_model`.

---

## Reference Implementation

```python
from typing import List, Optional
import torch
import torch.nn as nn


class BasicANN(nn.Module):
    """
    Standard modular Feedforward Neural Network (ANN / MLP) for Tabular Data.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int] = [64, 32, 16],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()

        layers = []
        in_features = input_dim

        # 1. Construct Hidden Layers Dynamically
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_features = units

        # 2. Output Decision Layer (Raw Logits / Unactivated values)
        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Applies Kaiming/He normal initialization to Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_basic_ann_model(
    input_dim: int,
    problem_type: str = "binary",
    num_classes: Optional[int] = None,
    hidden_units: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True,
    seed: Optional[int] = 42
) -> nn.Module:
    """
    Constructs and returns an instantiated PyTorch ANN model.

    Args:
        input_dim (int): Number of input features (from preprocessed X).
        problem_type (str): Type of task ('binary', 'multiclass', 'regression'). Default: 'binary'.
        num_classes (int, optional): Number of distinct target classes (required if multiclass).
        hidden_units (List[int], optional): Neuron dimensions per hidden layer. Default: [64, 32, 16].
        dropout_rate (float): Dropout probability for regularization. Default: 0.2.
        use_batch_norm (bool): Whether to include BatchNorm1d layers. Default: True.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        nn.Module: The configured basic ANN model ready for training.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if hidden_units is None:
        hidden_units = [64, 32, 16]

    # Resolve output node size
    task = problem_type.lower().strip()
    if task == "binary":
        output_dim = 1
    elif task == "multiclass":
        if num_classes is None or num_classes < 2:
            raise ValueError("num_classes must be >= 2 for multiclass classification")
        output_dim = num_classes
    elif task == "regression":
        output_dim = 1
    else:
        raise ValueError(f"Unsupported problem_type '{problem_type}'. Choose 'binary', 'multiclass', or 'regression'.")

    # Instantiate model
    ann_model = BasicANN(
        input_dim=input_dim,
        hidden_units=hidden_units,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )

    return ann_model


if __name__ == "__main__":
    # Example Quick Check
    sample_input_dim = 27  # Matching preprocessed tabular input dimension
    
    # 1. Binary Classification Model
    ann_model = create_basic_ann_model(
        input_dim=sample_input_dim,
        problem_type="binary",
        hidden_units=[32, 16, 8],
        dropout_rate=0.2
    )
    
    print("✅ Model created successfully:")
    print(ann_model)
    
    # Test forward pass with dummy batch
    dummy_x = torch.randn(4, sample_input_dim)
    dummy_out = ann_model(dummy_x)
    print(f"\nDummy Input Shape:  {dummy_x.shape}")
    print(f"Dummy Output Shape: {dummy_out.shape} (Raw Logits)")
```