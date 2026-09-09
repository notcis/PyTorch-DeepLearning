---
name: model-summary-inspector
description: Inspects PyTorch model architecture and parameter counts using torchinfo. Automatically checks and installs torchinfo if missing, then runs summary() with a user-specified or inferred input_size (batch_size=1).
---

# PyTorch Model Summary Inspector Guidelines

When summarizing or inspecting a PyTorch model using `torchinfo.summary`:

1. **Auto-Resolution & Import**:
   - Wrap the import inside a `try/except ImportError` block.
   - If missing, trigger `!pip install torchinfo` (or via Python environment installer) and import `from torchinfo import summary` immediately.
2. **Device Synchronization**:
   - Identify the model's current active device (`next(model.parameters()).device`) to prevent device mismatch errors during the dummy forward pass.
3. **Execution with `input_size`**:
   - Invoke `summary(model, input_size=(1, ...), device=device)` ensuring the batch dimension is explicitly set to 1.
   - For Computer Vision / Image models, use `(1, 3, H, W)` (e.g., `(1, 3, 224, 224)` or `(1, 3, 288, 288)` for EfficientNet).
   - For Tabular models, use `(1, num_features)`.

---

## Reference Implementation

```python
import sys
import subprocess
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

# Step 1: Try importing torchinfo, install if not present
try:
    from torchinfo import summary
except ImportError:
    print("📦 'torchinfo' is not installed. Installing now...")
    # Supports Jupyter/Colab shell magic (!pip install) or subprocess fallback
    try:
        get_ipython().system("pip install torchinfo")
    except NameError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary
    print("✅ Successfully imported summary from torchinfo.")


def inspect_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    col_names: Sequence[str] = ("input_size", "output_size", "num_params", "trainable"),
    depth: int = 3,
    verbose: int = 1
):
    """
    Summarizes a PyTorch model ensuring correct device allocation and input shape.
    
    Args:
        model (nn.Module): The target model to inspect.
        input_size (Tuple[int, ...]): Input tensor shape starting with batch size 1 (1, C, H, W) or (1, features).
        col_names (Sequence[str]): Columns to show in the summary table.
        depth (int): Depth of nested submodules to display. Default: 3.
        verbose (int): Output verbosity level (0, 1, or 2).
    """
    # Detect model device to prevent CPU vs CUDA runtime mismatch
    device = next(model.parameters()).device if list(model.parameters()) else "cpu"
    model.eval()

    # Step 2: Call summary with model, input_size, and device
    return summary(
        model=model,
        input_size=input_size,
        device=device,
        col_names=list(col_names),
        depth=depth,
        verbose=verbose
    )