---
name: model-artifact-saver
description: Saves PyTorch model weights (.pth state_dict) and target class_names (.txt) directly into the current working directory, including path verification and companion loading routines.
---

# PyTorch Model & Class Names Exporter Guidelines

When saving trained PyTorch models and label metadata to disk:

1. **Working Directory Targeting**: Save files directly into the active working directory (`Path.cwd()` or `./`) unless an explicit subdirectory is requested.
2. **Weight Persistence Standard**: Save `model.state_dict()` via `torch.save(obj=..., f=...)` with the `.pth` extension rather than pickling the full model object[cite: 1].
3. **Class Metadata Export**: Write `class_names` to a `.txt` file where each line corresponds to an index-aligned label (`\n`-separated) using `utf-8` encoding.
4. **Verification & Loader Pairing**: Confirm file generation and output companion loading code (`torch.load` + `model.load_state_dict`)[cite: 1].

---

## Reference Implementation

```python
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


def save_model_artifacts(
    model: nn.Module,
    class_names: List[str],
    model_filename: str = "model.pth",
    classes_filename: str = "class_names.txt",
    save_dir: Optional[Union[str, Path]] = None
) -> Tuple[Path, Path]:
    """
    Saves model state_dict (.pth) and class labels (.txt) into the current working directory.

    Args:
        model (nn.Module): Trained PyTorch model instance.
        class_names (List[str]): Ordered list of classification label strings.
        model_filename (str): Target filename for model weights. Default: 'model.pth'.
        classes_filename (str): Target filename for class names. Default: 'class_names.txt'.
        save_dir (str or Path, optional): Directory to save into. Defaults to Path.cwd().

    Returns:
        Tuple[Path, Path]: Resolved paths to (model_path, classes_path).
    """
    # 1. Resolve current directory
    target_dir = Path(save_dir) if save_dir is not None else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Format output file paths
    if not model_filename.endswith(".pth"):
        model_filename = f"{model_filename}.pth"
    if not classes_filename.endswith(".txt"):
        classes_filename = f"{classes_filename}.txt"

    model_path = target_dir / model_filename
    classes_path = target_dir / classes_filename

    # 3. Save model state_dict
    torch.save(obj=model.state_dict(), f=str(model_path))

    # 4. Save class_names into .txt (one label per line)
    with open(classes_path, mode="w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

    print("=" * 60)
    print("💾 ARTIFACTS SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model Weights (.pth) : {model_path.resolve()}")
    print(f"Class Names (.txt)   : {classes_path.resolve()} ({len(class_names)} classes)")
    print("=" * 60)

    return model_path, classes_path


def load_model_artifacts(
    model: nn.Module,
    model_filename: str = "model.pth",
    classes_filename: str = "class_names.txt",
    load_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[nn.Module, List[str]]:
    """
    Loads saved model state_dict (.pth) and class names (.txt) from disk.
    """
    source_dir = Path(load_dir) if load_dir is not None else Path.cwd()
    model_path = source_dir / model_filename
    classes_path = source_dir / classes_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model weight file not found at: {model_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Class names file not found at: {classes_path}")

    # 1. Load class labels
    with open(classes_path, mode="r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # 2. Determine target device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Restore state_dict into model architecture
    model.load_state_dict(torch.load(f=str(model_path), map_location=device))
    model.to(device)
    model.eval()

    print(f"✅ Loaded model weights from: {model_path}")
    print(f"✅ Loaded {len(class_names)} classes from: {classes_path}")

    return model, class_names
```