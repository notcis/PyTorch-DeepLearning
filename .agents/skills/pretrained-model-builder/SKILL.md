---
name: pretrained-model-builder
description: Builds a PyTorch transfer learning model with frozen backbone and replaced classification head for custom classes, returning (model, transforms) using torchvision Multi-Weight API.
---

# Pretrained Model & Transforms Builder Guidelines

When asked to create or fine-tune a pretrained model for computer vision:

1. **Weight Resolution & Transforms**: Always fetch the default ImageNet weights via torchvision's Multi-Weight API (`weights = torchvision.models.get_model_weights(model_name).DEFAULT`) and extract exact preprocessing transforms via `weights.transforms()`.
2. **Backbone Freezing**: Freeze all feature extraction / backbone parameters by setting `param.requires_grad = False`.
3. **Classifier Head Adaptation**: Identify the architecture type and replace the final linear projection layer with `num_classes` while keeping its `requires_grad = True`:
   - **ResNet / Inception**: `model.fc`
   - **EfficientNet / MobileNet**: `model.classifier[-1]` or `model.classifier[1]`
   - **Vision Transformer (ViT)**: `model.heads.head`
   - **ConvNeXt**: `model.classifier[2]`
4. **Output Interface**: Return a tuple of `(model, transforms)`.

---

## Reference Implementation

```python
from typing import Tuple, Any, Optional
import torch
import torch.nn as nn
import torchvision
from torchvision.models import get_model, get_model_weights


def create_pretrained_model(
    model_name: str,
    num_classes: int,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
    seed: Optional[int] = 42
) -> Tuple[nn.Module, Any]:
    """
    Creates a pretrained model, freezes the backbone, replaces the classifier head,
    and returns (model, transforms).

    Args:
        model_name (str): Name of the torchvision model (e.g., 'resnet50', 'efficientnet_b2', 'vit_b_16').
        num_classes (int): Number of target output classes.
        freeze_backbone (bool): Whether to freeze feature extraction layers. Default: True.
        dropout (float): Dropout probability for supported classifier heads. Default: 0.2.
        seed (int, optional): Random seed for reproducibility of head initialization.

    Returns:
        Tuple[nn.Module, Any]: (prepared_model, base_transforms)
    """
    model_name_clean = model_name.strip().lower()

    # 1. Load weights and transforms
    try:
        weights_enum = get_model_weights(model_name_clean).DEFAULT
        model = get_model(model_name_clean, weights=weights_enum)
        transforms = weights_enum.transforms()
    except Exception as e:
        raise ValueError(
            f"Error loading model '{model_name}'. Ensure the name is valid in torchvision.models. Details: {e}"
        )

    # 2. Freeze backbone parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 3. Deterministic head replacement
    if seed is not None:
        torch.manual_seed(seed)

    # 4. Adapt classifier head based on architecture
    # Case A: ResNet family
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    # Case B: EfficientNet / MobileNet family
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # Find the last Linear layer index in classifier
        last_idx = len(model.classifier) - 1
        for idx in reversed(range(len(model.classifier))):
            if isinstance(model.classifier[idx], nn.Linear):
                last_idx = idx
                break
        
        in_features = model.classifier[last_idx].in_features
        model.classifier[last_idx] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    # Case C: Vision Transformer (ViT)
    elif hasattr(model, "heads"):
        if hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features=in_features, out_features=num_classes)
        else:
            in_features = model.heads[0].in_features
            model.heads = nn.Linear(in_features=in_features, out_features=num_classes)

    # Case D: ConvNeXt family
    elif hasattr(model, "classifier") and hasattr(model.classifier, "__len__"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)

    else:
        raise NotImplementedError(
            f"Classifier adaptation not mapped for '{model_name}'. Please inspect model structure."
        )

    return model, transforms
```