---
name: basic-cnn-builder
description: Constructs a customizable baseline Convolutional Neural Network (CNN) in PyTorch using nn.Sequential or modular blocks (Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear), returning both the instantiated model and matched evaluation transforms.
---

# Basic CNN & Transforms Builder Guidelines

When creating custom baseline CNN architectures and their corresponding preprocessing pipelines, apply the following standards:

1. **Layer Hierarchy & Modular Blocks**:
   - Structure convolutional blocks using `nn.Conv2d` -> `nn.BatchNorm2d` -> `nn.ReLU` -> `nn.MaxPool2d`[cite: 1, 6, 13].
   - Insert non-linear activations (`nn.ReLU`) between linear/convolutional layers to enable deep representation learning[cite: 1, 6, 28].
   - Use `nn.AdaptiveAvgPool2d((output_h, output_w))` before flattening to allow flexible input image dimensions without breaking dense layer input sizes[cite: 2].
   - Vectorize feature maps using `nn.Flatten()` rather than manual tensor reshapes[cite: 1, 6].
2. **Transform Pipeline**:
   - Resize raw images to the model's target resolution[cite: 7].
   - Convert images using `transforms.ToTensor()`[cite: 7].
   - Apply standard channel-wise scaling via `transforms.Normalize(mean=..., std=...)`[cite: 7].
3. **Interface & Output**:
   - Return a tuple `(model, transform)`.

---

## Reference Implementation

```python
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import transforms


class BasicCNN(nn.Module):
    """
    Standard modular 2D Convolutional Neural Network baseline.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_units: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Feature extractor blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pool guarantees fixed spatial shape (4x4) regardless of input size
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units * 4 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def create_basic_cnn(
    num_classes: int,
    in_channels: int = 3,
    img_size: int = 224,
    hidden_units: int = 32,
    dropout: float = 0.2,
    seed: Optional[int] = 42
) -> Tuple[nn.Module, transforms.Compose]:
    """
    Instantiates BasicCNN and returns the model alongside its evaluation transforms.

    Args:
        num_classes (int): Number of target classification classes.
        in_channels (int): Channels in input images (3 for RGB, 1 for Grayscale). Default: 3.
        img_size (int): Standard height and width to resize input images. Default: 224.
        hidden_units (int): Initial filters for the first convolution layer. Default: 32.
        dropout (float): Dropout probability for dense layers. Default: 0.2.
        seed (int, optional): Random seed for reproducible weight initialization.

    Returns:
        Tuple[nn.Module, transforms.Compose]: (model, transform)
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = BasicCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout=dropout
    )

    # Set appropriate normalization stats based on channel count
    if in_channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    return model, transform