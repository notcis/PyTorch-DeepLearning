import torch.nn as nn
from torchvision.models import efficientnet_b2


def create_model(num_classes: int = 6, dropout: float = 0.2) -> nn.Module:
    model = efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=in_features, out_features=num_classes),
        ),
    )
    return model