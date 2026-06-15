import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def build_siran_model(pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    backbone = mobilenet_v3_small(weights=weights)

    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(256, 1),
    )

    return backbone
