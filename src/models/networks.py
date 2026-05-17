from __future__ import annotations

from typing import Final

from torch import Tensor, nn
from torchvision.models import resnet18


FEATURE_DIM: Final[int] = 512


class SplitResNet18(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")

        backbone = resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        backbone.maxpool = nn.Identity()
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1],
            nn.Flatten(start_dim=1),
        )
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return features, logits
