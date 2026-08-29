"""DeepConvNet (Schirrmeister et al., 2017) adapted for MODMA EEG.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _block(
    in_channels: int,
    out_channels: int,
    n_channels: int,
    pool: tuple[int, int],
    dropout: float,
    first: bool = False,
) -> nn.Sequential:
    layers = [nn.Conv2d(in_channels, out_channels, (1, 10))]
    if first:
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ELU())
        layers.append(nn.MaxPool2d(pool))
        layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)
    layers += [
        nn.BatchNorm2d(out_channels),
        nn.ELU(),
        nn.MaxPool2d(pool),
        nn.Dropout2d(dropout),
    ]
    return nn.Sequential(*layers)


class DeepConvNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 1,
        n_samples: int = 500,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.block1 = _block(1, 4, n_channels, (1, 3), dropout, first=True)
        self.block2 = nn.Sequential(
            nn.Conv2d(4, 8, (n_channels, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout2d(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(8, 16, (1, 10)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout2d(dropout),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(16, 64, (1, 10)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout2d(dropout),
        )

        dummy = torch.randn(1, 1, n_channels, n_samples)
        with torch.no_grad():
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
        self.fc_features = int(x.numel())
        self.classifier = nn.Linear(self.fc_features, n_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))
