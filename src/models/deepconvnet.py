"""DeepConvNet (Schirrmeister et al., 2017) adapted for MODMA EEG.

Structure follows the original paper: block1 stacks a temporal convolution with
a spatial (channel-wise) convolution before BatchNorm/ELU/max-pooling; the
following blocks are temporal convolutions with BatchNorm, ELU and max-pooling.

Adaptations for 2s @ 250 Hz windows (500 samples) and the small dataset: filter
counts are halved (8/16/32/64 vs 25/25/50/100/200) and the two last pools use
kernel 2 so the classifier keeps more temporal resolution (500 -> 7).
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
    layers = [nn.Conv2d(in_channels, out_channels, (1, 10), padding="same")]
    if first:
        layers.append(nn.Conv2d(out_channels, out_channels, (n_channels, 1)))
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

        self.block1 = _block(1, 8, n_channels, (1, 4), dropout, first=True)
        self.block2 = _block(8, 16, n_channels, (1, 4), dropout)
        self.block3 = _block(16, 32, n_channels, (1, 2), dropout)
        self.block4 = _block(32, 64, n_channels, (1, 2), dropout)

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
