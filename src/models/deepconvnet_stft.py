"""DeepConvNet for 2D STFT spectrograms of MODMA EEG.

Adaptation of the time-domain DeepConvNet (Schirrmeister et al., 2017) to the
2D time-frequency input used by Yousufi et al. (Brain Sci 2024). Each sample
is a log-magnitude STFT of shape ``[n_channels, n_freq, n_time]`` instead of a
raw ``[n_channels, n_samples]`` trace, so the convolutions run over the
(freq, time) plane rather than the time axis only.

The layout mirrors the proven DeepConvNet block structure (filters 4/8/16/64):

  * block1: conv over (freq, time) for every EEG channel independently
    (kernel size 1 across channels), so spectral-temporal features are learned
    per channel before any spatial mixing.
  * block2: spatial convolution with kernel ``(n_channels, 1, 1)`` that mixes
    all channels into a single spatial map (the analogue of DeepConvNet's
    channel-wise convolution).
  * block3/block4: further (freq, time) convolutions.

``fc_features`` is derived from a dummy forward pass, so the model adapts to
any ``(n_channels, n_freq, n_time)`` with no extra parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _block(
    in_channels: int,
    out_channels: int,
    kernel: tuple[int, int, int],
    pool: tuple[int, int, int],
    dropout: float,
    same: bool = True,
) -> nn.Sequential:
    layers = [
        nn.Conv3d(in_channels, out_channels, kernel, padding="same" if same else 0),
        nn.BatchNorm3d(out_channels),
        nn.ELU(),
        nn.MaxPool3d(pool),
        nn.Dropout3d(dropout),
    ]
    return nn.Sequential(*layers)


class DeepConvNetSTFT(nn.Module):
    def __init__(
        self,
        n_channels: int = 29,
        n_classes: int = 2,
        n_freq: int = 47,
        n_time: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        ft = (1, 5, 3)  # (freq, time) kernel; size 1 across the channel axis
        self.block1 = _block(1, 4, ft, (1, 2, 1), dropout)
        self.block2 = _block(4, 8, (n_channels, 1, 1), (1, 2, 1), dropout, same=False)
        self.block3 = _block(8, 16, ft, (1, 2, 2), dropout)
        self.block4 = _block(16, 64, ft, (1, 2, 2), dropout)

        dummy = torch.randn(1, 1, n_channels, n_freq, n_time)
        with torch.no_grad():
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
        self.fc_features = int(x.numel())
        self.classifier = nn.Linear(self.fc_features, n_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))
