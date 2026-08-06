"""EEGBackbone: learned subject-level EEG classifier that is also a feature extractor.

Two data paths feed one trainable head:

1. **CNN branch** — the paper-faithful DeepConvNet convolutional stack. It
   encodes each 2 s window to a deep embedding, then windows are *mean-pooled*
   per subject to a single vector ``z_cnn``.

2. **Engineered branch** — the static subject-level descriptors that validated
   the signal (band power mean/std, cross-channel topography, spectral entropy,
   17 dims, see :mod:`src.features.eeg_features`). Passed straight through a
   projection so the head learns their weighting jointly with the CNN.

The two are concatenated and classified by a trainable MLP. ``forward_features``
returns the concatenated representation ``z_eeg`` so the very same module can
serve later as the EEG backbone into the cross-modal attention model.

Methodology: the module operates on full-subject window tensors ``[S, W, C, T]``
(one batch item == one subject), so pooling never mixes train/test subjects.
No parameter is fit across folds; normalization uses per-fold train stats
(external to the module).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.features.eeg_features import band_powers, global_features
from src.models.deepconvnet import DeepConvNet


class EEGBackbone(nn.Module):
    """Subject-level classifier that doubles as an EEG feature extractor."""

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 1,
        n_samples: int = 500,
        dropout: float = 0.5,
        hidden: int = 32,
        engineered_dim: int = 17,
    ) -> None:
        super().__init__()
        self.engineered_dim = engineered_dim
        self.subject_level = True  # training loop flag: batch item == subject

        # CNN branch (per-window encoder, paper-faithful DeepConvNet blocks).
        self.encoder = DeepConvNet(
            n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
            dropout=dropout,
        )

        self.cnn_proj = nn.Linear(self.encoder.fc_features, hidden)
        self.eng_norm = nn.LayerNorm(engineered_dim)
        self.eng_proj = nn.Linear(engineered_dim, hidden)
        self.cnn_norm = nn.LayerNorm(hidden)
        self.z_norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Linear(hidden * 2, n_classes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._z_eeg: torch.Tensor | None = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Subject-level combined embedding ``z_eeg = [z_cnn | z_eng]``.

        x: ``[S, W, C, T]`` normalized subject windows (one item per subject).
        Returns ``[S, hidden * 2]``.
        """
        S, W, C, T = x.shape
        flat = x.reshape(S * W, C, T)                     # [S*W, C, T]
        emb = self.encoder.forward_features(flat)          # [S*W, fc]
        z_cnn = emb.view(S, W, -1).mean(dim=1)             # [S, fc] mean-pool
        z_cnn = self.cnn_norm(self.drop(nn.functional.gelu(self.cnn_proj(z_cnn))))

        z_eng = self.eng_proj(self.eng_norm(_batch_engineered(x).to(x.device)))
        z = self.z_norm(torch.cat([z_cnn, z_eng], dim=-1))  # [S, 2h]
        self._z_eeg = z
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        return self.head(z)                                # [S, n_classes]


def _batch_engineered(x: torch.Tensor) -> torch.Tensor:
    """Compute the 17 engineered features for each subject in the batch.

    x : x [S, W, C, T] (raw, pre-normalization scale is fine for ratios of
    band powers / entropies). Returns ``[S, 17]`` torch float32.
    """
    feats = []
    x = x.detach().cpu()
    for s in range(x.shape[0]):
        w = x[s].numpy()                                    # [W, C, T]
        bp, mag2 = band_powers(w.astype(np.float32))
        feats.append(global_features(bp, mag2))
    return torch.as_tensor(np.stack(feats), dtype=torch.float32)