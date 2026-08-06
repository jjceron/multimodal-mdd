"""EEGBackbone: learned EEG encoder + validated engineered features, gated fusion.

Design rationale (régimen: 53 subjects, ~33 train per fold):

* A plain DeepConvNet (Schirrmeister, built for motor-imagery) aggressively
  pools the time axis and destroys the static spectral/topographic signal that
  this dataset actually contains — it collapsed to chance repeatedly.
* The validated signal lives in *band-power/topography/dynamics/entropy*
  descriptors (17-dim probe, AUC ~0.69). So the model:

  1. **SpectralConvNet (CNN branch)** — a light, EEGNet-style depthwise
     convolutional encoder over the *time axis* that learns spectral filters;
     energy (``mean(x**2)`` + log) is pooled over time per filter, giving a
     per-window learned-band descriptor. It has ~6k params and is trained on
     **per-window** samples (~10k windows), evaluated at subject level.
  2. **Engineered branch** — the 17 static subject-level features computed on
     the RAW windows (same descriptors as the validated probe).
  3. **Gate (learned)** — per-element ``sigmoid`` weights between the CNN and
     engineered representations, so the model can *auto-limit* the CNN if it
     finds no signal without losing the engineered one.

  ``forward`` returns per-window logits ``[S*W, n_classes]`` (so the window
  training loop provides ~10k samples); evaluation pools per subject via mean
  softmax. ``forward_features`` returns the subject-level ``z_eeg [S, 2h]``
  for the future cross-modal fusion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.features.eeg_features import band_powers, global_features


class SpectralConvNet(nn.Module):
    """Per-window learned spectral encoder (depthwise over time axis).

    Input ``[B, C, T]`` -> per-window descriptor ``[B, hidden]``.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 500,
        hidden: int = 32,
        dropout: float = 0.5,
        n_filters: int = 16,
    ) -> None:
        super().__init__()
        depth = 4  # depthwise ratio
        self.net = nn.Sequential(
            nn.Conv1d(
                n_channels, n_channels * depth,
                kernel_size=32, padding=16, groups=n_channels,
            ),
            nn.BatchNorm1d(n_channels * depth),
            nn.ELU(),
            nn.Conv1d(n_channels * depth, n_filters, kernel_size=1),
            nn.BatchNorm1d(n_filters),
            nn.ELU(),
        )
        self.fc = nn.Linear(n_filters, hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                      # [B, n_filters, T]
        energy = torch.log1p(h.pow(2).mean(dim=2))  # [B, n_filters] log band energy
        return self.drop(F.gelu(self.fc(energy)))


class EEGBackbone(nn.Module):
    """Gated fusion of learned spectral CNN + validated engineered features.

    ``full_subject_input=True``: the training loop passes the whole subject
    tensor ``[S, W, C, T]`` (raw ``x_raw``) so the engineered branch can be
    computed on RAW windows; the model returns per-window logits.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 1,
        n_samples: int = 500,
        dropout: float = 0.5,
        hidden: int = 32,
        engineered_dim: int = 17,
        n_filters: int = 16,
    ) -> None:
        super().__init__()
        self.engineered_dim = engineered_dim
        self.full_subject_input = True  # expects [S, W, C, T] + x_raw

        self.cnn = SpectralConvNet(
            n_channels=n_channels, n_samples=n_samples,
            hidden=hidden, dropout=dropout, n_filters=n_filters,
        )
        self.cnn_norm = nn.LayerNorm(hidden)

        self.eng_norm = nn.LayerNorm(engineered_dim)
        self.eng_proj = nn.Linear(engineered_dim, hidden)

        self.gate = nn.Linear(hidden * 2, hidden)
        self.z_norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Linear(hidden * 2, n_classes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._z_eeg: torch.Tensor | None = None

    def _window_z(self, x: torch.Tensor, x_raw: torch.Tensor | None = None) -> torch.Tensor:
        """Per-window fused representation ``[S*W, 2h]``."""
        S, W, C, T = x.shape
        flat = x.reshape(S * W, C, T)                 # [S*W, C, T] normalized
        z_cnn = self.cnn_norm(self.cnn(flat))          # [S*W, h]

        raw = x_raw if x_raw is not None else x
        eng_subj = self.eng_proj(self.eng_norm(_batch_engineered(raw).to(x.device)))
        z_eng = eng_subj.repeat_interleave(W, dim=0)   # [S*W, h] (subject context)

        g = torch.sigmoid(self.gate(torch.cat([z_cnn, z_eng], dim=-1)))  # [S*W, h]
        z = self.z_norm(torch.cat([g * z_cnn, (1 - g) * z_eng], dim=-1))
        return z

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor | None = None) -> torch.Tensor:
        """Per-window logits ``[S*W, n_classes]``."""
        z = self._window_z(x, x_raw)
        return self.head(self.drop(z))

    def forward_features(self, x: torch.Tensor, x_raw: torch.Tensor | None = None) -> torch.Tensor:
        """Subject-level embedding ``z_eeg [S, 2h]`` (mean-pooled over windows)."""
        z = self._window_z(x, x_raw)
        S, W = x.shape[0], x.shape[1]
        z_subj = z.view(S, W, -1).mean(dim=1)
        self._z_eeg = z_subj
        return z_subj


def _batch_engineered(x: torch.Tensor) -> torch.Tensor:
    """Compute the 17 engineered features for each subject in the batch.

    x: ``[S, W, C, T]`` RAW windows. Returns ``[S, 17]`` torch float32.
    """
    feats = []
    x = x.detach().cpu()
    for s in range(x.shape[0]):
        w = x[s].numpy()                                   # [W, C, T]
        bp, mag2 = band_powers(w.astype(np.float32))
        feats.append(global_features(bp, mag2))
    return torch.as_tensor(np.stack(feats), dtype=torch.float32)