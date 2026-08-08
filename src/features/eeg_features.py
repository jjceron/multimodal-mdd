"""Subject-level engineered features from MODMA EEG windows.

These are the deterministic, *static* descriptors that validated the
subject-level signal (AUC ~0.677 / BACC ~0.585 on the 17-dim probe). They are
computed per subject over its windows and carry no learned parameters, so they
are reproducible and free of cross-fold leakage by construction. They form the
engineered branch of the EEG backbone and feed the multimodal fusion later.

window shape: ``[W, C, T]``.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing.modma_eeg import MODMADataset

FS = 250.0
EPS = 1e-12
BANDS = {
    "delta": (0.4, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def band_powers(windows: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """windows [W, C, T] -> ({band: [W, C] band power}, mag2 [W, C, F])."""
    mag2 = np.abs(np.fft.rfft(windows, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(windows.shape[-1], d=1.0 / FS)
    out: dict[str, np.ndarray] = {}
    for name, (f0, f1) in BANDS.items():
        mask = (freqs >= f0) & (freqs < f1)
        out[name] = mag2[:, :, mask].sum(axis=-1)
    return out, mag2


def global_features(bp: dict[str, np.ndarray], mag2: np.ndarray) -> np.ndarray:
    """Subject-level statistics over windows: [d] vector."""
    feats = []
    for name in BANDS:
        mean_acw = bp[name].mean(axis=0)  # [C] mean over windows
        std_acw = bp[name].std(axis=0)    # [C] std over windows (dynamics)
        feats += [mean_acw.mean(), mean_acw.std(), std_acw.mean()]
    p = mag2 / (mag2.sum(axis=-1, keepdims=True) + EPS)
    ent = -np.sum(p * np.log(p + EPS), axis=-1)  # [W, C]
    ent_mean_c = ent.mean(axis=0)                # [C]
    feats += [ent_mean_c.mean(), ent_mean_c.std()]
    return np.asarray(feats)


def subject_features(ds: MODMADataset) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-subject static features: (subjects, y, X [N, d])."""
    subjects, labels, feats = [], [], []
    for s in ds.samples:
        w = s["eeg"].numpy()
        bp, mag2 = band_powers(w)
        feats.append(global_features(bp, mag2))
        subjects.append(s["participant_id"])
        labels.append(int(s["label"].item()))
    return subjects, np.asarray(labels), np.asarray(feats)