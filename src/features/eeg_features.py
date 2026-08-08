"""Subject-level engineered features from MODMA EEG windows.

These are the deterministic, *static* descriptors that validated the
subject-level signal (AUC ~0.677 / BACC ~0.585 on the validated probe). They are
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
    """Subject-level statistics over windows: [d] vector (~31-d)."""
    feats = []
    for name in BANDS:
        mean_acw = bp[name].mean(axis=0)  # [C] mean over windows
        std_acw = bp[name].std(axis=0)    # [C] std over windows (dynamics)
        feats += [mean_acw.mean(), mean_acw.std(), std_acw.mean()]

    total = sum(bp.values())  # [W, C]
    for name in BANDS:
        rel = bp[name] / (total + EPS)  # [W, C] relative band power
        feats.append(rel.mean())
    theta_tot = bp["theta"].mean(axis=1)  # [W]
    alpha_tot = bp["alpha"].mean(axis=1)
    beta_tot = bp["beta"].mean(axis=1)
    feats += [
        (theta_tot / (alpha_tot + EPS)).mean(),
        (theta_tot / (beta_tot + EPS)).mean(),
    ]

    p = mag2 / (mag2.sum(axis=-1, keepdims=True) + EPS)
    ent = -np.sum(p * np.log(p + EPS), axis=-1)  # [W, C]
    ent_mean_c = ent.mean(axis=0)                # [C]
    feats += [ent_mean_c.mean(), ent_mean_c.std()]

    cum = np.cumsum(p, axis=-1)
    freqs = np.fft.rfftfreq(mag2.shape[-1], d=1.0 / FS)
    feats.append(freqs[np.argmax(cum >= 0.95, axis=-1)].mean())

    corr = np.corrcoef(total, rowvar=False)  # [C, C]
    tri = np.triu_indices(corr.shape[0], k=1)
    feats += [np.abs(corr[tri]).mean(), np.abs(corr[tri]).std()]
    return np.asarray(feats)


def hjorth_descriptors(windows: np.ndarray) -> np.ndarray:
    """Per-subject Hjorth statistics over raw windows: [4] vector."""
    d1 = np.diff(windows, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var = windows.var(axis=-1)           # [W, C]
    var1 = d1.var(axis=-1)
    var2 = d2.var(axis=-1)
    mobility = np.sqrt(var1 / (var + EPS))
    complexity = np.sqrt(var2 / (var1 + EPS)) / (mobility + EPS)
    return np.asarray(
        [var.mean(), mobility.mean(), mobility.std(axis=1).mean(), complexity.mean()]
    )


def subject_features(ds: MODMADataset) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-subject static features: (subjects, y, X [N, d])."""
    subjects, labels, feats = [], [], []
    for s in ds.samples:
        w = s["eeg"].numpy()
        bp, mag2 = band_powers(w)
        row = list(global_features(bp, mag2))
        row.extend(hjorth_descriptors(w))
        feats.append(row)
        subjects.append(s["participant_id"])
        labels.append(int(s["label"].item()))
    return subjects, np.asarray(labels), np.asarray(feats)