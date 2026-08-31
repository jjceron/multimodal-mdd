from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

mne.set_log_level("WARNING")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HUSM_DIR = PROJECT_ROOT / "data/raw/husm"

LABEL_MAP = {"H": 0, "MDD": 1}

HUSM_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz",
    "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "Cz", "Pz",
]


def _base_channel(name: str) -> str | None:
    m = re.match(r"\s*(?:EEG\s+)?([A-Za-z0-9]+)-?", name.strip())
    if not m:
        return None
    base = m.group(1)
    if base in HUSM_CHANNELS:
        return base
    return None


def _window(eeg, fs, window_sec, overlap):
    win = max(round(window_sec * fs), 1)
    stride = max(round(win * (1.0 - overlap)), 1)
    n = (eeg.shape[1] - win) // stride + 1
    if n < 1:
        n, stride = 1, max(eeg.shape[1] - win, 1)
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return np.transpose(eeg[:, idx], (1, 0, 2))


def _process_edf(path: Path, fs_target: float | None, lowcut, highcut,
                 window_sec, overlap) -> np.ndarray | None:
    raw = mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
    pick = []
    for ch in raw.ch_names:
        base = _base_channel(ch)
        if base is not None and base not in pick:
            pick.append(base)
    if len(pick) != len(HUSM_CHANNELS):
        missing = [c for c in HUSM_CHANNELS if c not in pick]
        print(f"  Missing {len(missing)} channels in {path.name}: {missing}")
        return None
    names = sorted(pick, key=HUSM_CHANNELS.index)
    raw.pick([ch for ch in raw.ch_names if _base_channel(ch) in names])
    raw.reorder_channels([ch for ch in raw.ch_names if _base_channel(ch) in names])
    raw.set_eeg_reference("average", verbose=False)
    fs = float(raw.info["sfreq"])
    if highcut is not None and highcut >= fs / 2:
        highcut = None
    if lowcut is not None or highcut is not None:
        raw.filter(l_freq=lowcut, h_freq=highcut, fir_design="firwin", verbose=False)
    if fs_target is not None and not np.isclose(fs, fs_target):
        raw.resample(fs_target, npad="auto", verbose=False)
    fs = float(raw.info["sfreq"])
    eeg = np.nan_to_num(raw.get_data().astype(np.float32), nan=0.0,
                        posinf=0.0, neginf=0.0)
    return _window(eeg, fs, window_sec, overlap)


class HUSMDataset(Dataset):
    """Window-level HUSM/MUMTAZ dataset (MDD vs HC via 10-20 EEG).

    Labels are read from the filename prefix: ``H S*.edf`` -> HC (0),
    ``MDD S*.edf`` -> MDD (1). _TASK files are ignored.
    """

    def __init__(
        self,
        root: str | Path = HUSM_DIR,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        fs_target: float = 256.0,
        window_sec: float = 2.0,
        overlap: float = 0.5,
        use_ec: bool = True,
        use_eo: bool = True,
    ):
        self.root = Path(root)
        self.lowcut, self.highcut = lowcut, highcut
        self.fs_target = fs_target
        self.window_sec, self.overlap = window_sec, overlap
        self.use_ec, self.use_eo = use_ec, use_eo

        self.windows, self.labels = [], []
        self._load()
        if not self.windows:
            raise ValueError("No HUSM windows loaded")

    def _filename_label(self, name: str) -> int | None:
        if "_TASK" in name or name.upper().endswith("TASK.EDF"):
            return None
        for prefix, label in LABEL_MAP.items():
            if name.startswith(prefix + " "):
                return label
        return None

    def _load(self) -> None:
        for path in sorted(self.root.glob("*.edf")):
            name = path.name
            label = self._filename_label(name)
            if label is None:
                continue
            cond = "eo" if re.search(r"\sEO\.edf$", name, re.I) else "ec"
            if cond == "ec" and not self.use_ec:
                continue
            if cond == "eo" and not self.use_eo:
                continue
            win = _process_edf(path, self.fs_target, self.lowcut, self.highcut,
                               self.window_sec, self.overlap)
            if win is None:
                continue
            self.windows.append(torch.from_numpy(win).float())
            self.labels.append(label)
        self._n_channels = self.windows[0].shape[1]
        self._n_samples = self.windows[0].shape[2]

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def __len__(self) -> int:
        return sum(len(w) for w in self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        for win, label in zip(self.windows, self.labels):
            if idx < len(win):
                return win[idx], label
            idx -= len(win)
        raise IndexError(idx)

    def window_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.cat([w for w in self.windows], dim=0)
        ys = torch.tensor(
            [lab for lab, w in zip(self.labels, self.windows) for _ in range(len(w))],
            dtype=torch.long,
        )
        return xs, ys
