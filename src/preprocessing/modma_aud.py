"""MODMA audio -> per-subject log-Mel windows (mirrors modma_eeg)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal as sg
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = PROJECT_ROOT / "data/raw/modma/aud/audio_lanzhou_2015"
AUDIO_META = AUDIO_DIR / "subjects_information_audio_lanzhou_2015.xlsx"
MAPPING_PATH = PROJECT_ROOT / "data/processed/multimodal_mapping.json"
CACHE_DIR = PROJECT_ROOT / "data/processed/aud_vad"

LABEL_MAP = {"MDD": 1, "HC": 0, "NC": 0}

SR_TARGET = 16000
N_MELS = 64
N_FFT = 1024
HOP = 160
N_FRAMES = 200          
OVERLAP = 0.5
N_WINS = 500            # cap of windows per subject
VAD_THRESHOLD = -60     # dB: drop windows below this (silence)
MIN_WINDOWS = 50        # floor: keep at least per subj
F_MIN, F_MAX = 20, 8000
TOP_DB = 80


def load_orig_to_bids() -> dict[str, str]:
    """orig audio id -> BIDS ``sub-XXX`` for the 38 paired subjects."""
    import json

    with open(MAPPING_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["orig_to_bids"]


def load_audio_labels() -> dict[str, int]:
    """orig audio id -> label (MDD=1/HC=0) for ALL audio subjects."""
    df = pd.read_excel(AUDIO_META)
    out: dict[str, int] = {}
    for subj, typ in zip(df["subject id"], df["type"]):
        orig = f"{int(subj):08d}"
        out[orig] = LABEL_MAP[str(typ).strip().upper()]
    return out


def iter_audio_subjects() -> dict[str, Path]:
    """orig audio id -> subject directory (only those with a label)."""
    labels = load_audio_labels()
    return {
        d.name: d
        for d in sorted(AUDIO_DIR.iterdir(), key=lambda p: p.name)
        if d.is_dir() and d.name in labels
    }


def _seg_audio(seg_dir: Path, sr: int) -> np.ndarray:
    import soundfile as sf

    pieces: list[np.ndarray] = []
    for wav in sorted(seg_dir.glob("*.wav"), key=lambda p: p.stem):
        try:
            data, fs = sf.read(wav.as_posix(), dtype="float32", always_2d=False)
        except Exception as e:  # noqa: BLE001 
            print(f"  skip {wav.name}: {e}")
            continue
        if fs != sr:
            data = _resample(data, fs, sr)
        pieces.append(data)
    if not pieces:
        raise FileNotFoundError(f"No readable .wav segments in {seg_dir}")
    return np.concatenate(pieces)


def _resample(data: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    import librosa

    if np.isclose(fs_in, fs_out):
        return data
    return librosa.resample(data, orig_sr=fs_in, target_sr=fs_out)


def _window_1d(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if x.shape[0] < win:
        pad = win - x.shape[0]
        x = np.pad(x, (0, pad))
    n = (x.shape[0] - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]  # [n, win]


def _sliding_windows(mel: np.ndarray, win: int, overlap: float) -> np.ndarray:
    """Full-clip log-Mel ``[n_mels, T]`` -> windows ``[W, n_mels, win]``."""
    stride = int(win * (1.0 - overlap))
    if mel.shape[1] < win:
        return np.empty((0,) + mel.shape[:1] + (win,), dtype=np.float32)
    n = (mel.shape[1] - win) // stride + 1
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return np.transpose(mel[:, idx], (1, 0, 2)).astype(np.float32)


def logmel_windows(seg_dir: Path, bandpass: bool = False,
                   overlap: float = OVERLAP) -> np.ndarray:
    """Per-window log-Mel ``[W', n_mels, N_FRAMES]`` (validated VAD recipe)."""
    import librosa

    y = _seg_audio(seg_dir, SR_TARGET)
    if bandpass:
        sos = sg.butter(4, [80, 4000], btype="band", fs=SR_TARGET, output="sos")
        y = sg.sosfiltfilt(sos, y).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR_TARGET, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        power=2.0, fmin=F_MIN, fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=TOP_DB)

    wins = _sliding_windows(mel_db, N_FRAMES, overlap)
    if wins.shape[0]:
        energy = wins.mean(axis=(1, 2))
        keep = energy >= VAD_THRESHOLD
        if keep.sum() < MIN_WINDOWS:
            order = np.argsort(energy)[::-1][:min(MIN_WINDOWS, wins.shape[0])]
            wins = wins[order]
        else:
            wins = wins[keep]
    if wins.shape[0] > N_WINS:
        rng = np.random.RandomState(42)
        wins = wins[rng.choice(wins.shape[0], N_WINS, replace=False)]
    return wins


class MODMAAudioDataset(Dataset):
    def __init__(
        self,
        root: str | Path = AUDIO_DIR,
        sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop: int = 160,
        window_sec: float = 2.0,
        overlap: float = 0.5,
        bandpass: bool = False,
        paired_only: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Audio root does not exist: {self.root}")

        self.window_sec, self.overlap = window_sec, overlap
        self.bandpass = bandpass

        self.orig_to_bids = load_orig_to_bids()
        self.labels = load_audio_labels()

        self.samples = []
        for seg_dir in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not seg_dir.is_dir():
                continue
            orig = seg_dir.name
            if orig not in self.labels:
                print(f"  skip {orig}: no label in audio metadata")
                continue
            bids = self.orig_to_bids.get(orig)
            if paired_only and bids is None:
                continue
            self.samples.append(
                {
                    "participant_id": bids or orig,
                    "bids_id": bids,
                    "orig_id": orig,
                    "dir": seg_dir,
                    "label": int(self.labels[orig]),
                }
            )
        if not self.samples:
            raise ValueError(f"No valid audio subjects in {self.root}")

        # Compute windows on the fly (no disk cache); keep per-subject variable W.
        for s in self.samples:
            s["logmel"] = self._load_logmel(s)
        self.min_windows = min(s["logmel"].shape[0] for s in self.samples)

    def _load_logmel(self, s: dict) -> torch.Tensor:
        arr = logmel_windows(s["dir"], bandpass=self.bandpass, overlap=self.overlap)
        return torch.from_numpy(arr).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s["participant_id"], s["logmel"], torch.tensor(s["label"], dtype=torch.long)


def build_audio_subjects(ds: MODMAAudioDataset) -> list[dict]:
    """Per-subject records: participant_id, orig_id, bids_id, label."""
    return [
        {
            "participant_id": s["participant_id"],
            "orig_id": s["orig_id"],
            "bids_id": s["bids_id"],
            "label": s["label"],
        }
        for s in ds.samples
    ]


def create_audio_dataloaders(dataset, k_folder=5, batch_size=16, shuffle=True,
                             split_seed=42, val_ratio=0.2, num_workers=0,
                             pin_memory=False):
    """Single-CV subject-aware dataloaders (variable windows per subject, no min truncation)."""
    subjects, labels, data = [], [], []
    for s in dataset.samples:
        subjects.append(s["participant_id"])
        labels.append(int(s["label"]))
        data.append(s["logmel"])

    class SubjectDataset(Dataset):
        def __init__(self, idx):
            self.X = [data[i] for i in idx]  # each [W_i, 64, F]
            self.y = torch.tensor([labels[i] for i in idx], dtype=torch.long)
            self.names = [subjects[i] for i in idx]

        def __len__(self):
            return len(self.names)

        def __getitem__(self, i):
            return self.names[i], self.X[i], self.y[i]

    class WindowDataset(Dataset):
        def __init__(self, idx):
            self.index = []
            self.y = []
            self.names = []
            for i in idx:
                for w in range(data[i].shape[0]):
                    self.index.append((i, w))
                    self.y.append(labels[i])
                    self.names.append(subjects[i])
            self.y = torch.tensor(self.y, dtype=torch.long)

        def __len__(self):
            return len(self.names)

        def __getitem__(self, i):
            si, wi = self.index[i]
            return self.names[i], data[si][wi], self.y[i]

    outer = StratifiedGroupKFold(n_splits=k_folder, shuffle=shuffle,
                                 random_state=split_seed)
    folds = []
    for tr_val_idx, test_idx in outer.split(data, labels, groups=subjects):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio,
                                          random_state=split_seed)
        val_labels = [labels[i] for i in tr_val_idx]
        tr_idx, val_idx = next(splitter.split(np.zeros(len(tr_val_idx)), val_labels))
        tr_sub = [tr_val_idx[i] for i in tr_idx]
        va_sub = [tr_val_idx[i] for i in val_idx]

        train_loader = DataLoader(WindowDataset(tr_sub), batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(WindowDataset(va_sub), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        outer_loader = DataLoader(WindowDataset(tr_val_idx),
                                  batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(SubjectDataset(test_idx),
                                 batch_size=1, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)
        folds.append((train_loader, val_loader, outer_loader, test_loader))
    return folds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MODMA audio preprocessing")
    p.add_argument("--root", type=str, default=str(AUDIO_DIR))
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--bandpass", action="store_true")
    p.add_argument("--show-subjects", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = MODMAAudioDataset(
        root=args.root, window_sec=args.window_sec, overlap=args.overlap,
        bandpass=args.bandpass,
    )
    n_windows = sum(s["logmel"].shape[0] for s in ds.samples)
    from collections import Counter

    counts = Counter(int(s["label"]) for s in ds.samples)
    print("=" * 60)
    print(" MODMA audio - dataset summary")
    print("-" * 60)
    print(f" Dataset : {len(ds)} subjects | window {tuple(ds.samples[0]['logmel'].shape)}")
    print(f" Labels  : HC={counts.get(0, 0)} | MDD={counts.get(1, 0)}")
    print(f" Windows : {n_windows} total | min/subj={ds.min_windows}")
    print("=" * 60)
    if args.show_subjects:
        for s in ds.samples:
            print(f"  {s['participant_id']:<10} orig={s['orig_id']} label={s['label']}")


if __name__ == "__main__":
    main()
