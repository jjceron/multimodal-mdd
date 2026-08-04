from __future__ import annotations

import argparse
import re
from collections import Counter
from io import StringIO
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.channel_selection import select_channel_names

mne.set_log_level("WARNING")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EEG_DIR = PROJECT_ROOT / "data/raw/modma/eeg/EEG_LZU_2015_2_resting state"
LABEL_MAP = {"MDD": 1, "HC": 0, "NC": 0}


def _window(eeg, fs, window_sec, overlap):
    win = max(round(window_sec * fs), 1)
    stride = max(round(win * (1.0 - overlap)), 1)
    n = (eeg.shape[1] - win) // stride + 1
    if n < 1:
        n, stride = 1, max(eeg.shape[1] - win, 1)
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return np.transpose(eeg[:, idx], (1, 0, 2))


class MODMADataset(Dataset):
    def __init__(
        self,
        root: str | Path = EEG_DIR,
        channels: str = "all",
        lowcut: float | None = 0.5,
        highcut: float | None = 60.0,
        notch: float | None = 50.0,
        target_fs: float | None = None,
        window_sec: float = 2.0,
        overlap: float = 0.5,
    ):
        self.root = Path(root)
        self.channels = channels
        self.lowcut, self.highcut, self.notch = lowcut, highcut, notch
        self.target_fs = None if target_fs is None else float(target_fs)
        self.window_sec, self.overlap = window_sec, overlap

        if not self.root.exists():
            raise FileNotFoundError(f"MODMA root does not exist: {self.root}")

        self.participants = self._load_participants()
        self.records = self._discover_records()
        if not self.records:
            raise ValueError(f"No EDF records found in {self.root}")

        self.channel_names = select_channel_names(channels)

        self.samples = []
        self._load_all()
        self._equalize_windows()

    def _load_participants(self) -> pd.DataFrame:
        path = self.root / "participants.tsv"
        if not path.exists():
            raise FileNotFoundError(f"participants.tsv not found at {path}")
        raw = path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(re.sub(r"\t+", "\t", raw)), sep="\t").dropna(
            subset=["group"]
        )
        df["label"] = df["group"].map(LABEL_MAP)
        missing = df["label"].isna()
        if missing.any():
            raise ValueError(
                f"Unknown group(s): {df.loc[missing, 'group'].unique().tolist()}"
            )
        df["label"] = df["label"].astype(int)
        return df

    def _discover_records(self) -> list[dict]:
        records = []
        for pid, label in zip(
            self.participants["participant_id"], self.participants["label"]
        ):
            edf = self.root / pid / "eeg" / f"{pid}_task-Resting-state_eeg.EDF"
            if edf.exists():
                records.append(
                    {"participant_id": pid, "label": int(label), "edf_path": edf}
                )
        return records

    def _load_all(self) -> None:
        for rec in self.records:
            eeg = self._process_edf(rec["edf_path"])
            if eeg is None:
                print(f"Warning: failed to load {rec['participant_id']}, skipping")
                continue
            self.samples.append(
                {
                    "participant_id": rec["participant_id"],
                    "eeg": torch.from_numpy(eeg).float(),
                    "label": torch.tensor(rec["label"], dtype=torch.long),
                }
            )

    def _equalize_windows(self) -> None:
        if not self.samples:
            raise ValueError(f"No samples loaded from {self.root}")
        min_windows = min(s["eeg"].shape[0] for s in self.samples)
        for s in self.samples:
            s["eeg"] = s["eeg"][:min_windows]
        self.min_windows = min_windows

    def _process_edf(self, path: Path) -> np.ndarray | None:
        raw = mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
        missing = [ch for ch in self.channel_names if ch not in raw.ch_names]
        if missing:
            print(f"  Missing channels in {path.name}: {missing}")
            return None

        raw.pick(self.channel_names)
        raw.reorder_channels(self.channel_names)
        raw.set_eeg_reference("average", verbose=False)

        fs = float(raw.info["sfreq"])
        if self.notch is not None and self.notch < fs / 2:
            raw.notch_filter([self.notch], verbose=False)

        highcut = self.highcut
        if highcut is not None:
            highcut = min(highcut, fs / 2 - 1e-3)
            if self.lowcut is not None and highcut <= self.lowcut:
                highcut = None

        if self.lowcut is not None or highcut is not None:
            raw.filter(
                l_freq=self.lowcut, h_freq=highcut, fir_design="firwin", verbose=False
            )

        if self.target_fs is not None and not np.isclose(fs, self.target_fs):
            raw.resample(self.target_fs, npad="auto", verbose=False)

        fs = float(raw.info["sfreq"])
        eeg = np.nan_to_num(
            raw.get_data().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        return _window(eeg, fs, self.window_sec, self.overlap)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s["participant_id"], s["eeg"], s["label"]


def create_dataloaders(
    dataset: MODMADataset,
    k_folder: int = 5,
    batch_size: int = 16,
    shuffle: bool = True,
    split_seed: int = 42,
    inner_split: int = 5,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> list[tuple[DataLoader, DataLoader, DataLoader]]:
    subjects, labels, eeg_data = [], [], []

    for s in dataset.samples:
        subjects.append(s["participant_id"])
        labels.append(int(s["label"].item()))
        eeg_data.append(s["eeg"])

    class FoldDataset(Dataset):
        def __init__(self, eeg_list, label_list, subject_indices):
            self.X, self.y, self.names = [], [], []

            for idx in subject_indices:
                self.X.append(eeg_list[idx])
                self.y.append(label_list[idx])
                self.names.append(subjects[idx])

            self.y = torch.tensor(self.y, dtype=torch.long)
            self.X = torch.stack(self.X)

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            return self.names[idx], self.X[idx], self.y[idx]

    outer_gkf = StratifiedGroupKFold(
        n_splits=k_folder,
        shuffle=shuffle,
        random_state=split_seed,
    )

    folds: list[tuple[DataLoader, DataLoader, DataLoader]] = []

    for train_val_idx, test_idx in outer_gkf.split(
        eeg_data,
        labels,
        groups=subjects,
    ):
        inner_gkf = StratifiedGroupKFold(
            n_splits=inner_split,
            shuffle=shuffle,
            random_state=split_seed,
        )

        train_idx, val_idx = next(
            inner_gkf.split(
                [eeg_data[i] for i in train_val_idx],
                [labels[i] for i in train_val_idx],
                groups=[subjects[i] for i in train_val_idx],
            )
        )

        train_subjects = [train_val_idx[i] for i in train_idx]
        val_subjects = [train_val_idx[i] for i in val_idx]

        train_dataset = FoldDataset(eeg_data, labels, train_subjects)
        val_dataset = FoldDataset(eeg_data, labels, val_subjects)
        test_dataset = FoldDataset(eeg_data, labels, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(EEG_DIR))
    parser.add_argument("--channels", type=str, default="10-20")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=60.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--show-subjects", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = MODMADataset(
        root=args.root,
        channels=args.channels,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        window_sec=args.window_sec,
        overlap=args.overlap,
    )

    n_windows = sum(s["eeg"].shape[0] for s in ds.samples)
    counts = Counter(int(s["label"]) for s in ds.samples)
    n_hc = counts.get(0, 0)
    n_mdd = counts.get(1, 0)
    split_seed = 42

    line = "=" * 60
    print(line)
    print(" MODMA EEG — dataset & nested-CV summary")
    print("-" * 60)
    print(f" Dataset : {len(ds)} subjects | {len(ds.channel_names)} channels ({args.channels})")
    print(f" Labels  : HC={n_hc} ({100 * n_hc / len(ds):.1f}%) | MDD={n_mdd} ({100 * n_mdd / len(ds):.1f}%)")
    print(f" Windows : {n_windows} total | per-subject shape {tuple(ds.samples[0]['eeg'].shape)}")
    print(f" CV      : outer k={args.k} (group+strat) | inner k={args.inner_splits} | seed={split_seed}")
    print(line)

    folds = create_dataloaders(
        ds,
        k_folder=args.k,
        batch_size=args.batch_size,
        inner_split=args.inner_splits,
        split_seed=split_seed,
    )

    def counts_str(dl):
        c = Counter(dl.dataset.y.tolist())
        return f"{c.get(0, 0)} / {c.get(1, 0)}"

    print(f"{'Fold':<6} {'Train':<9} {'HC/MDD':<13} {'Val':<8} {'HC/MDD':<12} {'Test':<9} {'HC/MDD':<10}")
    print(f"{'-'*6} {'-'*9} {'-'*13} {'-'*8} {'-'*12} {'-'*9} {'-'*10}")
    for i, (tr, va, te) in enumerate(folds):
        print(
            f"{i:<6} {len(tr.dataset):<9} {counts_str(tr):<13} "
            f"{len(va.dataset):<8} {counts_str(va):<12} "
            f"{len(te.dataset):<9} {counts_str(te):<10}"
        )

    if args.show_subjects:
        print()
        print(" Subject IDs per fold (sorted)")
        for i, (tr, va, te) in enumerate(folds):
            print(f" Fold {i}  tr=[{' '.join(sorted(tr.dataset.names))}]")
            print(f"           va=[{' '.join(sorted(va.dataset.names))}]")
            print(f"           te=[{' '.join(sorted(te.dataset.names))}]")


if __name__ == "__main__":
    main()
