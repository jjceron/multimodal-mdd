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

from src.preprocessing.channel_selection import (
    CLINICAL_10_20_EGI_MAP,
    select_channel_names,
)

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
        reference: str = "average",
    ):
        self.root = Path(root)
        self.channels = channels
        self.lowcut, self.highcut, self.notch = lowcut, highcut, notch
        self.target_fs = None if target_fs is None else float(target_fs)
        self.window_sec, self.overlap = window_sec, overlap
        self.reference = reference

        if not self.root.exists():
            raise FileNotFoundError(f"MODMA root does not exist: {self.root}")

        self.participants = self._load_participants()
        self.records = self._discover_records()
        if not self.records:
            raise ValueError(f"No EDF records found in {self.root}")

        self.channel_names = select_channel_names(channels)

        self.samples = []
        self._load_all()

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

    def _process_edf(self, path: Path) -> np.ndarray | None:
        raw = mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
        missing = [ch for ch in self.channel_names if ch not in raw.ch_names]
        if missing:
            print(f"  Missing channels in {path.name}: {missing}")
            return None

        raw.pick(self.channel_names)
        raw.reorder_channels(self.channel_names)
        self._set_reference(raw)

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
        windows = _window(eeg, fs, self.window_sec, self.overlap)
        return windows

    def _set_reference(self, raw: mne.io.BaseRaw) -> None:
        """Apply the EEG reference. ``cz`` requires Cz to be among the channels."""
        if self.reference == "cz":
            cz = f"E{CLINICAL_10_20_EGI_MAP['Cz'] + 1}"
            if cz not in raw.ch_names:
                raise ValueError(
                    f"reference='cz' requires Cz ({cz}) among the selected "
                    f"channels; preset {self.channels!r} does not include it"
                )
            raw.set_eeg_reference(ref_channels=[cz], verbose=False)
        else:
            raw.set_eeg_reference("average", verbose=False)

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
) -> list[tuple[list[tuple[DataLoader, DataLoader]], DataLoader, DataLoader]]:
    subjects, labels, eeg_data = [], [], []

    for s in dataset.samples:
        subjects.append(s["participant_id"])
        labels.append(int(s["label"].item()))
        eeg_data.append(s["eeg"])

    class SubjectDataset(Dataset):
        def __init__(self, subject_indices, n_windows):
            self.X = [eeg_data[idx][:n_windows] for idx in subject_indices]
            self.y = torch.tensor(
                [labels[idx] for idx in subject_indices],
                dtype=torch.long,
            )
            self.names = [subjects[idx] for idx in subject_indices]

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            return self.names[idx], self.X[idx], self.y[idx]

    class WindowDataset(Dataset):
        def __init__(self, subject_indices, n_windows):
            self.index = []
            self.y = []
            self.names = []
            for idx in subject_indices:
                for window_index in range(n_windows):
                    self.index.append((idx, window_index))
                    self.y.append(labels[idx])
                    self.names.append(subjects[idx])
            self.y = torch.tensor(self.y, dtype=torch.long)

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            subject_index, window_index = self.index[idx]
            return (
                self.names[idx],
                eeg_data[subject_index][window_index],
                self.y[idx],
            )

    outer_gkf = StratifiedGroupKFold(
        n_splits=k_folder,
        shuffle=shuffle,
        random_state=split_seed,
    )

    folds: list[tuple[list[tuple[DataLoader, DataLoader]], DataLoader, DataLoader]] = []

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

        n_windows = min(
            eeg_data[idx].shape[0]
            for idx in train_val_idx
        )

        inner_folds = []
        for train_idx, val_idx in inner_gkf.split(
                [eeg_data[i] for i in train_val_idx],
                [labels[i] for i in train_val_idx],
                groups=[subjects[i] for i in train_val_idx],
            ):
            train_subjects = [train_val_idx[i] for i in train_idx]
            val_subjects = [train_val_idx[i] for i in val_idx]
            train_dataset = WindowDataset(train_subjects, n_windows)
            val_dataset = WindowDataset(val_subjects, n_windows)
            inner_folds.append((
                DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory),
                DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory),
            ))

        outer_dataset = WindowDataset(train_val_idx, n_windows)
        test_dataset = SubjectDataset(test_idx, n_windows)

        outer_names = set(outer_dataset.names)
        test_names = set(test_dataset.names)
        if outer_names & test_names:
            raise RuntimeError("Subject overlap detected between EEG folds")

        outer_loader = DataLoader(
            outer_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        folds.append((inner_folds, outer_loader, test_loader))

    return folds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(EEG_DIR))
    parser.add_argument("--channels", type=str, default="10-20")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=50.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--reference", type=str, default="average",
                        choices=["average", "cz"])
    parser.add_argument("--split-seed", type=int, default=2509)
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
        reference=args.reference,
    )

    subject_ids = sorted(
        sample["participant_id"]
        for sample in ds.samples
    )

    subject_alias = {
        subject_id: f"S{index:02d}"
        for index, subject_id in enumerate(subject_ids, start=1)
    }

    label_by_subject = {
        sample["participant_id"]: int(sample["label"].item())
        for sample in ds.samples
    }

    windows_per_subject = [
        sample["eeg"].shape[0]
        for sample in ds.samples
    ]

    n_windows_total = sum(windows_per_subject)
    n_windows_min = min(windows_per_subject)
    n_windows_max = max(windows_per_subject)
    n_windows_mean = np.mean(windows_per_subject)

    counts = Counter(label_by_subject.values())

    print("=" * 88)
    print(" MODMA EEG - dataset and nested-CV summary")
    print("=" * 88)
    print(
        f" Dataset       : {len(ds)} subjects | "
        f"{len(ds.channel_names)} channels ({args.channels})"
    )
    print(
        f" Labels        : HC={counts.get(0, 0)} "
        f"({100 * counts.get(0, 0) / len(ds):.1f}%) | "
        f"MDD={counts.get(1, 0)} "
        f"({100 * counts.get(1, 0) / len(ds):.1f}%)"
    )
    print(
        f" Windows       : total={n_windows_total} | "
        f"min={n_windows_min} | max={n_windows_max} | "
        f"mean={n_windows_mean:.2f}"
    )
    print(
        f" Representation : [{n_windows_min}, "
        f"{len(ds.channel_names)}, "
        f"{ds.samples[0]['eeg'].shape[-1]}]"
    )
    print(
        f" CV            : SGKF | outer={args.k} | "
        f"inner={args.inner_splits} | seed={args.split_seed}"
    )
    print("=" * 88)

    folds = create_dataloaders(
        ds,
        k_folder=args.k,
        batch_size=args.batch_size,
        inner_split=args.inner_splits,
        split_seed=args.split_seed,
    )

    def unique_subjects(loader):
        return sorted(set(loader.dataset.names))

    def subject_counts(names):
        labels = [
            label_by_subject[name]
            for name in names
        ]
        return (
            f"HC={labels.count(0)}, "
            f"MDD={labels.count(1)}"
        )

    print()
    print(" OUTER FOLD SUMMARY")
    print("-" * 88)
    print(
        f"{'Fold':<8}"
        f"{'Train subjects':<18}"
        f"{'Train labels':<22}"
        f"{'Inner folds':<14}"
        f"{'Test subjects':<18}"
        f"{'Test labels':<20}"
    )
    print("-" * 88)

    fold_details = []

    for fold_index, (
        inner_folds,
        outer_loader,
        test_loader,
    ) in enumerate(folds, start=1):
        outer_train_subjects = unique_subjects(outer_loader)
        test_subjects = unique_subjects(test_loader)

        train_set = set(outer_train_subjects)
        test_set = set(test_subjects)

        if train_set & test_set:
            raise RuntimeError(
                f"Leakage in outer fold {fold_index}"
            )

        print(
            f"{fold_index:<8}"
            f"{len(outer_train_subjects):<18}"
            f"{subject_counts(outer_train_subjects):<22}"
            f"{len(inner_folds):<14}"
            f"{len(test_subjects):<18}"
            f"{subject_counts(test_subjects):<20}"
        )

        fold_details.append(
            (
                fold_index,
                inner_folds,
                outer_train_subjects,
                test_subjects,
            )
        )

    print("-" * 88)

    print()
    print(" SUBJECT ASSIGNMENT BY FOLD")
    print("-" * 88)

    for (
        fold_index,
        inner_folds,
        outer_train_subjects,
        test_subjects,
    ) in fold_details:
        print()
        print(f"OUTER FOLD {fold_index}")
        print("-" * 88)

        print(
            "Outer train+validation: "
            + " ".join(
                subject_alias[name]
                for name in outer_train_subjects
            )
        )

        print(
            "Outer test:             "
            + " ".join(
                subject_alias[name]
                for name in test_subjects
            )
        )

        print()
        print("INNER FOLDS")

        for inner_index, (
            inner_train_loader,
            inner_val_loader,
        ) in enumerate(inner_folds, start=1):
            inner_train_subjects = unique_subjects(
                inner_train_loader
            )
            inner_val_subjects = unique_subjects(
                inner_val_loader
            )

            inner_train_set = set(inner_train_subjects)
            inner_val_set = set(inner_val_subjects)

            if inner_train_set & inner_val_set:
                raise RuntimeError(
                    f"Leakage in outer fold {fold_index}, "
                    f"inner fold {inner_index}"
                )

            if inner_train_set & set(test_subjects):
                raise RuntimeError(
                    f"Outer test leakage in outer fold {fold_index}, "
                    f"inner fold {inner_index}"
                )

            if inner_val_set & set(test_subjects):
                raise RuntimeError(
                    f"Outer test leakage in outer fold {fold_index}, "
                    f"inner fold {inner_index}"
                )

            print()
            print(f"Inner fold {inner_index}")
            print(
                "  train: "
                + " ".join(
                    subject_alias[name]
                    for name in inner_train_subjects
                )
            )
            print(
                "  val:   "
                + " ".join(
                    subject_alias[name]
                    for name in inner_val_subjects
                )
            )
            print(
                f"  counts: train={len(inner_train_subjects)} "
                f"({subject_counts(inner_train_subjects)}), "
                f"val={len(inner_val_subjects)} "
                f"({subject_counts(inner_val_subjects)})"
            )

    print()
    print(" SUBJECT ALIAS TABLE")
    print("-" * 40)

    for subject_id in subject_ids:
        print(
            f"{subject_alias[subject_id]} -> "
            f"{subject_id} "
            f"({('MDD' if label_by_subject[subject_id] else 'HC')})"
        )

if __name__ == "__main__":
    main()
