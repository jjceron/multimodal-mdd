"""MODMA audio -> per-subject log-Mel windows (mirrors modma_eeg)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = PROJECT_ROOT / "data/raw/modma/aud/audio_lanzhou_2015"
AUDIO_META = AUDIO_DIR / "subjects_information_audio_lanzhou_2015.xlsx"
MAPPING_PATH = PROJECT_ROOT / "data/processed/multimodal_mapping.json"
CACHE_DIR = PROJECT_ROOT / "data/processed/aud"

LABEL_MAP = {"MDD": 1, "HC": 0, "NC": 0}


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
        except Exception as e:  # noqa: BLE001 - skip corrupt/unrecognized files
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


def logmel_windows(
    seg_dir: Path,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop: int = 160,
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """Per-window log-Mel spectrograms ``[W, n_mels, n_frames]``."""
    import librosa

    y = _seg_audio(seg_dir, sr)
    win = max(round(window_sec * sr), 1)
    stride = max(round(win * (1.0 - overlap)), 1)
    frames = _window_1d(y, win, stride)

    specs: list[np.ndarray] = []
    for w in frames:
        mel = librosa.feature.melspectrogram(
            y=w, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )
        specs.append(librosa.power_to_db(mel, ref=np.max))
    return np.asarray(specs, dtype=np.float32)


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
        cache: bool = True,
        paired_only: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Audio root does not exist: {self.root}")

        self.sr, self.n_mels = sr, n_mels
        self.n_fft, self.hop = n_fft, hop
        self.window_sec, self.overlap = window_sec, overlap
        self.cache = cache

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

        # Cache or compute windows for every subject.
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for s in self.samples:
            s["logmel"] = self._load_logmel(s)

        min_windows = min(s["logmel"].shape[0] for s in self.samples)
        for s in self.samples:
            s["logmel"] = s["logmel"][:min_windows]
        self.min_windows = min_windows

    def _load_logmel(self, s: dict) -> torch.Tensor:
        cache_path = CACHE_DIR / f"{s['orig_id']}.npy"
        if self.cache and cache_path.exists():
            arr = np.load(cache_path)
        else:
            arr = logmel_windows(
                s["dir"], self.sr, self.n_mels, self.n_fft, self.hop,
                self.window_sec, self.overlap,
            )
            if self.cache:
                np.save(cache_path, arr)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MODMA audio preprocessing")
    p.add_argument("--root", type=str, default=str(AUDIO_DIR))
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=160)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--show-subjects", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = MODMAAudioDataset(
        root=args.root, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft,
        hop=args.hop, window_sec=args.window_sec, overlap=args.overlap,
        cache=not args.no_cache,
    )
    n_windows = sum(s["logmel"].shape[0] for s in ds.samples)
    from collections import Counter

    counts = Counter(int(s["label"].item()) for s in ds.samples)
    print("=" * 60)
    print(" MODMA audio - dataset summary")
    print("-" * 60)
    print(f" Dataset : {len(ds)} subjects | window {tuple(ds.samples[0]['logmel'].shape)}")
    print(f" Labels  : HC={counts.get(0, 0)} | MDD={counts.get(1, 0)}")
    print(f" Windows : {n_windows} total | min/subj={ds.min_windows}")
    print(f" Paired  : {sum(1 for s in ds.samples if s['bids_id'])}")
    print(f" Cache   : {CACHE_DIR}")
    print("=" * 60)
    if args.show_subjects:
        for s in ds.samples:
            print(f"  {s['participant_id']:<10} orig={s['orig_id']} label={s['label']}")


if __name__ == "__main__":
    main()
