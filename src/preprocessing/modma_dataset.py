from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def paired_subjects(eeg_ds, aud_ds) -> list[str]:
    eeg_ids = {s["participant_id"] for s in eeg_ds.samples}
    aud_ids = {s["participant_id"] for s in aud_ds.samples}
    return sorted(eeg_ids & aud_ids)


def _label(s):
    return int(s["label"].item()) if hasattr(s["label"], "item") else int(s["label"])


def _to_dict(ds, key):
    out = {}
    for s in ds.samples:
        out[s["participant_id"]] = {"label": _label(s), key: s[key]}
    return out


def _non_paired(paired: set, ds, src_key, dst_key):
    out = {}
    for s in ds.samples:
        pid = s["participant_id"]
        if pid not in paired:
            out[pid] = {"label": _label(s), dst_key: s[src_key]}
    return out


class MODMASubjects:
    def __init__(self, eeg_ds, aud_ds):
        self.paired: dict[str, dict] = {}
        paired = set(paired_subjects(eeg_ds, aud_ds))
        eeg_by_id = _to_dict(eeg_ds, "eeg")
        aud_by_id = _to_dict(aud_ds, "logmel")
        for pid in sorted(paired):
            self.paired[pid] = {
                "label": eeg_by_id[pid]["label"],
                "eeg": eeg_by_id[pid]["eeg"],
                "aud": aud_by_id[pid]["logmel"],
            }
        self.non_paired_eeg = _non_paired(paired, eeg_ds, "eeg", "eeg")
        self.non_paired_aud = _non_paired(paired, aud_ds, "logmel", "aud")

    def pairs_arrays(self):
        ids = sorted(self.paired)
        labels = [self.paired[i]["label"] for i in ids]
        eeg = [self.paired[i]["eeg"] for i in ids]
        aud = [self.paired[i]["aud"] for i in ids]
        return ids, labels, eeg, aud

    def folds(self, k: int = 5, val_ratio: float = 0.2, split_seed: int = 42):
        ids, labels, _, _ = self.pairs_arrays()
        ids_np = np.array(ids)
        labels_np = np.array(labels)
        outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=split_seed)
        folds = []
        for train_val_idx, test_idx in outer.split(ids_np, labels_np, groups=ids_np):
            tvi = ids_np[train_val_idx]
            tei = ids_np[test_idx]
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=split_seed
            )
            val_labels = [self.paired[i]["label"] for i in tvi]
            tr_idx, val_idx = next(splitter.split(np.zeros(len(tvi)), val_labels))
            train_ids = [tvi[i] for i in tr_idx]
            val_ids = [tvi[i] for i in val_idx]
            test_ids = list(tei)
            folds.append((train_ids, val_ids, test_ids))
        return folds

    def get_subject_stats(self, modality: str = "both") -> dict:
        """Compute per-subject mean/std for normalization (like unimodal subj_stats)."""
        stats = {}
        if modality in ("both", "eeg"):
            for pid, data in self.paired.items():
                eeg = data["eeg"]
                stats[pid] = stats.get(pid, {})
                stats[pid]["eeg_mean"] = float(eeg.mean())
                stats[pid]["eeg_std"] = float(eeg.std()) + 1e-8
            for pid, data in self.non_paired_eeg.items():
                eeg = data["eeg"]
                stats[pid] = stats.get(pid, {})
                stats[pid]["eeg_mean"] = float(eeg.mean())
                stats[pid]["eeg_std"] = float(eeg.std()) + 1e-8
        if modality in ("both", "aud"):
            for pid, data in self.paired.items():
                aud = data["aud"]
                stats[pid] = stats.get(pid, {})
                stats[pid]["aud_mean"] = float(aud.mean())
                stats[pid]["aud_std"] = float(aud.std()) + 1e-8
            for pid, data in self.non_paired_aud.items():
                aud = data["aud"]
                stats[pid] = stats.get(pid, {})
                stats[pid]["aud_mean"] = float(aud.mean())
                stats[pid]["aud_std"] = float(aud.std()) + 1e-8
        return stats
