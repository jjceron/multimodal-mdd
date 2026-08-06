"""Subject-level EEG classification from global (aggregated) features.

Methodology (leakage-free by construction):

* Split: exactly the same StratifiedGroupKFold protocol as the window-level
  pipeline (groups=subject, outer k=5 + inner k=5, split_seed). A subject
  never crosses train/val/test, so there is no leakage at the subject level.
* Features: per-subject statistics over its windows (band-power mean,
  between-window dynamics, cross-channel topography, spectral entropy).
  These are *static* descriptors — no parameter is fit on data spanning
  folds, so nothing global can see the test subjects.
* Nested validation: within each outer fold the inner SGKF splits the
  training subjects again. StandardScaler is fit ONLY on the inner training
  portion. Logistic is trained with fixed, pre-validated hyperparameters
  (C=1.0, threshold 0.5). Test is never touched. No grid/threshold selection
  runs on the small inner validation set, which would otherwise fit noise.
* Metrics: subject-level BACC and AUC (threshold-free).

The exact same subject indices used by `create_dataloaders` are reproduced
here (same StratifiedGroupKFold construction + seeds), so folds match the
paper's partition 1:1.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import mne
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.preprocessing.modma_eeg import MODMADataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/results/unimodals/eeg"

FS = 250.0
EPS = 1e-12
BANDS = {
    "delta": (0.4, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
# no L2 regularisation grid: fixed C=1.0 validated in the subject-level probe


def band_powers(windows: np.ndarray) -> dict[str, np.ndarray]:
    """windows [W, C, T] -> {band: [W, C] band power}."""
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


def make_splits(subjects, y, X, k, inner_k, split_seed):
    """Replicate create_dataloaders' outer+inner SGKF partition 1:1."""
    outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=split_seed)
    folds = []
    for train_val_idx, test_idx in outer.split(X, y, groups=subjects):
        inner = StratifiedGroupKFold(
            n_splits=inner_k, shuffle=True, random_state=split_seed
        )
        tr_idx, va_idx = next(
            inner.split(
                X[train_val_idx], y[train_val_idx],
                groups=[subjects[i] for i in train_val_idx],
            )
        )
        train_idx = [train_val_idx[i] for i in tr_idx]
        val_idx = [train_val_idx[i] for i in va_idx]
        folds.append((train_idx, val_idx, test_idx))
    return folds


def run_subject_cv(ds, k=5, inner_k=5, split_seed=42) -> dict:
    subjects, y, X = subject_features(ds)
    folds = make_splits(subjects, y, X, k, inner_k, split_seed)

    results = []
    for fold_idx, (tr_idx, va_idx, te_idx) in enumerate(folds, start=1):
        subj_tr = [subjects[i] for i in tr_idx]
        subj_va = [subjects[i] for i in va_idx]
        subj_te = [subjects[i] for i in te_idx]
        assert set(subj_tr).isdisjoint(subj_va), "subject leak: train/val overlap"
        assert set(subj_tr).isdisjoint(subj_te), "subject leak: train/test overlap"
        assert set(subj_va).isdisjoint(subj_te), "subject leak: val/test overlap"

        # refit on train+val (still disjoint from test) with fixed C=1.0
        trva_idx = tr_idx + va_idx
        scaler = StandardScaler().fit(X[trva_idx])
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(
            scaler.transform(X[trva_idx]), y[trva_idx]
        )
        p = clf.predict_proba(scaler.transform(X[te_idx]))[:, 1]
        pred = (p >= 0.5).astype(int)

        yte = y[te_idx]
        auc = (
            float(roc_auc_score(yte, p))
            if len(set(yte.tolist())) > 1 else None
        )
        bacc = float(balanced_accuracy_score(yte, pred))
        results.append(
            {
                "fold": fold_idx,
                "test_subjects": subj_te,
                "test_true": yte.tolist(),
                "test_pred": pred.tolist(),
                "test_prob": p.tolist(),
                "test_auc": auc,
                "test_bacc": bacc,
                "hp": {"C": 1.0, "thr": 0.5},
            }
        )
        print(
            f"  fold {fold_idx}: BACC={bacc:.3f} AUC={auc if auc is None else round(auc, 3)} "
            f"(train {len(tr_idx)}+val {len(va_idx)} = {len(trva_idx)} | test {len(te_idx)})"
        )

    baccs = np.array([r["test_bacc"] for r in results])
    aucs = np.array([r["test_auc"] for r in results if r["test_auc"] is not None])
    summary = {
        "n_subjects": len(y),
        "k": k,
        "split_seed": split_seed,
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std()),
        "auc_mean": float(aucs.mean()) if len(aucs) else None,
        "auc_std": float(aucs.std()) if len(aucs) else None,
    }
    return {"summary": summary, "folds": results}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subject-level MODMA EEG features + logistic")
    p.add_argument("--channels", type=str, default="29",
                   choices=["all", "10-20", "f64", "29"])
    p.add_argument("--lowcut", type=float, default=0.4)
    p.add_argument("--highcut", type=float, default=45.0)
    p.add_argument("--notch", type=float, default=50.0)
    p.add_argument("--reference", type=str, default="average",
                   choices=["average", "cz"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--split-seed", type=int, nargs="+", default=[42])
    p.add_argument("--tag", type=str, default="subject")
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = MODMADataset(
        channels=args.channels,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        reference=args.reference,
    )
    gpu = "CPU"
    print("=" * 78)
    print(" MODAL / MODEL        : eeg / subject-features + logistic")
    print(f" DEVICE               : {gpu}")
    print(f" DATASET              : channels={args.channels} "
          f"({len(ds.channel_names)}) subjects={len(ds.samples)} "
          f"windows/subj={ds.samples[0]['eeg'].shape[0]}")
    print(f" TRAINING CONFIG      : k={args.k} inner={args.inner_splits} "
          f"seed(s)={args.split_seed} channels={args.channels}")
    print("=" * 78)

    _, _, Xfeat = subject_features(ds)
    n_features = int(Xfeat.shape[1])

    for split_seed in args.split_seed:
        print(f"\n=== SPLIT SEED {split_seed} ===")
        res = run_subject_cv(ds, k=args.k, inner_k=args.inner_splits,
                             split_seed=split_seed)
        s = res["summary"]
        print(
            f"  -> BACC {s['bacc_mean']:.3f}+/-{s['bacc_std']:.3f}  "
            f"AUC {s['auc_mean'] if s['auc_mean'] is None else round(s['auc_mean'], 3)}"
        )

        out_dir = (
            Path(args.output_root)
            / f"unimodals_sgkf_eeg_sseed{split_seed}_tag{args.tag}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "modal": "eeg",
            "tag": args.tag,
            "split_seed": split_seed,
            "channels": args.channels,
            "model": "subject-features+logistic",
            "n_channels": len(ds.channel_names),
            "config": {
                "data": {
                    "channels": args.channels,
                    "reference": args.reference,
                    "lowcut": args.lowcut,
                    "highcut": args.highcut,
                    "notch": args.notch,
                    "features": "band mean/topo/dyn + spectral entropy",
                    "n_features": n_features,
                },
                "cv": {
                    "k": args.k,
                    "inner_splits": args.inner_splits,
                    "split_seed": split_seed,
                },
                "model": "LogisticRegression L2 (fixed C=1.0, threshold 0.5)",
                "threshold": "fixed 0.5",
                "environment": {
                    "device": gpu,
                    "python": platform.python_version(),
                    "sklearn": sklearn.__version__,
                    "mne": mne.__version__,
                },
                "cli": vars(args),
            },
            "folds": res["folds"],
        }
        (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
        print(f"  saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
