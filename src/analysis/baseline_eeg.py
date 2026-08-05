"""Linear baseline to test whether EEG signal is present in MODMA.

Reuses the exact nested CV splits from ``modma_eeg.create_dataloaders`` and the
same ``MODMADataset``. Features are log band-power per channel per 2s window
(delta/theta/alpha/beta/gamma). A logistic regression is trained on window
features and evaluated per subject via probability averaging.

Also runs a subject-level permutation test: labels are shuffled across subjects
and the mean balanced accuracy is recomputed to obtain a p-value that the
observed performance is better than chance.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing.modma_eeg import MODMADataset, create_dataloaders
from src.utils.training_logger import ClassificationLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "tmp" / "baseline_eeg_results.json"

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 60.0),
}


def band_features(x: torch.Tensor, fs: float) -> torch.Tensor:
    """[S, W, C, T] -> [S, W, C, len(BANDS)] with log1p band power."""
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    spec = torch.fft.rfft(x, dim=-1).abs().square()
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(spec[..., mask].sum(dim=-1))
    return torch.log1p(torch.stack(feats, dim=-1))


def feature_bank(samples, fs: float) -> dict[str, torch.Tensor]:
    return {s["participant_id"]: band_features(s["eeg"], fs) for s in samples}


def fold_matrices(names, bank, y) -> tuple[np.ndarray, np.ndarray, int]:
    feats = torch.stack([bank[n] for n in names])  # [S, W, C, B]
    subjects, windows, n_channels, n_bands = feats.shape
    X = feats.reshape(subjects * windows, n_channels * n_bands).numpy()
    labels = np.repeat(np.asarray(y, dtype=int), windows)
    return X, labels, windows


def run_seed(ds, seed, k, inner, bank, batch_size):
    folds = create_dataloaders(
        ds,
        k_folder=k,
        inner_split=inner,
        split_seed=seed,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    fold_results = []
    for tr_loader, _val_loader, te_loader in folds:
        X_tr, y_tr, _ = fold_matrices(
            tr_loader.dataset.names, bank, tr_loader.dataset.y.tolist()
        )
        X_te, _, _ = fold_matrices(
            te_loader.dataset.names, bank, te_loader.dataset.y.tolist()
        )

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs"
        ).fit(X_tr, y_tr)

        prob_win = clf.predict_proba(X_te)[:, 1]
        subjects = len(te_loader.dataset.names)
        prob_subj = prob_win.reshape(subjects, -1).mean(axis=1)
        pred_subj = (prob_subj >= 0.5).astype(int)

        m = ClassificationLogger().metrics(te_loader.dataset.y.tolist(), pred_subj)
        auc = float(roc_auc_score(te_loader.dataset.y.tolist(), prob_subj))
        fold_results.append(
            {
                "fold": len(fold_results) + 1,
                "n_test_subjects": subjects,
                "test_subjects": list(te_loader.dataset.names),
                "bacc": m["bacc"],
                "sens": m["sens"],
                "spec": m["spec"],
                "auc": auc,
            }
        )
    return fold_results


def mean_bacc(fold_results) -> float:
    return float(np.mean([f["bacc"] for f in fold_results]))


def aggregate(fold_results):
    keys = ["bacc", "sens", "spec", "auc"]
    means = {k: float(np.mean([f[k] for f in fold_results])) for k in keys}
    stds = {k + "_std": float(np.std([f[k] for f in fold_results])) for k in keys}
    return {**means, **stds}


def permute_samples(samples, rng):
    permuted = copy.deepcopy(samples)
    labels = [s["label"] for s in permuted]
    rng.shuffle(labels)
    for s, lab in zip(permuted, labels):
        s["label"] = lab
    return permuted


class _Samples:
    def __init__(self, samples):
        self.samples = samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear EEG baseline (MODMA)")
    parser.add_argument("--channels", type=str, default="f64",
                        choices=["all", "10-20", "f64"])
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, nargs="+",
                        default=[42, 1825, 4013, 410, 4507])
    parser.add_argument("--perm-seed", type=int, default=42)
    parser.add_argument("--n-perms", type=int, default=50)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    print("Loading dataset ...")
    ds = MODMADataset(channels=args.channels, overlap=args.overlap)
    fs = ds.samples[0]["eeg"].shape[-1] / ds.window_sec
    print(f"  subjects={len(ds.samples)} channels={len(ds.channel_names)} "
          f"fs~{fs:.0f}Hz bands={list(BANDS)}")
    bank = feature_bank(ds.samples, fs)

    per_seed = {}
    all_folds = []
    for seed in args.split_seed:
        fold_results = run_seed(ds, seed, args.k, args.inner_splits, bank,
                                args.batch_size)
        per_seed[seed] = aggregate(fold_results)
        all_folds.extend(fold_results)
        print(f"\n=== seed={seed} ===")
        for f in fold_results:
            print(f"  Fold {f['fold']}: BACC={f['bacc']:.3f} "
                  f"Sens={f['sens']:.3f} Spec={f['spec']:.3f} "
                  f"AUC={f['auc']:.3f} (n_test={f['n_test_subjects']})")
        print(f"  mean bacc = {mean_bacc(fold_results):.3f}")

    overall = aggregate(all_folds)

    print("\n=== PERMUTATION TEST (subject-level label shuffle) ===")
    obs = mean_bacc(run_seed(ds, args.perm_seed, args.k, args.inner_splits,
                             bank, args.batch_size))
    rng = random.Random(args.perm_seed)
    null = []
    for i in range(args.n_perms):
        perm = permute_samples(ds.samples, rng)
        folds = run_seed(_Samples(perm), args.perm_seed, args.k,
                         args.inner_splits, bank, args.batch_size)
        null.append(mean_bacc(folds))
        if (i + 1) % 10 == 0:
            print(f"  perm {i + 1}/{args.n_perms} done")
    null = np.asarray(null)
    p_value = float((1 + (null >= obs).sum()) / (args.n_perms + 1))

    print(f"\n  observed mean bacc = {obs:.4f}")
    print(f"  null mean = {null.mean():.4f} +- {null.std():.4f}")
    print(f"  p-value (mean bacc) = {p_value:.4f}  -> "
          f"{'SIGNAL DETECTED' if p_value < 0.05 else 'no significant signal'}")

    out = {
        "config": {
            "channels": args.channels,
            "overlap": args.overlap,
            "k": args.k,
            "inner_splits": args.inner_splits,
            "split_seeds": args.split_seed,
            "bands": {k: v for k, v in BANDS.items()},
            "model": "LogisticRegression (balanced)",
            "window_sec": ds.window_sec,
            "fs": fs,
        },
        "per_seed": per_seed,
        "overall": overall,
        "permutation": {
            "n_perms": args.n_perms,
            "perm_seed": args.perm_seed,
            "observed_mean_bacc": obs,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "p_value": p_value,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
