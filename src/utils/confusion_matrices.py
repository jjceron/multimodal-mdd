"""Generate confusion matrices for EEG only, AUD only, and Cross-Attn Fusion (1x3 layout) - overall across all folds."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models.crossattnfusion import CrossAttnFusion
from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.preprocessing.modma_dataset import MODMASubjects
from src.training.modma_multimodal import (
    _subject_logit,
    load_dataset,
    PROJECT_ROOT,
)
from src.utils.get_seed import set_seed

matplotlib.rcParams.update({"font.size": 10})


def _load_training_config(tag, split_seed, PROJECT_ROOT):
    """Load training config from results.json to match dataset params."""
    import json
    results_dir = PROJECT_ROOT / "outputs/results/multimodals" / f"tag{tag}_sseed{split_seed}_sngkf_multimodal"
    config_path = results_dir / "results.json"
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        return data.get("config", {}).get("cli", {})
    return {}


def _build_dataset_args(args, train_config):
    """Build args object for load_dataset with required fields from training config."""
    class DatasetArgs:
        def __init__(self, a, tc):
            self.channels = tc.get("channels", "f64")
            self.target_fs = tc.get("target_fs", 250.0)
            self.lowcut = tc.get("lowcut", 0.5)
            self.highcut = tc.get("highcut", 50.0)
            self.notch = tc.get("notch", 50.0)
            self.reference = tc.get("reference", "average")
            self.overlap = tc.get("overlap", 0.5)
            self.window_sec = tc.get("window_sec", 2.0)
    return DatasetArgs(args, train_config)


def _window_norm(v):
    return (v - v.mean(dim=(1, 2), keepdim=True)) / (v.std(dim=(1, 2), keepdim=True) + 1e-8)


def _evaluate_single_modality(backbone, test_ids, subj, device, key, chunk=128):
    """Evaluate a single modality backbone (EEG or AUD) on test set."""
    backbone.eval()
    true, pred, prob = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_ids), 8):
            chunk_ids = test_ids[i : i + 8]
            blocks = []
            labels = []
            for pid in chunk_ids:
                blocks.append(subj.paired[pid][key])
                labels.append(subj.paired[pid]["label"])
            if not blocks:
                continue
            x = torch.stack(blocks).to(device)
            y = torch.tensor(labels, dtype=torch.float32, device=device)
            B, K = x.shape[0], x.shape[1]
            flat = x.reshape(B * K, *x.shape[2:])
            outs = []
            for j in range(0, flat.shape[0], chunk):
                outs.append(backbone(_window_norm(flat[j : j + chunk])).squeeze(-1))
            logits = torch.cat(outs, 0).reshape(B, K)
            logits_mean = logits.mean(dim=1)
            p = torch.sigmoid(logits_mean)
            prob.extend(p.cpu().tolist())
            pred.extend((p >= 0.5).long().cpu().tolist())
            true.extend(y.tolist())
    return np.array(true), np.array(pred), np.array(prob)


def _evaluate_fusion(beeg, baud, fusion, test_ids, subj, device):
    """Evaluate cross-attention fusion on test set."""
    beeg.eval()
    baud.eval()
    fusion.eval()
    true, prob = [], []
    with torch.no_grad():
        for i in range(0, len(test_ids), 8):
            chunk = test_ids[i : i + 8]
            e = torch.stack([subj.paired[pid]["eeg"] for pid in chunk]).to(device)
            a = torch.stack([subj.paired[pid]["aud"] for pid in chunk]).to(device)
            y = torch.tensor([subj.paired[pid]["label"] for pid in chunk], dtype=torch.float32, device=device)
            logit = _subject_logit(beeg, baud, fusion, e, a, device)
            prob.extend(torch.sigmoid(logit / fusion.temperature).tolist())
            true.extend(y.tolist())
    true = np.array(true, dtype=int)
    prob = np.array(prob)
    return true, (prob >= 0.5).astype(int), prob


def _load_fold_model(fold_idx, base, eeg_n_channels, eeg_n_samples, aud_n_channels, aud_n_samples, device):
    """Load model checkpoint for a specific fold."""
    model_dir = base / f"fold_{fold_idx}.pt"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_dir}")
    ckpt = torch.load(model_dir, map_location=device)
    beeg = DeepConvNet(eeg_n_channels, 1, eeg_n_samples, 0.5).to(device)
    baud = ShallowConvNet(aud_n_channels, 1, aud_n_samples, 0.5).to(device)
    dim_e = beeg.fc_features
    dim_a = baud.classifier.in_features
    fusion = CrossAttnFusion(dim_e, dim_a, 32, 2).to(device)
    beeg.load_state_dict(ckpt["beeg"])
    baud.load_state_dict(ckpt["baud"])
    fusion.load_state_dict(ckpt["fusion"])
    beeg.requires_grad_(False)
    baud.requires_grad_(False)
    return beeg, baud, fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_config = _load_training_config(args.tag, args.split_seed, PROJECT_ROOT)
    ds_args = _build_dataset_args(args, train_config)
    eeg_ds, aud_ds = load_dataset(ds_args)
    subj = MODMASubjects(eeg_ds, aud_ds)

    # Apply kmin truncation (same as training) to ensure uniform window counts
    eeg_min = min(
        [subj.paired[p]["eeg"].shape[0] for p in subj.paired]
        + [subj.non_paired_eeg[p]["eeg"].shape[0] for p in subj.non_paired_eeg]
    )
    aud_min = min(
        [subj.paired[p]["aud"].shape[0] for p in subj.paired]
        + [subj.non_paired_aud[p]["aud"].shape[0] for p in subj.non_paired_aud]
    )
    kmin = min(eeg_min, aud_min)
    for p in subj.paired:
        subj.paired[p]["eeg"] = subj.paired[p]["eeg"][:kmin]
        subj.paired[p]["aud"] = subj.paired[p]["aud"][:kmin]
    for p in subj.non_paired_eeg:
        subj.non_paired_eeg[p]["eeg"] = subj.non_paired_eeg[p]["eeg"][:kmin]
    for p in subj.non_paired_aud:
        subj.non_paired_aud[p]["aud"] = subj.non_paired_aud[p]["aud"][:kmin]
    print(f"Shared window count (kmin) = {kmin}")

    folds = subj.folds(k=5, val_ratio=0.3, split_seed=args.split_seed)

    eeg_n_channels = len(eeg_ds.channel_names)
    eeg_n_samples = int(eeg_ds.samples[0]["eeg"].shape[-1])
    aud_n_channels = int(aud_ds.samples[0]["logmel"].shape[1])
    aud_n_samples = int(aud_ds.samples[0]["logmel"].shape[-1])

    base = PROJECT_ROOT / "outputs/results/multimodals" / f"tag{args.tag}_sseed{args.split_seed}_sngkf_multimodal"

    # Accumulate predictions across all folds
    all_true_eeg, all_pred_eeg = [], []
    all_true_aud, all_pred_aud = [], []
    all_true_fus, all_pred_fus = [], []

    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_idx} ===")
        beeg, baud, fusion = _load_fold_model(
            fold_idx, base, eeg_n_channels, eeg_n_samples, aud_n_channels, aud_n_samples, device
        )

        print("  Evaluating EEG only...")
        t1, p1, _ = _evaluate_single_modality(beeg, test_ids, subj, device, "eeg")
        print("  Evaluating AUD only...")
        t2, p2, _ = _evaluate_single_modality(baud, test_ids, subj, device, "aud")
        print("  Evaluating Fusion...")
        t3, p3, _ = _evaluate_fusion(beeg, baud, fusion, test_ids, subj, device)

        all_true_eeg.append(t1)
        all_pred_eeg.append(p1)
        all_true_aud.append(t2)
        all_pred_aud.append(p2)
        all_true_fus.append(t3)
        all_pred_fus.append(p3)

    # Concatenate all fold predictions
    t1_all = np.concatenate(all_true_eeg)
    p1_all = np.concatenate(all_pred_eeg)
    t2_all = np.concatenate(all_true_aud)
    p2_all = np.concatenate(all_pred_aud)
    t3_all = np.concatenate(all_true_fus)
    p3_all = np.concatenate(all_pred_fus)

    cm1 = confusion_matrix(t1_all, p1_all)
    cm2 = confusion_matrix(t2_all, p2_all)
    cm3 = confusion_matrix(t3_all, p3_all)

    out = PROJECT_ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    titles = ["EEG only (overall)", "AUD only (overall)", "Cross-Attn Fusion (overall)"]
    cms = [cm1, cm2, cm3]

    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HC", "MDD"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["HC", "MDD"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Count")
    fig.savefig(out / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()