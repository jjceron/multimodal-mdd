"""Unimodal EEG classification on MODMA with nested group K-fold CV.

Each model is trained on 2-second EEG windows extracted per subject. Validation
tracks window-level metrics every epoch (early stopping on val accuracy). The
final evaluation is done per subject via majority vote over its windows.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.models.cnn_lstm import CNNLSTM
from src.models.deepconvnet import DeepConvNet
from src.models.eegnet import EEGNet
from src.models.shallowconvnet import ShallowConvNet
from src.preprocessing.modma_eeg import MODMADataset, create_dataloaders
from src.utils.get_seed import set_seed
from src.utils.training_logger import ClassificationLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/results/unimodals/eeg"
NUM_WORKERS = 4 if platform.system() != "Windows" else 0

MODEL_CLASSES = {
    "deepconvnet": DeepConvNet,
    "shallowconvnet": ShallowConvNet,
    "cnn_lstm": CNNLSTM,
    "eegnet": EEGNet,
}


def forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass returning only the class logits tensor."""
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def flatten_batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Flatten subject windows into the batch dimension."""
    n_subjects, windows, channels, samples = x.shape
    return x.reshape(n_subjects * windows, channels, samples), windows


def channel_stats(loader):
    """Per-channel mean/std over ALL train windows of a fold (no leakage)."""
    x = loader.dataset.X  # [S, W, C, T]
    mean = x.mean(dim=(0, 1, 3))
    std = x.std(dim=(0, 1, 3), unbiased=False)
    return mean, std


def normalize_channels(x: torch.Tensor, mean: torch.Tensor,
                       std: torch.Tensor) -> torch.Tensor:
    """Apply per-channel z-score using fold-level train statistics."""
    return (x - mean[None, :, None]) / (std[None, :, None] + 1e-8)


def expand_labels(y: torch.Tensor, windows: int) -> torch.Tensor:
    """Repeat each subject label once per window."""
    return y.view(-1, 1).repeat(1, windows).reshape(-1)


def build_model(
    name: str, n_channels: int, n_classes: int, n_samples: int, dropout: float
) -> torch.nn.Module:
    cls = MODEL_CLASSES[name]
    if name == "eegnet":
        return cls(n_channels=n_channels, n_classes=n_classes, dropout=dropout)
    return cls(
        n_channels=n_channels,
        n_classes=n_classes,
        n_samples=n_samples,
        dropout=dropout,
    )


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def confusion_matrix(true, pred) -> list[list[int]]:
    t, p = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    return [
        [int(((t == 0) & (p == 0)).sum()), int(((t == 0) & (p == 1)).sum())],
        [int(((t == 1) & (p == 0)).sum()), int(((t == 1) & (p == 1)).sum())],
    ]


def train_fold(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
    logger: ClassificationLogger,
    mean: torch.Tensor,
    std: torch.Tensor,
    label_smoothing: float,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    cls_counts = torch.bincount(torch.cat([y for _, _, y in train_loader]).long())
    cls_weights = (1.0 / cls_counts.float()).to(device)
    cls_weights = cls_weights / cls_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_bacc": [],
        "val_f1": [],
        "val_sens": [],
        "val_spec": [],
    }

    best_val_bacc, best_state, patience_left = -1.0, None, 0
    logger.log_header()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for _, x, y in train_loader:
            flat, windows = flatten_batch(x)
            flat = normalize_channels(flat, mean, std)
            yf = expand_labels(y, windows).to(device)
            optimizer.zero_grad()
            logits = forward_logits(model, flat.to(device))
            loss = criterion(logits, yf)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yf)
            tr_correct += (logits.argmax(1) == yf).sum().item()
            tr_total += len(yf)

        model.eval()
        val_loss, val_true, val_pred = 0.0, [], []
        val_subj_true, val_subj_pred = [], []
        with torch.no_grad():
            for _, x, y in val_loader:
                flat, windows = flatten_batch(x)
                flat = normalize_channels(flat, mean, std)
                yf = expand_labels(y, windows)
                logits = forward_logits(model, flat.to(device))
                val_loss += criterion(logits, yf.to(device)).item() * len(yf)
                val_true.extend(yf.tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())
                logits_subj = logits.view(x.shape[0], windows, -1)
                votes = logits_subj.argmax(dim=2).mode(dim=1).values
                val_subj_true.extend(y.cpu().tolist())
                val_subj_pred.extend(votes.cpu().tolist())

        tr_acc = tr_correct / max(tr_total, 1)
        tr_loss /= max(tr_total, 1)
        val_loss /= max(len(val_true), 1)
        vl_m = logger.metrics(val_true, val_pred)
        vs_m = logger.metrics(val_subj_true, val_subj_pred)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_m["acc"])
        history["val_bacc"].append(vl_m["bacc"])
        history["val_f1"].append(vl_m["f1"])
        history["val_sens"].append(vl_m["sens"])
        history["val_spec"].append(vl_m["spec"])

        if vs_m["bacc"] > best_val_bacc:
            best_val_bacc = vs_m["bacc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = 0
        else:
            patience_left += 1

        logger.log_epoch(
            epoch, tr_loss, val_loss, {"acc": tr_acc}, vl_m, patience_left
        )

        if patience_left >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def run_fold_test(model: torch.nn.Module, test_loader, device: str,
                  mean: torch.Tensor, std: torch.Tensor) -> dict:
    true_subj, pred_subj, prob_subj = [], [], []
    true_win, pred_win = [], []

    model.eval()
    with torch.no_grad():
        for _, x, y in test_loader:
            flat, windows = flatten_batch(x)
            flat = normalize_channels(flat, mean, std)
            logits = forward_logits(model, flat.to(device)).cpu()
            logits_subj = logits.view(x.shape[0], windows, -1)
            yf = expand_labels(y, windows)
            true_win.extend(yf.tolist())
            pred_win.extend(logits.argmax(1).tolist())

            votes = logits_subj.argmax(dim=2).mode(dim=1).values
            prob = logits_subj.softmax(dim=2)[:, :, 1].mean(dim=1)
            true_subj.extend(y.numpy().tolist())
            pred_subj.extend(votes.numpy().tolist())
            prob_subj.extend(prob.numpy().tolist())

    true, pred = np.asarray(true_subj), np.asarray(pred_subj)
    auc = float(roc_auc_score(true, prob_subj)) if len(set(true.tolist())) > 1 else None

    return {
        "test_true": true.tolist(),
        "test_pred": pred.tolist(),
        "test_cm_window": confusion_matrix(true_win, pred_win),
        "test_cm_subject": confusion_matrix(true, pred),
        "test_roc": {"y_true": true.tolist(), "y_prob": prob_subj},
        "test_auc": auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unimodal MODMA EEG classification")
    parser.add_argument("--modal", type=str, default="eeg",
                        choices=["eeg", "aud"])
    parser.add_argument("--channels", type=str, default="10-20",
                        choices=["all", "10-20", "f64"])
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--model", type=str, default="cnn_lstm",
                        choices=sorted(MODEL_CLASSES))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2509)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="base")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ds = MODMADataset(channels=args.channels, overlap=args.overlap)
    n_channels = len(ds.channel_names)
    n_samples = ds.samples[0]["eeg"].shape[-1]

    folds = create_dataloaders(
        ds,
        k_folder=args.k,
        inner_split=args.inner_splits,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    out_dir = (
        Path(args.output_root)
        / f"unimodals_sgkf_{args.modal}_sseed{args.split_seed}_tag{args.tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_per_subject = int(ds.samples[0]["eeg"].shape[0])
    model_hdr = build_model(
        args.model, n_channels, n_classes=2, n_samples=n_samples,
        dropout=args.dropout,
    )
    total, trainable, frozen = count_parameters(model_hdr)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 78)
    print(f" MODAL / MODEL        : {args.modal} / {args.model}")
    print(f" DEVICE               : {device}")
    print(f" GPU                  : {gpu}")
    print(f" DATASET              : channels={args.channels} ({n_channels}) "
          f"samples={n_samples} windows/subj={windows_per_subject} "
          f"overlap={args.overlap} subjects={len(ds.samples)}")
    print(f" INPUT SHAPES         : window [{n_channels}, {n_samples}] | "
          f"batch [{args.batch_size}, {windows_per_subject}, {n_channels}, {n_samples}] | "
          f"flatten [{args.batch_size * windows_per_subject}, {n_channels}, {n_samples}]")
    print(f" TRAINING CONFIG      : k={args.k} inner={args.inner_splits} "
          f"split_seed={args.split_seed} seed={args.seed} epochs={args.epochs} "
          f"batch={args.batch_size} lr={args.lr} wd={args.weight_decay} "
          f"dropout={args.dropout} label_smoothing={args.label_smoothing} "
          f"patience={args.patience}")
    print(f" MODEL PARAMS         : total={total:,} trainable={trainable:,} frozen={frozen:,}")
    print("=" * 78)

    logger = ClassificationLogger()
    results_folds = []

    def save_results() -> None:
        results = {
            "modal": args.modal,
            "tag": args.tag,
            "split_seed": args.split_seed,
            "seed": args.seed,
            "channels": args.channels,
            "model": args.model,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "folds": results_folds,
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_idx} ===")
        mean, std = channel_stats(train_loader)
        model = build_model(
            args.model, n_channels, n_classes=2, n_samples=n_samples,
            dropout=args.dropout,
        ).to(device)

        history = train_fold(
            model, train_loader, val_loader, args.epochs, args.lr,
            args.weight_decay, args.patience, device, logger,
            mean, std, args.label_smoothing,
        )

        fold_res = run_fold_test(model, test_loader, device, mean, std)
        fold_res["fold"] = fold_idx
        fold_res["history"] = history
        fold_res["test_metrics"] = logger.log_fold_test(
            fold_res["test_true"], fold_res["test_pred"]
        )
        results_folds.append(fold_res)

        if args.save_model:
            torch.save(
                model.state_dict(), out_dir / f"fold_{fold_idx}.pt"
            )
        save_results()

    logger.log_summary(n_folds=args.k, split_type="gkf")
    print(f"\nFinal results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
