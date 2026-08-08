"""Unimodal EEG classification on MODMA with nested group K-fold CV.

Each model is trained on 2-second EEG time windows extracted per subject.
Validation tracks window-level metrics every epoch (early stopping on val
accuracy). The final evaluation is done per subject via majority vote over
its windows.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.features.eeg_features import subject_features
from src.models.deepconvnet import DeepConvNet
from src.models.eeg_backbone import EEGBackbone
from src.preprocessing.modma_eeg import MODMADataset, create_dataloaders
from src.utils.get_seed import set_seed
from src.utils.training_logger import ClassificationLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/results/unimodals/eeg"
NUM_WORKERS = 4 if platform.system() != "Windows" else 0

MODEL_CLASSES = {
    "deepconvnet": DeepConvNet,
    "eeg_backbone": EEGBackbone,
}


def forward_logits(model: torch.nn.Module, x: torch.Tensor,
                   x_eng: torch.Tensor | None = None) -> torch.Tensor:
    """Forward pass returning only the class logits tensor."""
    out = model(x, x_eng=x_eng) if getattr(model, "full_subject_input", False) else model(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def flatten_batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Flatten subject windows into the batch dimension.

    Input is time windows ``[S, W, C, T]``; per-window trailing dims preserved.
    """
    n_subjects, windows = x.shape[0], x.shape[1]
    return x.reshape(n_subjects * windows, *x.shape[2:]), windows


def channel_stats(loader):
    """Per-channel mean/std over ALL train windows of a fold (no leakage)."""
    x = loader.dataset.X  # [S, W, C, T]
    reduce_dims = tuple(d for d in range(x.dim()) if d != 2)
    mean = x.mean(dim=reduce_dims)
    std = x.std(dim=reduce_dims, unbiased=False)
    return mean, std


def normalize_channels(x: torch.Tensor, mean: torch.Tensor,
                       std: torch.Tensor) -> torch.Tensor:
    """Apply per-channel z-score using fold-level train statistics."""
    shape = [1, -1] + [1] * (x.dim() - 2)
    return (x - mean.view(shape)) / (std.view(shape) + 1e-8)


def preprocess(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Per-channel z-score with fold-level train stats, then flatten windows."""
    flat, windows = flatten_batch(x)
    return normalize_channels(flat, mean, std), windows


def preprocess_subject(x: torch.Tensor, mean: torch.Tensor,
                       std: torch.Tensor) -> torch.Tensor:
    """Per-channel z-score keeping the ``[S, W, C, T]`` subject layout."""
    shape = [1, 1, -1] + [1] * (x.dim() - 3)
    return (x - mean.view(shape)) / (std.view(shape) + 1e-8)


def build_eng_feats(ds) -> dict[str, np.ndarray]:
    """per-subject engineered features: {participant_id: row}."""
    _, _, X = subject_features(ds)
    subjects = [s["participant_id"] for s in ds.samples]
    return {pid: X[i] for i, pid in enumerate(subjects)}


def eng_dim(eng_by_name: dict[str, np.ndarray]) -> int:
    """Dimensionality of the engineered feature rows."""
    return int(next(iter(eng_by_name.values())).shape[0])


def fit_eng_scaler(eng_feats: dict[str, np.ndarray],
                   train_names, val_names) -> dict[str, np.ndarray]:
    """Fit StandardScaler on train+val subjects only -> scaled {id: row}.

    Leak-free: the scaler never sees test subjects.
    """
    rows = np.stack([eng_feats[n] for n in (list(train_names) + list(val_names))])
    scaler = StandardScaler().fit(rows)
    return {n: scaler.transform(eng_feats[n].reshape(1, -1))[0] for n in eng_feats}


def batch_x_eng(names, eng_by_name: dict[str, np.ndarray], device: str) -> torch.Tensor:
    """Scaled per-subject engineered rows for a batch: ``[B, engineered_dim]`` on device."""
    rows = np.stack([eng_by_name[n] for n in names])
    return torch.as_tensor(rows, dtype=torch.float32).to(device)


def make_criterion(loss: str, device: str, label_smoothing: float, *,
                   train_loader=None, bce_pos_weight: bool = True):
    """Build the (possibly class-weighted) criterion for the given mode."""
    if loss == "bce":
        pos_weight = None
        if bce_pos_weight and train_loader is not None:
            cls_counts = torch.bincount(
                torch.cat([y for _, _, y in train_loader]).long()
            )
            if cls_counts.numel() > 1 and float(cls_counts[1]) > 0:
                pos_weight = (cls_counts[0].float() / cls_counts[1].float()).to(device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cls_counts = torch.bincount(
        torch.cat([y for _, _, y in train_loader]).long()
    )
    cls_weights = (1.0 / cls_counts.float()).to(device)
    cls_weights = cls_weights / cls_weights.mean()
    return torch.nn.CrossEntropyLoss(
        weight=cls_weights, label_smoothing=label_smoothing
    )


def step_loss(criterion, logits: torch.Tensor, yf: torch.Tensor,
              loss: str) -> torch.Tensor:
    """Compute loss for the given mode on already-expanded labels."""
    if loss == "bce":
        y_smooth = yf.float() * 0.95 + 0.025
        return criterion(logits[:, 1], y_smooth)
    return criterion(logits, yf)


def expand_labels(y: torch.Tensor, windows: int) -> torch.Tensor:
    """Repeat each subject label once per window."""
    return y.view(-1, 1).repeat(1, windows).reshape(-1)


def subject_prob(logits: torch.Tensor, n_subjects: int, windows: int) -> torch.Tensor:
    """Per-subject class-1 probability.

    Windows are pooled in *log space* first, then softmax once
    (``softmax(mean_w logits)``). Averaging in probability space
    (``mean_w softmax``) compresses every subject's score toward 0.5 when the
    per-window CNN output varies, erasing discriminative signal.
    """
    logits_subj = logits.view(n_subjects, windows, -1).mean(dim=1)
    return logits_subj.softmax(dim=1)[:, 1]


def forward_batch(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                  mean: torch.Tensor, std: torch.Tensor, device: str,
                  names=None, eng_by_name: dict[str, np.ndarray] | None = None):
    """Model-agnostic forward.

    Window-level models receive flattened ``[S*W, C, T]`` (labels expanded per
    window). Subject-level models (``subject_level``) receive the full
    ``[S, W, C, T]`` and return one logit row per subject directly.
    ``full_subject_input`` models get the full normalized subject tensor and
    the pre-scaled engineered features ``x_eng [S, engineered_dim]`` (per-fold
    StandardScaler), returning per-window logits.

    Returns ``(logits, labels)`` both on ``device``.
    """
    if getattr(model, "subject_level", False):
        xn = preprocess_subject(x, mean, std)
        return model(xn.to(device)), y.to(device)
    if getattr(model, "full_subject_input", False):
        xn = preprocess_subject(x, mean, std)
        x_eng = (
            batch_x_eng(list(names), eng_by_name, device)
            if eng_by_name is not None else None
        )
        logits = model(xn.to(device), x_eng=x_eng)
        labels = expand_labels(y, x.shape[1]).to(device)
        return logits, labels
    flat, windows = preprocess(x, mean, std)
    labels = expand_labels(y, windows).to(device)
    return model(flat.to(device)), labels


def build_model(
    name: str,
    n_channels: int,
    n_classes: int,
    n_samples: int,
    dropout: float,
    hidden: int = 32,
    n_filters: int = 16,
    engineered_dim: int = 17,
) -> torch.nn.Module:
    cls = MODEL_CLASSES[name]
    return cls(
        n_channels=n_channels,
        n_classes=n_classes,
        n_samples=n_samples,
        dropout=dropout,
        **({"hidden": hidden, "n_filters": n_filters, "engineered_dim": engineered_dim}
           if name == "eeg_backbone" else {}),
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
    loss: str,
    early_stop_on: str = "window-bacc",
    bce_pos_weight: bool = True,
    eng_by_name: dict[str, np.ndarray] | None = None,
    consistency_coef: float = 0.0,
) -> tuple[dict, int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15
    )
    criterion = make_criterion(
        loss, device, label_smoothing, train_loader=train_loader,
        bce_pos_weight=bce_pos_weight,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_bacc": [],
        "val_f1": [],
        "val_sens": [],
        "val_spec": [],
        "val_subj_bacc": [],
        "val_subj_f1": [],
        "val_subj_sens": [],
        "val_subj_spec": [],
    }

    best_val_bacc, best_state, best_epoch, patience_left = -1.0, None, 0, 0
    logger.log_header()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for name, x, y in train_loader:
            logits, yf = forward_batch(model, x, y, mean, std, device,
                                       names=name, eng_by_name=eng_by_name)
            optimizer.zero_grad()
            tloss = step_loss(criterion, logits, yf, loss)
            if consistency_coef > 0 and getattr(model, "full_subject_input", False):
                n_subjects, windows = x.shape[0], x.shape[1]
                subj_logits = logits.view(n_subjects, windows, -1).mean(dim=1)
                y_subj = (y.float() * 0.95 + 0.025).to(device)
                tloss = tloss + consistency_coef * criterion(subj_logits[:, 1], y_subj)
            tloss.backward()
            optimizer.step()
            tr_loss += tloss.item() * len(yf)
            tr_correct += (logits.argmax(1) == yf).sum().item()
            tr_total += len(yf)

        model.eval()
        val_loss, val_true, val_pred = 0.0, [], []
        val_subj_true, val_subj_pred = [], []
        with torch.no_grad():
            for name, x, y in val_loader:
                logits, yf = forward_batch(model, x, y, mean, std, device,
                                           names=name, eng_by_name=eng_by_name)
                val_loss += step_loss(criterion, logits, yf, loss).item() * len(yf)
                if getattr(model, "subject_level", False):
                    val_true.extend(y.tolist())
                    val_pred.extend(logits.argmax(1).cpu().tolist())
                    val_subj_true.extend(y.tolist())
                    prob = logits.softmax(dim=1)[:, 1].cpu()
                    val_subj_pred.extend((prob >= 0.5).long().tolist())
                else:
                    yf_cpu = yf
                    windows = x.shape[1]
                    val_true.extend(yf_cpu.tolist())
                    val_pred.extend(logits.argmax(1).cpu().tolist())
                    subj_prob = subject_prob(logits, x.shape[0], windows)
                    val_subj_true.extend(y.cpu().tolist())
                    val_subj_pred.extend((subj_prob >= 0.5).long().cpu().tolist())

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
        history["val_subj_bacc"].append(vs_m["bacc"])
        history["val_subj_f1"].append(vs_m["f1"])
        history["val_subj_sens"].append(vs_m["sens"])
        history["val_subj_spec"].append(vs_m["spec"])

        es_metric = vs_m["bacc"] if early_stop_on == "subject-bacc" else vl_m["bacc"]
        scheduler.step(es_metric)

        if es_metric > best_val_bacc:
            best_val_bacc = es_metric
            best_epoch = epoch
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
    if best_epoch == 0:
        best_epoch = len(history["train_loss"])
    return history, best_epoch


class _CombinedDataset(Dataset):
    def __init__(self, x, y, names):
        self.X, self.y, self.names = x, y, list(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx], self.X[idx], self.y[idx]


def refit_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    label_smoothing: float,
    batch_size: int,
    loss: str,
    bce_pos_weight: bool = True,
    eng_by_name: dict[str, np.ndarray] | None = None,
    consistency_coef: float = 0.0,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """Retrain ``model`` on train+val for exactly ``epochs`` epochs (no early stop)."""
    x = torch.cat([train_loader.dataset.X, val_loader.dataset.X])
    y = torch.cat([train_loader.dataset.y, val_loader.dataset.y])
    names = list(train_loader.dataset.names) + list(val_loader.dataset.names)
    loader = DataLoader(
        _CombinedDataset(x, y, names),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    reduce_dims = tuple(d for d in range(x.dim()) if d != 2)
    mean = x.mean(dim=reduce_dims)
    std = x.std(dim=reduce_dims, unbiased=False)

    criterion = make_criterion(
        loss, device, label_smoothing, train_loader=loader,
        bce_pos_weight=bce_pos_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    model.train()
    for _ in range(epochs):
        for name, xb, yb in loader:
            logits, yf = forward_batch(model, xb, yb, mean, std, device,
                                       names=name, eng_by_name=eng_by_name)
            optimizer.zero_grad()
            tloss = step_loss(criterion, logits, yf, loss)
            if consistency_coef > 0 and getattr(model, "full_subject_input", False):
                n_subjects, windows = xb.shape[0], xb.shape[1]
                subj_logits = logits.view(n_subjects, windows, -1).mean(dim=1)
                y_subj = (yb.float() * 0.95 + 0.025).to(device)
                tloss = tloss + consistency_coef * criterion(subj_logits[:, 1], y_subj)
            tloss.backward()
            optimizer.step()
    return model, mean, std


def run_fold_test(model: torch.nn.Module, test_loader, device: str,
                  mean: torch.Tensor, std: torch.Tensor,
                  eng_by_name: dict[str, np.ndarray] | None = None) -> dict:
    true_subj, pred_subj, prob_subj = [], [], []
    true_win, pred_win = [], []
    emb_subj, emb_names = [], []

    model.eval()
    with torch.no_grad():
        for name, x, y in test_loader:
            if getattr(model, "subject_level", False):
                logits, _ = forward_batch(model, x, y, mean, std, device,
                                          names=name, eng_by_name=eng_by_name)
                prob = logits.softmax(dim=1)[:, 1].cpu()
                true_subj.extend(y.numpy().tolist())
                pred_subj.extend((prob >= 0.5).long().numpy().tolist())
                prob_subj.extend(prob.numpy().tolist())
            elif getattr(model, "full_subject_input", False):
                logits, _ = forward_batch(model, x, y, mean, std, device,
                                          names=name, eng_by_name=eng_by_name)
                windows = x.shape[1]
                logits = logits.cpu()
                yf = expand_labels(y, windows)
                true_win.extend(yf.tolist())
                pred_win.extend(logits.argmax(1).tolist())
                prob = subject_prob(logits, x.shape[0], windows)
                true_subj.extend(y.numpy().tolist())
                pred_subj.extend((prob >= 0.5).long().numpy().tolist())
                prob_subj.extend(prob.numpy().tolist())
                if eng_by_name is not None:
                    x_eng_b = batch_x_eng(list(name), eng_by_name, device)
                    xn = preprocess_subject(x, mean, std)
                    emb = model.forward_features(xn.to(device), x_eng=x_eng_b)
                    emb_subj.append(emb.detach().cpu().numpy())
                    emb_names.extend(list(name))
            else:
                flat, windows = preprocess(x, mean, std)
                logits = forward_logits(model, flat.to(device)).cpu()
                yf = expand_labels(y, windows)
                true_win.extend(yf.tolist())
                pred_win.extend(logits.argmax(1).tolist())

                prob = subject_prob(logits, x.shape[0], windows)
                true_subj.extend(y.numpy().tolist())
                pred_subj.extend((prob >= 0.5).long().numpy().tolist())
                prob_subj.extend(prob.numpy().tolist())

    true, pred = np.asarray(true_subj), np.asarray(pred_subj)
    auc = float(roc_auc_score(true, prob_subj)) if len(set(true.tolist())) > 1 else None

    z_eeg = np.concatenate(emb_subj, axis=0) if emb_subj else None

    return {
        "test_true": true.tolist(),
        "test_pred": pred.tolist(),
        "test_cm_window": confusion_matrix(true_win, pred_win),
        "test_cm_subject": confusion_matrix(true, pred),
        "test_roc": {"y_true": true.tolist(), "y_prob": prob_subj},
        "test_auc": auc,
        "test_z_eeg": z_eeg.tolist() if z_eeg is not None else None,
        "test_emb_subjects": emb_names if emb_names else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unimodal MODMA EEG classification")
    parser.add_argument("--modal", type=str, default="eeg",
                        choices=["eeg"])
    parser.add_argument("--representation", type=str, default="time",
                        choices=["time"],
                        help="Input representation (accepted/validated; time only)")
    parser.add_argument("--norm", type=str, default="fold",
                        choices=["fold"],
                        help="Normalization scheme (accepted/validated; fold only)")
    parser.add_argument("--channels", type=str, default="29",
                        choices=["all", "10-20", "f64", "29"])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--reference", type=str, default="average",
                        choices=["average", "cz"],
                        help="EEG reference (cz requires Cz in the channels)")
    parser.add_argument("--lowcut", type=float, default=0.4)
    parser.add_argument("--highcut", type=float, default=45.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--model", type=str, default="deepconvnet",
                        choices=sorted(MODEL_CLASSES))
    parser.add_argument("--hidden", type=int, default=32,
                        help="Hidden dim for the eeg_backbone fusion head")
    parser.add_argument("--n-filters", type=int, default=16,
                        help="Spectral filter count for the eeg_backbone CNN")
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["ce", "bce"],
                        help="Loss: bce (BCE + label smoothing) or ce (weighted CE)")
    parser.add_argument("--early-stop-on", type=str, default="window-bacc",
                        choices=["subject-bacc", "window-bacc"],
                        help="Validation signal for scheduler + best-epoch "
                             "selection: subject-bacc (subject majority vote) "
                             "or window-bacc (window-level, more samples)")
    parser.add_argument("--no-bce-pos-weight", action="store_true",
                        help="Disable class-balanced pos_weight in BCE")
    parser.add_argument("--consistency-coef", type=float, default=0.0,
                        help="Weight of the subject-consistency auxiliary BCE "
                             "on per-subject pooled logits (0 = disabled)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, nargs="+", default=[2509])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="base")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--refit", action="store_true",
                        help="Retrain final model on train+val before testing")
    parser.add_argument("--refit-epochs", type=int, default=60,
                        help="Fixed number of epochs to retrain the final model "
                             "on train+val. Decoupled from the (noisy) best "
                             "val epoch so the tested model is actually trained")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ds = MODMADataset(
        channels=args.channels,
        overlap=args.overlap,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        reference=args.reference,
    )
    n_channels = len(ds.channel_names)
    # window shape: [W, C, T]
    n_samples = ds.samples[0]["eeg"].shape[-1]

    eng_feats = build_eng_feats(ds)  # deterministic per-subject features
    engineered_dim = eng_dim(eng_feats)

    windows = int(ds.samples[0]["eeg"].shape[0])
    model_hdr = build_model(
        args.model, n_channels, n_classes=2, n_samples=n_samples,
        dropout=args.dropout, hidden=args.hidden, n_filters=args.n_filters,
        engineered_dim=engineered_dim,
    )
    total, trainable, frozen = count_parameters(model_hdr)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    for split_seed in args.split_seed:
        print(f"\n{'=' * 78}\n  SPLIT SEED            : {split_seed}\n{'=' * 78}")

        folds = create_dataloaders(
            ds,
            k_folder=args.k,
            inner_split=args.inner_splits,
            split_seed=split_seed,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=(device == "cuda"),
        )

        out_dir = (
            Path(args.output_root)
            / f"unimodals_sgkf_{args.modal}_sseed{split_seed}_tag{args.tag}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "identity": {
                "modal": args.modal,
                "model": args.model,
                "tag": args.tag,
                "output_dir": str(out_dir),
            },
            "data": {
                "root": str(ds.root),
                "channel_names": ds.channel_names,
                "n_subjects": len(ds.samples),
                "n_channels": n_channels,
                "n_samples": n_samples,
                "windows_per_subject": windows,
                "window_sec": ds.window_sec,
                "overlap": args.overlap,
                "lowcut": ds.lowcut,
                "highcut": ds.highcut,
                "notch": ds.notch,
                "target_fs": ds.target_fs,
                "reference": ds.reference,
            },
            "cv": {
                "k": args.k,
                "inner_splits": args.inner_splits,
                "split_seed": split_seed,
                "shuffle": True,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "label_smoothing": args.label_smoothing,
                "patience": args.patience,
                "class_weighted_loss": args.loss == "ce",
                "early_stop_on": args.early_stop_on,
                "bce_pos_weight": not args.no_bce_pos_weight,
                "consistency_coef": args.consistency_coef,
                "refit": args.refit,
                "refit_epochs": args.refit_epochs,
                "loss": args.loss,
            },
            "model": {
                "constructor": {
                    "name": args.model,
                    "n_channels": n_channels,
                    "n_classes": 2,
                    "n_samples": n_samples,
                    "dropout": args.dropout,
                },
                "total_params": total,
                "trainable_params": trainable,
                "frozen_params": frozen,
            },
            "environment": {
                "device": device,
                "gpu": gpu,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "mne": mne.__version__,
            },
            "cli": vars(args),
        }

        print("=" * 78)
        print(f" MODAL / MODEL        : {args.modal} / {args.model}")
        print(f" DEVICE               : {device}")
        print(f" GPU                  : {gpu}")
        print(f" DATASET              : channels={args.channels} ({n_channels}) "
              f"samples={n_samples} windows/subj={windows} "
              f"overlap={args.overlap} subjects={len(ds.samples)}")
        win_shape = tuple(ds.samples[0]["eeg"].shape[1:])
        flat_shape = (args.batch_size * windows, *win_shape)
        print(f" INPUT SHAPES         : representation=time | "
              f"window {win_shape} | batch {(args.batch_size, windows, *win_shape)} | "
              f"flatten {flat_shape}")
        print(f" TRAINING CONFIG      : k={args.k} inner={args.inner_splits} "
              f"split_seed={split_seed} seed={args.seed} epochs={args.epochs} "
              f"batch={args.batch_size} lr={args.lr} wd={args.weight_decay} "
              f"dropout={args.dropout} label_smoothing={args.label_smoothing} "
              f"patience={args.patience} es={args.early_stop_on} "
              f"bce_pw={not args.no_bce_pos_weight}")
        print(f" MODEL PARAMS         : total={total:,} trainable={trainable:,} frozen={frozen:,}")
        print("=" * 78)

        logger = ClassificationLogger()
        results_folds = []

        def save_results(
            split_seed=split_seed,
            config=config,
            results_folds=results_folds,
            out_dir=out_dir,
        ) -> None:
            results = {
                "modal": args.modal,
                "tag": args.tag,
                "split_seed": split_seed,
                "seed": args.seed,
                "channels": args.channels,
                "model": args.model,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "config": config,
                "folds": results_folds,
            }
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
            print(f"\n=== Fold {fold_idx} ===")
            def cls_counts(dl):
                c = Counter(dl.dataset.y.tolist())
                return f"{c.get(0, 0)}/{c.get(1, 0)}"
            print(f"  subjects train={len(train_loader.dataset)} "
                  f"HC/MDD={cls_counts(train_loader)} | val={len(val_loader.dataset)} "
                  f"HC/MDD={cls_counts(val_loader)} | test={len(test_loader.dataset)} "
                  f"HC/MDD={cls_counts(test_loader)}")
            mean, std = channel_stats(train_loader)
            eng_by_name = fit_eng_scaler(
                eng_feats, train_loader.dataset.names, val_loader.dataset.names
            )
            model = build_model(
                args.model, n_channels, n_classes=2, n_samples=n_samples,
                dropout=args.dropout, hidden=args.hidden, n_filters=args.n_filters,
                engineered_dim=engineered_dim,
            ).to(device)

            history, best_epoch = train_fold(
                model, train_loader, val_loader, args.epochs, args.lr,
                args.weight_decay, args.patience, device, logger,
                mean, std, args.label_smoothing, args.loss,
                early_stop_on=args.early_stop_on,
                bce_pos_weight=not args.no_bce_pos_weight,
                eng_by_name=eng_by_name,
                consistency_coef=args.consistency_coef,
            )

            if args.refit:
                model, mean, std = refit_model(
                    build_model(
                        args.model, n_channels, n_classes=2, n_samples=n_samples,
                        dropout=args.dropout, hidden=args.hidden, n_filters=args.n_filters,
                        engineered_dim=engineered_dim,
                    ).to(device),
                    train_loader, val_loader, args.refit_epochs, args.lr,
                    args.weight_decay, device, args.label_smoothing,
                    args.batch_size, args.loss,
                    bce_pos_weight=not args.no_bce_pos_weight,
                    eng_by_name=eng_by_name,
                    consistency_coef=args.consistency_coef,
                )
            train_subjects = (
                (list(train_loader.dataset.names)
                 + list(val_loader.dataset.names))
                if args.refit else list(train_loader.dataset.names)
            )

            fold_res = run_fold_test(model, test_loader, device, mean, std,
                                     eng_by_name=eng_by_name)
            fold_res["fold"] = fold_idx
            fold_res["history"] = history
            fold_res["best_epoch"] = best_epoch
            fold_res["train_subjects"] = train_subjects
            fold_res["val_subjects"] = list(val_loader.dataset.names)
            fold_res["test_subjects"] = list(test_loader.dataset.names)
            fold_res["test_metrics"] = logger.log_fold_test(
                fold_res["test_true"], fold_res["test_pred"],
                test_auc=fold_res["test_auc"],
            )
            results_folds.append(fold_res)

            if args.save_model:
                torch.save(
                    model.state_dict(), out_dir / f"fold_{fold_idx}.pt"
                )
            save_results()

            del model
            torch.cuda.empty_cache()

        logger.log_summary(n_folds=args.k, split_type="gkf")
        print(f"\nFinal results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
