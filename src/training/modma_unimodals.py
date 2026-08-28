from __future__ import annotations

import argparse
import json
import platform
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


def forward_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    x_eng: torch.Tensor | None = None,
) -> torch.Tensor:
    out = (
        model(x, x_eng=x_eng)
        if getattr(model, "full_subject_input", False)
        else model(x)
    )

    if isinstance(out, tuple):
        out = out[0]

    return out


def flatten_batch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    n_subjects = x.shape[0]
    windows = x.shape[1]

    flat = x.reshape(
        n_subjects * windows,
        *x.shape[2:],
    )

    return flat, windows


def channel_stats(loader):
    total = None
    total_squared = None
    total_values = 0

    for _, x, _ in loader:
        channel_dimension = 1 if x.dim() == 3 else 2
        reduce_dims = tuple(
            dimension
            for dimension in range(x.dim())
            if dimension != channel_dimension
        )
        batch_total = x.sum(dim=reduce_dims)
        batch_squared = x.square().sum(dim=reduce_dims)
        total = batch_total if total is None else total + batch_total
        total_squared = (
            batch_squared
            if total_squared is None
            else total_squared + batch_squared
        )
        batch_size = x.numel() // x.shape[channel_dimension]
        total_values += batch_size

    if total is None or total_squared is None or total_values == 0:
        raise ValueError("Cannot calculate channel statistics from an empty loader")

    mean = total / total_values
    variance = total_squared / total_values - mean.square()
    return mean, variance.clamp_min(0).sqrt()


def normalize_channels(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    shape = [1, -1] + [1] * (x.dim() - 2)

    return (
        x - mean.view(shape)
    ) / (
        std.view(shape) + 1e-8
    )


def preprocess(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    flat, windows = flatten_batch(x)

    normalized = normalize_channels(
        flat,
        mean,
        std,
    )

    return normalized, windows


def preprocess_subject(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    shape = [1, 1, -1] + [1] * (x.dim() - 3)

    return (
        x - mean.view(shape)
    ) / (
        std.view(shape) + 1e-8
    )


def build_eng_feats(
    ds,
) -> dict[str, np.ndarray]:
    _, _, features = subject_features(ds)

    subjects = [
        sample["participant_id"]
        for sample in ds.samples
    ]

    return {
        participant_id: features[index]
        for index, participant_id in enumerate(subjects)
    }


def eng_dim(
    eng_by_name: dict[str, np.ndarray],
) -> int:
    first_value = next(
        iter(eng_by_name.values())
    )

    return int(first_value.shape[0])


def fit_eng_scaler(
    eng_feats: dict[str, np.ndarray],
    train_names,
    val_names,
) -> dict[str, np.ndarray]:
    names = list(train_names) + list(val_names)

    rows = np.stack(
        [
            eng_feats[name]
            for name in names
        ]
    )

    scaler = StandardScaler().fit(rows)

    return {
        name: scaler.transform(
            eng_feats[name].reshape(1, -1)
        )[0]
        for name in eng_feats
    }


def batch_x_eng(
    names,
    eng_by_name: dict[str, np.ndarray],
    device: str,
) -> torch.Tensor:
    rows = np.stack(
        [
            eng_by_name[name]
            for name in names
        ]
    )

    return torch.as_tensor(
        rows,
        dtype=torch.float32,
        device=device,
    )


def make_criterion(
    loss: str,
    device: str,
    label_smoothing: float,
    *,
    train_loader=None,
    bce_pos_weight: bool = True,
):
    if loss == "bce":
        pos_weight = None

        if bce_pos_weight and train_loader is not None:
            labels = train_loader.dataset.y.long()

            counts = torch.bincount(labels)

            if (
                counts.numel() > 1
                and float(counts[1]) > 0
            ):
                pos_weight = (
                    counts[0].float()
                    / counts[1].float()
                ).to(device)

        return torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )

    labels = train_loader.dataset.y.long()

    counts = torch.bincount(labels).float()
    weights = 1.0 / counts
    weights = weights / weights.mean()

    return torch.nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=label_smoothing,
    )


def step_loss(
    criterion,
    logits: torch.Tensor,
    yf: torch.Tensor,
    loss: str,
) -> torch.Tensor:
    if loss == "bce":
        smoothed = yf.float() * 0.95 + 0.025

        return criterion(
            logits.squeeze(-1),
            smoothed,
        )

    return criterion(
        logits,
        yf.long(),
    )


def expand_labels(
    y: torch.Tensor,
    windows: int,
) -> torch.Tensor:
    return (
        y.view(-1, 1)
        .repeat(1, windows)
        .reshape(-1)
    )


def binary_predictions(
    logits: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.sigmoid(
            logits.squeeze(-1)
        ) >= 0.5
    ).long()


def subject_prob(
    logits: torch.Tensor,
    n_subjects: int,
    windows: int,
) -> torch.Tensor:
    pooled = logits.reshape(
        n_subjects,
        windows,
        -1,
    ).mean(dim=1)

    if pooled.shape[-1] == 1:
        return torch.sigmoid(
            pooled[:, 0]
        )

    return pooled.softmax(
        dim=1
    )[:, 1]


def attention_subject_prob(
    model: torch.nn.Module,
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    names,
    eng_by_name: dict[str, np.ndarray] | None,
) -> torch.Tensor:
    normalized = preprocess_subject(
        x,
        mean,
        std,
    )

    engineered = (
        batch_x_eng(
            list(names),
            eng_by_name,
            device,
        )
        if eng_by_name is not None
        else None
    )

    logits = model.subject_logits(
        normalized.to(device),
        x_eng=engineered,
    )

    if logits.shape[-1] == 1:
        return torch.sigmoid(
            logits.squeeze(-1)
        ).cpu()

    return logits.softmax(
        dim=1
    )[:, 1].cpu()


def forward_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    names=None,
    eng_by_name: dict[str, np.ndarray] | None = None,
):
    if x.dim() == 3:
        normalized = normalize_channels(x, mean, std)
        return model(normalized.to(device)), y.to(device)

    if getattr(model, "subject_level", False):
        normalized = preprocess_subject(
            x,
            mean,
            std,
        )

        return (
            model(
                normalized.to(device)
            ),
            y.to(device),
        )

    if getattr(model, "full_subject_input", False):
        normalized = preprocess_subject(
            x,
            mean,
            std,
        )

        engineered = (
            batch_x_eng(
                list(names),
                eng_by_name,
                device,
            )
            if eng_by_name is not None
            else None
        )

        logits = model(
            normalized.to(device),
            x_eng=engineered,
        )

        labels = expand_labels(
            y,
            x.shape[1],
        ).to(device)

        return logits, labels

    flat, windows = preprocess(
        x,
        mean,
        std,
    )

    labels = expand_labels(
        y,
        windows,
    ).to(device)

    logits = model(
        flat.to(device)
    )

    return logits, labels


def build_model(
    name: str,
    n_channels: int,
    n_classes: int,
    n_samples: int,
    dropout: float,
    hidden: int = 32,
    n_filters: int = 16,
    engineered_dim: int = 17,
    pool: str = "mean",
) -> torch.nn.Module:
    model_class = MODEL_CLASSES[name]

    if name == "eeg_backbone":
        return model_class(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=n_samples,
            dropout=dropout,
            hidden=hidden,
            n_filters=n_filters,
            engineered_dim=engineered_dim,
            pool=pool,
        )

    return model_class(
        n_channels=n_channels,
        n_classes=n_classes,
        n_samples=n_samples,
        dropout=dropout,
    )


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int, int]:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = total - trainable

    return total, trainable, frozen


def confusion_matrix(
    true,
    pred,
) -> list[list[int]]:
    true_array = np.asarray(
        true,
        dtype=int,
    )

    pred_array = np.asarray(
        pred,
        dtype=int,
    )

    return [
        [
            int(
                (
                    (true_array == 0)
                    & (pred_array == 0)
                ).sum()
            ),
            int(
                (
                    (true_array == 0)
                    & (pred_array == 1)
                ).sum()
            ),
        ],
        [
            int(
                (
                    (true_array == 1)
                    & (pred_array == 0)
                ).sum()
            ),
            int(
                (
                    (true_array == 1)
                    & (pred_array == 1)
                ).sum()
            ),
        ],
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
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )

    criterion = make_criterion(
        loss,
        device,
        label_smoothing,
        train_loader=train_loader,
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

    best_metric = -1.0
    best_state = None
    best_epoch = 0
    patience_count = 0

    logger.log_header()

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for names, x, y in train_loader:
            logits, yf = forward_batch(
                model,
                x,
                y,
                mean,
                std,
                device,
                names=names,
                eng_by_name=eng_by_name,
            )

            optimizer.zero_grad()

            current_loss = step_loss(
                criterion,
                logits,
                yf,
                loss,
            )

            current_loss.backward()
            optimizer.step()

            train_loss += (
                current_loss.item()
                * len(yf)
            )

            if loss == "bce":
                predictions = binary_predictions(
                    logits
                )
            else:
                predictions = logits.argmax(
                    dim=1
                )

            train_correct += (
                predictions == yf.long()
            ).sum().item()

            train_total += len(yf)

        model.eval()

        val_loss = 0.0
        val_true = []
        val_pred = []
        val_names = []
        val_logits = []
        val_labels = []
        subject_true = []
        subject_pred = []

        with torch.no_grad():
            for names, x, y in val_loader:
                logits, yf = forward_batch(
                    model,
                    x,
                    y,
                    mean,
                    std,
                    device,
                    names=names,
                    eng_by_name=eng_by_name,
                )

                current_loss = step_loss(
                    criterion,
                    logits,
                    yf,
                    loss,
                )

                val_loss += (
                    current_loss.item()
                    * len(yf)
                )

                if loss == "bce":
                    window_predictions = binary_predictions(
                        logits
                    )
                else:
                    window_predictions = logits.argmax(
                        dim=1
                    )

                val_true.extend(
                    yf.cpu().tolist()
                )

                val_pred.extend(
                    window_predictions.cpu().tolist()
                )
                val_names.extend(list(names))
                val_logits.append(logits.detach().cpu())
                val_labels.append(y.detach().cpu())

                if x.dim() == 3:
                    continue
                if getattr(model, "subject_level", False):
                    subject_probabilities = (
                        logits.softmax(dim=1)[:, 1]
                    )
                elif getattr(model, "sub_att", None) is not None:
                    subject_probabilities = attention_subject_prob(
                        model,
                        x,
                        mean,
                        std,
                        device,
                        names,
                        eng_by_name,
                    )
                else:
                    subject_probabilities = subject_prob(
                        logits,
                        x.shape[0],
                        x.shape[1],
                    )

                subject_predictions = (
                    subject_probabilities >= 0.5
                ).long()

                subject_true.extend(
                    y.cpu().tolist()
                )

                subject_pred.extend(
                    subject_predictions.cpu().tolist()
                )

        if val_logits:
            subject_logit_values = {}
            subject_labels = {}
            for name, logit, label in zip(
                val_names,
                torch.cat(val_logits).reshape(-1),
                torch.cat(val_labels).reshape(-1),
            ):
                subject_logit_values.setdefault(name, []).append(logit)
                subject_labels[name] = int(label.item())
            subject_true = [subject_labels[name] for name in subject_logit_values]
            subject_pred = [
                int(torch.sigmoid(torch.stack(subject_logit_values[name]).mean()) >= 0.5)
                for name in subject_logit_values
            ]

        train_loss /= max(train_total, 1)
        val_loss /= max(len(val_true), 1)

        train_accuracy = (
            train_correct
            / max(train_total, 1)
        )

        window_metrics = logger.metrics(
            val_true,
            val_pred,
        )

        subject_metrics = logger.metrics(
            subject_true,
            subject_pred,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(window_metrics["acc"])
        history["val_bacc"].append(window_metrics["bacc"])
        history["val_f1"].append(window_metrics["f1"])
        history["val_sens"].append(window_metrics["sens"])
        history["val_spec"].append(window_metrics["spec"])
        history["val_subj_bacc"].append(subject_metrics["bacc"])
        history["val_subj_f1"].append(subject_metrics["f1"])
        history["val_subj_sens"].append(subject_metrics["sens"])
        history["val_subj_spec"].append(subject_metrics["spec"])

        if early_stop_on == "subject-bacc":
            stopping_metric = subject_metrics["bacc"]
        else:
            stopping_metric = window_metrics["bacc"]

        scheduler.step(stopping_metric)

        if stopping_metric > best_metric:
            best_metric = stopping_metric
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        logger.log_epoch(
            epoch,
            train_loss,
            val_loss,
            {"acc": train_accuracy},
            window_metrics,
            patience_count,
        )

        if patience_count >= patience:
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    if best_epoch == 0:
        best_epoch = len(
            history["train_loss"]
        )

    return history, best_epoch


def train_fixed_epochs(
    model: torch.nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    label_smoothing: float,
    loss: str,
    bce_pos_weight: bool = True,
    eng_by_name: dict[str, np.ndarray] | None = None,
) -> None:
    criterion = make_criterion(
        loss,
        device,
        label_smoothing,
        train_loader=train_loader,
        bce_pos_weight=bce_pos_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for _ in range(max(epochs, 1)):
        model.train()
        for names, x, y in train_loader:
            logits, yf = forward_batch(
                model,
                x,
                y,
                mean,
                std,
                device,
                names=names,
                eng_by_name=eng_by_name,
            )
            optimizer.zero_grad()
            step_loss(criterion, logits, yf, loss).backward()
            optimizer.step()


class _CombinedDataset(Dataset):
    def __init__(
        self,
        x,
        y,
        names,
    ):
        self.X = x
        self.y = y
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(
        self,
        idx,
    ):
        return (
            self.names[idx],
            self.X[idx],
            self.y[idx],
        )


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
) -> tuple[
    torch.nn.Module,
    torch.Tensor,
    torch.Tensor,
]:
    x = torch.cat(
        [
            train_loader.dataset.X,
            val_loader.dataset.X,
        ]
    )

    y = torch.cat(
        [
            train_loader.dataset.y,
            val_loader.dataset.y,
        ]
    )

    names = (
        list(train_loader.dataset.names)
        + list(val_loader.dataset.names)
    )

    dataset = _CombinedDataset(
        x,
        y,
        names,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    mean, std = channel_stats(
        type(
            "StatsLoader",
            (),
            {
                "dataset": type(
                    "StatsDataset",
                    (),
                    {"X": x},
                )(),
            },
        )()
    )

    criterion = make_criterion(
        loss,
        device,
        label_smoothing,
        train_loader=loader,
        bce_pos_weight=bce_pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    model.train()

    for _ in range(epochs):
        for names_batch, xb, yb in loader:
            logits, yf = forward_batch(
                model,
                xb,
                yb,
                mean,
                std,
                device,
                names=names_batch,
                eng_by_name=eng_by_name,
            )

            optimizer.zero_grad()

            current_loss = step_loss(
                criterion,
                logits,
                yf,
                loss,
            )

            current_loss.backward()
            optimizer.step()

    return model, mean, std


def run_fold_test(
    model: torch.nn.Module,
    test_loader,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    eng_by_name: dict[str, np.ndarray] | None = None,
) -> dict:
    true_subjects = []
    predicted_subjects = []
    subject_probabilities = []
    true_windows = []
    predicted_windows = []
    embeddings = []
    embedding_names = []

    model.eval()

    with torch.no_grad():
        for names, x, y in test_loader:
            if getattr(model, "subject_level", False):
                logits, _ = forward_batch(
                    model,
                    x,
                    y,
                    mean,
                    std,
                    device,
                    names=names,
                    eng_by_name=eng_by_name,
                )

                probabilities = (
                    logits.softmax(dim=1)[:, 1]
                    .cpu()
                )

                predictions = (
                    probabilities >= 0.5
                ).long()

                true_subjects.extend(
                    y.numpy().tolist()
                )

                predicted_subjects.extend(
                    predictions.numpy().tolist()
                )

                subject_probabilities.extend(
                    probabilities.numpy().tolist()
                )

                continue

            flat, windows = preprocess(
                x,
                mean,
                std,
            )

            logits = forward_logits(
                model,
                flat.to(device),
            ).cpu()

            labels = expand_labels(
                y,
                windows,
            )

            if logits.shape[-1] == 1:
                window_predictions = binary_predictions(
                    logits
                )
            else:
                window_predictions = logits.argmax(
                    dim=1
                )

            probabilities = subject_prob(
                logits,
                x.shape[0],
                windows,
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            true_windows.extend(
                labels.tolist()
            )

            predicted_windows.extend(
                window_predictions.tolist()
            )

            true_subjects.extend(
                y.numpy().tolist()
            )

            predicted_subjects.extend(
                predictions.numpy().tolist()
            )

            subject_probabilities.extend(
                probabilities.numpy().tolist()
            )

            if (
                getattr(model, "full_subject_input", False)
                and eng_by_name is not None
            ):
                engineered = batch_x_eng(
                    list(names),
                    eng_by_name,
                    device,
                )

                normalized = preprocess_subject(
                    x,
                    mean,
                    std,
                )

                embedding = model.forward_features(
                    normalized.to(device),
                    x_eng=engineered,
                )

                embeddings.append(
                    embedding.detach().cpu().numpy()
                )

                embedding_names.extend(
                    list(names)
                )

    true_subjects = np.asarray(
        true_subjects,
        dtype=int,
    )

    predicted_subjects = np.asarray(
        predicted_subjects,
        dtype=int,
    )

    auc = None

    if len(np.unique(true_subjects)) > 1:
        auc = float(
            roc_auc_score(
                true_subjects,
                subject_probabilities,
            )
        )

    subject_embeddings = (
        np.concatenate(
            embeddings,
            axis=0,
        )
        if embeddings
        else None
    )

    return {
        "test_true": true_subjects.tolist(),
        "test_pred": predicted_subjects.tolist(),
        "test_cm_window": confusion_matrix(
            true_windows,
            predicted_windows,
        ),
        "test_cm_subject": confusion_matrix(
            true_subjects,
            predicted_subjects,
        ),
        "test_roc": {
            "y_true": true_subjects.tolist(),
            "y_prob": subject_probabilities,
        },
        "test_auc": auc,
        "test_z_eeg": (
            subject_embeddings.tolist()
            if subject_embeddings is not None
            else None
        ),
        "test_emb_subjects": (
            embedding_names
            if embedding_names
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unimodal MODMA EEG classification"
    )

    parser.add_argument(
        "--modal",
        type=str,
        default="eeg",
        choices=["eeg"],
    )

    parser.add_argument(
        "--representation",
        type=str,
        default="time",
        choices=["time"],
    )

    parser.add_argument(
        "--norm",
        type=str,
        default="fold",
        choices=["fold"],
    )

    parser.add_argument(
        "--channels",
        type=str,
        default="f64",
        choices=["all", "10-20", "f64", "29"],
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--reference",
        type=str,
        default="average",
        choices=["average", "cz"],
    )

    parser.add_argument(
        "--lowcut",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--highcut",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--target-fs",
        type=float,
        default=250.0,
    )

    parser.add_argument(
        "--notch",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deepconvnet",
        choices=sorted(MODEL_CLASSES),
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--n-filters",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="bce",
        choices=["ce", "bce"],
    )

    parser.add_argument(
        "--early-stop-on",
        type=str,
        default="window-bacc",
        choices=["subject-bacc", "window-bacc"],
    )

    parser.add_argument(
        "--no-bce-pos-weight",
        action="store_true",
    )

    parser.add_argument(
        "--consistency-coef",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--inner-splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        nargs="+",
        default=[42],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="eeg_64ch",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--pool",
        type=str,
        default="mean",
        choices=["mean", "attention"],
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--refit",
        action="store_true",
    )

    parser.add_argument(
        "--refit-epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    if (
        args.model == "deepconvnet"
        and args.loss != "bce"
    ):
        raise SystemExit(
            "DeepConvNet debe ejecutarse con --loss bce."
        )

    if (
        args.pool == "attention"
        and args.consistency_coef == 0
    ):
        raise SystemExit(
            "El pooling attention requiere "
            "--consistency-coef > 0."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = MODMADataset(
        channels=args.channels,
        overlap=args.overlap,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        reference=args.reference,
    )

    n_channels = len(
        dataset.channel_names
    )

    n_samples = dataset.samples[0][
        "eeg"
    ].shape[-1]

    engineered_features = build_eng_feats(
        dataset
    )

    engineered_dim = eng_dim(
        engineered_features
    )

    n_classes = (
        1
        if args.loss == "bce"
        else 2
    )

    model_header = build_model(
        args.model,
        n_channels,
        n_classes=n_classes,
        n_samples=n_samples,
        dropout=args.dropout,
        hidden=args.hidden,
        n_filters=args.n_filters,
        engineered_dim=engineered_dim,
        pool=args.pool,
    )

    total_params, trainable_params, frozen_params = count_parameters(
        model_header
    )

    gpu = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    for split_seed in args.split_seed:
        folds = create_dataloaders(
            dataset,
            k_folder=args.k,
            inner_split=args.inner_splits,
            split_seed=split_seed,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=device == "cuda",
        )

        output_dir = (
            Path(args.output_root)
            / (
                f"unimodals_sgkf_{args.modal}"
                f"_sseed{split_seed}"
                f"_tag{args.tag}"
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        config = {
            "identity": {
                "modal": args.modal,
                "model": args.model,
                "tag": args.tag,
                "output_dir": str(output_dir),
            },
            "data": {
                "root": str(dataset.root),
                "channel_names": dataset.channel_names,
                "n_subjects": len(dataset.samples),
                "n_channels": n_channels,
                "n_samples": n_samples,
                "window_sec": dataset.window_sec,
                "overlap": args.overlap,
                "lowcut": dataset.lowcut,
                "highcut": dataset.highcut,
                "notch": dataset.notch,
                "target_fs": dataset.target_fs,
                "reference": dataset.reference,
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
                "loss": args.loss,
                "early_stop_on": args.early_stop_on,
                "bce_pos_weight": not args.no_bce_pos_weight,
                "refit": args.refit,
                "refit_epochs": args.refit_epochs,
            },
            "model": {
                "constructor": {
                    "name": args.model,
                    "n_channels": n_channels,
                    "n_classes": n_classes,
                    "n_samples": n_samples,
                    "dropout": args.dropout,
                },
                "total_params": total_params,
                "trainable_params": trainable_params,
                "frozen_params": frozen_params,
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

        logger = ClassificationLogger()
        fold_results = []

        def save_results() -> None:
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
                "folds": fold_results,
            }

            with open(
                output_dir / "results.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    results,
                    file,
                    indent=2,
                )

        for fold_index, (
            inner_folds,
            outer_train_loader,
            test_loader,
        ) in enumerate(folds, start=1):
            inner_best_epochs = []

            for inner_index, (
                inner_train_loader,
                inner_val_loader,
            ) in enumerate(inner_folds, start=1):
                train_mean, train_std = channel_stats(
                    inner_train_loader
                )
                scaled_engineered_features = fit_eng_scaler(
                    engineered_features,
                    inner_train_loader.dataset.names,
                    inner_val_loader.dataset.names,
                )
                model = build_model(
                    args.model,
                    n_channels,
                    n_classes=n_classes,
                    n_samples=n_samples,
                    dropout=args.dropout,
                    hidden=args.hidden,
                    n_filters=args.n_filters,
                    engineered_dim=engineered_dim,
                    pool=args.pool,
                ).to(device)
                _, best_epoch = train_fold(
                    model,
                    inner_train_loader,
                    inner_val_loader,
                    args.epochs,
                    args.lr,
                    args.weight_decay,
                    args.patience,
                    device,
                    logger,
                    train_mean,
                    train_std,
                    args.label_smoothing,
                    args.loss,
                    early_stop_on=args.early_stop_on,
                    bce_pos_weight=not args.no_bce_pos_weight,
                    eng_by_name=scaled_engineered_features,
                    consistency_coef=args.consistency_coef,
                )
                inner_best_epochs.append(best_epoch)
                del model
                torch.cuda.empty_cache()

            avg_best_ep = int(round(np.mean(inner_best_epochs)))
            train_mean, train_std = channel_stats(
                outer_train_loader
            )
            scaled_engineered_features = fit_eng_scaler(
                engineered_features,
                outer_train_loader.dataset.names,
                [],
            )
            model = build_model(
                args.model,
                n_channels,
                n_classes=n_classes,
                n_samples=n_samples,
                dropout=args.dropout,
                hidden=args.hidden,
                n_filters=args.n_filters,
                engineered_dim=engineered_dim,
                pool=args.pool,
            ).to(device)
            train_fixed_epochs(
                model,
                outer_train_loader,
                avg_best_ep,
                args.lr,
                args.weight_decay,
                device,
                train_mean,
                train_std,
                args.label_smoothing,
                args.loss,
                bce_pos_weight=not args.no_bce_pos_weight,
                eng_by_name=scaled_engineered_features,
            )
            history = {"inner_best_epochs": inner_best_epochs}
            best_epoch = avg_best_ep

            fold_result = run_fold_test(
                model,
                test_loader,
                device,
                train_mean,
                train_std,
                eng_by_name=scaled_engineered_features,
            )

            fold_result["fold"] = fold_index
            fold_result["history"] = history
            fold_result["best_epoch"] = best_epoch
            fold_result["train_subjects"] = (
                list(outer_train_loader.dataset.names)
            )
            fold_result["inner_best_epochs"] = inner_best_epochs
            fold_result["avg_best_ep"] = avg_best_ep
            fold_result["test_subjects"] = list(
                test_loader.dataset.names
            )

            fold_result["test_metrics"] = (
                logger.log_fold_test(
                    fold_result["test_true"],
                    fold_result["test_pred"],
                    fold_result["test_auc"],
                )
            )

            fold_results.append(
                fold_result
            )

            if args.save_model:
                torch.save(
                    model.state_dict(),
                    output_dir / f"fold_{fold_index}.pt",
                )

            save_results()

            del model
            torch.cuda.empty_cache()

        logger.log_summary(
            n_folds=args.k,
            split_type="gkf",
        )

        print(
            f"Final results: "
            f"{output_dir / 'results.json'}"
        )


if __name__ == "__main__":
    main()