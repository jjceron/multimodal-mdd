from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.utils.get_seed import set_seed
from src.utils.training_logger import ClassificationLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_WORKERS = 4 if platform.system() != "Windows" else 0

MODEL_CLASSES = {
    "deepconvnet": DeepConvNet,
    "shallowconvnet": ShallowConvNet,
}


def channel_stats(loader) -> tuple[torch.Tensor, torch.Tensor]:
    total = None
    total_squared = None
    total_values = 0
    for _, x, _ in loader:
        channel_dim = 1 if x.dim() == 3 else 2
        reduce_dims = tuple(d for d in range(x.dim()) if d != channel_dim)
        total = x.sum(reduce_dims) if total is None else total + x.sum(reduce_dims)
        total_squared = x.square().sum(reduce_dims) if total_squared is None else total_squared + x.square().sum(reduce_dims)
        total_values += x.numel() // x.shape[channel_dim]
    mean = total / total_values
    variance = total_squared / total_values - mean.square()
    return mean, variance.clamp_min(0).sqrt()


def normalize_channels(x, mean, std):
    shape = [1, -1] + [1] * (x.dim() - 2)
    return (x - mean.view(shape)) / (std.view(shape) + 1e-8)


def make_criterion(device, train_loader=None, pos_weight=None):
    pw = None
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), dtype=torch.float32).to(device)
    elif train_loader is not None:
        counts = torch.bincount(torch.cat([y for _, _, y in train_loader]).long())
        if counts.numel() > 1 and float(counts[1]) > 0:
            pw = (counts[0].float() / counts[1].float()).to(device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw)


def step_loss(criterion, logits, yf):
    smoothed = yf.float() * 0.95 + 0.025
    return criterion(logits.squeeze(-1), smoothed)


def expand_labels(y, windows):
    return y.view(-1, 1).repeat(1, windows).reshape(-1)


def binary_predictions(logits):
    return torch.sigmoid(logits.squeeze(-1)) >= 0.5


def subject_prob(logits, n_subjects, windows):
    pooled = logits.reshape(n_subjects, windows, -1).mean(dim=1)
    if pooled.shape[-1] == 1:
        return torch.sigmoid(pooled[:, 0])
    return pooled.softmax(dim=1)[:, 1]


def forward_batch(model, x, y, mean, std, device):
    if x.dim() == 4:
        flat, windows = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), x.shape[1]
        logits = model(normalize_channels(flat, mean, std).to(device))
        return logits, expand_labels(y, windows).to(device)
    normalized = normalize_channels(x, mean, std)
    return model(normalized.to(device)), y.to(device)


def build_model(name, n_channels, n_classes, n_samples, dropout):
    return MODEL_CLASSES[name](
        n_channels=n_channels, n_classes=n_classes,
        n_samples=n_samples, dropout=dropout,
    )


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def confusion_matrix(true, pred):
    t, p = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    return [
        [int(((t == 0) & (p == 0)).sum()), int(((t == 0) & (p == 1)).sum())],
        [int(((t == 1) & (p == 0)).sum()), int(((t == 1) & (p == 1)).sum())],
    ]


def train_fold(model, train_loader, val_loader, epochs, lr, weight_decay,
               patience, device, logger, mean, std, pos_weight=None, early_stop_on="subject-bacc"):
    
    criterion = make_criterion(device, train_loader=train_loader, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best, best_state, best_epoch, patience_left = -1.0, None, 0, 0
    logger.log_header()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for name, xb, yb in train_loader:
            logits, yf = forward_batch(model, xb, yb, mean, std, device)
            optimizer.zero_grad()
            tloss = step_loss(criterion, logits, yf)
            tloss.backward()
            optimizer.step()
            tr_loss += tloss.item() * len(yf)
            tr_correct += int((binary_predictions(logits).cpu() == yf.cpu()).sum())
            tr_total += len(yf)

        tr_loss /= max(tr_total, 1)
        model.eval()
        val_true, val_pred, val_loss = [], [], 0.0
        subj_logit, subj_count, subj_label = {}, {}, {}
        with torch.no_grad():
            for name, xb, yb in val_loader:
                logits, yf = forward_batch(model, xb, yb, mean, std, device)
                val_loss += step_loss(criterion, logits, yf).item() * len(yf)
                val_true.extend(yf.cpu().tolist())
                val_pred.extend(binary_predictions(logits).cpu().tolist())
                lgts = logits.squeeze(-1).cpu().tolist()
                for n, val in zip(name, lgts):
                    subj_logit[n] = subj_logit.get(n, 0.0) + val
                    subj_count[n] = subj_count.get(n, 0) + 1
                    subj_label[n] = int(yb[list(name).index(n)])
        val_loss /= max(len(val_true), 1)
        vl_m = logger.metrics(val_true, val_pred)

        if early_stop_on == "subject-bacc":
            s_true = [subj_label[n] for n in subj_logit]
            s_pred = [1 if (subj_logit[n] / subj_count[n]) >= 0.0 else 0 for n in subj_logit]
            metric = logger.metrics(s_true, s_pred)["bacc"]
        else:
            metric = vl_m["bacc"]
        scheduler.step(metric)
        if metric > best:
            best, best_epoch, patience_left = metric, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left += 1
        logger.log_epoch(epoch, tr_loss, val_loss, {"acc": tr_correct / max(tr_total, 1)}, vl_m, patience_left)
        if patience_left >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, max(best_epoch, 1), vl_m


def train_fixed_epochs(model, train_loader, epochs, lr, weight_decay, device,
                       mean, std, pos_weight=None):
    
    criterion = make_criterion(device, train_loader=train_loader, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(max(epochs, 1)):
        for name, xb, yb in train_loader:
            logits, yf = forward_batch(model, xb, yb, mean, std, device)
            optimizer.zero_grad()
            step_loss(criterion, logits, yf).backward()
            optimizer.step()
    return model


def run_fold_test(model, test_loader, device, mean, std) -> dict:
    true_subj, pred_subj, prob_subj = [], [], []
    true_win, pred_win = [], []

    model.eval()
    with torch.no_grad():
        for names, x, y in test_loader:
            flat, windows = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), x.shape[1]
            logits = model(normalize_channels(flat, mean, std).to(device)).cpu()
            yf = expand_labels(y, windows)
            true_win.extend(yf.tolist())
            pred_win.extend(binary_predictions(logits).tolist())
            prob = subject_prob(logits, x.shape[0], windows)
            true_subj.extend(y.numpy().tolist())
            pred_subj.extend((prob >= 0.5).long().numpy().tolist())
            prob_subj.extend(prob.numpy().tolist())

    true, pred = np.asarray(true_subj, dtype=int), np.asarray(pred_subj, dtype=int)
    auc = float(roc_auc_score(true, prob_subj)) if len(set(true.tolist())) > 1 else None
    metrics = ClassificationLogger().metrics(true, pred)
    return {
        "test_auc": auc,
        "test_cm_subject": confusion_matrix(true, pred),
        "test_metrics": metrics,
        "test_true": true.tolist(),
        "test_pred": pred.tolist(),
    }


def _load_dataset(args) -> tuple[object, int, int]:
    if args.modal == "eeg":
        from src.preprocessing.modma_eeg import MODMADataset
        ds = MODMADataset(
            channels=args.channels, overlap=args.overlap, lowcut=args.lowcut,
            highcut=args.highcut, notch=args.notch, target_fs=args.target_fs,
            reference=args.reference,
        )
        n_channels = len(ds.channel_names)
        n_samples = int(ds.samples[0]["eeg"].shape[-1])
        return ds, n_channels, n_samples

    from src.preprocessing.modma_aud import MODMAAudioDataset
    ds = MODMAAudioDataset(window_sec=args.window_sec, overlap=args.overlap)
    n_channels = int(ds.samples[0]["logmel"].shape[1])
    n_samples = int(ds.samples[0]["logmel"].shape[-1])
    return ds, n_channels, n_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unimodal classification for EEG and Audio MODMA modalities")
    parser.add_argument("--modal", type=str, default="eeg", choices=["eeg", "aud"])
    parser.add_argument("--model", type=str, default="deepconvnet", choices=sorted(MODEL_CLASSES))
    parser.add_argument("--channels", type=str, default="f64", choices=["all", "10-20", "f64", "29"])
    parser.add_argument("--target-fs", type=float, default=250.0)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=50.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--reference", type=str, default="average", choices=["average", "cz"])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--early-stop-on", type=str, default="subject-bacc", choices=["window-bacc", "subject-bacc"])
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="Override BCE pos_weight (None = auto class-balanced)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, nargs="+", default=[42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--tag", type=str, default="v1")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--save-model", action="store_true", help="Save the final model per-fold model checkpoint '.pt'")

    return parser.parse_args()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception: # noqa: BLE001
        return ""


def _aggregate(fold_results) -> dict:
    folds = fold_results.values() if isinstance(fold_results, dict) else fold_results
    keys = ("acc", "bacc", "f1", "sens", "spec")
    agg: dict[str, float] = {}
    for k in keys:
        vals = [f["test_metrics"][k] for f in folds]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    aucs = [f["test_auc"] for f in folds if f["test_auc"] is not None]
    agg["auc_mean"] = float(np.mean(aucs))
    agg["auc_std"] = float(np.std(aucs))
    return agg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, n_channels, n_samples = _load_dataset(args)

    windows = (
        int(min(s["eeg"].shape[0] for s in dataset.samples))
        if args.modal == "eeg"
        else None  # audio: variable window count per subject
    )
    n_classes = 1 

    header = build_model(args.model, n_channels, n_classes=n_classes, n_samples=n_samples, dropout=args.dropout)
    total, trainable, _ = count_parameters(header)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    output_root = Path(args.output_root) if args.output_root else (
        PROJECT_ROOT / "outputs/results/unimodals" / args.modal
    )

    for split_seed in args.split_seed:
        if args.modal == "aud":
            from src.preprocessing.modma_aud import create_audio_dataloaders
            loader_fn = create_audio_dataloaders
        else:
            from src.preprocessing.modma_eeg import create_dataloaders
            loader_fn = create_dataloaders

        folds = loader_fn(
            dataset, k_folder=args.k, inner_split=args.inner_splits,
            split_seed=split_seed, batch_size=args.batch_size,
            num_workers=NUM_WORKERS, pin_memory=device == "cuda",
        )

        ch_label = "64mel" if args.modal == "aud" else args.channels
        out_dir = output_root / (
            f"tag{args.tag}_sseed{split_seed}_sngkf_{args.modal}_{args.model}_{ch_label}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print(f" UNIMODAL | modal={args.modal} model={args.model} channels={args.channels} "
              f"subjects={len(dataset.samples)} n_ch={n_channels} n_samples={n_samples}")
        print(f" device={device} gpu={gpu} params(total={total}, trainable={trainable})")
        print("=" * 70)
 
        logger = ClassificationLogger()
        fold_results: dict[str, dict] = {}
        results = {
            "config": {
                "name": "unimodal_sngkf",
                "timestamp": datetime.now(UTC).isoformat(),
                "git_commit": _git_commit(),
                "windows": windows,
                "cli": vars(args),
            },
            "test": {},
            "folds": {},
        }

        def write_results(out_dir=out_dir, results=results) -> None:
            with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)

        for fold_index, (inner_folds, outer_loader, test_loader) in enumerate(folds, start=1):
            print(f"\n=== OUTER FOLD {fold_index} | inner={len(inner_folds)} ===")
            inner_best = []
            inner_metrics = []
            for inner_index, (inner_train_loader, inner_val_loader) in enumerate(inner_folds, start=1):
                print(f"  --- INNER FOLD {inner_index} ---")
                mean, std = channel_stats(inner_train_loader)
                model = build_model(args.model, n_channels, n_classes=n_classes,
                                    n_samples=n_samples, dropout=args.dropout).to(device)
                model, best_epoch, vl_m = train_fold(
                    model, inner_train_loader, inner_val_loader, args.epochs, args.lr,
                    args.weight_decay, args.patience, device, logger, mean, std,
                    pos_weight=args.pos_weight, early_stop_on=args.early_stop_on
                )
                inner_best.append(best_epoch)
                inner_metrics.append(vl_m)
                del model
                torch.cuda.empty_cache()

            avg_best_ep = round(float(np.mean(inner_best)))
            mean, std = channel_stats(outer_loader)
            model = build_model(args.model, n_channels, n_classes=n_classes,
                                n_samples=n_samples, dropout=args.dropout).to(device)
            model = train_fixed_epochs(
                model, outer_loader, avg_best_ep, args.lr, args.weight_decay, device,
                mean, std, pos_weight=args.pos_weight,
            )

            fres = run_fold_test(model, test_loader, device, mean, std)
            inner_means = {k: float(np.mean([m[k] for m in inner_metrics])) for k in ("bacc", "f1", "sens", "spec")}
            fold_res = {
                "test_metrics": fres["test_metrics"],
                "test_auc": fres["test_auc"],
                "test_cm": fres["test_cm_subject"],
                "n_train": len(set(outer_loader.dataset.names)),
                "n_test": len(set(test_loader.dataset.names)),
                "test_subjects": sorted(set(test_loader.dataset.names)),
                "inner_metrics": inner_means,
            }
            fold_results[f"fold {fold_index}"] = fold_res
            results["folds"] = fold_results
            results["test"] = _aggregate(fold_results)

            logger.log_fold_test(fres["test_true"], fres["test_pred"], fres["test_auc"])
            write_results()
            
            if args.save_model:
                torch.save(model.state_dict(), out_dir / f"fold_{fold_index}.pt")
                torch.save(
                    {"mean": mean.cpu(), "std": std.cpu()},
                    out_dir / f"fold_{fold_index}_stats.pt",
                )

            del model
            torch.cuda.empty_cache()

        logger.log_summary(n_folds=len(fold_results), split_type="gkf")
        print(f"\nFinal: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
