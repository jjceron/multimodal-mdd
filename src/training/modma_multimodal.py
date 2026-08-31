from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.models.crossattnfusion import CrossAttnFusion
from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.preprocessing.modma_dataset import MODMASubjects
from src.utils.get_seed import set_seed
from src.utils.training_logger import ClassificationLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _window_norm(v):
    return (v - v.mean(dim=(1, 2), keepdim=True)) / (v.std(dim=(1, 2), keepdim=True) + 1e-8)


def make_criterion(device, pos_weight=None):
    pw = torch.tensor(float(pos_weight)).to(device) if pos_weight else None
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw)


def _blocks(subject_ids, subj, device):
    e = torch.stack([subj[i]["eeg"] for i in subject_ids]).to(device)
    a = torch.stack([subj[i]["aud"] for i in subject_ids]).to(device)
    y = torch.tensor([subj[i]["label"] for i in subject_ids], dtype=torch.float32, device=device)
    return e, a, y


def _features(backbone, block, device, chunk=128):
    B, K = block.shape[0], block.shape[1]
    flat = block.reshape(B * K, *block.shape[2:])
    outs = []
    for i in range(0, flat.shape[0], chunk):
        outs.append(backbone.forward_features(_window_norm(flat[i : i + chunk])))
    return torch.cat(outs, 0).reshape(B, K, -1)


def _subject_logit(beeg, baud, fusion, e_block, a_block, device):
    e_feat = _features(beeg, e_block, device)
    a_feat = _features(baud, a_block, device)
    return fusion(e_feat, a_feat)


def pretrain_backbone(model, subjects, key, epochs, lr, wd, device, label):
    win_list = []
    for pid in sorted(subjects):
        for w in subjects[pid][key]:
            win_list.append((w, subjects[pid]["label"]))
    if not win_list:
        print(f"  {label}: no windows, skipped")
        return model
    print(f"  {label}: subjects={len(subjects)} windows={len(win_list)}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = make_criterion(device, pos_weight=1.0)
    n = len(win_list)
    model.train()
    for ep in range(1, max(epochs, 1) + 1):
        idx = torch.randperm(n)
        ep_loss, nb = 0.0, 0
        for i in range(0, n, 256):
            bi = idx[i : i + 256]
            xb = torch.stack([win_list[j][0] for j in bi]).to(device)
            yb = torch.tensor([win_list[j][1] for j in bi], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = crit(model(_window_norm(xb)).squeeze(-1), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(bi)
            nb += len(bi)
        if ep == 1 or ep % 20 == 0 or ep == max(epochs, 1):
            print(f"  {label} epoch {ep}/{max(epochs, 1)}  loss={ep_loss / max(nb, 1):.4f}")
    return model


def _training_fusion_criterion(device, train_ids, subj):
    labels = [subj[t]["label"] for t in train_ids]
    n_hc = sum(1 for lb in labels if lb == 0)
    n_mdd = sum(1 for lb in labels if lb == 1)
    pw = (n_hc / n_mdd) if n_mdd > 0 else 1.0
    return make_criterion(device, pos_weight=pw)


def train_husm_backbone(device, epochs=50, lr=3e-4, wd=1e-2, dropout=0.5,
                        batch_size=256, lowcut=0.5, highcut=50.0,
                        fs_target=256.0, window_sec=2.0, overlap=0.5,
                        save_path=None):
    from src.preprocessing.husm_dataset import HUSMDataset

    ds = HUSMDataset(lowcut=lowcut, highcut=highcut, fs_target=fs_target,
                     window_sec=window_sec, overlap=overlap)
    xs, ys = ds.window_tensors()
    xs = xs.to(device)
    ys = ys.float().to(device)
    model = DeepConvNet(ds.n_channels, 1, ds.n_samples, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, foreach=False)
    crit = make_criterion(device, pos_weight=1.0)
    n = len(xs)
    print(f"  HUSM: windows={n} (n_channels={ds.n_channels}, n_samples={ds.n_samples})")
    model.train()
    for ep in range(1, max(epochs, 1) + 1):
        idx = torch.randperm(n)
        ep_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            bi = idx[i : i + batch_size]
            xb, yb = xs[bi], ys[bi]
            opt.zero_grad()
            loss = crit(model(_window_norm(xb)).squeeze(-1), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(bi)
            nb += len(bi)
        if ep == 1 or ep % 10 == 0 or ep == max(epochs, 1):
            print(f"  HUSM epoch {ep}/{max(epochs, 1)}  loss={ep_loss / max(nb, 1):.4f}")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  HUSM backbone saved: {save_path}")
    return model


def load_husm_backbone(n_channels, n_samples, dropout, path, device):
    model = DeepConvNet(n_channels, 1, n_samples, dropout).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_pretrained_aud_backbone(aud_n_channels, aud_n_samples, dropout, seed, fold_index, device):
    base = PROJECT_ROOT / "outputs/results/unimodals/aud"
    model_dir = base / f"tagv1wnorm_sseed{seed}_sngkf_aud_shallowconvnet_64mel"
    pt_path = model_dir / f"fold_{fold_index}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Pretrained AUD backbone not found: {pt_path}")
    model = ShallowConvNet(aud_n_channels, 1, aud_n_samples, dropout).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    return model


def train_fusion(beeg, baud, fusion, train_ids, val_ids, subj, epochs, lr, wd, patience, device, logger, bs=4, finetune=False):
    beeg.requires_grad_(finetune)
    baud.requires_grad_(finetune)
    params = list(fusion.parameters())
    if finetune:
        params += list(beeg.parameters()) + list(baud.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, foreach=False)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    crit = _training_fusion_criterion(device, train_ids, subj)
    best, best_state, best_epoch, patience_left = -1.0, None, 0, 0
    history = {
        "train loss": [], "val loss": [], "train acc": [], "val acc": [],
        "train bacc": [], "val bacc": [],
    }
    logger.log_header()
    for epoch in range(1, epochs + 1):
        beeg.eval()
        baud.eval()
        fusion.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        tr_true, tr_pred = [], []
        idx = np.random.permutation(train_ids)
        for i in range(0, len(idx), bs):
            chunk = idx[i : i + bs]
            e, a, y = _blocks(chunk, subj, device)
            logit = _subject_logit(beeg, baud, fusion, e, a, device)
            opt.zero_grad()
            loss = crit(logit, y * 0.95 + 0.025)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            pred = (torch.sigmoid(logit) >= 0.5).long()
            tr_loss += loss.item() * len(y)
            tr_correct += int((pred == y.long()).sum())
            tr_total += len(y)
            tr_true.extend(y.long().tolist())
            tr_pred.extend(pred.tolist())
        tr_loss /= max(tr_total, 1)
        tr_m = logger.metrics(tr_true, tr_pred)

        beeg.eval()
        baud.eval()
        fusion.eval()
        val_true, val_pred, val_loss = [], [], 0.0
        with torch.no_grad():
            for i in range(0, len(val_ids), 16):
                chunk = val_ids[i : i + 16]
                e, a, y = _blocks(chunk, subj, device)
                logit = _subject_logit(beeg, baud, fusion, e, a, device)
                val_loss += crit(logit, y * 0.95 + 0.025).item() * len(y)
                val_true.extend(y.tolist())
                val_pred.extend((torch.sigmoid(logit) >= 0.5).long().tolist())
        val_loss /= max(len(val_true), 1)
        val_m = logger.metrics(val_true, val_pred)
        metric = val_m["bacc"]
        sched.step(metric)
        if metric > best:
            best, best_epoch, patience_left = metric, epoch, 0
            best_state = {
                "beeg": beeg.state_dict(),
                "baud": baud.state_dict(),
                "fusion": fusion.state_dict(),
            }
        else:
            patience_left += 1
        history["train loss"].append(float(tr_loss))
        history["val loss"].append(float(val_loss))
        history["train acc"].append(float(tr_m["acc"]))
        history["val acc"].append(float(val_m["acc"]))
        history["train bacc"].append(float(tr_m["bacc"]))
        history["val bacc"].append(float(val_m["bacc"]))
        logger.log_epoch(epoch, tr_loss, val_loss, tr_m, val_m, patience_left)
        if patience_left >= patience:
            break
    if best_state is not None:
        beeg.load_state_dict(best_state["beeg"])
        baud.load_state_dict(best_state["baud"])
        fusion.load_state_dict(best_state["fusion"])
    return beeg, baud, fusion, max(best_epoch, 1), val_m, history


def refit_fusion(beeg, baud, fusion, train_ids, val_ids, subj, epochs, lr, wd, device, bs=4, finetune=False):
    beeg.requires_grad_(finetune)
    baud.requires_grad_(finetune)
    params = list(fusion.parameters())
    if finetune:
        params += list(beeg.parameters()) + list(baud.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, foreach=False)
    crit = _training_fusion_criterion(device, train_ids + val_ids, subj)
    all_ids = list(train_ids) + list(val_ids)
    for _ in range(max(epochs, 1)):
        beeg.eval()
        baud.eval()
        fusion.train()
        idx = np.random.permutation(all_ids)
        for i in range(0, len(idx), bs):
            chunk = idx[i : i + bs]
            e, a, y = _blocks(chunk, subj, device)
            opt.zero_grad()
            logit = _subject_logit(beeg, baud, fusion, e, a, device)
            crit(logit, y * 0.95 + 0.025).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return beeg, baud, fusion


def evaluate_fusion(beeg, baud, fusion, test_ids, subj, device):
    beeg.eval()
    baud.eval()
    fusion.eval()
    true, prob = [], []
    with torch.no_grad():
        for i in range(0, len(test_ids), 8):
            chunk = test_ids[i : i + 8]
            e, a, y = _blocks(chunk, subj, device)
            logit = _subject_logit(beeg, baud, fusion, e, a, device)
            prob.extend(torch.sigmoid(logit / fusion.temperature).tolist())
            true.extend(y.tolist())
    true = np.array(true, dtype=int)
    prob = np.array(prob)
    return true, (prob >= 0.5).astype(int), prob


def fit_temperature(beeg, baud, fusion, val_ids, subj, device):
    beeg.eval()
    baud.eval()
    fusion.eval()
    logits, ys = [], []
    with torch.no_grad():
        for i in range(0, len(val_ids), 16):
            chunk = val_ids[i : i + 16]
            e, a, y = _blocks(chunk, subj, device)
            logit = _subject_logit(beeg, baud, fusion, e, a, device)
            logits.append(logit.cpu())
            ys.append(y.cpu())
    logits = torch.cat(logits)
    ys = torch.cat(ys)
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.3, 3.0, 55):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / t, ys).item()
        if loss < best_loss:
            best_loss, best_t = loss, t
    return float(best_t)


def _aggregate(fold_results):
    keys = ("acc", "bacc", "f1", "sens", "spec")
    agg = {}
    for k in keys:
        vals = [f["test_metrics"][k] for f in fold_results.values()]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    aucs = [f["test_auc"] for f in fold_results.values() if f["test_auc"] is not None]
    agg["auc_mean"] = float(np.mean(aucs)) if aucs else None
    agg["auc_std"] = float(np.std(aucs)) if aucs else None
    return agg

def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def load_dataset(args):
    from src.preprocessing.modma_aud import MODMAAudioDataset
    from src.preprocessing.modma_eeg import MODMADataset

    eeg_ds = MODMADataset(
        channels=args.channels, overlap=args.overlap, lowcut=args.lowcut,
        highcut=args.highcut, notch=args.notch, target_fs=args.target_fs,
        reference=args.reference,
    )
    aud_ds = MODMAAudioDataset(window_sec=args.window_sec, overlap=args.overlap)
    return eeg_ds, aud_ds


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-modal attention fusion for EEG+Audio")
    parser.add_argument("--channels", default="f64")
    parser.add_argument("--target-fs", type=float, default=250.0)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=50.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--reference", default="average")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, nargs="+", default=[42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bb-epochs", type=int, default=250)
    parser.add_argument("--bb-lr", type=float, default=3e-4)
    parser.add_argument("--bb-wd", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--fusion-dropout", type=float, default=0.6)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--n-self-attn-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--finetune", action="store_true", help="Fine-tune backbones jointly (end-to-end)")
    parser.add_argument("--save-model", action="store_true", help="Save multimodal model checkpoints per fold")
    parser.add_argument("--tag", type=str, default="v1")
    parser.add_argument("--pretrain-husm", action="store_true", help="Train the EEG backbone on the HUSM cohort (MDD/HC)")
    parser.add_argument("--husm-epochs", type=int, default=50)
    parser.add_argument("--husm-lr", type=float, default=3e-4)
    parser.add_argument("--husm-wd", type=float, default=1e-2)
    parser.add_argument("--husm-weights", type=str, default=str(PROJECT_ROOT / "outputs/pretrained/husm_deepconvnet_19_256.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    eeg_ds, aud_ds = load_dataset(args)
    subj = MODMASubjects(eeg_ds, aud_ds)

    # Shared minimum window count (uniform K per modality -> enables window-level averaging).
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

    eeg_n_channels = len(eeg_ds.channel_names)
    eeg_n_samples = int(eeg_ds.samples[0]["eeg"].shape[-1])
    aud_n_channels = int(aud_ds.samples[0]["logmel"].shape[1])
    aud_n_samples = int(aud_ds.samples[0]["logmel"].shape[-1])
    dim_e = DeepConvNet(eeg_n_channels, 1, eeg_n_samples, 0.5).fc_features
    dim_a = ShallowConvNet(aud_n_channels, 1, aud_n_samples, 0.5).classifier.in_features

    non_eeg = {pid: {"label": s["label"], "eeg": s["eeg"]} for pid, s in subj.non_paired_eeg.items()}
    non_aud = {pid: {"label": s["label"], "aud": s["aud"]} for pid, s in subj.non_paired_aud.items()}

    print("\n-- HUSM pretrain (leak-free external cohort for the EEG backbone) --")
    husm_weights = Path(args.husm_weights)
    if args.pretrain_husm or not husm_weights.exists():
        train_husm_backbone(
            device, epochs=args.husm_epochs, lr=args.husm_lr, wd=args.husm_wd,
            dropout=args.dropout, save_path=str(husm_weights),
        )
    else:
        print(f"  Using existing HUSM backbone: {husm_weights}")

    for split_seed in args.split_seed:
        folds = subj.folds(k=args.k, val_ratio=args.val_ratio, split_seed=split_seed)
        out_dir = PROJECT_ROOT / "outputs/results/multimodals" / f"tag{args.tag}_sseed{split_seed}_sngkf_multimodal"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = ClassificationLogger()
        fold_results = {}
        results = {
            "config": {
                "timestamp": datetime.now(UTC).isoformat(),
                "paired": len(subj.paired),
                "non_paired_eeg": len(subj.non_paired_eeg),
                "non_paired_aud": len(subj.non_paired_aud),
                "kmin": kmin,
                "cli": vars(args),
                "git_commit": _git_commit(),
            },
            "test": {},
            "folds": {},
            "training curves": {},
        }

        def write_results(out_dir, results):
            with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)

        for fold_index, (train_ids, val_ids, test_ids) in enumerate(folds, start=1):
            print(f"\n===== FOLD {fold_index} =====")
            if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
                raise RuntimeError("Subject overlap in multimodal fold")

            print("\n-- Phase 1: load frozen backbones (HUSM-EEG + pretrained-AUD) --")
            bb_eeg = {pid: {"label": subj.paired[pid]["label"], "eeg": subj.paired[pid]["eeg"]} for pid in train_ids}
            bb_eeg.update(non_eeg)
            bb_aud = {pid: {"label": subj.paired[pid]["label"], "aud": subj.paired[pid]["aud"]} for pid in train_ids}
            bb_aud.update(non_aud)
            if set(bb_eeg) & (set(val_ids) | set(test_ids)) or set(bb_aud) & (set(val_ids) | set(test_ids)):
                raise RuntimeError("Backbone leakage: backbone subjects overlap val/test")

            beeg = load_husm_backbone(eeg_n_channels, eeg_n_samples, args.dropout, husm_weights, device)
            baud = load_pretrained_aud_backbone(aud_n_channels, aud_n_samples, args.dropout, split_seed, fold_index, device)
            beeg.requires_grad_(False)
            baud.requires_grad_(False)

            fusion = CrossAttnFusion(
                dim_e, dim_a, args.hidden, args.n_heads,
                n_self_attn_layers=args.n_self_attn_layers,
                dropout=args.fusion_dropout, attn_dropout=args.attn_dropout,
            ).to(device)

            print("\n-- Phase 2: cross-modal fusion (early-stop on val) --")
            beeg, baud, fusion, best_ep, vl_m, history = train_fusion(
                beeg, baud, fusion, train_ids, val_ids, subj.paired, args.epochs,
                args.lr, args.weight_decay, args.patience, device, logger,
                args.batch_size, args.finetune,
            )
            results["training curves"][f"fold {fold_index}"] = history

            print(f"\n-- Phase 3: refit (train+val, epochs={best_ep}) --")
            beeg, baud, fusion = refit_fusion(
                beeg, baud, fusion, train_ids, val_ids, subj.paired, best_ep,
                args.lr, args.weight_decay, device, args.batch_size, args.finetune,
            )

            fusion.temperature.data = torch.tensor(1.0)
            print("  Temperature fixed to 1.0 (no calibration)")

            true, pred, prob = evaluate_fusion(beeg, baud, fusion, test_ids, subj.paired, device)
            auc = float(roc_auc_score(true, prob)) if len(set(true.tolist())) > 1 else None
            fm = logger.log_fold_test(true.tolist(), pred.tolist(), auc)
            fold_results[f"fold {fold_index}"] = {
                "test_metrics": fm,
                "test_auc": auc,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_test": len(test_ids),
                "val_metrics": {k: float(vl_m[k]) for k in ("bacc", "f1", "sens", "spec")},
                "best_epoch": best_ep,
            }
            results["folds"] = fold_results
            results["test"] = _aggregate(fold_results)
            if args.save_model:
                torch.save(
                    {"beeg": beeg.state_dict(), "baud": baud.state_dict(), "fusion": fusion.state_dict()},
                    out_dir / f"fold_{fold_index}.pt",
                )
            write_results()

        logger.log_summary(n_folds=len(fold_results), split_type="gkf")
        print(f"\nFinal: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
