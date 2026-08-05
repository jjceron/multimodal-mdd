"""Plot training diagnostics for MODMA results.

Reads the per-experiment results.json written by src/training/modma_unimodals.py
and renders training curves, per-fold / overall confusion matrices and ROC.

Modes (mutually selecting an output):
  --metric bacc|acc|f1|sens|spec   : train/val curves for the chosen metric
  --roc                             : per-fold ROC (+ AUC) curves
  --cm                              : per-fold confusion matrices (windows+subjects)
  --cm-overall                      : aggregated confusion matrix across folds
  --all                             : all folds (default when --fold is omitted)

Selecting the experiment:
  --type unimodal|multimodal  --modal eeg|aud  --tag <tag>  [--split-seed]

Examples:
  poetry run python -m src.utils.plot_training --type unimodal --modal eeg \
      --tag deepconvnet --metric bacc --all
  poetry run python -m src.utils.plot_training --type unimodal --modal eeg \
      --tag deepconvnet --roc --fold 1,3
  poetry run python -m src.utils.plot_training --type unimodal --modal eeg \
      --tag deepconvnet --cm-overall
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg" if not os.environ.get("DISPLAY") and os.name != "nt" else "TkAgg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "results")
FIGURES_ROOT = os.path.join(PROJECT_ROOT, "outputs", "figures")

METRIC_KEYS = {
    "bacc": "val_bacc",
    "acc": "val_acc",
    "f1": "val_f1",
    "sens": "val_sens",
    "spec": "val_spec",
}


def find_results_path(type_: str, modal: str, split_seed: int, tag: str) -> str:
    base = os.path.join(RESULTS_ROOT, type_, modal)
    pattern = os.path.join(
        base, "**", f"*sseed{split_seed}*tag{tag}", "results.json"
    )
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        pattern = os.path.join(base, "**", f"*tag{tag}", "results.json")
        matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        print(f"ERROR: no results found for --type {type_} --modal {modal} "
              f"--tag {tag}")
        sys.exit(1)
    return matches[-1]


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_folds(arg: str | None) -> list[int] | None:
    if arg is None:
        return None
    folds = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            folds.append(int(part))
    return folds


def select_folds(folds_data, fold_ids: list[int] | None) -> list[dict]:
    if fold_ids is None:
        return list(folds_data)
    by_num = {}
    for fe in folds_data:
        by_num[fe["fold"]] = fe
    selected = []
    for fid in fold_ids:
        if fid not in by_num:
            print(f"WARNING: fold {fid} not found in results (have "
                  f"{sorted(by_num)})")
            continue
        selected.append(by_num[fid])
    if not selected:
        print("ERROR: no valid folds selected")
        sys.exit(1)
    return selected


def _plot_metric_curves(axes, hist: dict, fold_num: int, metric: str,
                        show_legend: bool):
    ax_l, ax_r = axes
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    ax_l.plot(epochs, hist["train_loss"], color="blue", label="Train Loss")
    if hist.get("val_loss"):
        ax_l.plot(epochs, hist["val_loss"], color="orange", linestyle="--",
                  label="Val Loss")
    ax_l.set_title(f"Fold {fold_num} — Loss")
    ax_l.set_xlabel("Epoch")
    ax_l.grid(True, alpha=0.3)
    if show_legend:
        ax_l.legend(fontsize=7, loc="upper right")

    key = METRIC_KEYS[metric]
    if hist.get("train_acc"):
        ax_r.plot(epochs, hist["train_acc"], color="blue", label="Train Acc")
    if hist.get(key):
        ax_r.plot(epochs, hist[key], color="orange", linestyle="--",
                  label=f"Val {metric.upper()}")
    ax_r.set_title(f"Fold {fold_num} — {metric.upper()}")
    ax_r.set_xlabel("Epoch")
    ax_r.grid(True, alpha=0.3)
    if show_legend:
        ax_r.legend(fontsize=7, loc="lower right")


def _plot_roc(axes, fold_entry: dict, fold_num: int, show_legend: bool):
    ax_l, ax_r = axes
    ax_l.axis("off")
    ax_l.text(0.5, 0.5, f"Fold {fold_num}",
              ha="center", va="center", transform=ax_l.transAxes)

    roc = fold_entry.get("test_roc") or {}
    auc = fold_entry.get("test_auc")
    y_true = roc.get("y_true")
    y_prob = roc.get("y_prob")
    if y_true and y_prob:
        y_true = np.array(y_true, dtype=np.float64)
        y_prob = np.array(y_prob, dtype=np.float64)
        threshs = np.sort(np.unique(y_prob))[::-1]
        fpr, tpr = [0.0], [0.0]
        for t in threshs:
            yp = (y_prob >= t).astype(np.float64)
            tp = (yp * y_true).sum()
            fp = (yp * (1 - y_true)).sum()
            fn = ((1 - yp) * y_true).sum()
            tn = ((1 - yp) * (1 - y_true)).sum()
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        tpr.append(1.0)
        fpr.append(1.0)
        if auc is None:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_prob)
        ax_r.plot(fpr, tpr, color="orange", linewidth=1.5,
                  label=f"ROC (AUC={float(auc):.3f})")
    else:
        ax_r.text(0.5, 0.5, "No ROC data", ha="center", va="center",
                  transform=ax_r.transAxes)
    ax_r.plot([0, 1], [0, 1], color="blue", linestyle="--", label="Chance")
    ax_r.set_title(f"Fold {fold_num} — ROC")
    ax_r.grid(True, alpha=0.3)
    if show_legend:
        ax_r.legend(fontsize=7, loc="lower right")


def _trim_cm(cm):
    cm = np.array(cm)
    mx = cm.max()
    return cm, mx if mx > 0 else 1


def _plot_cm(table, ax, title):
    cm, vmax = _trim_cm(table)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["HC", "MDD"])
    ax.set_yticklabels(["HC", "MDD"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > vmax * 0.5 else "black")
    return im


def _plot_cm_pair(axes, fold_entry: dict, fold_num: int,
                  show_title: bool = True):
    kw = fold_entry.get("test_cm_window")
    ks = fold_entry.get("test_cm_subject")
    if isinstance(axes, np.ndarray):
        ax_w, ax_s = axes
    else:
        ax_w = ax_s = axes
    if kw is not None:
        _ = _plot_cm(kw, ax_w, f"Fold {fold_num} — Windows" if show_title else "Windows")
    if ks is not None:
        _ = _plot_cm(ks, ax_s, f"Fold {fold_num} — Subjects" if show_title else "Subjects")


def main():
    parser = argparse.ArgumentParser(description="Plot MODMA training diagnostics")
    parser.add_argument("--type", default="unimodal",
                        choices=["unimodal", "multimodal"])
    parser.add_argument("--modal", default="eeg", choices=["eeg", "aud"])
    parser.add_argument("--tag", required=True, help="Experiment tag")
    parser.add_argument("--split-seed", type=int, default=2509)
    parser.add_argument("--fold", type=str, default=None,
                        help="Comma-separated fold numbers, e.g. '1,3' (0-based). "
                             "Omit or use --all for all folds.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--metric", default=None,
                        choices=["bacc", "acc", "f1", "sens", "spec"])
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--cm", action="store_true")
    parser.add_argument("--cm-overall", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    args = parser.parse_args()

    results_path = find_results_path(args.type, args.modal,
                                     args.split_seed, args.tag)
    results = load_results(results_path)
    folds_data = results["folds"]
    print(f"Loaded: {results_path}  ({len(folds_data)} folds, "
          f"model={results.get('model')})")

    fold_ids = parse_folds(args.fold)
    if fold_ids is None and not args.all:
        fold_ids = None  # == all

    out_dir = None
    if args.save_png:
        rel = os.path.relpath(os.path.dirname(results_path), RESULTS_ROOT)
        out_dir = os.path.join(FIGURES_ROOT, rel)
        os.makedirs(out_dir, exist_ok=True)

    # ── Confusion matrix per fold ─────────────────────────────────────
    if args.cm:
        selected = select_folds(folds_data, fold_ids)
        has_window = any("test_cm_window" in f for f in selected)
        n_cols = 2 if has_window else 1
        k = len(selected)
        fig, axes = plt.subplots(
            k, n_cols, figsize=(7 if has_window else 3.5, 3.4 * k),
            constrained_layout=True)
        if k == 1:
            axes = np.array([axes])
        for i, fe in enumerate(selected):
            _plot_cm_pair(axes[i] if n_cols == 2 else axes, fe,
                          fe["fold"], show_title=k > 1)
        name = "all_folds_cm" if fold_ids is None else "cm"
        if fold_ids is not None:
            fig.suptitle(f"Folds {fold_ids} — Confusion matrices")
        _finish(fig, out_dir, name, args.save_png)
        return

    # ── Overall confusion matrix ─────────────────────────────────────
    if args.cm_overall:
        has_window = any("test_cm_window" in f for f in folds_data)
        if has_window:
            cm_w = np.sum([np.array(f["test_cm_window"]) for f in folds_data], axis=0)
            cm_s = np.sum([np.array(f["test_cm_subject"]) for f in folds_data], axis=0)
            fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), constrained_layout=True)
            _ = _plot_cm(cm_w, axes[0], "Overall — Windows")
            _ = _plot_cm(cm_s, axes[1], "Overall — Subjects")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.4), constrained_layout=True)
            _ = _plot_cm(np.sum([np.array(f["test_cm"]) for f in folds_data], axis=0),
                         ax, "Overall")
        _finish(fig, out_dir, "overall_cm", args.save_png)
        return

    # ── ROC per fold ─────────────────────────────────────────────────
    if args.roc:
        selected = select_folds(folds_data, fold_ids)
        k = len(selected)
        fig, axes = plt.subplots(k, 2, figsize=(10, 3.4 * k),
                                 constrained_layout=True)
        if k == 1:
            axes = np.array([axes])
        for i, fe in enumerate(selected):
            _plot_roc(axes[i], fe, fe["fold"], show_legend=i == 0)
        name = "all_folds_roc" if fold_ids is None else "roc"
        if fold_ids is not None:
            fig.suptitle(f"Folds {fold_ids} — ROC")
        _finish(fig, out_dir, name, args.save_png)
        return

    # ── Training curves per fold ─────────────────────────────────────
    metric = args.metric or "bacc"
    selected = select_folds(folds_data, fold_ids)
    k = len(selected)
    fig, axes = plt.subplots(k, 2, figsize=(10, 3.2 * k), constrained_layout=True)
    if k == 1:
        axes = np.array([axes])
    for i, fe in enumerate(selected):
        hist = fe.get("history") or {}
        if not hist.get("train_loss"):
            axes[i][0].text(0.5, 0.5, "No curves", ha="center", va="center",
                            transform=axes[i][0].transAxes)
            axes[i][1].text(0.5, 0.5, "No curves", ha="center", va="center",
                            transform=axes[i][1].transAxes)
            continue
        _plot_metric_curves(axes[i], hist, fe["fold"], metric,
                            show_legend=i == 0)
    name = f"all_folds_{metric}" if fold_ids is None else metric
    if fold_ids is not None:
        fig.suptitle(f"Folds {fold_ids} — {metric.upper()}")
    _finish(fig, out_dir, name, args.save_png)


def _finish(fig, out_dir, name, save_png):
    if save_png:
        fname = os.path.join(out_dir, f"{name}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()