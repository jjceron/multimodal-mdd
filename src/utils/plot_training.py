"""Plot training diagnostics for MODMA results.

Reads results.json produced by src/training (unimodals or multimodal) and
renders training curves (loss / acc per epoch), best-fold selection, ROC and
confusion matrices.

Modes:
  --curve (default)                : 1x2 (loss | <metric>) train/val curves
  --roc                            : per-fold ROC (+ AUC)
  --cm                             : per-fold confusion matrices (subjects)
  --best-fold                      : select the fold with best bacc, plot it alone
  --all                            : plot every fold

Selecting the experiment:
  --type unimodal|multimodal  --modal eeg|aud  --tag <tag>  [--split-seed]  [--metric]

Examples:
  poetry run python -m src.utils.plot_training --type multimodal --modal eeg \
      --tag v2 --split-seed 42 --curve --best-fold
  poetry run python -m src.utils.plot_training --type unimodal --modal eeg \
      --tag deepconvnet --curve --best-fold
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
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "results")
FIGURES_ROOT = os.path.join(PROJECT_ROOT, "outputs", "figures")

CURVE_KEYS = {
    "loss": ("train loss", "val loss"),
    "acc": ("train acc", "val acc"),
    "bacc": ("train bacc", "val bacc"),
    "f1": ("train f1", "val f1"),
}


def find_results_path(type_: str, modal: str, split_seed: int, tag: str) -> str:
    if type_ == "multimodal":
        base = os.path.join(RESULTS_ROOT, "multimodals")
    else:
        base = os.path.join(RESULTS_ROOT, "unimodals", modal)
    pool = sorted(glob.glob(os.path.join(base, "**", "results.json"), recursive=True))
    matches = [
        p for p in pool
        if f"sseed{split_seed}" in p and f"tag{tag}" in p
    ]
    if not matches:
        print(f"ERROR: no results found for type={type_} modal={modal} "
              f"seed={split_seed} tag={tag}")
        sys.exit(1)
    return matches[-1]


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize_folds(results: dict) -> list[dict]:
    folds = results.get("folds", {})
    curves = results.get("training curves") or {}
    out = []
    if isinstance(folds, dict):
        for name, fe in folds.items():
            num = int(str(name).split()[-1])
            fe = dict(fe)
            fe["fold"] = num
            fe["history"] = curves.get(name) or {}
            out.append(fe)
    else:
        for fe in folds:
            fe = dict(fe)
            fe["history"] = curves.get(f"fold {fe.get('fold')}") or {}
            out.append(fe)
    out.sort(key=lambda f: f["fold"])
    return out


def best_fold(folds: list[dict]) -> dict:
    return max(folds, key=lambda f: f.get("test_metrics", {}).get("bacc", -1.0))


def _plot_curve_1x2(ax, hist: dict, metric: str):
    trl, vll = CURVE_KEYS["loss"]
    tra, val = CURVE_KEYS[metric]
    n = len(hist.get(trl, []))
    epochs = np.arange(1, n + 1) if n else np.arange(1, 2)
    if hist.get(trl):
        ax[0].plot(epochs, hist[trl], color="tab:blue", label="Train")
        if hist.get(vll):
            ax[0].plot(epochs, hist[vll], color="tab:red", ls="--",
                       label="Validation")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    if hist.get(tra):
        ax[1].plot(epochs, hist[tra], color="tab:blue", label="Train")
        if hist.get(val):
            ax[1].plot(epochs, hist[val], color="tab:red", ls="--",
                       label="Validation")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(metric.upper())
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)


def plot_curves(folds: list[dict], metric: str, out_dir: str | None,
                save_png: bool, best_only: bool, tag: str):
    if best_only:
        folds = [best_fold(folds)]
    k = len(folds)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if k == 1:
        hist = folds[0].get("history") or {}
        if hist.get("train loss"):
            _plot_curve_1x2(axes, hist, metric)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No curves", ha="center", va="center",
                        transform=ax.transAxes)
    else:
        for i, fe in enumerate(folds):
            hist = fe.get("history") or {}
            if i == 0:
                _plot_curve_1x2(axes, hist, metric)
    name = f"tag{tag}_best_fold_{metric}" if best_only else f"tag{tag}_all_folds_{metric}"
    _finish(fig, out_dir, name, save_png)


def plot_roc(folds: list[dict], out_dir: str | None, save_png: bool):
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
    for fe in folds:
        y, p = fe.get("test_roc", {}).get("y_true"), fe.get("test_roc", {}).get("y_prob")
        if y and p:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
            auc = roc_auc_score(np.asarray(y), np.asarray(p))
            ax.plot(fpr, tpr, label=f"Fold {fe.get('fold')} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    _finish(fig, out_dir, "roc_curves", save_png)


def plot_cm(folds: list[dict], out_dir: str | None, save_png: bool):
    n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for i, fe in enumerate(folds):
        cm = np.zeros((2, 2))
        if "test_cm" in fe and fe["test_cm"] is not None:
            cm = np.array(fe["test_cm"])
        elif "test_cm_subject" in fe and fe["test_cm_subject"] is not None:
            cm = np.array(fe["test_cm_subject"])
        axes[i].imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["HC", "MDD"])
        axes[i].set_yticklabels(["HC", "MDD"])
        axes[i].set_xlabel(f"Fold {fe.get('fold')}")
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, str(int(cm[r, c])), ha="center", va="center",
                             fontsize=13,
                             color="white" if cm[r, c] > cm.max() * 0.5 else "black")
    _finish(fig, out_dir, "cm_folds", save_png)


def _finish(fig, out_dir, name, save_png):
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150,
                bbox_inches="tight") if save_png and out_dir else plt.show()
    if save_png and out_dir:
        print(f"Saved: {os.path.join(out_dir, name)}.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot MODMA training diagnostics")
    parser.add_argument("--type", default="unimodal", choices=["unimodal", "multimodal"])
    parser.add_argument("--modal", default="eeg", choices=["eeg", "aud"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--split-seed", type=int, default=2509)
    parser.add_argument("--metric", default="acc", choices=list(CURVE_KEYS))
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--best-fold", action="store_true")
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--cm", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    args = parser.parse_args()

    results_path = find_results_path(args.type, args.modal, args.split_seed, args.tag)
    results = load_results(results_path)
    folds = normalize_folds(results)
    print(f"Loaded: {results_path}  ({len(folds)} folds)")

    rel = os.path.relpath(os.path.dirname(results_path), RESULTS_ROOT)
    out_dir = os.path.join(FIGURES_ROOT, rel) if args.save_png else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.roc:
        plot_roc(folds, out_dir, args.save_png)
        return
    if args.cm:
        plot_cm(folds, out_dir, args.save_png)
        return
    plot_curves(folds, args.metric, out_dir, args.save_png,
                args.best_fold or args.fold is None, args.tag)


if __name__ == "__main__":
    main()
