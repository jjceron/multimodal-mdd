"""Plot MODMA training curves (loss/acc) and diagnostics."""
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
    base = os.path.join(RESULTS_ROOT, "multimodals") if type_ == "multimodal" \
        else os.path.join(RESULTS_ROOT, "unimodals", modal)
    pool = sorted(glob.glob(os.path.join(base, "**", "results.json"), recursive=True))
    matches = [p for p in pool if f"sseed{split_seed}" in p and f"tag{tag}" in p]
    if not matches:
        print(f"ERROR: no results for type={type_} modal={modal} "
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
            fe = dict(fe)
            fe["fold"] = int(str(name).split()[-1])
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


def plot_curves(folds: list[dict], metric: str, out_dir, save_png, best_only, tag):
    if best_only:
        folds = [best_fold(folds)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    hist = (folds[0].get("history") or {}) if folds else {}
    if hist.get("train loss"):
        _plot_curve_1x2(axes, hist, metric)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No curves", ha="center", va="center",
                    transform=ax.transAxes)
    name = f"tag{tag}_best_fold_{metric}" if best_only else f"tag{tag}_all_folds_{metric}"
    _finish(fig, out_dir, name, save_png)


def _finish(fig, out_dir, name, save_png):
    if save_png and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150,
                    bbox_inches="tight")
        print(f"Saved: {os.path.join(out_dir, name)}.png")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="unimodal", choices=["unimodal", "multimodal"])
    parser.add_argument("--modal", default="eeg", choices=["eeg", "aud"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--split-seed", type=int, default=2509)
    parser.add_argument("--metric", default="acc", choices=list(CURVE_KEYS))
    parser.add_argument("--best-fold", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    args = parser.parse_args()

    results_path = find_results_path(args.type, args.modal, args.split_seed, args.tag)
    folds = normalize_folds(load_results(results_path))
    print(f"Loaded: {results_path}  ({len(folds)} folds)")

    rel = os.path.relpath(os.path.dirname(results_path), RESULTS_ROOT)
    out_dir = os.path.join(FIGURES_ROOT, rel) if args.save_png else None
    plot_curves(folds, args.metric, out_dir, args.save_png, args.best_fold, args.tag)


if __name__ == "__main__":
    main()
