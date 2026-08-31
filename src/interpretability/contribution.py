"""EEG vs AUD contribution to the fusion logit, per fold (ablation)."""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PR = pathlib.Path(__file__).resolve().parents[2]

from src.interpretability.latent_space import _ctx  # noqa: E402
from src.training.modma_multimodal import _features  # noqa: E402


def _ablate(B, S, F, sj, dev):
    fe, fa = {}, {}
    for pid in sorted(sj.paired):
        e = torch.stack([sj.paired[pid]["eeg"]]).to(dev)
        a = torch.stack([sj.paired[pid]["aud"]]).to(dev)
        with torch.no_grad():
            fe[pid] = _features(B, e, dev)
            fa[pid] = _features(S, a, dev)
    me = torch.stack(list(fe.values())).mean(0)
    ma = torch.stack(list(fa.values())).mean(0)
    ce, ca = [], []
    for pid in fe:
        with torch.no_grad():
            full = F(fe[pid], fa[pid])
            eb = F(fe[pid], ma.expand_as(fa[pid]))
            ab = F(me.expand_as(fe[pid]), fa[pid])
            ce.append(abs(full - ab).item())
            ca.append(abs(full - eb).item())
    return float(np.mean(ce)), float(np.mean(ca))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1205)
    p.add_argument("--tag", type=str, default="v2")
    args = p.parse_args()

    folds, means_e, means_a = [], [], []
    for fold in range(1, 6):
        dev, sj, B, S, F = _ctx(args.seed, fold, args.tag)
        ce, ca = _ablate(B, S, F, sj, dev)
        folds.append(fold)
        means_e.append(ce)
        means_a.append(ca)

    out = PR / "outputs/figures/interpretability"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.arange(len(folds))
    w = 0.35
    ax.bar(x - w / 2, means_e, w, label="EEG", color="tab:blue")
    ax.bar(x + w / 2, means_a, w, label="AUD", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("|logit| contribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out / "modality_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out / "modality_contribution.png")


if __name__ == "__main__":
    main()
