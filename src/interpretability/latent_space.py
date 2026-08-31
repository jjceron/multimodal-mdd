"""Latent-space interpretability: EEG-only, AUD-only and fused multimodal."""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PR = pathlib.Path(__file__).resolve().parents[2]

from src.models.crossattnfusion import CrossAttnFusion  # noqa: E402
from src.models.deepconvnet import DeepConvNet  # noqa: E402
from src.models.shallowconvnet import ShallowConvNet  # noqa: E402
from src.preprocessing.modma_dataset import MODMASubjects  # noqa: E402
from src.training.modma_multimodal import _features, load_dataset  # noqa: E402


def _ctx(seed, fold, tag):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(
        PR / "outputs/results/multimodals" / f"tag{tag}_sseed{seed}_sngkf_multimodal"
        / f"fold_{fold}.pt", map_location=dev)
    eg, au = load_dataset(argparse.Namespace(
        channels="10-20", overlap=0.5, lowcut=0.5, highcut=50.0, notch=50.0,
        target_fs=256.0, reference="average", window_sec=2.0))
    sj = MODMASubjects(eg, au)
    k = min(min(s["eeg"].shape[0] for s in sj.paired.values()),
            min(s["aud"].shape[0] for s in sj.paired.values()))
    for s in sj.paired.values():
        s["eeg"], s["aud"] = s["eeg"][:k], s["aud"][:k]
    en = len(eg.channel_names)
    es = int(eg.samples[0]["eeg"].shape[-1])
    an = int(au.samples[0]["logmel"].shape[1])
    asx = int(au.samples[0]["logmel"].shape[-1])
    de = DeepConvNet(en, 1, es, 0.5).fc_features
    da = ShallowConvNet(an, 1, asx, 0.5).classifier.in_features
    B = DeepConvNet(en, 1, es, 0.6).to(dev)
    S = ShallowConvNet(an, 1, asx, 0.6).to(dev)
    F = CrossAttnFusion(de, da, 32, 2, n_self_attn_layers=1,
                        dropout=0.6, attn_dropout=0.1).to(dev)
    B.load_state_dict(ck["beeg"])
    S.load_state_dict(ck["baud"])
    F.load_state_dict(ck["fusion"])
    for m in (B, S, F):
        m.eval()
    return dev, sj, B, S, F


def _mod_feats(B, S, sj, dev):
    E, A, y = [], [], []
    for pid in sorted(sj.paired):
        e = torch.stack([sj.paired[pid]["eeg"]]).to(dev)
        a = torch.stack([sj.paired[pid]["aud"]]).to(dev)
        with torch.no_grad():
            E.append(_features(B, e, dev).mean(1).squeeze(0).cpu().numpy())
            A.append(_features(S, a, dev).mean(1).squeeze(0).cpu().numpy())
            y.append(sj.paired[pid]["label"])
    return np.array(E), np.array(A), np.array(y)


def _fused_feats(B, S, F, sj, dev):
    Z, y = [], []
    for pid in sorted(sj.paired):
        e = torch.stack([sj.paired[pid]["eeg"]]).to(dev)
        a = torch.stack([sj.paired[pid]["aud"]]).to(dev)
        with torch.no_grad():
            F(_features(B, e, dev), _features(S, a, dev))
            Z.append(F._pooled.detach().cpu().numpy().squeeze(0))
            y.append(sj.paired[pid]["label"])
    return np.array(Z), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1205)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--tag", type=str, default="v2")
    p.add_argument("--mode", choices=["modalities", "fusion"], default="modalities")
    args = p.parse_args()

    from sklearn.decomposition import PCA
    dev, sj, B, S, F = _ctx(args.seed, args.fold, args.tag)
    out = PR / "outputs/figures/interpretability"
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "modalities":
        E, A, y = _mod_feats(B, S, sj, dev)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for i, (X, lab) in enumerate([(E, "EEG"), (A, "AUD")]):
            z = PCA(2).fit_transform(X)
            for cl, c, m in ((0, "tab:blue", "o"), (1, "tab:red", "x")):
                s = y == cl
                ax[i].scatter(z[s, 0], z[s, 1], c=c, marker=m, s=22,
                              label="HC" if cl == 0 else "MDD")
            ax[i].set_xlabel("PC1")
            ax[i].set_ylabel(lab + " PC2")
            ax[i].legend()
            ax[i].grid(alpha=0.3)
        fig.savefig(out / "latent_modalities.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved:", out / "latent_modalities.png")
    else:
        Z, y = _fused_feats(B, S, F, sj, dev)
        z = PCA(2).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
        for cl, c, m in ((0, "tab:blue", "o"), (1, "tab:red", "x")):
            s = y == cl
            ax.scatter(z[s, 0], z[s, 1], c=c, marker=m, s=22,
                       label="HC" if cl == 0 else "MDD")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(out / "latent_fusion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved:", out / "latent_fusion.png")


if __name__ == "__main__":
    main()
