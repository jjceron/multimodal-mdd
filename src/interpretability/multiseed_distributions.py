"""Multi-seed distribution of multimodal bacc (boxplot, hist, means)."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PR = pathlib.Path(__file__).resolve().parents[2]
BASE = PR / "outputs/results/multimodals"


def _collect(tag, seeds):
    per_seed, all_b = {}, []
    for s in seeds:
        d = BASE / f"tag{tag}_sseed{s}_sngkf_multimodal"
        rj = d / "results.json"
        if not rj.exists():
            continue
        res = json.load(open(rj))
        bs = [f["test_metrics"]["bacc"] for f in res["folds"].values()]
        per_seed[s] = bs
        all_b += bs
    return per_seed, all_b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="v2")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1205, 2509, 3001])
    args = p.parse_args()

    per_seed, all_b = _collect(args.tag, args.seeds)
    seeds = list(per_seed.keys())
    means = [float(np.mean(per_seed[s])) for s in seeds]
    stds = [float(np.std(per_seed[s])) for s in seeds]

    out = PR / "outputs/figures/interpretability"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)

    ax[0].boxplot(all_b)
    ax[0].set_ylabel("bacc (per fold)")
    ax[0].set_xticks([1])
    ax[0].set_xticklabels(["All folds"])
    ax[0].grid(axis="y", alpha=0.3)

    ax[1].hist(all_b, bins=8, color="tab:gray", alpha=0.7, edgecolor="k")
    ax[1].set_xlabel("bacc (per fold)")
    ax[1].set_ylabel("count")
    ax[1].grid(alpha=0.3)

    x = np.arange(len(seeds))
    ax[2].bar(x, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.8)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"S{s}" for s in seeds])
    ax[2].set_ylabel("mean bacc")
    ax[2].axhline(float(np.mean(all_b)), color="tab:red", ls="--", lw=1,
                  label=f"overall {np.mean(all_b):.3f}")
    ax[2].legend()
    ax[2].grid(axis="y", alpha=0.3)

    fig.savefig(out / "multiseed_bacc_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out / "multiseed_bacc_distribution.png")


if __name__ == "__main__":
    main()
