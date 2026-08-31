from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_results(results_dir):
    pattern = str(
        results_dir / "tag*_sseed*_sngkf_multimodal" / "results.json"
    )

    results = {}

    for path in sorted(glob.glob(pattern)):
        path = Path(path)
        match = re.search(
            r"_sseed(\d+)_sngkf_multimodal$",
            path.parent.name,
        )

        if match is None:
            continue

        seed = int(match.group(1))

        with open(path, "r", encoding="utf-8") as fh:
            results[seed] = json.load(fh)

    if not results:
        raise RuntimeError(
            f"No multimodal results found in {results_dir}"
        )

    return results


def plot_results(results, output_path):
    metrics = {
        "Accuracy": "acc",
        "BACC": "bacc",
        "AUC": "auc",
        "F1": "f1",
    }

    seeds = sorted(results)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    ax = axes[0]

    distribution = []

    for key in metrics.values():
        values = []

        for seed in seeds:
            test = results[seed]["test"]
            value = test.get(f"{key}_mean")

            if value is not None:
                values.append(float(value))

        distribution.append(values)

    ax.boxplot(
        distribution,
        tick_labels=list(metrics.keys()),
        patch_artist=True,
        showmeans=False,
    )

    for index, values in enumerate(distribution, start=1):
        x = np.full(len(values), index, dtype=float)

        if len(values) > 1:
            x += np.linspace(-0.06, 0.06, len(values))

        ax.scatter(
            x,
            values,
            s=35,
            zorder=3,
        )

    ax.set_title("Distribution of Metrics Across Seeds")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]

    accuracy = []
    accuracy_std = []
    auc = []
    auc_std = []

    for seed in seeds:
        test = results[seed]["test"]

        accuracy.append(test["acc_mean"])
        accuracy_std.append(test["acc_std"])
        auc.append(test["auc_mean"])
        auc_std.append(test["auc_std"])

    x = np.arange(len(seeds))
    width = 0.32

    ax.bar(
        x - width / 2,
        accuracy,
        width,
        yerr=accuracy_std,
        capsize=5,
        label="Accuracy",
    )

    ax.bar(
        x + width / 2,
        auc,
        width,
        yerr=auc_std,
        capsize=5,
        label="AUC-ROC",
    )

    ax.set_title("Model Performance Across Different Seeds")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot multimodal performance across seeds"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(
            PROJECT_ROOT / "outputs/results/multimodals"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs/figures"
            / "multiseed_distributions.png"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results = load_results(
        Path(args.results_dir)
    )

    plot_results(
        results,
        Path(args.output),
    )


if __name__ == "__main__":
    main()