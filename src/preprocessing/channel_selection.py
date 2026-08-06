import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EEG_DIR = PROJECT_ROOT / "data/raw/modma/eeg/EEG_LZU_2015_2_resting state"

SUPPORTED_PRESETS = ("all", "10-20", "f64", "29")

CLINICAL_10_20_EGI_MAP = {
    "Fp1": 45, "Fp2": 33, "F7": 69, "F3": 59, "Fz": 30, "F4": 19, "F8": 21,
    "T3": 74, "C3": 71, "Cz": 6, "C4": 15, "T4": 14, "T5": 82,
    "P3": 84, "Pz": 79, "P4": 117, "T6": 8, "O1": 101, "O2": 115,
}
CLINICAL_10_20_NAMES = tuple(CLINICAL_10_20_EGI_MAP)
CLINICAL_10_20_EGI = tuple(sorted(CLINICAL_10_20_EGI_MAP.values()))

# 29-channel fronto-centro-parietal set (10-20 + 10-10 extended), replicating
# the channel count used by Yousufi et al. (Brain Sci 2024, brainsci-14-01018),
# who select 29 frontal/temporal/parietal channels per Hussain et al. The paper
# never names the 29 channels, so we fix a principled classic 10-20/10-10 set:
# the 19 standard 10-20 channels plus FC1 FC2 FC5 FC6 C5 C6 CP1 CP2 CP5 CP6.
# Each extra 10-10 position was placed by spherical interpolation between its
# verified 10-20 anchors and mapped to the nearest *unused* EGI-128 sensor
# (mean placement error ~1.7 cm, on par with the standard 10-20 assignments).
EXTENDED_29_EGI_MAP = {
    **CLINICAL_10_20_EGI_MAP,
    "FC1": 53, "FC2": 12, "FC5": 65, "FC6": 22,
    "C5": 70, "C6": 17,
    "CP1": 78, "CP2": 111, "CP5": 75, "CP6": 9,
}
EXTENDED_29_NAMES = tuple(EXTENDED_29_EGI_MAP)
EXTENDED_29_EGI = tuple(sorted(EXTENDED_29_EGI_MAP.values()))

_10_20_TARGETS = {
    "Fp1": (-2.7, 6.2, 1.8), "Fp2": (2.7, 6.2, 1.8),
    "F3": (-5.0, 4.0, 5.3), "F4": (5.0, 4.0, 5.3),
    "F7": (-7.7, 3.3, 0.4), "F8": (7.7, 3.3, 0.4),
    "Fz": (0.0, 2.6, 7.9), "C3": (-7.7, 0.4, 3.5),
    "C4": (7.7, 0.4, 3.5), "Cz": (0.0, 0.0, 8.8),
    "P3": (-5.0, -4.0, 5.3), "P4": (5.0, -4.0, 5.3),
    "Pz": (0.0, -2.6, 7.9), "O1": (-2.7, -6.2, 1.8),
    "O2": (2.7, -6.2, 1.8), "T3": (-8.6, 0.0, 0.4),
    "T4": (8.6, 0.0, 0.4), "T5": (-7.7, -3.3, 0.4),
    "T6": (7.7, -3.3, 0.4),
}


def select_channel_indices(preset: str) -> list[int]:
    if preset == "all":
        return list(range(128))
    if preset == "f64":
        return list(range(64))
    if preset == "10-20":
        return list(CLINICAL_10_20_EGI)
    if preset == "29":
        return list(EXTENDED_29_EGI)
    raise ValueError(f"Unknown preset: {preset!r}. Use one of {SUPPORTED_PRESETS}")


def select_channel_names(preset: str) -> list[str]:
    return [f"E{i + 1}" for i in select_channel_indices(preset)]


def _find_electrodes_path(root: Path | None = None) -> Path:
    root = Path(root) if root else EEG_DIR
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")):
        path = sub / "eeg" / f"{sub.name}_task-Resting-state_electrodes.tsv"
        if path.exists():
            return path
    raise FileNotFoundError(f"No electrodes.tsv found under {root}")


def _read_electrode_coords(path: Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].strip("'").startswith("E"):
                rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(rows, dtype=np.float32)


def verify_10_20(electrodes_path: Path) -> list[dict]:
    coords = _read_electrode_coords(electrodes_path)
    rows = []
    for name, idx in CLINICAL_10_20_EGI_MAP.items():
        t = np.asarray(_10_20_TARGETS[name], dtype=np.float32)
        dists = np.linalg.norm(coords - t, axis=1)
        nearest = int(np.argmin(dists))
        rows.append(
            {
                "name": name,
                "egi": idx,
                "nearest": nearest,
                "dist_egi": float(dists[idx]),
                "dist_nearest": float(dists[nearest]),
                "ok": idx == nearest,
            }
        )
    return rows


def _print_verification(electrodes_path: Path) -> None:
    rows = verify_10_20(electrodes_path)
    n_ok = sum(r["ok"] for r in rows)
    mean_d = np.mean([r["dist_egi"] for r in rows])
    print(f"\n  Verification against dataset: {electrodes_path}")
    print(f"  {'10-20':>5s} | {'EGI':>6s} | {'dataset nearest':>15s} | {'dist to EGI':>11s} | OK")
    print("  " + "-" * 56)
    for r in rows:
        print(
            f"  {r['name']:>5s} | E{r['egi'] + 1:>5d} | E{r['nearest'] + 1:>14d} | "
            f"{r['dist_egi']:>10.2f} cm | {'yes' if r['ok'] else 'NO'}"
        )
    print("  " + "-" * 56)
    print(f"  {n_ok}/{len(rows)} channels match the dataset nearest-neighbor "
          f"| mean dist to assigned EGI: {mean_d:.2f} cm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        choices=SUPPORTED_PRESETS,
        default="all",
        help="Channel configuration to select",
    )
    args = parser.parse_args()

    names = select_channel_names(args.channels)
    print(f"Channel selection: {args.channels}")
    print(f"  Count: {len(names)}")
    print(f"  Channels: {', '.join(names)}")

    if args.channels == "10-20":
        print("\n  Clinical 10-20 mapping (name -> EGI channel):")
        for name in CLINICAL_10_20_NAMES:
            print(f"    {name:>4s} -> E{CLINICAL_10_20_EGI_MAP[name] + 1}")
        try:
            _print_verification(_find_electrodes_path())
        except FileNotFoundError as e:
            print(f"\n  WARNING: could not verify against dataset: {e}")

    if args.channels == "29":
        print("\n  Extended 29-channel mapping (name -> EGI channel):")
        for name in EXTENDED_29_NAMES:
            print(f"    {name:>4s} -> E{EXTENDED_29_EGI_MAP[name] + 1}")


if __name__ == "__main__":
    main()
