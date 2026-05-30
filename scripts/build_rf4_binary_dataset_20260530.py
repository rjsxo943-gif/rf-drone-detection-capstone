#!/usr/bin/env python3
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path


SEED = 42

SRC_ROOT = Path("outputs/datasets/rf4_fixed_capture")
OUT_ROOT = Path("data/processed/rf4_binary_20260530_no_controller")

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def get_binary_label(folder_name: str) -> str | None:
    name = folder_name.lower()

    # Use only 20260530 data
    if not name.startswith("20260530_"):
        return None

    # Exclude controller-only data from training dataset
    if "controller" in name and not name.startswith("20260530_drone"):
        return None

    if "drone" in name:
        return "Drone"

    if (
        "background" in name
        or "wifi" in name
        or "bluetooth" in name
        or "mixed" in name
    ):
        return "NonDrone"

    return None


def get_source_type(folder_name: str) -> str:
    name = folder_name.lower()

    if "drone_motor_on_stick_taps" in name:
        return "drone_stick_taps"
    if "drone_linked_motor_on" in name:
        return "drone_motor_on"
    if "drone_linked_motor_off" in name:
        return "drone_motor_off"

    if "background" in name:
        return "background"
    if "mixed" in name and "wifi" in name and "bluetooth" in name:
        return "mixed_wifi_bluetooth"
    if "wifi" in name:
        return "wifi"
    if "bluetooth" in name:
        return "bluetooth"

    return "unknown"


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    files = list(files)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def main() -> None:
    random.seed(SEED)

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SRC_ROOT}")

    if OUT_ROOT.exists():
        raise FileExistsError(
            f"Output dataset already exists: {OUT_ROOT}\n"
            f"Delete it first if you want to rebuild."
        )

    rows: list[dict[str, str]] = []
    counts: dict[tuple[str, str, str], int] = {}

    folders = sorted(SRC_ROOT.glob("20260530_*"))

    for folder in folders:
        if not folder.is_dir():
            continue

        label = get_binary_label(folder.name)
        if label is None:
            print(f"[SKIP] {folder.name}")
            continue

        files = sorted(folder.glob("*.npy"))
        if not files:
            print(f"[SKIP EMPTY] {folder.name}")
            continue

        source_type = get_source_type(folder.name)
        split_map = split_files(files)

        for split, split_items in split_map.items():
            for src_file in split_items:
                dst_dir = OUT_ROOT / split / label
                dst_dir.mkdir(parents=True, exist_ok=True)

                dst_name = f"{folder.name}__{src_file.name}"
                dst_file = dst_dir / dst_name

                shutil.copy2(src_file, dst_file)

                row = {
                    "split": split,
                    "binary_label": label,
                    "source_type": source_type,
                    "source_folder": folder.name,
                    "src_file": str(src_file),
                    "dst_file": str(dst_file),
                }
                rows.append(row)

                key = (split, label, source_type)
                counts[key] = counts.get(key, 0) + 1

    metadata_path = OUT_ROOT / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "binary_label",
                "source_type",
                "source_folder",
                "src_file",
                "dst_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"[OK] dataset built: {OUT_ROOT}")
    print(f"[OK] metadata saved: {metadata_path}")
    print()

    print("=== Split / Label Counts ===")
    for split in ["train", "val", "test"]:
        for label in ["Drone", "NonDrone"]:
            n = sum(
                count
                for (s, l, _source), count in counts.items()
                if s == split and l == label
            )
            print(f"{split:5s} {label:8s}: {n}")

    print()
    print("=== Source Type Counts ===")
    for key in sorted(counts):
        split, label, source_type = key
        print(f"{split:5s} {label:8s} {source_type:24s}: {counts[key]}")


if __name__ == "__main__":
    main()
