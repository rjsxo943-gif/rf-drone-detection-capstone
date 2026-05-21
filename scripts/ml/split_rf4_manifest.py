from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows_by_label(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)

    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for label in sorted(by_label):
        label_rows = by_label[label]
        rng.shuffle(label_rows)

        n = len(label_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_part = label_rows[:n_train]
        val_part = label_rows[n_train:n_train + n_val]
        test_part = label_rows[n_train + n_val:]

        for row in train_part:
            row["split"] = "train"
        for row in val_part:
            row["split"] = "val"
        for row in test_part:
            row["split"] = "test"

        train_rows.extend(train_part)
        val_rows.extend(val_part)
        test_rows.extend(test_part)

    train_rows.sort(key=lambda x: (x["label"], x["filepath"]))
    val_rows.sort(key=lambda x: (x["label"], x["filepath"]))
    test_rows.sort(key=lambda x: (x["label"], x["filepath"]))

    return train_rows, val_rows, test_rows


def print_summary(name: str, rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = defaultdict(int)

    for row in rows:
        counts[row["label"]] += 1

    print()
    print(f"=== {name} ===")
    print(f"total: {len(rows)}")
    for label in sorted(counts):
        print(f"{label}: {counts[label]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/cnn_capture/manifests/manifest_rf4_balanced_v1.csv",
        help="Input balanced manifest csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/cnn_capture/splits/rf4_random_v1",
        help="Output split directory",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = read_csv(manifest_path)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train-ratio and val-ratio must be positive")

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be less than 1.0")

    train_rows, val_rows, test_rows = split_rows_by_label(
        rows=rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    fieldnames = list(rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")

    all_rows = train_rows + val_rows + test_rows
    all_rows.sort(key=lambda x: (x["split"], x["label"], x["filepath"]))

    write_csv(out_dir / "train.csv", train_rows, fieldnames)
    write_csv(out_dir / "val.csv", val_rows, fieldnames)
    write_csv(out_dir / "test.csv", test_rows, fieldnames)
    write_csv(out_dir / "split_manifest.csv", all_rows, fieldnames)

    print_summary("train", train_rows)
    print_summary("val", val_rows)
    print_summary("test", test_rows)

    print()
    print(f"[OK] saved: {out_dir / 'train.csv'}")
    print(f"[OK] saved: {out_dir / 'val.csv'}")
    print(f"[OK] saved: {out_dir / 'test.csv'}")
    print(f"[OK] saved: {out_dir / 'split_manifest.csv'}")


if __name__ == "__main__":
    main()
