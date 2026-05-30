from __future__ import annotations

from pathlib import Path

import csv
import matplotlib.pyplot as plt


OUT_DIR = Path("docs/report/figures/2026-05-31_external_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    out_path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved: {out_path}")


def plot_accuracy_f1_comparison() -> None:
    data = [
        {
            "method": "Baseline\nTH=0.50",
            "accuracy": 76.56,
            "f1": 74.85,
        },
        {
            "method": "Gain-aware\nTH",
            "accuracy": 91.00,
            "f1": 88.11,
        },
        {
            "method": "Gain-aware\n+ 3/2 vote",
            "accuracy": 95.33,
            "f1": 94.08,
        },
        {
            "method": "Gain-aware\n+ 5/3 vote",
            "accuracy": 97.33,
            "f1": 96.69,
        },
        {
            "method": "Gain-aware\n+ 5/2 vote",
            "accuracy": 99.17,
            "f1": 99.00,
        },
    ]

    x = range(len(data))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], [row["accuracy"] for row in data], width=width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], [row["f1"] for row in data], width=width, label="F1-score")

    plt.xticks(list(x), [row["method"] for row in data])
    plt.ylim(70, 102)
    plt.ylabel("Score (%)")
    plt.title("External Validation Performance Improvement")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    for i, row in enumerate(data):
        plt.text(i - width / 2, row["accuracy"] + 0.5, f"{row['accuracy']:.2f}", ha="center", fontsize=8)
        plt.text(i + width / 2, row["f1"] + 0.5, f"{row['f1']:.2f}", ha="center", fontsize=8)

    savefig("fig1_accuracy_f1_comparison.png")


def plot_fp_fn_comparison() -> None:
    data = [
        {
            "method": "Baseline\nTH=0.50",
            "fp": 300,
            "fn": 122,
        },
        {
            "method": "Gain-aware\nTH",
            "fp": 12,
            "fn": 150,
        },
        {
            "method": "Gain-aware\n+ 3/2 vote",
            "fp": 2,
            "fn": 82,
        },
        {
            "method": "Gain-aware\n+ 5/3 vote",
            "fp": 0,
            "fn": 48,
        },
        {
            "method": "Gain-aware\n+ 5/2 vote",
            "fp": 6,
            "fn": 9,
        },
    ]

    x = range(len(data))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], [row["fp"] for row in data], width=width, label="False Drone (FP)")
    plt.bar([i + width / 2 for i in x], [row["fn"] for row in data], width=width, label="Missed Drone (FN)")

    plt.xticks(list(x), [row["method"] for row in data])
    plt.ylabel("Number of samples")
    plt.title("False Positive and False Negative Reduction")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    for i, row in enumerate(data):
        plt.text(i - width / 2, row["fp"] + 5, f"{row['fp']}", ha="center", fontsize=8)
        plt.text(i + width / 2, row["fn"] + 5, f"{row['fn']}", ha="center", fontsize=8)

    savefig("fig2_fp_fn_comparison.png")


def load_g30_threshold_sweep() -> list[dict[str, float]]:
    csv_path = Path("outputs/ml/external_eval_20260528_slices/gain_g30_threshold_sweep.csv")

    if csv_path.exists():
        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "threshold": float(row["threshold"]),
                    "fp": int(float(row["fp"])),
                    "fn": int(float(row["fn"])),
                    "f1": float(row["f1"]),
                })
        return rows

    return [
        {"threshold": 0.50, "fp": 300, "fn": 91, "f1": 0.7013},
        {"threshold": 0.60, "fp": 94, "fn": 101, "f1": 0.8216},
        {"threshold": 0.70, "fp": 41, "fn": 112, "f1": 0.8513},
        {"threshold": 0.80, "fp": 11, "fn": 120, "f1": 0.8678},
        {"threshold": 0.85, "fp": 5, "fn": 123, "f1": 0.8697},
        {"threshold": 0.90, "fp": 2, "fn": 125, "f1": 0.8700},
        {"threshold": 0.95, "fp": 0, "fn": 134, "f1": 0.8613},
    ]


def plot_g30_threshold_fp_fn() -> None:
    rows = [
        r for r in load_g30_threshold_sweep()
        if 0.50 <= r["threshold"] <= 0.95
    ]

    thresholds = [r["threshold"] for r in rows]
    fps = [r["fp"] for r in rows]
    fns = [r["fn"] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, fps, marker="o", label="False Drone (FP)")
    plt.plot(thresholds, fns, marker="o", label="Missed Drone (FN)")

    plt.xlabel("Drone decision threshold")
    plt.ylabel("Number of samples")
    plt.title("g30 Threshold Sweep: FP/FN Trade-off")
    plt.grid(True, alpha=0.3)
    plt.legend()

    savefig("fig3_g30_threshold_fp_fn_tradeoff.png")


def plot_g30_threshold_f1() -> None:
    rows = [
        r for r in load_g30_threshold_sweep()
        if 0.50 <= r["threshold"] <= 0.95
    ]

    thresholds = [r["threshold"] for r in rows]
    f1_percent = [r["f1"] * 100 for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, f1_percent, marker="o", label="F1-score")

    best_i = max(range(len(rows)), key=lambda i: f1_percent[i])
    best_th = thresholds[best_i]
    best_f1 = f1_percent[best_i]

    plt.scatter([best_th], [best_f1], s=80)
    plt.text(
        best_th,
        best_f1 + 1.0,
        f"Best: TH={best_th:.2f}, F1={best_f1:.2f}%",
        ha="center",
    )

    plt.ylim(min(f1_percent) - 1.0, max(f1_percent) + 3.0)
    plt.xlabel("Drone decision threshold")
    plt.ylabel("F1-score (%)")
    plt.title("g30 Threshold Sweep: F1-score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    savefig("fig4_g30_threshold_f1.png")


def main() -> None:
    plot_accuracy_f1_comparison()
    plot_fp_fn_comparison()
    plot_g30_threshold_fp_fn()
    plot_g30_threshold_f1()

    print()
    print(f"[DONE] figures saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
