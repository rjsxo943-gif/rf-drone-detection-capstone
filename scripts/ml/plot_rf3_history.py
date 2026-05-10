from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"history file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or len(history) == 0:
        raise ValueError(f"history is empty or invalid: {path}")

    return history


def plot_loss(history: list[dict], out_path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_accuracy(history: list[dict], out_path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="Path to history.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="RF3 CNN",
        help="Plot title prefix",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)

    loss_path = out_dir / "loss_curve.png"
    acc_path = out_dir / "accuracy_curve.png"

    plot_loss(history, loss_path, args.title)
    plot_accuracy(history, acc_path, args.title)

    print(f"[OK] saved: {loss_path}")
    print(f"[OK] saved: {acc_path}")


if __name__ == "__main__":
    main()

