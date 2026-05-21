from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ml import (
    RF3SmallCNN,
    RFSpectrogramDataset,
    build_confusion_matrix,
    compute_spectrogram_mean_std,
    make_classification_report_text,
    read_manifest_csv,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_text,
)


RF4_CLASS_NAMES = [
    "Background",
    "WiFi",
    "Bluetooth",
    "Drone-like",
]


def num_rf4_classes() -> int:
    return len(RF4_CLASS_NAMES)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    rows: list[dict[str, str]],
    project_root: Path,
    mean: float,
    std: float,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = RFSpectrogramDataset(
        rows=rows,
        project_root=project_root,
        mean=mean,
        std=std,
        expected_shape=(128, 509),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((preds == y).sum().item())
        total_count += int(x.size(0))

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_preds: list[int] = []
    all_targets: list[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((preds == y).sum().item())
        total_count += int(x.size(0))

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())

    return total_loss / total_count, total_correct / total_count, all_preds, all_targets


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        type=str,
        default="data/processed/cnn_capture/splits/rf4_random_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/ml/rf4_cnn_baseline_v1",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path.cwd()
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest_csv(split_dir / "train.csv")
    val_rows = read_manifest_csv(split_dir / "val.csv")
    test_rows = read_manifest_csv(split_dir / "test.csv")

    print("[INFO] train:", len(train_rows))
    print("[INFO] val  :", len(val_rows))
    print("[INFO] test :", len(test_rows))
    print("[INFO] classes:", RF4_CLASS_NAMES)

    mean, std = compute_spectrogram_mean_std(
        rows=train_rows,
        project_root=project_root,
    )

    print(f"[INFO] train mean={mean:.6f}, std={std:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_loader = make_loader(
        rows=train_rows,
        project_root=project_root,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        rows=val_rows,
        project_root=project_root,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        rows=test_rows,
        project_root=project_root,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = RF3SmallCNN(num_classes=num_rf4_classes()).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_path = out_dir / "best_model.pt"

    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "input_shape": [1, 128, 509],
                    "num_classes": num_rf4_classes(),
                    "class_names": RF4_CLASS_NAMES,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_preds, test_targets = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    cm = build_confusion_matrix(
        preds=test_preds,
        targets=test_targets,
        num_classes=num_rf4_classes(),
    )

    save_confusion_matrix_csv(out_dir / "confusion_matrix.csv", cm)
    save_confusion_matrix_png(out_dir / "confusion_matrix.png", cm)

    report = make_classification_report_text(cm)
    save_text(out_dir / "classification_report.txt", report)

    summary = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "mean": mean,
        "std": std,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "num_classes": num_rf4_classes(),
        "class_names": RF4_CLASS_NAMES,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== Test Result ===")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc ={test_acc:.4f}")
    print()
    print(report)
    print()
    print(f"[OK] saved best model: {best_path}")
    print(f"[OK] saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()