from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["NotDrone", "Drone"]


@dataclass
class Sample:
    path: Path
    label_id: int
    label: str
    session: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary RF CNN: Drone vs NotDrone")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(manifest_path: Path) -> List[Sample]:
    samples: List[Sample] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_path = Path(row["path"])

            if raw_path.is_absolute():
                npy_path = raw_path
            else:
                # manifest 안의 path는 manifest 파일 위치 기준 상대경로로 해석
                npy_path = manifest_path.parent / raw_path

            samples.append(
                Sample(
                    path=npy_path,
                    label_id=int(row["label_id"]),
                    label=row["label"],
                    session=row["session"],
                    split=row["split"],
                )
            )

    return samples


class RFBinaryDataset(Dataset):
    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        arr = np.load(sample.path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        # 예상 shape: (freq, time)
        # CNN 입력 shape: (channel, freq, time)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.shape[0] != 1:
                raise ValueError(f"Unsupported 3D array shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

        x = torch.from_numpy(arr)
        y = torch.tensor(sample.label_id, dtype=torch.long)
        return x, y


class SmallRFBinaryCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_labels(samples: List[Sample]) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for sample in samples:
        counts[sample.label_id] += 1
    return counts


def make_class_weights(train_samples: List[Sample], device: torch.device) -> torch.Tensor:
    counts = count_labels(train_samples)
    total = counts[0] + counts[1]

    # CrossEntropyLoss weight: 적은 클래스에 더 큰 가중치
    weights = []
    for class_id in [0, 1]:
        count = max(counts[class_id], 1)
        weights.append(total / (2.0 * count))

    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    # confusion matrix:
    # rows = true, cols = pred
    cm = np.zeros((2, 2), dtype=np.int64)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        correct += int((pred == y).sum().item())

        for true_id, pred_id in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[int(true_id), int(pred_id)] += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    notdrone_total = cm[0].sum()
    drone_total = cm[1].sum()

    notdrone_acc = cm[0, 0] / max(notdrone_total, 1)
    drone_acc = cm[1, 1] / max(drone_total, 1)

    # Drone으로 오탐한 NotDrone 비율
    false_drone_rate = cm[0, 1] / max(notdrone_total, 1)

    # Drone을 놓친 비율
    missed_drone_rate = cm[1, 0] / max(drone_total, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "notdrone_acc": float(notdrone_acc),
        "drone_acc": float(drone_acc),
        "false_drone_rate": float(false_drone_rate),
        "missed_drone_rate": float(missed_drone_rate),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        correct += int((pred == y).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def save_history(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_confusion_csv(path: Path, metrics: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", "NotDrone", "Drone"])
        writer.writerow(["NotDrone", metrics["tn"], metrics["fp"]])
        writer.writerow(["Drone", metrics["fn"], metrics["tp"]])


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    samples = load_manifest(manifest_path)

    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "val"]
    test_samples = [s for s in samples if s.split == "test"]

    print("=== RF Binary CNN Training ===")
    print(f"manifest : {manifest_path}")
    print(f"out_dir  : {out_dir}")
    print(f"device   : {device}")
    print(f"epochs   : {args.epochs}")
    print(f"batch    : {args.batch_size}")
    print(f"lr       : {args.lr}")
    print()

    for name, split_samples in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        counts = count_labels(split_samples)
        print(
            f"{name:5s}: total={len(split_samples):4d} "
            f"NotDrone={counts[0]:4d} Drone={counts[1]:4d}"
        )

    print()

    train_loader = DataLoader(
        RFBinaryDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        RFBinaryDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        RFBinaryDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SmallRFBinaryCNN(num_classes=2).to(device)

    class_weights = make_class_weights(train_samples, device=device)
    print(f"class_weights: NotDrone={class_weights[0].item():.4f}, Drone={class_weights[1].item():.4f}")
    print()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_notdrone_acc": val_metrics["notdrone_acc"],
            "val_drone_acc": val_metrics["drone_acc"],
            "val_false_drone_rate": val_metrics["false_drone_rate"],
            "val_missed_drone_rate": val_metrics["missed_drone_rate"],
        }
        history.append(row)

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_DroneAcc={val_metrics['drone_acc']:.4f} "
            f"val_NotDroneAcc={val_metrics['notdrone_acc']:.4f} "
            f"FP_Drone={val_metrics['false_drone_rate']:.4f} "
            f"MissDrone={val_metrics['missed_drone_rate']:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "input_channels": 1,
            "num_classes": 2,
            "epoch": epoch,
            "val_acc": val_metrics["acc"],
            "args": vars(args),
        }

        torch.save(checkpoint, last_path)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(checkpoint, best_path)

    save_history(out_dir / "history.csv", history)

    # best model로 test 평가
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    save_confusion_csv(out_dir / "confusion_matrix_test.csv", test_metrics)

    with (out_dir / "test_metrics.txt").open("w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    print()
    print("=== Test Result ===")
    print(f"test_acc             : {test_metrics['acc']:.4f}")
    print(f"test_NotDrone_acc    : {test_metrics['notdrone_acc']:.4f}")
    print(f"test_Drone_acc       : {test_metrics['drone_acc']:.4f}")
    print(f"false_drone_rate     : {test_metrics['false_drone_rate']:.4f}")
    print(f"missed_drone_rate    : {test_metrics['missed_drone_rate']:.4f}")
    print(f"confusion TN FP FN TP: {test_metrics['tn']} {test_metrics['fp']} {test_metrics['fn']} {test_metrics['tp']}")
    print()
    print(f"[OK] best model: {best_path}")
    print(f"[OK] history   : {out_dir / 'history.csv'}")
    print(f"[OK] confusion : {out_dir / 'confusion_matrix_test.csv'}")


if __name__ == "__main__":
    main()
