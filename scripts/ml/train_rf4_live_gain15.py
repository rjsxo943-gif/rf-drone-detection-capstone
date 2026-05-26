from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.ml import RF3SmallCNN


PROJECT_ROOT = Path(".").resolve()

CLASS_NAMES = ["Background", "WiFi", "Bluetooth", "Drone-like"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

DATA_ROOT = Path("data/processed/cnn_capture")
OUT_ROOT = Path("data/processed/cnn_capture_live_gain15_v1")
SPLIT_DIR = OUT_ROOT / "splits" / "rf4_live_gain15_v1"
MODEL_OUT = Path("outputs/ml/rf4_cnn_live_gain15_v1")

EXPECTED_SHAPE = (128, 509)
SEED = 42


LABEL_DIRS = {
    "Background": [
        "background_live_gain15_alloff",
    ],
    "WiFi": [
        "wifi_live_gain15_ch6_range2425_2450",
    ],
    "Bluetooth": [
        "bluetooth_live_gain15_airpods_music_minpass1",
        "bluetooth_live_gain15_airpods_call_minpass1",
        "bluetooth_live_gain15_airpods_pairing_event_minpass1",
    ],
    "Drone-like": [
        "drone_like_live_gain15_front",
        "drone_like_live_gain15_front_center",
        "drone_like_live_gain15_front_left30",
        "drone_like_live_gain15_front_right30",
    ],
}


def load_spec(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    # npz capture format
    if hasattr(data, "files"):
        if "spectrogram" in data.files:
            x = data["spectrogram"]
        elif "cnn_input" in data.files:
            x = data["cnn_input"]
            if x.ndim == 3 and x.shape[-1] == 1:
                x = x[..., 0]
        else:
            raise KeyError(f"No spectrogram/cnn_input in {path}. keys={data.files}")
    else:
        x = data

    x = np.asarray(x, dtype=np.float32)

    if x.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {x.shape}, expected={EXPECTED_SHAPE}, path={path}")

    return x


def read_metadata(path: Path) -> dict:
    try:
        z = np.load(path, allow_pickle=True)
        if hasattr(z, "files") and "metadata_json" in z.files:
            return json.loads(str(z["metadata_json"]))
    except Exception:
        pass
    return {}


def build_manifest() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for label, dirs in LABEL_DIRS.items():
        for d in dirs:
            folder = DATA_ROOT / d
            files = sorted(folder.rglob("*.npz"))

            if not files:
                print(f"[WARN] no files: {folder}")

            for p in files:
                meta = read_metadata(p)
                session = str(meta.get("session_id", p.parent.name))
                center_freq = int(meta.get("center_freq", 0) or 0)
                center_freq_mhz = int(round(center_freq / 1e6)) if center_freq > 0 else ""

                rows.append({
                    "filepath": str(p),
                    "label": label,
                    "session": session,
                    "group": d,
                    "selected_dir": d,
                    "center_freq_mhz": center_freq_mhz,
                    "gain": "15",
                    "distance": "",
                })

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filepath", "label", "session", "group", "selected_dir",
            "center_freq_mhz", "gain", "distance"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("[manifest]", manifest_path)
    print("[counts]", dict(Counter(r["label"] for r in rows)))
    return rows


def split_rows(rows: list[dict[str, str]]) -> None:
    random.seed(SEED)

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    splits = {"train": [], "val": [], "test": []}

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_items in splits.items():
        random.shuffle(split_items)
        out = SPLIT_DIR / f"{split_name}.csv"

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filepath", "label", "session", "group", "selected_dir",
                "center_freq_mhz", "gain", "distance"
            ])
            writer.writeheader()
            writer.writerows(split_items)

        print(f"[{split_name}]", len(split_items), dict(Counter(r["label"] for r in split_items)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_mean_std(rows: list[dict[str, str]]) -> tuple[float, float]:
    total_sum = 0.0
    total_sq = 0.0
    total_n = 0

    for r in rows:
        x = load_spec(r["filepath"]).astype(np.float64)
        total_sum += float(x.sum())
        total_sq += float((x * x).sum())
        total_n += int(x.size)

    mean = total_sum / total_n
    var = total_sq / total_n - mean * mean
    std = float(np.sqrt(max(var, 1e-8)))
    return float(mean), std


class RF4LiveDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], mean: float, std: float) -> None:
        self.rows = rows
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        x = load_spec(r["filepath"])
        x = (x - self.mean) / (self.std + 1e-8)

        y = LABEL_TO_ID[r["label"]]
        return torch.from_numpy(x).unsqueeze(0).float(), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        loss_sum += float(loss.item()) * x.size(0)
        total += int(x.size(0))
        correct += int((pred == y).sum().item())

        for yt, yp in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[int(yt), int(yp)] += 1

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "cm": cm,
    }


def save_confusion_matrix_csv(cm: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + [int(v) for v in cm[i]])


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = build_manifest()
    split_rows(rows)

    train_rows = read_csv(SPLIT_DIR / "train.csv")
    val_rows = read_csv(SPLIT_DIR / "val.csv")
    test_rows = read_csv(SPLIT_DIR / "test.csv")

    mean, std = compute_mean_std(train_rows)
    print(f"[mean/std] mean={mean:.8f}, std={std:.8f}")

    train_ds = RF4LiveDataset(train_rows, mean, std)
    val_ds = RF4LiveDataset(val_rows, mean, std)
    test_ds = RF4LiveDataset(test_rows, mean, std)

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RF3SmallCNN(num_classes=len(CLASS_NAMES)).to(device)

    train_counts = Counter(r["label"] for r in train_rows)
    weights = []
    for name in CLASS_NAMES:
        weights.append(1.0 / max(train_counts[name], 1))
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history = []

    epochs = 25
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)

            loss_sum += float(loss.item()) * x.size(0)
            total += int(x.size(0))
            correct += int((pred == y).sum().item())

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        val = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
        }
        history.append(row)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}"
        )

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "input_shape": [1, EXPECTED_SHAPE[0], EXPECTED_SHAPE[1]],
                "num_classes": len(CLASS_NAMES),
                "class_names": CLASS_NAMES,
                "best_val_acc": best_val_acc,
                "args": {
                    "split_dir": str(SPLIT_DIR),
                    "out_dir": str(MODEL_OUT),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": 1e-3,
                    "seed": SEED,
                    "dataset": "live_gain15_v1",
                },
            }
            torch.save(ckpt, MODEL_OUT / "best_model.pt")
            print(f"  [saved] {MODEL_OUT / 'best_model.pt'}")

    test = evaluate(model, test_loader, device)

    with (MODEL_OUT / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    save_confusion_matrix_csv(test["cm"], MODEL_OUT / "confusion_matrix.csv")

    summary = {
        "out_dir": str(MODEL_OUT),
        "split_dir": str(SPLIT_DIR),
        "mean": mean,
        "std": std,
        "best_val_acc": best_val_acc,
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "class_names": CLASS_NAMES,
        "counts_total": dict(Counter(r["label"] for r in rows)),
        "counts_train": dict(Counter(r["label"] for r in train_rows)),
        "counts_val": dict(Counter(r["label"] for r in val_rows)),
        "counts_test": dict(Counter(r["label"] for r in test_rows)),
    }

    with (MODEL_OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=== DONE ===")
    print("best_model:", MODEL_OUT / "best_model.pt")
    print("test_acc:", test["acc"])
    print("confusion_matrix:", MODEL_OUT / "confusion_matrix.csv")


if __name__ == "__main__":
    main()
