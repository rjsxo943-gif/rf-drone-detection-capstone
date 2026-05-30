from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from scripts.ml.train_rf_binary_cnn import (
    SmallRFBinaryCNN,
    RFBinaryDataset,
    load_manifest,
)


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

    return torch.cat(all_probs), torch.cat(all_labels)


def compute_metrics(probs, labels, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).long()

    labels = labels.long()

    tp = int(((pred == 1) & (labels == 1)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    notdrone_acc = safe_div(tn, tn + fp)
    drone_acc = recall
    false_drone_rate = safe_div(fp, fp + tn)
    missed_drone_rate = safe_div(fn, fn + tp)

    return {
        "threshold": threshold,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "notdrone_acc": notdrone_acc,
        "drone_acc": drone_acc,
        "false_drone_rate": false_drone_rate,
        "missed_drone_rate": missed_drone_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    samples = load_manifest(Path(args.manifest))
    dataset = RFBinaryDataset(samples)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SmallRFBinaryCNN(num_classes=2)
    ckpt = torch.load(args.model, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)

    probs, labels = collect_probs(model, loader, device)

    thresholds = [round(x / 100, 2) for x in range(30, 96, 5)]
    rows = [compute_metrics(probs, labels, th) for th in thresholds]

    print("=== Threshold Sweep Result ===")
    print(f"manifest: {args.manifest}")
    print(f"total   : {len(labels)}")
    print()
    print(
        "thr   acc     prec    recall  f1      FP_rate Miss    TN   FP   FN   TP"
    )

    for r in rows:
        print(
            f"{r['threshold']:.2f}  "
            f"{r['acc']:.4f}  "
            f"{r['precision']:.4f}  "
            f"{r['recall']:.4f}  "
            f"{r['f1']:.4f}  "
            f"{r['false_drone_rate']:.4f}  "
            f"{r['missed_drone_rate']:.4f}  "
            f"{r['tn']:4d} {r['fp']:4d} {r['fn']:4d} {r['tp']:4d}"
        )

    best_f1 = max(rows, key=lambda r: r["f1"])
    print()
    print("=== Best by F1 ===")
    for k, v in best_f1.items():
        print(f"{k:20s}: {v}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
