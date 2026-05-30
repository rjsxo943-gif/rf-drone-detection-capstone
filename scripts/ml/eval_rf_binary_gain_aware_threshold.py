from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.ml.train_rf_binary_cnn import (
    SmallRFBinaryCNN,
    RFBinaryDataset,
    load_manifest,
)


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def get_threshold(sample_path: str, session: str, th_g25: float, th_g30: float, th_default: float) -> float:
    text = (sample_path + " " + session).lower()

    if "_g25_" in text:
        return th_g25
    if "_g30_" in text:
        return th_g30
    return th_default


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--th-g25", type=float, default=0.40)
    parser.add_argument("--th-g30", type=float, default=0.85)
    parser.add_argument("--th-default", type=float, default=0.50)
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
    model.eval()

    all_preds = []
    all_labels = []
    all_thresholds = []

    sample_index = 0

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()

        batch_size = len(y)

        for i in range(batch_size):
            sample = samples[sample_index]
            th = get_threshold(
                str(sample.path),
                sample.session,
                args.th_g25,
                args.th_g30,
                args.th_default,
            )

            pred = 1 if float(probs[i]) >= th else 0

            all_preds.append(pred)
            all_labels.append(int(y[i].item()))
            all_thresholds.append(th)

            sample_index += 1

    tp = sum(p == 1 and y == 1 for p, y in zip(all_preds, all_labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(all_preds, all_labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(all_preds, all_labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(all_preds, all_labels))

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_drone_rate = safe_div(fp, fp + tn)
    missed_drone_rate = safe_div(fn, fn + tp)

    print("=== Gain-aware Threshold Result ===")
    print(f"manifest            : {args.manifest}")
    print(f"threshold g25       : {args.th_g25}")
    print(f"threshold g30       : {args.th_g30}")
    print(f"threshold default   : {args.th_default}")
    print()
    print(f"acc                 : {acc}")
    print(f"precision           : {precision}")
    print(f"recall              : {recall}")
    print(f"f1                  : {f1}")
    print(f"false_drone_rate    : {false_drone_rate}")
    print(f"missed_drone_rate   : {missed_drone_rate}")
    print(f"tn                  : {tn}")
    print(f"fp                  : {fp}")
    print(f"fn                  : {fn}")
    print(f"tp                  : {tp}")


if __name__ == "__main__":
    main()
