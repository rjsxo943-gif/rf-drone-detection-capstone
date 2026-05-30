from __future__ import annotations

import argparse
import csv
from collections import deque
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


def compute_metrics(preds: list[int], labels: list[int]) -> dict[str, float]:
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_drone_rate": safe_div(fp, fp + tn),
        "missed_drone_rate": safe_div(fn, fn + tp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--th-g25", type=float, default=0.35)
    parser.add_argument("--th-g30", type=float, default=0.80)
    parser.add_argument("--th-default", type=float, default=0.50)

    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--vote-k", type=int, default=3)

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
    model.eval()

    rows = []
    sample_index = 0

    raw_preds: list[int] = []
    temporal_preds: list[int] = []
    labels: list[int] = []

    session_buffers: dict[str, deque[int]] = {}

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()

        for i in range(len(y)):
            sample = samples[sample_index]
            prob = float(probs[i].item())
            label = int(y[i].item())

            th = get_threshold(
                str(sample.path),
                sample.session,
                args.th_g25,
                args.th_g30,
                args.th_default,
            )

            raw_pred = 1 if prob >= th else 0

            if sample.session not in session_buffers:
                session_buffers[sample.session] = deque(maxlen=args.window)

            buf = session_buffers[sample.session]
            buf.append(raw_pred)

            # 실시간 causal 방식:
            # window가 아직 덜 찼으면 현재까지 들어온 프레임으로만 판단.
            # 예: 5개 중 3개 조건이면, 초반에는 ceil(현재길이 * 3/5) 이상이면 Drone.
            current_k = max(1, round(len(buf) * args.vote_k / args.window))
            temporal_pred = 1 if sum(buf) >= current_k else 0

            raw_preds.append(raw_pred)
            temporal_preds.append(temporal_pred)
            labels.append(label)

            rows.append({
                "index": sample_index,
                "session": sample.session,
                "path": str(sample.path),
                "label": label,
                "prob_drone": prob,
                "threshold": th,
                "raw_pred": raw_pred,
                "temporal_pred": temporal_pred,
            })

            sample_index += 1

    raw_metrics = compute_metrics(raw_preds, labels)
    temporal_metrics = compute_metrics(temporal_preds, labels)

    print("=== Temporal Vote Evaluation ===")
    print(f"manifest      : {args.manifest}")
    print(f"th_g25        : {args.th_g25}")
    print(f"th_g30        : {args.th_g30}")
    print(f"window/vote_k : {args.window}/{args.vote_k}")
    print()

    print("=== Raw Frame Result ===")
    for k, v in raw_metrics.items():
        print(f"{k:20s}: {v}")

    print()
    print("=== Temporal Vote Result ===")
    for k, v in temporal_metrics.items():
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
