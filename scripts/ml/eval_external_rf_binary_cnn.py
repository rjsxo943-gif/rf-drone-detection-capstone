from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.ml.train_rf_binary_cnn import (
    SmallRFBinaryCNN,
    RFBinaryDataset,
    load_manifest,
    evaluate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    samples = load_manifest(Path(args.manifest))
    dataset = RFBinaryDataset(samples)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
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

    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device)

    print("=== External Test Result ===")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()