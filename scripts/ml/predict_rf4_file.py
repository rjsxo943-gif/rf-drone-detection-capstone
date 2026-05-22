from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.ml import RF3SmallCNN


DEFAULT_CLASS_NAMES = [
    "Background",
    "WiFi",
    "Bluetooth",
    "Drone-like",
]


def load_spectrogram(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    x = np.load(path)

    if x.shape != expected_shape:
        raise ValueError(
            f"Unexpected spectrogram shape: {x.shape}, expected: {expected_shape}"
        )

    return x.astype(np.float32)


def prepare_input(
    spec: np.ndarray,
    mean: float,
    std: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = (spec - mean) / (std + eps)

    # shape: (H, W) -> (1, 1, H, W)
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(0).unsqueeze(0)

    return x


@torch.no_grad()
def predict(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[int, float, list[float]]:
    model.eval()

    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    prob_list = [float(p.item()) for p in probs.cpu()]

    return pred_id, confidence, prob_list


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="outputs/ml/rf4_cnn_baseline_v2/best_model.pt",
        help="Path to RF4 best_model.pt",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input spectrogram .npy file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for Unknown decision",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional output JSON path",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])
    input_shape = checkpoint.get("input_shape", [1, 128, 509])
    num_classes = int(checkpoint.get("num_classes", 4))
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)

    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length mismatch: len={len(class_names)}, num_classes={num_classes}"
        )

    expected_shape = (int(input_shape[1]), int(input_shape[2]))

    spec = load_spectrogram(input_path, expected_shape=expected_shape)
    x = prepare_input(spec, mean=mean, std=std)

    model = RF3SmallCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_id, confidence, probs = predict(model, x, device)

    pred_class = class_names[pred_id]
    final_class = pred_class if confidence >= args.threshold else "Unknown"

    result = {
        "input": str(input_path),
        "model": str(model_path),
        "pred_id": pred_id,
        "pred_class": pred_class,
        "final_class": final_class,
        "confidence": confidence,
        "threshold": args.threshold,
        "class_names": class_names,
        "probabilities": {
            class_name: prob for class_name, prob in zip(class_names, probs)
        },
        "mean": mean,
        "std": std,
        "expected_shape": list(expected_shape),
    }

    print()
    print("=== RF4 Prediction Result ===")
    print(f"input       : {input_path}")
    print(f"model       : {model_path}")
    print(f"pred_class  : {pred_class}")
    print(f"confidence  : {confidence:.4f}")
    print(f"threshold   : {args.threshold:.4f}")
    print(f"final_class : {final_class}")

    print()
    print("[probabilities]")
    for class_name, prob in result["probabilities"].items():
        print(f"{class_name:10s}: {prob:.6f}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"[OK] saved json: {json_path}")


if __name__ == "__main__":
    main()
