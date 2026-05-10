from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.ml.rf3_labels import ID_TO_LABEL, LABEL_TO_ID, num_rf3_classes
from src.ml.rf3_model import RF3SmallCNN


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return project_root / path


def load_spectrogram(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    x = np.load(path).astype(np.float32)

    if x.shape == expected_shape:
        return x

    # 혹시 (1, 128, 509) 형태로 저장되어 있으면 channel 차원 제거
    if x.ndim == 3 and x.shape[0] == 1 and x.shape[1:] == expected_shape:
        return x[0]

    raise ValueError(
        f"Unexpected input shape: {x.shape}, expected {expected_shape} or (1, {expected_shape[0]}, {expected_shape[1]})"
    )


def save_spectrogram_png(path: Path, x: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skip PNG saving.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(x, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Frequency bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="outputs/ml/rf3_cnn_baseline_v1/best_model.pt",
        help="Path to trained RF3 best_model.pt",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one spectrogram .npy file",
    )
    parser.add_argument(
        "--true-label",
        type=str,
        default="",
        choices=["", "Background", "Bluetooth", "WiFi"],
        help="Optional true label for comparison",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root for resolving relative paths",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--save-png",
        type=str,
        default="",
        help="Optional output PNG path for input spectrogram visualization",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    model_path = resolve_path(args.model, project_root)
    input_path = resolve_path(args.input, project_root)

    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])

    # checkpoint에 input_shape가 있으면 사용, 없으면 현재 RF3 기본값 사용
    input_shape = checkpoint.get("input_shape", [1, 128, 509])
    expected_shape = (int(input_shape[-2]), int(input_shape[-1]))

    model = RF3SmallCNN(num_classes=num_rf3_classes()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = load_spectrogram(input_path, expected_shape=expected_shape)

    x_norm = (x - mean) / (std + 1e-8)

    # Conv2d 입력: batch x channel x height x width
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0).float()
    x_tensor = x_tensor.to(device)

    logits = model(x_tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = ID_TO_LABEL[pred_id]
    confidence = float(probs[pred_id])

    result = {
        "input": str(input_path),
        "model": str(model_path),
        "pred_label": pred_label,
        "confidence": confidence,
        "probabilities": {
            "Background": float(probs[LABEL_TO_ID["Background"]]),
            "Bluetooth": float(probs[LABEL_TO_ID["Bluetooth"]]),
            "WiFi": float(probs[LABEL_TO_ID["WiFi"]]),
        },
        "mean": mean,
        "std": std,
        "input_shape": list(input_shape),
        "device": str(device),
    }

    if args.true_label:
        true_label = args.true_label
        result["true_label"] = true_label
        result["correct"] = bool(true_label == pred_label)

    print()
    print("=== RF3 Single File Prediction ===")
    print(f"input      : {input_path}")
    print(f"model      : {model_path}")
    print(f"device     : {device}")
    print(f"shape      : {x.shape}")
    print()
    print(f"predicted  : {pred_label}")
    print(f"confidence : {confidence:.6f}")
    print()
    print("[probabilities]")
    for label in ["Background", "Bluetooth", "WiFi"]:
        print(f"{label:10s}: {result['probabilities'][label]:.6f}")

    if args.true_label:
        print()
        print(f"true_label : {args.true_label}")
        print(f"correct    : {result['correct']}")

    if args.save_json:
        json_path = resolve_path(args.save_json, project_root)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"[OK] saved json: {json_path}")

    if args.save_png:
        png_path = resolve_path(args.save_png, project_root)
        title = f"pred={pred_label}, conf={confidence:.3f}"
        if args.true_label:
            title = f"true={args.true_label}, " + title

        save_spectrogram_png(
            path=png_path,
            x=x,
            title=title,
        )
        print(f"[OK] saved png : {png_path}")


if __name__ == "__main__":
    main()
