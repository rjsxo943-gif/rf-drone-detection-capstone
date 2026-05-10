from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.ml import (
    RF3SmallCNN,
    ID_TO_LABEL,
    LABEL_TO_ID,
    num_rf3_classes,
)


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_spectrogram_png(path: Path, x: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skip png saving.")
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
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/cnn_capture/splits/rf3_random_v1/test.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/ml/rf3_cnn_baseline_v1/error_analysis",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    rows = read_csv(csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])

    model = RF3SmallCNN(num_classes=num_rf3_classes()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prediction_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        filepath = row["filepath"]
        true_label = row["label"]

        path = resolve_path(filepath, project_root)

        x = np.load(path).astype(np.float32)

        if x.shape != (128, 509):
            raise ValueError(f"Unexpected shape {x.shape}: {path}")

        x_norm = (x - mean) / (std + 1e-8)

        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0).float()
        x_tensor = x_tensor.to(device)

        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = ID_TO_LABEL[pred_id]
        confidence = float(probs[pred_id])

        true_id = LABEL_TO_ID[true_label]
        correct = pred_id == true_id

        result = {
            "filepath": filepath,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": str(correct),
            "confidence": f"{confidence:.6f}",
            "prob_background": f"{float(probs[LABEL_TO_ID['Background']]):.6f}",
            "prob_bluetooth": f"{float(probs[LABEL_TO_ID['Bluetooth']]):.6f}",
            "prob_wifi": f"{float(probs[LABEL_TO_ID['WiFi']]):.6f}",
            "session": row.get("session", ""),
            "group": row.get("group", ""),
        }

        prediction_rows.append(result)

        if not correct:
            error_rows.append(result)

            png_name = (
                f"{idx:04d}__true_{true_label}"
                f"__pred_{pred_label}"
                f"__conf_{confidence:.3f}.png"
            )

            title = (
                f"true={true_label}, pred={pred_label}, "
                f"confidence={confidence:.3f}"
            )

            save_spectrogram_png(
                path=out_dir / "misclassified_png" / png_name,
                x=x,
                title=title,
            )

    fieldnames = [
        "filepath",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "prob_background",
        "prob_bluetooth",
        "prob_wifi",
        "session",
        "group",
    ]

    write_csv(out_dir / "predictions.csv", prediction_rows, fieldnames)
    write_csv(out_dir / "misclassified.csv", error_rows, fieldnames)

    print("=== Error Analysis ===")
    print(f"total samples       : {len(prediction_rows)}")
    print(f"misclassified count : {len(error_rows)}")
    print(f"saved predictions   : {out_dir / 'predictions.csv'}")
    print(f"saved errors        : {out_dir / 'misclassified.csv'}")
    print(f"saved error images  : {out_dir / 'misclassified_png'}")

    if error_rows:
        print()
        print("[misclassified]")
        for e in error_rows:
            print(
                f"{e['true_label']} -> {e['pred_label']} "
                f"conf={e['confidence']} | {e['filepath']}"
            )


if __name__ == "__main__":
    main()
