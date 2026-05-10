from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ml.rf3_inference import RF3Classifier


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return project_root / path


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
        "--threshold",
        type=float,
        default=0.0,
        help="If > 0, return Unknown when confidence is below this threshold",
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

    classifier = RF3Classifier(model_path=model_path)

    if args.threshold > 0:
        result = classifier.predict_file_with_threshold(
            path=input_path,
            confidence_threshold=args.threshold,
        )
    else:
        result = classifier.predict_file(path=input_path)

    result_dict = result.to_dict()
    result_dict["input"] = str(input_path)

    if args.true_label:
        result_dict["true_label"] = args.true_label
        result_dict["correct"] = bool(args.true_label == result.class_name)

    print()
    print("=== RF3 Single File Prediction ===")
    print(f"input      : {input_path}")
    print(f"model      : {model_path}")
    print(f"shape      : {result.input_shape}")
    print()
    print(f"predicted  : {result.class_name}")
    print(f"confidence : {result.confidence:.6f}")
    print(f"valid      : {result.valid}")
    if result.reason:
        print(f"reason     : {result.reason}")

    print()
    print("[probabilities]")
    for label, prob in result.probabilities.items():
        print(f"{label:10s}: {prob:.6f}")

    if args.true_label:
        print()
        print(f"true_label : {args.true_label}")
        print(f"correct    : {result_dict['correct']}")

    if args.save_json:
        json_path = resolve_path(args.save_json, project_root)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"[OK] saved json: {json_path}")

    if args.save_png:
        x = np.load(input_path).astype(np.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]

        png_path = resolve_path(args.save_png, project_root)
        title = f"pred={result.class_name}, conf={result.confidence:.3f}"
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
